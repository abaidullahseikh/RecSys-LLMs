"""Sparse program co-view graph and personalized PageRank for SWEET.TV.

Graph/profile fit accepts only the frozen Step 3 training_input handle. It never
accepts validation rows. User long histories are deterministically capped by
training qualifying-airing count, then watched seconds, then program ID.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import hashlib
import heapq
import math
import time

import numpy as np
from scipy.sparse import csr_matrix

from evaluation import FALLBACK_POLICY, FallbackRanking, candidate_pool, complete_top5, identity_hash, program_id
from validation import ROOT, training_input

EDGE_WEIGHTS = ("common_viewers", "cosine")
PERSONALIZATIONS = ("uniform", "frequency", "duration", "recency")
DEFAULT_HISTORY_CAP = 40
DEFAULT_MIN_COMMON = 3
DEFAULT_RESTART = 0.20
DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITER = 250
RECENCY_HALF_LIFE_SECONDS = 30 * 86400


@dataclass(frozen=True)
class TrainingProfiles:
    nodes: tuple
    index: dict
    profiles: dict
    distinct_viewers: dict
    training_sha256: str
    training_window_utc: tuple
    rows: int
    runtime_seconds: float


@dataclass(frozen=True)
class ProgramGraph:
    nodes: tuple
    matrix: csr_matrix  # row-stochastic P; edges are symmetric before normalization
    raw_pair_counts: dict  # undirected (lower node index, higher node index) -> viewers
    retained_edge_weights: dict
    weight_type: str
    max_neighbors: int
    history_cap: int
    min_common_viewers: int
    pair_occurrences_generated: int
    profile_programs_truncated: int
    runtime_seconds: float

    def statistics(self):
        degrees = np.diff(self.matrix.indptr)
        nnz = int(self.matrix.nnz)
        n = len(self.nodes)
        return {"nodes": n, "directed_nonzero_edges": nnz,
                "undirected_nonzero_edges": nnz // 2,
                "average_degree": float(degrees.mean()) if n else 0.0,
                "maximum_degree": int(degrees.max()) if n else 0,
                "directed_density": nnz / (n * n) if n else 0.0,
                "directed_sparsity": 1.0 - nnz / (n * n) if n else 1.0,
                "raw_undirected_copairs": len(self.raw_pair_counts),
                "retained_undirected_edges": len(self.retained_edge_weights),
                "pair_occurrences_generated": self.pair_occurrences_generated,
                "profile_programs_truncated": self.profile_programs_truncated,
                "max_neighbors_per_node_before_symmetrization": self.max_neighbors,
                "history_cap_per_user": self.history_cap,
                "minimum_common_viewers": self.min_common_viewers,
                "weight_type": self.weight_type,
                "row_transition_sums_min": float(np.asarray(self.matrix.sum(axis=1)).min()),
                "row_transition_sums_max": float(np.asarray(self.matrix.sum(axis=1)).max()),
                "non_dangling_row_sum_min": float(np.asarray(self.matrix.sum(axis=1)).ravel()[degrees > 0].min()) if np.any(degrees > 0) else None,
                "non_dangling_row_sum_max": float(np.asarray(self.matrix.sum(axis=1)).ravel()[degrees > 0].max()) if np.any(degrees > 0) else None,
                "dangling_nodes": int(np.count_nonzero(degrees == 0))}


def build_training_profiles(config):
    """The sole production data entry point: frozen Step 3 Week 11–20 handle."""
    return load_training_profiles(config)


def load_training_profiles(config):
    """Load one sparse user profile per user from the validated training-only API."""
    started = time.monotonic()
    source = training_input(config)
    profiles = defaultdict(dict)
    support = Counter()
    rows = 0
    for row in source.rows():
        rows += 1
        if row.program_id in profiles[row.user_id]:
            raise ValueError("Training input must contain one aggregate per user/program")
        profiles[row.user_id][row.program_id] = (row.qualifying_airings, row.watched_seconds,
                                                  row.last_watch_utc)
        support[row.program_id] += 1
    nodes = tuple(sorted(support, key=int))
    index = {item: i for i, item in enumerate(nodes)}
    return TrainingProfiles(nodes, index, {u: dict(p) for u, p in profiles.items()},
                            dict(support), source.sha256, tuple(source.window_utc), rows,
                            time.monotonic() - started)


def user_capped_indices(profile, nodes, *, max_programs=DEFAULT_HISTORY_CAP):
    items = list(profile.items())
    items.sort(key=lambda pair: (-pair[1][0], -pair[1][1], int(pair[0])))
    return [nodes[item] for item, _ in items[:max_programs]]


def build_copair_counts(data, *, history_cap=DEFAULT_HISTORY_CAP):
    if history_cap < 2:
        raise ValueError("History cap must be at least 2")
    counts = Counter()
    occurrences = truncated = 0
    for profile in data.profiles.values():
        chosen = user_capped_indices(profile, data.index, max_programs=history_cap)
        truncated += max(0, len(profile) - len(chosen))
        # A user contributes at most one count to each pair: profile has one row per item.
        for i, j in combinations(sorted(chosen), 2):
            counts[(i, j)] += 1
            occurrences += 1
    return dict(counts), occurrences, truncated


def _edge_weight(common, viewer_i, viewer_j, weight_type):
    if weight_type == "common_viewers":
        return float(common)
    if weight_type == "cosine":
        return common / math.sqrt(viewer_i * viewer_j)
    raise ValueError(f"Unknown edge weighting: {weight_type}")


def build_graph_from_pair_counts(nodes, viewer_counts, pair_counts, *, weight_type="common_viewers",
                                 max_neighbors=30, min_common_viewers=3, history_cap=DEFAULT_HISTORY_CAP,
                                 pair_occurrences_generated=0, profile_programs_truncated=0):
    """Bound graph degree with per-node heaps, then symmetrize their union."""
    if weight_type not in EDGE_WEIGHTS or max_neighbors <= 0 or min_common_viewers <= 0:
        raise ValueError("Invalid graph parameters")
    n = len(nodes)
    heaps = [[] for _ in range(n)]
    pair_counts = {tuple(pair): int(count) for pair, count in pair_counts.items()}
    for (i, j), common in pair_counts.items():
        if not (0 <= i < j < n) or common <= 0:
            raise ValueError("Malformed co-view pair count")
        if common < min_common_viewers:
            continue
        weight = _edge_weight(common, viewer_counts[nodes[i]], viewer_counts[nodes[j]], weight_type)
        for node, other in ((i, j), (j, i)):
            # Larger heap key is better: greater weight, then lower numeric ID.
            key = (weight, -int(nodes[other]), other, common)
            if len(heaps[node]) < max_neighbors:
                heapq.heappush(heaps[node], key)
            elif key > heaps[node][0]:
                heapq.heapreplace(heaps[node], key)
    retained = set()
    for node, heap in enumerate(heaps):
        for weight, _, other, common in heap:
            retained.add((min(node, other), max(node, other)))
    weights, common_retained = {}, {}
    for i, j in retained:
        common = pair_counts[(i, j)]
        weights[(i, j)] = _edge_weight(common, viewer_counts[nodes[i]], viewer_counts[nodes[j]], weight_type)
        common_retained[(i, j)] = common
    rows, cols, vals = [], [], []
    for (i, j), weight in weights.items():
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((weight, weight))
    raw = csr_matrix((np.asarray(vals, dtype=np.float64),
                      (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                     shape=(n, n), dtype=np.float64)
    sums = np.asarray(raw.sum(axis=1)).ravel()
    inv = np.zeros_like(sums)
    nonzero = sums > 0
    inv[nonzero] = 1.0 / sums[nonzero]
    # Sparse row scaling avoids ever materializing an n-by-n dense diagonal.
    transition = raw.multiply(inv[:, None]).tocsr()
    transition.sort_indices()
    return ProgramGraph(nodes, transition, pair_counts, weights, weight_type, max_neighbors,
                        history_cap, min_common_viewers, pair_occurrences_generated,
                        profile_programs_truncated, 0.0)


def build_graph(data, *, weight_type="common_viewers", max_neighbors=30,
                history_cap=DEFAULT_HISTORY_CAP, min_common_viewers=DEFAULT_MIN_COMMON):
    started = time.monotonic()
    pair_counts, occurrences, truncated = build_copair_counts(data, history_cap=history_cap)
    graph = build_graph_from_pair_counts(data.nodes, data.distinct_viewers, pair_counts,
                                         weight_type=weight_type, max_neighbors=max_neighbors,
                                         min_common_viewers=min_common_viewers, history_cap=history_cap,
                                         pair_occurrences_generated=occurrences,
                                         profile_programs_truncated=truncated)
    return ProgramGraph(graph.nodes, graph.matrix, graph.raw_pair_counts,
                        graph.retained_edge_weights, graph.weight_type, graph.max_neighbors,
                        graph.history_cap, graph.min_common_viewers, graph.pair_occurrences_generated,
                        graph.profile_programs_truncated, time.monotonic() - started)


def personalization_vector(profile, data, *, variant="uniform", half_life_seconds=RECENCY_HALF_LIFE_SECONDS):
    if variant not in PERSONALIZATIONS:
        raise ValueError(f"Unknown PPR personalization: {variant}")
    if not profile:
        return {}
    end = data.training_window_utc[1]
    weights = {}
    for item, (airings, seconds, last) in profile.items():
        if variant == "uniform":
            weight = 1.0
        elif variant == "frequency":
            weight = float(airings)
        elif variant == "duration":
            weight = float(seconds)
        else:
            age = max(0, end - last)
            weight = math.exp2(-age / half_life_seconds)
        if weight > 0 and math.isfinite(weight):
            weights[item] = weight
    total = math.fsum(weights.values())
    if total <= 0:
        return {}
    return {item: weight / total for item, weight in weights.items()}


def personalized_pagerank(graph, restart, *, restart_probability=DEFAULT_RESTART,
                          tolerance=DEFAULT_TOL, max_iterations=DEFAULT_MAX_ITER):
    """Solve r=alpha*p+(1-alpha)*(P.T*r+dangling_mass*p) deterministically."""
    if not (0 < restart_probability <= 1) or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("Invalid PageRank convergence settings")
    n = len(graph.nodes)
    p = np.zeros(n, dtype=np.float64)
    # Graph stores sorted nodes; construct the exact string->row map here.
    index = {item: i for i, item in enumerate(graph.nodes)}
    for item, value in restart.items():
        if item not in index or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid personalization entry")
        p[index[item]] = value
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Personalization vector must have positive mass")
    p /= total
    dangling = np.diff(graph.matrix.indptr) == 0
    rank = p.copy()
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        dangling_mass = float(rank[dangling].sum())
        next_rank = (restart_probability * p
                     + (1.0 - restart_probability)
                     * (graph.matrix.transpose().dot(rank) + dangling_mass * p))
        residual = float(np.abs(next_rank - rank).sum())
        rank = np.asarray(next_rank).ravel()
        if residual <= tolerance:
            break
    return rank, {"iterations": iteration, "l1_residual": residual,
                  "converged": residual <= tolerance, "rank_sum": float(rank.sum()),
                  "dangling_mass_final": float(rank[dangling].sum())}


def predict_user(graph, data, user, candidates, popularity_fallback, *, personalization="uniform",
                 restart_probability=DEFAULT_RESTART, tolerance=DEFAULT_TOL, max_iterations=DEFAULT_MAX_ITER,
                 recency_half_life_seconds=RECENCY_HALF_LIFE_SECONDS):
    """Return (five IDs, solver info); absent/cold users use shared fallback."""
    profile = data.profiles.get(user, {})
    candidates = candidate_pool(candidates)
    popularity_fallback = tuple(program_id(item) for item in popularity_fallback)
    if len(candidates) < 5:
        raise ValueError("At least five validation candidates are required")
    if (len(popularity_fallback) < 5 or len(set(popularity_fallback)) != len(popularity_fallback)
            or not set(popularity_fallback).issubset(set(candidates))):
        raise ValueError("Popularity fallback must be distinct positive validation candidates")
    if not profile:
        return tuple(popularity_fallback[:5]), {"iterations": 0, "l1_residual": 0.0,
                                                  "converged": True, "rank_sum": 0.0,
                                                  "dangling_mass_final": 0.0,
                                                  "cold_start": True, "fallback_slots": 5}
    restart = personalization_vector(profile, data, variant=personalization,
                                     half_life_seconds=recency_half_life_seconds)
    scores, info = personalized_pagerank(graph, restart, restart_probability=restart_probability,
                                         tolerance=tolerance, max_iterations=max_iterations)
    index = {item: i for i, item in enumerate(graph.nodes)}
    fallback_order = tuple(popularity_fallback)
    tie_rank = {item: rank for rank, item in enumerate(fallback_order)}
    candidate_set = frozenset(candidates)
    # Only positive-score graph candidates are generated by PPR. Unseen/zero-score
    # candidates are completed by the exact shared training-only popularity order.
    scored = [item for item in candidates if item in index and scores[index[item]] > 0]
    scored.sort(key=lambda item: (-scores[index[item]], tie_rank.get(item, len(tie_rank)), int(item)))
    selected, seen = [], set()
    for item in scored:
        if item in candidate_set and item not in seen:
            selected.append(item)
            seen.add(item)
        if len(selected) == 5:
            break
    ppr_slots = len(selected)
    for item in fallback_order:
        if len(selected) >= 5:
            break
        if item in candidate_set and item not in seen:
            selected.append(item)
            seen.add(item)
    if len(selected) != 5 or len(set(selected)) != 5 or any(int(item) <= 0 for item in selected):
        raise AssertionError(f"PPR Top-5 completion failed: selected={selected!r}, candidates={len(candidate_set)}, fallback={len(fallback_order)}")
    info.update(cold_start=False, fallback_slots=5 - ppr_slots,
                ppr_scored_candidates=ppr_slots)
    return tuple(selected), info


def predict_cohort(graph, data, cohort, candidates, fallback, *, personalization="uniform",
                   restart_probability=DEFAULT_RESTART, tolerance=DEFAULT_TOL,
                   max_iterations=DEFAULT_MAX_ITER, half_life_seconds=RECENCY_HALF_LIFE_SECONDS):
    predictions, infos = {}, {}
    for user in cohort:
        predictions[user], infos[user] = predict_user(
            graph, data, user, candidates, fallback, personalization=personalization,
            restart_probability=restart_probability, tolerance=tolerance,
            max_iterations=max_iterations, recency_half_life_seconds=half_life_seconds)
    return predictions, infos


def graph_npz_arrays(graph):
    """Stable compact graph export for verification/reuse."""
    co_count = np.zeros(graph.matrix.nnz, dtype=np.int32)
    for row in range(len(graph.nodes)):
        for pos in range(graph.matrix.indptr[row], graph.matrix.indptr[row + 1]):
            col = graph.matrix.indices[pos]
            key = (min(row, col), max(row, col))
            co_count[pos] = graph.raw_pair_counts.get(key, 0)
    undirected = sorted(graph.retained_edge_weights)
    return {"nodes": np.asarray(graph.nodes, dtype="U32"),
            "indptr": graph.matrix.indptr.astype(np.int32),
            "indices": graph.matrix.indices.astype(np.int32),
            "transition_data": graph.matrix.data.astype(np.float64),
            "common_viewer_count_per_directed_edge": co_count,
            "undirected_edge_i": np.asarray([i for i, _ in undirected], dtype=np.int32),
            "undirected_edge_j": np.asarray([j for _, j in undirected], dtype=np.int32),
            "undirected_edge_weight": np.asarray([graph.retained_edge_weights[e] for e in undirected], dtype=np.float64),
            "undirected_common_viewers": np.asarray([graph.raw_pair_counts[e] for e in undirected], dtype=np.int32),
            "weight_type": np.asarray(graph.weight_type),
            "max_neighbors": np.asarray(graph.max_neighbors),
            "history_cap": np.asarray(graph.history_cap),
            "min_common_viewers": np.asarray(graph.min_common_viewers)}
