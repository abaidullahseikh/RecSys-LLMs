"""Sparse pairwise Association Rules for SWEET.TV (training-only)."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import csv
import gzip
import math
import time

import numpy as np
from scipy.sparse import csr_matrix, triu

from evaluation import candidate_pool, program_id
from validation import ROOT, WINDOWS, iter_interactions, training_input

TRANSACTION_TYPES = ("user_history", "time_window")
SCORE_TYPES = ("confidence", "confidence_log_lift")
ANTECEDENT_WEIGHTS = ("uniform", "log_frequency")
WINDOW_SIZES_DAYS = (7, 14)
OVERLAP_COLUMNS = ["user_id", "channel_id", "broadcast_start_utc", "broadcast_end_utc",
                   "tv_show_id", "overlap_seconds", "first_overlap_utc", "last_overlap_utc"]


@dataclass(frozen=True)
class TransactionDataset:
    transaction_type: str
    window_days: int | None
    transactions: tuple
    item_index: dict
    index_item: tuple
    item_support: dict
    pair_counts: dict
    transaction_count: int
    pair_occurrences: int
    average_transaction_size: float
    max_transaction_size: int
    p95_transaction_size: int
    training_sha256: str
    training_window_utc: tuple
    source_sha256: str
    build_seconds: float
    pair_mining_seconds: float


@dataclass(frozen=True)
class AssociationModel:
    dataset: TransactionDataset
    rules_by_antecedent: dict
    min_support: float
    min_confidence: float
    min_cooccurrence: int
    min_lift: float
    top_neighbors: int
    build_seconds: float

    @property
    def rule_count(self):
        return sum(len(rules) for rules in self.rules_by_antecedent.values())


def canonical_transaction(items):
    return tuple(sorted({program_id(item) for item in items}, key=int))


def user_history_transactions(training_rows):
    """One transaction per training user, consisting of distinct program IDs."""
    baskets = defaultdict(set)
    for row in training_rows:
        baskets[row.user_id].add(program_id(row.program_id))
    users = tuple(sorted(baskets))
    return tuple((user, canonical_transaction(baskets[user])) for user in users if baskets[user])


def window_index(timestamp, window, window_seconds):
    if window_seconds <= 0:
        raise ValueError("Window duration must be positive")
    lo, hi = window
    if not lo <= timestamp < hi:
        raise ValueError("Timestamp is outside the half-open training window")
    return (timestamp - lo) // window_seconds


def time_window_transactions_from_airings(rows, window, window_days, threshold_seconds=120):
    """One (user, fixed UTC window) set of qualifying aired programs.

    `first_overlap_utc` is within the Step 2 clipped training period and serves as
    the deterministic airing anchor. Each broadcast is counted once per user
    within the window. Boundary-crossing interactions cannot enter validation.
    """
    seconds = int(window_days) * 86400
    if window_days <= 0 or threshold_seconds <= 0:
        raise ValueError("Window size and qualifying threshold must be positive")
    baskets = defaultdict(set)
    for row in rows:
        user = row["user_id"]
        item = int(row["tv_show_id"])
        overlap = int(row["overlap_seconds"])
        first = int(row["first_overlap_utc"])
        if item <= 0 or overlap < threshold_seconds:
            continue
        bucket = window_index(first, window, seconds)
        baskets[(user, bucket)].add(str(item))
    return tuple((user, bucket, canonical_transaction(items))
                 for (user, bucket), items in sorted(baskets.items()) if items)


def load_window_transactions(config, window_days, threshold_seconds=120):
    source = training_input(config)
    path = ROOT / "data" / "step2" / "train_airing_overlaps.csv.gz"
    manifest = config["input_products"].get("data/step2/join_manifest.json")
    if not manifest:
        raise ValueError("Step 3 config does not register the Step 2 join manifest")
    # The artifact hash is registered in the join manifest, not the Step 3 product map.
    join_path = ROOT / "data" / "step2" / "join_manifest.json"
    import json
    join = json.loads(join_path.read_text())
    expected = join["artifacts"][path.name]["sha256"]
    if validation_hash(path) != expected:
        raise ValueError("Training airing-overlap artifact hash mismatch")
    baskets = defaultdict(set)
    lo, hi = source.window_utc
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != OVERLAP_COLUMNS:
            raise ValueError("Training airing-overlap schema mismatch")
        for row in reader:
            user, item = row["user_id"], int(row["tv_show_id"])
            overlap, first, last = int(row["overlap_seconds"]), int(row["first_overlap_utc"]), int(row["last_overlap_utc"])
            if not (lo <= first < last <= hi):
                raise ValueError("Training airing overlap escapes Week 11–20 boundary")
            if item > 0 and overlap >= threshold_seconds:
                bucket = window_index(first, source.window_utc, int(window_days) * 86400)
                baskets[(user, bucket)].add(str(item))
    transactions = tuple((user, bucket, canonical_transaction(items))
                         for (user, bucket), items in sorted(baskets.items()) if items)
    return transactions, expected, source


def validation_hash(path):
    import hashlib
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mine_sparse_pair_counts(transactions):
    """Construct binary transaction-item CSR and mine all distinct co-pair counts."""
    started = time.monotonic()
    baskets = [canonical_transaction(tx[-1]) for tx in transactions]
    items = tuple(sorted({item for basket in baskets for item in basket}, key=int))
    index = {item: i for i, item in enumerate(items)}
    rows, cols = [], []
    pair_occurrences, sizes = 0, []
    for row_num, basket in enumerate(baskets):
        ids = [index[item] for item in basket]
        sizes.append(len(ids))
        rows.extend([row_num] * len(ids))
        cols.extend(ids)
        pair_occurrences += len(ids) * (len(ids) - 1) // 2
    matrix = csr_matrix((np.ones(len(rows), dtype=np.int32),
                         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
                        shape=(len(baskets), len(items)), dtype=np.int32)
    support_array = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    item_support = {item: int(support_array[i]) for i, item in enumerate(items)}
    co = (matrix.T @ matrix).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    upper = triu(co, k=1, format="coo")
    pair_counts = {(int(i), int(j)): int(count) for i, j, count in zip(upper.row, upper.col, upper.data)}
    sizes_sorted = sorted(sizes)
    p95 = sizes_sorted[max(0, math.ceil(.95 * len(sizes_sorted)) - 1)] if sizes_sorted else 0
    return items, index, item_support, pair_counts, pair_occurrences, sizes, p95, time.monotonic() - started


def build_dataset(transactions, *, transaction_type, window_days, training_sha256,
                  training_window_utc, source_sha256, transaction_build_seconds=0):
    if transaction_type not in TRANSACTION_TYPES:
        raise ValueError("Unknown transaction type")
    mined = mine_sparse_pair_counts(transactions)
    items, index, supports, pairs, pair_occ, sizes, p95, mine_seconds = mined
    return TransactionDataset(transaction_type, window_days, tuple(transactions), index, items,
        supports, pairs, len(transactions), pair_occ,
        sum(sizes) / len(sizes) if sizes else 0.0, max(sizes, default=0), p95,
        training_sha256, tuple(training_window_utc), source_sha256,
        transaction_build_seconds, mine_seconds)


def build_history_dataset(config):
    started = time.monotonic()
    source = training_input(config)
    transactions = user_history_transactions(source.rows())
    build_seconds = time.monotonic() - started
    return build_dataset(transactions, transaction_type="user_history", window_days=None,
                         training_sha256=source.sha256, training_window_utc=source.window_utc,
                         source_sha256=source.sha256, transaction_build_seconds=build_seconds)


def build_window_dataset(config, window_days):
    started = time.monotonic()
    transactions, source_hash, source = load_window_transactions(config, window_days)
    build_seconds = time.monotonic() - started
    return build_dataset(transactions, transaction_type="time_window", window_days=window_days,
                         training_sha256=source.sha256, training_window_utc=source.window_utc,
                         source_sha256=source_hash, transaction_build_seconds=build_seconds)


def association_metrics(pair_count, antecedent_count, consequent_count, transaction_count):
    if min(pair_count, antecedent_count, consequent_count, transaction_count) <= 0 or pair_count > min(antecedent_count, consequent_count):
        raise ValueError("Invalid association counts")
    support = pair_count / transaction_count
    confidence = pair_count / antecedent_count
    lift = confidence / (consequent_count / transaction_count)
    return support, confidence, lift


def mine_rules(dataset, *, min_support=0.0005, min_confidence=0.01, min_cooccurrence=3,
               min_lift=1.0, top_neighbors=50):
    if min_support < 0 or not (0 <= min_confidence <= 1) or min_cooccurrence < 1 or min_lift < 0 or top_neighbors < 1:
        raise ValueError("Invalid association-rule filters")
    started = time.monotonic()
    by_a = defaultdict(list)
    n = dataset.transaction_count
    for (i, j), pair_count in dataset.pair_counts.items():
        a, b = dataset.index_item[i], dataset.index_item[j]
        for antecedent, consequent in ((a,b),(b,a)):
            support, confidence, lift = association_metrics(pair_count,
                dataset.item_support[antecedent], dataset.item_support[consequent], n)
            if (pair_count >= min_cooccurrence and support >= min_support
                    and confidence >= min_confidence and lift >= min_lift):
                by_a[antecedent].append({"antecedent":antecedent,"consequent":consequent,
                    "pair_count":pair_count,"support":support,"confidence":confidence,"lift":lift})
    for antecedent in by_a:
        by_a[antecedent].sort(key=lambda rule:(-rule["confidence"],-rule["lift"],-rule["pair_count"],int(rule["consequent"])))
        by_a[antecedent]=tuple(by_a[antecedent][:top_neighbors])
    return AssociationModel(dataset,dict(by_a),min_support,min_confidence,min_cooccurrence,
                            min_lift,top_neighbors,time.monotonic()-started)


def score_user(model, training_profile, candidates, *, score_type="confidence", antecedent_weight="log_frequency"):
    if score_type not in SCORE_TYPES or antecedent_weight not in ANTECEDENT_WEIGHTS:
        raise ValueError("Unknown rule scoring/antecedent weighting")
    eligible=frozenset(candidate_pool(candidates))
    if antecedent_weight=="uniform":
        weights={item:1.0 for item in training_profile}
    else:
        weights={item:math.log1p(values[0]) for item,values in training_profile.items()}
    total=math.fsum(weights.values())
    if total<=0:
        return {}, False
    normalized={item:w/total for item,w in weights.items()}
    scores=Counter()
    matched=False
    for antecedent,weight in normalized.items():
        for rule in model.rules_by_antecedent.get(antecedent,()):
            consequent=rule["consequent"]
            if consequent not in eligible:
                continue
            matched=True
            if score_type=="confidence":
                value=rule["confidence"]
            else:
                # Log normalization damps extreme lifts from rare consequents.
                value=rule["confidence"]*math.log1p(min(rule["lift"],100.0))
            scores[consequent]+=weight*value
    return dict(scores), matched


def recommend_user(model, training_profile, candidates, popularity_fallback, *,
                   score_type="confidence", antecedent_weight="log_frequency"):
    pool=candidate_pool(candidates)
    fallback=tuple(program_id(i) for i in popularity_fallback)
    if len(pool)<5 or len(fallback)<5 or len(set(fallback))!=len(fallback) or not set(fallback).issubset(set(pool)):
        raise ValueError("Invalid validation candidate pool or popularity fallback")
    scores,matched=score_user(model,training_profile,pool,score_type=score_type,
                              antecedent_weight=antecedent_weight)
    ranked=sorted(scores,key=lambda item:(-scores[item],-model.dataset.item_support.get(item,0),int(item)))
    seen=set(ranked)
    result=ranked[:5]
    seen=set(result)
    fallback_slots=5-len(result)
    for item in fallback:
        if len(result)>=5:
            break
        if item not in seen:
            result.append(item)
            seen.add(item)
    if len(result)!=5 or len(set(result))!=5:
        raise AssertionError("Association Rules did not produce a strict Top-5")
    return tuple(result),{"matched_rule_user":matched,"fallback_slots":fallback_slots,
                          "cold_start":not bool(training_profile),"scored_candidate_count":len(scores)}
