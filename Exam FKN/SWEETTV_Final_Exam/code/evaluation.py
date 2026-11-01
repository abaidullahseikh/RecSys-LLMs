"""FKN AP@5/MAP@5, strict prediction validation, and fallback consumption only.

No recommendation method or popularity statistics are fitted in this module.
User keys are exact strings. Program tokens are canonical positive decimal IDs;
Python ints are accepted losslessly, but floats, bools and normalized text are not.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

K = 5
FALLBACK_POLICY = "distinct_training_users_desc,qualifying_airings_desc,numeric_program_id_asc"


def program_id(value):
    if type(value) is int and value > 0:
        return str(value)
    if (type(value) is str and value.isascii() and value.isdecimal()
            and value[0] != "0"):
        return value
    raise ValueError(f"Not a canonical positive program ID: {value!r}")


def user_id(value):
    if type(value) is not str or not value or not value.isascii() or not value.isdecimal():
        raise ValueError("User IDs must be unchanged decimal strings")
    return value


def ordered_cohort(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("A nonempty ordered cohort is required")
    result = tuple(user_id(u) for u in values)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate cohort user")
    return result


def candidate_pool(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("Candidate pool must be a collection of IDs")
    values = tuple(program_id(v) for v in values)
    if len(values) != len(set(values)):
        raise ValueError("Duplicate candidate ID")
    return tuple(sorted(values, key=int))


def identity_hash(values):
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def strict_top5(predictions, candidates):
    if not isinstance(predictions, (list, tuple)) or len(predictions) != K:
        raise ValueError("Production predictions must be an ordered list of exactly five IDs")
    if candidates is None:
        raise ValueError("Strict evaluation requires an eligible candidate pool")
    items = tuple(program_id(p) for p in predictions)
    candidates = {program_id(p) for p in candidates}
    if len(set(items)) != K:
        raise ValueError("Duplicate recommendation ID")
    if any(p not in candidates for p in items):
        raise ValueError("Unknown or unavailable program ID")
    return items


def average_precision_at_5(relevant, predictions, *, strict=False, candidates=None):
    """AP denominator=min(5, unique relevant items); empty relevance gives zero.

    Non-strict mathematical mode allows short/empty/duplicate rankings. Invalid
    or ineligible tokens receive no credit and still consume their original rank.
    Only the first five positions are scored. Production callers use strict=True.
    """
    if isinstance(relevant, (str, bytes)) or not isinstance(predictions, (list, tuple)):
        raise ValueError("Relevance must be a collection and predictions must be ordered")
    truth = {program_id(p) for p in relevant}
    eligible = None if candidates is None else {program_id(p) for p in candidates}
    if eligible is not None and not truth.issubset(eligible):
        raise ValueError("Ground truth contains an ineligible ID")
    if strict:
        predictions = strict_top5(predictions, eligible)
    seen, hits, contributions = set(), 0, []
    for rank, raw in enumerate(predictions[:K], 1):
        try:
            item = program_id(raw)
        except ValueError:
            if strict:
                raise
            continue
        if item in seen:
            continue
        seen.add(item)
        if item in truth and (eligible is None or item in eligible):
            hits += 1
            contributions.append(hits / rank)
    return math.fsum(contributions) / min(K, len(truth)) if truth else 0.0


def evaluate_predictions(predictions, truth, cohort, candidates, *, strict=True):
    """Return primary all-user MAP, separate nonempty diagnostic, and ordered APs."""
    cohort = ordered_cohort(cohort)
    eligible = frozenset(candidate_pool(candidates))
    if not isinstance(predictions, Mapping) or not isinstance(truth, Mapping):
        raise ValueError("Predictions and truth must be user-keyed mappings")
    for mapping in (predictions, truth):
        for u in mapping:
            user_id(u)
    cohort_set = set(cohort)
    if set(truth) != cohort_set:
        raise ValueError("Truth must explicitly contain every cohort user, including empty labels")
    if set(predictions) - cohort_set or (strict and set(predictions) != cohort_set):
        raise ValueError("Prediction cohort mismatch or missing user")
    per_user, labelled = {}, []
    for u in cohort:
        # Validate even empty-label users' predictions; invalid rows never hide behind AP=0.
        if isinstance(truth[u], (str, bytes)):
            raise ValueError("Each user's truth must be a collection of IDs")
        labels = tuple(program_id(p) for p in truth[u])
        ap = average_precision_at_5(labels, predictions.get(u, []), strict=strict, candidates=eligible)
        per_user[u] = ap
        if labels:
            labelled.append(ap)
    return {"primary_map_at_5": math.fsum(per_user.values()) / len(cohort),
            "diagnostic_map_at_5_nonempty": math.fsum(labelled) / len(labelled) if labelled else None,
            "per_user_ap_at_5": per_user, "primary_users": len(cohort),
            "nonempty_users": len(labelled), "empty_users": len(cohort) - len(labelled),
            "mode": "strict_production" if strict else "lenient_math_diagnostic",
            "cohort_order_sha256": identity_hash(cohort),
            "candidate_ids_sha256": identity_hash(sorted(eligible, key=int))}


def parse_prediction_records(records):
    """Parse exact two-field CSV-style records without losing duplicate user rows."""
    result = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {"user_id", "tv_show_id"}:
            raise ValueError("Prediction record must have exactly user_id,tv_show_id")
        u = user_id(row["user_id"])
        value = row["tv_show_id"]
        if u in result or type(value) is not str:
            raise ValueError("Duplicate user or malformed prediction field")
        tokens = value.split(" ")
        if len(tokens) != K:
            raise ValueError("Exactly five single-space-separated tokens required")
        items = tuple(program_id(t) for t in tokens)
        if len(set(items)) != K:
            raise ValueError("Duplicate recommendation ID")
        result[u] = items
    return result


@dataclass(frozen=True)
class FallbackRanking:
    """Step 4 will supply this training-only ranking; Step 3 only consumes it.

    Provenance assertions prevent accidental split/catalog mismatch. They do not
    establish that a future model honestly computed its statistics; verify that
    fitting implementation when it exists. No method to fit a ranking is provided.
    """

    program_ids: tuple
    training_sha256: str
    training_window_utc: tuple
    candidate_ids_sha256: str
    policy: str = FALLBACK_POLICY


def complete_top5(ranked_ids, fallback, candidates, *, training_sha256, training_window_utc):
    """Shared deterministic completion; requires an externally supplied fallback."""
    pool = candidate_pool(candidates)
    if len(pool) < K or not isinstance(ranked_ids, (tuple, list)):
        raise ValueError("Need five candidates and an ordered ranking")
    if not isinstance(fallback, FallbackRanking):
        raise ValueError("A provenance-tagged fallback ranking is required")
    if (fallback.training_sha256 != training_sha256
            or tuple(fallback.training_window_utc) != tuple(training_window_utc)
            or fallback.candidate_ids_sha256 != identity_hash(pool)
            or fallback.policy != FALLBACK_POLICY):
        raise ValueError("Fallback training/catalog provenance mismatch")
    order = tuple(program_id(p) for p in fallback.program_ids)
    if len(set(order)) != len(order) or set(order) != set(pool):
        raise ValueError("Fallback must rank every eligible candidate exactly once")
    allowed, seen, result = set(pool), set(), []
    for raw in (*ranked_ids, *order):
        item = program_id(raw)
        if item in allowed and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) == K:
            return tuple(result)
    raise AssertionError("Unable to complete five eligible IDs")
