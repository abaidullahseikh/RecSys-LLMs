"""Frozen evaluation protocol over existing Step 2 files. No preprocessing/model fit.

Training input access is role- and hash-checked. Validation labels are separately
loaded for evaluation. Test schedule access is availability-only. These are API
guardrails, not an OS sandbox for future code that bypasses them.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

from data_pipeline import parse_timestamp, SCHEDULE_COLUMNS
from evaluation import (candidate_pool, evaluate_predictions, FALLBACK_POLICY,
                        identity_hash, ordered_cohort, program_id, user_id)

ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "data" / "step2"
INTERACTION_COLUMNS = ["user_id", "tv_show_id", "qualifying_airings", "watched_seconds", "first_watch_utc", "last_watch_utc"]
TRUTH_COLUMNS = ["user_id", "relevant_program_ids", "relevant_count"]
WINDOWS = {"train": (1583704800, 1589749200), "validation": (1589749200, 1595797200),
           "history": (1583704800, 1595797200), "test": (1595797200, 1603054800)}
POLICY = {"metric": "MAP@5", "k": 5, "ap_denominator": "min(5,distinct_relevant_items)",
          "primary_population": "all_template_users_in_original_order", "empty_label_ap": 0,
          "diagnostic_population": "same_cohort_restricted_to_nonempty_labels",
          "ground_truth": "all_distinct_validation_programs_with_qualifying_airing",
          "ground_truth_top5_truncation": False, "threshold_seconds_per_airing": 120,
          "repeat_previously_watched": True, "duplicates_within_ranking": "reject",
          "production_prediction_length": 5, "unknown_or_zero_predictions": "reject",
          "timezone": "Europe/Kyiv", "timezone_status": "evidence-supported inference, not official",
          "fallback_policy": FALLBACK_POLICY, "fallback_fitting_stage": 4,
          "hidden_Kaggle_labels_reproduced": False}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def records(path, header):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != header:
            raise ValueError(f"Schema mismatch: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed record at {path}:{line}")
            yield row


def load_cohort(path):
    return ordered_cohort([user_id(row[0]) for row in records(path, ["user_id"])])


def load_candidates(path, expected_hash=None):
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError("Candidate file checksum mismatch")
    supplied = tuple(program_id(row[0]) for row in records(path, ["tv_show_id"]))
    canonical = candidate_pool(supplied)
    if supplied != canonical:
        raise ValueError("Candidate file must use deterministic numeric-ID order")
    if len(canonical) < 5:
        raise ValueError("At least five candidates required")
    return canonical


def candidates_from_airings(airings, window):
    """Availability only: (start_utc, end_utc, raw_program_id), no user argument."""
    ids = set()
    for start, end, raw in airings:
        if raw in (0, "0"):
            continue
        item = program_id(raw)
        if start < window[1] and end > window[0] and end > start:
            ids.add(item)
    return candidate_pool(ids)


def schedule_program_id(raw):
    """Match Step 2's exact numeric-ID representation, including raw leading zeros.

    Raw CSV text is never changed. This conversion is confined to schedule
    metadata; prediction and processed-data tokens remain strictly canonical.
    """
    if type(raw) is not str or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"Invalid raw numeric schedule ID: {raw!r}")
    return str(int(raw))


def schedule_catalogs(schedule_paths):
    """Recheck metadata-only schedule availability; never joins viewing records."""
    pools = {p: set() for p in ("train", "validation", "test")}
    for path in schedule_paths:
        for row in records(path, SCHEDULE_COLUMNS):
            item = schedule_program_id(row[5])
            if item == "0":
                continue
            start = parse_timestamp(row[2], schedule=True)
            duration = int(row[3])
            if duration <= 0:
                continue
            for period, pool in pools.items():
                lo, hi = WINDOWS[period]
                if start < hi and start + duration > lo:
                    pool.add(item)
    return {p: candidate_pool(ids) for p, ids in pools.items()}


def positive_integer(value):
    if not value.isascii() or not value.isdecimal() or value.startswith("0"):
        raise ValueError(f"Noncanonical positive integer field: {value!r}")
    return int(value)


@dataclass(frozen=True)
class Interaction:
    user_id: str
    program_id: str
    qualifying_airings: int
    watched_seconds: int
    first_watch_utc: int
    last_watch_utc: int


def iter_interactions(path, window, candidates, *, threshold=120, expected_hash=None):
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError("Interaction source checksum mismatch")
    allowed = frozenset(candidates)
    previous = None
    for line, row in enumerate(records(path, INTERACTION_COLUMNS), 2):
        u, item = user_id(row[0]), program_id(row[1])
        count, seconds, first, last = map(positive_integer, row[2:])
        key = (u, int(item))
        if previous is not None and key <= previous:
            raise ValueError(f"Duplicate or unsorted user/program record: {path}:{line}")
        previous = key
        if item not in allowed:
            raise ValueError(f"Program outside period catalog: {path}:{line}, {item}")
        if not (window[0] <= first < last <= window[1]):
            raise ValueError(f"Temporal leakage at {path}:{line}: {first},{last} outside {window}")
        if seconds < count * threshold:
            raise ValueError(f"Inconsistent qualifying-airing duration at {path}:{line}")
        yield Interaction(u, item, count, seconds, first, last)


def summarize(rows, cohort, collect_truth=False):
    cohort_set = set(cohort)
    counts, users, items, per_user = Counter(), set(), set(), Counter()
    truth = {u: set() for u in cohort} if collect_truth else None
    first, last = None, None
    for r in rows:
        counts["pairs"] += 1
        counts["qualifying_airings"] += r.qualifying_airings
        counts["watched_seconds"] += r.watched_seconds
        users.add(r.user_id)
        items.add(r.program_id)
        first = min(first, r.first_watch_utc) if first is not None else r.first_watch_utc
        last = max(last, r.last_watch_utc) if last is not None else r.last_watch_utc
        if r.user_id in cohort_set:
            per_user[r.user_id] += 1
            if collect_truth:
                truth[r.user_id].add(r.program_id)
    return {**counts, "users": len(users), "programs": len(items), "template_users": len(per_user),
            "template_empty": len(cohort) - len(per_user), "template_pairs": sum(per_user.values()),
            "first_watch_utc": first, "last_watch_utc": last}, per_user, truth


def save_truth(path, cohort, truth):
    if set(truth) != set(cohort):
        raise ValueError("Truth/cohort mismatch")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRUTH_COLUMNS)
        for u in cohort:
            labels = candidate_pool(truth[u])
            writer.writerow((u, " ".join(labels), len(labels)))


def load_truth(path, cohort, candidates):
    truth, actual_order = {}, []
    allowed = frozenset(candidates)
    for raw_user, field, raw_count in records(path, TRUTH_COLUMNS):
        u = user_id(raw_user)
        if u in truth:
            raise ValueError("Duplicate truth user")
        labels = tuple(field.split(" ")) if field else ()
        normalized = candidate_pool(labels)
        if labels != normalized or str(len(labels)) != raw_count or not set(labels).issubset(allowed):
            raise ValueError("Malformed or ineligible ground truth")
        truth[u] = frozenset(labels)
        actual_order.append(u)
    if tuple(actual_order) != tuple(cohort):
        raise ValueError("Ground-truth user population/order mismatch")
    return truth


def describe(path, **extra):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size, **extra}


def verified_path(spec):
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or sha256(path) != spec["sha256"]:
        raise ValueError(f"Registered file checksum/path mismatch: {path}")
    return path


def build(output=ROOT / "data" / "step3"):
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite the frozen Step 3 protocol; use a fresh directory")
    inspection = json_read(ROOT / "results" / "step3_input_inspection.json")
    for relative, info in inspection.items():
        if sha256(ROOT / relative) != info["sha256"]:
            raise ValueError(f"Input changed since Step 3 inspection: {relative}")
    join = json_read(STEP2 / "join_manifest.json")
    selected = json_read(STEP2 / "threshold_120s" / "interaction_manifest.json")
    if ({p: tuple(w) for p, w in join["windows_utc"].items()} != WINDOWS
            or selected["threshold_seconds"] != 120 or not json_read(ROOT / "results" / "step2_verification.json")["all_current_checks_passed"]):
        raise ValueError("Step 2 verification, windows or threshold differ from frozen protocol")
    cohort = load_cohort(STEP2 / "template_users.csv")
    original_template = join["source_files_before"]["submission.csv"]["path"]
    if tuple(row[0] for row in records(original_template, ["user_id", "tv_show_id"])) != cohort:
        raise ValueError("Original template user IDs/order changed")
    pools = {p: load_candidates(STEP2 / f"{p}_candidates.csv", join["artifacts"][f"{p}_candidates.csv"]["sha256"])
             for p in ("train", "validation", "test")}
    schedules = [join["source_files_before"][name]["path"] for name in (
        "export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv")]
    availability = schedule_catalogs(schedules)
    if pools != availability:
        raise ValueError("Candidate IDs disagree with timestamp-clipped schedule metadata")
    summaries, presence, truth = {}, {}, None
    specs = {}
    for period in ("train", "validation"):
        path = STEP2 / "threshold_120s" / f"{period}_interactions.csv.gz"
        expected = selected["artifacts"][path.name]["sha256"]
        summary, counts, labels = summarize(iter_interactions(path, WINDOWS[period], pools[period], expected_hash=expected),
                                           cohort, collect_truth=(period == "validation"))
        summaries[period], presence[period] = summary, counts
        specs[period] = describe(path, role="training_only" if period == "train" else "labels_only", window_utc=WINDOWS[period])
        old = selected["periods"][period]
        for key, old_key in (("pairs", "unique_user_program_pairs"), ("users", "unique_users"), ("programs", "unique_programs"),
                             ("template_users", "template_users_with_interactions"), ("template_pairs", "template_user_program_pairs")):
            if summary[key] != old[old_key]:
                raise ValueError(f"Step 2 count mismatch: {period}/{key}")
        if labels is not None:
            truth = labels
    if (len(cohort), summaries["train"]["template_users"], summaries["validation"]["template_users"]) != (1260, 1250, 1246):
        raise ValueError("Unexpected cohort coverage: investigate instead of dropping users")
    output.mkdir(parents=True, exist_ok=False)
    save_truth(output / "validation_truth.csv", cohort, truth)
    with (output / "evaluation_cohort.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "has_training_history", "validation_relevant_count"])
        for u in cohort:
            writer.writerow([u, int(presence["train"][u] > 0), len(truth[u])])
    input_products = {relative: {"sha256": info["sha256"], "path": relative} for relative, info in inspection.items()}
    config = {"schema_version": 1, "policies": POLICY, "windows_utc": WINDOWS,
              "cohort_ids": cohort, "cohort_order_sha256": identity_hash(cohort),
              "training_input": specs["train"], "validation_labels_source": specs["validation"],
              "history_input_policy": "for later final fitting only; forbidden in hold-out training API",
              "truth": describe(output / "validation_truth.csv"), "cohort_file": describe(output / "evaluation_cohort.csv"),
              "candidate_pools": {p: describe(STEP2 / f"{p}_candidates.csv", count=len(ids), ids_sha256=identity_hash(ids),
                                              window_utc=WINDOWS[p], source="schedule_only", sort="numeric_id_ascending")
                                  for p, ids in pools.items()},
              "input_products": input_products, "coverage": summaries,
              "schedule_source_hashes": {str(Path(p)): sha256(p) for p in schedules},
              "python": platform.python_version(), "dependencies": "Python standard library; system IANA timezone data",
              "production_random_seed": None, "test_random_seed": 42}
    semantic = {k: config[k] for k in ("policies", "windows_utc", "cohort_order_sha256", "input_products")}
    config["protocol_id"] = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = output / "evaluation_config.json"
    json_write(path, config)
    (output / "evaluation_config.sha256").write_text(sha256(path) + "\n")
    result = {"seconds": round(time.monotonic() - started, 3), "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "config_sha256": sha256(path), "protocol_id": config["protocol_id"], "coverage": summaries,
              "candidate_counts": {p: len(ids) for p, ids in pools.items()}, "schedule_catalogs_exact_match": True}
    json_write(ROOT / "results" / "step3_build_stats.json", result)
    print(json.dumps(result, indent=2))
    return config


def load_config(path=ROOT / "data" / "step3" / "evaluation_config.json"):
    path = Path(path)
    if sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen evaluation configuration checksum mismatch")
    config = json_read(path)
    if config["policies"] != POLICY or {p: tuple(w) for p, w in config["windows_utc"].items()} != WINDOWS:
        raise ValueError("Evaluation policy/windows changed")
    cohort = ordered_cohort(config["cohort_ids"])
    if identity_hash(cohort) != config["cohort_order_sha256"]:
        raise ValueError("Cohort identity/order hash mismatch")
    return config


@dataclass(frozen=True)
class TrainingInput:
    path: Path
    sha256: str
    window_utc: tuple
    candidate_ids: tuple

    def rows(self):
        return iter_interactions(self.path, self.window_utc, self.candidate_ids, expected_hash=self.sha256)


def training_input(config, proposed_path=None):
    """Only the registered Week 11–20 file can enter the hold-out training API."""
    spec = config["training_input"]
    registered = ROOT / "data" / "step2" / "threshold_120s" / "train_interactions.csv.gz"
    known = config["input_products"][str(registered.relative_to(ROOT))]["sha256"]
    if spec["role"] != "training_only" or tuple(spec["window_utc"]) != WINDOWS["train"] or spec["sha256"] != known:
        raise ValueError("Invalid training role/window/provenance")
    path = verified_path(spec)
    if path != registered.resolve() or (proposed_path is not None and Path(proposed_path).resolve() != path):
        raise ValueError("Validation/history/future files are forbidden as hold-out training input")
    pool = load_candidates(verified_path(config["candidate_pools"]["train"]))
    return TrainingInput(path, spec["sha256"], WINDOWS["train"], pool)


class EvaluationProtocol:
    def __init__(self, config_path=ROOT / "data" / "step3" / "evaluation_config.json"):
        self.config = load_config(config_path)
        self.cohort = ordered_cohort(self.config["cohort_ids"])
        self.candidates = load_candidates(verified_path(self.config["candidate_pools"]["validation"]))
        self.truth = load_truth(verified_path(self.config["truth"]), self.cohort, self.candidates)
        self.config_sha256 = sha256(config_path)

    def evaluate(self, predictions):
        result = evaluate_predictions(predictions, self.truth, self.cohort, self.candidates, strict=True)
        result.update(protocol_id=self.config["protocol_id"], evaluation_config_sha256=self.config_sha256,
                      truth_sha256=self.config["truth"]["sha256"])
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "step3")
    args = parser.parse_args()
    build(args.output_dir)
