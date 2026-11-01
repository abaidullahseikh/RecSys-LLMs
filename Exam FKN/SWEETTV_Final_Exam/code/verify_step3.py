"""Step 3 input snapshot and independent validation evidence; no model fitting."""

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
from datetime import datetime, timezone
from fractions import Fraction
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import resource
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "data" / "step2"
REQUIRED = [STEP2 / "threshold_120s" / f"{p}_interactions.csv.gz" for p in ("train", "validation", "history")]
REQUIRED += [STEP2 / f"{p}_candidates.csv" for p in ("train", "validation", "test")]
REQUIRED += [STEP2 / "template_users.csv", STEP2 / "join_manifest.json", STEP2 / "threshold_120s" / "interaction_manifest.json"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def snapshot():
    started = time.monotonic()
    target = ROOT / "results" / "step3_inputs_before.json"
    if target.exists():
        raise FileExistsError("Preserve the existing Step 3 initial integrity snapshot")
    originals = read_json(ROOT / "results" / "original_sha256_before.json")
    paths = {Path(p) for p in originals}
    paths.add(Path(read_json(ROOT / "results" / "homework_guidelines_sha256.json")["path"]))
    for p in ROOT.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if "step3" in str(p.relative_to(ROOT)) or p.name in {"evaluation.py", "validation.py", "fallback.py"}:
            continue
        paths.add(p)
    protected = {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(paths)}
    for path, old in originals.items():
        if protected[path] != old:
            raise ValueError(f"Original differs from Step 1: {path}")
    for manifest_path in (STEP2 / "join_manifest.json", STEP2 / "threshold_120s" / "interaction_manifest.json",
                          STEP2 / "threshold_300s" / "interaction_manifest.json"):
        for name, old in read_json(manifest_path)["artifacts"].items():
            if protected[str(manifest_path.parent / name)] != old:
                raise ValueError(f"Step 2 artifact differs: {name}")
    previews = {}
    for path in REQUIRED:
        if path.suffix == ".json":
            previews[str(path.relative_to(ROOT))] = {"sha256": digest(path), "top_level_keys": list(read_json(path))}
        else:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                previews[str(path.relative_to(ROOT))] = {"sha256": digest(path), "header": next(reader), "sample_rows": list(itertools.islice(reader, 2))}
    save_json(target, {"files": protected, "seconds": round(time.monotonic() - started, 3)})
    save_json(ROOT / "results" / "step3_input_inspection.json", previews)
    print(json.dumps({"protected_files": len(protected), "previews": previews}, indent=2))


def inspect_schedule_ids():
    join = read_json(STEP2 / "join_manifest.json")
    result = {}
    for name, spec in join["source_files_before"].items():
        if not name.startswith("export"):
            continue
        variants = defaultdict(set)
        counts, examples = Counter(), []
        with Path(spec["path"]).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw = row["tv_show_id"]
                numeric = int(raw)
                counts["records"] += 1
                if len(raw) > 1 and raw.startswith("0"):
                    counts["leading_zero_rows"] += 1
                    variants[str(numeric)].add(raw)
                    if len(examples) < 5:
                        examples.append({"raw": raw, "processed_numeric_id": str(numeric), "title": row["tv_show_title"]})
        result[name] = {**counts, "distinct_affected_numeric_ids": len(variants),
                        "variants": {k: sorted(v) for k, v in variants.items()}, "examples": examples}
    save_json(ROOT / "results" / "step3_schedule_id_inspection.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def verify():
    import evaluation as ev
    import validation as val
    from test_step3 import reference_ap

    started = time.monotonic()
    protocol = val.EvaluationProtocol()
    config = protocol.config
    checks = []

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    def reject(name, operation):
        try:
            operation()
        except ValueError as exc:
            check(name, True, str(exc))
        else:
            check(name, False, "Expected rejection did not occur")

    before = read_json(ROOT / "results" / "step3_inputs_before.json")
    integrity = {}
    for path, expected in before["files"].items():
        current = {"bytes": Path(path).stat().st_size, "sha256": digest(path)}
        integrity[path] = current
        check("preserved:" + str(Path(path).name), current == expected)
    check("cohort_1260_exact_order", len(protocol.cohort) == 1260 and protocol.cohort == val.load_cohort(STEP2 / "template_users.csv"))
    join = read_json(STEP2 / "join_manifest.json")
    original_cohort = tuple(r[0] for r in val.records(join["source_files_before"]["submission.csv"]["path"], ["user_id", "tv_show_id"]))
    check("original_template_IDs_preserved", original_cohort == protocol.cohort)
    check("cohort_order_hash", ev.identity_hash(protocol.cohort) == config["cohort_order_sha256"])
    check("cohort_manifest_hash", digest(ROOT / config["cohort_file"]["path"]) == config["cohort_file"]["sha256"])
    pools = {p: val.load_candidates(val.verified_path(spec)) for p, spec in config["candidate_pools"].items()}
    for p, pool in pools.items():
        spec = config["candidate_pools"][p]
        check(f"candidates:{p}:count_order_identity", len(pool) == spec["count"] and ev.identity_hash(pool) == spec["ids_sha256"])
    # Independent schedule check: strptime + aware UTC conversion, no Step 2 parser.
    schedule_sets = {p: set() for p in pools}
    zone = ZoneInfo("Europe/Kyiv")
    for path in config["schedule_source_hashes"]:
        check("schedule_original_hash:" + Path(path).name, digest(path) == config["schedule_source_hashes"][path])
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                item, duration = int(r["tv_show_id"]), int(r["duration"])
                if item <= 0 or duration <= 0:
                    continue
                start = int(datetime.strptime(r["start_time"], "%d.%m.%Y %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc).timestamp())
                for period in schedule_sets:
                    lo, hi = config["windows_utc"][period]
                    if start < hi and start + duration > lo:
                        schedule_sets[period].add(str(item))
    for period in pools:
        check(f"independent_schedule_exact_candidate_set:{period}", set(pools[period]) == schedule_sets[period])

    # Independent stream of actual Step 2 files, without the production label loader.
    independent_truth = {u: set() for u in protocol.cohort}
    coverage = {}
    for period in ("train", "validation", "history"):
        path = STEP2 / "threshold_120s" / f"{period}_interactions.csv.gz"
        lo, hi = config["windows_utc"][period]
        users, programs, template_users = set(), set(), set()
        row_count = violations = 0
        first, last, previous_key = None, None, None
        with gzip.open(path, "rt", newline="") as stream:
            reader = csv.DictReader(stream)
            check(f"actual_interaction_schema:{period}", reader.fieldnames == val.INTERACTION_COLUMNS)
            for r in reader:
                u, item = r["user_id"], r["tv_show_id"]
                a, b = int(r["first_watch_utc"]), int(r["last_watch_utc"])
                key = (u, int(item))
                violations += not (lo <= a < b <= hi)
                violations += not (u.isascii() and u.isdecimal() and int(item) > 0 and str(int(item)) == item)
                violations += previous_key is not None and key <= previous_key
                if period in pools:
                    violations += item not in schedule_sets[period]
                previous_key = key
                row_count += 1
                users.add(u)
                programs.add(item)
                first = min(first, a) if first is not None else a
                last = max(last, b) if last is not None else b
                if u in independent_truth:
                    template_users.add(u)
                    if period == "validation":
                        independent_truth[u].add(item)
        coverage[period] = {"pairs": row_count, "users": len(users), "programs": len(programs),
                            "template_users": len(template_users), "template_empty": len(protocol.cohort) - len(template_users),
                            "first_watch_utc": first, "last_watch_utc": last, "violations": violations}
        check(f"full_file_temporal_identity_checks:{period}", violations == 0, coverage[period])
        if period != "history":
            check(f"independent_coverage_matches_config:{period}", all(config["coverage"][period][k] == v for k, v in coverage[period].items() if k != "violations"))
    check("truth_equals_all_distinct_validation_programs", independent_truth == protocol.truth)
    check("truth_total_pairs", sum(map(len, protocol.truth.values())) == 198517)
    check("labelled_and_empty_users", (sum(bool(v) for v in protocol.truth.values()), sum(not v for v in protocol.truth.values())) == (1246, 14))
    check("training_cold_coverage", (coverage["train"]["template_users"], coverage["train"]["template_empty"]) == (1250, 10))
    check("repeat_allowed_fixed_policy", config["policies"]["repeat_previously_watched"] is True)
    check("no_top5_truth_truncation", config["policies"]["ground_truth_top5_truncation"] is False and max(map(len, protocol.truth.values())) > 5)
    # Exercise the actual training-only accessor and observe every CSV opened by it.
    access = val.training_input(config)
    opened, access_count = [], 0
    actual_records = val.records

    def guarded_records(path, header):
        opened.append(str(Path(path).resolve()))
        if Path(path).resolve() != access.path:
            raise AssertionError("Training accessor attempted a nontraining CSV")
        yield from actual_records(path, header)

    with patch.object(val, "records", guarded_records):
        for row in access.rows():
            access_count += 1
    check("training_accessor_reads_training_only", opened == [str(access.path)] and access_count == 716796,
          {"csv_paths_opened": opened, "rows": access_count, "exposed_fields": [f.name for f in fields(access)]})
    for prohibited in ("validation", "history", "test"):
        path = STEP2 / "threshold_120s" / f"{prohibited}_interactions.csv.gz"
        reject(f"training_rejects_{prohibited}_source", lambda p=path: val.training_input(config, p))
    check("no_test_view_source_registered", all("test_interactions" not in path for path in config["input_products"]))

    # Synthetic API exercise only: cyclic ID positions, no learned statistics.
    candidate_list = list(protocol.candidates)
    predictions = {u: [candidate_list[(index * 7 + rank) % len(candidate_list)] for rank in range(5)]
                   for index, u in enumerate(protocol.cohort)}
    metric_started = time.monotonic()
    synthetic = protocol.evaluate(predictions)
    metric_seconds = time.monotonic() - metric_started
    reference = {u: reference_ap(protocol.truth[u], predictions[u]) for u in protocol.cohort}
    expected = sum(reference.values(), Fraction(0)) / len(protocol.cohort)
    labelled = [v for u, v in reference.items() if protocol.truth[u]]
    expected_diagnostic = sum(labelled, Fraction(0)) / len(labelled)
    check("synthetic_full_cohort_AP_reference", all(math.isclose(synthetic["per_user_ap_at_5"][u], float(reference[u]), rel_tol=0, abs_tol=1e-15) for u in protocol.cohort))
    check("synthetic_primary_MAP_reference", math.isclose(synthetic["primary_map_at_5"], float(expected), rel_tol=0, abs_tol=1e-15))
    check("synthetic_nonempty_MAP_reference", math.isclose(synthetic["diagnostic_map_at_5_nonempty"], float(expected_diagnostic), rel_tol=0, abs_tol=1e-15))
    check("synthetic_empty_users_AP_zero", all(synthetic["per_user_ap_at_5"][u] == 0 for u in protocol.cohort if not protocol.truth[u]))
    check("synthetic_deterministic_replay", synthetic == protocol.evaluate(dict(reversed(list(predictions.items())))))
    first_user = protocol.cohort[0]
    for name, altered in (("duplicate_program", [candidate_list[0]] * 5), ("short_list", candidate_list[:4]),
                          ("zero_program", ["0", *candidate_list[:4]]),
                          ("unknown_program", [str(int(candidate_list[-1]) + 1), *candidate_list[:4]])):
        bad = {**predictions, first_user: altered}
        reject(f"strict_production_rejects_{name}", lambda p=bad: protocol.evaluate(p))
    cold = next(r[0] for r in val.records(ROOT / config["cohort_file"]["path"],
                                        ["user_id", "has_training_history", "validation_relevant_count"]) if r[1] == "0")
    reject("strict_production_rejects_dropped_cold_user", lambda: protocol.evaluate({u: v for u, v in predictions.items() if u != cold}))
    empty_user = next(u for u in protocol.cohort if not protocol.truth[u])
    reject("strict_production_rejects_dropped_empty_label_user", lambda: protocol.evaluate({u: v for u, v in predictions.items() if u != empty_user}))
    save_json(ROOT / "results" / "step3_synthetic_checks.json", {
        "synthetic_only_not_model_performance": True, "construction": "cyclic candidate IDs using cohort position only; no fitted statistics",
        "evaluation_seconds": metric_seconds, "result": synthetic,
        "independent_primary_fraction": str(expected), "independent_diagnostic_fraction": str(expected_diagnostic)})
    unit_runs = [read_json(p) for p in sorted((ROOT / "results" / "step3_test_runs").glob("run_*.json"))]
    check("latest_unit_suite_passed", unit_runs[-1]["all_passed"])
    check("unit_suite_matches_current_sources", all(digest(ROOT / "code" / n) == h for n, h in unit_runs[-1]["source_sha256"].items()))
    check("all_Python_inside_code", all(p.is_relative_to(ROOT / "code") for p in ROOT.rglob("*.py")))
    check("no_Kaggle_prediction_csv", not list((ROOT / "submission").glob("*.csv")))
    result = {"stage": "Step 3 evaluation framework only", "all_passed": all(c["passed"] for c in checks),
              "passed_checks": sum(c["passed"] for c in checks), "failed_checks": sum(not c["passed"] for c in checks),
              "checks": checks, "unit_test_attempts": unit_runs,
              "recorded_failures_and_fixes": read_json(ROOT / "results" / "step3_failures.json"),
              "coverage": coverage, "truth_pairs": sum(map(len, protocol.truth.values())),
              "truth_unique_programs": len(set().union(*protocol.truth.values())),
              "truth_count_min": min(map(len, protocol.truth.values())), "truth_count_max": max(map(len, protocol.truth.values())),
              "candidate_counts": {p: len(v) for p, v in pools.items()},
              "config_sha256": protocol.config_sha256, "cohort_order_sha256": config["cohort_order_sha256"],
              "cohort_file_sha256": config["cohort_file"]["sha256"], "truth_sha256": config["truth"]["sha256"],
              "protocol_id": config["protocol_id"], "preserved_files": len(integrity), "integrity_after": integrity,
              "python": platform.python_version(), "dependencies": "standard library only; system IANA zone database",
              "verification_seconds": round(time.monotonic() - started, 3),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "synthetic_evaluation_seconds": metric_seconds,
              "production_models_fitted": 0, "model_performance_results": None,
              "implementation_sha256": {n: digest(ROOT / "code" / n) for n in ("evaluation.py", "validation.py", "test_step3.py", "verify_step3.py")}}
    save_json(ROOT / "results" / "step3_verification.json", result)
    print(json.dumps({k: result[k] for k in ("all_passed", "passed_checks", "failed_checks", "coverage", "truth_pairs", "truth_unique_programs",
                                             "truth_count_max", "candidate_counts", "verification_seconds", "peak_rss_mib")}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "inspect-ids", "verify"])
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot()
    elif args.action == "inspect-ids":
        inspect_schedule_ids()
    else:
        verify()
