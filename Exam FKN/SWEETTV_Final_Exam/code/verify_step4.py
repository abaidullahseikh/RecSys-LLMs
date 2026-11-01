"""Independent verification of Step 4 training aggregates, rankings and metrics."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import evaluation
import validation

ROOT = validation.ROOT
OUT = ROOT / "results" / "step4"
MODELS = {"popularity": ("distinct_users", "qualifying_airings", "watched_seconds"),
          "personal_frequency": ("qualifying_airings", "watched_seconds")}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def records(path, header):
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader) != header:
            raise ValueError(f"Bad header {path}")
        yield from reader


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main():
    started = time.monotonic()
    protocol = validation.EvaluationProtocol()
    cohort, candidates = protocol.cohort, protocol.candidates
    config = protocol.config
    run = load_json(OUT / "step4_run.json")
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    # Independently recompute counts from the hashed Weeks 11-20 interaction file.
    raw_path = ROOT / config["training_input"]["path"]
    distinct = defaultdict(set)
    airs, seconds = Counter(), Counter()
    personal_airings, personal_seconds = defaultdict(Counter), defaultdict(Counter)
    train_users = set()
    raw_rows = 0
    for r in records(raw_path, validation.INTERACTION_COLUMNS):
        u, item = r[0], r[1]
        n, sec = int(r[2]), int(r[3])
        train_users.add(u)
        distinct[item].add(u)
        airs[item] += n
        seconds[item] += sec
        personal_airings[u][item] += n
        personal_seconds[u][item] += sec
        raw_rows += 1
    eligible = frozenset(candidates)
    dusers = {p: len(us) for p, us in distinct.items()}
    eligible_count = {p: dusers.get(p, 0) for p in eligible}
    eligible_airs = {p: airs.get(p, 0) for p in eligible}
    eligible_seconds = {p: seconds.get(p, 0) for p in eligible}
    fallback = tuple(sorted(eligible, key=lambda p: (-eligible_count[p], -eligible_airs[p], int(p))))
    reference_rankings = {
        "distinct_users": tuple(sorted(eligible, key=lambda p: (-eligible_count[p], -eligible_airs[p], int(p)))),
        "qualifying_airings": tuple(sorted(eligible, key=lambda p: (-eligible_airs[p], -eligible_count[p], int(p)))),
        "watched_seconds": tuple(sorted(eligible, key=lambda p: (-eligible_seconds[p], -eligible_count[p], -eligible_airs[p], int(p))))}
    check("training_file_hash_matches_step3", digest(raw_path) == config["training_input"]["sha256"])
    check("training_only_rows_read", raw_rows == 716796 == run["training_rows"]["popularity"] == run["training_rows"]["personal_frequency"])
    check("training_split_only", tuple(config["training_input"]["window_utc"]) == validation.WINDOWS["train"])
    check("all_training_program_user_counts_match", len(distinct) == 2948 and sum(map(len, distinct.values())) == raw_rows)
    check("training_users_match_frozen_coverage", len(train_users & set(cohort)) == 1250)
    check("fallback_ranking_exact_independent_reference", tuple(run["fallback_ranking"]) == fallback)
    for variant in MODELS["popularity"]:
        check(f"popularity_all_counts:{variant}", run["popularity_statistics"][variant] == {
            "distinct_users": dusers,
            "qualifying_airings": dict(airs),
            "watched_seconds": dict(seconds)}[variant])
    check("step3_config_hash_unchanged", run["evaluation_config_sha256"] == protocol.config_sha256)
    check("protocol_id_unchanged", run["protocol_id"] == config["protocol_id"])
    check("cohort_is_exact_1260", run["cohort_users"] == len(cohort) == 1260)
    # Original Step 1 files and Step 3 frozen evaluation outputs must remain bit-identical.
    original_manifest = load_json(ROOT / "results" / "original_sha256_before.json")
    for original_path, expected_info in original_manifest.items():
        check("original_unchanged:" + Path(original_path).name, digest(original_path) == expected_info["sha256"])
    step3_snapshot = load_json(ROOT / "results" / "step3_inputs_before.json")["files"]
    step3_snapshot_changes = []
    for path, expected_info in step3_snapshot.items():
        actual = {"bytes": Path(path).stat().st_size, "sha256": digest(path)}
        if actual != expected_info:
            step3_snapshot_changes.append(path)
    check("all_step1_step2_inputs_outputs_evidence_preserved", not step3_snapshot_changes, step3_snapshot_changes)
    checkpoint_hashes = {}
    for line in (ROOT / "evidence" / "step2_checkpoint_sha256.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith(".json"):
            checkpoint_hashes[parts[1]] = parts[0]
    step3_checkpoint = (ROOT / "evidence" / "step3_checkpoint_sha256.txt").read_text().splitlines()[0].split()[0]
    checkpoint_hashes["step3_opencode_session.json"] = step3_checkpoint
    for filename, expected_hash in checkpoint_hashes.items():
        check("OpenCode_export_unchanged:" + filename, digest(ROOT / "evidence" / filename) == expected_hash)
    frozen_step3 = {
        ROOT / "data" / "step3" / "evaluation_config.json": "400b00eb6d660a663222e2e083ee59b2132469b2a06da28918b11323c88ec758",
        ROOT / "data" / "step3" / "evaluation_config.sha256": digest(ROOT / "data" / "step3" / "evaluation_config.sha256"),
        ROOT / "data" / "step3" / "validation_truth.csv": "166778fecc198e1acf73db2c3860580c3678275d5e5ccc180a17405b8181773d",
        ROOT / "data" / "step3" / "evaluation_cohort.csv": "5e8309d71cdbea025cd87c33899fba4b2c37cde666512a9cb336dfa1e6a191c9",
    }
    for path, expected_hash in frozen_step3.items():
        check("step3_input_preserved:" + path.name, digest(path) == expected_hash)

    predictions = {}
    ap_outputs = {}
    for family, names in MODELS.items():
        for variant in names:
            model_name = f"{family}_{variant}"
            metrics = run["variants"][model_name]
            pred_path = ROOT / metrics["prediction_file"]
            check(f"prediction_file_hash:{model_name}", digest(pred_path) == metrics["prediction_sha256"])
            raw = read_csv(pred_path)
            check(f"prediction_schema_and_users:{model_name}",
                  [r["user_id"] for r in raw] == list(cohort) and all(set(r) == {"user_id", "tv_show_id"} for r in raw))
            parsed = evaluation.parse_prediction_records(raw)
            check(f"strict_top5:{model_name}", set(parsed) == set(cohort) and all(len(row) == 5 for row in parsed.values()))
            check(f"candidate_eligible:{model_name}", all(item in eligible for row in parsed.values() for item in row))
            predictions[model_name] = parsed
            evaluated = protocol.evaluate(parsed)
            for key in ("primary_map_at_5", "diagnostic_map_at_5_nonempty", "primary_users", "nonempty_users", "empty_users"):
                check(f"exact_step3_metric:{model_name}:{key}", evaluated[key] == metrics[key], {"recomputed": evaluated[key], "saved": metrics[key]})
            check(f"per_user_ap_matches_csv:{model_name}", digest(ROOT / metrics["user_ap_file"]) == metrics["user_ap_sha256"])
            ap_rows = read_csv(ROOT / metrics["user_ap_file"])
            check(f"per_user_ap_population:{model_name}", [r["user_id"] for r in ap_rows] == list(cohort))
            check(f"per_user_ap_exact:{model_name}", all(float(r["ap_at_5"]) == evaluated["per_user_ap_at_5"][r["user_id"]] for r in ap_rows))
            ap_outputs[model_name] = evaluated["per_user_ap_at_5"]
            flat = [p for u in cohort for p in parsed[u]]
            unique_recs = set(flat)
            check(f"recommended_program_stats:{model_name}", len(unique_recs) == metrics["distinct_recommended_programs"]
                  and len(unique_recs) / len(eligible) == metrics["candidate_coverage_fraction"])
            hits = sum(bool(set(parsed[u]) & protocol.truth[u]) for u in cohort)
            check(f"users_with_hits:{model_name}", hits == metrics["users_with_at_least_one_relevant_recommendation"])
            warm_ap = [evaluated["per_user_ap_at_5"][u] for u in cohort if u in train_users]
            cold_ap = [evaluated["per_user_ap_at_5"][u] for u in cohort if u not in train_users]
            check(f"warm_cold_metrics:{model_name}",
                  math.fsum(warm_ap) / len(warm_ap) == metrics["warm_primary_map_at_5"]
                  and math.fsum(cold_ap) / len(cold_ap) == metrics["cold_primary_map_at_5"])

    # Reconstruct each popularity variant's ordered list from independent counters.
    for variant in MODELS["popularity"]:
        expected = reference_rankings[variant][:5]
        pred = predictions[f"popularity_{variant}"]
        check(f"popularity_rank_matches_independent_reference:{variant}", all(pred[u] == expected for u in cohort))
        check(f"popularity_is_same_global_top5:{variant}", len({pred[u] for u in cohort}) == 1)
    # Independent personal rankings: per-user count/duration, validation-candidate filter, exact global fallback.
    for variant in MODELS["personal_frequency"]:
        pred = predictions[f"personal_frequency_{variant}"]
        personal = personal_airings if variant == "qualifying_airings" else personal_seconds
        backfill_slots, backfill_users, full_fallback_users = 0, 0, 0
        for u in cohort:
            own = personal.get(u, {})
            eligible_own = set(own) & eligible
            own_order = sorted(eligible_own, key=lambda p: (-own[p], int(p)))
            expected = list(own_order[:5])
            seen = set(expected)
            if len(expected) < 5:
                backfill_users += 1
                for item in fallback:
                    if item not in seen:
                        expected.append(item)
                        seen.add(item)
                        if len(expected) == 5:
                            break
            backfill_slots += 5 - min(5, len(own_order))
            full_fallback_users += not bool(own)
            check(f"personal_rank_matches_independent_reference:{variant}:{u}", tuple(expected) == pred[u],
                  None if tuple(expected) == pred[u] else {"expected": expected, "actual": pred[u],
                                                           "own_order_first10": own_order[:10],
                                                           "fallback_first10": fallback[:10]})
        metrics = run["variants"][f"personal_frequency_{variant}"]
        check(f"personal_backfill_counts:{variant}", backfill_users == metrics["users_using_popularity_backfill_any_slot"]
              and backfill_slots == metrics["popularity_backfill_slots_exact"]
              and full_fallback_users == metrics["users_using_full_popularity_fallback"])

    global_ranking = predictions["popularity_distinct_users"][cohort[0]]
    similarity = {}
    for name, rows in predictions.items():
        overlaps = [len(set(rows[u]) & set(global_ranking)) for u in cohort]
        familiarity = sum(item in personal_airings.get(u, {}) for u in cohort for item in rows[u])
        similarity[name] = {"mean_top5_overlap_with_popularity_distinct_users": sum(overlaps) / len(overlaps),
                            "users_identical_to_popularity_distinct_users": sum(overlap == 5 for overlap in overlaps),
                            "recommendation_slots_matching_training_history": familiarity,
                            "fraction_slots_matching_training_history": familiarity / (5 * len(cohort))}

    comparisons = read_csv(OUT / "step4_baseline_comparison.csv")
    check("comparison_contains_all_five_variants", len(comparisons) == 5)
    published_comparison = ROOT / "results" / "step4_baseline_comparison.csv"
    check("requested_root_comparison_matches_run_copy",
          published_comparison.read_bytes() == (OUT / "step4_baseline_comparison.csv").read_bytes()
          and len(read_csv(published_comparison)) == 5)
    provisional = load_json(OUT / "provisional_two_users.json")
    check("two_user_selection_is_distinct_and_warm", provisional["heavy_user_id"] != provisional["light_user_id"]
          and provisional["heavy_training_distinct_programs"] > provisional["light_training_distinct_programs"] > 0)
    check("two_user_lists_cover_each_method", all(set(provisional["users"][role]["recommendations"]) == set(run["variants"])
          for role in ("heavy", "light")))
    check("no_kaggle_submission_created", not list((ROOT / "submission").glob("*.csv")))
    check("no_test_behavior_input", all("test_interactions" not in p and "test_views" not in p
                                        for p in config["input_products"]))
    source_hashes = {name: digest(ROOT / "code" / name) for name in
                     ("popularity.py", "personal_frequency.py", "run_step4.py", "verify_step4.py", "test_step4.py")}
    test_run = load_json(sorted((ROOT / "results" / "step4_test_runs").glob("step4_test_run_*.json"))[-1])
    check("latest_step4_tests_pass", test_run["all_passed"] and test_run["tests_run"] == 10)
    check("test_hashes_match_current_sources", all(digest(ROOT / "code" / n) == h for n, h in test_run["source_sha256"].items()))
    output_manifest = load_json(OUT / "step4_outputs.json")
    for name, metadata in output_manifest.items():
        check("output_manifest:" + name, (OUT / name).stat().st_size == metadata["bytes"] and digest(OUT / name) == metadata["sha256"])
    run_hash = digest(OUT / "step4_run.json")
    result = {"stage": "Step 4 baseline verification only", "all_passed": all(c["passed"] for c in checks),
              "passed_checks": sum(c["passed"] for c in checks), "failed_checks": sum(not c["passed"] for c in checks),
              "checks": checks, "independent_training_rows": raw_rows, "independent_training_users": len(train_users),
              "independent_training_programs": len(distinct), "fallback_ranking": fallback,
              "baseline_run_sha256": run_hash, "baseline_outputs_manifest_sha256": digest(OUT / "step4_outputs.json"),
              "unit_test_run": test_run, "implementation_sha256": source_hashes,
              "list_similarity_and_familiarity": similarity,
              "verification_runtime_seconds": round(time.monotonic() - started, 3),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "preserved_step3_config_sha256": protocol.config_sha256,
              "no_recommendation_submission": True, "no_step5_methods": True}
    attempt = 1 + len(list(OUT.glob("step4_verification_attempt_*.json")))
    write_json(OUT / f"step4_verification_attempt_{attempt:03d}.json", result)
    # Stable pointer to the latest attempt; prior attempt records remain preserved.
    write_json(OUT / "step4_verification.json", result)
    write_json(ROOT / "results" / "step4_verification.json", result)
    print(json.dumps({"all_passed": result["all_passed"], "passed_checks": result["passed_checks"],
                      "failed_checks": result["failed_checks"], "runtime_seconds": result["verification_runtime_seconds"],
                      "peak_rss_mib": result["peak_rss_mib"]}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
