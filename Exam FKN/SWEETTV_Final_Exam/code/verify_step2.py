"""Read-only independent full-output checks, Step 1 reconciliation, and evidence.

SQL aggregation independently checks the production per-user aggregation. History
is checked by a separate streaming merge of the two unthresholded partition files.
No recommendation metric is computed. Original session exports are never written.
"""

from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import resource
import sqlite3
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
JOIN = ROOT / "data" / "step2"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def raw_records(period):
    for r in csv_records(JOIN / f"{period}_airing_overlaps.csv.gz"):
        # Match file order: channel, exact user string, broadcast time, program.
        key = (int(r["channel_id"]), r["user_id"], int(r["broadcast_start_utc"]),
               int(r["broadcast_end_utc"]), int(r["tv_show_id"]))
        value = (int(r["overlap_seconds"]), int(r["first_overlap_utc"]), int(r["last_overlap_utc"]))
        yield key, value


def main():
    begun = time.monotonic()
    join = read_json(JOIN / "join_manifest.json")
    selected = {t: read_json(JOIN / f"threshold_{t}s" / "interaction_manifest.json") for t in (120, 300)}
    previous = read_json(ROOT / "results" / "step1_intervals_kyiv.json")
    original = read_json(ROOT / "results" / "original_sha256_before.json")
    checks, comparisons = [], []

    def check(name, condition, detail=None):
        entry = {"name": name, "passed": bool(condition)}
        if detail is not None:
            entry["detail"] = detail
        checks.append(entry)

    def compare(name, measured, expected):
        comparisons.append({"name": name, "measured": measured, "step1": expected, "difference": measured - expected})
        check("step1:" + name, measured == expected)

    hashes = {}
    for filename, info in join["source_files_before"].items():
        current = digest(info["path"])
        hashes[filename] = current
        check("original_unchanged:" + filename, current == info["sha256"] == original[info["path"]]["sha256"])
    step1_evidence = read_json(ROOT / "results" / "step1_verification.json")["native_session_checkpoint"]
    session_hash = digest(ROOT / "evidence" / "step1_opencode_session.json")
    check("original_OpenCode_export_unchanged", session_hash == step1_evidence["sha256"])
    current_code = digest(ROOT / "code" / "data_pipeline.py")
    check("join_matches_current_pipeline_source", current_code == join["pipeline_sha256"])
    for base, manifest in [(JOIN, join), *[(JOIN / f"threshold_{t}s", selected[t]) for t in (120, 300)]]:
        for name, info in manifest["artifacts"].items():
            check("artifact_sha256:" + str((base / name).relative_to(ROOT)), digest(base / name) == info["sha256"])
    for t in selected:
        check(f"threshold_{t}_uses_same_join", selected[t]["join_manifest_sha256"] == digest(JOIN / "join_manifest.json"))
        check(f"threshold_{t}_matches_current_source", selected[t]["pipeline_sha256"] == current_code)
    template = [r["user_id"] for r in csv_records(join["source_files_before"]["submission.csv"]["path"])]
    check("exact_template_identity_order", template == [r["user_id"] for r in csv_records(JOIN / "template_users.csv")])
    check("all_1260_template_users_retained", len(template) == len(set(template)) == 1260)
    original_users = {r["user_id"] for r in csv_records(join["source_files_before"]["dataset11-30.csv"]["path"])}
    check("raw_user_population", len(original_users) == 4914)
    full_scan = {}
    with tempfile.TemporaryDirectory(prefix="step2_verify_", dir=ROOT / "data") as temporary:
        db = sqlite3.connect(Path(temporary) / "reference.sqlite")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA temp_store=FILE")
        for period in ("train", "validation", "history"):
            db.execute("DROP TABLE IF EXISTS raw")
            db.execute("CREATE TABLE raw(u TEXT,item INTEGER,seconds INTEGER,first INTEGER,last INTEGER)")
            lo, hi = join["windows_utc"][period]
            catalog = {int(r["tv_show_id"]) for r in csv_records(JOIN / f"{period}_candidates.csv")}
            batch, last_key, failures = [], None, Counter()
            n = seconds_sum = 0
            boundary = Counter()
            for key, value in raw_records(period):
                channel, user, start, end, item = key
                seconds, first, last = value
                n += 1
                seconds_sum += seconds
                failures["nonunique_or_unsorted_airing_key"] += last_key is not None and key <= last_key
                failures["unknown_or_modified_user_id"] += user not in original_users
                failures["invalid_or_ineligible_program"] += item <= 0 or item not in catalog
                failures["duration_or_boundary_violation"] += not (0 < seconds <= last - first <= end - start and max(lo, start) <= first < last <= min(hi, end))
                last_key = key
                batch.append((user, item, seconds, first, last))
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
                    batch.clear()
            db.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
            db.commit()
            full_scan[period] = {"rows_scanned": n, "seconds": seconds_sum, "violations": dict(failures)}
            check(f"all_overlap_row_invariants:{period}", not any(failures.values()), dict(failures))
            check(f"overlap_row_count:{period}", n == join["stats"]["periods"][period]["positive_user_airings"])
            check(f"overlap_seconds:{period}", seconds_sum == join["stats"]["periods"][period]["labelled_overlap_seconds"])
            # Different implementation: SQL GROUP BY rather than production's Python grouping.
            db.execute("CREATE INDEX raw_user_item ON raw(u,item)")
            for threshold in (120, 300):
                expected = db.execute("SELECT u,item,COUNT(*),SUM(seconds),MIN(first),MAX(last) FROM raw WHERE seconds>=? GROUP BY u,item ORDER BY u,item", (threshold,))
                actual = csv_records(JOIN / f"threshold_{threshold}s" / f"{period}_interactions.csv.gz")
                bad, row_count, airing_count, credited = 0, 0, 0, 0
                users, programs = set(), set()
                for sql, csv_row in itertools.zip_longest(expected, actual):
                    if sql is None or csv_row is None:
                        bad += 1
                        continue
                    row = (csv_row["user_id"], *[int(csv_row[k]) for k in ("tv_show_id", "qualifying_airings", "watched_seconds", "first_watch_utc", "last_watch_utc")])
                    bad += tuple(sql) != row
                    row_count += 1
                    airing_count += row[2]
                    credited += row[3]
                    users.add(row[0])
                    programs.add(row[1])
                measures = {"unique_user_program_pairs": row_count, "qualifying_user_airings": airing_count,
                            "qualifying_seconds": credited, "unique_users": len(users), "unique_programs": len(programs),
                            "template_users_with_interactions": len(users & set(template))}
                check(f"independent_SQL_vs_all_output_rows:{period}:{threshold}", bad == 0, {"rows": row_count, "mismatches": bad})
                check(f"interaction_manifest_counts:{period}:{threshold}", all(selected[threshold]["periods"][period][k] == v for k, v in measures.items()), measures)
            print("Verified all rows", period, n, flush=True)
        db.close()

    # Independently merge raw partition streams and compare every history-airing row.
    combined = heapq.merge(raw_records("train"), raw_records("validation"), key=lambda r: r[0])
    history = iter(raw_records("history"))
    history_mismatches, cross_boundary = 0, Counter()
    for key, group in itertools.groupby(combined, lambda r: r[0]):
        values = [r[1] for r in group]
        expected = (sum(v[0] for v in values), min(v[1] for v in values), max(v[2] for v in values))
        observed = next(history, None)
        history_mismatches += observed != (key, expected)
        if len(values) == 2:
            cross_boundary["user_airings_in_both_partitions"] += 1
            for t in (120, 300):
                qualifiers = sum(v[0] >= t for v in values)
                cross_boundary[f"both_sides_qualify_{t}s"] += qualifiers == 2
                cross_boundary[f"only_combined_qualifies_{t}s"] += qualifiers == 0 and expected[0] >= t
    history_mismatches += next(history, None) is not None
    check("full_history_equals_unthresholded_partition_merge", history_mismatches == 0,
          {"mismatches": history_mismatches, **cross_boundary})

    for period in ("train", "validation"):
        old = previous["periods"][period]
        new = join["stats"]["periods"][period]
        compare(f"{period}:positive_user_airings", new["positive_user_airings"], old["union_user_airing_overlap_seconds"]["n"])
        compare(f"{period}:raw_view_rows", new["raw_records_touching_period"], old["counts"]["raw_view_rows_touching_period"])
        compare(f"{period}:view_users", new["view_users"], old["users_with_views"])
        for threshold in (120, 300):
            stats = selected[threshold]["periods"][period]
            compare(f"{period}:qualifying_airings_{threshold}", stats["qualifying_user_airings"], old["counts"][f"user_airings_at_least_{threshold}s"])
            compare(f"{period}:users_{threshold}", stats["unique_users"], old["users_at_threshold"][str(threshold)])
            compare(f"{period}:template_users_{threshold}", stats["template_users_with_interactions"], old["template_users_at_threshold"][str(threshold)])
        compare(f"{period}:user_program_pairs_120", selected[120]["periods"][period]["unique_user_program_pairs"], old["unique_user_program_pairs_120s"])
        compare(f"{period}:watched_programs_120", selected[120]["periods"][period]["unique_programs"], old["unique_watched_programs_120s"])
    for period in ("train", "validation", "test"):
        compare(f"catalog:{period}", join["stats"]["candidate_counts"][period], previous["candidate_catalogs_by_actual_overlap"][period])
    map_names = {"raw_rows_labelled_overlap": "raw_view_rows_any_labelled_overlap",
                 "raw_rows_schedule_without_label": "raw_view_rows_schedule_but_no_label",
                 "raw_rows_without_schedule_overlap": "raw_view_rows_no_schedule_overlap",
                 "union_history_seconds": "union_user_channel_seconds",
                 "union_labelled_seconds": "union_user_channel_labelled_seconds",
                 "union_schedule_seconds": "union_user_channel_schedule_seconds",
                 "merged_history_segments": "merged_user_channel_segments"}
    for new, old in map_names.items():
        compare(new, join["stats"]["views"][new], previous["counters"][old])
    compare("exact_duplicate_events", join["stats"]["views"]["exact_duplicate_records_removed"], 17)
    compare("history_template_cold_users", selected[120]["periods"]["history"]["template_users_without_interactions"], previous["validation_diagnostics_120s"]["full_history_template_users_without_120s_airing"])

    # Investigate the 8,234 file-union versus 8,233 timestamp-clipped catalog.
    candidates = {int(r["tv_show_id"]) for r in csv_records(JOIN / "history_candidates.csv")}
    outside = []
    historical_ids = set()
    zone = ZoneInfo("Europe/Kyiv")
    for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        for r in csv_records(join["source_files_before"][name]["path"]):
            item = int(r["tv_show_id"])
            if item > 0:
                historical_ids.add(item)
                if item not in candidates:
                    start = int(datetime.strptime(r["start_time"], "%d.%m.%Y %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc).timestamp())
                    end = start + int(r["duration"])
                    lo, hi = join["windows_utc"]["history"]
                    outside.append({"file": name, "id": item, "title": r["tv_show_title"], "start_local": r["start_time"],
                                    "duration": int(r["duration"]), "outside_history": end <= lo or start >= hi})
    check("history_catalog_difference_explained_by_time_filter", len(historical_ids) == 8234 and len(candidates) == 8233 and all(r["outside_history"] for r in outside), outside)
    unit_runs = [read_json(p) for p in sorted((ROOT / "results" / "step2_test_runs").glob("run_*.json"))]
    check("latest_independent_tests_pass", unit_runs[-1]["all_passed"] and unit_runs[-1]["pipeline_sha256"] == current_code)
    check("latest_test_source_matches", unit_runs[-1]["test_source_sha256"] == digest(ROOT / "code" / "test_data_pipeline.py"))
    check("all_python_inside_code", all(p.is_relative_to(ROOT / "code") for p in ROOT.rglob("*.py")))
    check("no_prediction_submission_csv", not list((ROOT / "submission").glob("*.csv")))
    for threshold in (120, 300):
        rows = list(csv_records(JOIN / f"threshold_{threshold}s" / "template_user_coverage.csv"))
        check(f"complete_ordered_template_coverage:{threshold}", [r["user_id"] for r in rows] == template)
        for period in ("train", "validation", "history"):
            count = sum(int(r[f"{period}_programs"]) > 0 for r in rows)
            check(f"template_coverage_counts:{period}:{threshold}", count == selected[threshold]["periods"][period]["template_users_with_interactions"])
    result = {"stage": "Step 2 preprocessing only", "all_current_checks_passed": all(c["passed"] for c in checks),
              "checks": checks, "passed_checks": sum(c["passed"] for c in checks), "failed_checks": sum(not c["passed"] for c in checks),
              "unit_test_attempts": unit_runs, "known_failure_and_fix": "Initial test rejected 24:00:00 but Python 3.14 fromisoformat accepted it; explicit clock-field bounds added and suite rerun.",
              "step1_comparisons": comparisons, "full_output_scans": full_scan,
              "cross_boundary_recombination": dict(cross_boundary), "catalog_difference_evidence": outside,
              "join_stats": join["stats"], "interaction_stats": {t: selected[t]["periods"] for t in selected},
              "original_sha256": hashes, "original_step1_session_sha256": session_hash,
              "verification_seconds": round(time.monotonic() - begun, 3),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "scope_note": "Fixtures and SQL checks are agent-executed verification, not claimed student personal verification; no hidden Kaggle labels verified."}
    path = ROOT / "results" / "step2_verification.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VERIFICATION", result["passed_checks"], "passed;", result["failed_checks"], "failed; seconds", result["verification_seconds"], flush=True)
    print("Boundary reconciliation", dict(cross_boundary))
    print("Catalog difference", outside)
    if not result["all_current_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
