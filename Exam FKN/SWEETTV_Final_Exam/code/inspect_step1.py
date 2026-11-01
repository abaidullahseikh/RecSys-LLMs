"""Read-only input audit. No recommender, model fitting, or predictions.

Uses streaming CSV, exact disk-backed duplicate checks, and compact intervals.
Outputs diagnostic JSON and a disposable SQLite inspection cache only.
"""

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import sqlite3
import sys
import time
import csv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "sweettv-tv-program-recommender"
EPOCH = datetime(1970, 1, 1)
FILES = ["dataset11-30.csv", "export_arh_11-20-final.csv",
         "export_arh_21-30-final.csv", "export_arh_31-42-final.csv", "submission.csv"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(value, schedule=False):
    if schedule:
        value = value[6:10] + "-" + value[3:5] + "-" + value[:2] + value[10:]
    return int((datetime.fromisoformat(value) - EPOCH).total_seconds())


def iso(value):
    return (EPOCH + timedelta(seconds=value)).isoformat(sep=" ") if value is not None else None


def distribution(counts):
    n = sum(counts.values())
    if not n:
        return {"n": 0}
    ordered = sorted(counts.items())
    quantiles = {}
    for q in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1):
        target = max(1, math.ceil(q * n))
        total = 0
        for value, count in ordered:
            total += count
            if total >= target:
                quantiles[str(q)] = value
                break
    return {"n": n, "quantiles_nearest_rank": quantiles,
            "mean": sum(k * v for k, v in counts.items()) / n}


def database():
    db = sqlite3.connect(ROOT / "results" / "step1_inspection.sqlite")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    return db


def profile():
    started = time.monotonic()
    paths = [SOURCE / name for name in FILES] + [ROOT.parent / "exam_quiz_FKN.md",
                                               ROOT.parent / "LLM4Rec. Midterm Quiz.pdf"]
    manifest = {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}
    (ROOT / "results" / "original_sha256_before.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    db = database()
    db.execute("DROP TABLE IF EXISTS views")
    db.execute("CREATE TABLE views (u TEXT, c INTEGER, s INTEGER, e INTEGER)")
    report = {"files": {}, "environment": {"python": sys.version, "platform": platform.platform(),
                                           "sqlite": sqlite3.sqlite_version}}
    for package in ("numpy", "scipy", "pandas", "polars", "duckdb", "networkx", "scikit-learn",
                    "implicit", "mlxtend", "pytest", "kaggle", "pypdf", "psutil"):
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][package] = None
    item_sets = {}
    user_sets = {}
    for name in FILES:
        file_started = time.monotonic()
        schedule = name.startswith("export")
        view = name.startswith("dataset")
        db.execute("DROP TABLE IF EXISTS duplicate_rows")
        db.execute("CREATE TABLE duplicate_rows (row_value TEXT)")
        batch, view_batch = [], []
        users, items, channels = set(), set(), set()
        minima, maxima = {}, {}
        durations = Counter()
        by_week = Counter()
        intervals = defaultdict(list)
        metadata = defaultdict(lambda: [set(), set()])
        counts = Counter()
        with (SOURCE / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            missing = Counter({key: 0 for key in header})
            whitespace = Counter({key: 0 for key in header})
            sentinels = {key: Counter() for key in header}
            integer = dict.fromkeys(header, True)
            examples = None
            for row in reader:
                counts["records"] += 1
                if len(row) != len(header):
                    counts["wrong_field_count"] += 1
                    continue
                if examples is None:
                    examples = dict(zip(header, row))
                batch.append((json.dumps(row, ensure_ascii=False, separators=(",", ":")),))
                for key, value in zip(header, row):
                    if not value.strip():
                        missing[key] += 1
                        if value:
                            whitespace[key] += 1
                    elif value.strip().lower() in {"null", "none", "nan", "na", "n/a"}:
                        sentinels[key][value] += 1
                    if value and not value.lstrip("-").isdigit():
                        integer[key] = False
                try:
                    if view:
                        u, c, s, e, d = row
                        users.add(u)
                        c, s, e, d = int(c), stamp(s), stamp(e), int(d)
                        view_batch.append((u, c, s, e))
                        counts["duration_mismatch"] += d != e - s
                        counts["nonpositive_interval"] += e <= s
                        counts["crosses_train_boundary"] += s < stamp("2020-05-18 00:00:00") < e
                        counts["crosses_test_boundary"] += s < stamp("2020-07-27 00:00:00") < e
                        channels.add(c)
                    elif schedule:
                        c, s, d, item = int(row[0]), stamp(row[2], True), int(row[3]), int(row[5])
                        e = s + d
                        channels.add(c)
                        items.add(item)
                        counts["zero_id"] += item == 0
                        counts["negative_id"] += item < 0
                        counts["nonpositive_interval"] += d <= 0
                        counts["schedule_seconds"] += d
                        counts["labelled_schedule_seconds"] += d if item > 0 else 0
                        intervals[c].append((s, e, item))
                        if item:
                            metadata[item][0].add(row[4])
                            metadata[item][1].add(row[6])
                    else:
                        users.add(row[0])
                        tokens = row[1].split()
                        counts["not_five_tokens"] += len(tokens) != 5
                        counts["all_zero_rows"] += tokens == ["0"] * 5
                        counts["nondigit_tokens"] += any(not t.isdigit() for t in tokens)
                        items.update(int(t) for t in tokens if t.isdigit())
                    if view or schedule:
                        durations[d] += 1
                        by_week["%d-W%02d" % (EPOCH + timedelta(seconds=s)).isocalendar()[:2]] += 1
                        for key, value in (("start", s), ("end", e)):
                            minima[key] = min(minima.get(key, value), value)
                            maxima[key] = max(maxima.get(key, value), value)
                except (ValueError, OverflowError):
                    counts["parse_errors"] += 1
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO duplicate_rows VALUES (?)", batch)
                    batch.clear()
                    if view_batch:
                        db.executemany("INSERT INTO views VALUES (?,?,?,?)", view_batch)
                        view_batch.clear()
            db.executemany("INSERT INTO duplicate_rows VALUES (?)", batch)
            if view_batch:
                db.executemany("INSERT INTO views VALUES (?,?,?,?)", view_batch)
            db.commit()
        duplicate_groups, duplicate_excess = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(n-1),0) FROM "
            "(SELECT COUNT(*) n FROM duplicate_rows GROUP BY row_value HAVING COUNT(*)>1)"
        ).fetchone()
        entry = {**manifest[str(SOURCE / name)], "path": str(SOURCE / name),
                 "columns": header, "missing_empty_or_whitespace": dict(missing),
                 "whitespace_only": dict(whitespace),
                 "text_null_markers": {k: dict(v) for k, v in sentinels.items() if v},
                 "inferred_types": {k: ("integer-form text" if integer[k] else "text") for k in header},
                 "counts": dict(counts), "duplicate_groups": duplicate_groups,
                 "duplicate_excess_rows": duplicate_excess,
                 "unique_users": len(users) if not schedule else None,
                 "unique_program_ids_including_zero": len(items) if not view else None,
                 "unique_positive_program_ids": len(items - {0}) if not view else None,
                 "unique_channels": len(channels) if view or schedule else None,
                 "channels": sorted(channels),
                 "time_ranges": {k: [iso(minima[k]), iso(maxima[k])] for k in minima},
                 "duration_seconds": distribution(durations), "records_by_iso_start_week": dict(sorted(by_week.items())),
                 "first_record": examples}
        if users:
            ints = [int(u) for u in users]
            entry["user_id_range"] = [str(min(ints)), str(max(ints))]
            entry["unique_users_above_2pow53"] = sum(i > 2**53 for i in ints)
            entry["unique_users_above_signed_int64"] = sum(i > 2**63 - 1 for i in ints)
            entry["user_ids_with_leading_zero"] = sum(u.startswith("0") and len(u) > 1 for u in users)
            user_sets[name] = users
        if schedule:
            conflicts, overlaps, overlap_seconds = 0, 0, 0
            labels_overlaps = 0
            for records in intervals.values():
                records.sort()
                max_end, labelled_end = -1, -1
                starts = Counter()
                for s, e, item in records:
                    starts[s] += 1
                    if s < max_end:
                        overlaps += 1
                        overlap_seconds += max(0, min(e, max_end) - s)
                    max_end = max(max_end, e)
                    if item:
                        labels_overlaps += s < labelled_end
                        labelled_end = max(labelled_end, e)
                conflicts += sum(n - 1 for n in starts.values() if n > 1)
            entry["schedule_overlap"] = {"rows_overlapping_prior_coverage": overlaps,
                                         "excess_seconds_over_union": overlap_seconds,
                                         "labelled_rows_overlapping_prior_labelled": labels_overlaps,
                                         "same_channel_start_excess": conflicts}
            entry["ids_with_multiple_titles"] = sum(len(m[0]) > 1 for m in metadata.values())
            entry["ids_with_multiple_categories"] = sum(len(m[1]) > 1 for m in metadata.values())
            entry["positive_id_categories"] = sorted(set().union(*(m[1] for m in metadata.values())))
            item_sets[name] = items - {0}
        entry["elapsed_seconds"] = round(time.monotonic() - file_started, 3)
        report["files"][name] = entry
        print(name, counts["records"], "rows;", duplicate_excess, "duplicates;", entry["elapsed_seconds"], "seconds", flush=True)
        (ROOT / "results" / "step1_profile.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    train, val, test = [item_sets[n] for n in FILES[1:4]]
    report["cross_file"] = {"validation_ids_in_train": len(val & train),
                            "validation_ids_absent_train_schedule": len(val - train),
                            "test_ids_in_history_schedule": len(test & (train | val)),
                            "test_ids_absent_history_schedule": len(test - (train | val)),
                            "historical_positive_ids": len(train | val),
                            "all_positive_ids": len(train | val | test),
                            "template_users_absent_view_log": len(user_sets[FILES[-1]] - user_sets[FILES[0]])}
    db.execute("DROP TABLE duplicate_rows")
    db.execute("CREATE INDEX IF NOT EXISTS view_order ON views(c,u,s,e)")
    db.commit()
    db.close()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (ROOT / "results" / "step1_profile.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PROFILE COMPLETE", report["elapsed_seconds"], "seconds; peak MiB", report["peak_rss_mib"], flush=True)


def integrity():
    before = json.loads((ROOT / "results" / "original_sha256_before.json").read_text())
    after = {p: {"bytes": Path(p).stat().st_size, "sha256": digest(Path(p))} for p in before}
    reference_path = ROOT / "results" / "homework_guidelines_sha256.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    reference_matches = (digest(Path(reference["path"])) == reference["sha256"]) if reference else None
    result = {"files": after, "all_match_before": before == after,
              "additional_homework_guidelines_hash_matches": reference_matches}
    (ROOT / "results" / "original_sha256_after.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if before != after or reference_matches is False:
        raise RuntimeError("Original file integrity changed")


def reference():
    path = ROOT.parent.parent / "Week 2" / "homework_guidelines (2).pdf"
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    (ROOT / "results" / "homework_guidelines_sha256.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def verify():
    profile = json.loads((ROOT / "results" / "step1_profile.json").read_text())
    intervals = json.loads((ROOT / "results" / "step1_intervals_kyiv.json").read_text())
    clock = json.loads((ROOT / "results" / "step1_clock.json").read_text())
    integrity = json.loads((ROOT / "results" / "original_sha256_after.json").read_text())
    checks = {}
    checks["exam_copy_byte_identical"] = (ROOT / "exam_quiz_FKN.md").read_bytes() == (ROOT.parent / "exam_quiz_FKN.md").read_bytes()
    checks["original_seven_files_unchanged"] = integrity["all_match_before"]
    checks["homework_guidelines_unchanged_since_hash"] = integrity["additional_homework_guidelines_hash_matches"]
    checks["all_csv_rows_have_expected_width_and_parse"] = all(
        not f["counts"].get("parse_errors", 0) and not f["counts"].get("wrong_field_count", 0)
        for f in profile["files"].values())
    checks["four_data_files_total_matches_exam"] = sum(profile["files"][n]["counts"]["records"] for n in FILES[:4]) == 9265994
    checks["interval_pass_covers_every_view_row"] = intervals["counters"]["raw_view_rows"] == profile["files"][FILES[0]]["counts"]["records"]
    c = intervals["counters"]
    checks["view_match_partition_sums_to_total"] = (c["raw_view_rows_any_labelled_overlap"] + c["raw_view_rows_schedule_but_no_label"] + c["raw_view_rows_no_schedule_overlap"]) == c["raw_view_rows"]
    checks["duration_fields_agree_under_kyiv_hypothesis"] = not intervals["duration_difference_supplied_minus_timestamp"]
    checks["all_naive_mismatches_cross_spring_transition"] = clock["counts"]["views_crossing_04_transition"] == profile["files"][FILES[0]]["counts"]["duration_mismatch"] == 117
    checks["clip_boundary_counts_reconcile"] = sum(p["counts"]["raw_view_rows_touching_period"] for p in intervals["periods"].values()) == c["raw_view_rows"] + 1007
    checks["template_1260_unique_all_zero_five_tokens"] = profile["files"][FILES[-1]]["unique_users"] == profile["files"][FILES[-1]]["counts"]["all_zero_rows"] == 1260
    checks["python_files_only_in_code"] = all(p.is_relative_to(ROOT / "code") for p in ROOT.rglob("*.py"))
    checks["no_generated_submission_csv"] = not list((ROOT / "submission").glob("*.csv"))
    checks["all_fifteen_audit_sections_present"] = all(f"## {i}. " in (ROOT / "results" / "step1_audit.md").read_text() for i in range(1, 16))
    result = {"checks": checks, "all_passed": all(checks.values()),
              "scope": "Inspection consistency and input integrity only; no model/evaluator verification.",
              "inspection_cache_bytes": (ROOT / "results" / "step1_inspection.sqlite").stat().st_size}
    session = ROOT / "evidence" / "step1_opencode_session.json"
    if session.exists():
        native = json.loads(session.read_text())
        result["native_session_checkpoint"] = {"bytes": session.stat().st_size, "sha256": digest(session),
                                               "top_level_keys": list(native),
                                               "session_id": native.get("info", {}).get("id"),
                                               "messages": len(native.get("messages", []))}
        assert result["native_session_checkpoint"]["session_id"] == "ses_e7371ba5affeMuqWtLibI0NrzK"
    (ROOT / "results" / "step1_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise RuntimeError("Inspection consistency check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["profile", "integrity", "reference", "verify"])
    args = parser.parse_args()
    globals()[args.action]()
