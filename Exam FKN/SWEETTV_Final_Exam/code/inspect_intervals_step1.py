"""Descriptive interval diagnostics only; exports counts, never model inputs or predictions."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import argparse
import csv
from datetime import timedelta
import itertools
import json
import resource
import time
from zoneinfo import ZoneInfo

from inspect_step1 import ROOT, SOURCE, FILES, EPOCH, database, distribution, stamp as wall_stamp, iso

CLOCK = "naive"
ZONE = ZoneInfo("Europe/Kyiv")


def utc_seconds(wall):
    return int((EPOCH + timedelta(seconds=wall)).replace(tzinfo=ZONE).timestamp())


def stamp(value, schedule=False):
    wall = wall_stamp(value, schedule)
    return utc_seconds(wall) if CLOCK == "kyiv" else wall


def merge(records):
    result = []
    for s, e in sorted(records):
        if result and s <= result[-1][1]:
            result[-1] = (result[-1][0], max(e, result[-1][1]))
        else:
            result.append((s, e))
    return result


def coverage_index(records):
    records = merge(records)
    starts, ends, prefix = [], [], [0]
    for s, e in records:
        starts.append(s)
        ends.append(e)
        prefix.append(prefix[-1] + e - s)
    return starts, ends, prefix


def covered(index, s, e):
    starts, ends, prefix = index
    left, right = bisect_right(ends, s), bisect_left(starts, e)
    if left >= right:
        return 0
    return prefix[right] - prefix[left] - max(0, s - starts[left]) - max(0, ends[right - 1] - e)


def main():
    begun = time.monotonic()
    boundaries = [stamp(s) for s in ("2020-03-09 00:00:00", "2020-05-18 00:00:00",
                                     "2020-07-27 00:00:00", "2020-10-19 00:00:00")]
    names = ("train", "validation", "test")
    schedules = defaultdict(list)
    candidate_ids = [set(), set(), set()]
    file_extras = {}
    boundary_raw = set()
    cross_file_exact_duplicates = 0
    for file_number, name in enumerate(FILES[1:4]):
        extra = Counter()
        year_examples = set()
        categories = Counter()
        bad_examples = []
        with (SOURCE / name).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                s, d = stamp(row["start_time"], True), int(row["duration"])
                e, c, item = s + d, int(row["channel_id"]), int(row["tv_show_id"])
                categories[row["tv_show_category"]] += 1
                if row["year_of_production"] and not row["year_of_production"].isdigit():
                    extra["nonnumeric_year_rows"] += 1
                    if len(year_examples) < 10:
                        year_examples.add(row["year_of_production"])
                if d <= 0:
                    bad_examples.append(row)
                    continue
                extra["start_before_named_window"] += s < boundaries[file_number]
                extra["start_at_or_after_named_window_end"] += s >= boundaries[file_number + 1]
                extra["crosses_named_window_end"] += s < boundaries[file_number + 1] < e
                # Only boundary bands can overlap the immediately following file.
                if s < boundaries[file_number] + 86400:
                    cross_file_exact_duplicates += tuple(row.values()) in boundary_raw
                if file_number < 2 and e >= boundaries[file_number + 1]:
                    boundary_raw.add(tuple(row.values()))
                for period in range(3):
                    if item > 0 and s < boundaries[period + 1] and e > boundaries[period]:
                        candidate_ids[period].add(item)
                if file_number < 2:
                    schedules[c].append((s, e, item))
        file_extras[name] = {**extra, "nonnumeric_year_examples": sorted(year_examples),
                             "all_categories": dict(categories), "invalid_interval_examples": bad_examples}
    with (SOURCE / "submission.csv").open(newline="") as stream:
        target_users = {r["user_id"] for r in csv.DictReader(stream)}
    counters = Counter()
    raw_overlap = [Counter(), Counter()]
    airing_overlap = [Counter(), Counter()]
    airing_fraction = [Counter(), Counter()]
    period_counts = [Counter(), Counter()]
    users_with_views = [set(), set()]
    users_with_any_label = [set(), set()]
    threshold_users = [{t: set() for t in (60, 120, 300, 600)} for _ in range(2)]
    histories_120 = [defaultdict(set), defaultdict(set)]
    counts_120 = [Counter(), Counter()]
    channel_ids = set()
    conflict_examples = []
    db = database()
    for c, grouped in itertools.groupby(db.execute("SELECT c,u,s,e FROM views ORDER BY c,u,s,e"), lambda r: r[0]):
        channel_ids.add(c)
        original = schedules.pop(c, [])
        records = sorted(set(original))
        counters["historical_duplicate_airing_keys_on_viewed_channels"] += len(original) - len(records)
        all_index = coverage_index([(s, e) for s, e, _ in records])
        labelled = [(s, e, item) for s, e, item in records if item > 0]
        label_index = coverage_index([(s, e) for s, e, _ in labelled])
        label_starts = [r[0] for r in labelled]
        max_ends, end = [], -1
        for s, e, item in labelled:
            if s < end:
                counters["historical_conflicting_labelled_airings_on_viewed_channels"] += 1
                if len(conflict_examples) < 10:
                    conflict_examples.append({"channel": c, "start": iso(s), "end": iso(e), "id": item})
            end = max(end, e)
            max_ends.append(end)

        def overlaps(s, e):
            for i in range(bisect_right(max_ends, s), bisect_left(label_starts, e)):
                a, b, item = labelled[i]
                seconds = min(e, b) - max(s, a)
                if seconds > 0:
                    yield i, seconds

        for u, user_group in itertools.groupby(grouped, lambda r: r[1]):
            spans = []
            for _, _, s, e in user_group:
                if CLOCK == "kyiv":
                    s, e = utc_seconds(s), utc_seconds(e)
                spans.append((s, e))
                counters["raw_view_rows"] += 1
                counters["raw_timestamp_view_seconds"] += e - s
                cov, lab = covered(all_index, s, e), covered(label_index, s, e)
                counters["raw_view_rows_no_schedule_channel"] += not records
                counters["raw_view_rows_any_schedule_overlap"] += cov > 0
                counters["raw_view_rows_any_labelled_overlap"] += lab > 0
                counters["raw_view_rows_schedule_but_no_label"] += cov > 0 and lab == 0
                counters["raw_view_rows_no_schedule_overlap"] += cov == 0
                for p in range(2):
                    a, b = max(s, boundaries[p]), min(e, boundaries[p + 1])
                    if a < b:
                        users_with_views[p].add(u)
                        period_counts[p]["raw_view_rows_touching_period"] += 1
                        for _, seconds in overlaps(a, b):
                            raw_overlap[p][seconds] += 1
            merged = merge(spans)
            counters["merged_user_channel_segments"] += len(merged)
            counters["union_user_channel_seconds"] += sum(e - s for s, e in merged)
            totals = [Counter(), Counter()]
            for s, e in merged:
                counters["union_user_channel_schedule_seconds"] += covered(all_index, s, e)
                counters["union_user_channel_labelled_seconds"] += covered(label_index, s, e)
                for p in range(2):
                    a, b = max(s, boundaries[p]), min(e, boundaries[p + 1])
                    if a < b:
                        period_counts[p]["merged_segments_touching_period"] += 1
                        period_counts[p]["union_view_seconds"] += b - a
                        for i, seconds in overlaps(a, b):
                            totals[p][i] += seconds
            for p in range(2):
                for i, seconds in totals[p].items():
                    s, e, item = labelled[i]
                    assert 0 < seconds <= e - s
                    users_with_any_label[p].add(u)
                    airing_overlap[p][seconds] += 1
                    airing_fraction[p][round(10000 * seconds / (e - s))] += 1
                    for t in (60, 120, 300, 600):
                        if seconds >= t:
                            period_counts[p][f"user_airings_at_least_{t}s"] += 1
                            threshold_users[p][t].add(u)
                    if seconds >= 120:
                        histories_120[p][u].add(item)
                        counts_120[p][u] += 1
    # Cross-channel concurrency diagnostic: union all channel intervals per user.
    global_union = 0
    for u, grouped in itertools.groupby(db.execute("SELECT u,s,e FROM views ORDER BY u,s,e"), lambda r: r[0]):
        last_s, last_e = None, None
        for _, s, e in grouped:
            if CLOCK == "kyiv":
                s, e = utc_seconds(s), utc_seconds(e)
            if last_s is None:
                last_s, last_e = s, e
            elif s <= last_e:
                last_e = max(last_e, e)
            else:
                global_union += last_e - last_s
                last_s, last_e = s, e
        global_union += last_e - last_s
    counters["union_user_all_channel_seconds"] = global_union
    counters["cross_channel_concurrent_excess_seconds"] = counters["union_user_channel_seconds"] - global_union
    counters["same_channel_duplicate_or_overlap_seconds"] = counters["raw_timestamp_view_seconds"] - counters["union_user_channel_seconds"]
    # Inspect the discrepancy rather than trusting the misspelled duration field.
    mismatches, mismatch_examples = Counter(), []
    with (SOURCE / FILES[0]).open(newline="") as stream:
        for row in csv.DictReader(stream):
            delta = int(row["duraton"]) - (stamp(row["stop_time"]) - stamp(row["start_time"]))
            if delta:
                mismatches[delta] += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(row)
    output = {"clock_interpretation": CLOCK, "counters": dict(counters), "file_extras": file_extras,
              "cross_file_exact_duplicate_boundary_rows": cross_file_exact_duplicates,
              "conflicting_labelled_airing_examples": conflict_examples,
              "duration_difference_supplied_minus_timestamp": dict(mismatches),
              "duration_mismatch_examples": mismatch_examples,
              "periods": {}, "candidate_catalogs_by_actual_overlap": {}}
    for p in range(3):
        output["candidate_catalogs_by_actual_overlap"][names[p]] = len(candidate_ids[p])
    output["candidate_catalogs_by_actual_overlap"]["validation_unseen_in_train_schedule"] = len(candidate_ids[1] - candidate_ids[0])
    output["candidate_catalogs_by_actual_overlap"]["test_unseen_in_history_schedule"] = len(candidate_ids[2] - candidate_ids[0] - candidate_ids[1])
    for p in range(2):
        output["periods"][names[p]] = {
            "counts": dict(period_counts[p]),
            "raw_row_to_airing_overlap_seconds": distribution(raw_overlap[p]),
            "union_user_airing_overlap_seconds": distribution(airing_overlap[p]),
            "union_user_airing_fraction_basis_points": distribution(airing_fraction[p]),
            "users_with_views": len(users_with_views[p]),
            "users_with_any_label": len(users_with_any_label[p]),
            "users_at_threshold": {t: len(s) for t, s in threshold_users[p].items()},
            "template_users_with_views": len(target_users & users_with_views[p]),
            "template_users_with_any_label": len(target_users & users_with_any_label[p]),
            "template_users_at_threshold": {t: len(s & target_users) for t, s in threshold_users[p].items()},
            "unique_user_program_pairs_120s": sum(map(len, histories_120[p].values())),
            "unique_watched_programs_120s": len(set().union(*histories_120[p].values())),
            "unique_programs_per_usable_user_120s": distribution(Counter(map(len, histories_120[p].values()))),
            "qualifying_airings_per_usable_user_120s": distribution(Counter(counts_120[p].values()))}
    relevant_targets = {u for u in target_users if histories_120[1].get(u)}
    train_targets = {u for u in target_users if histories_120[0].get(u)}
    val_pairs = sum(len(histories_120[1].get(u, set())) for u in target_users)
    repeated_pairs = sum(len(histories_120[1].get(u, set()) & histories_120[0].get(u, set())) for u in target_users)
    output["validation_diagnostics_120s"] = {
        "template_users_with_nonempty_relevance": len(relevant_targets),
        "template_users_empty_relevance": len(target_users - relevant_targets),
        "template_users_cold_at_train_cutoff": len(target_users - train_targets),
        "cold_users_with_nonempty_validation_relevance": len(relevant_targets - train_targets),
        "template_user_validation_program_pairs": val_pairs,
        "those_pairs_seen_by_same_user_in_train": repeated_pairs,
        "full_history_template_users_without_120s_airing": len(target_users - threshold_users[0][120] - threshold_users[1][120]),
        "all_train_user_basket_pair_enumerations": sum(len(s) * (len(s) - 1) // 2 for s in histories_120[0].values())}
    output["elapsed_seconds"] = round(time.monotonic() - begun, 3)
    output["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    filename = "step1_intervals_kyiv.json" if CLOCK == "kyiv" else "step1_intervals.json"
    (ROOT / "results" / filename).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    db.close()
    print("Saved", filename, "; seconds", output["elapsed_seconds"], "; peak MiB", output["peak_rss_mib"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", choices=["naive", "kyiv"], default="naive")
    CLOCK = parser.parse_args().clock
    main()
