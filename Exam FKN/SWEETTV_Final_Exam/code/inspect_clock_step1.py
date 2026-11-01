"""Check the observed one-hour duration discrepancy; no timezone is assumed official."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from zoneinfo import ZoneInfo

from inspect_step1 import ROOT, SOURCE, FILES, stamp


def main():
    zone = ZoneInfo("Europe/Kyiv")
    counts = Counter()
    mismatch_dates = Counter()
    for row in csv.DictReader((SOURCE / FILES[0]).open(newline="")):
        start, end = row["start_time"], row["stop_time"]
        duration = int(row["duraton"])
        naive_delta = stamp(end) - stamp(start)
        if start[:10] == "2020-03-29":
            counts["view_starts_on_transition_date"] += 1
            counts["view_starts_in_local_hour_03"] += start[11:13] == "03"
        if end[:10] == "2020-03-29":
            counts["view_stops_in_local_hour_03"] += end[11:13] == "03"
        if start < "2020-03-29 04:00:00" <= end:
            counts["views_crossing_04_transition"] += 1
        if naive_delta != duration:
            mismatch_dates[start[:10] + " to " + end[:10]] += 1
        # Outside the only spring offset transition, both clocks yield identical deltas.
        if start < "2020-03-30" and end >= "2020-03-28":
            a = datetime.fromisoformat(start).replace(tzinfo=zone).astimezone(timezone.utc)
            b = datetime.fromisoformat(end).replace(tzinfo=zone).astimezone(timezone.utc)
            counts["near_transition_rows_checked"] += 1
            counts["kyiv_utc_elapsed_mismatches_near_transition"] += int((b - a).total_seconds()) != duration
    day_intervals = defaultdict(list)
    for row in csv.DictReader((SOURCE / FILES[1]).open(newline="", encoding="utf-8")):
        value = row["start_time"]
        if value[:10] in {"28.03.2020", "29.03.2020", "30.03.2020"}:
            s, d = stamp(value, True), int(row["duration"])
            day_intervals[int(row["channel_id"])].append((s, d, int(row["tv_show_id"]), value))
    next_gaps = Counter()
    examples = []
    for channel, records in day_intervals.items():
        records.sort()
        for current, nxt in zip(records, records[1:]):
            s, d, item, value = current
            if s < stamp("2020-03-29 04:00:00") <= nxt[0]:
                gap = nxt[0] - s - d
                next_gaps[gap] += 1
                if len(examples) < 10:
                    examples.append({"channel": channel, "start": value, "duration": d,
                                     "next_start": nxt[3], "wall_clock_gap_minus_duration": gap})
    profile = json.loads((ROOT / "results" / "step1_profile.json").read_text())
    viewed = set(profile["files"][FILES[0]]["channels"])
    historical = set(profile["files"][FILES[1]]["channels"]) | set(profile["files"][FILES[2]]["channels"])
    result = {"hypothesis": "Europe/Kyiv local wall clock; not confirmed by competition",
              "counts": dict(counts), "naive_duration_mismatch_date_pairs": dict(mismatch_dates),
              "schedule_transition_next_start_minus_start_minus_duration": dict(next_gaps),
              "schedule_transition_examples": examples,
              "view_channels_absent_both_historical_schedules": sorted(viewed - historical)}
    (ROOT / "results" / "step1_clock.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
