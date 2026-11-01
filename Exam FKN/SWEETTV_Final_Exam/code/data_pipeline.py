"""Threshold-independent, chronological SWEET.TV preprocessing (standard library).

Europe/Kyiv is an evidence-supported inference, NOT an official competition rule.
All views are localized independently and all arithmetic uses UTC integer seconds.
No recommenders, prediction lists, or recommendation metrics are implemented here.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import re
import resource
import sqlite3
import time
from typing import NamedTuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FILES = ("dataset11-30.csv", "export_arh_11-20-final.csv",
         "export_arh_21-30-final.csv", "export_arh_31-42-final.csv", "submission.csv")
VIEW_COLUMNS = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
SCHEDULE_COLUMNS = ["channel_id", "channel_title", "start_time", "duration", "tv_show_title",
                    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                    "tv_show_genre_3", "year_of_production", "director", "actors"]
OVERLAP_COLUMNS = ["user_id", "channel_id", "broadcast_start_utc", "broadcast_end_utc",
                   "tv_show_id", "overlap_seconds", "first_overlap_utc", "last_overlap_utc"]
ZONE = ZoneInfo("Europe/Kyiv")
EPOCH = datetime(1970, 1, 1)
VIEW_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\Z")
SCHEDULE_PATTERN = re.compile(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}\Z")
LOCAL_BOUNDARIES = ("2020-03-09 00:00:00", "2020-05-18 00:00:00",
                    "2020-07-27 00:00:00", "2020-10-19 00:00:00")


class View(NamedTuple):
    user: str
    channel: int
    start: int
    end: int


class Airing(NamedTuple):
    channel: int
    start: int
    end: int
    item: int


class Overlap(NamedTuple):
    seconds: int
    first: int
    last: int


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@lru_cache(maxsize=1024)
def stable_day_offset(day):
    """Fast path only for Kyiv dates with the same offset throughout the day.

    Transition days use explicit fold/round-trip validation below. The fixed zone
    has no same-day pair of offset transitions; no fixed offset is used globally.
    """
    base = datetime.combine(day, datetime.min.time())
    offsets = {(base + timedelta(hours=h)).replace(tzinfo=ZONE).utcoffset()
               for h in (0, 12, 24)}
    return int(next(iter(offsets)).total_seconds()) if len(offsets) == 1 else None


def parse_timestamp(value, schedule=False, fold=None):
    pattern = SCHEDULE_PATTERN if schedule else VIEW_PATTERN
    if not pattern.fullmatch(value):
        raise ValueError(f"Unexpected timestamp format: {value!r}")
    if not (0 <= int(value[11:13]) < 24 and 0 <= int(value[14:16]) < 60
            and 0 <= int(value[17:19]) < 60):
        raise ValueError(f"Out-of-range clock fields: {value!r}")
    if fold not in (None, 0, 1):
        raise ValueError("Invalid fold")
    text = value[6:10] + "-" + value[3:5] + "-" + value[:2] + value[10:] if schedule else value
    naive = datetime.fromisoformat(text)
    offset = stable_day_offset(naive.date())
    if offset is not None:
        return int((naive - EPOCH).total_seconds()) - offset
    options = {}
    for candidate_fold in (0, 1):
        aware = naive.replace(tzinfo=ZONE, fold=candidate_fold)
        utc = aware.astimezone(timezone.utc)
        if utc.astimezone(ZONE).replace(tzinfo=None) == naive:
            options[candidate_fold] = int(utc.timestamp())
    if not options:
        raise ValueError(f"Nonexistent local time in inferred Europe/Kyiv zone: {value}")
    if len(set(options.values())) > 1 and fold is None:
        raise ValueError(f"Ambiguous local time needs an explicit fold: {value}")
    if fold is not None and fold not in options:
        raise ValueError("Invalid fold")
    return options[fold] if fold is not None else next(iter(options.values()))


def windows():
    a, b, c, d = map(parse_timestamp, LOCAL_BOUNDARIES)
    return {"train": (a, b), "validation": (b, c), "history": (a, c), "test": (c, d)}


def csv_rows(path, columns):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != columns:
            raise ValueError(f"Schema mismatch: {path}")
        for number, row in enumerate(reader, 2):
            if len(row) != len(columns):
                raise ValueError(f"Malformed CSV record: {path}:{number}")
            yield number, row


def parse_view(row):
    user, channel, start, end, duration = row
    if not user or not user.isascii() or not user.isdecimal():
        raise ValueError("user_id must be an unmodified decimal string")
    start, end = parse_timestamp(start), parse_timestamp(end)
    if end <= start or end - start != int(duration):
        raise ValueError("Invalid viewing duration or elapsed-time mismatch")
    return View(user, int(channel), start, end)


def parse_airing(row):
    start = parse_timestamp(row[2], schedule=True)
    duration, item = int(row[3]), int(row[5])
    if item < 0 or duration < 0:
        raise ValueError("Negative program ID or broadcast duration")
    # Addition occurs AFTER conversion to UTC, never on a DST wall clock.
    return Airing(int(row[0]), start, start + duration, item)


def union_intervals(spans, window=None):
    clipped = []
    for start, end in spans:
        if end <= start:
            raise ValueError("Nonpositive viewing interval")
        if window is not None:
            start, end = max(start, window[0]), min(end, window[1])
        if start < end:
            clipped.append((start, end))
    merged = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class AiringIndex:
    """Sorted positive-ID airings for exactly one channel; fail on ambiguity."""

    def __init__(self, airings):
        self.airings = sorted({a for a in airings if a.item > 0 and a.end > a.start},
                              key=lambda a: (a.start, a.end, a.item))
        if len({a.channel for a in self.airings}) > 1:
            raise ValueError("An airing index must contain exactly one channel")
        self.starts, self.ends = [], []
        last_end = None
        for airing in self.airings:
            if last_end is not None and airing.start < last_end:
                raise ValueError(f"Conflicting labelled airings: {airing}")
            self.starts.append(airing.start)
            self.ends.append(airing.end)
            last_end = airing.end


def match_channel(spans, index, window):
    """Optimized interval join; accumulate fragments before applying ANY threshold."""
    totals = {}
    for start, end in union_intervals(spans, window):
        left, right = bisect_right(index.ends, start), bisect_left(index.starts, end)
        for i in range(left, right):
            airing = index.airings[i]
            first, last = max(start, airing.start), min(end, airing.end)
            if first < last:
                old = totals.get(airing)
                totals[airing] = Overlap(last - first + (old.seconds if old else 0),
                                        min(old.first, first) if old else first,
                                        max(old.last, last) if old else last)
    for airing, value in totals.items():
        if not (0 < value.seconds <= airing.end - airing.start
                and window[0] <= value.first < value.last <= window[1]):
            raise AssertionError("Invalid credited seconds or temporal leakage")
    return totals


def match_views(views, airings, window):
    """Small-data API, also used to independently test production matching logic."""
    schedules, groups = defaultdict(list), defaultdict(list)
    for airing in airings:
        schedules[airing.channel].append(airing)
    for view in views:
        groups[(view.user, view.channel)].append((view.start, view.end))
    indexes = {c: AiringIndex(a) for c, a in schedules.items()}
    result = {}
    empty = AiringIndex([])
    for (user, channel), spans in groups.items():
        for airing, overlap in match_channel(spans, indexes.get(channel, empty), window).items():
            result[(user, *airing)] = overlap
    return result


def combine_period_matches(first, second):
    """Combine disjoint-window overlaps, BEFORE thresholding, for full history."""
    result = dict(first)
    for airing, value in second.items():
        old = result.get(airing)
        result[airing] = Overlap(old.seconds + value.seconds,
                                min(old.first, value.first), max(old.last, value.last)) if old else value
        if result[airing].seconds > airing.end - airing.start:
            raise AssertionError("Period merge double-counted a broadcast")
    return result


class CoverageIndex:
    """Union schedule coverage, including ID zero, for missing-label diagnostics."""

    def __init__(self, spans):
        merged = union_intervals(spans)
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]
        self.prefix = [0]
        for s, e in merged:
            self.prefix.append(self.prefix[-1] + e - s)

    def seconds(self, start, end):
        left, right = bisect_right(self.ends, start), bisect_left(self.starts, end)
        if left >= right:
            return 0
        return (self.prefix[right] - self.prefix[left]
                - max(0, start - self.starts[left]) - max(0, self.ends[right - 1] - end))


@contextmanager
def compressed_csv(path):
    # Stable gzip bytes: no embedded file name or creation time.
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                yield csv.writer(text, lineterminator="\n")


def input_manifest(source):
    return {name: {"path": str((source / name).resolve()), "bytes": (source / name).stat().st_size,
                   "sha256": sha256(source / name)} for name in FILES}


def check_expected_hashes(manifest, expected_path):
    if expected_path is None:
        return
    expected = {Path(p).name: info["sha256"] for p, info in json.loads(Path(expected_path).read_text()).items()}
    for name, info in manifest.items():
        if expected.get(name) != info["sha256"]:
            raise ValueError(f"Original hash mismatch before preprocessing: {name}")


def open_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    return db


def build(source, output, expected_path=None):
    started = time.monotonic()
    source, output = Path(source).resolve(), Path(output).resolve()
    before = input_manifest(source)
    check_expected_hashes(before, expected_path)
    output.mkdir(parents=True, exist_ok=False)
    bounds = windows()
    stats = {"schedule_files": {}, "views": {}, "periods": {}, "stage_seconds": {}}
    template = []
    for _, row in csv_rows(source / FILES[-1], ["user_id", "tv_show_id"]):
        if not row[0].isascii() or not row[0].isdecimal():
            raise ValueError("Invalid template user ID")
        template.append(row[0])
    if len(set(template)) != len(template):
        raise ValueError("Duplicate template user IDs")
    with (output / "template_users.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id"])
        writer.writerows((u,) for u in template)
    db = open_db(output / "working.sqlite")
    db.execute("CREATE TABLE schedule(c INTEGER,s INTEGER,e INTEGER,item INTEGER)")
    db.execute("CREATE TABLE views(c INTEGER,u TEXT,s INTEGER,e INTEGER,original TEXT)")
    catalogs = {p: set() for p in bounds}
    stage = time.monotonic()
    for name in FILES[1:4]:
        counts, batch = Counter(), []
        for _, row in csv_rows(source / name, SCHEDULE_COLUMNS):
            counts["records"] += 1
            airing = parse_airing(row)
            counts["zero_id_rows"] += airing.item == 0
            if airing.end == airing.start:
                counts["zero_duration_rows_excluded"] += 1
                continue
            for period, (lo, hi) in bounds.items():
                if airing.item > 0 and airing.start < hi and airing.end > lo:
                    catalogs[period].add(airing.item)
            # Future schedules are availability-only, never joined to user behavior.
            if name != FILES[3] and airing.start < bounds["history"][1] and airing.end > bounds["history"][0]:
                batch.append(tuple(airing))
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO schedule VALUES (?,?,?,?)", batch)
                    batch.clear()
        db.executemany("INSERT INTO schedule VALUES (?,?,?,?)", batch)
        db.commit()
        stats["schedule_files"][name] = dict(counts)
    db.execute("CREATE INDEX schedule_order ON schedule(c,s,e,item)")
    for period, items in catalogs.items():
        with (output / f"{period}_candidates.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["tv_show_id"])
            writer.writerows((item,) for item in sorted(items))
    stats["candidate_counts"] = {p: len(v) for p, v in catalogs.items()}
    stats["stage_seconds"]["schedule_ingestion"] = round(time.monotonic() - stage, 3)
    print("Schedules ingested", stats["candidate_counts"], flush=True)
    stage = time.monotonic()
    counts, batch = Counter(), []
    users_all = set()
    view_users = {p: set() for p in ("train", "validation", "history")}
    period_counts = {p: Counter() for p in view_users}
    for _, row in csv_rows(source / FILES[0], VIEW_COLUMNS):
        counts["raw_records"] += 1
        view = parse_view(row)
        users_all.add(view.user)
        counts["raw_elapsed_seconds"] += view.end - view.start
        counts["crosses_train_validation_boundary"] += view.start < bounds["train"][1] < view.end
        for period in view_users:
            lo, hi = bounds[period]
            if view.start < hi and view.end > lo:
                view_users[period].add(view.user)
                period_counts[period]["raw_records_touching_period"] += 1
        if view.start >= bounds["history"][1] or view.end <= bounds["history"][0]:
            counts["outside_history_records_excluded"] += 1
            continue
        batch.append((view.channel, view.user, view.start, view.end,
                      json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
        if len(batch) >= 10000:
            db.executemany("INSERT INTO views VALUES (?,?,?,?,?)", batch)
            batch.clear()
    db.executemany("INSERT INTO views VALUES (?,?,?,?,?)", batch)
    db.commit()
    db.execute("CREATE INDEX view_order ON views(c,u,s,e,original)")
    db.commit()
    counts["unique_raw_users"] = len(users_all)
    stats["stage_seconds"]["view_ingestion_and_index"] = round(time.monotonic() - stage, 3)
    print("Views ingested", counts["raw_records"], flush=True)
    stage = time.monotonic()
    overlap_users = {p: set() for p in view_users}
    overlap_items = {p: set() for p in view_users}
    with ExitStack() as stack:
        writers = {p: stack.enter_context(compressed_csv(output / f"{p}_airing_overlaps.csv.gz")) for p in view_users}
        for writer in writers.values():
            writer.writerow(OVERLAP_COLUMNS)
        cursor = db.execute("SELECT c,u,s,e,original FROM views ORDER BY c,u,s,e,original")
        for channel, rows in itertools.groupby(cursor, lambda r: r[0]):
            airings = [Airing(*r) for r in db.execute("SELECT c,s,e,item FROM schedule WHERE c=? ORDER BY s,e,item", (channel,))]
            all_coverage = CoverageIndex((a.start, a.end) for a in airings)
            index = AiringIndex(airings)
            label_coverage = CoverageIndex((a.start, a.end) for a in index.airings)
            for user, group in itertools.groupby(rows, lambda r: r[1]):
                spans, previous = [], None
                for _, _, start, end, original in group:
                    # Match-status statistics include duplicates to compare with Step 1.
                    cov, lab = all_coverage.seconds(start, end), label_coverage.seconds(start, end)
                    counts["raw_rows_labelled_overlap"] += lab > 0
                    counts["raw_rows_schedule_without_label"] += cov > 0 and lab == 0
                    counts["raw_rows_without_schedule_overlap"] += cov == 0
                    counts["raw_rows_missing_schedule_channel"] += not airings
                    if original == previous:
                        counts["exact_duplicate_records_removed"] += 1
                        continue
                    previous = original
                    spans.append((start, end))
                merged = union_intervals(spans, bounds["history"])
                counts["merged_history_segments"] += len(merged)
                counts["union_history_seconds"] += sum(e - s for s, e in merged)
                counts["union_labelled_seconds"] += sum(label_coverage.seconds(s, e) for s, e in merged)
                counts["union_schedule_seconds"] += sum(all_coverage.seconds(s, e) for s, e in merged)
                matches = {p: match_channel(spans, index, bounds[p]) for p in ("train", "validation")}
                matches["history"] = combine_period_matches(matches["train"], matches["validation"])
                for period, values in matches.items():
                    for airing, overlap in sorted(values.items()):
                        writers[period].writerow((user, *airing, *overlap))
                        period_counts[period]["positive_user_airings"] += 1
                        period_counts[period]["labelled_overlap_seconds"] += overlap.seconds
                        overlap_users[period].add(user)
                        overlap_items[period].add(airing.item)
    db.close()
    (output / "working.sqlite").unlink()  # Only this run's owned disposable workspace.
    counts["unique_events_after_exact_dedup"] = counts["raw_records"] - counts["exact_duplicate_records_removed"] - counts["outside_history_records_excluded"]
    stats["views"] = dict(counts)
    for period in view_users:
        stats["periods"][period] = {**period_counts[period], "view_users": len(view_users[period]),
                                    "overlap_users": len(overlap_users[period]), "overlap_programs": len(overlap_items[period]),
                                    "template_view_users": len(set(template) & view_users[period]),
                                    "template_overlap_users": len(set(template) & overlap_users[period])}
    with (output / "user_coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "in_template", *[f"{p}_has_view" for p in view_users]])
        for user in sorted(users_all | set(template)):
            writer.writerow([user, int(user in set(template)), *[int(user in view_users[p]) for p in view_users]])
    stats["stage_seconds"]["interval_join"] = round(time.monotonic() - stage, 3)
    after = input_manifest(source)
    if before != after:
        raise RuntimeError("Original inputs changed during preprocessing")
    artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()}
    stats["runtime_seconds"] = round(time.monotonic() - started, 3)
    stats["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    manifest = {"schema_version": 1, "timezone": "Europe/Kyiv", "timezone_status": "evidence-supported inference, not official",
                "windows_utc": bounds, "local_boundaries": LOCAL_BOUNDARIES,
                "source_files_before": before, "source_files_after": after, "source_integrity_passed": True,
                "pipeline_sha256": sha256(__file__), "python": platform.python_version(), "stats": stats,
                "artifacts": artifacts, "threshold_applied": False,
                "history_policy": "combine disjoint train/validation overlaps per airing before any threshold"}
    write_json(output / "join_manifest.json", manifest)
    print("JOIN COMPLETE", stats["runtime_seconds"], "seconds; peak MiB", stats["peak_rss_mib"], flush=True)
    return manifest


def aggregate_interactions(rows, threshold):
    """Aggregate qualifying broadcasts; suitable for one user's rows at a time."""
    if threshold <= 0:
        raise ValueError("Threshold must be positive seconds")
    result = {}
    for item, seconds, first, last in rows:
        if item <= 0:
            raise ValueError("Unlabelled program in overlap artifact")
        if seconds < threshold:
            continue
        old = result.get(item)
        result[item] = (old[0] + 1, old[1] + seconds, min(old[2], first), max(old[3], last)) if old else (1, seconds, first, last)
    return result


def derive(join_dir, threshold=120, output=None):
    """Re-threshold only persisted overlaps; does not read original viewing data."""
    started = time.monotonic()
    join_dir = Path(join_dir).resolve()
    output = Path(output).resolve() if output else join_dir / f"threshold_{threshold}s"
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    manifest = json.loads((join_dir / "join_manifest.json").read_text())
    for name, info in manifest["artifacts"].items():
        if sha256(join_dir / name) != info["sha256"]:
            raise ValueError(f"Derived input checksum mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    with (join_dir / "template_users.csv").open(newline="") as stream:
        template = [r["user_id"] for r in csv.DictReader(stream)]
    result = {"threshold_seconds": threshold, "periods": {}, "join_manifest_sha256": sha256(join_dir / "join_manifest.json"),
              "pipeline_sha256": sha256(__file__), "no_raw_join_rebuilt": True}
    coverage = {u: {} for u in template}
    db = open_db(output / "working.sqlite")
    for period in ("train", "validation", "history"):
        db.execute("DROP TABLE IF EXISTS qualifying")
        db.execute("CREATE TABLE qualifying(u TEXT,item INTEGER,seconds INTEGER,first INTEGER,last INTEGER)")
        batch, positives, qualifying, total_seconds = [], 0, 0, 0
        with gzip.open(join_dir / f"{period}_airing_overlaps.csv.gz", "rt", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != OVERLAP_COLUMNS:
                raise ValueError("Overlap artifact schema mismatch")
            for row in reader:
                positives += 1
                seconds, item = int(row["overlap_seconds"]), int(row["tv_show_id"])
                if item <= 0:
                    raise ValueError("Nonpositive program ID in overlap cache")
                if seconds >= threshold:
                    qualifying += 1
                    total_seconds += seconds
                    batch.append((row["user_id"], item, seconds, int(row["first_overlap_utc"]), int(row["last_overlap_utc"])))
                    if len(batch) >= 10000:
                        db.executemany("INSERT INTO qualifying VALUES (?,?,?,?,?)", batch)
                        batch.clear()
        db.executemany("INSERT INTO qualifying VALUES (?,?,?,?,?)", batch)
        db.commit()
        db.execute("CREATE INDEX qualifying_user ON qualifying(u,item)")
        users, items, pair_count, template_pairs = set(), set(), 0, 0
        with compressed_csv(output / f"{period}_interactions.csv.gz") as writer:
            writer.writerow(["user_id", "tv_show_id", "qualifying_airings", "watched_seconds", "first_watch_utc", "last_watch_utc"])
            for user, group in itertools.groupby(db.execute("SELECT u,item,seconds,first,last FROM qualifying ORDER BY u,item"), lambda r: r[0]):
                pairs = aggregate_interactions((r[1:] for r in group), threshold)
                users.add(user)
                for item, values in sorted(pairs.items()):
                    writer.writerow((user, item, *values))
                    items.add(item)
                    pair_count += 1
                if user in coverage:
                    coverage[user][period] = len(pairs)
                    template_pairs += len(pairs)
        result["periods"][period] = {"positive_user_airings": positives, "qualifying_user_airings": qualifying,
                                      "unique_user_program_pairs": pair_count, "unique_users": len(users), "unique_programs": len(items),
                                      "qualifying_seconds": total_seconds, "template_users_with_interactions": len(users & set(template)),
                                      "template_users_without_interactions": len(set(template) - users), "template_user_program_pairs": template_pairs}
    with (output / "template_user_coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "train_programs", "validation_programs", "history_programs"])
        for user in template:
            writer.writerow([user, *[coverage[user].get(p, 0) for p in ("train", "validation", "history")]])
    db.close()
    (output / "working.sqlite").unlink()
    result["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()}
    result["runtime_seconds"] = round(time.monotonic() - started, 3)
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(output / "interaction_manifest.json", result)
    print("DERIVE COMPLETE", threshold, "seconds threshold;", result["runtime_seconds"], "seconds; peak MiB", result["peak_rss_mib"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--source-dir", type=Path, default=ROOT.parent / "sweettv-tv-program-recommender")
    make.add_argument("--output-dir", type=Path, default=ROOT / "data" / "step2")
    make.add_argument("--expected-hashes", type=Path, default=ROOT / "results" / "original_sha256_before.json")
    label = sub.add_parser("derive")
    label.add_argument("--join-dir", type=Path, default=ROOT / "data" / "step2")
    label.add_argument("--threshold-seconds", type=int, default=120)
    label.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.source_dir, args.output_dir, args.expected_hashes)
    else:
        derive(args.join_dir, args.threshold_seconds, args.output_dir)


if __name__ == "__main__":
    main()
