"""Independent second-by-second oracle, hand fixtures, and pipeline integration tests.

The oracle deliberately does not use the production union, overlap, timestamp
parser, or indexing code. It is only for short deterministic test intervals.
"""

from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

import data_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


def brute_force(views, airings, window):
    """Enumerate each elapsed second and ask if ANY same-channel event covers it."""
    result = {}
    for user in sorted({v[0] for v in views}):
        for channel, start, end, item in set(map(tuple, airings)):
            if item <= 0 or end <= start:
                continue
            watched = []
            for second in range(max(start, window[0]), min(end, window[1])):
                if any(u == user and c == channel and s <= second < e for u, c, s, e in views):
                    watched.append(second)
            if watched:
                result[(user, channel, start, end, item)] = (len(watched), watched[0], watched[-1] + 1)
    return result


def optimized(views, airings, window):
    return pipeline.match_views([pipeline.View(*v) for v in views], [pipeline.Airing(*a) for a in airings], window)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class PipelineTests(unittest.TestCase):
    def test_seeded_brute_force_equivalence_100_cases(self):
        rng = random.Random(42)
        for number in range(100):
            airings = [[c, s, s + rng.randint(5, 20), rng.choice([0, 11, 22, 33])]
                       for c in (1, 2) for s in range(0, 120, 20)]
            views = [[rng.choice(["0001", "18446744073709551615", "cold"]), rng.randint(1, 3),
                      s, s + rng.randint(1, 70)] for s in [rng.randint(-20, 120) for _ in range(20)]]
            views.extend(views[:3])
            rng.shuffle(views)
            window = (rng.randint(0, 40), rng.randint(80, 150))
            with self.subTest(case=number):
                self.assertEqual(optimized(views, airings, window), brute_force(views, airings, window))

    def test_no_future_behavior_changes_training(self):
        schedules = [[1, 0, 200, 11], [1, 200, 500, 22]]
        base = [["u", 1, 50, 100]]
        future = [["u", 1, 100, 400], ["new", 1, 150, 500]]
        self.assertEqual(optimized(base, schedules, (0, 100)), optimized(base + future, schedules, (0, 100)))

    def test_spring_dst_elapsed_seconds_and_broadcast_end(self):
        start = pipeline.parse_timestamp("2020-03-29 02:30:00")
        end = pipeline.parse_timestamp("2020-03-29 04:30:00")
        expected_start = int(datetime(2020, 3, 29, 0, 30, tzinfo=timezone.utc).timestamp())
        self.assertEqual(start, expected_start)
        self.assertEqual(end - start, 3600)
        row = ["1", "channel", "29.03.2020 02:30:00", "3600", "title", "11", "category", "", "", "", "", "", ""]
        airing = pipeline.parse_airing(row)
        self.assertEqual(airing.end, end)
        view = pipeline.parse_view(["18440717181607000261", "1", "2020-03-29 02:45:00", "2020-03-29 04:15:00", "1800"])
        self.assertEqual(next(iter(pipeline.match_views([view], [airing], (start, end)).values())).seconds, 1800)

    def test_nonexistent_local_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonexistent"):
            pipeline.parse_timestamp("2020-03-29 03:30:00")

    def test_autumn_dst_ambiguous_requires_fold(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            pipeline.parse_timestamp("2020-10-25 03:30:00")
        first = pipeline.parse_timestamp("2020-10-25 03:30:00", fold=0)
        second = pipeline.parse_timestamp("2020-10-25 03:30:00", fold=1)
        self.assertEqual(second - first, 3600)

    def test_explicit_timestamp_schemas_and_invalid_dates(self):
        self.assertEqual(pipeline.parse_timestamp("2020-05-18 12:34:56"), pipeline.parse_timestamp("18.05.2020 12:34:56", schedule=True))
        for value in ("2020-5-18 12:34:56", "05/18/2020 12:34:56", "2020-02-30 00:00:00", "2020-05-18 24:00:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.parse_timestamp(value)

    def test_calendar_boundaries_use_different_offsets(self):
        b = pipeline.windows()
        expected = [(2020, 3, 8, 22), (2020, 5, 17, 21), (2020, 7, 26, 21), (2020, 10, 18, 21)]
        actual = [b["train"][0], b["train"][1], b["validation"][1], b["test"][1]]
        self.assertEqual(actual, [int(datetime(*v, tzinfo=timezone.utc).timestamp()) for v in expected])

    def test_aggregation_before_threshold_and_threshold_reuse(self):
        views = [["u", 1, 0, 70], ["u", 1, 150, 220]]
        matches = optimized(views, [[1, 0, 300, 11]], (0, 300))
        value = next(iter(matches.values()))
        rows = [(11, *value)]
        self.assertEqual(pipeline.aggregate_interactions(rows, 120)[11][1], 140)
        self.assertEqual(pipeline.aggregate_interactions(rows, 141), {})
        self.assertEqual(pipeline.aggregate_interactions([(11, 119, 0, 119)], 120), {})
        self.assertIn(11, pipeline.aggregate_interactions([(11, 120, 0, 120)], 120))

    def test_separate_airings_do_not_jointly_pass_threshold(self):
        self.assertEqual(pipeline.aggregate_interactions([(11, 70, 0, 70), (11, 70, 200, 270)], 120), {})

    def test_boundary_airing_recombined_before_history_threshold(self):
        a = pipeline.Airing(1, 50, 300, 11)
        index = pipeline.AiringIndex([a])
        train = pipeline.match_channel([(70, 230)], index, (0, 150))
        val = pipeline.match_channel([(70, 230)], index, (150, 300))
        combined = pipeline.combine_period_matches(train, val)
        self.assertEqual(train[a].seconds, 80)
        self.assertEqual(val[a].seconds, 80)
        self.assertEqual(combined[a].seconds, 160)
        self.assertEqual(combined, pipeline.match_channel([(70, 230)], index, (0, 300)))

    def test_duplicate_airing_and_conflicting_schedule(self):
        a = pipeline.Airing(1, 0, 100, 11)
        self.assertEqual(len(pipeline.AiringIndex([a, a]).airings), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            pipeline.AiringIndex([a, pipeline.Airing(1, 50, 150, 22)])
        with self.assertRaisesRegex(ValueError, "one channel"):
            pipeline.AiringIndex([a, pipeline.Airing(2, 0, 100, 22)])

    def test_empty_history_and_unlabelled_only_are_empty(self):
        self.assertEqual(optimized([], [[1, 0, 200, 11]], (0, 200)), {})
        self.assertEqual(optimized([["u", 1, 0, 200]], [[1, 0, 200, 0]], (0, 200)), {})

    def test_coverage_counts_union_not_sum(self):
        index = pipeline.CoverageIndex([(0, 100), (50, 200), (250, 300)])
        self.assertEqual(index.seconds(75, 275), 150)
        self.assertEqual(index.seconds(200, 250), 0)

    def test_original_user_id_preserved_by_parser(self):
        for user in ("000123", "18446744073709551615", "36938861791891792"):
            view = pipeline.parse_view([user, "1", "2020-05-18 00:00:00", "2020-05-18 00:02:00", "120"])
            self.assertIsInstance(view.user, str)
            self.assertEqual(view.user, user)

    def test_integration_csv_dedup_threshold_reuse_and_reproducibility(self):
        with tempfile.TemporaryDirectory(prefix="sweettv_test_", dir=ROOT / "tests") as temporary:
            directory = Path(temporary)
            source = directory / "originals"
            source.mkdir()
            def write(name, header, rows):
                with (source / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(rows)
            def schedule(start, duration, item):
                return ["1", "channel", start, str(duration), "title", str(item), "category", "", "", "", "", "", ""]
            views = [
                ["000123", "1", "2020-05-17 23:58:40", "2020-05-18 00:01:20", "160"],
                ["000123", "1", "2020-05-17 23:58:40", "2020-05-18 00:01:20", "160"],
                ["18446744073709551615", "1", "2020-03-09 00:00:00", "2020-03-09 00:02:00", "120"],
                ["000123", "1", "2020-07-27 00:00:00", "2020-07-27 00:02:00", "120"],
                ["99", "1", "2020-05-18 00:10:00", "2020-05-18 00:12:00", "120"]]
            write(pipeline.FILES[0], pipeline.VIEW_COLUMNS, views)
            write(pipeline.FILES[1], pipeline.SCHEDULE_COLUMNS,
                  [schedule("09.03.2020 00:00:00", 300, 11), schedule("17.05.2020 23:58:00", 300, 22),
                   schedule("29.03.2020 04:00:00", 0, 0)])
            write(pipeline.FILES[2], pipeline.SCHEDULE_COLUMNS, [schedule("18.05.2020 00:10:00", 300, 0)])
            write(pipeline.FILES[3], pipeline.SCHEDULE_COLUMNS, [schedule("27.07.2020 00:00:00", 300, 33)])
            ids = ["000123", "18446744073709551615", "555", "99"]
            write(pipeline.FILES[4], ["user_id", "tv_show_id"], [[u, "0 0 0 0 0"] for u in ids])
            before = pipeline.input_manifest(source)
            join = directory / "join"
            built = pipeline.build(source, join)
            selected = pipeline.derive(join, 120)
            self.assertEqual(built["stats"]["views"]["exact_duplicate_records_removed"], 1)
            self.assertEqual(built["stats"]["views"]["outside_history_records_excluded"], 1)
            self.assertEqual(selected["periods"]["train"]["unique_user_program_pairs"], 1)
            self.assertEqual(selected["periods"]["validation"]["unique_user_program_pairs"], 0)
            self.assertEqual(selected["periods"]["history"]["unique_user_program_pairs"], 2)
            coverage = read_csv(join / "threshold_120s" / "template_user_coverage.csv")
            self.assertEqual([r["user_id"] for r in coverage], ids)
            self.assertEqual(coverage[2]["history_programs"], "0")
            self.assertEqual(coverage[3]["history_programs"], "0")
            self.assertEqual(pipeline.input_manifest(source), before)
            # Hide all raw inputs: thresholding must depend ONLY on the join cache.
            source.rename(directory / "hidden_originals")
            alternate = pipeline.derive(join, 60)
            self.assertEqual(alternate["periods"]["train"]["unique_user_program_pairs"], 2)
            self.assertEqual(alternate["periods"]["validation"]["unique_user_program_pairs"], 1)
            replay = pipeline.derive(join, 120, directory / "replay")
            self.assertEqual(selected["artifacts"], replay["artifacts"])
            # Rebuild twice from originals to check stable join and gzip ordering.
            rebuilt = pipeline.build(directory / "hidden_originals", directory / "join_replay")
            self.assertEqual(built["artifacts"], rebuilt["artifacts"])
            # Cache tampering must be detected before producing a new threshold.
            with (join / "train_candidates.csv").open("a") as stream:
                stream.write("999999\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                pipeline.derive(join, 121)


def fixture_test(case):
    def test(self):
        actual = optimized(case["views"], case["airings"], case["window"])
        expected = {tuple(r[:5]): tuple(r[5:]) for r in case["expected"]}
        self.assertEqual(brute_force(case["views"], case["airings"], case["window"]), expected)
        self.assertEqual(actual, expected)
    return test


for fixture in json.loads((ROOT / "tests" / "interval_fixtures.json").read_text())["cases"]:
    setattr(PipelineTests, "test_fixture_" + fixture["name"], fixture_test(fixture))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed", "traceback": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error", "traceback": self._exc_info_to_string(err, test)})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.records.append({"test": subtest.id(), "status": "failed", "traceback": self._exc_info_to_string(err, test)})


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    runs = ROOT / "results" / "step2_test_runs"
    runs.mkdir(exist_ok=True)
    number = len(list(runs.glob("run_*.json"))) + 1
    pipeline.write_json(runs / f"run_{number:03d}.json", {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "all_passed": result.wasSuccessful(), "seconds": round(time.monotonic() - started, 3),
        "test_source_sha256": pipeline.sha256(__file__), "pipeline_sha256": pipeline.sha256(pipeline.__file__),
        "independent_oracle": "second-by-second membership, no optimized helpers",
        "seeded_random_cases": 100, "checks": result.records})
    raise SystemExit(0 if result.wasSuccessful() else 1)
