"""Independent rational AP reference and strict evaluation/temporal contract tests."""

from dataclasses import replace
from fractions import Fraction
import csv
import gzip
import json
from pathlib import Path
import random
import resource
import tempfile
import time
import unittest

import evaluation as ev
import validation as val

ROOT = Path(__file__).resolve().parents[1]


def reference_ap(relevant, predictions):
    """Independent exact arithmetic: find each relevant item's first rank.

    Unlike production's running hit counter, this iterates relevant items and
    recomputes the distinct relevant prefix for their first appearance.
    Synthetic integer/string tokens only; no production helpers are reused.
    """
    truth = set(map(str, relevant))
    ranking = list(map(str, predictions[:5]))
    total = Fraction(0)
    for item in truth:
        if item in ranking:
            rank = ranking.index(item) + 1
            prefix_hits = len(truth.intersection(set(ranking[:rank])))
            total += Fraction(prefix_hits, rank)
    return total / min(5, len(truth)) if truth else Fraction(0)


class MetricTests(unittest.TestCase):
    def test_example_A_hand_computed(self):
        expected = (Fraction(1, 1) + Fraction(2, 3) + Fraction(3, 5)) / 3
        self.assertEqual(expected, Fraction(34, 45))
        self.assertAlmostEqual(ev.average_precision_at_5([10, 20, 30], [10, 99, 20, 88, 30]), float(expected), places=15)

    def test_example_B_hand_computed(self):
        expected = (Fraction(1, 2) + Fraction(2, 5)) / 2
        self.assertEqual(expected, Fraction(9, 20))
        self.assertAlmostEqual(ev.average_precision_at_5([1, 2], [3, 1, 4, 5, 2]), float(expected), places=15)

    def test_all_five_correct(self):
        self.assertEqual(ev.average_precision_at_5([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), 1)

    def test_no_correct_predictions(self):
        self.assertEqual(ev.average_precision_at_5([1, 2], [3, 4, 5, 6, 7]), 0)

    def test_order_changes_relevant_positions(self):
        score = ev.average_precision_at_5([10, 20, 30], [99, 10, 20, 88, 30])
        self.assertAlmostEqual(score, float(Fraction(53, 90)))
        self.assertLess(score, ev.average_precision_at_5([10, 20, 30], [10, 99, 20, 88, 30]))

    def test_single_relevant_item_at_rank_five(self):
        self.assertEqual(ev.average_precision_at_5([1], [2, 3, 4, 5, 1]), .2)

    def test_more_than_five_relevant_denominator_capped(self):
        self.assertEqual(ev.average_precision_at_5(list(range(1, 11)), [1, 2, 3, 4, 5]), 1)

    def test_repeated_truth_is_binary(self):
        self.assertEqual(ev.average_precision_at_5([1, 1, 1], [1]), 1)

    def test_duplicate_prediction_gets_no_repeated_credit(self):
        self.assertAlmostEqual(ev.average_precision_at_5([1, 2], [1, 1, 2, 9, 8]), float(Fraction(5, 6)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ev.average_precision_at_5([1, 2], [1, "1", 2, 9, 8], strict=True, candidates=list(range(1, 10)))

    def test_empty_prediction_and_short_ranking(self):
        self.assertEqual(ev.average_precision_at_5([1], []), 0)
        self.assertEqual(ev.average_precision_at_5([1, 2], [1]), .5)
        for ranking in ([], [1], [1, 2, 3, 4, 5, 6]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                ev.average_precision_at_5([1], ranking, strict=True, candidates=range(1, 7))

    def test_only_first_five_scored_in_mathematical_mode(self):
        self.assertEqual(ev.average_precision_at_5([6], [1, 2, 3, 4, 5, 6]), 0)

    def test_empty_truth_zero_but_still_strictly_validates(self):
        self.assertEqual(ev.average_precision_at_5([], [1, 2, 3, 4, 5], strict=True, candidates=range(1, 6)), 0)
        with self.assertRaises(ValueError):
            ev.average_precision_at_5([], [], strict=True, candidates=range(1, 6))

    def test_invalid_ids_and_unknown_eligibility(self):
        for item in (0, -1, True, 1.0, "1.0", "1e3", "01", " 1", "", None):
            with self.subTest(item=item), self.assertRaises(ValueError):
                ev.average_precision_at_5([], [item, 2, 3, 4, 5], strict=True, candidates=range(1, 6))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ev.average_precision_at_5([], [99, 2, 3, 4, 5], strict=True, candidates=range(1, 6))
        # Lenient invalid tokens consume rank; they do not compact the ranking.
        self.assertEqual(ev.average_precision_at_5([1], [0, 1]), .5)
        with self.assertRaises(ValueError):
            ev.average_precision_at_5([0], [1])

    def test_large_program_ids_are_lossless(self):
        item = 9007199254740993
        self.assertEqual(ev.program_id(item), "9007199254740993")
        self.assertEqual(ev.average_precision_at_5([str(item)], [item]), 1)

    def test_strict_requires_candidates_and_ordered_predictions(self):
        with self.assertRaises(ValueError):
            ev.average_precision_at_5([1], [1, 2, 3, 4, 5], strict=True)
        for ranking in ({1, 2, 3, 4, 5}, "1 2 3 4 5", None):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                ev.average_precision_at_5([1], ranking)

    def test_seeded_independent_fraction_reference_500_cases(self):
        rng = random.Random(42)
        for case in range(500):
            relevant = rng.sample(range(1, 31), rng.randrange(0, 20))
            ranking = [rng.randrange(0, 35) for _ in range(rng.randrange(0, 10))]
            with self.subTest(case=case):
                actual = ev.average_precision_at_5(relevant, ranking)
                self.assertAlmostEqual(actual, float(reference_ap(relevant, ranking)), places=15)
                self.assertTrue(0 <= actual <= 1)

    def test_macro_map_with_empty_user_and_per_user_values(self):
        cohort = ["0001", "18446744073709551615", "03"]
        truth = {"0001": [10, 20, 30], cohort[1]: [1, 2], "03": []}
        predictions = {"0001": [10, 99, 20, 88, 30], cohort[1]: [3, 1, 4, 5, 2], "03": [1, 2, 3, 4, 5]}
        result = ev.evaluate_predictions(predictions, truth, cohort, range(1, 100))
        self.assertAlmostEqual(result["primary_map_at_5"], float((Fraction(34, 45) + Fraction(9, 20)) / 3))
        self.assertAlmostEqual(result["diagnostic_map_at_5_nonempty"], float((Fraction(34, 45) + Fraction(9, 20)) / 2))
        self.assertEqual(list(result["per_user_ap_at_5"]), cohort)
        self.assertEqual(result["per_user_ap_at_5"]["03"], 0)
        self.assertEqual(result, ev.evaluate_predictions(dict(reversed(list(predictions.items()))), truth, cohort, reversed(range(1, 100))))

    def test_all_empty_diagnostic_is_null(self):
        result = ev.evaluate_predictions({"1": [1, 2, 3, 4, 5]}, {"1": []}, ["1"], range(1, 6))
        self.assertEqual(result["primary_map_at_5"], 0)
        self.assertIsNone(result["diagnostic_map_at_5_nonempty"])

    def test_missing_and_extra_users_rejected(self):
        for predictions in ({}, {"2": [1, 2, 3, 4, 5]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                ev.evaluate_predictions(predictions, {"1": [1]}, ["1"], range(1, 6))
        result = ev.evaluate_predictions({}, {"1": [1]}, ["1"], range(1, 6), strict=False)
        self.assertEqual(result["primary_map_at_5"], 0)
        self.assertEqual(result["mode"], "lenient_math_diagnostic")

    def test_malformed_records_and_duplicate_user_rows(self):
        valid = {"user_id": "000123", "tv_show_id": "1 2 3 4 5"}
        self.assertEqual(list(ev.parse_prediction_records([valid])), ["000123"])
        for records in ([valid, valid], [{**valid, "user_id": 123}], [{**valid, "extra": 0}],
                        [{**valid, "tv_show_id": "1  2 3 4 5"}], [{**valid, "tv_show_id": "1 1 3 4 5"}],
                        [{**valid, "tv_show_id": [1, 2, 3, 4, 5]}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                ev.parse_prediction_records(records)

    def test_user_cohort_and_truth_contract(self):
        for cohort in ([], ["1", "1"], [1], {"1"}):
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                ev.ordered_cohort(cohort)
        with self.assertRaises(ValueError):
            ev.evaluate_predictions({"1": [1, 2, 3, 4, 5]}, {}, ["1"], range(1, 6))
        with self.assertRaises(ValueError):
            ev.evaluate_predictions({"1": [1, 2, 3, 4, 5]}, {"1": "12"}, ["1"], range(1, 6))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="step3_test_", dir=ROOT / "tests")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, header, rows):
        path = self.directory / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_candidate_nonzero_distinct_sorted(self):
        path = self.write("c.csv", ["tv_show_id"], [[i] for i in range(1, 6)])
        self.assertEqual(val.load_candidates(path), ("1", "2", "3", "4", "5"))
        for values in ([0, 1, 2, 3, 4], [1, 1, 2, 3, 4], [5, 4, 3, 2, 1]):
            bad = self.write("bad.csv", ["tv_show_id"], [[i] for i in values])
            with self.subTest(values=values), self.assertRaises(ValueError):
                val.load_candidates(bad)

    def test_candidates_use_airing_overlap_not_viewing(self):
        schedule = [(0, 10, "1"), (10, 20, "2"), (20, 30, "3"), (0, 30, "0"), (15, 15, "4")]
        self.assertEqual(val.candidates_from_airings(schedule, (10, 20)), ("2",))
        self.assertEqual(val.candidates_from_airings(reversed(schedule), (10, 20)), ("2",))

    def test_raw_schedule_leading_zero_ids_match_step2_numeric_identity(self):
        self.assertEqual(val.schedule_program_id("00052641"), "52641")
        self.assertEqual(val.schedule_program_id("0000"), "0")
        self.assertEqual(val.schedule_program_id("09007199254740993"), "9007199254740993")
        with self.assertRaises(ValueError):
            ev.program_id("00052641")  # Prediction/processed tokens stay canonical.
        row = ["1", "channel", "18.05.2020 00:00:00", "300", "title", "00052641", "category", "", "", "", "", "", ""]
        path = self.write("schedule.csv", val.SCHEDULE_COLUMNS, [row, [*row[:5], "0000", *row[6:]]])
        pools = val.schedule_catalogs([path])
        self.assertEqual(pools["validation"], ("52641",))
        self.assertEqual(pools["train"], ())

    def test_interaction_boundaries_including_exclusive_stop(self):
        lo, hi = val.WINDOWS["train"]
        valid = self.write("valid.csv.gz", val.INTERACTION_COLUMNS, [["001", "1", 1, 120, hi - 120, hi]])
        self.assertEqual(next(val.iter_interactions(valid, (lo, hi), ["1"])).user_id, "001")
        for first, last in ((lo - 1, lo + 120), (hi - 119, hi + 1), (hi, hi + 120)):
            path = self.write("invalid.csv.gz", val.INTERACTION_COLUMNS, [["1", "1", 1, 120, first, last]])
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "Temporal leakage"):
                list(val.iter_interactions(path, (lo, hi), ["1"]))
        next_period = self.write("val.csv.gz", val.INTERACTION_COLUMNS, [["1", "1", 1, 120, hi, hi + 120]])
        self.assertEqual(len(list(val.iter_interactions(next_period, val.WINDOWS["validation"], ["1"]))), 1)

    def test_duplicate_pair_ineligible_id_and_bad_threshold_counts(self):
        lo = val.WINDOWS["train"][0]
        row = ["1", "1", 1, 120, lo, lo + 120]
        for rows, pool in (([row, row], ["1"]), ([row], ["2"]), ([["1", "1", 2, 239, lo, lo + 239]], ["1"])):
            path = self.write("bad.csv.gz", val.INTERACTION_COLUMNS, rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                list(val.iter_interactions(path, val.WINDOWS["train"], pool))

    def test_truth_uses_all_distinct_programs_and_preserves_empty_users(self):
        cohort = ("001", "18446744073709551615", "3")
        truth = {cohort[0]: {str(i) for i in range(1, 9)}, cohort[1]: {"9"}, cohort[2]: set()}
        path = self.directory / "truth.csv"
        val.save_truth(path, cohort, truth)
        loaded = val.load_truth(path, cohort, list(map(str, range(1, 10))))
        self.assertEqual(loaded, truth)
        self.assertEqual(len(loaded["001"]), 8)
        self.assertEqual(loaded["3"], set())
        first_hash = val.sha256(path)
        val.save_truth(path, cohort, truth)
        self.assertEqual(val.sha256(path), first_hash)

    def test_malformed_truth_not_silently_dropped(self):
        for labels, count in (("0", 1), ("1 1", 2), ("9", 1), ("1", 2)):
            path = self.write("truth.csv", val.TRUTH_COLUMNS, [["001", labels, count]])
            with self.subTest(labels=labels, count=count), self.assertRaises(ValueError):
                val.load_truth(path, ("001",), ["1", "2"])

    def test_cohort_summary_retains_cold_and_empty(self):
        rows = [val.Interaction("001", "1", 2, 240, 1, 500), val.Interaction("001", "2", 1, 120, 600, 720),
                val.Interaction("2", "3", 1, 120, 1, 121)]
        stats, counts, truth = val.summarize(rows, ("001", "2", "3"), collect_truth=True)
        self.assertEqual((stats["template_users"], stats["template_empty"]), (2, 1))
        self.assertEqual(truth, {"001": {"1", "2"}, "2": {"3"}, "3": set()})
        self.assertEqual(counts["001"], 2)  # Repeated airings do not duplicate relevance.

    def test_source_integrity_rejects_tampering(self):
        path = self.write("candidate.csv", ["tv_show_id"], [[i] for i in range(1, 6)])
        expected = val.sha256(path)
        with path.open("a") as stream:
            stream.write("6\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            val.load_candidates(path, expected)

    def test_config_checksum_and_policy_lock(self):
        config = {"policies": val.POLICY, "windows_utc": val.WINDOWS, "cohort_ids": ["001"], "cohort_order_sha256": ev.identity_hash(["001"])}
        path = self.directory / "config.json"
        val.json_write(path, config)
        path.with_suffix(".sha256").write_text(val.sha256(path))
        self.assertEqual(val.load_config(path)["cohort_ids"], ["001"])
        config["cohort_ids"] = ["1"]
        val.json_write(path, config)
        with self.assertRaisesRegex(ValueError, "checksum"):
            val.load_config(path)
        path.with_suffix(".sha256").write_text(val.sha256(path))
        with self.assertRaisesRegex(ValueError, "Cohort"):
            val.load_config(path)


class FallbackInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.pool = tuple(map(str, range(1, 7)))
        self.fallback = ev.FallbackRanking(("5", "4", "3", "2", "1", "6"), "a" * 64,
                                           val.WINDOWS["train"], ev.identity_hash(self.pool))

    def complete(self, ranked, fallback=None):
        return ev.complete_top5(ranked, fallback or self.fallback, self.pool,
                               training_sha256="a" * 64, training_window_utc=val.WINDOWS["train"])

    def test_cold_users_get_same_supplied_fallback(self):
        self.assertEqual(self.complete([]), ("5", "4", "3", "2", "1"))
        self.assertEqual(self.complete([]), self.complete([]))

    def test_repeat_allowed_and_deduplicated_completion(self):
        self.assertEqual(self.complete(["1", "1", "999"]), ("1", "5", "4", "3", "2"))

    def test_validation_or_wrong_candidate_provenance_rejected(self):
        for fallback in (replace(self.fallback, training_window_utc=val.WINDOWS["validation"]),
                         replace(self.fallback, training_sha256="b" * 64),
                         replace(self.fallback, candidate_ids_sha256="x"),
                         replace(self.fallback, program_ids=("1",) * 6)):
            with self.subTest(fallback=fallback), self.assertRaises(ValueError):
                self.complete([], fallback)


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
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (MetricTests, ValidationTests, FallbackInterfaceTests))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    runs = ROOT / "results" / "step3_test_runs"
    runs.mkdir(exist_ok=True)
    number = len(list(runs.glob("run_*.json"))) + 1
    val.json_write(runs / f"run_{number:03d}.json", {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "all_passed": result.wasSuccessful(), "checks": result.records,
        "seconds": round(time.monotonic() - started, 3), "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "seed": 42, "independent_reference_cases": 500,
        "source_sha256": {name: val.sha256(ROOT / "code" / name) for name in ("evaluation.py", "validation.py", "test_step3.py")},
        "hand_computed_examples": {"A": {"fraction": "34/45", "actual": ev.average_precision_at_5([10, 20, 30], [10, 99, 20, 88, 30])},
                                   "B": {"fraction": "9/20", "actual": ev.average_precision_at_5([1, 2], [3, 1, 4, 5, 2])}},
        "scores_are_synthetic_tests_not_model_performance": True})
    raise SystemExit(0 if result.wasSuccessful() else 1)
