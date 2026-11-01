"""Synthetic independent-reference tests for Step 4 baselines and output contract."""

from collections import Counter, defaultdict
from dataclasses import replace
import json
import resource
from types import SimpleNamespace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import evaluation
import personal_frequency
import popularity
import validation


def brute_popularity(rows, candidates):
    users, airs, seconds = defaultdict(set), Counter(), Counter()
    for row in rows:
        users[row.program_id].add(row.user_id)
        airs[row.program_id] += row.qualifying_airings
        seconds[row.program_id] += row.watched_seconds
    distinct = {p: len(users[p]) for p in users}
    result = {}
    for name, counts in (("distinct_users", distinct), ("qualifying_airings", airs), ("watched_seconds", seconds)):
        result[name] = tuple(sorted(candidates, key=lambda p: (-counts.get(p, 0), -distinct.get(p, 0),
                                                               -airs.get(p, 0), int(p))))
    return result, distinct, airs, seconds


class PopularityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [SimpleNamespace(user_id="0001", program_id="10", qualifying_airings=2, watched_seconds=300),
                     SimpleNamespace(user_id="0001", program_id="20", qualifying_airings=1, watched_seconds=120),
                     SimpleNamespace(user_id="2", program_id="10", qualifying_airings=1, watched_seconds=140),
                     SimpleNamespace(user_id="2", program_id="30", qualifying_airings=4, watched_seconds=900),
                     SimpleNamespace(user_id="3", program_id="30", qualifying_airings=1, watched_seconds=180),
                     SimpleNamespace(user_id="3", program_id="99", qualifying_airings=9, watched_seconds=9999)]
        self.pool = ("10", "20", "30", "40", "50")
        self.source = SimpleNamespace(rows=lambda: iter(self.rows), sha256="a" * 64,
                                      window_utc=validation.WINDOWS["train"])

    def fit(self):
        with patch.object(popularity, "training_input", return_value=self.source):
            return popularity.fit_popularity({"input": "training-only"}, self.pool)

    def test_all_popularity_counts_match_independent_reference(self):
        model = self.fit()
        reference, distinct, airs, seconds = brute_popularity(self.rows, self.pool)
        self.assertEqual(model.distinct_users, distinct)
        self.assertEqual(model.qualifying_airings, dict(airs))
        self.assertEqual(model.watched_seconds, dict(seconds))
        for variant in popularity.VARIANTS:
            self.assertEqual(model.ranking(variant), reference[variant])

    def test_candidate_restriction_and_zero_support_ties(self):
        model = self.fit()
        self.assertEqual(model.ranking("distinct_users")[-2:], ("40", "50"))
        self.assertNotIn("99", model.ranking("distinct_users"))
        self.assertTrue(all(p in self.pool and p != "0" for variant in popularity.VARIANTS for p in model.ranking(variant)))

    def test_global_top5_same_for_all_original_string_ids(self):
        model = self.fit()
        cohort = ("0001", "18446744073709551615", "003")
        result = popularity.predictions(model, "distinct_users", cohort)
        self.assertEqual(tuple(result), cohort)
        self.assertTrue(all(result[u] == model.ranking("distinct_users")[:5] for u in cohort))
        self.assertTrue(all(len(v) == 5 and len(set(v)) == 5 for v in result.values()))

    def test_deterministic_ties_and_training_provenance(self):
        first, second = self.fit(), self.fit()
        self.assertEqual(first.ranking("qualifying_airings"), second.ranking("qualifying_airings"))
        self.assertEqual(first.training_sha256, "a" * 64)
        self.assertEqual(first.training_window_utc, validation.WINDOWS["train"])
        with self.assertRaises(ValueError):
            first.ranking("validation_labels")


class PersonalFrequencyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [SimpleNamespace(user_id="0001", program_id="10", qualifying_airings=2, watched_seconds=300),
                     SimpleNamespace(user_id="0001", program_id="20", qualifying_airings=1, watched_seconds=400),
                     SimpleNamespace(user_id="0001", program_id="99", qualifying_airings=8, watched_seconds=9000),
                     SimpleNamespace(user_id="2", program_id="30", qualifying_airings=2, watched_seconds=500)]
        self.pool = ("10", "20", "30", "40", "50")
        self.source = SimpleNamespace(rows=lambda: iter(self.rows), sha256="b" * 64,
                                      window_utc=validation.WINDOWS["train"])
        with patch.object(personal_frequency, "training_input", return_value=self.source):
            self.model = personal_frequency.fit_personal_frequency({"input": "training-only"}, self.pool)
        self.pop = popularity.PopularityModel(self.pool, {"10": 2, "20": 1, "30": 1},
                                              {"10": 2, "20": 1, "30": 2},
                                              {"10": 440, "20": 400, "30": 500}, "b" * 64,
                                              validation.WINDOWS["train"], 4, .001)

    def test_frequency_values_match_manual_baskets(self):
        self.assertEqual(self.model.user_airings["0001"], {"10": 2, "20": 1, "99": 8})
        self.assertEqual(self.model.user_seconds["0001"], {"10": 300, "20": 400, "99": 9000})

    def test_event_frequency_and_duration_have_distinct_rankings(self):
        fallback = self.pop.ranking("distinct_users")
        event = self.model.user_ranking("0001", "qualifying_airings", fallback)
        duration = self.model.user_ranking("0001", "watched_seconds", fallback)
        self.assertEqual(event[:2], ("10", "20"))
        self.assertEqual(duration[:2], ("20", "10"))

    def test_unavailable_history_program_excluded_and_popularity_fills(self):
        result = self.model.user_ranking("0001", "qualifying_airings", self.pop.ranking("distinct_users"))
        self.assertNotIn("99", result[:5])
        self.assertEqual(result[:5], ("10", "20", "30", "40", "50"))

    def test_cold_and_empty_eligible_history_users_receive_same_fallback(self):
        fallback = self.pop.ranking("distinct_users")
        expected = fallback[:5]
        for user in ("new", "history_only_outside_pool"):
            self.assertEqual(self.model.user_ranking(user, "qualifying_airings", fallback)[:5], expected)

    def test_repeat_allow_and_source_mismatch_rejection(self):
        fallback = self.pop.ranking("distinct_users")
        self.assertIn("10", self.model.user_ranking("0001", "qualifying_airings", fallback))
        other = replace(self.pop, training_sha256="c" * 64)
        with self.assertRaises(ValueError):
            personal_frequency.predict(self.model, "qualifying_airings", ("0001",), other)
        other = replace(self.pop, candidates=tuple(reversed(self.pool)))
        with self.assertRaises(ValueError):
            personal_frequency.predict(self.model, "qualifying_airings", ("0001",), other)


class EvaluationIntegrationTests(unittest.TestCase):
    def test_step3_evaluator_is_called_with_same_protocol(self):
        cohort = ("001", "2")
        pool = ("10", "20", "30", "40", "50")
        truth = {"001": {"10"}, "2": set()}
        predictions = {"001": ("10", "20", "30", "40", "50"), "2": ("10", "20", "30", "40", "50")}
        result = evaluation.evaluate_predictions(predictions, truth, cohort, pool, strict=True)
        self.assertEqual(result["primary_map_at_5"], .5)
        self.assertEqual(result["diagnostic_map_at_5_nonempty"], 1)
        self.assertEqual(result["per_user_ap_at_5"], {"001": 1, "2": 0})


if __name__ == "__main__":
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = validation.ROOT
    runs = root / "results" / "step4_test_runs"
    runs.mkdir(exist_ok=True)
    attempts = sorted(runs.glob("step4_test_run_*.json"))
    number = len(attempts) + 1
    record = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "all_passed": result.wasSuccessful(), "seconds": round(time.monotonic() - started, 6),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "seed": None, "synthetic_reference": "independent explicit sample aggregates and expected order",
              "source_sha256": {name: validation.sha256(root / "code" / name)
                                for name in ("evaluation.py", "validation.py", "popularity.py", "personal_frequency.py", "test_step4.py")}}
    (runs / f"step4_test_run_{number:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
