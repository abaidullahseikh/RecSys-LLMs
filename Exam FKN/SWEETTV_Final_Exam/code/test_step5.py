"""Independent toy-graph and PPR contract checks (never synthetic model results)."""

from collections import Counter
from types import SimpleNamespace
import math
import json
import unittest
from unittest.mock import patch
import time
import resource
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import evaluation
import ppr
import validation
from unittest.mock import patch


def independent_reference(matrix, restart, alpha, tol=1e-13, max_iter=10000):
    """Plain Python dense implementation, independent of SciPy sparse matvec."""
    n = len(restart)
    r = list(restart)
    rows = [[float(matrix[i, j]) for j in range(n)] for i in range(n)]
    for iteration in range(1, max_iter + 1):
        dangling = sum(r[i] for i in range(n) if sum(rows[i]) == 0)
        nxt = []
        for j in range(n):
            incoming = sum(rows[i][j] * r[i] for i in range(n))
            nxt.append(alpha * restart[j] + (1-alpha) * (incoming + dangling * restart[j]))
        delta = sum(abs(a-b) for a,b in zip(nxt,r))
        r = nxt
        if delta <= tol:
            return r, iteration, delta
    return r, max_iter, delta


def graph_from_dense(nodes, values):
    matrix = csr_matrix(np.asarray(values, dtype=np.float64))
    return ppr.ProgramGraph(tuple(nodes), matrix, {}, {}, "toy", 10, 10, 1, 0, 0, 0.0)


class GraphConstructionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = ("10", "20", "30")
        self.support = {"10": 2, "20": 3, "30": 2}
        self.pairs = {(0, 1): 2, (0, 2): 1}

    def test_graph_nodes_follow_valid_training_programs(self):
        graph = ppr.build_graph_from_pair_counts(self.nodes, self.support, self.pairs,
                                                  min_common_viewers=1, max_neighbors=5)
        self.assertEqual(graph.nodes, self.nodes)
        self.assertEqual(graph.matrix.shape, (3, 3))

    def test_common_viewer_counts_and_edges(self):
        graph = ppr.build_graph_from_pair_counts(self.nodes, self.support, self.pairs,
                                                  weight_type="common_viewers", min_common_viewers=1,
                                                  max_neighbors=5)
        self.assertEqual(graph.raw_pair_counts[(0, 1)], 2)
        self.assertEqual(graph.raw_pair_counts[(0, 2)], 1)
        self.assertEqual(graph.matrix.nnz, 4)
        self.assertEqual(graph.retained_edge_weights[(0, 1)], 2)

    def test_cosine_weights_match_hand_calculation(self):
        graph = ppr.build_graph_from_pair_counts(self.nodes, self.support, self.pairs,
                                                  weight_type="cosine", min_common_viewers=1,
                                                  max_neighbors=5)
        self.assertAlmostEqual(graph.retained_edge_weights[(0, 1)], 2 / math.sqrt(6), places=15)
        self.assertAlmostEqual(graph.retained_edge_weights[(0, 2)], 1 / 2, places=15)

    def test_minimum_common_viewers_and_neighbor_cap(self):
        graph = ppr.build_graph_from_pair_counts(self.nodes, self.support, self.pairs,
                                                  min_common_viewers=2, max_neighbors=1)
        self.assertNotIn((0, 2), graph.retained_edge_weights)
        self.assertEqual(graph.matrix.nnz, 2)

    def test_symmetrization_and_transition_row_normalization(self):
        graph = ppr.build_graph_from_pair_counts(self.nodes, self.support, self.pairs,
                                                  min_common_viewers=1, max_neighbors=5)
        self.assertGreater(graph.matrix[0, 1], 0)
        self.assertGreater(graph.matrix[1, 0], 0)
        self.assertEqual(graph.retained_edge_weights[(0, 1)], 2.0)
        sums = np.asarray(graph.matrix.sum(axis=1)).ravel()
        self.assertTrue(np.allclose(sums[:2], 1.0))
        self.assertEqual(sums[2], 1.0)  # Node 30 has an edge to 10.
        self.assertTrue(graph.matrix.has_sorted_indices)

    def test_training_cap_pair_counts_and_truncation(self):
        profiles = {"u": {"1": (4, 500, 100), "2": (3, 400, 100), "3": (2, 300, 100)},
                    "v": {"1": (1, 100, 100), "2": (2, 200, 100)}}
        data = ppr.TrainingProfiles(tuple("123"), {x:i for i,x in enumerate("123")}, profiles,
                                    {"1":2,"2":2,"3":1}, "a"*64, validation.WINDOWS["train"], 5, 0.0)
        counts, occurrences, truncated = ppr.build_copair_counts(data, history_cap=2)
        self.assertEqual(counts, {(0,1): 2})
        self.assertEqual(occurrences, 2)
        self.assertEqual(truncated, 1)

    def test_training_profile_loader_uses_only_guarded_training_source(self):
        rows = [SimpleNamespace(user_id="0001", program_id="10", qualifying_airings=2, watched_seconds=240, last_watch_utc=1589000000),
                SimpleNamespace(user_id="0001", program_id="20", qualifying_airings=1, watched_seconds=120, last_watch_utc=1589500000),
                SimpleNamespace(user_id="2", program_id="10", qualifying_airings=1, watched_seconds=121, last_watch_utc=1589600000)]
        source = SimpleNamespace(rows=lambda: iter(rows), sha256="c"*64, window_utc=validation.WINDOWS["train"])
        with patch.object(ppr, "training_input", return_value=source) as opened:
            data = ppr.load_training_profiles({"role":"training_only"})
        opened.assert_called_once()
        self.assertEqual(data.rows, 3)
        self.assertEqual(data.distinct_viewers, {"10": 2, "20": 1})
        self.assertEqual(data.profiles["0001"]["10"], (2,240,1589000000))
        self.assertEqual(data.training_window_utc, validation.WINDOWS["train"])


class PersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.data = ppr.TrainingProfiles(("10","20","30"), {"10":0,"20":1,"30":2},
            {"u": {"10":(2,100,1587000000), "20":(4,900,1589700000), "30":(1,300,1589000000)}},
            {"10":2,"20":3,"30":2}, "d"*64, validation.WINDOWS["train"], 3, 0)

    def test_uniform_frequency_duration_recency_personalization(self):
        profile = self.data.profiles["u"]
        u = ppr.personalization_vector(profile, self.data, variant="uniform")
        f = ppr.personalization_vector(profile, self.data, variant="frequency")
        d = ppr.personalization_vector(profile, self.data, variant="duration")
        r = ppr.personalization_vector(profile, self.data, variant="recency", half_life_seconds=86400)
        self.assertAlmostEqual(sum(u.values()), 1)
        self.assertAlmostEqual(u["10"], 1/3)
        self.assertAlmostEqual(f["20"], 4/7)
        self.assertAlmostEqual(d["20"], 900/1300)
        self.assertGreater(r["20"], r["30"])
        self.assertGreater(r["30"], r["10"])
        for variant in ppr.PERSONALIZATIONS:
            self.assertTrue(all(v > 0 for v in ppr.personalization_vector(profile, self.data, variant=variant).values()))

    def test_sparse_empty_profile_is_empty(self):
        self.assertEqual(ppr.personalization_vector({}, self.data, variant="uniform"), {})


class PageRankTests(unittest.TestCase):
    def test_small_independent_pagerank_calculation(self):
        # Node 0 -> node 1; node 1 -> node 0; node 2 is dangling.
        graph = graph_from_dense(("10","20","30"), [[0,1,0],[1,0,0],[0,0,0]])
        p = {"10":1.0,"20":0.0,"30":0.0}
        rank, info = ppr.personalized_pagerank(graph, p, restart_probability=.2, tolerance=1e-14,
                                               max_iterations=500)
        self.assertAlmostEqual(rank[0], 5/9, places=12)
        self.assertAlmostEqual(rank[1], 4/9, places=12)
        self.assertAlmostEqual(rank[2], 0, places=12)
        expected, iterations, residual = independent_reference(graph.matrix.toarray(), [1,0,0], .2)
        self.assertGreaterEqual(info["iterations"], iterations)
        self.assertTrue(np.allclose(rank, expected, rtol=0, atol=1e-13))
        self.assertLessEqual(info["l1_residual"], 1e-14)
        self.assertTrue(info["converged"])

    def test_dangling_mass_returns_to_personalization(self):
        graph = graph_from_dense(("1","2","3"), [[0,1,0],[1,0,0],[0,0,0]])
        rank, info = ppr.personalized_pagerank(graph, {"3":1}, restart_probability=.2)
        self.assertEqual(tuple(rank), (0.0,0.0,1.0))
        self.assertTrue(info["converged"])

    def test_graph_power_iteration_agrees_with_independent_dense_reference(self):
        graph = graph_from_dense(("1","2","3"), [[0,.5,.5],[1,0,0],[0,0,0]])
        restart = {"1":.5,"3":.5}
        actual, stats = ppr.personalized_pagerank(graph, restart, restart_probability=.35, tolerance=1e-12)
        expected, _, _ = independent_reference(graph.matrix.toarray(), [.5,0,.5], .35, tol=1e-13)
        self.assertTrue(np.allclose(actual, expected, rtol=0, atol=1e-11))
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=12)
        self.assertTrue(stats["converged"])

    def test_max_iteration_nonconvergence_is_reported(self):
        graph = graph_from_dense(("1","2"), [[0,1],[1,0]])
        _, stats = ppr.personalized_pagerank(graph, {"1":1}, restart_probability=.1,
                                             tolerance=1e-30, max_iterations=1)
        self.assertEqual(stats["iterations"], 1)
        self.assertFalse(stats["converged"])

    def test_deterministic_sparse_top5_candidate_filter_and_fallback(self):
        graph = graph_from_dense(("10","20","30"), [[0,1,0],[1,0,0],[0,0,0]])
        data = ppr.TrainingProfiles(("10","20","30"), {"10":0,"20":1,"30":2},
            {"u":{"10":(1,120,100),"20":(1,120,110)}}, {"10":1,"20":1,"30":1},
            "a"*64, validation.WINDOWS["train"], 2, 0)
        candidates = ("20","30","40","50","60","70")
        fallback = ("30","40","50","60","70","20")
        first, info = ppr.predict_user(graph, data, "u", candidates, fallback,
                                       personalization="uniform", restart_probability=.2)
        second, _ = ppr.predict_user(graph, data, "u", candidates, fallback,
                                     personalization="uniform", restart_probability=.2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(set(first).issubset(candidates))
        self.assertNotIn("0", first)
        self.assertGreaterEqual(info["fallback_slots"], 0)

    def test_full_ppr_top5_does_not_append_extra_fallback_items(self):
        nodes = tuple(str(i) for i in range(1, 6))
        graph = graph_from_dense(nodes, [[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]])
        profile = {item:(1,120,100) for item in nodes}
        data = ppr.TrainingProfiles(nodes, {item:i for i,item in enumerate(nodes)}, {"u":profile},
                                    {item:1 for item in nodes}, "a"*64, validation.WINDOWS["train"], 5, 0)
        result, info = ppr.predict_user(graph, data, "u", nodes, tuple(reversed(nodes)),
                                        personalization="uniform", restart_probability=.2)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(set(result)), 5)
        self.assertEqual(info["fallback_slots"], 0)

    def test_cold_start_returns_shared_fallback_exactly(self):
        graph = graph_from_dense(("10","20"), [[0,1],[1,0]])
        data = ppr.TrainingProfiles(("10","20"), {"10":0,"20":1}, {}, {"10":1,"20":1},
                                    "a"*64, validation.WINDOWS["train"], 0, 0)
        fallback = ("20","10","30","40","50")
        ranked, info = ppr.predict_user(graph, data, "cold", ("10","20","30","40","50"), fallback)
        self.assertEqual(ranked, fallback[:5])
        self.assertTrue(info["cold_start"])
        self.assertEqual(info["fallback_slots"], 5)

    def test_cold_start_rejects_bad_fallback_inventory(self):
        graph = graph_from_dense(("1","2"), [[0,1],[1,0]])
        data = ppr.TrainingProfiles(("1","2"), {"1":0,"2":1}, {}, {"1":1,"2":1},
                                    "a"*64, validation.WINDOWS["train"], 0, 0)
        for fallback in (("1","1","2","3","4"), ("0","1","2","3","4"), ("1","2")):
            with self.subTest(fallback=fallback), self.assertRaises(ValueError):
                ppr.predict_user(graph, data, "cold", ("1","2","3","4","5"), fallback)

    def test_common_viewer_edge_and_cosine_by_hand(self):
        counts = {(0,1):3,(0,2):2,(1,2):1}
        support = {"1":4,"2":9,"3":4}
        g1 = ppr.build_graph_from_pair_counts(("1","2","3"), support, counts, weight_type="common_viewers", min_common_viewers=1, max_neighbors=10)
        g2 = ppr.build_graph_from_pair_counts(("1","2","3"), support, counts, weight_type="cosine", min_common_viewers=1, max_neighbors=10)
        self.assertEqual(g1.retained_edge_weights[(0,1)], 3)
        self.assertAlmostEqual(g2.retained_edge_weights[(0,1)], 3/6)
        self.assertAlmostEqual(g2.retained_edge_weights[(1,2)], 1/6)

    def test_candidate_and_program_id_zero_rejected(self):
        graph = graph_from_dense(("1","2"), [[0,1],[1,0]])
        data = ppr.TrainingProfiles(("1","2"), {"1":0,"2":1}, {"u":{"1":(1,120,10)}}, {"1":1}, "a"*64, validation.WINDOWS["train"], 1, 0)
        with self.assertRaises(ValueError):
            ppr.predict_user(graph, data, "u", ("0","1","2","3","4"), ("1","2","3","4","5"))


class LeakageAndFallbackTests(unittest.TestCase):
    def test_loader_requests_training_handle_only(self):
        rows = [SimpleNamespace(user_id="u", program_id="1", qualifying_airings=2, watched_seconds=240, last_watch_utc=1589000000)]
        source = SimpleNamespace(rows=lambda:iter(rows), sha256="a"*64, window_utc=validation.WINDOWS["train"])
        with patch.object(ppr, "training_input", return_value=source) as accessor:
            data = ppr.build_training_profiles({"role":"training_only"})
        accessor.assert_called_once()
        self.assertEqual(data.rows, 1)
        self.assertEqual(data.profiles["u"]["1"], (2,240,1589000000))
        self.assertEqual(data.training_sha256, "a"*64)

    def test_validation_history_or_future_rows_cannot_enter_training_loader(self):
        with patch.object(ppr, "training_input", side_effect=ValueError("Validation/history/test forbidden")):
            with self.assertRaisesRegex(ValueError, "Validation/history/test"):
                ppr.build_training_profiles({"role":"validation"})

    def test_top5_and_map_call_existing_step3_evaluator(self):
        cohort = ("001", "002")
        candidates = ("1","2","3","4","5","6")
        truth = {"001": {"1","2"}, "002": set()}
        predictions = {"001": ("1","3","2","4","5"), "002": ("1","2","3","4","5")}
        result = evaluation.evaluate_predictions(predictions, truth, cohort, candidates, strict=True)
        self.assertAlmostEqual(result["per_user_ap_at_5"]["001"], (1 + 2/3) / 2)
        self.assertEqual(result["per_user_ap_at_5"]["002"], 0)
        self.assertAlmostEqual(result["primary_map_at_5"], ((1 + 2/3) / 2) / 2)


if __name__ == "__main__":
    class Result(unittest.TextTestResult):
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

    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    root = validation.ROOT
    out = root / "results" / "step5_test_runs"
    out.mkdir(parents=True, exist_ok=True)
    number = len(list(out.glob("run_*.json"))) + 1
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "all_passed": result.wasSuccessful(), "checks": result.records,
               "seconds": round(time.monotonic() - started, 6),
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "numpy_version": np.__version__, "scipy_version": __import__("scipy").__version__,
               "independent_reference": "dense pure-Python power iteration and hand arithmetic",
               "source_sha256": {name: validation.sha256(root / "code" / name)
                                 for name in ("ppr.py", "test_step5.py", "validation.py", "evaluation.py")}}
    (out / f"run_{number:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
