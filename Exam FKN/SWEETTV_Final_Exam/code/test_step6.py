"""Independent hand-calculated Association Rule and leakage contract tests."""

from collections import Counter
from types import SimpleNamespace
from fractions import Fraction
import unittest
from unittest.mock import patch
import time
import resource
import json
import math
from pathlib import Path

import association_rules as ar
import evaluation
import validation


class TransactionTests(unittest.TestCase):
    def test_history_transactions_distinct_user_item_sets(self):
        rows=[SimpleNamespace(user_id="0001",program_id="10"),SimpleNamespace(user_id="0001",program_id="10"),
              SimpleNamespace(user_id="0001",program_id="20"),SimpleNamespace(user_id="2",program_id="30")]
        self.assertEqual(ar.user_history_transactions(rows),(("0001",("10","20")),("2",("30",))))

    def test_window_transaction_baskets_repeat_pair_support_per_window(self):
        lo,hi=1000,1000+14*86400
        rows=[{"user_id":"u","tv_show_id":"10","overlap_seconds":"120","first_overlap_utc":str(lo),"last_overlap_utc":str(lo+120)},
              {"user_id":"u","tv_show_id":"20","overlap_seconds":"240","first_overlap_utc":str(lo+5),"last_overlap_utc":str(lo+245)},
              {"user_id":"u","tv_show_id":"10","overlap_seconds":"160","first_overlap_utc":str(lo+7*86400),"last_overlap_utc":str(lo+7*86400+160)},
              {"user_id":"u","tv_show_id":"30","overlap_seconds":"300","first_overlap_utc":str(lo+7*86400),"last_overlap_utc":str(lo+7*86400+300)},
              {"user_id":"v","tv_show_id":"0","overlap_seconds":"1000","first_overlap_utc":str(lo+9),"last_overlap_utc":str(lo+1009)},
              {"user_id":"v","tv_show_id":"40","overlap_seconds":"119","first_overlap_utc":str(lo+9),"last_overlap_utc":str(lo+128)}]
        tx=ar.time_window_transactions_from_airings(rows,(lo,hi),7)
        self.assertEqual(tx,( ("u",0,("10","20")),("u",1,("10","30")) ))

    def test_window_boundary_fixed_and_exclusive(self):
        lo,hi=100,100+14*86400
        self.assertEqual(ar.window_index(lo,(lo,hi),7*86400),0)
        self.assertEqual(ar.window_index(lo+7*86400,(lo,hi),7*86400),1)
        with self.assertRaises(ValueError): ar.window_index(hi,(lo,hi),7*86400)
        with self.assertRaises(ValueError): ar.window_index(lo-1,(lo,hi),7*86400)

    def test_transactions_do_not_cross_validation_boundary(self):
        lo,hi=validation.WINDOWS["train"]
        rows=[{"user_id":"u","tv_show_id":"10","overlap_seconds":"120","first_overlap_utc":str(hi),"last_overlap_utc":str(hi+120)}]
        with self.assertRaises(ValueError): ar.time_window_transactions_from_airings(rows,(lo,hi),7)

    def test_transaction_statistics_and_pair_occurrences(self):
        tx=(("u",("1","2","3")),("v",("1","2")),("w",("2","3")),("x",("1",)))
        items,index,support,pairs,pair_occ,sizes,p95,elapsed=ar.mine_sparse_pair_counts(tx)
        self.assertEqual(items,("1","2","3"))
        self.assertEqual(support,{"1":3,"2":3,"3":2})
        self.assertEqual(pairs,{(0,1):2,(0,2):1,(1,2):2})
        self.assertEqual(pair_occ,5)
        self.assertEqual(sizes,[3,2,2,1])


class RuleMathTests(unittest.TestCase):
    def test_hand_calculated_support_confidence_and_lift(self):
        # Four baskets: {A,B}, {A,B}, {A,C}, {B,C}.
        support,confidence,lift=ar.association_metrics(2,3,3,4)
        self.assertEqual(Fraction(2,4),Fraction(support).limit_denominator())
        self.assertEqual(Fraction(2,3),Fraction(confidence).limit_denominator())
        self.assertEqual(Fraction(8,9),Fraction(lift).limit_denominator())
        support,confidence,lift=ar.association_metrics(1,3,2,4)
        self.assertEqual((Fraction(support).limit_denominator(),Fraction(confidence).limit_denominator(),Fraction(lift).limit_denominator()),
                         (Fraction(1,4),Fraction(1,3),Fraction(2,3)))

    def test_rule_mining_metrics_and_directionality(self):
        transactions=(("u1",("1","2")),("u2",("1","2")),("u3",("1","3")),("u4",("2","3")))
        dataset=ar.build_dataset(transactions,transaction_type="user_history",window_days=None,
            training_sha256="a"*64,training_window_utc=validation.WINDOWS["train"],source_sha256="a"*64)
        model=ar.mine_rules(dataset,min_support=.25,min_confidence=0,min_cooccurrence=1,min_lift=0,top_neighbors=20)
        ab=next(r for r in model.rules_by_antecedent["1"] if r["consequent"]=="2")
        ba=next(r for r in model.rules_by_antecedent["2"] if r["consequent"]=="1")
        self.assertAlmostEqual(ab["support"],.5)
        self.assertAlmostEqual(ab["confidence"],2/3)
        self.assertAlmostEqual(ab["lift"],8/9)
        self.assertAlmostEqual(ba["confidence"],2/3)
        self.assertEqual(model.rule_count,6)

    def test_min_filters_and_neighbor_truncation(self):
        tx=tuple((f"u{i}",("1","2","3")) for i in range(3))+(("u4",("1","2")),)
        dataset=ar.build_dataset(tx,transaction_type="user_history",window_days=None,
            training_sha256="a"*64,training_window_utc=validation.WINDOWS["train"],source_sha256="a"*64)
        model=ar.mine_rules(dataset,min_support=.5,min_confidence=.5,min_cooccurrence=3,min_lift=1,
                            top_neighbors=1)
        self.assertTrue(all(len(rules)<=1 for rules in model.rules_by_antecedent.values()))
        for a,rules in model.rules_by_antecedent.items():
            for r in rules:
                self.assertGreaterEqual(r["support"],.5)
                self.assertGreaterEqual(r["confidence"],.5)
                self.assertGreaterEqual(r["pair_count"],3)
                self.assertGreaterEqual(r["lift"],1)


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        tx=(("a",("1","2")),("b",("1","2")),("c",("1","3")),("d",("1","4")),
            ("e",("2","3")),("f",("2","4")),("g",("3","5")),("h",("4","5")))
        self.ds=ar.build_dataset(tx,transaction_type="user_history",window_days=None,
            training_sha256="b"*64,training_window_utc=validation.WINDOWS["train"],source_sha256="b"*64)
        self.model=ar.mine_rules(self.ds,min_support=.1,min_confidence=0,min_cooccurrence=1,min_lift=0,top_neighbors=10)
        self.pool=("1","2","3","4","5","6")
        self.fallback=("1","2","3","4","5","6")
        self.profile={"1":(2,300,100),"2":(1,120,100)}

    def test_score_aggregation_confidence_and_normalized_lift(self):
        conf,matched=ar.score_user(self.model,self.profile,self.pool,score_type="confidence",antecedent_weight="uniform")
        lift,_=ar.score_user(self.model,self.profile,self.pool,score_type="confidence_log_lift",antecedent_weight="uniform")
        self.assertTrue(matched)
        self.assertGreater(conf["2"],0)
        self.assertGreater(lift["2"],0)
        freq,_=ar.score_user(self.model,self.profile,self.pool,score_type="confidence",antecedent_weight="log_frequency")
        self.assertTrue(all(math.isfinite(v) and v>=0 for v in (*conf.values(),*lift.values(),*freq.values())))

    def test_exact_five_unique_positive_candidates_and_repeat_policy(self):
        ranking,info=ar.recommend_user(self.model,self.profile,self.pool,self.fallback)
        self.assertEqual(len(ranking),5)
        self.assertEqual(len(set(ranking)),5)
        self.assertTrue(set(ranking).issubset(self.pool))
        self.assertNotIn("0",ranking)
        # Previously watched items are not excluded if rules rank them.
        for item in self.profile:
            if item in self.pool and item in self.model.rules_by_antecedent:
                self.assertTrue(item in ranking or item not in self.model.rules_by_antecedent)

    def test_cold_and_sparse_users_use_shared_fallback(self):
        cold,diag=ar.recommend_user(self.model,{},self.pool,self.fallback)
        self.assertEqual(cold,self.fallback[:5])
        self.assertTrue(diag["cold_start"])
        sparse,_=ar.recommend_user(self.model,{"6":(1,120,100)},self.pool,self.fallback)
        self.assertEqual(len(sparse),5)

    def test_deterministic_scores_and_ties(self):
        first,_=ar.recommend_user(self.model,self.profile,self.pool,self.fallback,
                                  score_type="confidence_log_lift",antecedent_weight="log_frequency")
        second,_=ar.recommend_user(self.model,self.profile,self.pool,self.fallback,
                                   score_type="confidence_log_lift",antecedent_weight="log_frequency")
        self.assertEqual(first,second)

    def test_invalid_candidate_and_variant_rejected(self):
        with self.assertRaises(ValueError): ar.recommend_user(self.model,self.profile,("0","1","2","3","4"),self.fallback)
        with self.assertRaises(ValueError): ar.score_user(self.model,self.profile,self.pool,score_type="probability")

    def test_strict_map_uses_step3_evaluator(self):
        cohort=("0001","02")
        truth={"0001":{"1","2"},"02":set()}
        ranked={"0001":("1","3","2","4","5"),"02":("1","2","3","4","5")}
        result=evaluation.evaluate_predictions(ranked,truth,cohort,self.pool,strict=True)
        self.assertAlmostEqual(result["per_user_ap_at_5"]["0001"],(1+2/3)/2)
        self.assertEqual(result["per_user_ap_at_5"]["02"],0)
        self.assertAlmostEqual(result["primary_map_at_5"],(1+2/3)/4)


if __name__=="__main__":
    class Result(unittest.TextTestResult):
        def __init__(self,*a,**kw): super().__init__(*a,**kw); self.records=[]
        def addSuccess(self,t): super().addSuccess(t); self.records.append({"test":t.id(),"status":"passed"})
        def addFailure(self,t,e): super().addFailure(t,e); self.records.append({"test":t.id(),"status":"failed","traceback":self._exc_info_to_string(e,t)})
        def addError(self,t,e): super().addError(t,e); self.records.append({"test":t.id(),"status":"error","traceback":self._exc_info_to_string(e,t)})
        def addSubTest(self,t,s,e):
            super().addSubTest(t,s,e)
            if e:self.records.append({"test":s.id(),"status":"failed","traceback":self._exc_info_to_string(e,t)})
    start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    out=validation.ROOT/"results"/"step6_test_runs"; out.mkdir(exist_ok=True)
    n=1+len(list(out.glob("run_*.json")))
    payload={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "all_passed":result.wasSuccessful(),"checks":result.records,"seconds":round(time.monotonic()-start,6),
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "independent_examples":"4-basket hand-calculated support/confidence/lift and a separate pair-count reference",
        "source_sha256":{name:validation.sha256(validation.ROOT/"code"/name) for name in
            ("association_rules.py","run_step6.py","verify_step6.py","test_step6.py")
            if (validation.ROOT/"code"/name).is_file()}}
    (out/f"run_{n:03d}.json").write_text(json.dumps(payload,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
