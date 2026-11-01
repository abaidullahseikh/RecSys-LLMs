"""Focused deterministic checks for the Step 8 rank-fusion/diagnostic helpers."""

import unittest

import run_step8 as s8


class Step8RankingTests(unittest.TestCase):
    def test_equal_rankings_preserve_order_under_reciprocal_rank_fusion(self):
        users=("u",);candidates=("1","2","3","4","5")
        ranking={"u":candidates};fallback=candidates
        positions=s8.rank_position_matrix(ranking,users,candidates)
        fused=s8.rrf_from_positions(positions,positions,users,candidates,fallback,first_weight=.5)
        self.assertEqual(fused["u"],candidates)

    def test_reciprocal_rank_fusion_combines_personal_and_graph_order(self):
        users=("u",);candidates=("1","2","3","4","5")
        ppr={"u":("1","2","3","4","5")}
        personal={"u":("2","3","4","5","1")}
        a=s8.rank_position_matrix(ppr,users,candidates)
        b=s8.rank_position_matrix(personal,users,candidates)
        result=s8.rrf_from_positions(a,b,users,candidates,candidates,first_weight=.5)
        self.assertEqual(result["u"],("2","3","1","4","5"))

    def test_reciprocal_rank_ties_use_shared_popularity_order(self):
        users=("u",);candidates=("1","2","3","4","5")
        ppr={"u":("1","2","3","4","5")}
        personal={"u":("2","1","3","4","5")}
        fallback=("2","1","3","4","5")
        a=s8.rank_position_matrix(ppr,users,candidates)
        b=s8.rank_position_matrix(personal,users,candidates)
        result=s8.rrf_from_positions(a,b,users,candidates,fallback,first_weight=.5)
        self.assertEqual(result["u"][:2],("2","1"))

    def test_pairwise_diversity_counts_near_identical_lists(self):
        users=("a","b","c")
        predictions={"a":("1","2","3","4","5"),
                     "b":("1","2","3","4","6"),
                     "c":("8","9","10","11","12")}
        result=s8.pairwise_diversity(predictions,users)
        self.assertEqual(result["unique_recommended_programs"],11)
        self.assertAlmostEqual(result["mean_pairwise_top5_item_overlap"],4/3)
        self.assertAlmostEqual(result["fraction_user_pairs_sharing_at_least_4_items"],1/3)
        self.assertAlmostEqual(result["fraction_user_pairs_sharing_at_least_3_items"],1/3)
        self.assertEqual(result["fraction_user_pairs_identical_ranked_list"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)
