"""Synthetic checks for sparse CBF feature fitting, profiles, and fallback."""

import unittest

import content_based as cbf
import evaluation
import run_cbf_step9 as step9


class ContentBasedTests(unittest.TestCase):
    def setUp(self):
        self.training={
            "1":{"category":{"CATEGORY__news"},"genre":{"GENRE__politics"},"title":{"TITLE__alpha"},"actors":{"ACTORS__alice"},"channel":{"CHANNEL__channel_a"}},
            "2":{"category":{"CATEGORY__news"},"genre":{"GENRE__politics"},"title":{"TITLE__beta"},"actors":{"ACTORS__alice"},"channel":{"CHANNEL__channel_b"}},
            "3":{"category":{"CATEGORY__sport"},"genre":{"GENRE__football"},"title":{"TITLE__gamma"},"actors":{"ACTORS__bob"},"channel":{"CHANNEL__channel_a"}},
        }
        self.candidates={**self.training,
            "4":{"category":{"CATEGORY__news"},"genre":{"GENRE__politics"},"title":{"TITLE__unseen","TITLE__alpha"},"actors":{"ACTORS__new_person"},"channel":{"CHANNEL__channel_c"}},
            "5":{"category":{"CATEGORY__documentary"},"genre":{"GENRE__science"},"title":{"TITLE__delta"},"actors":{"ACTORS__other"},"channel":{"CHANNEL__channel_d"}}}
        self.weights={"category":2,"genre":1.5,"title":1,"actors":.5,"channel":.25}

    def test_training_only_vocabulary_transforms_unseen_metadata_sparsely(self):
        space=cbf.FeatureSpace.fit(self.training,self.weights)
        matrix=space.transform(self.candidates,tuple(sorted(self.candidates,key=int)),self.weights)
        self.assertEqual(matrix.shape,(5,len(space.vocabulary)))
        self.assertNotIn("TITLE__unseen",space.vocabulary)
        self.assertGreater(matrix[3].nnz,0)
        self.assertLessEqual(max(space.training_matrix.getnnz(axis=1)),len(space.vocabulary))

    def test_metadata_tokens_normalize_unicode_and_keep_field_identity(self):
        tokens=cbf.metadata_tokens({"tv_show_category":"Фільми", "tv_show_genre_1":"Документальні фільми",
            "tv_show_title":"Тест 2", "year_of_production":"2019", "channel_id":"0012", "channel_title":"Телеканал"})
        self.assertIn("CATEGORY__фільми",tokens["category"])
        self.assertIn("GENRE__документальні_фільми",tokens["genre"])
        self.assertIn("TITLE__тест",tokens["title"])
        self.assertIn("YEAR__decade_2010s",tokens["year"])
        self.assertIn("CHANNEL__id_12",tokens["channel"])

    def test_user_profile_uses_history_and_cold_user_gets_shared_fallback(self):
        space=cbf.FeatureSpace.fit(self.training,self.weights)
        items=space.training_items;idx={item:i for i,item in enumerate(items)}
        profiles={"warm":{"1":(3,600,100),"3":(1,200,200)},"cold":{}}
        users,stats=cbf.build_user_profiles(profiles,items,idx,space.training_matrix,("warm","cold"))
        self.assertEqual(stats["users_with_profile"],1)
        self.assertEqual(stats["cold_users"],1)
        candidate_ids=tuple(sorted(self.candidates,key=int))
        candidates=space.transform(self.candidates,candidate_ids,self.weights)
        ranks,details=cbf.rank_candidates(candidates,users,candidate_ids,tuple(reversed(candidate_ids)))
        self.assertIn("1",ranks[0])  # Previously watched programs remain recommendable.
        self.assertEqual(ranks[1][:5],tuple(reversed(candidate_ids))[:5])
        self.assertTrue(details[1]["profile_is_nonempty"] is False)
        self.assertEqual(len(set(ranks[0][:5])),5)

    def test_fast_cbf_metric_matches_frozen_step3_map_definition(self):
        class ToyProtocol:
            cohort=("1001","1002")
            candidates=("1","2","3","4","5","6")
            truth={"1001":frozenset(("1","3","5")),"1002":frozenset()}
        protocol=ToyProtocol()
        secondary={"1001":frozenset(("2","3")),"1002":frozenset()}
        fraction={"1001":frozenset(("3",)),"1002":frozenset(("5",))}
        rankings={"1001":("1","2","3","4","5","6"),"1002":("6","5","4","3","2","1")}
        result=step9.metric_record("toy",rankings,protocol,secondary,fraction,0.0)
        pred={u:tuple(rankings[u][:5]) for u in protocol.cohort}
        expected=evaluation.evaluate_predictions(pred,protocol.truth,protocol.cohort,protocol.candidates,strict=True)
        expected_secondary=evaluation.evaluate_predictions(pred,secondary,protocol.cohort,protocol.candidates,strict=True)
        expected_fraction=evaluation.evaluate_predictions(pred,fraction,protocol.cohort,protocol.candidates,strict=True)
        self.assertAlmostEqual(result["primary_map_at_5"],expected["primary_map_at_5"],places=15)
        self.assertAlmostEqual(result["secondary_map_at_5"],expected_secondary["primary_map_at_5"],places=15)
        self.assertAlmostEqual(result["fraction80_top5_sensitivity_map_at_5"],expected_fraction["primary_map_at_5"],places=15)
        self.assertAlmostEqual(result["candidate_coverage_fraction"],len({x for rec in pred.values() for x in rec})/6,places=15)


if __name__=="__main__":
    unittest.main(verbosity=2)
