"""Frequency grades, qualifying occasions and leakage-boundary regressions."""
import unittest
import numpy as np
import data_pipeline as dp
from v12_common import grades,parameters,grouping

class V12Tests(unittest.TestCase):
    def test_explicit_grade_boundaries_and_invalid_counts(self):
        self.assertEqual(grades(np.asarray([0,1,2,3,4,50],dtype=np.int32)).tolist(),[0,1,2,2,3,3])
        with self.assertRaises(ValueError):grades(np.asarray([-1],dtype=np.int32))
        with self.assertRaises(ValueError):grades(np.asarray([1.5]))

    def test_fragments_union_then_threshold_one_occasion_per_airing(self):
        air=[dp.Airing(1,0,500,11),dp.Airing(1,1000,1500,11)]
        views=[dp.View('18446744073709551615',1,0,80),dp.View('18446744073709551615',1,60,120),dp.View('18446744073709551615',1,1000,1119)]
        overlaps=dp.match_views(views,air,(0,1500));qualifying=sum(v.seconds>=120 for v in overlaps.values())
        self.assertEqual(qualifying,1);self.assertEqual(grades(np.asarray([qualifying])).tolist(),[1])
        views[-1]=dp.View('18446744073709551615',1,1000,1120)
        qualifying=sum(v.seconds>=120 for v in dp.match_views(views,air,(0,1500)).values())
        self.assertEqual(qualifying,2);self.assertEqual(grades(np.asarray([qualifying])).tolist(),[2])

    def test_only_grade_gain_configuration_changes(self):
        a=parameters('E3');b=parameters('E4');self.assertEqual(a['label_gain'],[0,1]);self.assertEqual(b['label_gain'],[0,1,3,7])
        self.assertEqual({k:v for k,v in a.items() if k!='label_gain'},{k:v for k,v in b.items() if k!='label_gain'})
        self.assertLessEqual(a['num_threads'],4)

    def test_future_airing_outside_training_window_excluded(self):
        overlaps=dp.match_views([dp.View('000123',1,0,120),dp.View('000123',1,1000,1120)],
            [dp.Airing(1,0,500,11),dp.Airing(1,1000,1500,12)],(0,500))
        self.assertEqual(len(overlaps),1);self.assertEqual(next(iter(overlaps.values())).seconds,120)

    def test_metric_denominator_keeps_unretrieved_relevant_labels(self):
        from verify_v12 import evaluation_metric
        user='18446744073709551615'
        self.assertEqual(evaluation_metric({user:('1','2','3','4','5')},{user:['1','6']},np.asarray([user]),np.asarray(['1','2','3','4','5','6'])),.5)

if __name__=='__main__':unittest.main(verbosity=2)
