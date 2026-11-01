"""Focused V8 tests without claiming the hypothesis is an official rule."""
from collections import Counter
from datetime import datetime
import unittest
import numpy as np
import data_pipeline as dp
from features_v5 import completion_flags,ZONE
from build_v8 import values_for
from train_v8 import columns_for
from v8_common import top_counts,partition,ap


class V8Tests(unittest.TestCase):
    def test_fragments_aggregate_per_airing_before_completion_threshold(self):
        a=[dp.Airing(1,0,500,11)]
        v=[dp.View('18446744073709551615',1,0,200),dp.View('18446744073709551615',1,300,500)]
        result=dp.match_views(v,a,(0,500));overlap=next(iter(result.values()))
        self.assertEqual(overlap.seconds,400)
        self.assertTrue(completion_flags(overlap.seconds,500)[1])

    def test_overlapping_fragments_not_double_counted(self):
        v=[dp.View('100',1,0,250),dp.View('100',1,100,300)]
        overlap=next(iter(dp.match_views(v,[dp.Airing(1,0,500,11)],(0,500)).values()))
        self.assertEqual(overlap.seconds,300)
        self.assertFalse(completion_flags(overlap.seconds,500)[1])

    def test_channel_zero_and_future_behavior_excluded(self):
        v=[dp.View('100',1,0,500),dp.View('100',2,0,500),dp.View('100',1,600,900)]
        r=dp.match_views(v,[dp.Airing(1,0,500,0),dp.Airing(3,0,500,11),dp.Airing(1,600,900,12)],(0,500))
        self.assertEqual(r,{})

    def test_fraction_inclusive_and_duration_normalization(self):
        self.assertTrue(completion_flags(400,500)[1]);self.assertFalse(completion_flags(399,500)[1])
        self.assertFalse(completion_flags(400,1000)[1])
        with self.assertRaises(ValueError):completion_flags(501,500)

    def test_hypothesis_counts_and_our_numeric_tie_convention(self):
        counts={'18446744073709551615':Counter({'10':2,'2':2,'3':4,'4':1,'5':1,'6':1})}
        self.assertEqual(top_counts(counts)['18446744073709551615'],['3','2','10','4','5'])
        self.assertEqual(partition('18446744073709551615'),partition('18446744073709551615'))

    def test_new_signals_are_distinct_and_safe_for_empty_history(self):
        u='100';i='10'
        self.assertEqual(values_for(u,i,{(u,i):0b1011},{(u,i):2},{u:8}),[2,.25])
        self.assertEqual(values_for(u,i,{}, {},{}),[0,0.])
        self.assertEqual(len(columns_for('E1_binary75')),75)
        self.assertEqual(columns_for('E2_streak')[-1],75)
        self.assertEqual(columns_for('E2_recent_share')[-1],76)
        self.assertEqual(len(columns_for('E2_both')),77)

    def test_map5_empty_duplicate_and_cutoff(self):
        self.assertEqual(ap({'1'},['1']*5),1)
        self.assertEqual(ap({'6'},['1','2','3','4','5','6']),0)
        self.assertEqual(ap(set(),['1','2','3','4','5']),0)


if __name__=='__main__':unittest.main(verbosity=2)
