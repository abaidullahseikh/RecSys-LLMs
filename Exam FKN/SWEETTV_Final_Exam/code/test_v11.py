"""Meaningful sampling/retrieval/group-boundary regressions for V11."""
import unittest
import numpy as np
from v11_common import sample_negatives,digest_split,grouping
from v11_retrieval import additional
from features_v5 import week_age,ZONE
from datetime import datetime

class V11Tests(unittest.TestCase):
    def test_hard_negative_positive_retention_and_fixed_budget(self):
        u='18446744073709551615';n=65;z={'users':np.asarray([u]*n),'items':np.asarray([str(i+1) for i in range(n)]),'columns':np.asarray(['popularity_rank']),
            'X':np.arange(n,dtype=np.float32).reshape(-1,1),'binary_label':np.asarray([1]*5+[0]*60)};s=np.arange(n,dtype=float)
        chosen=sample_negatives(z,np.arange(n),s)
        self.assertEqual(len(chosen),45);self.assertTrue(set(range(5))<=set(chosen));self.assertTrue(set(range(45,65))<=set(chosen))
        self.assertEqual(set(chosen),set(sample_negatives(z,np.arange(n)[::-1],s)))

    def test_retrieval_excludes_seen_original_zero_and_has_cap(self):
        rank=('1','2','3','4','5');scores={'1':.9,'2':.8,'3':.7,'4':0.,'5':.5}
        self.assertEqual(additional(rank,scores,{'1'},{'2'},1),('3',))
        self.assertEqual(additional(rank,scores,{'1'},{'2'},50),('3','5'))

    def test_group_contiguity_and_exact_large_user_ids(self):
        self.assertEqual(grouping(np.asarray(['18446744073709551615']*2+['000123']*3)).tolist(),[2,3])
        with self.assertRaises(ValueError):grouping(np.asarray(['1','2','1']))
        self.assertIn(digest_split('18446744073709551615'),(0,1))

    def test_local_calendar_boundary_across_dst(self):
        cutoff=int(datetime(2020,4,13,tzinfo=ZONE).timestamp());stamp=int(datetime(2020,3,23,tzinfo=ZONE).timestamp())
        self.assertEqual(week_age(stamp,cutoff),2)
        self.assertEqual(week_age(cutoff-1,cutoff),0)

if __name__=='__main__':unittest.main(verbosity=2)
