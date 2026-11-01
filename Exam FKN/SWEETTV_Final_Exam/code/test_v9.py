"""Small deterministic regressions for the one targeted calibration and fusion."""
import unittest
from unittest.mock import patch
import numpy as np
from train_v9 import new_matrix,fuse,rankings


class V9Tests(unittest.TestCase):
    def test_calibration_smoothing_and_history_only_denominator(self):
        z={'columns':np.asarray(['ppr_score','pop_distinct_users']),
            'X':np.asarray([[.2,80],[.2,380],[0,0]],dtype=np.float32)}
        with patch('train_v9.read',return_value={'selected_columns':[0,1]}):X,names=new_matrix(z,1000)
        self.assertAlmostEqual(X[0,-1],.2*np.sqrt(10),places=6)
        self.assertAlmostEqual(X[1,-1],.2*np.sqrt(2.5),places=6)
        self.assertEqual(X[2,-1],0)
        self.assertEqual(names[-1],'ppr_support_adjusted_score')
        with self.assertRaises(ValueError):new_matrix(z,0)

    def test_identical_rankings_and_large_ids(self):
        u='18446744073709551615';r={u:('1','2','3','4','5','6')}
        self.assertEqual(fuse(r,r,.75,(u,))[u],('1','2','3','4','5'))
        with self.assertRaises(ValueError):fuse(r,{u:('1','2','3','4','5')},.5,(u,))

    def test_rank_ties_and_cold_fallback(self):
        u='000123';z={'columns':np.asarray(['user_cold_start','popularity_rank']),
            'X':np.asarray([[0,2],[0,0],[0,1]],dtype=float),'users':np.asarray([u]*3),'items':np.asarray(['1','2','3'])}
        self.assertEqual(rankings([.5,.5,.4],z,[0,1,2],(u,))[u],('2','1','3'))
        z['X'][:,0]=1
        self.assertEqual(rankings([99,2,3],z,[0,1,2],(u,))[u],('2','3','1'))


if __name__=='__main__':unittest.main(verbosity=2)
