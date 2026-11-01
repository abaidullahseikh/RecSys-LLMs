"""Deterministic calibration and user-group regression checks."""
import unittest
import numpy as np
from run_v10 import rank,group_indices


class V10Tests(unittest.TestCase):
    def setUp(self):
        self.z={'columns':np.asarray(['popularity_rank']),'X':np.asarray([[1],[0],[2],[3],[4],[5]],dtype=float),
            'users':np.asarray(['18446744073709551615']*6),'items':np.asarray(['1','2','3','4','5','6'])}
        self.u=('18446744073709551615',);self.ix=group_indices(self.z)
        self.seen=np.asarray([True]*5+[False]);self.cold=np.zeros(6,dtype=bool);self.light=np.ones(6,dtype=bool)
        self.s=np.asarray([.9,.8,.7,.6,.5,.49])

    def test_zero_adjustment_does_not_force_unseen(self):
        self.assertEqual(rank(self.s,self.z,self.ix,self.u,self.seen,self.cold,self.light)[self.u[0]],('1','2','3','4','5'))

    def test_positive_negative_corrections_and_light_gate(self):
        self.assertEqual(rank(self.s,self.z,self.ix,self.u,self.seen,self.cold,self.light,.02)[self.u[0]][-1],'6')
        self.assertEqual(rank(self.s,self.z,self.ix,self.u,self.seen,self.cold,self.light,-.02)[self.u[0]][-1],'5')
        self.light[:]=False
        self.assertEqual(rank(self.s,self.z,self.ix,self.u,self.seen,self.cold,self.light,.02,'light1to20')[self.u[0]][-1],'5')

    def test_cold_fallback_ignores_model_and_bonus(self):
        self.cold[:]=True
        self.assertEqual(rank(self.s,self.z,self.ix,self.u,self.seen,self.cold,self.light,99)[self.u[0]],('2','1','3','4','5'))

    def test_deterministic_tie_and_large_exact_user(self):
        p=rank(np.ones(6),self.z,self.ix,self.u,self.seen,self.cold,self.light)
        self.assertEqual(p[self.u[0]],('2','1','3','4','5'))
        self.assertEqual(p,rank(np.ones(6),self.z,self.ix,self.u,self.seen,self.cold,self.light))


if __name__=='__main__':unittest.main(verbosity=2)
