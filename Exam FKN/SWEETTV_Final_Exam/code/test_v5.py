"""Regression tests for corrected feature semantics and V5 scoring."""
import unittest
from datetime import datetime
from fractions import Fraction
import numpy as np
from features_v5 import week_age,window_start,completion_flags,ZONE
from train_v5 import predictions,ap,user_hash


class Tests(unittest.TestCase):
    def test_calendar_week_after_dst(self):
        end=int(datetime(2020,5,18,tzinfo=ZONE).timestamp())
        monday=int(datetime(2020,5,11,0,20,tzinfo=ZONE).timestamp())
        sunday=int(datetime(2020,5,10,23,59,tzinfo=ZONE).timestamp())
        self.assertEqual(week_age(monday,end),0)
        self.assertEqual(week_age(sunday,end),1)
        self.assertEqual(datetime.fromtimestamp(window_start(end),ZONE),datetime(2020,3,9,tzinfo=ZONE))

    def test_completion_threshold_boundaries(self):
        self.assertEqual(completion_flags(50,100),(True,False,False))
        self.assertEqual(completion_flags(80,100),(True,True,False))
        self.assertEqual(completion_flags(95,100),(True,True,True))
        self.assertEqual(completion_flags(79,100),(True,False,False))
        with self.assertRaises(ValueError):completion_flags(101,100)

    def test_ap_independent_hand_value(self):
        self.assertAlmostEqual(ap({'1','3','5'},('1','2','3','4','5')),float(Fraction(34,45)),places=15)
        self.assertEqual(ap(set(),('1','2','3','4','5')),0)

    def test_score_ties_and_cold_users(self):
        names=['ppr_rank','personal_frequency_rank','popularity_rank','user_cold_start']
        X=np.array([[j,j,5-j,0] for j in range(6)],dtype=float)
        u='18446744073709551615';users=np.array([u]*6);items=np.array(['1','2','3','4','5','6'])
        scores=np.array([0,1,2,3,4,5],dtype=float)
        self.assertEqual(predictions(scores,X,users,items,(u,),names,1.)[u],('6','5','4','3','2'))
        X[:,-1]=1
        self.assertEqual(predictions(-scores,X,users,items,(u,),names,1.)[u],('6','5','4','3','2'))
        self.assertEqual(user_hash(u),user_hash(u))


if __name__=='__main__':unittest.main(verbosity=2)
