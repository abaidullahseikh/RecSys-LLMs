import unittest
import numpy as np
import evaluation
from v6_data import corrected_ap,affinity,window_contains
from features_v5 import completion_flags,week_age,ZONE
from datetime import datetime


class V6Tests(unittest.TestCase):
    def test_ap_duplicate_and_cutoff_regression(self):
        for r,p in [({'1'},['1']*5),({'6'},['1','2','3','4','5','6']), (set(),['1','2','3','4','5'])]:
            self.assertEqual(corrected_ap(r,p),evaluation.average_precision_at_5(r,p))
        self.assertEqual(corrected_ap({'1'},['1']*5),1)

    def test_affinity_consistent_schedule_population(self):
        self.assertEqual(affinity({'a':3,'b':1},{'a':1,'b':1}),.5)
        self.assertEqual(affinity({'a':3,'b':1},{'a':3,'b':1}),.625)
        self.assertEqual(affinity({}, {'a':1}),0)
        self.assertEqual(affinity({'a':1}, {}),0)

    def test_no_temporal_boundary_crossing(self):
        self.assertTrue(window_contains(10,20,10,20))
        self.assertFalse(window_contains(19,21,10,20))
        self.assertFalse(window_contains(9,11,10,20))
        self.assertFalse(window_contains(20,21,10,20))

    def test_calendar_week_and_fraction_exactness(self):
        end=int(datetime(2020,5,18,tzinfo=ZONE).timestamp())
        stamp=int(datetime(2020,5,11,0,20,tzinfo=ZONE).timestamp())
        self.assertEqual(week_age(stamp,end),0)
        self.assertEqual(completion_flags(799,1000),(True,False,False))
        self.assertEqual(completion_flags(950,1000),(True,True,True))


if __name__=='__main__':unittest.main(verbosity=2)
