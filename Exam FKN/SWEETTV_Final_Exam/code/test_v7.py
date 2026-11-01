"""Small regressions for query grouping and the genuinely new repeat signal."""
import unittest
import numpy as np
from run_v7 import grouping,streak,model_params


class V7Tests(unittest.TestCase):
    def test_group_lengths_and_exact_large_string_ids(self):
        users=np.array(['18446744073709551615']*3+['000123']*2+['123']*4)
        np.testing.assert_array_equal(grouping(users),[3,2,4])
        self.assertEqual(int(grouping(users).sum()),len(users))

    def test_interleaved_user_groups_rejected(self):
        with self.assertRaises(ValueError):grouping(np.array(['1','2','1']))

    def test_streak_is_not_active_week_count_or_recent_count(self):
        self.assertEqual(streak(0),0)
        self.assertEqual(streak(0b10101),1)
        self.assertEqual(streak(0b00111),3)
        self.assertEqual(streak(0b11110),0)
        self.assertEqual(streak(0b1111111111),10)
        # Both have three active weeks and two in recent four, but different streaks.
        self.assertEqual((0b10011).bit_count(),(0b10101).bit_count())
        self.assertNotEqual(streak(0b10011),streak(0b10101))

    def test_lambdarank_cutoff_seed_threads(self):
        p=model_params('E2_LambdaRank',{'learning_rate':.05})
        self.assertEqual(p['objective'],'lambdarank')
        self.assertEqual(p['eval_at'],[5])
        self.assertEqual(p['num_threads'],4)
        self.assertEqual(p['seed'],2026)
        self.assertEqual(p['label_gain'],[0,1])


if __name__=='__main__':unittest.main(verbosity=2)
