"""Regression tests for verification/schema repairs; no model training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from verify_v6 import choose_inner,blend_reference,ranked,reference_ap
from create_v6_test_features import validate_existing


class VerifierRepairs(unittest.TestCase):
    def test_selection_uses_inner_summaries_not_outer_scores(self):
        a={'recipe':'temporal50','weight':.75,'metrics':{'secondary':.464,'primary':.838}}
        b={'recipe':'expanded100','weight':1.,'metrics':{'secondary':.459,'primary':.836}}
        ex={'development_oof_results':[a,b],'outer_comparisons':{'expanded100':{'secondary':.493}}}
        selection={'recipes':[{'id':'temporal50','k':50},{'id':'expanded100','k':100}]}
        self.assertEqual(choose_inner(ex,selection),a)

    def test_blend_uses_full_V5_rank_and_skips_missing_baseline_vote(self):
        new={'100':('6','2','1','3','4','5')};base={'100':('1','2','3','4','5')}
        result=blend_reference(new,base,.5,['100'])['100']
        # .5/62 + .5/60 is slightly greater than 1/61; this is not a tie.
        self.assertEqual(result,('1','2','3','4','5'))

    def test_model_rank_reference_cold_start_and_ties(self):
        z={'columns':np.array(['user_cold_start','popularity_rank']),
           'users':np.array(['18446744073709551615']*3),'items':np.array(['1','2','3']),
           'X':np.array([[0,2],[0,0],[0,1]],dtype=float)}
        u='18446744073709551615'
        self.assertEqual(ranked([.5,.5,.4],z,[0,1,2],[u])[u],('2','1','3'))
        z['X'][:,0]=1
        self.assertEqual(ranked([9,2,3],z,[0,1,2],[u])[u],('2','3','1'))

    def test_rational_AP_truncates_and_suppresses_duplicate_credit(self):
        self.assertEqual(reference_ap({'1'},['1']*5),1)
        self.assertEqual(reference_ap({'6'},['1','2','3','4','5','6']),0)

    def test_existing_cache_check_is_read_only_and_requires_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.csv';source.write_text('source evidence\n')
            with self.assertRaises(FileNotFoundError):validate_existing(root,root)
            np.savez_compressed(root/'test_features.npz',X=np.array([[2.]],dtype=np.float32),
                users=np.array(['18446744073709551615']),items=np.array(['123']),columns=np.array(['program_airings_recent_2w']))
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            meta={'rows':1,'features':1,'columns':['program_airings_recent_2w'],'source_path':'source.csv',
                'source_sha256':digest(source),'feature_file_sha256':digest(root/'test_features.npz'),'recent_2w_nonzero_rows':1}
            m=root/'test_feature_manifest.json';m.write_text(json.dumps(meta));before=m.read_bytes()
            self.assertTrue(validate_existing(root,root)['all_passed'])
            self.assertEqual(before,m.read_bytes())
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source checksum'):validate_existing(root,root)


if __name__=='__main__':unittest.main(verbosity=2)
