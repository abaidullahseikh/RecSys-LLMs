"""Regression checks for compressed candidates, cold starts, and CBF orientation."""
import csv
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from types import SimpleNamespace

import train_lightgbm_step10 as training
import train_final_submission_v4 as final
import popularity


class Regressions(unittest.TestCase):
    def test_compressed_candidate_loader_preserves_large_ids(self):
        user='18446744073709551615'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'c.csv.gz'
            with gzip.open(path,'wt',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['user_id','candidate_program_ids','candidate_count'])
                writer.writerow([user,'1 2 3 4 5',5])
            self.assertEqual(training.candidate_sets(path,(user,))[user],('1','2','3','4','5'))

    def test_cold_start_ignores_classifier_scores(self):
        class Model:
            def predict(self,X):return np.arange(len(X),dtype=float)
        items=['1','2','3','4','5'];u='100'
        pred=training.predictions_for_group(Model(),np.ones((5,1)),[u]*5,items,['held']*5,'held',50,
            {u:items},items,[u],0)
        self.assertEqual(pred[u],tuple(items))

    def test_cbf_scores_include_all_candidate_rows(self):
        ids=('1','2','3','4','5');u='100'
        pop=popularity.PopularityModel(ids,{}, {}, {},'hash',(0,1),0,0)
        graph=SimpleNamespace(nodes=())
        data=SimpleNamespace(profiles={})
        matrix=csr_matrix(np.array([[.1],[.2],[.3],[.4],[.5]]))
        ranks,scores,_=final.ranks_for_models(graph,data,pop,{},None,matrix,csr_matrix([[1.]]),ids,(u,))
        self.assertEqual(ranks['cbf'][u],tuple(reversed(ids)))
        self.assertEqual(len(scores['cbf'][u]),5)


if __name__=='__main__':unittest.main(verbosity=2)
