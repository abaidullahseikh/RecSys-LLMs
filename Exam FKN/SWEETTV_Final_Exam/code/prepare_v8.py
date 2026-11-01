"""Preserve prior evidence and independently reproduce historical V5 before V8."""
import json,time
import numpy as np
import lightgbm as lgb
import evaluation,validation
from verify_v5 import rank_reference,ap
from v8_common import ROOT,OUT,SOURCE,sha,read,dump,load,records


def main():
    if OUT.exists():raise FileExistsError('Preserve V8 partial results; investigate before running again')
    OUT.mkdir();t=time.monotonic()
    protected=[]
    for folder in ('code','results','submission','evidence'):
        protected += [p for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents
            and '__pycache__' not in str(p) and 'v8' not in p.name and p.suffix not in ('.sqlite','.pyc')]
    protected += [ROOT/'exam_quiz_FKN.md']+list(SOURCE.glob('*.csv'))
    hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in protected}
    dump('preservation_before.json',hashes)
    ex=read(ROOT/'results/v5_experiment/experiment.json');fm=read(ROOT/'results/v5_experiment/feature_manifest.json')
    z=load(ROOT/'results/v5_experiment/validation_features.npz');outer=ex['outer_users'];mask=np.isin(z['users'],outer)
    if sha(ROOT/'results/v5_experiment/validation_features.npz')!=fm['validation']['features_sha256']:raise ValueError('V5 feature checksum')
    if z['columns'].tolist()!=ex['selection']['feature_names']:raise ValueError('V5 feature schema')
    m=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/evaluation_model.txt'))
    if sha(ROOT/'results/v5_experiment/evaluation_model.txt')!=ex['evaluation_model_sha256']:raise ValueError('V5 model checksum')
    pred=rank_reference(m.predict(z['X'][mask],num_threads=4),z['X'][mask],z['users'][mask],z['items'][mask],outer,z['columns'].tolist(),1.)
    original=evaluation.parse_prediction_records(list(records(ROOT/'results/v5_experiment/heldout_v5.csv')))
    truth={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    score=sum(ap(truth[u],pred[u]) for u in outer)/len(outer)
    if pred!=original or abs(score-.4886328125)>1e-14:raise ValueError('V5 failed reproduction; stop')
    if not set(z['items'])<=set(validation.EvaluationProtocol().candidates):raise ValueError('V5 candidate eligibility')
    if fm['validation']['trailing_history_utc']!=[1583704800,1589749200]:raise ValueError('V5 history window')
    dump('V5_reproduction.json',{'exact':True,'outer_users':len(outer),'MAP_original_top5_120':score,
        'feature_sha256':sha(ROOT/'results/v5_experiment/validation_features.npz'),
        'evaluation_model_sha256':sha(ROOT/'results/v5_experiment/evaluation_model.txt'),
        'saved_predictions_sha256':sha(ROOT/'results/v5_experiment/heldout_v5.csv'),'schema':z['columns'].tolist(),
        'history_window':fm['validation']['trailing_history_utc'],'target_window':fm['validation']['target_window'],
        'candidate_pool_count':len(validation.EvaluationProtocol().candidates),'seconds':time.monotonic()-t})
    dump('official_labels.json',{'data_url':'https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data',
        'retrieved_data_page':'JavaScript shell only, no accessible label definition',
        'official_api_url':'https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionId=24201',
        'confirmed_metric':'MAP@{K}, higher is better; exam specifies K5','official_80pct_rule_confirmed':False,
        'official_frequency_top5_construction_confirmed':False,'official_label_tie_break':'Not accessible/documented in retrieved sources',
        'choice':'Preserve >=120s top-five-by-qualifying-airing-count as primary V8 selection protocol; >=80%-duration Top5 is a hypothesis, reported separately; numeric item-ID tie break is our choice.',
        'another_participant_report_used_as_official_proof':False})
    print({'V5_reproduction_exact':True,'MAP':score,'protected_files':len(hashes),'seconds':time.monotonic()-t})


if __name__=='__main__':main()
