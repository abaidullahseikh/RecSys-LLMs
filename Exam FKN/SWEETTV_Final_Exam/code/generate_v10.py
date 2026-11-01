"""Generate genuine new V10 experiment only after correctness verification."""
import csv,json,time,resource
from pathlib import Path
import numpy as np
import lightgbm as lgb
from v8_common import ROOT,SOURCE,sha,load,read,records
from run_v10 import OUT,V9,V8,history_flags,group_indices,rank,dump


def main():
    t=time.monotonic();destination=ROOT/'submission/final_submission_v10_experimental.csv'
    if destination.exists():raise FileExistsError('Never overwrite submission')
    if not read(OUT/'validation_verification.json')['all_passed']:raise ValueError('Validation correctness failed')
    ex=read(OUT/'experiment.json');strategy=ex['strategy'];original=load(V8/'test.npz');z=load(V9/'test_features.npz')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if len(template)!=1260 or tuple(dict.fromkeys(z['users']))!=template:raise ValueError('Template/source order mismatch')
    seen,sizes,cold,light,history=history_flags(original,'history');ix=group_indices(original)
    score9=lgb.Booster(model_file=str(V9/'final_model.txt')).predict(z['X'],num_threads=4)
    if not np.array_equal(score9,np.load(V9/'V9_test_scores.npy',allow_pickle=False)):raise ValueError('Original V9 inference scores do not reproduce')
    if strategy['type']=='pairwise_V9_features':
        path=OUT/'final_pairwise.txt';model=lgb.Booster(model_file=str(path))
        if sha(path)!=ex['pairwise_info']['final_model_sha256']:raise ValueError('Pairwise final checksum differs')
        if model.feature_name()!=z['columns'].tolist():raise ValueError('Final feature schema differs from training')
        score=model.predict(z['X'],num_threads=4);pred=rank(score,original,ix,template,seen,cold,light)
    else:
        path=V9/'final_model.txt';model=lgb.Booster(model_file=str(path));score=score9
        if strategy['unseen_bonus']==0.:raise ValueError('Cannot rename unchanged V9 as V10')
        pred=rank(score,original,ix,template,seen,cold,light,strategy['unseen_bonus'],strategy['policy'])
    previous={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(ROOT/'submission/final_submission_v9_experimental.csv')}
    if pred==previous:raise ValueError('V10 is identical to V9; do not rename prior prediction file')
    pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
    if set(pred)!=set(template) or any(len(p)!=5 or len(set(p))!=5 or not set(p)<=pool for p in pred.values()):raise ValueError('Invalid inference Top5')
    with destination.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(pred[u])) for u in template)
    protected=read(OUT/'preservation_before.json')
    if any(sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=h for p,h in protected.items()):raise ValueError('Protected prior work changed')
    dump('submission_manifest.json',{'version':'V10_experimental','strategy':strategy,'status':'experimental_not_Kaggle_scored',
        'model_path':str(path.relative_to(ROOT)),'model_sha256':sha(path),'feature_names':model.feature_name(),
        'model_parameters':ex['pairwise_info']['parameters'] if ex['pairwise_info'] else read(V9/'experiment.json')['parameters'],
        'source_feature_cache':'results/v9_experiment/test_features.npz','feature_sha256':sha(V9/'test_features.npz'),
        'source_V8_feature_cache_sha256':sha(V8/'test.npz'),'history_classification':history,
        'history_context':'Weeks11-30 permitted behavior; original V9 features and final candidate context reused, not future viewing',
        'target_schedule_use':'Weeks31-42 broadcast metadata and candidate eligibility only',
        'users':1260,'predictions_per_user':5,'candidate_pool_size':len(pool),'template_sha256':sha(SOURCE/'submission.csv'),
        'schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),'original_V9_score_replay':True,
        'changed_lists_vs_V9':sum(pred[u]!=previous[u] for u in template),'submission_path':str(destination.relative_to(ROOT)),
        'submission_sha256':sha(destination),'development_MAP':ex['selected_development_MAP'],'assessment_MAP':ex['selected_assessment_MAP'],
        'recommended_final':'submission/final_submission_v8_experimental.csv','V8_private_MAP_user_reported':.39908,
        'Kaggle_score':None,'automatically_submitted':False,'future_user_viewing_used':False,
        'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'reason_generated_despite_regression':'Mandatory separately trained valid experimental V10 model; no positive unseen calibration forced'})
    print({'path':str(destination),'sha256':sha(destination),'strategy':strategy,'seconds':time.monotonic()-t})


if __name__=='__main__':main()
