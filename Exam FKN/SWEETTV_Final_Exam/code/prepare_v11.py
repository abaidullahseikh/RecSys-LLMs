"""Read-only V10 reproduction and preservation baseline for V11."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import time,json,resource
import numpy as np
import lightgbm as lgb
import evaluation,validation
from v8_common import ROOT,SOURCE,read,load,sha,records,inner_fold,map5
from train_v9 import new_matrix
from verify_v10 import actual_flags,rank_ref,map5 as independent_map
from run_v7 import grouping

OUT=ROOT/'results/v11_experiment';V8=ROOT/'results/v8_experiment';V9=ROOT/'results/v9_experiment';V10=ROOT/'results/v10_experiment'

def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')

def main():
    t=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'baseline_reproduction.json').exists():raise FileExistsError('Completed reproduction exists')
    if not (OUT/'preservation_before.json').exists():
        paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('.venv','.venv-lightgbm','__pycache__','.git')) and OUT not in p.parents and 'v11' not in p.name and p.suffix!='.pyc']+list(SOURCE.glob('*.csv'))
        dump('preservation_before.json',{str(p):sha(p) for p in paths})
    checks=[]
    def ck(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok:raise ValueError(name)
    ex=read(V10/'experiment.json');sm=read(V10/'submission_manifest.json');e9=read(V9/'experiment.json');z=load(V8/'validation.npz');pop=read(V8/'population.json');truth=read(V8/'labels.json')['secondary']
    ck('V10_CSV_checksum_matches_record',sha(ROOT/sm['submission_path'])==sm['submission_sha256']=='39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e')
    ck('V10_model_checksum_matches_record',sha(V10/'final_pairwise.txt')==sm['model_sha256']==ex['pairwise_info']['final_model_sha256'])
    ck('V10_prior_verification_passed',read(V10/'submission_verification.json')['all_passed'])
    ck('feature_cache_hash',sha(V8/'validation.npz')==e9['feature_source_sha256'])
    X,names=new_matrix(z,pop['historical_users']);flags,_,_,_=actual_flags(z,'train')
    ck('historical_flags_independent',all(np.array_equal(flags[k],v) for k,v in load(V10/'validation_flags.npz').items()))
    dev=ex['development_users'];ass=ex['assessment_users'];dm=np.isin(z['users'],dev);fold=np.asarray([inner_fold(str(u)) for u in z['users']]);scores=np.zeros(len(X));models=[]
    for f in range(4):
        vi=np.flatnonzero(dm&(fold==f)) if f<3 else np.flatnonzero(~dm)
        ti=np.flatnonzero(dm&(fold!=f)) if f<3 else np.flatnonzero(dm)
        path=V10/(f'pairwise_inner{f}.txt' if f<3 else 'pairwise_assessment.txt');bst=lgb.Booster(model_file=str(path))
        ck(f'fold{f}_supervised_user_separation',not set(z['users'][ti])&set(z['users'][vi]) and not set(z['users'][ti])&set(ass))
        ck(f'fold{f}_feature_objective_rounds',bst.feature_name()==names and bst.params['objective']=='lambdarank' and bst.current_iteration()==78)
        ck(f'fold{f}_contiguous_groups',grouping(z['users'][ti]).sum()==len(ti) and grouping(z['users'][vi]).sum()==len(vi))
        if f<3:ck(f'fold{f}_model_hash',sha(path)==ex['pairwise_info']['folds'][f]['model_sha256'])
        scores[vi]=bst.predict(X[vi],num_threads=4);models.append({'path':str(path),'sha256':sha(path),'fit_users':len(set(z['users'][ti])),'evaluation_users':len(set(z['users'][vi]))})
    pred=rank_ref(scores,z,pop['users'],flags)
    ck('all_crossfit_predictions_exact',pred==evaluation.parse_prediction_records(list(records(V10/'selected_all_crossfit.csv'))))
    for stage,users in [('development',dev),('assessment',ass)]:
        ck(stage+'_MAP_exact',map5(pred,truth,users)==ex['selected_'+stage+'_MAP'])
        ck(stage+'_independent_MAP',abs(independent_map(pred,truth,users)-ex['selected_'+stage+'_MAP'])<1e-14)
    tz=load(V9/'test_features.npz');original=load(V8/'test.npz');tf,_,_,_=actual_flags(original,'history');template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    ck('test_cache_hash_schema',sha(V9/'test_features.npz')==sm['feature_sha256'] and tz['columns'].tolist()==names)
    bst=lgb.Booster(model_file=str(V10/'final_pairwise.txt'));score=bst.predict(tz['X'],num_threads=4);final=rank_ref(score,original,template,tf)
    ck('all_1260_final_V10_predictions_exact',final==evaluation.parse_prediction_records(list(records(ROOT/sm['submission_path']))))
    ck('permitted_historical_bounds',read(V8/'data_manifest.json')['history_provenance']['hi']==1589749200 and sm['history_classification']['window'][1]==1595797200)
    np.save(OUT/'V10_crossfit_scores.npy',scores,allow_pickle=False)
    dump('baseline_reproduction.json',{'exact':True,'checks':checks,'development_MAP':map5(pred,truth,dev),'assessment_MAP':map5(pred,truth,ass),'users':len(pop['users']),
        'development_users':dev,'assessment_users':ass,'feature_names':names,'parameters':ex['pairwise_info']['parameters'],'rounds':78,'models':models,
        'final_model_path':str(V10/'final_pairwise.txt'),'model_sha256':sha(V10/'final_pairwise.txt'),'submission_sha256':sha(ROOT/sm['submission_path']),
        'candidate_policy':'Frozen union top50 PPR/personal-frequency/association-rules/CBF/popularity, eligible schedule IDs, repeats allowed; original template75 rows preserved by V8',
        'training_window':[1583704800,1589749200],'label_window':[1589749200,1595797200],
        'relevance':'Top5 by >=120-second qualifying-airing frequency, numeric ID tie-break; official hidden target construction unconfirmed',
        'ordering':'Score desc, popularity rank asc, numeric program ID asc; common popularity fallback for cold users',
        'Kaggle_private_user_reported':.40637,'Kaggle_public_user_reported':.42051,'Kaggle_target_aspirational':.44801,
        'old_report_status':'V10 report predates newly user-reported Kaggle scores and remains unmodified',
        'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    print({'exact':True,'checks':len(checks),'development_MAP':map5(pred,truth,dev),'assessment_MAP':map5(pred,truth,ass),'seconds':time.monotonic()-t},flush=True)

if __name__=='__main__':main()
