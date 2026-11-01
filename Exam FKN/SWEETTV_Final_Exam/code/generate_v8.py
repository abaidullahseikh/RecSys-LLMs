"""Mandatory experimental V8: valid selected new model, regardless of baseline gain."""
import csv,time,resource,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
import validation
from v8_common import ROOT,OUT,SOURCE,SIGNALS,sha,read,dump,load,records,rank
from build_v8 import added_signals,values_for
from train_v8 import columns_for


def main():
    t=time.monotonic();destination=ROOT/'submission/final_submission_v8_experimental.csv'
    if destination.exists():raise FileExistsError('Never overwrite submission')
    verified=read(OUT/'validation_verification.json')
    if not verified['all_passed']:raise ValueError('Essential validation checks did not pass')
    ex=read(OUT/'experiment.json');name=ex['selected']
    if name=='E1_binary75':raise ValueError('Cannot relabel original baseline as experimental V8')
    if sha(OUT/'final_model.txt')!=ex['final_model_sha256']:raise ValueError('Final model checksum differs')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if len(template)!=1260 or len(set(template))!=1260:raise ValueError('Template cohort invalid')
    z=load(ROOT/'results/v5_experiment/test_features.npz');fm=read(ROOT/'results/v5_experiment/feature_manifest.json')
    if sha(ROOT/'results/v5_experiment/test_features.npz')!=fm['test']['features_sha256']:raise ValueError('Final source cache checksum differs')
    if z['columns'].tolist()!=ex['feature_names'][:75]:raise ValueError('Source schema differs from training')
    if tuple(dict.fromkeys(z['users']))!=template:raise ValueError('Source inference user order differs')
    wanted=set(zip(z['users'].tolist(),z['items'].tolist()))
    bit,recent,totals,provenance=added_signals('history',1595797200,set(template),wanted)
    extra=np.asarray([values_for(str(u),str(i),bit,recent,totals) for u,i in zip(z['users'],z['items'])],dtype=np.float32)
    X=np.column_stack([z['X'],extra]);columns=z['columns'].tolist()+list(SIGNALS)
    if columns!=ex['feature_names'] or not np.isfinite(X).all():raise ValueError('Final feature schema/values inconsistent')
    np.savez_compressed(OUT/'test.npz',X=X,users=z['users'],items=z['items'],columns=np.asarray(columns))
    model=lgb.Booster(model_file=str(OUT/'final_model.txt'));cols=columns_for(name)
    if model.feature_name()!=[columns[i] for i in cols]:raise ValueError('Model feature order inconsistent')
    s=model.predict(X[:,cols],num_threads=4)
    inference={'X':X,'users':z['users'],'items':z['items'],'columns':np.asarray(columns)}
    pred=rank(s,inference,np.arange(len(s)),template)
    pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
    if len(pool)!=6636:raise ValueError('Eligible pool changed')
    for u,p in pred.items():
        if len(p)!=5 or len(set(p))!=5 or not set(p)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in p):raise ValueError('Invalid inference Top5')
    with destination.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(pred[u])) for u in template)
    protected=read(OUT/'preservation_before.json')
    if any(sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=h for p,h in protected.items()):raise ValueError('Prior artifact changed')
    dump('test_feature_manifest.json',{'source_v5_cache_sha256':sha(ROOT/'results/v5_experiment/test_features.npz'),
        'test_npz_sha256':sha(OUT/'test.npz'),'rows':len(s),'columns':columns,'selected_columns':cols,
        'new_signal_provenance':provenance,'history_behavior_through_week30_only':True,
        'existing75_signal_context':'Verified V5 cache; candidate graph/profiles fitted on full Weeks11-30; rolling behavioral75 features Weeks21-30. Same transformations as V5 baseline.',
        'future_schedule_use':'Weeks31-42 metadata/availability only, not user behavior'})
    manifest={'version':'V8_experimental','selected':name,'status':'experiment_not_Kaggle_scored',
        'selected_development_metrics':ex['comparisons'][name]['development'],
        'baseline_development_metrics':ex['comparisons']['E1_binary75']['development'],
        'selected_assessment_metrics':ex['comparisons'][name]['assessment'],
        'model_params':ex['selected_params'],'rounds':ex['rounds'],'selected_feature_names':model.feature_name(),
        'model_path':str((OUT/'final_model.txt').relative_to(ROOT)),'model_sha256':sha(OUT/'final_model.txt'),
        'data_manifest_sha256':sha(OUT/'data_manifest.json'),'experiment_sha256':sha(OUT/'experiment.json'),
        'training_example_source':'All4893 eligible historical users, features only Weeks11-20, historical labels Weeks21-30; final candidate context through Week30',
        'template_sha256':sha(SOURCE/'submission.csv'),'schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),
        'submission_path':str(destination.relative_to(ROOT)),'submission_sha256':sha(destination),'users':1260,
        'predictions_per_user':5,'eligible_programs':6636,'source_cache_rows':len(s),'repeat_previously_viewed_allowed':True,
        'cold_start_policy':'Same common training-history popularity ordering; no learned score for cold profiles',
        'Kaggle_score':None,'automatic_upload':False,'future_viewing_used':False,
        'recommended_final':'submission/final_submission_v5.csv','V5_private_MAP_user_reported':.39706,
        'local_regression_did_not_block_experimental_CSV':True,'seconds':time.monotonic()-t,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'versions':{'python':sys.version,'lightgbm':lgb.__version__,'numpy':np.__version__}}
    dump('submission_manifest.json',manifest)
    print({k:manifest[k] for k in ('selected','submission_path','submission_sha256','users','seconds','peak_rss_mib')})


if __name__=='__main__':main()
