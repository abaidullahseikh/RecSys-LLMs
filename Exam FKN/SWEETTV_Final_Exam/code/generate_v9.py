"""Generate the selected genuinely trained V9 experimental configuration only."""
import csv,json,time,resource,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from v8_common import ROOT,SOURCE,load,read,sha,records
from train_v9 import OUT,V8,new_matrix,rankings,fuse,dump


def main():
    t=time.monotonic();destination=ROOT/'submission/final_submission_v9_experimental.csv'
    if destination.exists():raise FileExistsError('Never overwrite a submission')
    if not read(OUT/'validation_verification.json')['all_passed']:raise ValueError('V9 validation correctness not verified')
    ex=read(OUT/'experiment.json');selected=ex['selected']
    if ex['status']!='trained_and_evaluated' or selected['id']=='E1_V8':raise ValueError('V9 must use a valid new model')
    if sha(OUT/'final_model.txt')!=ex['final_model_sha256']:raise ValueError('Final model integrity')
    z=load(V8/'test.npz');V8manifest=read(V8/'test_feature_manifest.json')
    if sha(V8/'test.npz')!=V8manifest['test_npz_sha256']:raise ValueError('V8 inference cache changed')
    history_manifest=read(ROOT/'data/step2/threshold_120s/interaction_manifest.json')
    n=history_manifest['periods']['history']['unique_users'];X,names=new_matrix(z,n)
    if names!=ex['feature_names']:raise ValueError('Inference schema mismatch')
    model=lgb.Booster(model_file=str(OUT/'final_model.txt'))
    if model.feature_name()!=names or model.num_feature()!=X.shape[1]:raise ValueError('Model order mismatch')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if len(template)!=1260 or len(set(template))!=1260 or tuple(dict.fromkeys(z['users']))!=template:raise ValueError('Template identity mismatch')
    score=model.predict(X,num_threads=4);new=rankings(score,z,np.arange(len(score)),template)
    baseline=np.load(OUT/'V8_test_scores.npy',allow_pickle=False);base=rankings(baseline,z,np.arange(len(score)),template)
    pred=fuse(new,base,selected['new_weight'],template)
    pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
    for u,p in pred.items():
        if len(p)!=5 or len(set(p))!=5 or not set(p)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in p):raise ValueError('Invalid output Top5')
    np.savez_compressed(OUT/'test_features.npz',X=X,users=z['users'],items=z['items'],columns=np.asarray(names))
    np.save(OUT/'V9_test_scores.npy',score,allow_pickle=False)
    protected=read(OUT/'preservation_before.json')
    if any(sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=h for p,h in protected.items()):raise ValueError('Protected artifact changed')
    with destination.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(pred[u])) for u in template)
    dump('test_feature_manifest.json',{'source_V8_features_sha256':sha(V8/'test.npz'),'feature_file_sha256':sha(OUT/'test_features.npz'),
        'names':names,'rows':len(X),'features':X.shape[1],'historical_users_denominator':n,
        'formula':'ppr_score*sqrt(historical_qualifying_users/(historical_distinct_program_viewers+20))',
        'source_historical_manifest_sha256':sha(ROOT/'data/step2/threshold_120s/interaction_manifest.json'),
        'feature_signal_window':'Weeks11-30 full-history PPR score/support only; original V8 rolling signals remain as recorded',
        'future_viewing_used':False,'target_schedule':'Weeks31-42 availability/metadata only'})
    dump('submission_manifest.json',{'version':'V9_experimental','selected':selected,'status':'experimental_unscored',
        'model_path':str((OUT/'final_model.txt').relative_to(ROOT)),'model_sha256':sha(OUT/'final_model.txt'),
        'feature_manifest_sha256':sha(OUT/'test_feature_manifest.json'),'feature_names':names,'model_params':ex['parameters'],'rounds':ex['rounds'],
        'training_population':'All4893 historical V8 eligible users; features Weeks11-20 and historical target labels Weeks21-30. Candidate context refitted through Week30 at final inference.',
        'candidate_policy':'Same verified V8 Top50 multi-generator union, all1260 template users preserved',
        'target_schedule_path':str(SOURCE/'export_arh_31-42-final.csv'),'schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),
        'template_sha256':sha(SOURCE/'submission.csv'),'users':len(template),'predictions_per_user':5,'eligible_ids':len(pool),
        'submission_path':str(destination.relative_to(ROOT)),'submission_sha256':sha(destination),
        'source_experiment_sha256':sha(OUT/'experiment.json'),'source_V8_submission_sha256':sha(ROOT/'submission/final_submission_v8_experimental.csv'),
        'new_configuration_development_MAP':ex['comparison']['selected']['development_MAP'],
        'V8_development_MAP':ex['comparison']['E1_V8']['development_MAP'],
        'new_configuration_assessment_MAP':ex['comparison']['selected']['assessment_MAP'],
        'recommended_final':'submission/final_submission_v8_experimental.csv','V8_private_MAP_user_reported':.39908,
        'Kaggle_score':None,'automatically_submitted':False,'future_viewing_used':False,
        'cold_start_policy':'Same common historical popularity fallback, ranks rather than learned cold-profile scores',
        'repeat_previously_watched_allowed':True,'generation_seconds':time.monotonic()-t,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'versions':{'python':sys.version,'lightgbm':lgb.__version__,'numpy':np.__version__}})
    print({'path':str(destination),'sha256':sha(destination),'selected':selected,'seconds':time.monotonic()-t})


if __name__=='__main__':main()
