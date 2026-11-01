"""Refit selected new full-expanded-pool ranker and emit a separately verified V12 CSV."""
import sys
from v12_common import *

def main():
    t=time.monotonic();destination=ROOT/'submission/final_submission_v12_experimental.csv';path=OUT/'final_model.txt'
    if destination.exists() or path.exists():raise FileExistsError('Preserve existing final model/submission')
    vr=read(OUT/'validation_verification.json')
    if not vr['all_passed']:raise ValueError('Final feature/leakage/validation gate did not pass')
    selection=read(OUT/'selection.json');chosen=selection['selected'];z,_=data();targets=load(OUT/'validation_targets.npz');targetname='binary' if chosen=='E3' else 'graded';y=targets[targetname]
    if chosen not in ('E3','E4'):raise ValueError('Not a newly trained V12 strategy')
    indices=np.arange(len(z['users']));groups=grouping(z['users']);np.save(OUT/'final_groups.npy',groups,allow_pickle=False);bst,fit_seconds=fit(z,y,indices,path,chosen)
    test,_=data('test');cached=load(OUT/'test_model_features.npz')
    if bst.feature_name()!=test['columns'].tolist() or not np.array_equal(test['X'],cached['X']) or not np.isfinite(test['X']).all() or test['X'].dtype!=np.float32:raise ValueError('Final schema or transformations differ')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if len(template)!=1260 or tuple(dict.fromkeys(test['users']))!=template:raise ValueError('Exact original test identities/order differ')
    ti=time.monotonic();score=bst.predict(test['X'],num_threads=4);inference_seconds=time.monotonic()-ti
    if not np.array_equal(score,bst.predict(test['X'],num_threads=4)):raise ValueError('Nondeterministic model inference')
    pred=ranks(score,test,template);pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0};ix=defaultdict(dict)
    for j,(u,i) in enumerate(zip(test['users'],test['items'])):ix[str(u)][str(i)]=j
    missing=test['columns'].tolist().index('missing_target_availability')
    for u,p in pred.items():
        q=p[:5]
        if len(q)!=5 or len(set(q))!=5 or not set(q)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') or test['X'][ix[u][i],missing] for i in q):raise ValueError('Invalid or unavailable final Top5')
    prior={v:{r['user_id']:tuple(r['tv_show_id'].split()) for r in records(ROOT/f'submission/final_submission_{v.lower()}_experimental.csv')} for v in ('V10','V11')}
    if any(all(pred[u][:5]==p[u] for u in template) for p in prior.values()):raise ValueError('Cannot rename unchanged historical predictions as V12')
    protected=read(OUT/'preservation_before.json')
    if any(sha(p)!=h for p,h in protected.items()):raise ValueError('Protected prior artifact changed')
    np.save(OUT/'final_scores.npy',score,allow_pickle=False)
    with destination.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(pred[u][:5])) for u in template)
    dump('submission_manifest.json',{'version':'V12_experimental','selected':chosen,'selection_sha256':sha(OUT/'selection.json'),'model_path':str(path),'model_sha256':sha(path),
        'objective':'lambdarank','parameters':parameters(chosen),'rounds':78,'feature_names':bst.feature_name(),'features':77,'training_rows':len(indices),'training_users':len(groups),
        'training_group_sum':int(groups.sum()),'final_groups_sha256':sha(OUT/'final_groups.npy'),'training_label_kind':targetname,'training_label_histogram':dict(Counter(str(int(v)) for v in y)),
        'training_history':[1583704800,1589749200],'historical_training_target':[1589749200,1595797200],
        'final_history':[1583704800,1595797200],'final_target':list(validation.WINDOWS['test']),
        'training_scope':'All4893 eligible historical users; all expanded candidate rows; historical target labels only throughWeek30. Final retrieval refitted on all permitted behavior throughWeek30, with trailing10wk behavioral feature definitions.',
        'candidate_policy':'Original verified five-generator Top50 union plus first50 positive-PPR historical-unseen programs absent from original pool; no forced ground-truth positives; no negative subsampling',
        'final_features_recomputed_for_consistency':True,'test_build_sha256':sha(OUT/'test_build.json'),'feature_cache_sha256':sha(OUT/'test_model_features.npz'),
        'target_schedule_use':'Availability and metadata only; no future user viewing','target_schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),
        'cold_start':'Ignore learned scores for absence of full historical qualifying profile; common historical popularity rank then numeric program-ID tie break',
        'users':1260,'candidate_rows':len(test['users']),'additional_pairs':read(OUT/'test_build.json')['additional_pairs'],'eligible_programs':len(pool),
        'changed_lists_vs_previous':{v:sum(pred[u][:5]!=p[u] for u in template) for v,p in prior.items()},
        'submission_path':str(destination.relative_to(ROOT)),'submission_sha256':sha(destination),'local_development_MAP':selection['development'][chosen]['MAP5'],
        'local_assessment_MAP':read(OUT/'experiment.json')['comparison'][chosen]['assessment']['MAP5'],
        'Kaggle_private_MAP':None,'Kaggle_public_MAP':None,'automatic_upload':False,'best_genuinely_scored':'V10','best_private_user_reported':.40637,
        'training_seconds':fit_seconds,'inference_seconds':inference_seconds,'seconds':time.monotonic()-t,'peak_rss_mib':rss(),
        'versions':{'python':sys.version,'lightgbm':lgb.__version__,'numpy':np.__version__}})
    print({'selected':chosen,'submission':str(destination),'sha256':sha(destination),'changed_lists':{v:sum(pred[u][:5]!=p[u] for u in template) for v,p in prior.items()},'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
