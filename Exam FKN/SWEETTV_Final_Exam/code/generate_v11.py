"""Refit the frozen genuinely new V11 strategy and generate checked inference."""
import time
import numpy as np
import lightgbm as lgb
from v11_common import *

def main():
    t=time.monotonic();destination=ROOT/'submission/final_submission_v11_experimental.csv'
    if destination.exists() or (OUT/'final_model.txt').exists():raise FileExistsError('Preserve final artifacts')
    if not read(OUT/'validation_verification.json')['all_passed']:raise ValueError('Validation correctness not verified')
    selection=read(OUT/'selection.json');chosen=selection['selected'];z,truth=model_data();params=read(OUT/'baseline_reproduction.json')['parameters']
    if chosen not in ('E2','E3','E4'):raise ValueError('Not a genuine V11 strategy')
    if chosen in ('E3','E4'):
        ix=np.flatnonzero(z['original']) if chosen=='E3' else np.arange(len(z['users']));score=np.load(OUT/'expanded_V10_crossfit_scores.npy',allow_pickle=False);sample=sample_negatives(z,ix,score)
        # Original V10 cross-fits exclude the scored user's labels, including assessment-user models fitted on development only.
        if not set(ix[z['binary_label'][ix]>0])<=set(sample):raise ValueError('Positive omitted')
        np.save(OUT/'final_training_indices.npy',sample,allow_pickle=False);path=OUT/'final_model.txt';bst=fit(z,sample,path,params)
    else:path=V10/'final_pairwise.txt';bst=lgb.Booster(model_file=str(path));sample=None
    if chosen=='E3':
        raw=load(V8/'test.npz');denominator=read(ROOT/'data/step2/threshold_120s/interaction_manifest.json')['periods']['history']['unique_users'];X,names=new_matrix(raw,denominator)
        cached=load(V9/'test_features.npz')
        if not np.array_equal(X,cached['X']) or names!=cached['columns'].tolist():raise ValueError('Final transformations differ from verified V9 cache')
        test={**raw,'X':X,'columns':np.asarray(names)};feature_path=V9/'test_features.npz';expanded=False
    else:
        test,_=model_data('test');feature_path=OUT/'test.npz';denominator=read(OUT/'test_build.json')['context']['historical_users'];expanded=True
    if bst.feature_name()!=test['columns'].tolist() or test['X'].dtype!=np.float32 or not np.isfinite(test['X']).all():raise ValueError('Final schema/dtype mismatch')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if len(template)!=1260 or tuple(dict.fromkeys(test['users']))!=template:raise ValueError('Test user identity/order mismatch')
    scores=bst.predict(test['X'],num_threads=4)
    if not np.array_equal(scores,bst.predict(test['X'],num_threads=4)):raise ValueError('Nondeterministic predictions')
    pred=ranks(scores,test,template);original={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(ROOT/'submission/final_submission_v10_experimental.csv')}
    if all(pred[u][:5]==original[u] for u in template):raise ValueError('V11 must not be unchanged V10 renamed')
    pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
    for u,p in pred.items():
        top=p[:5]
        if len(top)!=5 or len(set(top))!=5 or not set(top)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in top):raise ValueError('Invalid final list')
    protected=read(OUT/'preservation_before.json')
    if any(sha(p)!=h for p,h in protected.items()):raise ValueError('Protected prior artifact changed')
    np.save(OUT/'final_scores.npy',scores,allow_pickle=False)
    with destination.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(pred[u][:5])) for u in template)
    dump('submission_manifest.json',{'version':'V11_experimental','selected':chosen,'selection_sha256':sha(OUT/'selection.json'),'model_path':str(path),'model_sha256':sha(path),
        'parameters':params,'rounds':78,'feature_names':bst.feature_name(),'feature_count':77,'feature_path':str(feature_path),'feature_cache_sha256':sha(feature_path),
        'training_rows':len(sample) if sample is not None else None,'training_indices_sha256':sha(OUT/'final_training_indices.npy') if sample is not None else None,
        'training':'Refit all4893 historical users, features beforeWeek21 and targetsWeeks21-30; all information throughWeek30 only. Final behavioral/PPR context throughWeek30. No supervision fromWeeks31-42.',
        'final_candidate_configuration':'Original verified five-generator Top50 union' if not expanded else 'Original union plus top50 new unseen positive PPR candidates',
        'expanded_candidates_used':expanded,'historical_user_denominator':denominator,'final_history':[1583704800,1595797200],
        'target_schedule_use':'Weeks31-42 metadata and availability only; no future user viewing','target_schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),
        'cold_start':'Ignore model scores for users with no qualifying full historical profile; popularity rank then numeric program ID. Same V10 fallback.',
        'users':1260,'eligible_ids':len(pool),'changed_lists_vs_V10':sum(pred[u][:5]!=original[u] for u in template),'submission_path':str(destination.relative_to(ROOT)),
        'submission_sha256':sha(destination),'development_MAP':selection['development'][chosen]['MAP5'],
        'Kaggle_private_MAP':None,'Kaggle_public_MAP':None,'automatic_upload':False,'best_scored_submission':'submission/final_submission_v10_experimental.csv',
        'best_scored_private_user_reported':.40637,'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'path':str(destination),'sha256':sha(destination),'selected':chosen,'changed_users':sum(pred[u][:5]!=original[u] for u in template),'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
