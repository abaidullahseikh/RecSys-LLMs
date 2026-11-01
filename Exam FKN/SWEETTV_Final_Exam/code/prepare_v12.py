"""Preserve V1--V11 and reproduce both scored baselines before any V12 fitting."""
import time
import numpy as np
import lightgbm as lgb
from v12_common import *
from verify_v11 import reference_rank,independent_ap
from verify_v10 import actual_flags

def main():
    t=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'baseline_reproduction.json').exists():raise FileExistsError('Preserve completed reproduction')
    if not (OUT/'preservation_before.json').exists():
        pp=[p for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents and not any(k in p.parts for k in ('.venv','.venv-lightgbm','__pycache__','.git')) and 'v12' not in p.name and p.suffix!='.pyc']+list(SOURCE.glob('*.csv'))
        dump('preservation_before.json',{str(p):sha(p) for p in pp})
    checks=[]
    def ck(n,ok):
        checks.append({'name':n,'passed':bool(ok)})
        if not ok:raise ValueError(n)
    baseline=read(V11/'baseline_reproduction.json');z,truth=data();allusers=tuple(dict.fromkeys(z['users']));dev=baseline['development_users'];ass=baseline['assessment_users'];dm=np.isin(z['users'],dev);fold=np.asarray([inner_fold(str(u)) for u in z['users']]);scores={n:np.zeros(len(z['users'])) for n in ('V10','V11')};times={n:0. for n in scores};models={n:[] for n in scores}
    ck('V11_verification_passed',read(V11/'submission_verification.json')['all_passed'] and read(V11/'validation_verification.json')['all_passed'])
    ck('identical_user_partition',dev==read(V10/'experiment.json')['development_users'] and ass==read(V10/'experiment.json')['assessment_users'] and not set(dev)&set(ass))
    ck('verified_expansion_config',read(V11/'retrieval_diagnosis.json')['chosen_generator']=='ppr' and read(V11/'retrieval_diagnosis.json')['limit_extra_unseen']==50)
    ck('native77feature_schema',z['columns'].tolist()==baseline['feature_names'] and z['X'].dtype==np.float32)
    params=read(V10/'experiment.json')['pairwise_info']['parameters'];prior11=read(V11/'hard_negative_experiment.json')['comparison']['E3']
    for n in scores:
        sm=read((V10 if n=='V10' else V11)/'submission_manifest.json');csvpath=ROOT/sm['submission_path'];ck(n+'_CSV_sha_matches_manifest',sha(csvpath)==sm['submission_sha256'])
        ck(n+'_final_model_hash',sha(Path(sm['model_path']) if Path(sm['model_path']).is_absolute() else ROOT/sm['model_path'])==sm['model_sha256'])
        for f in range(4):
            vi=np.flatnonzero((dm&(fold==f)) if f<3 else ~dm);vi=vi[z['original'][vi]] if n=='V11' else vi
            ti=np.flatnonzero(dm&(fold!=f)) if f<3 else np.flatnonzero(dm)
            path=V10/(f'pairwise_inner{f}.txt' if f<3 else 'pairwise_assessment.txt') if n=='V10' else V11/f'E3_inner{f}.txt'
            if n=='V11':
                sample=np.load(V11/f'E3_inner{f}_training_indices.npy',allow_pickle=False);ck(f'{n}_fold{f}_sample_training_scope',set(sample)<=set(ti) and np.all(z['original'][sample]))
            ck(f'{n}_fold{f}_no_supervised_user_leakage',not set(z['users'][ti])&set(z['users'][vi]) and not set(z['users'][ti])&set(ass))
            ck(f'{n}_fold{f}_group_sizes',grouping(z['users'][ti]).sum()==len(ti) and grouping(z['users'][vi]).sum()==len(vi))
            bst=lgb.Booster(model_file=str(path));ck(f'{n}_fold{f}_schema_config',bst.feature_name()==baseline['feature_names'] and bst.params['objective']=='lambdarank' and bst.current_iteration()==78 and bst.params['num_threads']<=4)
            ti0=time.monotonic();scores[n][vi]=bst.predict(z['X'][vi],num_threads=4);times[n]+=time.monotonic()-ti0;models[n].append({'path':str(path),'sha256':sha(path)})
        saved=np.load(V11/('expanded_V10_crossfit_scores.npy' if n=='V10' else 'E3_crossfit_scores.npy'),allow_pickle=False)
        ck(n+'_exact_saved_scores',np.array_equal(saved,scores[n]))
    result={};prof=history_profiles()
    for n in scores:
        mask=z['original'];rr=reference_rank(scores[n],z,allusers,mask);saved={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(V10/'selected_all_crossfit.csv' if n=='V10' else V11/'E3_all_crossfit.csv')}
        ck(n+'_exact_all_crossfit_lists',{u:r[:5] for u,r in rr.items()}==saved)
        result[n]={'development':metrics(rr,truth,dev,prof),'assessment':metrics(rr,truth,ass,prof),'models':models[n],'replay_inference_seconds':times[n]}
        for stage,uu in [('development',dev),('assessment',ass)]:
            expected=baseline[stage+'_MAP'] if n=='V10' else prior11[stage]['MAP5'];actual=sum(independent_ap(truth[u],rr[u]) for u in uu)/len(uu)
            ck(n+'_'+stage+'_MAP_reproduction',abs(actual-expected)<1e-14 and result[n][stage]['MAP5']==expected)
    tz=load(V9/'test_features.npz');traw=load(V8/'test.npz');flags,_,_,_=actual_flags(traw,'history');template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    for n in scores:
        sm=read((V10 if n=='V10' else V11)/'submission_manifest.json');mp=Path(sm['model_path']);mp=mp if mp.is_absolute() else ROOT/mp;bst=lgb.Booster(model_file=str(mp));s=bst.predict(tz['X'],num_threads=4)
        pred=reference_rank(s,tz,template);stored={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(ROOT/sm['submission_path'])}
        ck(n+'_all_final_lists_exact',{u:p[:5] for u,p in pred.items()}==stored);ck(n+'_final_cold_history_flag',np.array_equal(tz['X'][:,tz['columns'].tolist().index('user_cold_start')].astype(bool),flags['cold']))
    y,rebuilt,targetmeta=target_counts(z,'validation');ck('local_labels_independently_reconstructed',rebuilt==truth);ck('binary_label_alignment',np.array_equal(z['binary_label'],[str(i) in truth[str(u)] for u,i in zip(z['users'],z['items'])]))
    np.savez_compressed(OUT/'validation_targets.npz',counts=y,graded=grades(y),binary=z['binary_label']);np.save(OUT/'V10_expanded_scores.npy',scores['V10'],allow_pickle=False)
    original=load(V8/'validation.npz');flags,_,_,_=actual_flags(original,'train');ck('historical_boundaries_match_V10',all(np.array_equal(v,load(V10/'validation_flags.npz')[k]) for k,v in flags.items()))
    dump('baseline_reproduction.json',{'exact':True,'checks':checks,'comparison':result,'development_users':dev,'assessment_users':ass,'feature_names':z['columns'].tolist(),'parameters':params,
        'V11_E4_distinction':'V11 E4 retained positives +20hard+20sampled negatives per query. V12 E3/E4 fit every expanded row, without mining or sampling; E4 uses counts-based grades on all qualifying programs.',
        'target_provenance':targetmeta,'target_cache_sha256':sha(OUT/'validation_targets.npz'),'expanded_cache_path':str(V11/'validation.npz'),'expanded_cache_sha256':sha(V11/'validation.npz'),
        'training_history':[1583704800,1589749200],'target_window':[1589749200,1595797200],
        'Kaggle_private_user_reported':{'V5':.39706,'V8':.39908,'V10':.40637,'V11':.40634},'Kaggle_target_aspirational':.44801,'remaining_best_private_gap':.04164,
        'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'exact':True,'checks':len(checks),'comparison':{n:{s:result[n][s]['MAP5'] for s in ('development','assessment')} for n in result},'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
