"""Independent V11 label/metric/sampling/model/CSV replay and preservation checks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from collections import defaultdict,Counter
from fractions import Fraction
from pathlib import Path
import csv,json,hashlib,math,time,traceback,argparse,io,unittest
import numpy as np
import lightgbm as lgb
import validation,evaluation
from v8_common import ROOT,SOURCE,read,load,sha,records,inner_fold,partition
from prepare_v11 import OUT,V8,V9,V10

def independent_ap(labels,ranking):
    labels=set(labels);ranking=list(ranking[:5]);value=Fraction(0)
    for i in labels:
        if i in ranking:
            k=ranking.index(i)+1;value+=Fraction(len(labels&set(ranking[:k])),k)
    return float(value/min(5,len(labels))) if labels else 0.

def reference_rank(scores,z,users,mask=None):
    names=z['columns'].tolist();pi=names.index('popularity_rank');ci=names.index('user_cold_start');groups=defaultdict(list)
    for j,u in enumerate(z['users']):
        if mask is None or mask[j]:groups[str(u)].append(j)
    result={}
    for u in users:
        cold=bool(z['X'][groups[u][0],ci]);order=sorted(groups[u],key=lambda n:(0 if cold else -float(scores[n]),float(z['X'][n,pi]),int(z['items'][n])))
        result[u]=tuple(str(z['items'][n]) for n in order)
    return result

def reference_sample(z,ix,scores):
    groups=defaultdict(list);pi=z['columns'].tolist().index('popularity_rank');out=[]
    for j in ix:groups[str(z['users'][j])].append(int(j))
    for u,js in groups.items():
        pos=[j for j in js if z['binary_label'][j]];neg=sorted((j for j in js if not z['binary_label'][j]),key=lambda j:(-float(scores[j]),float(z['X'][j,pi]),int(z['items'][j])))
        out.extend(sorted(pos+neg[:20]+sorted(neg[20:],key=lambda j:hashlib.sha256(f'2026:v11-negative:{u}:{z["items"][j]}'.encode()).digest())[:20]))
    return np.asarray(out,dtype=np.int64)

def independent_metrics(r,truth,users,prof):
    d=Counter()
    for u in users:
        rel=set(truth[u])
        for i in rel:
            flag='seen' if i in prof.get(u,set()) else 'unseen';d[flag+'_labels']+=1;d[flag+'_candidate_hits']+=i in r[u]
        for k in (5,10,20):d[f'hits{k}']+=len(set(r[u][:k])&rel)
    den=d['seen_labels']+d['unseen_labels']
    return {'MAP5':math.fsum(independent_ap(truth[u],r[u]) for u in users)/len(users),'candidate_recall':sum(d[g+'_candidate_hits'] for g in ('seen','unseen'))/den,
        **{g+'_candidate_recall':d[g+'_candidate_hits']/d[g+'_labels'] if d[g+'_labels'] else None for g in ('seen','unseen')},
        **{f'Recall{k}':d[f'hits{k}']/den for k in (5,10,20)},'average_candidates':sum(len(r[u]) for u in users)/len(users)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validation','submission']);a=parser.parse_args();t=time.monotonic();checks=[];details={}
    path=OUT/f'{a.phase}_verification.json'
    if path.exists():path=OUT/f'{a.phase}_verification_recheck_{time.time_ns()}.json'
    def ck(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(name)
    try:
        baseline=read(OUT/'baseline_reproduction.json');raw=load(OUT/'validation.npz');meta=read(OUT/'validation_build.json');e9=read(V9/'experiment.json');original=load(V8/'validation.npz');users=tuple(dict.fromkeys(raw['users']));dev=baseline['development_users'];ass=baseline['assessment_users'];truth=read(V8/'labels.json')['secondary']
        ck('V10_reproduction_and_checksum',baseline['exact'] and all(c['passed'] for c in baseline['checks']) and sha(ROOT/'submission/final_submission_v10_experimental.csv')==baseline['submission_sha256'])
        ck('expanded_cache_hash_and_dtype',sha(OUT/'validation.npz')==meta['data_sha256'] and raw['X'].dtype==np.float32 and np.isfinite(raw['X']).all())
        ck('all_string_unique_positive_ID_pairs',raw['users'].dtype.kind in 'US' and raw['items'].dtype.kind in 'US' and len(set(zip(raw['users'],raw['items'])))==len(raw['users']) and all(int(i)>0 for i in raw['items']))
        ck('eligible_candidates',set(raw['items'])<=set(validation.EvaluationProtocol().candidates))
        oldmap={(str(u),str(i)):j for j,(u,i) in enumerate(zip(original['users'],original['items']))};oldix=np.flatnonzero(raw['original']);correspond=np.asarray([oldmap[str(raw['users'][j]),str(raw['items'][j])] for j in oldix])
        ck('every_V10_candidate_and_feature_preserved_exactly',len(oldix)==len(original['users']) and len(set(correspond))==len(oldix) and np.array_equal(raw['X'][oldix],original['X'][correspond]))
        # Independent target reconstruction from matched held-out airings.
        count={u:Counter() for u in users};prof=defaultdict(set)
        for r in records(validation.STEP2/'validation_airing_overlaps.csv.gz'):
            f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc']);s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
            if not 1589749200<=f<l<=1595797200 or not 0<s<=d or int(r['tv_show_id'])<=0:raise ValueError('Invalid target overlap boundary/join')
            if s>=120 and r['user_id'] in count:count[r['user_id']][r['tv_show_id']]+=1
        computed={u:sorted(v,key=lambda i:(-v[i],int(i)))[:5] for u,v in count.items()}
        ck('all_primary_relevance_labels_reconstructed',computed==truth)
        for r in records(validation.STEP2/'threshold_120s/train_interactions.csv.gz'):
            if not 1583704800<=int(r['first_watch_utc'])<int(r['last_watch_utc'])<=1589749200:raise ValueError('Future historical viewing')
            prof[r['user_id']].add(r['tv_show_id'])
        ck('new_binary_labels_exact_alignment',np.array_equal(raw['binary_label'],[str(i) in truth[str(u)] for u,i in zip(raw['users'],raw['items'])]))
        ck('all_added_candidates_are_historically_unseen',all(str(raw['items'][j]) not in prof[str(raw['users'][j])] for j in np.flatnonzero(~raw['original'])))
        counts=Counter(str(u) for u in raw['users'][~raw['original']]);ck('maximum50_added_per_user',max(counts.values())<=50)
        names=raw['columns'].tolist();p=names.index('ppr_score');s=names.index('pop_distinct_users');den=meta['context']['historical_users']
        M=np.column_stack([raw['X'][:,e9['base_V8_indices']],(raw['X'][:,p].astype(float)/np.sqrt((raw['X'][:,s].astype(float)+20.)/den)).astype(np.float32)])
        z={**raw,'X':M,'columns':np.asarray(e9['feature_names'])};ck('native_V10_feature_schema_count',M.shape[1]==77 and z['columns'].tolist()==baseline['feature_names'])
        ck('historical_context_and_vocabulary_boundary',meta['context']['history']==[1583704800,1589749200] and meta['context']['metadata_vocabulary_fitted_before_cutoff'] and not meta['context']['future_viewing_used'])
        dm=np.isin(z['users'],dev);folds=np.asarray([inner_fold(str(u)) for u in z['users']]);bscore=np.zeros(len(M))
        for f in range(4):
            vi=np.flatnonzero(dm&(folds==f)) if f<3 else np.flatnonzero(~dm);model=lgb.Booster(model_file=str(V10/(f'pairwise_inner{f}.txt' if f<3 else 'pairwise_assessment.txt')))
            bscore[vi]=model.predict(M[vi],num_threads=4)
        ck('expanded_V10_scores_reproduced',np.array_equal(bscore,np.load(OUT/'expanded_V10_crossfit_scores.npy',allow_pickle=False)))
        result={**read(OUT/'E1_E2.json')['comparison'],**read(OUT/'hard_negative_experiment.json')['comparison']};mining=read(OUT/'hard_negative_experiment.json')['mining'];allr={}
        for f in range(4):
            fitusers=set(u for u in dev if inner_fold(u)!=f) if f<3 else set(dev);vu=set(u for u in dev if inner_fold(u)==f) if f<3 else set(ass);ti=np.flatnonzero(np.isin(z['users'],list(fitusers)));vi=np.flatnonzero(np.isin(z['users'],list(vu)))
            ck(f'outer{f}_fit_evaluation_assessment_disjoint',not fitusers&vu and not fitusers&set(ass))
            if f<3:
                mined=np.full(len(M),np.nan)
                for k,mm in enumerate(mining[f]['metadata']):
                    fu=set(mm['fit_users']);su=set(mm['score_users']);ck(f'outer{f}_miner{k}_no_target_contamination',not fu&su and fu|su==fitusers and not (fu|su)&vu)
                    pathm=OUT/f'outer{f}_miner{k}.txt';ck(f'outer{f}_miner{k}_hash',sha(pathm)==mm['model_sha256']);si=np.flatnonzero(np.isin(z['users'],list(su)));model=lgb.Booster(model_file=str(pathm));mined[si]=model.predict(M[si],num_threads=4)
                if not np.isfinite(mined[ti]).all():raise ValueError('Incomplete miner scores')
            else:mined=bscore
            for n in ('E3','E4'):
                ii=ti[z['original'][ti]] if n=='E3' else ti;actual=np.load(OUT/f'{n}_inner{f}_training_indices.npy',allow_pickle=False);expected=reference_sample(z,ii,mined)
                ck(f'{n}_outer{f}_independent_negative_sample',np.array_equal(actual,expected) and set(actual)<=set(ii) and set(ii[z['binary_label'][ii]>0])<=set(actual))
                modelpath=OUT/f'{n}_inner{f}.txt';model=lgb.Booster(model_file=str(modelpath));entry=result[n]['models'][f]
                ck(f'{n}_outer{f}_model_hash_schema_config',sha(modelpath)==entry['model_sha256'] and model.feature_name()==z['columns'].tolist() and model.current_iteration()==78 and model.params['objective']=='lambdarank' and model.params['num_threads']<=4)
        for n in ('E1','E2','E3','E4'):
            scores=bscore.copy() if n in ('E1','E2') else np.zeros(len(M))
            if n in ('E3','E4'):
                for f in range(4):
                    vi=np.flatnonzero(dm&(folds==f)) if f<3 else np.flatnonzero(~dm)
                    if n=='E3':vi=vi[z['original'][vi]]
                    scores[vi]=lgb.Booster(model_file=str(OUT/f'{n}_inner{f}.txt')).predict(M[vi],num_threads=4)
                ck(n+'_all_crossfit_scores',np.array_equal(scores,np.load(OUT/f'{n}_crossfit_scores.npy',allow_pickle=False)))
            ranking=reference_rank(scores,z,users,z['original'] if n in ('E1','E3') else None);allr[n]=ranking
            saved={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(OUT/f'{n}_all_crossfit.csv')}
            ck(n+'_all_rankings_replayed',saved=={u:v[:5] for u,v in ranking.items()})
            for stage,uu in [('development',dev),('assessment',ass)]:
                actual=independent_metrics(ranking,truth,uu,prof)
                ck(n+'_'+stage+'_metrics_independent',all((v is None and result[n][stage][k] is None) or (v is not None and math.isclose(v,result[n][stage][k],rel_tol=0,abs_tol=1e-14)) for k,v in actual.items()))
                production=evaluation.evaluate_predictions({u:ranking[u][:5] for u in uu},{u:truth[u] for u in uu},uu,validation.EvaluationProtocol().candidates)['primary_map_at_5']
                ck(n+'_'+stage+'_production_MAP_parity',math.isclose(actual['MAP5'],production,rel_tol=0,abs_tol=1e-14))
        selection=read(OUT/'selection.json');tie={'E3':2,'E2':1,'E4':0};chosen=max(('E2','E3','E4'),key=lambda n:(result[n]['development']['MAP5'],tie[n]))
        ck('development_only_selection_recomputed',selection['selected']==chosen and selection['frozen_before_assessment_scoring'])
        early=load(OUT/'early.npz');earlytruth=read(OUT/'early_labels.json');earlymeta=read(OUT/'early_build.json');cut=earlymeta['context']['history'][1];eusers=tuple(dict.fromkeys(early['users']));edev=tuple(u for u in eusers if partition(u)!=0);ecount={u:Counter() for u in eusers};eprof=defaultdict(set)
        for r in records(validation.STEP2/'train_airing_overlaps.csv.gz'):
            f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
            if int(r['overlap_seconds'])<120:continue
            if 1583704800<=f<l<=cut:eprof[r['user_id']].add(r['tv_show_id'])
            if cut<=f<l<=1589749200 and r['user_id'] in ecount:ecount[r['user_id']][r['tv_show_id']]+=1
        ck('earlier_targets_strictly_after_history',earlytruth=={u:sorted(v,key=lambda i:(-v[i],int(i)))[:5] for u,v in ecount.items()} and earlymeta['context']['history']==[1583704800,1586725200])
        ep=early['columns'].tolist().index('ppr_score');es=early['columns'].tolist().index('pop_distinct_users');em=np.column_stack([early['X'][:,e9['base_V8_indices']],(early['X'][:,ep].astype(float)*np.sqrt(earlymeta['context']['historical_users']/(early['X'][:,es].astype(float)+20))).astype(np.float32)])
        ez={**early,'X':em,'columns':np.asarray(e9['feature_names'])};chrono=read(OUT/'chronological_validation.json');edm=np.isin(ez['users'],edev);efold=np.asarray([inner_fold(str(u)) for u in ez['users']])
        for n in ('E1','E2','E3','E4'):
            ss=np.zeros(len(em))
            for f in range(3):
                ix=np.flatnonzero(edm&(efold==f));pathm=OUT/f'early_{"E1" if n=="E2" else n}_inner{f}.txt';ss[ix]=lgb.Booster(model_file=str(pathm)).predict(em[ix],num_threads=4)
            er=reference_rank(ss,ez,edev,ez['original'] if n in ('E1','E3') else None);mm=independent_metrics(er,earlytruth,edev,eprof)
            ck('early_'+n+'_model_metrics_replayed',all(v is None or math.isclose(v,chrono['early']['comparison'][n][k],rel_tol=0,abs_tol=1e-14) for k,v in mm.items()))
        for fm in chrono['early']['models']:
            fu=set(fm['fit_users']);vu=set(fm['evaluation_users']);ck('early_outer'+str(fm['fold'])+'_no_future_user_label_training',not fu&vu and all(partition(u)!=0 for u in fu|vu) and all(not set(m['fit_users'])&set(m['score_users']) and set(m['fit_users'])|set(m['score_users'])==fu for m in fm['mining']))
        ck('chronological_check_did_not_retune_selection',chrono['selection_sha256']==sha(OUT/'selection.json'))
        # Replay fresh candidate generation and every feature for three deterministic development users.
        if a.phase=='validation':
            from v11_retrieval import context,signals,additional,feature
            sampleusers=tuple(sorted(dev,key=lambda u:hashlib.sha256(('v11-audit:'+u).encode()).digest())[:3]);pool=validation.EvaluationProtocol().candidates;c=context(1583704800,1589749200,1595797200,'train',sampleusers,pool);index=defaultdict(dict)
            for j,(u,i) in enumerate(zip(raw['users'],raw['items'])):
                if str(u) in sampleusers:index[str(u)][str(i)]=j
            for un,u in enumerate(sampleusers):
                rs,rr,_=signals(c,u,un);rs2,rr2,_=signals(c,u,un);old={i for i,j in index[u].items() if raw['original'][j]};expected=set(additional(rr['ppr'],rs['ppr'],c['h']['data'].profiles.get(u,{}),old));actual={i for i,j in index[u].items() if not raw['original'][j]}
                ck(u+'_deterministic_full_generator_replay',rs==rs2 and rr==rr2 and expected==actual)
                pos={g:{i:k for k,i in enumerate(v)} for g,v in rr.items()}
                ck(u+'_all_extra_features_reconstructed',all(np.array_equal(feature(c,u,i,rs,pos,raw['columns'].tolist()),raw['X'][index[u][i]]) for i in actual))
            modules=('test_data_pipeline','test_step3','test_step4','test_step5','test_step6','test_step8','test_content_based','test_lightgbm_v4','test_v5','test_v6','test_v6_verification','test_v7','test_v8','test_v9','test_v10','test_v11')
            suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules);log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            with (OUT/'regression_tests.txt').open('x') as f:f.write(log.getvalue())
            details['regression_tests']={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)};ck('all_relevant_regression_tests',r.wasSuccessful(),details['regression_tests'])
        if a.phase=='submission':
            sm=read(OUT/'submission_manifest.json');ck('validation_prechecks_passed',read(OUT/'validation_verification.json')['all_passed'])
            test=load(Path(sm['feature_path']));ck('test_cache_hash_and_schema',sha(Path(sm['feature_path']))==sm['feature_cache_sha256'] and test['columns'].tolist()==e9['feature_names'])
            bst=lgb.Booster(model_file=sm['model_path']);ck('final_model_hash_parameters_schema',sha(sm['model_path'])==sm['model_sha256'] and bst.feature_name()==test['columns'].tolist() and bst.params['objective']=='lambdarank' and bst.params['num_threads']<=4 and bst.current_iteration()==78)
            if chosen in ('E3','E4'):
                ix=np.flatnonzero(z['original']) if chosen=='E3' else np.arange(len(z['users']));sample=np.load(OUT/'final_training_indices.npy',allow_pickle=False)
                ck('final_refit_sample_independent',np.array_equal(sample,reference_sample(z,ix,bscore)) and len(sample)==sm['training_rows'] and sha(OUT/'final_training_indices.npy')==sm['training_indices_sha256'])
            template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'));score=bst.predict(test['X'],num_threads=4);ck('deterministic_final_inference',np.array_equal(score,bst.predict(test['X'],num_threads=4)) and np.array_equal(score,np.load(OUT/'final_scores.npy',allow_pickle=False)))
            expected=reference_rank(score,test,template);submission=ROOT/sm['submission_path']
            with submission.open(newline='',encoding='utf-8') as f:
                rr=csv.reader(f);header=next(rr);data=list(rr)
            ck('exact_submission_schema',header==['user_id','tv_show_id'] and all(len(r)==2 for r in data))
            ck('exact1260_original_users_order',len(data)==1260 and len({r[0] for r in data})==1260 and tuple(r[0] for r in data)==template)
            pool={str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0};pred={r[0]:tuple(r[1].split(' ')) for r in data}
            ck('five_distinct_positive_canonical_eligible_programs',all(len(p)==5 and len(set(p))==5 and set(p)<=pool and all(i.isascii() and i.isdecimal() and not i.startswith('0') for i in p) for p in pred.values()))
            ck('all1260_final_lists_replayed',pred=={u:r[:5] for u,r in expected.items()})
            ck('genuine_new_configuration_and_not_V10_renamed',chosen!='E1' and any(pred[r['user_id']]!=tuple(r['tv_show_id'].split()) for r in records(ROOT/'submission/final_submission_v10_experimental.csv')))
            ck('final_history_and_schedule_provenance',sm['final_history']==[1583704800,1595797200] and sm['target_schedule_sha256']==sha(SOURCE/'export_arh_31-42-final.csv') and not sm['automatic_upload'])
            ck('final_CSV_SHA256',sha(submission)==sm['submission_sha256']);details['submission_sha256']=sha(submission);details['submission_path']=str(submission)
            checksum=OUT/'submission_sha256.txt'
            if checksum.exists():
                if checksum.read_text().strip()!=sha(submission):raise ValueError('Recorded checksum differs')
            else:
                with checksum.open('x') as f:f.write(sha(submission)+'\n')
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
        ck('all_V1_V10_data_models_code_exports_unchanged',not changed,changed)
        details['protected_files']=len(protected);details['protected_session_exports']=[p for p in protected if 'session' in p.lower() and p.endswith('.json')]
        details['selected']=chosen
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':str(error)})
        details['traceback']=traceback.format_exc()
    report={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':details,
        'seconds':time.monotonic()-t,'peak_rss_mib':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024,
        'scope':'Independent labels/MAP/recall/negative samples, all main model scores and earlier-fold metrics replayed. Fresh sampled generator/features replay uses shared historical generator implementations, not an independent reimplementation. Prior input/join behavior covered by preservation hashes and regression tests.'}
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print({k:v for k,v in report.items() if k!='checks'},flush=True)
    if not report['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
