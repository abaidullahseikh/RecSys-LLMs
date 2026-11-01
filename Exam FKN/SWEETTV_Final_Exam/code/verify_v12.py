"""Independent count/grade/metric replay, full-pool grouping and final CSV checks."""
import argparse,io,unittest,traceback,math
from v12_common import *
from verify_v11 import reference_rank,independent_metrics,independent_ap

def measured_reference(pred,truth,users,profiles):
    result=independent_metrics(pred,truth,users,profiles);d=Counter()
    for u in users:
        rel=set(truth[u])
        for i in rel:
            g='seen' if i in profiles.get(u,set()) else 'unseen';d[g+'_labels']+=1;d[g+'_hits5']+=i in pred[u][:5]
        for i in pred[u][:5]:
            g='seen' if i in profiles.get(u,set()) else 'unseen';d[g+'_slots']+=1;d[g+'_hits']+=i in rel
    for g in ('seen','unseen'):
        result[g+'_relevant_recall5']=d[g+'_hits5']/d[g+'_labels'] if d[g+'_labels'] else None
        result[g+'_top5_precision']=d[g+'_hits']/d[g+'_slots'] if d[g+'_slots'] else None
    return result

def independent_targets(z,phase):
    users=set(str(u) for u in z['users']);count=defaultdict(Counter)
    if phase=='validation':
        # Independent aggregated interaction-table source, instead of the fitting overlap-row scan.
        for r in records(validation.STEP2/'threshold_120s/validation_interactions.csv.gz'):
            if not 1589749200<=int(r['first_watch_utc'])<int(r['last_watch_utc'])<=1595797200:raise ValueError('Invalid future label window')
            n=int(r['qualifying_airings'])
            if n<1 or int(r['watched_seconds'])<120*n or int(r['tv_show_id'])<=0:raise ValueError('Invalid threshold/label aggregation')
            if r['user_id'] in users:count[r['user_id']][r['tv_show_id']]=n
    else:
        for r in records(validation.STEP2/'train_airing_overlaps.csv.gz'):
            if 1586725200<=int(r['first_overlap_utc'])<int(r['last_overlap_utc'])<=1589749200 and int(r['overlap_seconds'])>=120 and r['user_id'] in users:count[r['user_id']][r['tv_show_id']]+=1
    y=np.asarray([count[str(u)].get(str(i),0) for u,i in zip(z['users'],z['items'])],dtype=np.int32)
    # Separate interval assignment rather than invoking grades().
    g=np.zeros(len(y),dtype=np.int8);g[y==1]=1;g[(y>=2)&(y<=3)]=2;g[y>=4]=3
    truth={u:sorted(count[u],key=lambda i:(-count[u][i],int(i)))[:5] for u in users}
    return y,g,truth

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validation','submission']);a=parser.parse_args();start=time.monotonic();checks=[];details={};report=OUT/f'{a.phase}_verification.json'
    if report.exists():report=OUT/f'{a.phase}_verification_recheck_{time.time_ns()}.json'
    def ck(n,ok,detail=None):
        checks.append({'name':n,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(n)
    try:
        base=read(OUT/'baseline_reproduction.json');ex=read(OUT/'experiment.json');chrono=read(OUT/'chronological_validation.json');selection=read(OUT/'selection.json');dev=tuple(base['development_users']);ass=tuple(base['assessment_users'])
        ck('exact_V10_V11_reproduction',base['exact'] and all(x['passed'] for x in base['checks']))
        for version,folder in [('V10',V10),('V11',V11)]:
            sm=read(folder/'submission_manifest.json');ck(version+'_CSV_checksum_preserved',sha(ROOT/sm['submission_path'])==sm['submission_sha256'])
        # Check both time cutoffs, each model, every score, query group and reported metric.
        for phase in ('validation','early'):
            z,truth=data(phase);raw=load(V11/f'{phase}.npz');meta=read(V11/f'{phase}_build.json');names=raw['columns'].tolist();e9=read(V9/'experiment.json');p=names.index('ppr_score');s=names.index('pop_distinct_users')
            ref=np.column_stack([raw['X'][:,e9['base_V8_indices']],(raw['X'][:,p].astype(float)/np.sqrt((raw['X'][:,s].astype(float)+20)/meta['context']['historical_users'])).astype(np.float32)])
            ck(phase+'_independent_feature_transform',np.allclose(z['X'],ref,rtol=0,atol=1e-7) and z['columns'].tolist()==base['feature_names'] and z['X'].dtype==np.float32 and np.isfinite(z['X']).all())
            ck(phase+'_unique_nonzero_string_pairs',z['users'].dtype.kind in 'US' and z['items'].dtype.kind in 'US' and len(set(zip(z['users'],z['items'])))==len(z['users']) and all(int(i)>0 for i in z['items']))
            target=load(OUT/f'{phase}_targets.npz');y,g,indtruth=independent_targets(z,phase)
            ck(phase+'_independent_counts_grades_and_truth',np.array_equal(y,target['counts']) and np.array_equal(g,target['graded']) and indtruth==truth)
            ck(phase+'_binary_alignment',np.array_equal(target['binary'],z['binary_label']) and np.array_equal(target['binary'],[str(i) in truth[str(u)] for u,i in zip(z['users'],z['items'])]))
            ck(phase+'_labels_not_feature_columns',not any('future' in n or 'graded' in n or 'binary_label' in n for n in z['columns']))
            ck(phase+'_metadata_before_cutoff',meta['context']['metadata_vocabulary_fitted_before_cutoff'] and not meta['context']['future_viewing_used'] and meta['context']['history'][1]==meta['context']['target'][0])
            prof=history_profiles(phase);extra=np.flatnonzero(~z['original']);ck(phase+'_expansion_is_unseen_positive_PPR',all(str(z['items'][j]) not in prof[str(z['users'][j])] for j in extra) and np.all(z['X'][extra,z['columns'].tolist().index('ppr_score')]>0))
            newcounts=Counter(str(u) for u in z['users'][extra]);ck(phase+'_bounded_expansion',max(newcounts.values())<=50)
            users=tuple(dict.fromkeys(z['users']));du=dev if phase=='validation' else tuple(u for u in users if partition(u)!=0);au=ass if phase=='validation' else tuple(u for u in users if partition(u)==0);dm=np.isin(z['users'],du);fold=np.asarray([inner_fold(str(u)) for u in z['users']]);prefix='' if phase=='validation' else 'early_';er=ex if phase=='validation' else chrono;ranked={}
            for n in ('E3','E4'):
                scores=np.zeros(len(z['users']))
                for f in range(4 if phase=='validation' else 3):
                    ti=np.flatnonzero(dm&(fold!=f)) if f<3 else np.flatnonzero(dm);vi=np.flatnonzero(dm&(fold==f)) if f<3 else np.flatnonzero(~dm);m=er['comparison'][n]['models'][f]
                    ck(f'{phase}_{n}_fold{f}_user_separation',not set(z['users'][ti])&set(z['users'][vi]) and not set(z['users'][ti])&set(au) and set(m['fit_users'])==set(z['users'][ti]) and set(m['evaluation_users'])==set(z['users'][vi]))
                    # Full row coverage, no V11 sampled-negative distribution.
                    starts=np.r_[0,np.flatnonzero(z['users'][ti][1:]!=z['users'][ti][:-1])+1];gg=np.diff(np.r_[starts,len(ti)]).astype(np.int32)
                    ck(f'{phase}_{n}_fold{f}_full_pool_groups',len(set(z['users'][ti][starts]))==len(starts) and np.array_equal(gg,np.load(OUT/f'{prefix}inner{f}_groups.npy',allow_pickle=False)) and gg.sum()==m['training_rows']==len(ti))
                    path=Path(m['model_path']);bst=lgb.Booster(model_file=str(path));expected=parameters(n);actualgain=bst.params.get('label_gain');actualgain=[int(v) for v in actualgain.split(',')] if isinstance(actualgain,str) else list(actualgain)
                    ck(f'{phase}_{n}_fold{f}_model_schema_hash_gains',sha(path)==m['model_sha256'] and bst.feature_name()==base['feature_names'] and bst.current_iteration()==78 and bst.params['objective']=='lambdarank' and bst.params['num_threads']<=4 and actualgain==expected['label_gain'])
                    hist=Counter(str(int(v)) for v in target['binary' if n=='E3' else 'graded'][ti]);ck(f'{phase}_{n}_fold{f}_fit_label_histogram',hist==m['label_histogram'])
                    scores[vi]=bst.predict(z['X'][vi],num_threads=4)
                ck(phase+'_'+n+'_all_scores_replayed',np.array_equal(scores,np.load(OUT/f'{prefix}{n}_scores.npy',allow_pickle=False)))
                rr=reference_rank(scores,z,users if phase=='validation' else du);ranked[n]=rr;saved={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(OUT/f'{prefix}{n}_crossfit.csv')}
                ck(phase+'_'+n+'_all_Top5_lists_replayed',saved=={u:p[:5] for u,p in rr.items()})
                for stage,uu in [('development',du)]+([('assessment',au)] if phase=='validation' else []):
                    actual=measured_reference(rr,truth,uu,prof);stored=er['comparison'][n][stage]
                    ck(phase+'_'+n+'_'+stage+'_all_metrics_independent',all((v is None and stored[k] is None) or (v is not None and math.isclose(v,stored[k],rel_tol=0,abs_tol=1e-14)) for k,v in actual.items()))
                    # Evaluation universe includes eligible labels missing from retrieval;
                    # this does not add any ground-truth ID to a prediction/candidate row.
                    eligible=set(validation.EvaluationProtocol().candidates) if phase=='validation' else {str(i) for i in z['items']}|{i for labels in truth.values() for i in labels}
                    production=evaluation_metric(rr,truth,uu,eligible)
                    ck(phase+'_'+n+'_'+stage+'_production_MAP_parity',math.isclose(actual['MAP5'],production,rel_tol=0,abs_tol=1e-14))
            if phase=='validation':
                best=max(('E3','E4'),key=lambda n:(er['comparison'][n]['development']['MAP5'],n=='E3'));ck('development_only_new_model_selection',best==selection['selected'] and selection['frozen_before_new_assessment_fits'])
        ck('chronological_check_kept_selection_frozen',sha(OUT/'selection.json')==chrono['selection_frozen_sha256'])
        # Final expanded feature cache is required even for validation-stage correctness gates.
        tz,_=data('test');raw=load(OUT/'test.npz');tm=read(OUT/'test_build.json');cached=load(OUT/'test_model_features.npz');old=load(V8/'test.npz');template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
        ck('final_expanded_feature_cache_integrity',sha(OUT/'test_model_features.npz')==tm['model_feature_sha256'] and np.array_equal(tz['X'],cached['X']) and tz['columns'].tolist()==base['feature_names'])
        ck('final_original_candidates_preserved',set(zip(old['users'],old['items']))==set(zip(raw['users'][raw['original']],raw['items'][raw['original']])))
        ck('final_exact_template_groups',tuple(dict.fromkeys(raw['users']))==template and len(template)==1260 and grouping(raw['users']).sum()==len(raw['users']))
        ck('final_duplicate_free_positive_eligible_candidates',len(set(zip(raw['users'],raw['items'])))==len(raw['users']) and set(raw['items'])<=set(final_pool()) and all(int(i)>0 for i in raw['items']))
        fullprof=history_profiles('test');extras=np.flatnonzero(~raw['original']);nc=Counter(str(u) for u in raw['users'][extras]);ck('final_extra_unseen_positive_PPR_max50',max(nc.values())<=50 and all(str(raw['items'][j]) not in fullprof[str(raw['users'][j])] for j in extras) and np.all(tz['X'][extras,tz['columns'].tolist().index('ppr_score')]>0))
        ck('final_behavior_cutoff_and_no_target_viewing',tm['context']['history']==[1583704800,1595797200] and not tm['target_behavior_read'] and tm['context']['target']==list(validation.WINDOWS['test']))
        ck('final_cold_flags_independent',np.array_equal(tz['X'][:,tz['columns'].tolist().index('user_cold_start')].astype(bool),[not fullprof[str(u)] for u in tz['users']]))
        if a.phase=='validation':
            # Fresh full-history generator/feature replay for deterministic final users.
            from v11_retrieval import context,signals,additional,feature
            sampleusers=(template[0],template[len(template)//2],template[-1]);c=context(1583704800,1595797200,validation.WINDOWS['test'][1],'history',sampleusers,tuple(sorted(final_pool(),key=int)));index=defaultdict(dict)
            for j,(u,i) in enumerate(zip(raw['users'],raw['items'])):
                if str(u) in sampleusers:index[str(u)][str(i)]=j
            for un,u in enumerate(sampleusers):
                rs,rr,_=signals(c,u,un);s2,r2,_=signals(c,u,un);oldset={i for i,j in index[u].items() if raw['original'][j]};newset={i for i,j in index[u].items() if not raw['original'][j]};expected=set(additional(rr['ppr'],rs['ppr'],c['h']['data'].profiles.get(u,{}),oldset,50))
                ck(u+'_final_candidate_repeat_replay',rs==s2 and rr==r2 and newset==expected);pos={g:{i:k for k,i in enumerate(v)} for g,v in rr.items()}
                ck(u+'_all_final_features_replayed',all(np.array_equal(feature(c,u,i,rs,pos,raw['columns'].tolist()),raw['X'][j]) for i,j in index[u].items()))
            modules=('test_data_pipeline','test_step3','test_step4','test_step5','test_step6','test_step8','test_content_based','test_lightgbm_v4','test_v5','test_v6','test_v6_verification','test_v7','test_v8','test_v9','test_v10','test_v11','test_v12')
            suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules);log=io.StringIO();test=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            lp=OUT/'regression_tests.txt'
            if lp.exists():lp=OUT/f'regression_tests_recheck_{time.time_ns()}.txt'
            with lp.open('x') as f:f.write(log.getvalue())
            details['regression_tests']={'tests_run':test.testsRun,'failures':len(test.failures),'errors':len(test.errors)};ck('all_existing_and_V12_regressions',test.wasSuccessful(),details['regression_tests'])
        if a.phase=='submission':
            sm=read(OUT/'submission_manifest.json');ck('validation_gate_passed',read(OUT/'validation_verification.json')['all_passed']);bst=lgb.Booster(model_file=str(OUT/'final_model.txt'))
            ck('final_model_hash_schema_gain_config',sha(OUT/'final_model.txt')==sm['model_sha256'] and bst.feature_name()==base['feature_names'] and bst.params['objective']=='lambdarank' and bst.params['num_threads']<=4 and bst.current_iteration()==78)
            mainz,_=data();finalgroups=np.load(OUT/'final_groups.npy',allow_pickle=False);yy=load(OUT/'validation_targets.npz')['binary' if selection['selected']=='E3' else 'graded']
            ck('final_refit_every_expanded_row_group_and_target',sm['training_rows']==len(mainz['users']) and np.array_equal(finalgroups,grouping(mainz['users'])) and finalgroups.sum()==len(yy) and sha(OUT/'final_groups.npy')==sm['final_groups_sha256'] and Counter(str(int(v)) for v in yy)==sm['training_label_histogram'])
            gain=bst.params['label_gain'];gain=[int(v) for v in gain.split(',')] if isinstance(gain,str) else list(gain)
            ck('final_selected_label_gains_match',gain==parameters(selection['selected'])['label_gain'] and sm['parameters']==parameters(selection['selected']))
            score=bst.predict(tz['X'],num_threads=4);ck('final_deterministic_scores',np.array_equal(score,bst.predict(tz['X'],num_threads=4)) and np.array_equal(score,np.load(OUT/'final_scores.npy',allow_pickle=False)))
            expected=reference_rank(score,tz,template);destination=ROOT/sm['submission_path']
            with destination.open(newline='',encoding='utf-8') as f:reader=csv.reader(f);header=next(reader);rows=list(reader)
            ck('exact_Kaggle_header_and_row_width',header==['user_id','tv_show_id'] and all(len(r)==2 for r in rows))
            ck('1260_unique_original_integer_IDs_exact_order',len(rows)==1260 and len({r[0] for r in rows})==1260 and tuple(r[0] for r in rows)==template and all(r[0].isascii() and r[0].isdecimal() for r in rows))
            pred={r[0]:tuple(r[1].split(' ')) for r in rows};pool=final_pool();ck('five_distinct_canonical_positive_scheduled_IDs',all(len(p)==5 and len(set(p))==5 and set(p)<=pool and all(i.isascii() and i.isdecimal() and not i.startswith('0') for i in p) for p in pred.values()))
            ck('all1260_lists_independently_replayed',pred=={u:r[:5] for u,r in expected.items()});ck('final_selected_new_model',sm['selected']==selection['selected'] and sm['selected'] in ('E3','E4') and sm['training_rows']==len(mainz['users']))
            index={(str(u),str(i)):j for j,(u,i) in enumerate(zip(tz['users'],tz['items']))};missing=tz['columns'].tolist().index('missing_target_availability')
            ck('all_final_recommendations_have_timestamp_clipped_availability',all(tz['X'][index[u,i],missing]==0 for u,p in pred.items() for i in p))
            ck('genuine_V12_not_previous_file_renamed',all(sha(destination)!=read(folder/'submission_manifest.json')['submission_sha256'] for folder in (V10,V11)))
            ck('final_CSV_sha256',sha(destination)==sm['submission_sha256']);details['submission_sha256']=sha(destination);details['submission_path']=str(destination)
            checksum=OUT/'submission_sha256.txt'
            if checksum.exists():
                if checksum.read_text().strip()!=sha(destination):raise ValueError('Recorded SHA changed')
            else:
                with checksum.open('x') as f:f.write(sha(destination)+'\n')
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];ck('all_V1_V11_artifacts_and_exports_unchanged',not changed,changed)
        details['protected_files']=len(protected);details['protected_session_exports']=[p for p in protected if 'session' in p.lower() and p.endswith('.json')];details['selected']=selection['selected']
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':str(error)})
        details['traceback']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':details,'seconds':time.monotonic()-start,'peak_rss_mib':rss(),
        'scope':'Independent count/grade source and MAP/recall/precision/group checks; every new model replayed at both cutoffs. Fresh sampled final candidates/features replay shares audited generator functions. Original viewing join is protected by hashes and tested, not independently reimplemented.'}
    with report.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print({k:v for k,v in result.items() if k!='checks'},flush=True)
    if not result['all_passed']:raise SystemExit(1)

def final_pool():return {str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}

def evaluation_metric(r,truth,users,pool):
    import evaluation
    users=tuple(str(u) for u in users);pool=tuple(str(i) for i in pool)
    return evaluation.evaluate_predictions({u:r[u][:5] for u in users},{u:truth[u] for u in users},users,pool)['primary_map_at_5']

if __name__=='__main__':main()
