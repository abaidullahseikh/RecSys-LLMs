"""Read-only method replay and strict-temporal, evaluation-only V10 examples.

Writes new Requirement7 evidence only; never modifies historical artifacts.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import csv,json,math,time,resource
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix
import validation,evaluation,ppr,association_rules as ar,content_based as cbf
from run_cbf_step9 import source_schedule_metadata
from train_v9 import new_matrix,rankings
from run_v7 import grouping
from v8_common import ROOT,SOURCE,read,load,sha,records,partition,inner_fold

USERS=('13571199046623737884','8150816946177670084')
OUT=ROOT/'results/final_requirement7_evidence'
METHODS=('Popularity','Personal frequency','Personalized PageRank','Association Rules','Content-Based Filtering','V10 configuration — temporal evaluation proxy')
TRAIN=(1583704800,1589749200);VALID=(1589749200,1595797200);EARLY=1586725200

def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')

def predictions(path):
    return {r['user_id']:tuple(r['tv_show_id'].split()) for r in records(path) if r['user_id'] in USERS}

def ap_reference(truth,pred):
    truth=set(truth);hits=0;value=Fraction(0)
    for rank,item in enumerate(pred[:5],1):
        if item in truth:hits+=1;value+=Fraction(hits,rank)
    return float(value/min(5,len(truth))) if truth else 0.

def metadata_for(ids):
    """Raw first-in-file eligible validation airing; keep variants and row provenance."""
    result={};variants=defaultdict(set)
    for line,r in enumerate(records(SOURCE/'export_arh_21-30-final.csv'),2):
        i=str(int(r['tv_show_id']))
        if i not in ids or int(r['duration'])<=0:continue
        stamp=validation.data_pipeline.parse_timestamp(r['start_time'],schedule=True) if hasattr(validation,'data_pipeline') else __import__('data_pipeline').parse_timestamp(r['start_time'],schedule=True)
        if not stamp<VALID[1] or stamp+int(r['duration'])<=VALID[0]:continue
        variants[i].add((r['tv_show_title'],r['tv_show_category']))
        if i not in result:result[i]={'title':r['tv_show_title'],'category':r['tv_show_category'],'schedule_row':line,'airing_start':r['start_time'],'airing_start_utc':stamp}
    for i in ids:
        result.setdefault(i,{'title':'[metadata unavailable in validation schedule]','category':'[metadata unavailable in validation schedule]','schedule_row':None})
        result[i]['metadata_complete']=bool(result[i]['schedule_row'] and result[i]['title'].strip() and result[i]['category'].strip())
        result[i]['variant_count']=len(variants[i]);result[i]['title_category_variants']=[{'title':t,'category':c} for t,c in sorted(variants[i])]
    return result

def main():
    t=time.monotonic()
    if OUT.exists() or (ROOT/'results/final_requirement7_examples.md').exists():raise FileExistsError('Preserve existing Requirement7 evidence')
    OUT.mkdir();protected=[p for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents and not any(x in p.parts for x in ('.venv','.venv-lightgbm','__pycache__','.git')) and p!=Path(__file__).resolve()]+list(SOURCE.glob('*.csv'))
    snapshot={str(p):sha(p) for p in protected};dump('preservation_before.json',snapshot)
    checks=[]
    def ck(n,ok,detail=None):
        checks.append({'name':n,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(n)
    protocol=validation.EvaluationProtocol();config=protocol.config;pool=tuple(protocol.candidates);poolset=set(pool);oldusers=read(ROOT/'results/step4/provisional_two_users.json')
    ck('exact_fixed_user_strings',USERS==(oldusers['heavy_user_id'],oldusers['light_user_id']) and all(type(u)is str for u in USERS))
    ck('frozen_protocol_windows_and_primary_truth',config['windows_utc']['train']==list(TRAIN) and config['windows_utc']['validation']==list(VALID) and config['policies']['ground_truth_top5_truncation'] is False)
    v10=ROOT/'results/v10_experiment';sm=read(v10/'submission_manifest.json');mp=ROOT/sm['model_path']
    ck('designated_production_V10_model_and_CSV_identity',sha(mp)==sm['model_sha256'] and sha(ROOT/sm['submission_path'])==sm['submission_sha256']=='39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e')
    data=ppr.build_training_profiles(config);air=Counter();seconds=Counter()
    for profile in data.profiles.values():
        for i,(a,s,last) in profile.items():air[i]+=a;seconds[i]+=s
    fallback=tuple(sorted(pool,key=lambda i:(-data.distinct_viewers.get(i,0),-air[i],int(i))))
    ck('actual_saved_training_only_popularity_order',fallback==tuple(read(ROOT/'results/step4/step4_run.json')['fallback_ranking']))
    sizes={u:len(data.profiles.get(u,{})) for u in protocol.cohort};warm=[u for u in protocol.cohort if sizes[u]>0]
    ck('heavy_light_selection_from_history_only',max(warm,key=lambda u:sizes[u])==USERS[0] and min((u for u in warm if u!=USERS[0]),key=lambda u:sizes[u])==USERS[1])
    stats={u:{'distinct_programs':sizes[u],'qualifying_airings':sum(x[0] for x in data.profiles[u].values()),'watched_seconds':sum(x[1] for x in data.profiles[u].values()),
        'eligible_history_programs':sum(i in poolset for i in data.profiles[u]),'first_qualifying_overlap_utc':None,'last_qualifying_overlap_utc':max(v[2] for v in data.profiles[u].values())} for u in USERS}
    # Independently reconstruct selected-user aggregates and all-user14day supports from matched historical airings.
    baskets=defaultdict(set);independent={u:defaultdict(lambda:[0,0,0]) for u in USERS};earlycount=defaultdict(Counter);selected_weeks=defaultdict(set);channels=defaultdict(set)
    for r in records(validation.STEP2/'train_airing_overlaps.csv.gz'):
        f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc']);n=int(r['overlap_seconds']);i=r['tv_show_id'];u=r['user_id']
        if not TRAIN[0]<=f<l<=TRAIN[1] or int(i)<=0:raise ValueError('Historical boundary or ID error')
        if n<120:continue
        baskets[u,(f-TRAIN[0])//(14*86400)].add(i)
        if f>=EARLY:earlycount[u][i]+=1
        if u in USERS:
            x=independent[u][i];x[0]+=1;x[1]+=n;x[2]=max(x[2],l);stats[u]['first_qualifying_overlap_utc']=min(f,stats[u]['first_qualifying_overlap_utc'] or f)
            dt=__import__('datetime').datetime.fromtimestamp(f,__import__('zoneinfo').ZoneInfo('Europe/Kyiv'));selected_weeks[u].add((dt.isocalendar().year,dt.isocalendar().week));channels[u].add(r['channel_id'])
    for u in USERS:
        ck(u+'_independent_historical_activity',{i:tuple(v) for i,v in independent[u].items()}==data.profiles[u]);stats[u]['active_calendar_weeks']=len(selected_weeks[u]);stats[u]['channels_with_qualifying_airings']=len(channels[u])
    ck('heavy_activity_expected',stats[USERS[0]]['distinct_programs']==757 and stats[USERS[0]]['qualifying_airings']==2358 and stats[USERS[0]]['watched_seconds']==5118834)
    ck('light_activity_expected',stats[USERS[1]]['distinct_programs']==1 and stats[USERS[1]]['qualifying_airings']==1 and stats[USERS[1]]['watched_seconds']==2753)
    item_support=Counter()
    for items in baskets.values():item_support.update(items)
    ck('AR_training_transaction_count',len(baskets)==23310)
    del baskets
    paths={METHODS[0]:ROOT/'results/step4/popularity_distinct_users_predictions.csv',METHODS[1]:ROOT/'results/step4/personal_frequency_qualifying_airings_predictions.csv',
        METHODS[2]:ROOT/'results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv',METHODS[3]:ROOT/'results/step6/experiments/time_window_14d_lift_log/predictions.csv',
        METHODS[4]:ROOT/'results/step9_cbf/predictions/cbf_channel_context_emphasis_airings.csv'}
    pred={m:predictions(p) for m,p in paths.items()};diag={m:{} for m in METHODS};rawscores={m:{} for m in METHODS};sourcehashes={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    s4=read(ROOT/'results/step4/step4_outputs.json');pc=read(ROOT/'results/step5/ppr_model_config.json');ac=read(ROOT/'results/step6/association_best_config.json');cc=read(ROOT/'results/step9_cbf/step9_validation_results.json')
    ck('saved_pop_and_PF_prediction_hashes',all(sha(paths[m])==s4[paths[m].name]['sha256'] for m in METHODS[:2]))
    ck('saved_PPR_AR_config_prediction_hashes',sha(paths[METHODS[2]])==pc['best_prediction_sha256'] and sha(paths[METHODS[3]])==ac['prediction_sha256'])
    ck('selected_original_method_configs',pc['best_config_id']=='cosine_pfrequency_a0.35_k30' and ac['config_id']=='time_window_14d_lift_log' and cc['cbf_best_individual_config_id']=='cbf_channel_context_emphasis_airings')
    ck('same_original_protocol_ids',pc['protocol_id']==ac['protocol_id']==cc['protocol_id']==config['protocol_id'])
    for u in USERS:
        ck(u+'_popularity_replay',pred[METHODS[0]][u]==fallback[:5]);diag[METHODS[0]][u]={'personalized_candidates':None,'fallback_positions':[],'kind':'Global popularity baseline; not a personalized fallback event'};rawscores[METHODS[0]][u]=data.distinct_viewers
        profile=data.profiles[u];personal=sorted((i for i in profile if i in poolset),key=lambda i:(-profile[i][0],int(i)));order=personal+[i for i in fallback if i not in set(personal)]
        ck(u+'_personal_frequency_replay',tuple(order[:5])==pred[METHODS[1]][u]);diag[METHODS[1]][u]={'personalized_candidates':len(personal),'fallback_positions':list(range(min(5,len(personal))+1,6))};rawscores[METHODS[1]][u]={i:v[0] for i,v in profile.items()}
    graphpath=ROOT/'results/step5/graph_cosine_top30.npz';ck('PPR_saved_graph_hash',sha(graphpath)==pc['graph_artifacts']['cosine_top30']['sha256']);g=load(graphpath);nodes=tuple(str(i) for i in g['nodes']);matrix=csr_matrix((g['transition_data'],g['indices'],g['indptr']),shape=(len(nodes),len(nodes)))
    pairs={(int(i),int(j)):int(n) for i,j,n in zip(g['undirected_edge_i'],g['undirected_edge_j'],g['undirected_common_viewers'])};weights={(int(i),int(j)):float(w) for i,j,w in zip(g['undirected_edge_i'],g['undirected_edge_j'],g['undirected_edge_weight'])}
    graph=ppr.ProgramGraph(nodes,matrix,pairs,weights,'cosine',30,40,3,pc['best_metrics']['graph']['pair_occurrences_generated'],pc['best_metrics']['graph']['profile_programs_truncated'],0.)
    ck('graph_node_identity',nodes==data.nodes)
    for u in USERS:
        result,info=ppr.predict_user(graph,data,u,pool,fallback,personalization='frequency',restart_probability=.35,tolerance=1e-8,max_iterations=250)
        ck(u+'_PPR_replay_and_convergence',result==pred[METHODS[2]][u] and info['converged']);restart=ppr.personalization_vector(data.profiles[u],data,variant='frequency');s,_=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
        positives={i:float(s[data.index[i]]) for i in pool if i in data.index and s[data.index[i]]>0};rawscores[METHODS[2]][u]=positives
        diag[METHODS[2]][u]={'personalized_candidates':len(positives),'fallback_positions':list(range(6-info['fallback_slots'],6)),'converged':True,'iterations':info['iterations']}
    rulepath=ROOT/ac['rules_file'];ck('saved_AR_rules_hash',sha(rulepath)==ac['rules_sha256']);rules=defaultdict(list)
    for r in records(rulepath):rules[r['antecedent_id']].append({'antecedent':r['antecedent_id'],'consequent':r['consequent_id'],'pair_count':int(r['transaction_support_count']),'support':float(r['support']),'confidence':float(r['confidence']),'lift':float(r['lift'])})
    model=SimpleNamespace(dataset=SimpleNamespace(item_support=item_support),rules_by_antecedent=rules);rule_evidence={}
    for u in USERS:
        result,info=ar.recommend_user(model,data.profiles[u],pool,fallback,score_type='confidence_log_lift',antecedent_weight='log_frequency');ck(u+'_AR_replay',result==pred[METHODS[3]][u])
        ars,_=ar.score_user(model,data.profiles[u],pool,score_type='confidence_log_lift',antecedent_weight='log_frequency');rawscores[METHODS[3]][u]=ars;diag[METHODS[3]][u]={'personalized_candidates':len(ars),'fallback_positions':list(range(6-info['fallback_slots'],6))}
        norm=sum(math.log1p(v[0]) for v in data.profiles[u].values());evidence=[]
        for a,v in data.profiles[u].items():
            for r in rules.get(a,[]):
                if r['consequent'] in result:evidence.append({**r,'contribution':math.log1p(v[0])/norm*r['confidence']*math.log1p(min(r['lift'],100.))})
        rule_evidence[u]=sorted(evidence,key=lambda r:-r['contribution'])[:12]
    trainmeta=source_schedule_metadata([SOURCE/'export_arh_11-20-final.csv'],TRAIN);validmeta=source_schedule_metadata([SOURCE/f'export_arh_{p}-final.csv' for p in ('11-20','21-30','31-42')],VALID)
    cw=cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'];space=cbf.FeatureSpace.fit(trainmeta,cw);cm=space.transform(validmeta,pool,cw);uv,uvmeta=cbf.build_user_profiles(data.profiles,space.training_items,{i:j for j,i in enumerate(space.training_items)},space.training_matrix,USERS,weighting='airings');ranked,cd=cbf.rank_candidates(cm,uv,pool,fallback)
    for j,u in enumerate(USERS):
        ck(u+'_saved_CBF_prediction_replay',ranked[j][:5]==pred[METHODS[4]][u]);sim=(uv[j]@cm.T).toarray().ravel();rawscores[METHODS[4]][u]={i:float(s) for i,s in zip(pool,sim) if s>0};diag[METHODS[4]][u]={'personalized_candidates':cd[j]['positive_similarity_candidates'],'fallback_positions':list(range(6-cd[j]['fallback_slots_top5'],6))}
    # A separate strict-chronological proxy: no21-30 outcomes are used in its fit.
    ep=ROOT/'results/v11_experiment/early.npz';em=read(ROOT/'results/v11_experiment/early_build.json');ck('strict_temporal_training_cache_hash',sha(ep)==em['data_sha256'] and em['context']['history']==[TRAIN[0],EARLY] and em['context']['target']==[EARLY,TRAIN[1]])
    ez=load(ep);emask=ez['original'];ez={k:(v[emask] if k in ('X','users','items','binary_label','original') else v) for k,v in ez.items()};X,names=new_matrix(ez,em['context']['historical_users']);ey=np.zeros(len(ez['users']),dtype=np.int8);et={u:sorted(c,key=lambda i:(-c[i],int(i)))[:5] for u,c in earlycount.items()}
    ey[:]=[str(i) in et.get(str(u),[]) for u,i in zip(ez['users'],ez['items'])];ck('early_labels_pre_validation_reconstructed',np.array_equal(ey,ez['binary_label']))
    params=read(v10/'experiment.json')['pairwise_info']['parameters'];groups=grouping(ez['users']);ck('strict_temporal_full_query_groups',groups.sum()==len(ey)==133605)
    bst=lgb.train(params,lgb.Dataset(X,label=ey,group=groups,feature_name=names),num_boost_round=78);evalpath=OUT/'v10_temporal_evaluation_model.txt';bst.save_model(str(evalpath))
    vp=ROOT/'results/v8_experiment/validation.npz';e9=read(ROOT/'results/v9_experiment/experiment.json');ck('pre_validation_feature_cache_hash',sha(vp)==e9['feature_source_sha256'])
    with np.load(vp,allow_pickle=False) as zz:full={k:zz[k] for k in ('X','users','items','columns')}
    wanted=np.isin(full['users'],USERS);vz={k:(v[wanted] if k!='columns' else v) for k,v in full.items()};M,names2=new_matrix(vz,read(ROOT/'results/v8_experiment/population.json')['historical_users']);ck('strict_temporal_same_V10_schema_config',names==names2==sm['feature_names'] and bst.feature_name()==names and params==sm['model_parameters'])
    s=bst.predict(M,num_threads=4);fullrank=rankings(s,vz,np.arange(len(s)),USERS);pred[METHODS[5]]={u:r[:5] for u,r in fullrank.items()}
    if not np.array_equal(s,bst.predict(M,num_threads=4)):raise ValueError('Proxy inference nondeterministic')
    for u in USERS:
        diag[METHODS[5]][u]={'personalized_candidates':sum(vz['users']==u).item(),'fallback_positions':[],'kind':'Learned reranking over the original V10 multigenerator union; neither fixed user is cold'}
        rawscores[METHODS[5]][u]={str(i):float(v) for i,v,uu in zip(vz['items'],s,vz['users']) if str(uu)==u}
    proxyconfig={'purpose':'Evaluation-only fixed-V10-configuration proxy, not production model or an optimized replacement','history':[TRAIN[0],EARLY],'supervised_targets':[EARLY,TRAIN[1]],'evaluation_features':list(TRAIN),'evaluation_target':list(VALID),
        'training_rows':len(ey),'training_users':len(groups),'feature_names':names,'parameters':params,'rounds':78,'training_feature_source':str(ep.relative_to(ROOT)),'training_feature_sha256':sha(ep),
        'model_path':str(evalpath.relative_to(ROOT)),'model_sha256':sha(evalpath),'no_validation_target_used_for_fit':True,'limitation':'Same77features/configuration but earlier5wk feature/5wk label horizon and1260-user fitting pool, versus broader production V10 fit; not identical learned weights.'}
    dump('v10_temporal_evaluation_config.json',proxyconfig)
    # Labels are evaluated after every recommendation list has been reconstructed.
    actual={u:Counter() for u in USERS};validseconds={u:Counter() for u in USERS}
    for r in records(validation.STEP2/'threshold_120s/validation_interactions.csv.gz'):
        if r['user_id'] not in USERS:continue
        ck('validation_boundary_'+r['user_id']+'_'+r['tv_show_id'],VALID[0]<=int(r['first_watch_utc'])<int(r['last_watch_utc'])<=VALID[1])
        actual[r['user_id']][r['tv_show_id']]=int(r['qualifying_airings']);validseconds[r['user_id']][r['tv_show_id']]=int(r['watched_seconds'])
    secondary={u:sorted(c,key=lambda i:(-c[i],int(i)))[:5] for u,c in actual.items()}
    for u in USERS:ck(u+'_frozen_truth_equals_actual_validation',set(actual[u])==protocol.truth[u])
    allitems={i for byuser in pred.values() for p in byuser.values() for i in p}|{i for u in USERS for i in secondary[u]};allitems|=set(data.profiles[USERS[1]])
    meta=metadata_for(allitems);examples=[];summary=[]
    for u in USERS:
        stats[u].update(validation_distinct_programs=len(actual[u]),validation_qualifying_airings=sum(actual[u].values()),validation_qualifying_seconds=sum(validseconds[u].values()))
        for method in METHODS:
            p=pred[method][u];ck(u+'_'+method+'_strict_five',len(p)==5 and len(set(p))==5 and set(p)<=poolset and all(type(i)is str and i.isascii() and i.isdecimal() and not i.startswith('0') for i in p))
            ap=ap_reference(actual[u],p);ap5=ap_reference(secondary[u],p);ck(u+'_'+method+'_AP_independent',abs(ap-evaluation.average_precision_at_5(set(actual[u]),p,strict=True,candidates=pool))<1e-15)
            fpos=diag[method][u]['fallback_positions'];summary.append({'user_id':u,'method':method,'hits_all_distinct':len(set(p)&set(actual[u])),'AP5_all_distinct':ap,'hits_frequency_Top5':len(set(p)&set(secondary[u])),'AP5_frequency_Top5':ap5,'historically_seen_recommendations':sum(i in data.profiles[u] for i in p),'fallback_positions':fpos})
            for rank,i in enumerate(p,1):
                mm=meta[i];fallback_used=rank in fpos
                if fallback_used:ck(u+'_'+method+f'_rank{rank}_actual_fallback',i in fallback and (method!=METHODS[1] or i not in data.profiles[u]))
                ck(u+'_'+method+f'_rank{rank}_metadata_in_validation_schedule',mm['metadata_complete'])
                examples.append({'user_id':u,'method':method,'rank':rank,'tv_show_id':i,'title':mm['title'],'category':mm['category'],'in_validation_truth':i in actual[u],
                    'validation_qualifying_airings':actual[u].get(i,0),'validation_watched_seconds':validseconds[u].get(i,0),'in_frequency_Top5_truth':i in secondary[u],
                    'in_training_history':i in data.profiles[u],'training_qualifying_airings':data.profiles[u].get(i,(0,0,0))[0],'fallback_used':fallback_used,'native_score':rawscores[method][u].get(i),
                    'metadata_row':mm['schedule_row'],'metadata_variants':mm['variant_count']})
    ck('all60_recommendation_positions_present',len(examples)==60 and len(summary)==12)
    # Earlier OOF lists are preserved for audit only; do not mistake them for a strict temporal example.
    existing_oof=predictions(v10/'selected_all_crossfit.csv')
    sourcehashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'exam_quiz_FKN.md',ROOT/'data/step3/evaluation_config.json',ROOT/'data/step3/validation_truth.csv',ROOT/'data/step2/threshold_120s/train_interactions.csv.gz',ROOT/'data/step2/threshold_120s/validation_interactions.csv.gz',graphpath,rulepath,v10/'experiment.json',v10/'selected_all_crossfit.csv',ROOT/'results/step9_cbf/step9_validation_results.json')})
    changed=[p for p,h in snapshot.items() if not Path(p).is_file() or sha(p)!=h];ck('every_historical_artifact_and_original_export_unchanged',not changed,changed)
    dump('examples.json',{'users':list(USERS),'history':list(TRAIN),'validation':list(VALID),'protocol_id':config['protocol_id'],'candidate_count':len(pool),'primary_truth_definition':'All distinct>=120s qualifying validation programs; no truncation; supplementary frequency-Top5 separately marked',
        'statistics':stats,'recommendations':examples,'summary':summary,'fallback_diagnostics':diag,'metadata':meta,'actual_validation_counts':actual,'actual_validation_seconds':validseconds,'frequency_Top5':secondary,
        'rule_evidence':rule_evidence,'CBF_config':{'id':cc['cbf_best_individual_config_id'],'field_weights':cw,'profile_weighting':'airings','train_documents':space.document_count,'vocabulary_size':len(space.vocabulary),'fit_history_end':TRAIN[1]},
        'V10_temporal_proxy':proxyconfig,'original_V10_user_OOF_for_audit_only':existing_oof,'original_V10_OOF_warning':'These exclude the evaluated user but fit other usersWeek21-30targets. Not used in strict temporal main comparison.',
        'source_prediction_paths':{m:str(p.relative_to(ROOT)) for m,p in paths.items()},'source_sha256':sourcehashes,'production_V10_model_sha256':sm['model_sha256'],'production_V10_CSV_sha256':sm['submission_sha256'],
        'designated_final':'V10 production LambdaRank','Kaggle_private_MAP_user_reported':.40637,'automatic_submission':False})
    with (OUT/'recommendations.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(examples[0]),lineterminator='\n');w.writeheader();w.writerows(examples)
    def esc(x):return str(x).replace('|','\\|').replace('\n',' ')
    text=[]
    for role,u in zip(('Heavy viewer','Light viewer'),USERS):
        text.extend([f'## {role}: `{u}`','', '| Method | Rank | tv_show_id | Title from validation schedule | Category | Actual validation match | Seen in training | Popularity fallback |','| --- | ---: | --- | --- | --- | --- | --- | --- |'])
        for r in examples:
            if r['user_id']==u:text.append('| '+' | '.join([esc(r['method']),str(r['rank']),'`'+r['tv_show_id']+'`',esc(r['title']),esc(r['category']),'Yes' if r['in_validation_truth'] else 'No','Yes' if r['in_training_history'] else 'No','Yes' if r['fallback_used'] else 'No'])+' |')
        text.append('')
    with (OUT/'tables.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    dump('verification.json',{'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),
        'checks':checks,'protected_files':len(snapshot),'protected_session_exports':[p for p in snapshot if 'session' in p.lower() and p.endswith('.json')],
        'generated_examples_sha256':sha(OUT/'examples.json'),'tables_sha256':sha(OUT/'tables.md'),'recommendations_sha256':sha(OUT/'recommendations.csv'),
        'production_V10_model_sha256':sha(mp),'production_V10_submission_sha256':sha(ROOT/sm['submission_path']),'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'scope':'Replayed original five selected methods from original configuration/artifacts; independently reconstructed historical activity and primary validation truth; strict-temporal V10-configuration proxy, metadata and all60rows verified. Production V10 never evaluated on its own historical fitting labels.'})
    print(json.dumps({'statistics':stats,'summary':summary,'fallbacks':diag,'V10_proxy_predictions':pred[METHODS[5]],'original_V10_OOF_audit':existing_oof,
        'passed_checks':len(checks),'failed_checks':0,'seconds':time.monotonic()-t},indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
