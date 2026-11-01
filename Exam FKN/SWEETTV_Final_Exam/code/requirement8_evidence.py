"""Read-only Requirement8 measurement ledger and report verification; never fits models."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
from collections import Counter
import argparse,csv,json,math,time,re,traceback
import numpy as np
import lightgbm as lgb
import evaluation,validation
from v8_common import ROOT,SOURCE,read,load,sha,records

OUT=ROOT/'results/final_requirement8_evidence';REPORT=ROOT/'results/final_requirement8_business_analysis.md'
USERS=('13571199046623737884','8150816946177670084')

def write(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')

def preds(path):return {r['user_id']:tuple(r['tv_show_id'].split()) for r in records(path)}

def prepare():
    start=time.monotonic()
    if OUT.exists() or REPORT.exists():raise FileExistsError('Preserve completed Requirement8 work')
    OUT.mkdir();protected=[p for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents and not any(k in p.parts for k in ('.venv','.venv-lightgbm','__pycache__','.git')) and p!=Path(__file__).resolve()]+list(SOURCE.glob('*.csv'))
    write('preservation_before.json',{str(p):sha(p) for p in protected});checks=[]
    def ck(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok:raise ValueError(name)
    r7=read(ROOT/'results/final_requirement7_evidence/examples.json');vr7=read(ROOT/'results/final_requirement7_evidence/report_verification.json')
    ck('Requirement7_report_and_evidence_integrity',vr7['all_passed'] and sha(ROOT/'results/final_requirement7_examples.md')==vr7['details']['report_sha256'] and tuple(r7['users'])==USERS)
    for path,h in r7['source_sha256'].items():ck('Requirement7_source_'+path,sha(ROOT/path)==h)
    cfg=read(ROOT/'data/step3/evaluation_config.json');protocol=validation.EvaluationProtocol();pop=read(ROOT/'results/step4/step4_run.json');pc=read(ROOT/'results/step5/ppr_model_config.json');ac=read(ROOT/'results/step6/association_best_config.json');arun=read(ROOT/'results/step6/association_run.json');cb=read(ROOT/'results/step9_cbf/step9_validation_results.json')
    secondary={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')};cbrow=next(r for r in cb['experiments'] if r['config_id']==cb['cbf_best_individual_config_id'])
    specs={
        'Popularity':('results/step4/popularity_distinct_users_predictions.csv',pop['variants']['popularity_distinct_users']['primary_map_at_5'],None),
        'Personal Frequency':('results/step4/personal_frequency_qualifying_airings_predictions.csv',pop['variants']['personal_frequency_qualifying_airings']['primary_map_at_5'],None),
        'Personalized PageRank':('results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv',pc['best_metrics']['primary_map_at_5'],pc['best_metrics']['secondary_top5_sensitivity_map_at_5']),
        'Association Rules':(ac['prediction_file'],float(ac['primary_map_at_5']),float(ac['secondary_top5_sensitivity_map_at_5'])),
        'Content-Based Filtering':('results/step9_cbf/predictions/'+cb['cbf_best_individual_config_id']+'.csv',cbrow['primary_map_at_5'],cbrow['secondary_map_at_5'])}
    metrics={}
    for m,(path,expected,esec) in specs.items():
        prediction=preds(ROOT/path);a=evaluation.evaluate_predictions(prediction,protocol.truth,protocol.cohort,protocol.candidates)['primary_map_at_5'];b=evaluation.evaluate_predictions(prediction,secondary,protocol.cohort,protocol.candidates)['primary_map_at_5']
        ck(m+'_saved_predictions_primary_MAP_recomputed',math.isclose(a,expected,rel_tol=0,abs_tol=1e-14))
        if esec is not None:ck(m+'_saved_predictions_secondary_MAP_recomputed',math.isclose(b,esec,rel_tol=0,abs_tol=1e-14))
        metrics[m]={'primary_all_distinct_MAP5':a,'secondary_frequency_Top5_MAP5':b,'users':len(protocol.cohort),'prediction_source':path,'sha256':sha(ROOT/path)}
    v10=read(ROOT/'results/v10_experiment/experiment.json');sm=read(ROOT/'results/v10_experiment/submission_manifest.json');bst=lgb.Booster(model_file=str(ROOT/sm['model_path']))
    ck('production_V10_model_and_CSV_hashes',sha(ROOT/sm['model_path'])==sm['model_sha256']==r7['production_V10_model_sha256'] and sha(ROOT/sm['submission_path'])==sm['submission_sha256']==r7['production_V10_CSV_sha256'])
    ck('production_V10_real_schema_objective_rounds',bst.feature_name()==sm['feature_names']==v10['pairwise_info']['feature_names'] and len(bst.feature_name())==77 and bst.params['objective']=='lambdarank' and bst.current_iteration()==78)
    allpred=preds(ROOT/'results/v10_experiment/selected_all_crossfit.csv');labels=read(ROOT/'results/v8_experiment/labels.json')['secondary'];v10metrics={}
    for stage in ('development','assessment'):
        uu=v10[stage+'_users'];val=math.fsum(evaluation.average_precision_at_5(labels[u],allpred[u]) for u in uu)/len(uu);ck('V10_'+stage+'_MAP_from_actual_crossfits',math.isclose(val,v10['selected_'+stage+'_MAP'],rel_tol=0,abs_tol=1e-14));v10metrics[stage]={'MAP5':val,'users':len(uu)}
    # Candidate-only recall is independent of which reranker orders the identical frozen rows.
    z=load(ROOT/'results/v8_experiment/validation.npz');candidates={u:set() for u in labels}
    for u,i in zip(z['users'],z['items']):candidates[str(u)].add(str(i))
    hits=sum(len(candidates[u]&set(t)) for u,t in labels.items());den=sum(len(t) for t in labels.values());v10metrics['candidate_recall']={'hits':hits,'labels':den,'recall':hits/den,'users':len(labels),'candidate_rows':len(z['users'])}
    ck('original_candidate_recall_approximately85pct',round(100*hits/den,2)==85.03)
    bm=pc['best_metrics'];runtime={
        'Popularity':{'preparation_seconds':pop['training_runtime_seconds']['popularity'],'prediction_seconds':pop['prediction_runtime_seconds']['popularity_distinct_users'],'peak_rss_mib':pop['peak_rss_mib'],'peak_scope':'Entire shared5variant Step4 run; not isolated method peak','post_fit_rss_mib':pop['rss_after_popularity_fit_mib'],'users':1260,'source':'results/step4/step4_run.json'},
        'Personal Frequency':{'preparation_seconds':pop['training_runtime_seconds']['personal_frequency'],'prediction_seconds':pop['prediction_runtime_seconds']['personal_frequency_qualifying_airings'],'peak_rss_mib':pop['peak_rss_mib'],'peak_scope':'Same shared Step4 run','post_fit_rss_mib':pop['rss_after_personal_fit_mib'],'users':1260,'source':'results/step4/step4_run.json'},
        'Personalized PageRank':{'preparation_seconds':bm['ppr_training_runtime_seconds'],'fallback_preparation_seconds':bm['popularity_fallback_training_runtime_seconds'],'prediction_seconds':bm['prediction_generation_seconds'],'peak_rss_mib':pc['peak_process_rss_mib'],'peak_scope':'Entire16configuration Step5 run','users':1260,'source':'results/step5/ppr_model_config.json'},
        'Association Rules':{'preparation_seconds':ac['rule_training_runtime_seconds'],'prediction_seconds':float(ac['prediction_runtime_seconds']),'peak_rss_mib':arun['peak_rss_mib'],'peak_scope':'Entire11variant Step6 run','users':1260,'source':'results/step6/association_best_config.json; results/step6/association_run.json'},
        'Content-Based Filtering':{'preparation_seconds':cb['feature_spaces']['channel_context_emphasis']['fit_and_transform_seconds'],'preparation_scope':'Selected feature-space fit/transform; not complete user-profile preparation','prediction_seconds':cbrow['prediction_runtime_seconds'],'prediction_scope':'User-profile construction and ranking for selected CBF','peak_rss_mib':cb['peak_rss_mib'],'peak_scope':'Entire Step9 CBF/hybrid run','users':1260,'source':'results/step9_cbf/step9_validation_results.json'},
        'V10 LambdaRank':{'isolated_training_seconds':None,'isolated_prediction_seconds':None,'complete_experiment_seconds':v10['seconds'],'complete_experiment_peak_rss_mib':v10['peak_rss_mib'],'cached_feature_generation_seconds':sm['seconds'],'generation_peak_rss_mib':sm['peak_rss_mib'],'generation_users':sm['users'],'source':'results/v10_experiment/experiment.json; results/v10_experiment/submission_manifest.json'}}
    rows=0;named=0
    for r in records(SOURCE/'export_arh_11-20-final.csv'):rows+=1;named+=int(r['tv_show_id'])>0
    ck('first_schedule_label_coverage_verified',rows==1074235 and named==115613)
    rules=read(ROOT/'results/final_requirement7_evidence/examples.json')['rule_evidence'];rule_refs={}
    for u,a,b in ((USERS[0],'20088','2400480'),(USERS[1],'90053098','90075859')):
        rule_refs[u]=next(r for r in rules[u] if r['antecedent']==a and r['consequent']==b)
    rr=next(r for r in records(ROOT/ac['rules_file']) if r['antecedent_id']=='90053098' and r['consequent_id']=='90075859')
    ck('light_rule_exact_source_confidence_lift',float(rr['confidence'])==rule_refs[USERS[1]]['confidence'] and float(rr['lift'])==rule_refs[USERS[1]]['lift'])
    s8=read(ROOT/'results/v8_experiment/submission_manifest.json');env={'recorded_hardware_source':'results/step1_audit.md; CPU Intel Corei7-6600U2.60GHz,2cores/4logicalCPUs,11GiBRAM','Step4':'Python3.14.4, standard library','Step5':{'python':pc['python_version'],'numpy':pc['numpy_version'],'scipy':pc['scipy_version']},'Step6':{k:arun[k] for k in ('python','numpy','scipy')},'Step9':cb['environment'],'LightGBM_pipeline_context_from_V8':s8['versions'],'V10_separate_full_environment_recorded':False}
    ledger={'metrics':metrics,'V10_local_metrics':v10metrics,'user_statistics':r7['statistics'],'user_method_summary':r7['summary'],'recommendations':r7['recommendations'],'fallback_diagnostics':r7['fallback_diagnostics'],
        'runtime':runtime,'environment':env,'V10_parameters':v10['pairwise_info']['parameters'],'V10_features':sm['feature_names'],'V10_family_best':v10['family_best'],'V10_strategy':v10['strategy'],
        'V10_comparison':v10['comparison'],'V10_feature_rows':len(z['users']),'production_V10_model_sha256':sm['model_sha256'],'production_V10_submission_sha256':sm['submission_sha256'],
        'schedule_coverage':{'positive_ID_rows':named,'rows':rows,'percentage':100*named/rows},'rules':rule_refs,
        'Kaggle':{'private':.40637,'public':.42051,'source':'User-confirmed returned Kaggle scores in current task; consistent with previous baseline records; not independently queried from Kaggle by this task'},
        'optional_requirement6_available':(ROOT/'results/final_requirement6_verification.md').exists(),'sources':{str(p):sha(p) for p in protected if ('results/' in str(p) or 'code/' in str(p)) and ('step4_run.json' in p.name or p.name in ('ppr_model_config.json','association_best_config.json','association_run.json','step9_validation_results.json'))}}
    ck('private_score_consistent_with_existing_user_record',r7['Kaggle_private_MAP_user_reported']==ledger['Kaggle']['private'])
    write('verified_measurements.json',ledger);write('preparation_verification.json',{'all_passed':all(c['passed'] for c in checks),'passed_checks':len(checks),'failed_checks':0,'checks':checks,'seconds':time.monotonic()-start})
    print(json.dumps({k:ledger[k] for k in ('metrics','V10_local_metrics','runtime','environment','schedule_coverage','optional_requirement6_available')},indent=2),flush=True)

def claims():
    """Build quantitative text checks directly from already verified source values."""
    d=read(OUT/'verified_measurements.json');entries=[]
    def add(name,text,keys,source,value=None,evidence=None):
        v=d if evidence is None else read(ROOT/evidence)
        for key in keys:v=v[key]
        c={'id':name,'text':text,'ledger_path':keys,'value':v if value is None else value,'source':source}
        if evidence:c['evidence_path']=evidence
        entries.append(c)
    add('Kaggle_private_score',f"{d['Kaggle']['private']:.5f}",['Kaggle','private'],'Current user-confirmed genuine Kaggle private score; consistent with Requirement7')
    add('Kaggle_public_score',f"{d['Kaggle']['public']:.5f}",['Kaggle','public'],'Current user-confirmed genuine Kaggle public score')
    for m,v in d['metrics'].items():
        add(m+'_local_MAP_table',f"| {m} | {v['primary_all_distinct_MAP5']:.6f} | {v['secondary_frequency_Top5_MAP5']:.6f} |",['metrics',m],v['prediction_source']+'; recomputed primary and secondary MAP')
    h=d['user_statistics'][USERS[0]];l=d['user_statistics'][USERS[1]]
    add('heavy_historical_activity',f"{h['distinct_programs']} distinct historical programs, {h['qualifying_airings']:,} qualifying airings and {h['watched_seconds']:,} watched seconds",['user_statistics',USERS[0]],'Requirement7 independent history reconstruction')
    add('heavy_validation_size',f"{h['validation_distinct_programs']} distinct programs",['user_statistics',USERS[0],'validation_distinct_programs'],'Requirement7 actual qualifying validation programs')
    add('light_historical_activity',f"one qualifying historical program and one airing, totaling {l['watched_seconds']:,} watched seconds",['user_statistics',USERS[1]],'Requirement7 history; exactly one program/airing')
    add('light_validation_size',f"{l['validation_distinct_programs']} distinct programs",['user_statistics',USERS[1],'validation_distinct_programs'],'Requirement7 actual qualifying validation programs')
    add('light_PF_fallback_positions','2–5',['fallback_diagnostics','Personal frequency',USERS[1],'fallback_positions'],'Requirement7 actual fallback',[2,3,4,5])
    for u,label in ((USERS[0],'heavy'),(USERS[1],'light')):
        for field in ('support','confidence','lift'):
            v=d['rules'][u][field];add(label+'_rule_'+field,f'{v:.6f}',['rules',u,field],'Saved14day historical rule evidence; rules.csv',v)
    for stage in ('development','assessment'):
        row=d['V10_local_metrics'][stage];add('V10_'+stage+'_MAP',f"{row['MAP5']:.6f} on {row['users']:,} {stage} users",['V10_local_metrics',stage],'V10 historical cross-fitted predictions; frequency-Top5 truth')
    val=d['V10_comparison']['E1_V8']['development']['MAP5'];add('V8_local_development_baseline',f'{val:.6f}',['V10_comparison','E1_V8','development','MAP5'],'V10 experiment.json recorded baseline')
    for m,v in d['runtime'].items():
        if m=='Popularity':text=f"| {m} | {v['preparation_seconds']:.3f} s, shared popularity statistics | {v['prediction_seconds']:.6f} s, selected variant, 1,260 users | {v['peak_rss_mib']:.2f} MiB, entire shared Step 4 run | [S4] |"
        elif m=='Personal Frequency':text=f"| {m} | {v['preparation_seconds']:.3f} s, personal tables; shared popularity preparation separate | {v['prediction_seconds']:.6f} s, selected variant, 1,260 users | {v['peak_rss_mib']:.2f} MiB, same shared Step 4 run | [S4] |"
        elif m=='Personalized PageRank':text=f"| {m} | {v['preparation_seconds']:.3f} s, selected profile/pair/graph preparation; plus {v['fallback_preparation_seconds']:.3f} s popularity fallback preparation | {v['prediction_seconds']:.6f} s, 1,260 users | {v['peak_rss_mib']:.2f} MiB, entire staged Step 5 search | [S5] |"
        elif m=='Association Rules':text=f"| {m} | {v['preparation_seconds']:.3f} s, selected model preparation | {v['prediction_seconds']:.6f} s, 1,260 users | {v['peak_rss_mib']:,.2f} MiB, entire Step 6 comparison | [S6] |"
        elif m=='Content-Based Filtering':text=f"| {m} | {v['preparation_seconds']:.6f} s, selected metadata-space fit/transform, not full pipeline preparation | {v['prediction_seconds']:.6f} s, selected profile construction and ranking, 1,260 users | {v['peak_rss_mib']:,.2f} MiB, entire Step 9 comparison | [S7] |"
        else:text=f"| {m} | Fit-only not separately recorded; complete V10 experiment {v['complete_experiment_seconds']:.3f} s | Score-only not separately recorded; cached-feature final generation {v['cached_feature_generation_seconds']:.3f} s, 1,260 users | {v['complete_experiment_peak_rss_mib']:.2f} MiB, experiment; {v['generation_peak_rss_mib']:.2f} MiB, final generation | [S3] |"
        add(m+'_cost_row',text,['runtime',m],v['source'])
    c=d['V10_local_metrics']['candidate_recall'];add('candidate_recall_counts',f"{c['hits']:,} of {c['labels']:,} relevant Top-5 labels",['V10_local_metrics','candidate_recall'],'Original V8/V10 candidate IDs and labels, independently recounted')
    add('candidate_recall_percentage',f"{100*c['recall']:.2f}% candidate recall",['V10_local_metrics','candidate_recall'],'Same candidate recall, micro denominator over all4893 users')
    sc=d['schedule_coverage'];add('schedule_label_counts',f"{sc['positive_ID_rows']:,} of {sc['rows']:,} rows",['schedule_coverage'],'Direct count of original Week11-20 schedule rows')
    add('schedule_label_approximate_percentage',f"{sc['percentage']:.1f}%",['schedule_coverage'],'Direct schedule count, rounded to1decimal')
    p=d['V10_parameters'];add('V10_real_configuration',f"78 boosting rounds, {p['num_leaves']} leaves, learning rate {p['learning_rate']}, minimum leaf size {p['min_data_in_leaf']}, L2 regularization {p['lambda_l2']}, binary gains {p['label_gain']}, truncation level {p['lambdarank_truncation_level']}, seed {p['seed']} and four threads",['V10_parameters'],'V10 pairwise_info.parameters; source fixed78rounds and actual Booster header')
    add('native_feature_count',f"{len(d['V10_features'])}-feature",['V10_features'],'Production V10 feature names and model header')
    add('zero_unseen_bonus','zero unseen adjustment',['V10_family_best'],'Both calibration families selected unseen_bonus0')
    add('original_eval_cohort','1,260-user',['metrics','Popularity','users'],'Frozen Step3 cohort and actual prediction-row count')
    add('threshold','120 unioned overlap seconds',['policies','threshold_seconds_per_airing'],'Frozen Step3 threshold',120,'data/step3/evaluation_config.json')
    add('strict_proxy_feature_period','Weeks 11–15',['V10_temporal_proxy','history'],'Requirement7 strict temporal proxy',[1583704800,1586725200],'results/final_requirement7_evidence/examples.json')
    add('strict_proxy_label_period','Weeks 16–20',['V10_temporal_proxy','supervised_targets'],'Requirement7 strict temporal proxy',[1586725200,1589749200],'results/final_requirement7_evidence/examples.json')
    add('LightGBM_pipeline_environment','Python 3.12.14, LightGBM 4.6.0 and NumPy 2.2.6',['environment','LightGBM_pipeline_context_from_V8'],'Recorded V8 manifest, context only, not independent V10 environment capture')
    add('original_numeric_environment','Python 3.14.4, NumPy 2.5.4 and SciPy 1.18.1',['environment','Step9'],'Recorded Step9 software environment; Step5/6 manifests match')
    add('historical_hardware','Intel Core i7-6600U at 2.60 GHz, two physical cores and four logical CPUs, with approximately 11 GiB RAM',['environment','recorded_hardware_source'],'results/step1_audit.md hardware table')
    write('report_claims.json',entries);print({'quantitative_claim_checks':len(entries)})

def verify():
    start=time.monotonic();checks=[];details={};path=OUT/'report_verification.json'
    if path.exists():path=OUT/f'report_verification_recheck_{time.time_ns()}.json'
    def ck(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(name)
    try:
        ledger=read(OUT/'verified_measurements.json');text=REPORT.read_text();flat=' '.join(text.split());prep=read(OUT/'preparation_verification.json');r7=read(ROOT/'results/final_requirement7_evidence/examples.json')
        ck('measurement_verification_passed',prep['all_passed']);ck('fixed_users_present_exactly',all('`'+u+'`' in text for u in USERS))
        # Check every explicitly tagged quantitative report claim against a source-linked ledger.
        registry=read(OUT/'report_claims.json')
        for c in registry:
            ck(c['id'],' '.join(c['text'].split()) in flat,c['source'])
            value=read(ROOT/c['evidence_path']) if 'evidence_path' in c else ledger
            for key in c['ledger_path']:value=value[key]
            if 'value' in c:
                ck(c['id']+'_underlying_value',value==c['value'] if not isinstance(value,float) else math.isclose(value,c['value'],rel_tol=0,abs_tol=c.get('tolerance',1e-12)))
        known=set(r['tv_show_id'] for r in r7['recommendations'])|{r['antecedent'] for r in ledger['rules'].values()}|{r['consequent'] for r in ledger['rules'].values()}
        mentioned=re.findall(r'`([0-9]+)`',text)
        ck('every_mentioned_program_or_user_ID_genuine',all(i in known or i in USERS or i=='0' for i in mentioned),mentioned)
        for method in ('Popularity','Personal Frequency','Personalized PageRank','Association Rules','Content-Based Filtering','V10 temporal proxy'):
            # Report table is an explicit claim, not inferred from free text.
            line=next(line for line in text.splitlines() if line.startswith('| '+method+' |') and line.count('|')==6)
            parts=[p.strip() for p in line.split('|')[1:-1]];lookup='Personal frequency' if method=='Personal Frequency' else 'V10 configuration — temporal evaluation proxy' if method=='V10 temporal proxy' else method
            hh=next(r for r in r7['summary'] if r['user_id']==USERS[0] and r['method']==lookup);ll=next(r for r in r7['summary'] if r['user_id']==USERS[1] and r['method']==lookup)
            ck(method+'_actual_user_hits_and_AP_table',parts[1:]==[str(hh['hits_all_distinct'])+'/5',f"{hh['AP5_all_distinct']:.6f}",str(ll['hits_all_distinct'])+'/5',f"{ll['AP5_all_distinct']:.6f}"])
        ck('matrix_factorization_unmeasured_explicit','Matrix Factorization was not implemented as an independently evaluated method' in text and 'no empirical comparison with it' in text)
        ck('proxy_distinct_from_production_explicit','different learned weights' in flat and 'strictly before Week 21' in flat)
        ck('local_and_Kaggle_targets_not_equated','not directly comparable' in text and 'genuine Kaggle' in text)
        ck('no_invented_fusion_weights','no manually assigned fusion weights' in text)
        ck('no_claimed_online_or_business_outcomes','No live A/B test was performed' in flat and 'No revenue, retention, satisfaction or causal business improvement was measured' in flat)
        ck('timing_scope_and_missing_measurements_disclosed','not strictly comparable' in flat and 'not separately recorded' in flat and 'not per-request service latencies' in flat)
        ck('missing_requirement6_disclosed',ledger['optional_requirement6_available'] or 'final_requirement6_verification.md' in text and 'not available' in text)
        # Check all displayed program titles used as examples against R7 schedule metadata.
        for i in ('240081','12001682','20088','90053098','90075859','10003148','90082673'):
            title=r7['metadata'][i]['title'];ck(i+'_exact_observed_title_present',' '.join(title.split()) in flat,title)
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];ck('all_preexisting_files_and_exports_unchanged',not changed,changed)
        ck('production_V10_exact_model_and_CSV_hashes',sha(ROOT/'results/v10_experiment/final_pairwise.txt')==ledger['production_V10_model_sha256'] and sha(ROOT/'submission/final_submission_v10_experimental.csv')==ledger['production_V10_submission_sha256'])
        details.update(report_path=str(REPORT),report_sha256=sha(REPORT),protected_files=len(protected),protected_session_exports=[p for p in protected if 'session' in p.lower() and p.endswith('.json')],
            quantitative_claims=len(registry),production_model_sha256=ledger['production_V10_model_sha256'],production_submission_sha256=ledger['production_V10_submission_sha256'],
            limitations=['Optional Requirement6 verification file not available','Kaggle scores user-confirmed, not independently queried here','Historical cost scopes differ; isolated V10 fitting/scoring latency not recorded','Temporal proxy learned weights and fitting horizon differ from production V10'])
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':str(error)})
        details['traceback']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':details,'seconds':time.monotonic()-start}
    write(path.name,result);print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)
    if not result['all_passed']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','claims','verify']);args=parser.parse_args();{'prepare':prepare,'claims':claims,'verify':verify}[args.phase]()
