"""Independently audit the finished Requirement7 Markdown against source data."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from collections import Counter,defaultdict
from pathlib import Path
from fractions import Fraction
import csv,json,re,time,io,unittest,traceback
import numpy as np
import lightgbm as lgb
import data_pipeline as dp,evaluation,validation
from v8_common import ROOT,SOURCE,read,load,sha,records
from train_v9 import new_matrix,rankings

OUT=ROOT/'results/final_requirement7_evidence';REPORT=ROOT/'results/final_requirement7_examples.md'
USERS=('13571199046623737884','8150816946177670084');LO,CUT,HI=1583704800,1589749200,1595797200

def main():
    t=time.monotonic();checks=[];detail={};path=OUT/'report_verification.json'
    if path.exists():path=OUT/f'report_verification_recheck_{time.time_ns()}.json'
    def ck(name,ok,info=None):
        checks.append({'name':name,'passed':bool(ok),'detail':info})
        if not ok:raise ValueError(name)
    try:
        ex=read(OUT/'examples.json');previous=read(OUT/'verification.json');proxy=read(OUT/'v10_temporal_evaluation_config.json');config=read(ROOT/'data/step3/evaluation_config.json');text=REPORT.read_text()
        ck('extraction_evidence_checks_passed',previous['all_passed'] and previous['failed_checks']==0 and sha(OUT/'examples.json')==previous['generated_examples_sha256'])
        ck('exact_user_strings_and_protocol',tuple(ex['users'])==USERS and ex['history']==[LO,CUT] and ex['validation']==[CUT,HI] and ex['protocol_id']==config['protocol_id'])
        ck('designated_model_and_score_clearly_stated','0.40637' in text and 'Designated final model' in text and 'not Kaggle scores' in text)
        # Separate matched-overlap source to verify actual label counts, not narrative or aggregates.
        counts={u:Counter() for u in USERS};secs={u:Counter() for u in USERS}
        for r in records(validation.STEP2/'validation_airing_overlaps.csv.gz'):
            u=r['user_id']
            if u not in counts:continue
            first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc']);s=int(r['overlap_seconds']);duration=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
            if not CUT<=first<last<=HI or not 0<s<=duration or int(r['tv_show_id'])<=0:raise ValueError('Validation matching/boundary invalid')
            if s>=120:counts[u][r['tv_show_id']]+=1;secs[u][r['tv_show_id']]+=s
        history={u:{} for u in USERS}
        for r in records(validation.STEP2/'threshold_120s/train_interactions.csv.gz'):
            if r['user_id'] in USERS:history[r['user_id']][r['tv_show_id']]=int(r['qualifying_airings'])
        for u in USERS:
            ck(u+'_counts_independent_from_actual_overlaps',dict(counts[u])==ex['actual_validation_counts'][u] and dict(secs[u])==ex['actual_validation_seconds'][u])
            ck(u+'_history_size_and_validation_size',len(history[u])==ex['statistics'][u]['distinct_programs'] and len(counts[u])==ex['statistics'][u]['validation_distinct_programs'])
        expected={(r['user_id'],r['method'],r['rank']):r for r in ex['recommendations']};selected=set(r['tv_show_id'] for r in ex['recommendations']);metadata={}
        for line,r in enumerate(records(SOURCE/'export_arh_21-30-final.csv'),2):
            i=r['tv_show_id']
            if i not in selected or int(r['duration'])<=0:continue
            stamp=dp.parse_timestamp(r['start_time'],schedule=True)
            if stamp<HI and stamp+int(r['duration'])>CUT and i not in metadata:metadata[i]=(r['tv_show_title'],r['tv_show_category'],line)
        for n,r in enumerate(ex['recommendations'],1):
            ck(f'position{n}_metadata_and_actual_flags',metadata.get(r['tv_show_id'])==(r['title'],r['category'],r['metadata_row']) and r['in_validation_truth']==(r['tv_show_id'] in counts[r['user_id']]) and r['in_training_history']==(r['tv_show_id'] in history[r['user_id']]))
        # Parse the actual report tables, rather than trusting the rendering intermediate.
        actual={};user=None
        for line in text.splitlines():
            if line.startswith('## ') and USERS[0] in line:user=USERS[0]
            elif line.startswith('## ') and USERS[1] in line:user=USERS[1]
            if not line.startswith('|'):continue
            cells=[c.strip().replace('\\|','|') for c in re.split(r'(?<!\\)\|',line)[1:-1]]
            if len(cells)!=8 or not cells[1].isdigit() or cells[0] not in ex['fallback_diagnostics']:continue
            key=(user,cells[0],int(cells[1]));actual[key]=cells
        ck('report_contains_exact60_positions',set(actual)==set(expected) and len(actual)==60)
        for n,(key,r) in enumerate(expected.items(),1):
            c=actual[key];ck(f'report_position{n}_exact_source_values',c==[r['method'],str(r['rank']),'`'+r['tv_show_id']+'`',r['title'],r['category'],'Yes' if r['in_validation_truth'] else 'No','Yes' if r['in_training_history'] else 'No','Yes' if r['fallback_used'] else 'No'])
        pool=validation.EvaluationProtocol().candidates
        for summary in ex['summary']:
            u=summary['user_id'];m=summary['method'];rr=sorted((r for r in ex['recommendations'] if r['user_id']==u and r['method']==m),key=lambda r:r['rank']);pred=tuple(r['tv_show_id'] for r in rr)
            ck(u+'_'+m+'_strict_five_and_AP',len(pred)==5 and len(set(pred))==5 and set(pred)<=set(pool) and abs(evaluation.average_precision_at_5(set(counts[u]),pred,strict=True,candidates=pool)-summary['AP5_all_distinct'])<1e-15)
            fpos=[r['rank'] for r in rr if r['fallback_used']];ck(u+'_'+m+'_fallback_consistency',fpos==summary['fallback_positions']==ex['fallback_diagnostics'][m][u]['fallback_positions'])
        for m,p in ex['source_prediction_paths'].items():
            saved={r['user_id']:r['tv_show_id'].split() for r in records(ROOT/p) if r['user_id'] in USERS}
            ck(m+'_report_matches_saved_original_predictions',all(saved[u]==[r['tv_show_id'] for r in ex['recommendations'] if r['user_id']==u and r['method']==m] for u in USERS))
        proxyname='V10 configuration — temporal evaluation proxy';model=lgb.Booster(model_file=str(ROOT/proxy['model_path']))
        ck('proxy_strict_temporal_provenance_and_fixed_parameters',proxy['supervised_targets'][1]<=CUT and proxy['history'][1]==proxy['supervised_targets'][0] and proxy['no_validation_target_used_for_fit'] and model.feature_name()==proxy['feature_names'] and sha(ROOT/proxy['model_path'])==proxy['model_sha256'] and model.params['objective']=='lambdarank' and model.current_iteration()==78)
        with np.load(ROOT/'results/v8_experiment/validation.npz',allow_pickle=False) as z:raw={k:z[k] for k in ('X','users','items','columns')}
        mask=np.isin(raw['users'],USERS);raw={k:(v[mask] if k!='columns' else v) for k,v in raw.items()};X,names=new_matrix(raw,read(ROOT/'results/v8_experiment/population.json')['historical_users']);score=model.predict(X,num_threads=4);ranked=rankings(score,raw,np.arange(len(score)),USERS)
        ck('proxy_report_predictions_replay',all(tuple(r['tv_show_id'] for r in ex['recommendations'] if r['user_id']==u and r['method']==proxyname)==ranked[u][:5] for u in USERS))
        ck('proxy_is_explicitly_distinct_from_production','not identical' in text or 'different' in text)
        modules=('test_step3','test_step4','test_step5','test_step6','test_content_based','test_v10')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules);log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        logfile=OUT/'regression_tests.txt'
        if logfile.exists():logfile=OUT/f'regression_tests_recheck_{time.time_ns()}.txt'
        with logfile.open('x') as f:f.write(log.getvalue())
        detail['regressions']={'tests_run':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors)};ck('relevant_existing_regressions_passed',tests.wasSuccessful(),detail['regressions'])
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];ck('all_prior_artifacts_and_original_exports_preserved',not changed,changed)
        ck('production_V10_model_and_submission_exact_hashes',sha(ROOT/'results/v10_experiment/final_pairwise.txt')==previous['production_V10_model_sha256'] and sha(ROOT/'submission/final_submission_v10_experimental.csv')==previous['production_V10_submission_sha256'])
        detail.update(report_sha256=sha(REPORT),protected_files=len(protected),recommendation_positions=len(actual),user_method_lists=len(ex['summary']),report_path=str(REPORT),
            implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'code/final_requirement7.py')})
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':str(error)})
        detail['traceback']=traceback.format_exc()
    report={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':detail,'seconds':time.monotonic()-t}
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2,ensure_ascii=False),flush=True)
    if not report['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
