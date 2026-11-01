"""Read-only diagnosis of saved V2/V3/V4 models and data. Writes new audit evidence only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo
import bisect
import csv
import gzip
import hashlib
import json
import math
import random
import time
import unittest
import io

import numpy as np
import lightgbm as lgb
import evaluation
import validation
import test_data_pipeline
import test_step3

ROOT=validation.ROOT
SOURCE=ROOT.parent/'sweettv-tv-program-recommender'
OUT=ROOT/'results'/'v2_v4_diagnosis'
STEP10=ROOT/'results'/'step10_lightgbm'
ZONE=ZoneInfo('Europe/Kyiv')
WEEK=604800


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as stream:yield from csv.DictReader(stream)


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def table(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def load_preds(path):
    return evaluation.parse_prediction_records(list(records(path)))


def reference_ap(labels,ranking):
    # Independent relevant-item/prefix enumeration with rational arithmetic.
    labels=set(labels);ranking=list(ranking[:5]);value=Fraction(0)
    for item in labels:
        if item in ranking:
            k=ranking.index(item)+1
            value+=Fraction(len(set(ranking[:k])&labels),k)
    return float(value/min(5,len(labels))) if labels else 0.


def local_stamp(text,schedule=False):
    if schedule:text=text[6:10]+'-'+text[3:5]+'-'+text[:2]+text[10:]
    return int(datetime.fromisoformat(text).replace(tzinfo=ZONE).timestamp())


def local_date(stamp):return datetime.fromtimestamp(stamp,ZONE).date()


def calendar_bucket(stamp,lower):return (local_date(stamp)-local_date(lower)).days//7


def contributions(labels,ranking):
    hits=0;out={}
    for k,item in enumerate(ranking,1):
        if item in labels:
            hits+=1;out[item]=hits/k/min(5,len(labels))
    return out


def coverage(pred,candidates,users):
    missed=[]
    for u in users:
        for k,item in enumerate(pred[u],1):
            if item not in candidates[u]:missed.append({'user_id':u,'rank':k,'program_id':item})
    return {'slots':5*len(users),'missing_slots':len(missed),'users_with_omissions':len({x['user_id'] for x in missed}),
            'missing_programs':dict(Counter(x['program_id'] for x in missed))},missed


def pair_comparison(v2,v4,truth,users):
    a=np.array([reference_ap(truth[u],v2[u]) for u in users]);b=np.array([reference_ap(truth[u],v4[u]) for u in users])
    delta=b-a;rng=np.random.default_rng(2026)
    samples=np.array([delta[rng.integers(0,len(users),len(users))].mean() for _ in range(2000)])
    return {'v2_map':float(a.mean()),'v4_map':float(b.mean()),'delta':float(delta.mean()),
        'improved_users':int(np.sum(delta>1e-12)),'worsened_users':int(np.sum(delta< -1e-12)),
        'equal_users':int(np.sum(np.abs(delta)<=1e-12)),
        'paired_bootstrap_95pct_delta':np.quantile(samples,[.025,.975]).tolist()}


def tests():
    suite=unittest.TestSuite()
    for cls in (test_data_pipeline.PipelineTests,test_step3.MetricTests,test_step3.ValidationTests,test_step3.FallbackInterfaceTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    log=io.StringIO();run=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/'existing_test_output.txt').write_text(log.getvalue())
    return {'tests_run':run.testsRun,'failures':len(run.failures),'errors':len(run.errors),'passed':run.wasSuccessful()}


def raw_join_sample(selected):
    # Independent union of raw viewing fragments, then direct same-channel schedule scan.
    spans=defaultdict(list);count=0;lo,hi=validation.WINDOWS['history']
    for r in records(SOURCE/'dataset11-30.csv'):
        if r['user_id'] not in selected:continue
        s,e=local_stamp(r['start_time']),local_stamp(r['stop_time'])
        if e-s!=int(r['duraton']):raise ValueError('Raw elapsed duration mismatch')
        s,e=max(lo,s),min(hi,e)
        if s<e:spans[(r['user_id'],int(r['vsetv_id']))].append((s,e));count+=1
    merged={};group_channels=defaultdict(list)
    for (u,ch),values in spans.items():
        union=[]
        for s,e in sorted(values):
            if union and s<=union[-1][1]:union[-1]=(union[-1][0],max(e,union[-1][1]))
            else:union.append((s,e))
        merged[(u,ch)]=(union,[e for s,e in union]);group_channels[ch].append(u)
    airings=defaultdict(set)
    for name in ('export_arh_11-20-final.csv','export_arh_21-30-final.csv'):
        for r in records(SOURCE/name):
            ch=int(r['channel_id']);item=int(r['tv_show_id']);duration=int(r['duration'])
            if ch not in group_channels or item<=0 or duration<=0:continue
            s=local_stamp(r['start_time'],True);e=s+duration
            if s<hi and e>lo:airings[ch].add((s,e,str(item)))
    expected={}
    for ch,values in airings.items():
        for u in group_channels[ch]:
            union,ends=merged[(u,ch)]
            for s,e,item in values:
                fragments=[]
                for j in range(bisect.bisect_right(ends,s),len(union)):
                    a,b=union[j]
                    if a>=e:break
                    if max(s,a)<min(e,b):fragments.append((max(s,a),min(e,b)))
                if fragments:expected[(u,ch,s,e,item)]=(sum(b-a for a,b in fragments),fragments[0][0],fragments[-1][1])
    actual={}
    for r in records(validation.STEP2/'history_airing_overlaps.csv.gz'):
        if r['user_id'] in selected:
            actual[(r['user_id'],int(r['channel_id']),int(r['broadcast_start_utc']),int(r['broadcast_end_utc']),r['tv_show_id'])]=tuple(int(r[c]) for c in ('overlap_seconds','first_overlap_utc','last_overlap_utc'))
    return {'selected_users':sorted(selected),'raw_view_rows_intersecting_history':count,'expected_overlap_rows':len(expected),
        'saved_overlap_rows':len(actual),'exact_dictionary_match':expected==actual,
        'scope':'Four users, all their raw views and labeled historical airings; independent merge/join and timestamp conversion'}


def weekly_and_labels(cohort):
    stats={};truth80={u:Counter() for u in cohort};truth120={u:Counter() for u in cohort}
    dates={}
    def day(stamp):
        if stamp not in dates:dates[stamp]=local_date(stamp)
        return dates[stamp]
    for period in ('train','validation','history'):
        lo,hi=validation.WINDOWS[period];base=local_date(lo)
        counters=Counter();examples=[];pairs_affected=set()
        for r in records(validation.STEP2/f'{period}_airing_overlaps.csv.gz'):
            first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
            s,e=int(r['broadcast_start_utc']),int(r['broadcast_end_utc']);seconds=int(r['overlap_seconds'])
            counters['rows']+=1
            if not lo<=first<last<=hi or not 0<seconds<=e-s:counters['invalid_bounds']+=1
            if int(r['tv_show_id'])<=0:counters['invalid_program']+=1
            actual_week=(day(first)-base).days//7;fixed=(first-lo)//WEEK
            if fixed!=actual_week:
                counters['fixed_utc_bucket_differs_from_calendar_first_overlap']+=1
                pairs_affected.add((r['user_id'],r['tv_show_id']))
                if len(examples)<4:examples.append({'user_id':r['user_id'],'program_id':r['tv_show_id'],
                    'first_overlap_local':datetime.fromtimestamp(first,ZONE).isoformat(),'fixed_bucket':fixed,'calendar_bucket':actual_week})
            if day(s).isocalendar()[:2]!=day(first).isocalendar()[:2]:counters['broadcast_start_week_differs_from_first_view_week']+=1
            if period=='validation' and r['user_id'] in truth80:
                if seconds>=120:truth120[r['user_id']][r['tv_show_id']]+=1
                if seconds*5>=4*(e-s):truth80[r['user_id']][r['tv_show_id']]+=1
        stats[period]={**counters,'affected_user_program_pairs':len(pairs_affected),'examples':examples}
    top=lambda counts:{u:set(sorted(c,key=lambda i:(-c[i],int(i)))[:5]) for u,c in counts.items()}
    return stats,top(truth120),top(truth80)


def schedule_week_audit():
    output={};metadata={};cache={}
    for name,period in [('export_arh_11-20-final.csv','train'),('export_arh_21-30-final.csv','validation'),('export_arh_31-42-final.csv','test')]:
        lo,hi=validation.WINDOWS[period];base=local_date(lo);nweeks=10 if period!='test' else 12
        c=Counter();weeks=Counter();examples=[]
        for r in records(SOURCE/name):
            c['all_rows']+=1
            text=r['start_time']
            if text not in cache:
                dt=datetime.strptime(text,'%d.%m.%Y %H:%M:%S').replace(tzinfo=ZONE)
                cache[text]=(int(dt.timestamp()),dt.date(),dt.isocalendar().week)
            s,day,iso=cache[text];duration=int(r['duration']);weeks[iso]+=1
            item=str(int(r['tv_show_id']))
            if item!='0':metadata.setdefault(item,{'title':r['tv_show_title'],'category':r['tv_show_category']})
            if s<lo or s>=hi:c['starts_outside_named_window']+=1
            if duration<=0 or item=='0':continue
            c['positive_valid_airings']+=1
            fixed=(s-lo)//WEEK;calendar=(day-base).days//7
            if fixed!=calendar:c['utc_vs_calendar_bucket_mismatches']+=1
            if period=='test' and not s<hi and s+duration>lo:c['v4_positive_airings_after_test_window']+=1
            if period=='test' and s+duration<=lo:c['v4_positive_airings_before_test_window']+=1
            if not 0<=calendar<nweeks:
                c['calendar_bucket_outside_nominal_range']+=1
                if len(examples)<3:examples.append({'item':item,'start':text,'calendar_bucket':calendar,'clamped_bucket':max(0,min(nweeks-1,fixed))})
        output[name]={**c,'actual_iso_week_row_counts':dict(sorted(weeks.items())),'examples':examples}
    return output,metadata


def main():
    started=time.monotonic()
    if OUT.exists():raise FileExistsError('Preserve audit evidence; use a new output directory')
    OUT.mkdir()
    protected=list((ROOT/'submission').glob('*.csv'))+list((ROOT/'evidence').glob('*session*.json'))
    protected+=list(STEP10.glob('*.json'))+list((STEP10/'models').glob('*.txt'))
    protected+=[STEP10/'candidate_features_top50.csv.gz',STEP10/'v4_test_candidate_features.csv.gz']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    test_results=tests();print('Join/evaluator tests complete',test_results,flush=True)
    protocol=validation.EvaluationProtocol();users=protocol.cohort
    training=json.loads((STEP10/'lightgbm_training_result.json').read_text());columns=training['feature_columns']
    tr=list(records(STEP10/'candidate_features_top50.csv.gz'));te=list(records(STEP10/'v4_test_candidate_features.csv.gz'))
    X=np.array([[float(r[c]) for c in columns] for r in tr],dtype=np.float32)
    TX=np.array([[float(r[c]) for c in columns] for r in te],dtype=np.float32)
    held=tuple(u for u in users if any(r['user_id']==u and r['group_split']=='evaluation_users' for r in tr))
    hset=set(held);mask=np.array([r['user_id'] in hset for r in tr]);hr=[r for r in tr if r['user_id'] in hset]
    model=lgb.Booster(model_file=str(ROOT/training['model_path']))
    scores=model.predict(X[mask],num_threads=4);grouped=defaultdict(list)
    for r,s in zip(hr,scores):grouped[r['user_id']].append((r,float(s)))
    v2=load_preds(ROOT/'results/step8/validation_predictions/rrf_ppr_personal_frequency_ppr_weight_0.50/predictions.csv')
    v3=load_preds(ROOT/'results/step9_cbf/best_validation_candidate.csv')
    pop=load_preds(ROOT/'results/step4/popularity_distinct_users_predictions.csv')
    v4={};alt_tie={};tied5=0
    for u in held:
        order=sorted(grouped[u],key=lambda x:(-x[1],int(x[0]['tv_show_id'])))
        alt=sorted(grouped[u],key=lambda x:(-x[1],float(x[0]['popularity_rank']),int(x[0]['tv_show_id'])))
        cold=float(order[0][0]['user_cold_start'])>.5
        v4[u]=pop[u] if cold else tuple(r['tv_show_id'] for r,s in order[:5])
        alt_tie[u]=pop[u] if cold else tuple(r['tv_show_id'] for r,s in alt[:5])
        tied5+=len(order)>5 and order[4][1]==order[5][1]
    validation_candidates=defaultdict(set);test_candidates=defaultdict(set)
    for r in tr:validation_candidates[r['user_id']].add(r['tv_show_id'])
    for r in te:test_candidates[r['user_id']].add(r['tv_show_id'])
    val_coverage,missing=coverage(v2,validation_candidates,held);table('validation_v2_candidate_omissions.csv',missing)
    final_v2=load_preds(ROOT/'submission/final_submission_v2.csv');final_v4=load_preds(ROOT/'submission/final_submission_v4.csv')
    test_coverage,missing_test=coverage(final_v2,test_candidates,users);table('submission_v2_candidate_omissions.csv',missing_test)
    print('Saved V4 evaluation model replayed',flush=True)
    week_stats,secondary,fraction80=weekly_and_labels(set(users))
    saved_sec={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    if secondary!=saved_sec:raise ValueError('Independent 120-second Top5 labels disagree')
    schedule_stats,metadata=schedule_week_audit()
    metrics={}
    for name,truth in [('primary_all_120',protocol.truth),('top5_120',secondary),('top5_fraction80',fraction80)]:
        metrics[name]=pair_comparison(v2,v4,truth,held)
        metrics[name]['v3_map_same_users']=float(np.mean([reference_ap(truth[u],v3[u]) for u in held]))
    per_user=[];programs=defaultdict(Counter);history_sizes={}
    for r in hr:history_sizes[r['user_id']]=int(float(r['user_history_program_count']))
    for u in held:
        a,b=reference_ap(secondary[u],v2[u]),reference_ap(secondary[u],v4[u])
        aa,bb=contributions(secondary[u],v2[u]),contributions(secondary[u],v4[u])
        per_user.append({'user_id':u,'history_programs':history_sizes[u],'v2_ap':a,'v4_ap':b,'delta':b-a,
            'v2_top5':' '.join(v2[u]),'v4_top5':' '.join(v4[u]),'relevant_top5':' '.join(sorted(secondary[u],key=int)),
            'v2_relevant_omitted_from_candidates':' '.join(sorted((set(v2[u])&secondary[u])-validation_candidates[u],key=int)),
            'relevant_dropped':' '.join(sorted((set(v2[u])-set(v4[u]))&secondary[u],key=int))})
        for i in set(v2[u])|set(v4[u]):
            p=programs[i];p['v2_recommended_users']+=i in v2[u];p['v4_recommended_users']+=i in v4[u]
            p['relevant_drop_users']+=i in secondary[u] and i in v2[u] and i not in v4[u]
            p['relevant_gain_users']+=i in secondary[u] and i in v4[u] and i not in v2[u]
            p['v2_ap_contribution']+=aa.get(i,0);p['v4_ap_contribution']+=bb.get(i,0)
    per_user.sort(key=lambda r:(r['delta'],r['user_id']));table('paired_users.csv',per_user)
    pr=[]
    for i,p in programs.items():
        pr.append({'program_id':i,**metadata.get(i,{'title':'','category':''}),**{k:p[k] for k in ('v2_recommended_users','v4_recommended_users','relevant_drop_users','relevant_gain_users','v2_ap_contribution','v4_ap_contribution')},
            'map_contribution_delta':(p['v4_ap_contribution']-p['v2_ap_contribution'])/len(held)})
    pr.sort(key=lambda r:r['map_contribution_delta']);table('paired_programs.csv',pr)
    strata={}
    for name,lo,hi in [('cold',0,0),('light_1_20',1,20),('medium_21_100',21,100),('heavy_101_plus',101,100000)]:
        us=[u for u in held if lo<=history_sizes[u]<=hi]
        if us:strata[name]={'users':len(us),**pair_comparison(v2,v4,secondary,us)}
    shifts=[]
    for j,c in enumerate(columns):
        a=X[:,j];b=TX[:,j]
        shifts.append({'feature':c,'train_min':float(a.min()),'train_median':float(np.median(a)),
            'train_p95':float(np.quantile(a,.95)),'train_max':float(a.max()),'test_min':float(b.min()),
            'test_median':float(np.median(b)),'test_p95':float(np.quantile(b,.95)),'test_max':float(b.max()),
            'test_fraction_above_train_max':float(np.mean(b>a.max())),'train_constant':bool(np.all(a==a[0])),
            'train_nonzero_rows':int(np.count_nonzero(a)),'test_nonzero_rows':int(np.count_nonzero(b))})
    table('feature_distribution_shift.csv',shifts)
    full=lgb.Booster(model_file=str(ROOT/training['final_refit_model_path']))
    gain=full.feature_importance('gain');importance=[{'feature':c,'gain':float(g),'split_count':int(s)} for c,g,s in zip(columns,gain,full.feature_importance('split'))]
    importance.sort(key=lambda r:-r['gain']);table('final_model_importance.csv',importance)
    random_check=random.Random(2026);max_error=0
    for _ in range(1000):
        labels=set(random_check.sample([str(i) for i in range(1,20)],random_check.randrange(20)))
        ranking=[str(random_check.randrange(1,20)) for _ in range(random_check.randrange(8))]
        max_error=max(max_error,abs(reference_ap(labels,ranking)-evaluation.average_precision_at_5(labels,ranking)))
    strict_errors=[]
    for truth in (protocol.truth,secondary,fraction80):
        for preds in (v2,v4,v3):
            measured=evaluation.evaluate_predictions({u:preds[u] for u in held},{u:truth[u] for u in held},held,protocol.candidates)['primary_map_at_5']
            strict_errors.append(abs(measured-float(np.mean([reference_ap(truth[u],preds[u]) for u in held]))))
    selected=set([per_user[0]['user_id'],per_user[1]['user_id'],'13571199046623737884','8150816946177670084'])
    print('Independent calendar-week and paired-loss scans complete; checking raw join sample',flush=True)
    join_sample=raw_join_sample(selected)
    original_ids=[r['user_id'] for r in records(SOURCE/'submission.csv')]
    identity={}
    for p in (ROOT/'submission').glob('*.csv'):
        rows=list(records(p));identity[p.name]={'rows':len(rows),'exact_original_order':[r['user_id'] for r in rows]==original_ids,
            'unique_users':len({r['user_id'] for r in rows}),'sha256':sha(p)}
    after={name:sha(ROOT/name) for name in before};preserved=before==after
    group_train={r['user_id'] for r in tr if r['group_split']=='training_users'}
    summary={'user_reported_kaggle_scores':{'V1':[.37269,.39316],'V2':[.37880,.40074],'V3':[.35282,.37196],'V4':[.35210,.36523]},
        'test_suites':test_results,'metric_max_random_error':max_error,'metric_max_full_data_error':max(strict_errors),
        'heldout_users':len(held),'supervised_training_users':len(group_train),'supervised_groups_disjoint':not group_train&hset,
        'paired_metrics':metrics,'history_strata':strata,'worst_users':per_user[:8],'worst_programs':pr[:12],
        'v2_candidate_coverage_validation':val_coverage,'v2_candidate_coverage_submission':test_coverage,
        'heldout_tie_break_policy_changes':{'fifth_sixth_score_ties_users':tied5,'ranking_changes':sum(v4[u]!=alt_tie[u] for u in held),
            'popularity_tiebreak_secondary_map':float(np.mean([reference_ap(secondary[u],alt_tie[u]) for u in held]))},
        'labels':{'primary_mean_relevant':float(np.mean([len(protocol.truth[u]) for u in users])),
            'top5_120_nonempty':sum(bool(secondary[u]) for u in users),'top5_80_nonempty':sum(bool(fraction80[u]) for u in users),
            'top5_120_vs80_mean_set_overlap':float(np.mean([len(secondary[u]&fraction80[u]) for u in users]))},
        'weekly_overlap_audit':week_stats,'calendar_schedule_audit':schedule_stats,'independent_raw_join':join_sample,
        'feature_shifts':shifts,'importance_top15':importance[:15],'model_rounds':full.current_iteration(),
        'model_parameters':full.params,'saved_cv_results':training['cv_results'],'saved_ablations':training['ablations'],
        'user_identity':identity,'original_ids_changed_if_float64_used':sum(str(int(float(u)))!=u for u in original_ids),
        'protected_files_unchanged':preserved,'protected_hashes':before,'elapsed_seconds':time.monotonic()-started,
        'no_new_model_trained':True,'no_new_submission_written':True}
    dump('audit.json',summary)
    print(json.dumps({k:summary[k] for k in ('paired_metrics','v2_candidate_coverage_validation','v2_candidate_coverage_submission','independent_raw_join','protected_files_unchanged','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
