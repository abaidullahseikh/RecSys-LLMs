"""Independent saved-model/group/metric/streak checks; no fitting or old writes."""
from collections import defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from fractions import Fraction
import csv
import gzip
import hashlib
import json
import math
import time
import numpy as np
import lightgbm as lgb
import evaluation
import validation

ROOT=validation.ROOT;OUT=ROOT/'results/v7_experiment';V5=ROOT/'results/v5_experiment'
ZONE=ZoneInfo('Europe/Kyiv')


def read(p):return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def rational_ap(truth,pred):
    labels=set(truth);pred=list(pred[:5]);n=Fraction(0)
    for item in labels:
        if item in pred:
            r=pred.index(item)+1;n+=Fraction(len(set(pred[:r])&labels),r)
    return float(n/min(5,len(labels))) if labels else 0.


def map5(pred,truth,users):return math.fsum(rational_ap(truth[u],pred[u]) for u in users)/len(users)


def replay(model,M,us,items,cohort,names):
    scores=model.predict(M,num_threads=4);groups=defaultdict(list)
    p=names.index('popularity_rank');c=names.index('user_cold_start')
    for n,u in enumerate(us):groups[str(u)].append(n)
    result={}
    for u in cohort:
        ix=groups[u];cold=M[ix[0],c]>.5
        order=sorted(ix,key=lambda n:((0 if cold else -float(scores[n])),float(M[n,p]),int(items[n])))
        result[u]=tuple(str(items[n]) for n in order[:5])
    return result


def main():
    t=time.monotonic();ex=read(OUT/'experiment.json');plan=read(OUT/'plan.json');selection=read(OUT/'selection.json')
    v5ex=read(V5/'experiment.json');fm=read(V5/'feature_manifest.json');checks=[]
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(name)
    protocol=validation.EvaluationProtocol();cohort=protocol.cohort;eligible=set(protocol.candidates)
    z=load(V5/'validation_features.npz');names=z['columns'].tolist();X=z['X'];us=z['users'];items=z['items']
    dev=tuple(ex['development_users']);outer=tuple(ex['outer_users']);devset=set(dev);oset=set(outer)
    dm=np.isin(us,dev);om=np.isin(us,outer)
    truth={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    check('same_V5_feature_schema_hash',names==v5ex['columns']==ex['feature_schema'] and sha(V5/'validation_features.npz')==fm['validation']['features_sha256'])
    check('same_V5_user_partition',list(dev)==v5ex['development_users'] and list(outer)==v5ex['outer_users'] and not devset&oset and devset|oset==set(cohort))
    check('exact_string_IDs_eligible_unique_pairs',us.dtype.kind in 'US' and items.dtype.kind in 'US' and set(items)<=eligible and len(set(zip(us,items)))==len(us))
    check('historical_boundaries',fm['validation']['trailing_history_utc']==[1583704800,1589749200] and fm['validation']['target_window']==[1589749200,1595797200])
    repeat=np.load(OUT/'repeat_validation.npy',allow_pickle=False);rm=read(OUT/'repeat_feature.json')
    check('repeat_column_hash_shape',sha(OUT/'repeat_validation.npy')==rm['feature_sha256'] and len(repeat)==len(X) and np.isfinite(repeat).all())
    # Different reference algorithm: materialize active Monday dates and walk
    # backward from the last completed week, rather than bit operations.
    weeks=defaultdict(set);wanted=set(zip(us.tolist(),items.tolist()));lo,hi=rm['new']['history_window_utc'];path=ROOT/rm['new']['source_path']
    check('historical_overlap_hash',sha(path)==rm['new']['source_sha256'])
    for r in records(path):
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not first<last<=hi:raise ValueError('Future behavior in repeat history')
        key=r['user_id'],r['tv_show_id']
        if key not in wanted or first<lo or int(r['overlap_seconds'])<120:continue
        d=datetime.fromtimestamp(first,ZONE).date();weeks[key].add(d-timedelta(days=d.weekday()))
    monday=datetime.fromtimestamp(hi,ZONE).date()-timedelta(weeks=1);expected={}
    for key,active in weeks.items():
        count=0;cursor=monday
        while cursor in active and count<10:count+=1;cursor-=timedelta(weeks=1)
        expected[key]=count
    ref=np.asarray([expected.get((str(u),str(i)),0) for u,i in zip(us,items)],dtype=np.float32)
    check('all_repeat_features_reconstructed_from_historical_calendar_dates',np.array_equal(ref,repeat))
    innerfold=np.asarray([int(hashlib.sha256(('v5-inner:'+str(u)).encode()).hexdigest()[:8],16)%3 for u in us])
    print('Repeat feature independently checked; replaying three experiments',flush=True)
    recomputed={}
    for name in plan['experiments']:
        M=np.column_stack([X,repeat]) if name=='E3_repeat_streak' else X
        fn=names+['trailing_consecutive_active_weeks'] if name=='E3_repeat_streak' else names
        oof={}
        for f in range(3):
            ti=np.flatnonzero(dm&(innerfold!=f));vi=np.flatnonzero(dm&(innerfold==f))
            train=set(us[ti]);valid=set(us[vi]);vu=tuple(u for u in dev if u in valid)
            check(f'{name}_fold{f}_disjoint',not train&valid and not (train|valid)&oset)
            # Verify contiguous user query blocks independently from grouping().
            for ix in (ti,vi):
                values=us[ix].tolist();blocks=[values[0]]+[values[n] for n in range(1,len(values)) if values[n]!=values[n-1]]
                if len(blocks)!=len(set(blocks)):raise ValueError('Interleaved ranking query groups')
            model=lgb.Booster(model_file=str(OUT/f'{name}_inner{f}.txt'))
            check(f'{name}_fold{f}_model_schema_rounds',model.feature_name()==fn and model.current_iteration()==selection['cv'][name]['rounds'][f])
            if name=='E2_LambdaRank':check(f'{name}_fold{f}_ranking_objective',model.params['objective']=='lambdarank' and model.params['lambdarank_truncation_level']==8)
            pred=replay(model,M[vi],us[vi],items[vi],vu,fn);oof.update(pred)
        oof={u:oof[u] for u in dev}
        saved=evaluation.parse_prediction_records(list(records(OUT/f'{name}_oof.csv')))
        check(name+'_OOF_predictions_replayed',oof==saved)
        inner=map5(oof,truth,dev)
        check(name+'_OOF_MAP_independent',math.isclose(inner,selection['cv'][name]['oof_MAP'],rel_tol=0,abs_tol=1e-14))
        m=lgb.Booster(model_file=str(OUT/f'{name}_evaluation.txt'))
        check(name+'_outer_model_schema_rounds',m.feature_name()==fn and m.current_iteration()==ex['comparison'][name]['rounds'])
        pred=replay(m,M[om],us[om],items[om],outer,fn)
        saved=evaluation.parse_prediction_records(list(records(OUT/f'{name}_outer.csv')))
        check(name+'_outer_prediction_replay',pred==saved)
        for p in pred.values():evaluation.strict_top5(p,eligible)
        value=map5(pred,truth,outer);primary=map5(pred,protocol.truth,outer)
        check(name+'_outer_metrics_independent',math.isclose(value,ex['comparison'][name]['outer_MAP'],rel_tol=0,abs_tol=1e-14) and math.isclose(primary,ex['comparison'][name]['outer_primary_MAP'],rel_tol=0,abs_tol=1e-14))
        prod=evaluation.evaluate_predictions(pred,{u:truth[u] for u in outer},outer,protocol.candidates)['primary_map_at_5']
        check(name+'_production_evaluator_parity',math.isclose(prod,value,rel_tol=0,abs_tol=1e-14))
        recomputed[name]={'OOF_MAP':inner,'outer_MAP':value,'outer_primary_MAP':primary}
    old=evaluation.parse_prediction_records(list(records(V5/'heldout_v5.csv')))
    check('V5_exact_outer_reproduction',old==evaluation.parse_prediction_records(list(records(OUT/'E1_V5_outer.csv'))) and math.isclose(recomputed['E1_V5']['outer_MAP'],.4886328125,abs_tol=1e-15))
    best=max(plan['experiments'],key=lambda n:recomputed[n]['OOF_MAP'])
    check('development_only_selection_recomputed',best==selection['selected']==ex['selected'])
    gate=best!='E1_V5' and recomputed[best]['OOF_MAP']>recomputed['E1_V5']['OOF_MAP']+1e-12 and recomputed[best]['outer_MAP']>recomputed['E1_V5']['outer_MAP']+1e-12
    check('predeclared_submission_gate',gate==ex['submission_gate_passed'])
    protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if sha(ROOT/p)!=h]
    check('V1_V6_results_and_session_exports_preserved',not changed,changed)
    check('no_forced_V7_submission',gate or not (ROOT/'submission/final_submission_v7_experimental.csv').exists())
    report={'all_passed':True,'passed_checks':len(checks),'failed_checks':0,'checks':checks,
        'independent_scores':recomputed,'selected':best,'CSV_generated':False,'seconds':time.monotonic()-t,
        'scope':'All saved inner and outer model rankings replayed; rational AP/production evaluator agreement; all new streak values independently reconstructed; source hashes, splits, query grouping, schema and prior evidence preservation checked. Original 75-feature engineering reused, not rebuilt.'}
    path=OUT/'verification.json'
    if path.exists():raise FileExistsError('Preserve previous verification')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
