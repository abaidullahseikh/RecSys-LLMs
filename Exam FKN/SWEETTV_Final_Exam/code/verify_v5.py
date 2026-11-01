"""Audit V5 scores, sampled feature arithmetic, selection, and optional submission.

No fitting or submission generation. Uses independent prefix AP and candidate
sorting; uses native LightGBM only to replay saved models.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import json
import math
import time
import lightgbm as lgb
import numpy as np
import evaluation
import validation

ROOT=validation.ROOT;OUT=ROOT/'results/v5_experiment';ZONE=ZoneInfo('Europe/Kyiv')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def ap(labels,ranking):
    labels=set(labels);ranking=list(ranking);value=Fraction(0)
    for item in labels:
        if item in ranking:
            k=ranking.index(item)+1;value+=Fraction(len(set(ranking[:k])&labels),k)
    return float(value/min(5,len(labels))) if labels else 0


def rank_reference(scores,X,users,items,cohort,names,weight):
    grouped=defaultdict(list);ci=names.index('user_cold_start');pi=names.index('popularity_rank')
    ai=names.index('ppr_rank');bi=names.index('personal_frequency_rank')
    for n,u in enumerate(users):grouped[str(u)].append(n)
    pred={}
    for u in cohort:
        ix=grouped[u]
        fallback=sorted(ix,key=lambda n:(X[n,pi],int(items[n])))
        if X[ix[0],ci]>.5:ordered=fallback
        else:
            a=sorted(ix,key=lambda n:(-float(scores[n]),X[n,pi],int(items[n])))
            b=sorted(ix,key=lambda n:(-(.5/(60+float(X[n,ai]))+.5/(60+float(X[n,bi]))),X[n,pi],int(items[n])))
            apos={n:j for j,n in enumerate(a)};bpos={n:j for j,n in enumerate(b)}
            ordered=sorted(ix,key=lambda n:(-(weight/(60+apos[n])+(1-weight)/(60+bpos[n])),X[n,pi],int(items[n])))
        pred[u]=tuple(str(items[n]) for n in ordered[:5])
    return pred


def feature_reference(z,period,end):
    """Check 24 deterministic positive/empty candidate pairs against raw saved overlaps."""
    names=z['columns'].tolist();X=z['X'];users=z['users'];items=z['items']
    has=X[:,names.index('completion50_count')]>0
    positive=np.flatnonzero(has);negative=np.flatnonzero(~has)
    ix=np.concatenate([positive[np.linspace(0,len(positive)-1,12,dtype=int)],negative[np.linspace(0,len(negative)-1,12,dtype=int)]])
    wanted={(str(users[n]),str(items[n])):int(n) for n in ix};stats={k:Counter() for k in wanted};weekcounts={k:Counter() for k in wanted}
    end_dt=datetime.fromtimestamp(end,ZONE);start=int((end_dt-timedelta(weeks=10)).timestamp())
    for r in records(validation.STEP2/f'{period}_airing_overlaps.csv.gz'):
        key=r['user_id'],r['tv_show_id']
        if key not in wanted:continue
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if first<start:continue
        if not first<last<=end:raise ValueError('Feature reference temporal boundary')
        secs=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        s=stats[key];s['completion50_count']+=Fraction(secs,d)>=Fraction(1,2)
        s['completion80_count']+=Fraction(secs,d)>=Fraction(4,5)
        s['completion95_count']+=Fraction(secs,d)>=Fraction(19,20)
        if secs>=120:
            age=((end_dt.date()-datetime.fromtimestamp(first,ZONE).date()).days-1)//7
            weekcounts[key][age]+=1
            for w in (2,4,8):s[f'program_airings_recent_{w}w']+=age<w
    for key,n in wanted.items():
        for name in ('completion50_count','completion80_count','completion95_count','program_airings_recent_2w','program_airings_recent_4w','program_airings_recent_8w'):
            if X[n,names.index(name)]!=stats[key][name]:raise ValueError(f'Independent feature mismatch {key}/{name}')
        if X[n,names.index('program_active_weeks')]!=len(weekcounts[key]):raise ValueError('Active week mismatch')
    return len(wanted)


def main():
    start=time.monotonic();checks=[]
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(name)
    ex=json.loads((OUT/'experiment.json').read_text());fm=json.loads((OUT/'feature_manifest.json').read_text())
    selection=ex['selection'];names=ex['columns'];cols=selection['feature_indices']
    for name in ('validation','test'):check(name+'_feature_integrity',sha(OUT/f'{name}_features.npz')==fm[name]['features_sha256'])
    for name in ('evaluation','final'):check(name+'_model_integrity',sha(OUT/f'{name}_model.txt')==ex[name+'_model_sha256'])
    z=np.load(OUT/'validation_features.npz');X=z['X'];users=z['users'];items=z['items']
    check('feature_names_order',z['columns'].tolist()==names)
    outer=ex['outer_users'];dev=ex['development_users'];outerset=set(outer)
    check('supervised_user_groups_disjoint',not set(dev)&outerset and len(dev)==1004 and len(outer)==256)
    check('hashed_outer_user_split',all(int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5==0 for u in outer))
    choices=[(r,v) for r in ex['inner_cv'] for v in r['blend_oof_metrics']]
    best,blend=max(choices,key=lambda rv:(rv[1]['secondary_top5_120'],rv[1]['primary_all120'],-rv[0]['spec']['num_leaves']))
    check('inner_only_selection',best['spec']==selection['spec'] and blend['lgb_weight']==selection['lgb_weight'] and int(np.median(best['rounds']))==selection['rounds'])
    check('history_window_and_cutoff',fm['validation']['trailing_history_utc']==[1583704800,1589749200] and fm['test']['trailing_history_utc']==[1589749200,1595797200])
    check('corrected_features_independent_reference',feature_reference(z,'train',1589749200)==24)
    tz=np.load(OUT/'test_features.npz')
    check('final_features_independent_reference',feature_reference(tz,'history',1595797200)==24)
    mask=np.array([u in outerset for u in users]);model=lgb.Booster(model_file=str(OUT/'evaluation_model.txt'))
    scores=model.predict(X[mask][:,cols],num_threads=4)
    predicted=rank_reference(scores,X[mask],users[mask],items[mask],outer,names,selection['lgb_weight'])
    saved=evaluation.parse_prediction_records(list(records(OUT/'heldout_v5.csv')))
    check('evaluation_model_replay',predicted==saved)
    protocol=validation.EvaluationProtocol()
    truth={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    own=math.fsum(ap(truth[u],predicted[u]) for u in outer)/len(outer)
    prod=evaluation.evaluate_predictions(predicted,{u:truth[u] for u in outer},outer,protocol.candidates)['primary_map_at_5']
    check('independent_AP_and_production_evaluator',math.isclose(own,prod,abs_tol=1e-15) and math.isclose(own,ex['outer_comparisons']['V5_selected_blend']['secondary_top5_120'],abs_tol=1e-15))
    baseline=evaluation.parse_prediction_records(list(records(ROOT/'results/step8/validation_predictions/rrf_ppr_personal_frequency_ppr_weight_0.50/predictions.csv')))
    bm=math.fsum(ap(truth[u],baseline[u]) for u in outer)/len(outer)
    check('same_population_v2_MAP',math.isclose(bm,ex['outer_comparisons']['V2']['secondary_top5_120'],abs_tol=1e-15))
    candidate_sets=defaultdict(set)
    for u,i in zip(users,items):candidate_sets[str(u)].add(str(i))
    candidate_recall=sum(len(candidate_sets[u]&truth[u]) for u in outer)/sum(len(truth[u]) for u in outer)
    check('candidate_eligibility',all(v<=set(protocol.candidates) for v in candidate_sets.values()))
    # Historical submissions and native session exports were snapshotted before this work.
    audit=json.loads((ROOT/'results/v2_v4_diagnosis/audit.json').read_text())
    check('all_prior_protected_files_preserved',all(sha(ROOT/p)==h for p,h in audit['protected_hashes'].items()))
    submission=ROOT/'submission/final_submission_v5.csv';subhash=None
    if submission.exists():
        template=[r['user_id'] for r in records(ROOT.parent/'sweettv-tv-program-recommender/submission.csv')]
        with submission.open(newline='',encoding='utf-8') as f:
            reader=csv.reader(f);check('exact_Kaggle_header',next(reader)==['user_id','tv_show_id']);rows=list(reader)
        check('1260_unique_users_exact_order',len(rows)==1260 and len({r[0] for r in rows})==1260 and [r[0] for r in rows]==template)
        pool={str(int(r['tv_show_id'])) for r in records(ROOT.parent/'sweettv-tv-program-recommender/export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
        finalpred={}
        for r in rows:
            if len(r)!=2:raise ValueError('Extra/missing CSV column')
            p=r[1].split(' ')
            if len(p)!=5 or len(set(p))!=5 or not set(p)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in p):raise ValueError('Invalid submission Top5')
            finalpred[r[0]]=tuple(p)
        check('five_unique_positive_scheduled_ids',True)
        m=lgb.Booster(model_file=str(OUT/'final_model.txt'));tx=tz['X'];ts=m.predict(tx[:,cols],num_threads=4)
        reference=rank_reference(ts,tx,tz['users'],tz['items'],template,names,selection['lgb_weight'])
        check('final_model_and_rankings_replay',finalpred==reference)
        subhash=sha(submission)
        manifest=json.loads((ROOT/'results/final_model_v5_manifest.json').read_text())
        check('manifest_submission_hash',subhash==manifest['submission_sha256'])
        (ROOT/'results/final_submission_v5_sha256.txt').write_text(subhash+'\n')
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':len(checks),'failed_checks':0,'checks':checks,
        'secondary_MAP_V5':own,'secondary_MAP_V2':bm,'outer_candidate_recall_micro':candidate_recall,
        'submission_sha256':subhash,'seconds':time.monotonic()-start,
        'scope':'Independent AP, calendar/completion arithmetic for 48 pairs, saved-model/ranking replay, IDs/order/eligibility and preservation. Candidate generators shared with original implementation; full feature reconstruction not duplicated.'}
    path=OUT/('submission_verification.json' if submission.exists() else 'validation_verification.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
