"""Independent CSV verification and saved-model prediction replay for V4.

Replays saved features, not an independent reimplementation of feature engineering.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import re
import time

import lightgbm as lgb
import numpy as np
import validation
import evaluation

ROOT=validation.ROOT
OUT=ROOT/'results'/'step10_lightgbm'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_json(path):return json.loads(path.read_text())


def read_csv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)


def main():
    started=time.monotonic();checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    m=read_json(ROOT/'results/final_model_v4_manifest.json')
    t=read_json(OUT/'lightgbm_training_result.json')
    join=read_json(validation.STEP2/'join_manifest.json')
    source=ROOT.parent/'sweettv-tv-program-recommender'
    header,template=read_csv(source/'submission.csv');users=[r['user_id'] for r in template]
    check('original_template_integrity',sha(source/'submission.csv')==join['source_files_before']['submission.csv']['sha256'])
    pool=set()
    with (source/'export_arh_31-42-final.csv').open(newline='',encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            raw=r['tv_show_id']
            check_id=bool(re.fullmatch(r'[0-9]+',raw))
            if not check_id:raise ValueError('Schedule ID is not numeric')
            if int(raw)>0 and int(r['duration'])>0:pool.add(str(int(raw)))
    check('test_schedule_integrity',sha(source/'export_arh_31-42-final.csv')==join['source_files_before']['export_arh_31-42-final.csv']['sha256'])
    check('candidate_count',len(pool)==m['test_candidate_count']==6636)
    header,records=read_csv(ROOT/m['submission_path'])
    check('exact_header',header==['user_id','tv_show_id'])
    check('exact_template_order_1260_unique_users',[r['user_id'] for r in records]==users and len(set(users))==len(users)==1260)
    pred={}
    for r in records:
        items=r['tv_show_id'].split(' ')
        if set(r)!={'user_id','tv_show_id'} or len(items)!=5 or len(set(items))!=5:raise ValueError('Bad output row')
        if any(not re.fullmatch(r'[1-9][0-9]*',i) or i not in pool for i in items):raise ValueError('Bad output ID')
        pred[r['user_id']]=tuple(items)
    check('five_distinct_positive_canonical_scheduled_ids',len(pred)==1260)
    history=validation.STEP2/'threshold_120s/history_interactions.csv.gz'
    check('historical_input_hash',sha(history)==m['training_history_sha256'])
    support=Counter();airings=Counter();warm=set();n=0
    with gzip.open(history,'rt',newline='',encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            a,s,f,l=map(int,(r['qualifying_airings'],r['watched_seconds'],r['first_watch_utc'],r['last_watch_utc']))
            if not 1583704800<=f<l<=1595797200 or a<=0 or s<120*a:raise ValueError('Historical period/threshold violation')
            support[r['tv_show_id']]+=1;airings[r['tv_show_id']]+=a;warm.add(r['user_id']);n+=1
    check('all_history_rows_within_weeks11_30',n==m['training_history_rows'])
    fallback=sorted(pool,key=lambda i:(-support[i],-airings[i],int(i)))
    tie={i:k for k,i in enumerate(fallback)}
    check('cold_start_shared_fallback',all(pred[u]==tuple(fallback[:5]) for u in users if u not in warm))
    path=ROOT/m['candidate_features_path']
    check('saved_inference_feature_hash',sha(path)==m['candidate_features_sha256'])
    header,rows=read_csv(path);columns=t['feature_columns']
    check('feature_order_and_no_target_columns',header==['user_id','tv_show_id',*columns] and not any('label' in c for c in columns))
    X=np.asarray([[float(r[c]) for c in columns] for r in rows],dtype=np.float32)
    check('finite_feature_matrix',np.isfinite(X).all())
    check('model_hash',sha(ROOT/m['lightgbm_model_path'])==m['lightgbm_model_sha256']==t['final_refit_model_sha256'])
    model=lgb.Booster(model_file=str(ROOT/m['lightgbm_model_path']))
    scores=model.predict(X,num_threads=4)
    grouped=defaultdict(list)
    for r,s in zip(rows,scores):
        if r['user_id'] not in pred or r['tv_show_id'] not in pool:raise ValueError('Inference candidate outside cohort/pool')
        grouped[r['user_id']].append((r['tv_show_id'],float(s)))
    replay={}
    for u in users:
        order=sorted(grouped[u],key=lambda pair:(-pair[1],tie[pair[0]],int(pair[0])))
        replay[u]=tuple(i for i,_ in order[:5]) if u in warm else tuple(fallback[:5])
    check('saved_model_replays_all_1260_rankings',pred==replay)
    # Independently re-evaluate the held-out-user model, never the all-user refit.
    _,training=read_csv(OUT/'candidate_features_top50.csv.gz')
    train_users={r['user_id'] for r in training if r['group_split']=='training_users'}
    held_users={r['user_id'] for r in training if r['group_split']=='evaluation_users'}
    check('disjoint_supervised_user_groups',not train_users&held_users and len(train_users)==1004 and len(held_users)==256)
    check('hash_based_groups',all(r['group_split']==('evaluation_users' if int(hashlib.sha256(r['user_id'].encode()).hexdigest()[:8],16)%5==0 else 'training_users') for r in training))
    held=[r for r in training if r['user_id'] in held_users]
    HX=np.asarray([[float(r[c]) for c in columns] for r in held],dtype=np.float32)
    check('evaluation_model_hash',sha(ROOT/t['model_path'])==t['model_sha256'])
    held_model=lgb.Booster(model_file=str(ROOT/t['model_path']))
    hs=held_model.predict(HX,num_threads=4);groups=defaultdict(list)
    for r,s in zip(held,hs):groups[r['user_id']].append((r['tv_show_id'],float(s),float(r['user_cold_start'])))
    _,pop_rows=read_csv(ROOT/'results/step4/popularity_distinct_users_predictions.csv')
    pop_pred=evaluation.parse_prediction_records(pop_rows)
    hp={u:(pop_pred[u] if groups[u][0][2]>.5 else tuple(i for i,_,_ in sorted(groups[u],key=lambda r:(-r[1],int(r[0])))[:5])) for u in users if u in held_users}
    protocol=validation.EvaluationProtocol();secondary={}
    for r in read_csv(ROOT/'results/step5/secondary_top5_truth.csv')[1]:secondary[r['user_id']]=set(r['top5_validation_program_ids'].split())
    order=tuple(hp);truth={u:secondary[u] for u in order}
    lm=evaluation.evaluate_predictions(hp,truth,order,protocol.candidates)['primary_map_at_5']
    _,v2_rows=read_csv(ROOT/'results/step8/validation_predictions/rrf_ppr_personal_frequency_ppr_weight_0.50/predictions.csv')
    vp=evaluation.parse_prediction_records(v2_rows)
    vm=evaluation.evaluate_predictions({u:vp[u] for u in order},truth,order,protocol.candidates)['primary_map_at_5']
    check('heldout_lightgbm_metric_recomputed',math.isclose(lm,t['heldout_metrics']['top50_secondary_map_at_5'],abs_tol=1e-15))
    check('identical_users_labels_v2_metric_recomputed',math.isclose(vm,t['same_heldout_baselines']['v2_ppr_personal_frequency']['secondary_map_at_5'],abs_tol=1e-15))
    for version,file in [('v1','final_submission.csv'),('v2','final_submission_v2.csv'),('v3','final_submission_v3.csv')]:
        check(version+'_preserved',sha(ROOT/'submission'/file)==m[version+'_sha256_preserved'])
    check('submission_hash',sha(ROOT/m['submission_path'])==m['submission_sha256'])
    (ROOT/'results/final_submission_v4_sha256.txt').write_text(m['submission_sha256']+'\n')
    result={'all_passed':True,'passed_checks':len(checks),'failed_checks':0,'checks':checks,
        'verification_scope':'CSV, provenance, full-history time bounds, saved-feature/model replay, held-out evaluator parity; shared feature implementation is not independently rebuilt',
        'submission_sha256':m['submission_sha256'],'heldout_lightgbm_map_at_5':lm,'heldout_v2_map_at_5':vm,
        'runtime_seconds':time.monotonic()-started,'kaggle_score':None}
    (ROOT/'results/final_submission_v4_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
