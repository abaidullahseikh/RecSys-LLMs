"""V5 source-backed replay, drift, and candidate audit; no old artifacts written."""
from collections import Counter,defaultdict
import csv
import json
import time
import numpy as np
import lightgbm as lgb
import evaluation
import validation
from features_v5 import ROOT,sha,records
from verify_v5 import rank_reference,ap

OUT=ROOT/'results/v6_experiment'


def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def write_table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def moments(values):
    finite=values[np.isfinite(values)]
    return {'min':float(finite.min()),'max':float(finite.max()),'mean':float(finite.mean()),'std':float(finite.std()),
        'zero_rate':float(np.mean(values==0)),'missing_rate':float(np.mean(~np.isfinite(values))),
        'p10':float(np.quantile(finite,.1)),'median':float(np.median(finite)),'p90':float(np.quantile(finite,.9))}


def main():
    t=time.monotonic()
    if (OUT/'audit.json').exists():raise FileExistsError('Preserve completed V6 audit')
    OUT.mkdir(exist_ok=True)
    paths=list((ROOT/'submission').glob('*.csv'))+list((ROOT/'evidence').glob('*session*.json'))
    paths+=list((ROOT/'results/v5_experiment').glob('*'))+[ROOT/'results/final_model_v5_manifest.json']
    paths+=list((ROOT/'code').glob('*v5*.py'))
    if (OUT/'preservation_before.json').exists():
        before=json.loads((OUT/'preservation_before.json').read_text())
        if any(sha(ROOT/p)!=h for p,h in before.items() if p!='code/audit_v5_source.py'):raise ValueError('Prior artifact changed')
    else:
        before={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
        write_json('preservation_before.json',before)
    ex=json.loads((ROOT/'results/v5_experiment/experiment.json').read_text());s=ex['selection'];outer=ex['outer_users']
    with np.load(ROOT/'results/v5_experiment/validation_features.npz') as z:v={k:z[k] for k in z.files}
    with np.load(ROOT/'results/v5_experiment/test_features.npz') as z:f={k:z[k] for k in z.files}
    names=v['columns'].tolist();mask=np.isin(v['users'],outer)
    vm=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/evaluation_model.txt'))
    scores=vm.predict(v['X'][mask][:,s['feature_indices']],num_threads=4)
    pred=rank_reference(scores,v['X'][mask],v['users'][mask],v['items'][mask],outer,names,s['lgb_weight'])
    saved=evaluation.parse_prediction_records(list(records(ROOT/'results/v5_experiment/heldout_v5.csv')))
    if pred!=saved:raise ValueError('V5 validation replay mismatch')
    users=validation.EvaluationProtocol().cohort
    fm=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/final_model.txt'))
    fp=rank_reference(fm.predict(f['X'][:,s['feature_indices']],num_threads=4),f['X'],f['users'],f['items'],users,names,s['lgb_weight'])
    if fp!=evaluation.parse_prediction_records(list(records(ROOT/'submission/final_submission_v5.csv'))):raise ValueError('V5 final replay mismatch')
    print('V5 validation and final submission reproduced exactly',flush=True)
    truth={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    metric=np.mean([ap(truth[u],pred[u]) for u in outer])
    if not np.isclose(metric,.4886328125,rtol=0,atol=1e-15):raise ValueError('V5 metric mismatch')
    drift=[];peruser=[];indices=[]
    for z in (v,f):
        idx=defaultdict(list)
        for n,u in enumerate(z['users']):idx[str(u)].append(n)
        indices.append(idx)
    for j,name in enumerate(names):
        row={'feature':name}
        for label,z,idx in zip(('validation','inference'),(v,f),indices):
            row.update({f'{label}_{k}':val for k,val in moments(z['X'][:,j].astype(float)).items()})
            means=[];stds=[]
            for u in users:
                x=z['X'][idx[u],j].astype(float);means.append(float(x.mean()));stds.append(float(x.std()))
                peruser.append({'feature':name,'period':label,'user_id':u,**moments(x)})
            row[f'{label}_user_mean_std']=float(np.std(means));row[f'{label}_user_mean_p10']=float(np.quantile(means,.1))
            row[f'{label}_user_mean_p90']=float(np.quantile(means,.9));row[f'{label}_mean_within_user_std']=float(np.mean(stds))
        drift.append(row)
    write_table('feature_drift_all75.csv',drift);write_table('feature_drift_by_user.csv',peruser)
    original=list(records(ROOT/'results/step10_lightgbm/candidate_features_top50.csv.gz'))
    full=defaultdict(set);sources={g:defaultdict(set) for g in ('ppr','personal_frequency','association_rules','cbf','popularity')}
    for r in original:
        u=r['user_id'];i=r['tv_show_id'];full[u].add(i)
        for g,values in sources.items():
            if int(float(r['source_'+g])):values[u].add(i)
    missing=[{'user_id':u,'program_id':i} for u in outer for i in sorted(truth[u]-full[u],key=int)]
    write_table('missing_outer_relevant_programs.csv',missing)
    coverage={}
    denom=sum(len(truth[u]) for u in outer)
    for g,values in sources.items():
        other={u:set().union(*(sources[x][u] for x in sources if x!=g)) for u in outer}
        coverage[g]={'label_hits':sum(len(values[u]&truth[u]) for u in outer),
            'unique_label_hits':sum(len((values[u]-other[u])&truth[u]) for u in outer),
            'recall':sum(len(values[u]&truth[u]) for u in outer)/denom}
    write_json('audit.json',{'v5_validation_replayed':True,'v5_submission_replayed':True,'secondary_map':metric,
        'users':len(outer),'feature_count':len(names),'missing_labels':len(missing),'total_labels':denom,
        'candidate_recall':1-len(missing)/denom,'generator_contributions':coverage,
        'missing_program_counts':Counter(r['program_id'] for r in missing),
        'all_prior_hashes_unchanged':all(sha(ROOT/p)==h for p,h in before.items() if p!='code/audit_v5_source.py'),'seconds':time.monotonic()-t,
        'source_findings':[
          {'file':'code/features_v5.py','lines':'116-148','finding':'Some behavioral totals use last10 weeks, inherited model/channel-affinity/popularity signals retain full20 history at inference. Explicitly documented hybrid time scopes; modeling drift, not future leakage.'},
          {'file':'code/features_v5.py','lines':'72-78','finding':'Airing crossing a calendar-week boundary is credited to only its starting/first-overlap week for schedule availability. Measure before calling a material bug.'},
          {'file':'code/train_v5.py','lines':'53-57','finding':'The local AP helper does not guard duplicates or truncate after rank5. Current saved predictions are unique length5, so reported metrics unaffected; reusable scorer contract is defective.'},
          {'file':'code/features_v5.py','lines':'125-138','finding':'candidate_channel_affinity and candidate_hour_affinity remain inherited from pre-clipping, non-deduplicated V4 target schedule while target counts are replaced. Same-named schedule feature family uses inconsistent target populations.'}
        ]})
    print(json.dumps({'replayed':True,'MAP':metric,'candidate_recall':1-len(missing)/denom,'sources':coverage,'seconds':time.monotonic()-t},indent=2))


if __name__=='__main__':main()
