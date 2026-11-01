"""V11 model schema, ranking, metrics and deterministic hard-negative sampling."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from collections import defaultdict,Counter
import hashlib,csv,resource
import numpy as np
import lightgbm as lgb
from v8_common import ROOT,SOURCE,read,load,sha,records,map5,inner_fold,partition
from train_v9 import new_matrix,rankings
from prepare_v11 import OUT,V8,V9,V10,dump
from run_v7 import grouping

def model_data(phase='validation'):
    raw=load(OUT/f'{phase}.npz');meta=read(OUT/f'{phase}_build.json');M,names=new_matrix(raw,meta['context']['historical_users'])
    z={**raw,'X':M,'columns':np.asarray(names)}
    if phase=='test':truth=None
    elif phase=='early':truth=read(OUT/'early_labels.json')
    else:truth=read(V8/'labels.json')['secondary']
    return z,truth

def subset(z,mask):
    return {k:(v[mask] if k!='columns' and isinstance(v,np.ndarray) and v.ndim and len(v)==len(z['users']) else v) for k,v in z.items()}

def ranks(score,z,users,mask=None):
    ix=np.arange(len(score)) if mask is None else np.flatnonzero(mask)
    return rankings(np.asarray(score)[ix],z,ix,users)

def profiles(period):
    prof=defaultdict(set)
    for r in records(ROOT/f'data/step2/threshold_120s/{period}_interactions.csv.gz'):prof[r['user_id']].add(r['tv_show_id'])
    return prof

def measured(pred,truth,users,prof):
    d=Counter();n=0
    for u in users:
        r=pred[u];n+=len(r);rel=set(truth[u])
        for i in rel:
            flag='seen' if i in prof.get(u,set()) else 'unseen';d[flag+'_labels']+=1;d[flag+'_candidate_hits']+=i in r
        for k in (5,10,20):d[f'hits{k}']+=len(set(r[:k])&rel)
    den=d['seen_labels']+d['unseen_labels']
    return {'users':len(users),'MAP5':map5(pred,truth,users),'candidate_recall':(d['seen_candidate_hits']+d['unseen_candidate_hits'])/den if den else None,
        **{g+'_candidate_recall':d[g+'_candidate_hits']/d[g+'_labels'] if d[g+'_labels'] else None for g in ('seen','unseen')},
        **{f'Recall{k}':d[f'hits{k}']/den if den else None for k in (5,10,20)},'average_candidates':n/len(users),'counts':dict(d)}

def save_pred(name,pred,full=False):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(v if full else v[:5])) for u,v in pred.items())

def digest_split(u):return int(hashlib.sha256(('v11-miner:'+u).encode()).hexdigest()[:8],16)%2

def sample_negatives(z,indices,scores,hard=20,random_count=20):
    """Keep every positive plus top20 OOF negatives and20 seeded other negatives."""
    selected=[];groups=defaultdict(list);names=z['columns'].tolist();pop=names.index('popularity_rank')
    for n in indices:groups[str(z['users'][n])].append(int(n))
    for u,ix in groups.items():
        positives=[n for n in ix if z['binary_label'][n]];negative=[n for n in ix if not z['binary_label'][n]]
        negative.sort(key=lambda n:(-float(scores[n]),float(z['X'][n,pop]),int(z['items'][n])))
        high=negative[:hard];remaining=negative[hard:]
        # Hash ordering is stable across row order and avoids process-randomized Python hashing.
        remaining.sort(key=lambda n:hashlib.sha256(('2026:v11-negative:'+u+':'+str(z['items'][n])).encode()).digest())
        selected.extend(sorted(positives+high+remaining[:random_count]))
    return np.asarray(selected,dtype=np.int64)

def fit(z,indices,path,params):
    if path.exists():raise FileExistsError('Preserve saved model')
    group=grouping(z['users'][indices]);bst=lgb.train(params,lgb.Dataset(z['X'][indices],label=z['binary_label'][indices],group=group,feature_name=z['columns'].tolist()),num_boost_round=78)
    bst.save_model(str(path));return bst

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
