"""V12 fixed expanded-pool experiment: read-only prior caches and explicit targets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from collections import Counter,defaultdict
from pathlib import Path
import json,csv,time,resource
import numpy as np
import lightgbm as lgb
import validation
from v8_common import ROOT,SOURCE,read,load,sha,records,map5,inner_fold,partition
from train_v9 import new_matrix,rankings
from run_v7 import grouping

OUT=ROOT/'results/v12_experiment';V8=ROOT/'results/v8_experiment';V9=ROOT/'results/v9_experiment';V10=ROOT/'results/v10_experiment';V11=ROOT/'results/v11_experiment'

def dump(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')

def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def data(phase='validation'):
    folder=OUT if phase=='test' else V11;raw=load(folder/f'{phase}.npz');meta=read(folder/f'{phase}_build.json')
    if sha(folder/f'{phase}.npz')!=meta['data_sha256']:raise ValueError('Feature cache hash differs')
    X,names=new_matrix(raw,meta['context']['historical_users']);z={**raw,'X':X,'columns':np.asarray(names)}
    truth=None if phase=='test' else read(V8/'labels.json')['secondary'] if phase=='validation' else read(V11/'early_labels.json')
    return z,truth

def ranks(scores,z,users,mask=None):
    indices=np.arange(len(z['users'])) if mask is None else np.flatnonzero(mask)
    return rankings(np.asarray(scores)[indices],z,indices,users)

def save_pred(name,pred):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(v[:5])) for u,v in pred.items())

def grades(counts):
    a=np.asarray(counts)
    if not np.issubdtype(a.dtype,np.integer) or np.any(a<0):raise ValueError('Counts must be nonnegative integers')
    return np.where(a==0,0,np.where(a==1,1,np.where(a<=3,2,3))).astype(np.int8)

def parameters(name):
    params={**read(V10/'experiment.json')['pairwise_info']['parameters']}
    params['label_gain']=[0,1] if name=='E3' else [0,1,3,7]
    if name not in ('E3','E4'):raise ValueError('Only two new V12 configurations')
    return params

def fit(z,y,indices,path,name):
    if path.exists():raise FileExistsError('Preserve model')
    g=grouping(z['users'][indices])
    if g.sum()!=len(indices):raise ValueError('Bad query group sizes')
    t=time.monotonic();bst=lgb.train(parameters(name),lgb.Dataset(z['X'][indices],label=y[indices],group=g,feature_name=z['columns'].tolist()),num_boost_round=78)
    bst.save_model(str(path));return bst,time.monotonic()-t

def history_profiles(phase='validation'):
    prof=defaultdict(set)
    if phase=='early':
        for r in records(validation.STEP2/'train_airing_overlaps.csv.gz'):
            if 1583704800<=int(r['first_overlap_utc'])<int(r['last_overlap_utc'])<=1586725200 and int(r['overlap_seconds'])>=120:prof[r['user_id']].add(r['tv_show_id'])
    else:
        period='history' if phase=='test' else 'train';lo,hi=validation.WINDOWS[period]
        for r in records(validation.STEP2/f'threshold_120s/{period}_interactions.csv.gz'):
            if not lo<=int(r['first_watch_utc'])<int(r['last_watch_utc'])<=hi:raise ValueError('Historical boundary corruption')
            prof[r['user_id']].add(r['tv_show_id'])
    return prof

def metrics(pred,truth,users,profiles):
    c=Counter();n=0
    for u in users:
        rel=set(truth[u]);p=pred[u];n+=len(p)
        for i in rel:
            g='seen' if i in profiles.get(u,set()) else 'unseen';c[g+'_labels']+=1;c[g+'_candidate_hits']+=i in p;c[g+'_hits5']+=i in p[:5]
        for k in (5,10,20):c[f'hits{k}']+=len(rel&set(p[:k]))
        for i in p[:5]:
            g='seen' if i in profiles.get(u,set()) else 'unseen';c[g+'_top5_slots']+=1;c[g+'_top5_hits']+=i in rel
    den=c['seen_labels']+c['unseen_labels']
    return {'users':len(users),'MAP5':map5(pred,truth,users),'candidate_recall':(c['seen_candidate_hits']+c['unseen_candidate_hits'])/den if den else None,
        **{g+'_candidate_recall':c[g+'_candidate_hits']/c[g+'_labels'] if c[g+'_labels'] else None for g in ('seen','unseen')},
        **{g+'_relevant_recall5':c[g+'_hits5']/c[g+'_labels'] if c[g+'_labels'] else None for g in ('seen','unseen')},
        **{g+'_top5_precision':c[g+'_top5_hits']/c[g+'_top5_slots'] if c[g+'_top5_slots'] else None for g in ('seen','unseen')},
        **{f'Recall{k}':c[f'hits{k}']/den if den else None for k in (5,10,20)},'average_candidates':n/len(users),'counts':dict(c)}

def target_counts(z,phase):
    """Candidate matrix already exists before scanning any future occasions."""
    if phase=='validation':
        period='validation';lo,hi=1589749200,1595797200
    elif phase=='early':period='train';lo,hi=1586725200,1589749200
    else:raise ValueError('No test-window viewing labels exist or are permitted')
    users=set(str(u) for u in z['users']);counts=defaultdict(Counter);p=validation.STEP2/f'{period}_airing_overlaps.csv.gz';jm=read(validation.STEP2/'join_manifest.json')
    if sha(p)!=jm['artifacts'][p.name]['sha256']:raise ValueError('Matched-airing source hash differs')
    for r in records(p):
        f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc']);s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if not 0<s<=d or int(r['tv_show_id'])<=0:raise ValueError('Invalid matched airing')
        if not lo<=f<l<=hi:continue
        if r['user_id'] in users and s>=120:counts[r['user_id']][r['tv_show_id']]+=1
    y=np.asarray([counts[str(u)].get(str(i),0) for u,i in zip(z['users'],z['items'])],dtype=np.int32)
    truth={u:sorted(counts[u],key=lambda i:(-counts[u][i],int(i)))[:5] for u in users}
    return y,truth,{'source_path':str(p),'sha256':sha(p),'target':[lo,hi],'qualifying_seconds':120,
        'occasion':'One matched user/broadcast airing after fragment union, >=120 total overlapping seconds; repeated broadcasts count separately; ID0 excluded',
        'grade_definition':{'0':'count0','1':'count1','2':'count2or3','3':'count4plus'},
        'grade_positive_scope':'All qualifying target programs, including programs outside local frequency Top5; not confirmed official relevance',
        'binary_positive_scope':'Unchanged local frequency Top5 membership, as V10/V11; not all count-positive programs',
        'retrieval_before_labels':True}
