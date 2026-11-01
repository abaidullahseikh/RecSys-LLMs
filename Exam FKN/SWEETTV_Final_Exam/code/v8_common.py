"""Shared read-only I/O and exact V8 ranking/label conventions."""
from collections import defaultdict
from pathlib import Path
import csv,gzip,hashlib,json,math
import numpy as np
import validation

ROOT=validation.ROOT;OUT=ROOT/'results/v8_experiment';SOURCE=ROOT.parent/'sweettv-tv-program-recommender'
SIGNALS=('completion80_active_week_streak','recent2w_personal_airing_share')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())


def dump(name,obj):
    p=OUT/name
    if p.exists():raise FileExistsError(f'Preserve existing artifact {p}')
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def records(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def ap(truth,pred):
    labels=set(truth);seen=set();hit=0;terms=[]
    for k,i in enumerate(pred[:5],1):
        if i in labels and i not in seen:hit+=1;terms.append(hit/k)
        seen.add(i)
    return math.fsum(terms)/min(5,len(labels)) if labels else 0.


def map5(pred,truth,users):
    return math.fsum(ap(truth[u],pred[u]) for u in users)/len(users) if users else None


def partition(u):return int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5


def inner_fold(u):return int(hashlib.sha256(('v8-inner:'+u).encode()).hexdigest()[:8],16)%3


def rank(scores,z,ix,users):
    names=z['columns'].tolist();pop=names.index('popularity_rank');cold=names.index('user_cold_start')
    grouped=defaultdict(list)
    for j,s in zip(ix,scores):grouped[str(z['users'][j])].append((int(j),float(s)))
    pred={}
    for u in users:
        g=grouped[u]
        if not g:raise ValueError('User lacks candidates')
        isc=z['X'][g[0][0],cold]>.5
        g.sort(key=lambda p:((0 if isc else -p[1]),float(z['X'][p[0],pop]),int(z['items'][p[0]])))
        pred[u]=tuple(str(z['items'][j]) for j,s in g[:5])
    return pred


def save_pred(name,pred):
    p=OUT/name
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(v)) for u,v in pred.items())


def top_counts(counts):
    return {u:sorted(c,key=lambda i:(-c[i],int(i)))[:5] for u,c in counts.items()}
