"""Bounded expansion of existing generators; no outcome labels enter retrieval."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from collections import Counter,defaultdict
import time,resource,math
import numpy as np
import ppr,association_rules as ar,content_based as cbf,validation
import v6_data as legacy
from features_v5 import window_start,week_age
from v8_common import ROOT,SOURCE,read,load,records,sha
from train_v9 import new_matrix
from prepare_v11 import OUT,dump,V8,V9

LIMIT=50
GENERATORS=('ppr','association_rules','cbf')

def context(lo,cut,end,period,users,pool):
    t=time.monotonic();h=legacy.histories(lo,cut,period);data=h['data'];paths=legacy.SCHEDULES
    hs=legacy.schedule(lo,cut,paths[:1] if cut<=legacy.MID else paths[:2]);ts=legacy.schedule(cut,end,paths,set(pool))
    rollhs=hs if lo>=window_start(cut) else legacy.schedule(window_start(cut),cut,paths[:2])
    graph=ppr.build_graph(data,weight_type='cosine',max_neighbors=30,history_cap=40,min_common_viewers=3)
    tx=tuple((u,b,tuple(sorted(items,key=int))) for (u,b),items in sorted(h['baskets'].items()))
    ds=ar.build_dataset(tx,transaction_type='time_window',window_days=14,training_sha256=data.training_sha256,training_window_utc=(lo,cut),source_sha256=data.training_sha256)
    rules=ar.mine_rules(ds,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.,top_neighbors=50)
    weights=cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'];space=cbf.FeatureSpace.fit(hs['docs'],weights);cm=space.transform(ts['docs'],pool,weights)
    uv,_=cbf.build_user_profiles(data.profiles,space.training_items,{i:j for j,i in enumerate(space.training_items)},space.training_matrix,users,weighting='airings')
    fallback=sorted(pool,key=lambda i:(-data.distinct_viewers.get(i,0),-h['pop_air'][i],int(i)));poppos={i:j for j,i in enumerate(fallback)}
    prefs={f:defaultdict(Counter) for f in ('category','genre')}
    for u in users:
        for i,(n,_,_) in data.profiles.get(u,{}).items():
            for f in prefs:
                for term in hs['docs'].get(i,{}).get(f,set()):prefs[f][u][term]+=n
    bits=defaultdict(int)
    recent_totals=Counter()
    for (u,i),p in h['pairs'].items():recent_totals[u]+=sum(p['week_counts'][:2])
    for r in records(validation.STEP2/f'{period}_airing_overlaps.csv.gz'):
        f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not max(lo,window_start(cut))<=f<l<=cut:continue
        secs=int(r['overlap_seconds']);dur=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if 5*secs>=4*dur:bits[r['user_id'],r['tv_show_id']]|=1<<week_age(f,cut)
    return dict(h=h,hs=hs,rollhs=rollhs,ts=ts,graph=graph,ds=ds,rules=rules,cm=cm,uv=uv,pool=tuple(pool),poolset=set(pool),fallback=fallback,poppos=poppos,prefs=prefs,bits=bits,recent_totals=recent_totals,
        metadata={'history':[lo,cut],'target':[cut,end],'period':period,'history_provenance':h['provenance'],'historical_users':len(data.profiles),'context_seconds':time.monotonic()-t,
            'metadata_vocabulary_fitted_before_cutoff':True,'future_viewing_used':False,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})

def signals(c,u,un):
    h=c['h'];data=h['data'];prof=data.profiles.get(u,{});pool=c['pool'];poppos=c['poppos'];times={}
    t=time.monotonic();ps={}
    if prof:
        restart=ppr.personalization_vector(prof,data,variant='frequency');s,info=ppr.personalized_pagerank(c['graph'],restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
        if not info['converged']:raise ValueError('PPR convergence')
        ps={i:float(s[data.index[i]]) for i in pool if i in data.index and s[data.index[i]]>0}
    times['ppr']=time.monotonic()-t;t=time.monotonic()
    ars,_=ar.score_user(c['rules'],prof,pool,score_type='confidence_log_lift',antecedent_weight='log_frequency');times['association_rules']=time.monotonic()-t;t=time.monotonic()
    cs=(c['uv'][un]@c['cm'].T).toarray().ravel();cbfs={i:float(cs[j]) for j,i in enumerate(pool) if cs[j]>0};times['cbf']=time.monotonic()-t
    raw={'ppr':ps,'association_rules':ars,'cbf':cbfs}
    def finish(a):
        used=set(a);return a+[i for i in c['fallback'] if i not in used]
    rank={g:sorted(d,key=lambda i:(-d[i],-c['ds'].item_support.get(i,0) if g=='association_rules' else poppos[i],int(i))) for g,d in raw.items()}
    rank={g:finish(r) for g,r in rank.items()}
    rank['personal_frequency']=finish(sorted((i for i in prof if i in c['poolset']),key=lambda i:(-prof[i][0],int(i))))
    rank['popularity']=c['fallback']
    return raw,rank,times

def additional(rank,raw,profile,original,limit=LIMIT):
    return tuple(i for i in rank if i not in original and i not in profile and raw.get(i,0)>0)[:limit]

def feature(c,u,i,raw,ranks,names):
    pos={g:ranks[g][i] for g in ranks};src={g+'_rank':v for g,v in pos.items()};src.update({'source_'+g:int(v<50) for g,v in pos.items()})
    src.update(candidate_source_count=sum(v<50 for v in pos.values()),ppr_score=raw['ppr'].get(i,0),association_rules_score=raw['association_rules'].get(i,0),cbf_similarity=raw['cbf'].get(i,0))
    x=legacy.feature_row(u,i,c['h'],c['rollhs'],c['ts'],src,names[:75],c['prefs'])
    bit=c['bits'].get((u,i),0);streak=0
    while bit&1:streak+=1;bit>>=1
    num=sum(c['h']['pairs'].get((u,i),{}).get('week_counts',[0]*10)[:2])
    return np.r_[x,streak,num/max(1,c['recent_totals'][u])].astype(np.float32)

def metrics(candidates,truth,users,profiles):
    d=Counter();n=0
    for u in users:
        cs=set(candidates[u]);n+=len(cs)
        for i in truth[u]:
            flag='seen' if i in profiles.get(u,{}) else 'unseen';d[flag+'_labels']+=1;d[flag+'_hits']+=i in cs
    return {'users':len(users),'unique_pairs':n,'average_candidates':n/len(users),'candidate_recall':sum(d[k+'_hits'] for k in ('seen','unseen'))/sum(d[k+'_labels'] for k in ('seen','unseen')),
        **{k+'_candidate_recall':d[k+'_hits']/d[k+'_labels'] if d[k+'_labels'] else None for k in ('seen','unseen')},'counts':dict(d)}
