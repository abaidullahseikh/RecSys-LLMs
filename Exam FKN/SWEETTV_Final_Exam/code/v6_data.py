"""Generic chronological feature/candidate builder for V6, preserving V5 arrays.

Old candidate rows reuse exact V5 features except audited schedule affinities.
Additional rows and earlier folds use the same 75-column definitions.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from functools import lru_cache
import csv
import gzip
import json
import math
import resource
import time
import numpy as np
import ppr
import association_rules as ar
import content_based as cbf
import validation
import features_v5 as old

ROOT=validation.ROOT;OUT=ROOT/'results/v6_experiment';SOURCE=old.SOURCE;ZONE=old.ZONE
SCHEDULES=[SOURCE/f'export_arh_{s}-final.csv' for s in ('11-20','21-30','31-42')]
GENERATORS=('ppr','personal_frequency','association_rules','cbf','popularity')
START=1583704800;MID=1589749200;END=1595797200
EARLY=int(datetime(2020,4,13,tzinfo=ZONE).timestamp())
NEW_NAMES=list(old.NEW_NAMES)


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


@lru_cache(maxsize=150000)
def local(stamp):return datetime.fromtimestamp(stamp,ZONE)


@lru_cache(maxsize=150000)
def parse(text):return old.local_stamp(text)


def corrected_ap(labels,items):
    """AP@5 contract includes truncation and duplicate suppression."""
    relevant=set(labels);seen=set();hits=0;terms=[]
    for k,item in enumerate(items[:5],1):
        if item in relevant and item not in seen:hits+=1;terms.append(hits/k)
        seen.add(item)
    return math.fsum(terms)/min(5,len(relevant)) if relevant else 0.


def affinity(preference,schedule):
    a=sum(preference.values());b=sum(schedule.values())
    if not a or not b:return 0.
    return sum(preference.get(k,0)*v for k,v in schedule.items())/(a*b)


def window_contains(first,last,lo,hi):return lo<=first<last<=hi


def histories(lo,hi,period):
    path=validation.STEP2/f'{period}_airing_overlaps.csv.gz'
    join=json.loads((validation.STEP2/'join_manifest.json').read_text())
    if old.sha(path)!=join['artifacts'][path.name]['sha256']:raise ValueError('Overlap integrity')
    full=defaultdict(dict);pair={};channels=defaultdict(Counter);hours=defaultdict(Counter)
    totals=Counter();seconds=Counter();weeks=defaultdict(set);unique=defaultdict(set);baskets=defaultdict(set)
    rolllo=max(lo,old.window_start(hi));cross=Counter();scope=Counter();viewers=Counter();pop_air=Counter();pop_sec=Counter()
    rows=0
    for r in old.records(path):
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not window_contains(first,last,lo,hi):
            if first<hi and last>lo:cross['source_boundary_rows_excluded']+=1
            continue
        u=r['user_id'];i=r['tv_show_id'];secs=int(r['overlap_seconds'])
        dur=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc']);start=int(r['broadcast_start_utc'])
        if int(i)<=0 or not 0<secs<=dur:raise ValueError('Invalid matched airing')
        rows+=1
        if secs>=120:
            x=full[u].setdefault(i,[0,0,0]);x[0]+=1;x[1]+=secs;x[2]=max(x[2],last)
            pop_air[i]+=1;pop_sec[i]+=secs
            baskets[(u,(first-lo)//(14*86400))].add(i)
        if first<rolllo:
            cross['rolling_boundary_rows_excluded']+=last>rolllo
            continue
        key=(u,i)
        d=pair.setdefault(key,{'n':0,'f50':0,'f80':0,'f95':0,'fraction_sum':0.,'seconds':0,'duration':0,
            'last':0,'a120':0,'a300':0,'a600':0,'qualifying_seconds':0,'week_counts':[0]*10})
        f50,f80,f95=old.completion_flags(secs,dur);age=old.week_age(first,hi)
        if not 0<=age<10:raise ValueError('Bad local calendar age')
        d['n']+=1;d['f50']+=f50;d['f80']+=f80;d['f95']+=f95;d['fraction_sum']+=secs/dur
        d['seconds']+=secs;d['duration']+=dur;d['a300']+=secs>=300;d['a600']+=secs>=600
        if secs>=120:
            d['a120']+=1;d['qualifying_seconds']+=secs;d['last']=max(d['last'],last);d['week_counts'][age]+=1
            channels[u][r['channel_id']]+=1;hours[u][local(start).hour]+=1
            totals[u]+=1;seconds[u]+=secs;weeks[u].add(age);unique[u].add(i)
    for u,prof in full.items():viewers.update(prof.keys())
    nodes=tuple(sorted(viewers,key=int));model=ppr.TrainingProfiles(nodes,{i:n for n,i in enumerate(nodes)},dict(full),dict(viewers),old.sha(path),(lo,hi),sum(len(x) for x in full.values()),0.)
    return {'data':model,'pairs':pair,'channels':channels,'hours':hours,'totals':totals,'seconds':seconds,
        'weeks':weeks,'unique':unique,'pop_air':pop_air,'pop_sec':pop_sec,'baskets':baskets,
        'provenance':{'lo':lo,'hi':hi,'rolllo':rolllo,'period':period,'source_sha256':old.sha(path),'accepted_airing_rows':rows,**cross}}


def schedule(lo,hi,paths,eligible=None):
    counts=Counter();duration=Counter();mx=Counter();weeks=defaultdict(set);channels=defaultdict(Counter);hours=defaultdict(Counter)
    docs=defaultdict(lambda:defaultdict(set));seen=set();namedpool=set();clipped=set()
    for path in paths:
        for r in old.records(path):
            i=str(int(r['tv_show_id']));d=int(r['duration'])
            if i=='0' or d<=0:continue
            namedpool.add(i)
            s=parse(r['start_time'])
            if s>=hi or s+d<=lo:continue
            clipped.add(i)
            if eligible is not None and i not in eligible:continue
            key=(r['channel_id'],s,s+d,i)
            if key in seen:continue
            seen.add(key);counts[i]+=1;duration[i]+=d;mx[i]=max(mx[i],d)
            dt=local(max(s,lo));weeks[i].add((dt.isocalendar().year,dt.isocalendar().week))
            channels[i][r['channel_id']]+=1;hours[i][local(s).hour]+=1
            for f,terms in cbf.metadata_tokens(r).items():docs[i][f].update(terms)
    return {'counts':counts,'duration':duration,'max':mx,'weeks':weeks,'channels':channels,'hours':hours,
        'docs':{i:dict(v) for i,v in docs.items()},'clipped_pool':clipped,'named_pool':namedpool,
        'weeks_length':(local(hi).date()-local(lo).date()).days/7,'airing_identities':len(seen)}


def labels(lo,hi,period,users,pool):
    counts={u:Counter() for u in users};f80={u:Counter() for u in users};path=validation.STEP2/f'{period}_airing_overlaps.csv.gz'
    for r in old.records(path):
        u=r['user_id']
        if u not in counts:continue
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not window_contains(first,last,lo,hi):continue
        i=r['tv_show_id'];s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if i not in pool:raise ValueError('Label outside target catalog')
        if s>=120:counts[u][i]+=1
        if s*5>=d*4:f80[u][i]+=1
    top=lambda c:{u:sorted(v,key=lambda i:(-v[i],int(i)))[:5] for u,v in c.items()}
    return {'secondary':top(counts),'fraction80':top(f80),'primary':{u:sorted(c,key=int) for u,c in counts.items()}}


def feature_row(u,i,history,hs,ts,source,names,full_preferences):
    data=history['data'];pair=history['pairs'].get((u,i),{});n=pair.get('n',0);a=pair.get('a120',0)
    wc=pair.get('week_counts',[0]*10);active=sum(x>0 for x in wc);last=pair.get('last',0);mean=sum(wc)/10
    fraction=pair.get('fraction_sum',0)/n if n else 0;hi=data.training_window_utc[1]
    tc=ts['counts'][i];targetweeks=ts['weeks_length'];histweeks=hs['weeks_length'];hc=hs['counts'][i]
    prof=data.profiles.get(u,{});fullold=prof.get(i,[0,0,0])
    ua=history['totals'][u];uw=history['weeks'][u]
    cat=full_preferences['category'][u];genre=full_preferences['genre'][u];catn=sum(cat.values()) or 1;genren=sum(genre.values()) or 1
    doc=ts['docs'].get(i,{})
    row=dict(source)
    row.update(pop_distinct_users=data.distinct_viewers.get(i,0),pop_qualifying_airings=history['pop_air'][i],pop_watched_seconds=history['pop_sec'][i],
        user_history_program_count=len(history['unique'][u]),user_history_airings_total=ua,user_history_seconds_total=history['seconds'][u],
        user_cold_start=int(not prof),user_history_active_weeks=len(uw),user_program_seen=int(a>0),user_program_airings=a,
        user_program_watched_seconds=pair.get('qualifying_seconds',0),user_program_log_airings=math.log1p(a),
        user_program_recency_days=(hi-last)/86400 if last else 71.,airings_ge_120=a,airings_ge_300=pair.get('a300',0),airings_ge_600=pair.get('a600',0),
        airings_fraction80=pair.get('f80',0),mean_airing_fraction=fraction,program_active_weeks=active,program_week_consistency=active/10,
        program_airings_recent_2w=sum(wc[:2]),program_airings_recent_4w=sum(wc[:4]),program_airings_recent_8w=sum(wc[:8]),
        target_airing_count=tc*10/targetweeks,target_duration_total=ts['duration'][i]*10/targetweeks,
        target_duration_mean=ts['duration'][i]/tc if tc else 0,target_duration_max=ts['max'][i],target_channel_count=len(ts['channels'][i]),
        target_schedule_week_count=len(ts['weeks'][i])*10/targetweeks,target_unique_hours=len(ts['hours'][i]),
        target_hour_entropy=-sum((v/tc)*math.log(v/tc) for v in ts['hours'][i].values()) if tc else 0,
        candidate_channel_affinity=affinity(history['channels'][u],ts['channels'][i]),
        candidate_hour_affinity=affinity(history['hours'][u],ts['hours'][i]),
        user_category_affinity=sum(cat.get(x,0) for x in doc.get('category',set()))/catn,
        user_genre_affinity=sum(genre.get(x,0) for x in doc.get('genre',set()))/genren,
        personal_frequency_score=fullold[0])
    vals=[pair.get('f50',0),pair.get('f80',0),pair.get('f95',0),fraction,pair.get('f80',0)/n if n else 0,
        pair.get('f95',0)/n if n else 0,pair.get('seconds',0)/pair.get('duration',1) if n else 0,
        old.week_age(last-1,hi) if last else 10,sum(wc[:2])/2-sum(wc[2:])/8,active/10,max(0,a-1)/a if a else 0,
        active/max(1,len(uw)),float(np.std(wc))/mean if mean else 0,a/max(1,ua),hc/10,
        hs['duration'][i]/hc if hc else 0,tc/targetweeks,len(ts['weeks'][i])/targetweeks,
        (tc/targetweeks)/(hc/10) if hc else 0,a/max(1,hc),int(not a),int(not tc),
        affinity(history['channels'][u],ts['channels'][i]),len(uw),ua/10]
    row.update(zip(NEW_NAMES,vals))
    return np.asarray([row[c] for c in names],dtype=np.float32)


def build(name,lo,cut,end,period,target_period,old_npz=None):
    started=time.monotonic();protocol=validation.EvaluationProtocol();cohort=protocol.cohort
    oldz=None;old_index={};original_pool=None
    if old_npz:
        with np.load(old_npz) as z:oldz={k:z[k] for k in z.files}
        old_index={(str(u),str(i)):n for n,(u,i) in enumerate(zip(oldz['users'],oldz['items']))}
    with np.load(ROOT/'results/v5_experiment/validation_features.npz') as z:names=z['columns'].tolist()
    # Metadata availability only: no target watching is read until generators are fit.
    targetpaths=SCHEDULES if name!='test' else [SCHEDULES[2]]
    ts=schedule(cut,end,targetpaths)
    if name=='main':pool=protocol.candidates
    elif name=='test':pool=tuple(sorted(ts['named_pool'],key=int))
    else:pool=tuple(sorted(ts['clipped_pool'],key=int))
    poolset=set(pool)
    h=histories(lo,cut,period);data=h['data']
    historypaths=SCHEDULES[:1] if cut<=MID else SCHEDULES[:2]
    hs=schedule(lo,cut,historypaths)
    rollhs=hs if lo==max(lo,old.window_start(cut)) else schedule(max(lo,old.window_start(cut)),cut,historypaths)
    graph=ppr.build_graph(data,weight_type='cosine',max_neighbors=30,history_cap=40,min_common_viewers=3)
    tx=tuple((u,b,tuple(sorted(items,key=int))) for (u,b),items in sorted(h['baskets'].items()))
    ds=ar.build_dataset(tx,transaction_type='time_window',window_days=14,training_sha256=data.training_sha256,
        training_window_utc=(lo,cut),source_sha256=data.training_sha256)
    rules=ar.mine_rules(ds,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.,top_neighbors=50)
    space=cbf.FeatureSpace.fit(hs['docs'],cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'])
    cm=space.transform(ts['docs'],pool,cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'])
    uv,_=cbf.build_user_profiles(data.profiles,space.training_items,{i:k for k,i in enumerate(space.training_items)},space.training_matrix,cohort,weighting='airings')
    fallback=sorted(pool,key=lambda i:(-data.distinct_viewers.get(i,0),-h['pop_air'][i],int(i)));poppos={i:n for n,i in enumerate(fallback)}
    pp={f:defaultdict(Counter) for f in ('category','genre')}
    for u in cohort:
        for i,(cnt,_,_) in data.profiles.get(u,{}).items():
            for f in pp:
                for term in hs['docs'].get(i,{}).get(f,set()):pp[f][u][term]+=cnt
    rows=[];users=[];items=[];masks={20:[],50:[],100:[]};changed=Counter();ranknames={g:g+'_rank' for g in GENERATORS}
    # Candidate IDs are strings; numeric conversion is limited to deterministic item tie breaks.
    for un,u in enumerate(cohort):
        prof=data.profiles.get(u,{});ps={}
        if prof:
            restart=ppr.personalization_vector(prof,data,variant='frequency')
            score,info=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            if not info['converged']:raise ValueError('PPR did not converge')
            ps={i:float(score[data.index[i]]) for i in pool if i in data.index and score[data.index[i]]>0}
        ars,_=ar.score_user(rules,prof,pool,score_type='confidence_log_lift',antecedent_weight='log_frequency')
        cs=(uv[un]@cm.T).toarray().ravel()
        cbfs={i:float(cs[j]) for j,i in enumerate(pool) if cs[j]>0}
        def finish(order):
            seen=set(order);return order+[i for i in fallback if i not in seen]
        rank={
            'ppr':finish(sorted(ps,key=lambda i:(-ps[i],poppos[i],int(i)))),
            'personal_frequency':finish(sorted((i for i in prof if i in poolset),key=lambda i:(-prof[i][0],int(i)))),
            'association_rules':finish(sorted(ars,key=lambda i:(-ars[i],-ds.item_support.get(i,0),int(i)))),
            'cbf':finish(sorted(cbfs,key=lambda i:(-cbfs[i],poppos[i],int(i)))), 'popularity':fallback}
        pos={g:{i:j for j,i in enumerate(r)} for g,r in rank.items()}
        sets={k:set().union(*(set(r[:k]) for r in rank.values())) for k in masks}
        if oldz:
            olditems={str(i) for i in oldz['items'][oldz['users']==u]}
            sets[50]=olditems;sets[100]|=olditems;sets[20]&=olditems
        for i in sorted(sets[100],key=int):
            src={ranknames[g]:pos[g][i] for g in GENERATORS}
            src.update({'source_'+g:int(pos[g][i]<50) for g in GENERATORS})
            src.update(candidate_source_count=sum(pos[g][i]<50 for g in GENERATORS),ppr_score=ps.get(i,0),
                association_rules_score=ars.get(i,0),cbf_similarity=cbfs.get(i,0))
            row=feature_row(u,i,h,rollhs,ts,src,names,pp)
            oldn=old_index.get((u,i))
            if oldn is not None:
                preserved=oldz['X'][oldn].copy()
                for c in ('candidate_channel_affinity','candidate_hour_affinity'):
                    j=names.index(c)
                    if abs(float(preserved[j])-float(row[j]))>1e-7:changed[c]+=1
                    preserved[j]=row[j]
                row=preserved
            rows.append(row);users.append(u);items.append(i)
            for k in masks:masks[k].append(i in sets[k])
        if un%400==0:print(name,'users',un,flush=True)
    target=labels(cut,end,target_period,cohort,poolset) if name!='test' else None
    X=np.vstack(rows)
    if not np.isfinite(X).all():raise ValueError('Nonfinite feature')
    np.savez_compressed(OUT/f'{name}_data.npz',X=X,users=np.asarray(users),items=np.asarray(items),columns=np.asarray(names),
        **{f'mask{k}':np.asarray(v) for k,v in masks.items()})
    if target:dump(f'{name}_labels.json',target)
    meta={'name':name,'history':h['provenance'],'target_window':[cut,end],'rows':len(users),'users':len(cohort),'eligible_items':len(pool),
        'fixed_feature_changes':dict(changed),'candidate_limits':[20,50,100],'data_sha256':old.sha(OUT/f'{name}_data.npz'),
        'source_v5_npz':str(old_npz) if old_npz else None,'generation_seconds':time.monotonic()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'zero_label_forced_candidates':True,'metadata_fit_end':cut,'target_behavior_read_only_after_generators':target is not None,
        'shorter_early_history_note':'Early fold has five weeks of history; V5 fixed ten-week-rate denominator retained to represent partial observed history.'}
    dump(f'{name}_build.json',meta);print(meta,flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('period',choices=['main','early','test']);args=parser.parse_args()
    if (OUT/f'{args.period}_data.npz').exists():raise FileExistsError('Preserve completed feature table')
    if args.period=='main':build('main',START,MID,END,'train','validation',ROOT/'results/v5_experiment/validation_features.npz')
    elif args.period=='early':build('early',START,EARLY,MID,'train','train')
    else:build('test',START,END,validation.WINDOWS['test'][1],'history',None,ROOT/'results/v5_experiment/test_features.npz')


if __name__=='__main__':main()
