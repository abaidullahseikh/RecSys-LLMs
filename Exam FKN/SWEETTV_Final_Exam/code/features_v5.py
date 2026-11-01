"""Corrected calendar-week/completion features layered over frozen V4 candidates.

All behavioral additions use a trailing ten-calendar-week window. Target
schedule counters are timestamp-clipped and broadcast-identity deduplicated.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import validation

ROOT=validation.ROOT
OUT=ROOT/'results'/'v5_experiment'
SOURCE=ROOT.parent/'sweettv-tv-program-recommender'
ZONE=ZoneInfo('Europe/Kyiv')
NEW_NAMES=('completion50_count','completion80_count','completion95_count','completion_mean',
    'completion80_share','completion95_share','watched_airing_fraction_mean','last_watched_week_age',
    'recent_frequency_trend','active_week_fraction','repeat_airing_share','viewing_stickiness',
    'program_week_frequency_cv','personal_airing_share','historical_airings_per_week','historical_duration_mean',
    'target_airings_per_week','target_availability_week_fraction','target_to_history_airing_rate',
    'personal_to_scheduled_airings','missing_personal_history','missing_target_availability',
    'recent_channel_affinity','user_recent_active_weeks','user_recent_airings_per_week')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def local_stamp(s):
    return int(datetime.strptime(s,'%d.%m.%Y %H:%M:%S').replace(tzinfo=ZONE).timestamp())


def window_start(end,weeks=10):
    return int((datetime.fromtimestamp(end,ZONE)-timedelta(weeks=weeks)).timestamp())


def week_age(timestamp,end):
    """0 denotes the last completed ISO calendar week before a Monday cutoff."""
    a=datetime.fromtimestamp(timestamp,ZONE).date()
    b=datetime.fromtimestamp(end,ZONE).date()
    return ((b-a).days-1)//7


def completion_flags(seconds,duration):
    if not 0<seconds<=duration:raise ValueError('Invalid airing overlap/duration')
    return (2*seconds>=duration,5*seconds>=4*duration,20*seconds>=19*duration)


def schedule_catalog(paths,lo,hi,allowed):
    counts=Counter();duration=Counter();max_duration=Counter();weeks=defaultdict(set)
    channels=defaultdict(Counter);hours=defaultdict(Counter);seen=set();cache={}
    for path in paths:
        for r in records(path):
            item=str(int(r['tv_show_id']));d=int(r['duration'])
            if item not in allowed or d<=0:continue
            raw=r['start_time']
            if raw not in cache:
                s=local_stamp(raw);dt=datetime.fromtimestamp(s,ZONE)
                cache[raw]=(s,(dt.isocalendar().year,dt.isocalendar().week),dt.hour)
            s,w,h=cache[raw]
            if s>=hi or s+d<=lo:continue
            key=(r['channel_id'],s,s+d,item)
            if key in seen:continue
            seen.add(key);counts[item]+=1;duration[item]+=d;max_duration[item]=max(max_duration[item],d)
            # Availability belongs to the target portion of a boundary-crossing airing.
            effective=datetime.fromtimestamp(max(s,lo),ZONE).isocalendar()
            weeks[item].add((effective.year,effective.week));channels[item][r['channel_id']]+=1;hours[item][h]+=1
    return counts,duration,max_duration,weeks,channels,hours


def behavioral_features(period,end,requested,users):
    lo=window_start(end);path=validation.STEP2/f'{period}_airing_overlaps.csv.gz'
    join=json.loads((validation.STEP2/'join_manifest.json').read_text())
    if sha(path)!=join['artifacts'][path.name]['sha256']:raise ValueError('Overlap input changed')
    stats={};user_air=Counter();user_sec=Counter();user_weeks=defaultdict(set);user_channels=defaultdict(Counter)
    user_items=defaultdict(set);cross=0;recent_nonzero=Counter();pairs=set(requested)
    for r in records(path):
        u=r['user_id']
        if u not in users:continue
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if first>=end or last>end:raise ValueError('Future behavior in historical input')
        if first<lo:
            cross+=last>lo
            continue  # No attribution of unobserved fragment locations at rolling boundary.
        item=r['tv_show_id'];secs=int(r['overlap_seconds']);duration=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        f50,f80,f95=completion_flags(secs,duration);age=week_age(first,end)
        if not 0<=age<10:raise ValueError('Calendar week outside history')
        if secs>=120:
            user_air[u]+=1;user_sec[u]+=secs;user_weeks[u].add(age);user_channels[u][r['channel_id']]+=1;user_items[u].add(item)
        key=(u,item)
        if key not in pairs:continue
        if key not in stats:stats[key]={'n':0,'seconds':0,'duration':0,'f50':0,'f80':0,'f95':0,'fraction_sum':0.,
            'last':0,'week_counts':[0]*10,'a120':0,'a300':0,'a600':0,'qualifying_seconds':0}
        d=stats[key];d['n']+=1;d['seconds']+=secs;d['duration']+=duration
        d['f50']+=f50;d['f80']+=f80;d['f95']+=f95;d['fraction_sum']+=secs/duration
        d['a300']+=secs>=300;d['a600']+=secs>=600
        if secs>=120:
            d['a120']+=1;d['qualifying_seconds']+=secs;d['last']=max(d['last'],last);d['week_counts'][age]+=1
    return stats,user_air,user_sec,user_weeks,user_channels,user_items,cross


def build(source_path,period,end,target_window,target_paths,names):
    rows=list(records(source_path));users={r['user_id'] for r in rows};requested={(r['user_id'],r['tv_show_id']) for r in rows}
    items={i for u,i in requested};stats,ua,us,uw,uc,ui,cross=behavioral_features(period,end,requested,users)
    history_paths=[SOURCE/'export_arh_11-20-final.csv'] if period=='train' else [SOURCE/'export_arh_11-20-final.csv',SOURCE/'export_arh_21-30-final.csv']
    hc,hd,_,_,_,_=schedule_catalog(history_paths,window_start(end),end,items)
    tc,td,tm,tw,tch,th=schedule_catalog(target_paths,*target_window,items)
    target_weeks=(datetime.fromtimestamp(target_window[1],ZONE).date()-datetime.fromtimestamp(target_window[0],ZONE).date()).days/7
    X=np.empty((len(rows),len(names)+len(NEW_NAMES)),dtype=np.float32)
    for k,r in enumerate(rows):
        u,i=r['user_id'],r['tv_show_id'];d=stats.get((u,i),{})
        n=d.get('n',0);a=d.get('a120',0);wc=d.get('week_counts',[0]*10);active=sum(x>0 for x in wc)
        mean=sum(wc)/10;last=d.get('last',0);fraction=d.get('fraction_sum',0)/n if n else 0
        base={c:float(r[c]) for c in names}
        base.update(user_history_program_count=len(ui[u]),user_history_airings_total=ua[u],user_history_seconds_total=us[u],
            user_history_active_weeks=len(uw[u]),user_program_seen=int(a>0),user_program_airings=a,
            user_program_watched_seconds=d.get('qualifying_seconds',0),
            user_program_log_airings=math.log1p(a),user_program_recency_days=(end-last)/86400 if last else 71.,
            airings_ge_120=a,airings_ge_300=d.get('a300',0),airings_ge_600=d.get('a600',0),
            airings_fraction80=d.get('f80',0),mean_airing_fraction=fraction,program_active_weeks=active,
            program_week_consistency=active/10,program_airings_recent_2w=sum(wc[:2]),
            program_airings_recent_4w=sum(wc[:4]),program_airings_recent_8w=sum(wc[:8]),
            target_airing_count=tc[i],target_duration_total=td[i],target_duration_mean=td[i]/tc[i] if tc[i] else 0,
            target_duration_max=tm[i],target_channel_count=len(tch[i]),target_schedule_week_count=len(tw[i]),target_unique_hours=len(th[i]))
        base['target_hour_entropy']=-sum((v/tc[i])*math.log(v/tc[i]) for v in th[i].values()) if tc[i] else 0
        # Raw count features are rates on a common ten-week equivalent horizon.
        for c in ('target_airing_count','target_duration_total','target_schedule_week_count'):base[c]*=10/target_weeks
        new=dict(zip(NEW_NAMES,[
            d.get('f50',0),d.get('f80',0),d.get('f95',0),fraction,
            d.get('f80',0)/n if n else 0,d.get('f95',0)/n if n else 0,
            d.get('seconds',0)/d.get('duration',1) if n else 0,week_age(last-1,end) if last else 10,
            sum(wc[:2])/2-sum(wc[2:])/8,active/10,max(0,a-1)/a if a else 0,
            active/max(1,len(uw[u])),float(np.std(wc))/mean if mean else 0,
            a/max(1,ua[u]),hc[i]/10,hd[i]/hc[i] if hc[i] else 0,
            tc[i]/target_weeks,len(tw[i])/target_weeks,(tc[i]/target_weeks)/(hc[i]/10) if hc[i] else 0,
            a/max(1,hc[i]),int(not a),int(not tc[i]),
            sum(uc[u][ch]*count for ch,count in tch[i].items())/(max(1,ua[u])*max(1,tc[i])),len(uw[u]),ua[u]/10]))
        X[k]=[base[c] for c in names]+[new[c] for c in NEW_NAMES]
    if not np.isfinite(X).all():raise ValueError('Nonfinite corrected features')
    return X,rows,{'rows':len(rows),'trailing_history_utc':[window_start(end),end],
        'boundary_crossing_rows_excluded_conservatively':cross,'target_window':target_window,
        'recent_2w_nonzero_rows':int(np.count_nonzero(X[:,names.index('program_airings_recent_2w')]))}


def main():
    import time
    t=time.monotonic()
    if OUT.exists():raise FileExistsError('Preserve V5 experiment folder')
    OUT.mkdir()
    old=json.loads((ROOT/'results/step10_lightgbm/lightgbm_training_result.json').read_text())
    names=old['feature_columns'];paths=[SOURCE/f'export_arh_{p}-final.csv' for p in ('11-20','21-30','31-42')]
    spec=[('validation','train',validation.WINDOWS['train'][1],validation.WINDOWS['validation'],paths,
        ROOT/'results/step10_lightgbm/candidate_features_top50.csv.gz'),
        ('test','history',validation.WINDOWS['history'][1],validation.WINDOWS['test'],[paths[2]],
        ROOT/'results/step10_lightgbm/v4_test_candidate_features.csv.gz')]
    evidence={}
    for name,period,end,target,sp,source_path in spec:
        X,rows,meta=build(source_path,period,end,target,sp,names)
        np.savez_compressed(OUT/f'{name}_features.npz',X=X,users=np.array([r['user_id'] for r in rows]),
            items=np.array([r['tv_show_id'] for r in rows]),columns=np.array(names+list(NEW_NAMES)))
        evidence[name]={**meta,'source_path':str(source_path.relative_to(ROOT)),'source_sha256':sha(source_path),
            'features_sha256':sha(OUT/f'{name}_features.npz')}
        print(name,meta,flush=True)
    evidence['features']=names+list(NEW_NAMES)
    evidence['policy']='Trailing 10 local calendar weeks for behavioral additions. Whole airing assigned by first observed overlap date; boundary-straddling rolling-window rows excluded rather than guessed. Target schedule clipped/deduplicated; source candidate eligibility remains frozen.'
    evidence['retained_signals']='Existing PPR/PF/AR/CBF/popularity candidate signals preserved; final signals use full Week11-30 history. Corrected behavioral features use last10 weeks. user_cold_start remains full-history absence.'
    evidence['seconds']=time.monotonic()-t
    (OUT/'feature_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
