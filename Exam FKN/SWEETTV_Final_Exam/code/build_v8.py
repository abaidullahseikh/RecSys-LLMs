"""Broader leak-safe validation candidates with exact V5 rows reused for template users.

Only two added signals: >=80%-completion active-week streak and share of the
user's last-two-week qualifying airings attributed to the program.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import math,time,resource
import numpy as np
import ppr,association_rules as ar,content_based as cbf,validation
import v6_data as legacy
from features_v5 import week_age
from v8_common import ROOT,OUT,SOURCE,SIGNALS,records,read,dump,load,sha,top_counts

START,MID,END=1583704800,1589749200,1595797200
PATHS=[SOURCE/f'export_arh_{s}-final.csv' for s in ('11-20','21-30','31-42')]


def targets(users):
    counts={u:Counter() for u in users};c80={u:Counter() for u in users}
    for r in records(validation.STEP2/'validation_airing_overlaps.csv.gz'):
        u=r['user_id']
        if u not in counts:continue
        f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not MID<=f<l<=END:raise ValueError('Target viewing outside validation')
        s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc']);i=r['tv_show_id']
        if not 0<s<=d or int(i)<=0:raise ValueError('Invalid overlap target')
        if s>=120:counts[u][i]+=1
        if s*5>=4*d:c80[u][i]+=1
    return counts,c80


def added_signals(period,end,users,requested):
    lower=int((datetime.fromtimestamp(end,legacy.ZONE)-timedelta(weeks=10)).timestamp())
    bit=defaultdict(int);recent=Counter();user_recent=Counter();accepted=0
    path=validation.STEP2/f'{period}_airing_overlaps.csv.gz'
    join=read(validation.STEP2/'join_manifest.json')
    if sha(path)!=join['artifacts'][path.name]['sha256']:raise ValueError('Signal input integrity')
    for r in records(path):
        u=r['user_id'];f,l=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not f<l<=end:raise ValueError('Future viewing feature contamination')
        if u not in users or f<lower:continue
        s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        age=week_age(f,end)
        if not 0<=age<10:raise ValueError('Invalid local week')
        if s>=120 and age<2:user_recent[u]+=1
        key=(u,r['tv_show_id'])
        if requested is not None and key not in requested:continue
        if s*5>=4*d:bit[key]|=1<<age
        if s>=120 and age<2:recent[key]+=1
        accepted+=1
    return bit,recent,user_recent,{'period':period,'window':[lower,end],'source_path':str(path.relative_to(ROOT)),
        'source_sha256':sha(path),'accepted_candidate_airings':accepted,'signals':list(SIGNALS)}


def values_for(u,i,bit,recent,totals):
    mask=bit.get((u,i),0);streak=0
    while mask&1:streak+=1;mask>>=1
    return [streak,recent.get((u,i),0)/max(1,totals.get(u,0))]


def main():
    t=time.monotonic()
    if (OUT/'validation.npz').exists():raise FileExistsError('Preserve V8 data')
    if not read(OUT/'V5_reproduction.json')['exact']:raise ValueError('Reproduction not complete')
    protocol=validation.EvaluationProtocol();template=protocol.cohort;pool=protocol.candidates;pset=set(pool)
    h=legacy.histories(START,MID,'train');data=h['data']
    outcome_users={r['user_id'] for r in records(validation.STEP2/'threshold_120s/validation_interactions.csv.gz')}
    all_users=tuple(sorted(set(data.profiles)|outcome_users|set(template)))
    count,c80=targets(all_users)
    labels={'secondary':top_counts(count),'fraction80':top_counts(c80),'primary':{u:sorted(v,key=int) for u,v in count.items()}}
    for which,path in [('secondary',ROOT/'results/step5/secondary_top5_truth.csv')]:
        saved={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(path)}
        if any(set(labels[which][u])!=saved[u] for u in template):raise ValueError('Independent >=120s label reconstruction mismatch')
    hs=legacy.schedule(START,MID,[PATHS[0]])
    ts=legacy.schedule(MID,END,PATHS,pset)
    graph=ppr.build_graph(data,weight_type='cosine',max_neighbors=30,history_cap=40,min_common_viewers=3)
    tx=tuple((u,b,tuple(sorted(items,key=int))) for (u,b),items in sorted(h['baskets'].items()))
    ds=ar.build_dataset(tx,transaction_type='time_window',window_days=14,training_sha256=data.training_sha256,
        training_window_utc=(START,MID),source_sha256=data.training_sha256)
    rules=ar.mine_rules(ds,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.,top_neighbors=50)
    space=cbf.FeatureSpace.fit(hs['docs'],cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'])
    cm=space.transform(ts['docs'],pool,cbf.FIELD_WEIGHT_CONFIGS['channel_context_emphasis'])
    uv,_=cbf.build_user_profiles(data.profiles,space.training_items,{i:j for j,i in enumerate(space.training_items)},space.training_matrix,all_users,weighting='airings')
    fallback=sorted(pool,key=lambda i:(-data.distinct_viewers.get(i,0),-h['pop_air'][i],int(i)));poppos={i:j for j,i in enumerate(fallback)}
    v5=load(ROOT/'results/v5_experiment/validation_features.npz');names=v5['columns'].tolist()
    oldidx=defaultdict(list)
    for n,u in enumerate(v5['users']):oldidx[str(u)].append(n)
    preferences={f:defaultdict(Counter) for f in ('category','genre')}
    for u in all_users:
        for i,(n,_,_) in data.profiles.get(u,{}).items():
            for f in preferences:
                for term in hs['docs'].get(i,{}).get(f,set()):preferences[f][u][term]+=n
    bit,recent,recent_total,signalmeta=added_signals('train',MID,set(all_users),None)
    values=[];rowusers=[];rowitems=[];base_source=Counter();recall_by_user={};user_sizes={}
    for un,u in enumerate(all_users):
        prof=data.profiles.get(u,{});user_sizes[u]=len(prof)
        if u in oldidx:
            for n in oldidx[u]:
                i=str(v5['items'][n]);values.append(np.r_[v5['X'][n],values_for(u,i,bit,recent,recent_total)].astype(np.float32));rowusers.append(u);rowitems.append(i)
            base_source['exact_frozen_template_rows']+=len(oldidx[u])
            cand={str(v5['items'][n]) for n in oldidx[u]}
        else:
            pscore={}
            if prof:
                restart=ppr.personalization_vector(prof,data,variant='frequency');s,info=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
                if not info['converged']:raise ValueError('PPR convergence')
                pscore={i:float(s[data.index[i]]) for i in pool if i in data.index and s[data.index[i]]>0}
            ascore,_=ar.score_user(rules,prof,pool,score_type='confidence_log_lift',antecedent_weight='log_frequency')
            cs=(uv[un]@cm.T).toarray().ravel();cscore={i:float(cs[j]) for j,i in enumerate(pool) if cs[j]>0}
            def finish(a):
                seen=set(a);return a+[i for i in fallback if i not in seen]
            rank={
                'ppr':finish(sorted(pscore,key=lambda i:(-pscore[i],poppos[i],int(i)))),
                'personal_frequency':finish(sorted((i for i in prof if i in pset),key=lambda i:(-prof[i][0],int(i)))),
                'association_rules':finish(sorted(ascore,key=lambda i:(-ascore[i],-ds.item_support.get(i,0),int(i)))),
                'cbf':finish(sorted(cscore,key=lambda i:(-cscore[i],poppos[i],int(i)))),'popularity':fallback}
            pos={g:{i:j for j,i in enumerate(v)} for g,v in rank.items()};cand=set().union(*(set(v[:50]) for v in rank.values()))
            for i in sorted(cand,key=int):
                src={g+'_rank':pos[g][i] for g in rank};src.update({'source_'+g:int(pos[g][i]<50) for g in rank})
                src.update(candidate_source_count=sum(pos[g][i]<50 for g in rank),ppr_score=pscore.get(i,0),association_rules_score=ascore.get(i,0),cbf_similarity=cscore.get(i,0))
                x=legacy.feature_row(u,i,h,hs,ts,src,names,preferences)
                values.append(np.r_[x,values_for(u,i,bit,recent,recent_total)].astype(np.float32));rowusers.append(u);rowitems.append(i)
            base_source['new_nontemplate_rows']+=len(cand)
        recall_by_user[u]={'candidates':len(cand),'relevant':len(labels['secondary'][u]),'hits':len(cand&set(labels['secondary'][u])),
            'hits80':len(cand&set(labels['fraction80'][u])),'relevant80':len(labels['fraction80'][u])}
        if un%1000==0:print('Broader candidates built',un,'/',len(all_users),flush=True)
    X=np.vstack(values)
    if not np.isfinite(X).all():raise ValueError('Nonfinite V8 matrix')
    label=np.asarray([i in labels['secondary'][u] for u,i in zip(rowusers,rowitems)],dtype=np.int8)
    counts=np.asarray([count[u].get(i,0) for u,i in zip(rowusers,rowitems)],dtype=np.int32)
    counts80=np.asarray([c80[u].get(i,0) for u,i in zip(rowusers,rowitems)],dtype=np.int32)
    np.savez_compressed(OUT/'validation.npz',X=X,users=np.asarray(rowusers),items=np.asarray(rowitems),columns=np.asarray(names+list(SIGNALS)),
        binary_label=label,future_counts=counts,future_counts80=counts80)
    dump('labels.json',labels);dump('future_airing_counts.json',{'threshold120':count,'threshold80':c80})
    dump('population.json',{'users':list(all_users),'template_users':list(template),'history_sizes':user_sizes,'candidate_recall_by_user':recall_by_user,
        'all_users':len(all_users),'historical_users':len(data.profiles),'outcome_users':len(outcome_users),
        'history_and_outcome_users':len(set(data.profiles)&outcome_users),'union_history_outcome':len(set(data.profiles)|outcome_users),
        'template_absent_from_both':sorted(set(template)-set(data.profiles)-outcome_users),
        'policy':'Union of qualifying train-history or validation-outcome users, plus all template users; includes and reports cold/empty targets. Both-history/outcome cohort reported separately; population chosen independently of model success.',
        'heavy_rule':'101+ distinct qualifying training programs; light1-20; weak1-5; cold0'})
    changed_users=sum(set(labels['secondary'][u])!=set(labels['fraction80'][u]) for u in all_users)
    added=sum(len(set(labels['fraction80'][u])-set(labels['secondary'][u])) for u in all_users)
    removed=sum(len(set(labels['secondary'][u])-set(labels['fraction80'][u])) for u in all_users)
    dump('data_manifest.json',{'columns':names+list(SIGNALS),'rows':len(X),'features':X.shape[1],'candidate_pool':list(pool),
        'feature_file_sha256':sha(OUT/'validation.npz'),'history_provenance':h['provenance'],
        'metadata_fitted_on_training_schedule_only':True,'new_signals':signalmeta,'row_sources':base_source,
        'count_target':'actual >=120s qualifying-airing count for each candidate; log1p transformation at fitting only',
        'unchanged_V5_labels':True,'alternative80_status':'unverified hypothesis, not corrected official truth',
        'label_difference_all':{'users_with_changed_Top5':changed_users,'added_user_program_labels':added,'removed_user_program_labels':removed,'symmetric_difference':added+removed},
        'source_files':{str(p.relative_to(ROOT)):sha(p) for p in [validation.STEP2/'train_airing_overlaps.csv.gz',validation.STEP2/'validation_airing_overlaps.csv.gz']},
        'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    print({'users':len(all_users),'rows':len(X),'features':X.shape[1],'seconds':time.monotonic()-t,'peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},flush=True)


if __name__=='__main__':main()
