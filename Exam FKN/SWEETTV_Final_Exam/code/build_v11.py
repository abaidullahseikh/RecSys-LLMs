"""Development-only retrieval diagnosis, then one selected expansion with exact old rows."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import argparse,csv,time,resource
from collections import Counter,defaultdict
import numpy as np
import validation,v6_data as legacy
from v8_common import ROOT,SOURCE,read,load,records,sha
from prepare_v11 import OUT,V8,V9,dump
from v11_retrieval import context,signals,additional,metrics,GENERATORS,LIMIT

def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['diagnose','validation','test','early']);a=parser.parse_args();t=time.monotonic()
    if (OUT/(a.phase+'_build.json')).exists():raise FileExistsError('Preserve existing build')
    baseline=read(OUT/'baseline_reproduction.json');allnames=read(V8/'experiment.json')['feature_names'];truth=None
    if a.phase in ('diagnose','validation'):
        z=load(V8/'validation.npz');users=tuple(read(V8/'population.json')['users']);pool=validation.EvaluationProtocol().candidates;lo,cut,end,period=legacy.START,legacy.MID,legacy.END,'train'
        if a.phase=='diagnose':users=tuple(baseline['development_users'])
        truth=read(V8/'labels.json')['secondary']
    elif a.phase=='test':
        z=load(V8/'test.npz');users=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'));lo,cut,end,period=legacy.START,legacy.END,validation.WINDOWS['test'][1],'history'
        pool=tuple(sorted({str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0},key=int))
    else:
        early=load(ROOT/'results/v6_experiment/early_data.npz');mask=early['mask50'];z={k:early[k][mask] for k in ('X','users','items')};z['columns']=early['columns'];users=tuple(dict.fromkeys(z['users']));truth=read(ROOT/'results/v6_experiment/early_labels.json')['secondary']
        lo,cut,end,period=legacy.START,legacy.EARLY,legacy.MID,'train';pool=tuple(sorted(legacy.schedule(cut,end,legacy.SCHEDULES)['clipped_pool'],key=int))
    index=defaultdict(list)
    for j,u in enumerate(z['users']):index[str(u)].append(j)
    old={u:{str(z['items'][j]) for j in index[u]} for u in users}
    c=context(lo,cut,end,period,users,pool);profiles=c['h']['data'].profiles
    if a.phase=='diagnose':
        orig=metrics(old,truth,users,profiles);expanded={g:{} for g in GENERATORS};generated={g:0 for g in GENERATORS};times=Counter();missing=[];summary=Counter()
        for un,u in enumerate(users):
            raw,rank,tm=signals(c,u,un);times.update(tm);prof=profiles.get(u,{})
            for g in GENERATORS:
                add=additional(rank[g],raw[g],prof,old[u]);expanded[g][u]=old[u]|set(add);generated[g]+=len(add)
            historical_cat=set(c['prefs']['category'][u]);historical_genre=set(c['prefs']['genre'][u])
            for i in sorted(set(truth[u])-old[u],key=int):
                doc=c['ts']['docs'].get(i,{});eligible=i in c['poolset'];available=c['ts']['counts'].get(i,0)>0
                row={'user_id':u,'tv_show_id':i,'seen':i in prof,'matching_category':bool(historical_cat&doc.get('category',set())),
                    'matching_genre':bool(historical_genre&doc.get('genre',set())),'association_positive':raw['association_rules'].get(i,0)>0,'ppr_positive':raw['ppr'].get(i,0)>0,
                    'cbf_positive':raw['cbf'].get(i,0)>0,'eligible':eligible,'target_airings':c['ts']['counts'].get(i,0),'available_in_target_window':available,
                    **{g+'_recovered':i in expanded[g][u] for g in GENERATORS}}
                missing.append(row)
                for k,v in row.items():
                    if isinstance(v,bool):summary[k]+=int(v)
            if un%1000==0:print('diagnosed',un,len(users),flush=True)
        candidates={g:{**metrics(expanded[g],truth,users,profiles),'additional_pairs':generated[g],'additional_relevant_recovered':sum(len((expanded[g][u]-old[u])&set(truth[u])) for u in users),
            'generator_seconds':times[g],'shared_context_seconds':c['metadata']['context_seconds'],'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} for g in GENERATORS}
        # Equal fixed50 unseen extra cap; avoid full-pool explosion. This picks one retrieval hypothesis, not the final ranking configuration.
        chosen=max(GENERATORS,key=lambda g:(candidates[g]['additional_relevant_recovered'],-candidates[g]['additional_pairs'],-times[g]))
        table('missing_development_pairs.csv',missing)
        dump('retrieval_diagnosis.json',{'original':orig,'generators':candidates,'missing_pairs':len(missing),'missing_flags_overlapping':summary,'chosen_generator':chosen,
            'limit_extra_unseen':LIMIT,'definition':'First50 positive-score unseen candidates absent from original pool, in unchanged native generator order; no label-forced candidates',
            'selection':'One bounded retrieval hypothesis chosen by development recovery with equal per-user caps, candidate cost and runtime tie-breaks. Final V11 selection requires ranking MAP, not recall alone.',
            'audit':'Reuse unchanged existing CBF/PPR/AR implementations; only targeted retrieval depth/exclusion is new. Historical-frequency and popularity top50 remain in original union.',
            'no_future_labels_in_generators':True,'context':c['metadata']})
        dump('diagnose_build.json',{'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        print({'chosen':chosen,'original':orig,'generators':candidates,'missing_flags':summary},flush=True);return
    choice=read(OUT/'retrieval_diagnosis.json');g=choice['chosen_generator'];xs=[];us=[];items=[];isold=[];y=[];counts=Counter();diffs=Counter();maxdiff=Counter();alignment=[]
    # Precompute per-user recent totals once, for exact archived recent-share semantics.
    recenttot=Counter()
    for (u,i),p in c['h']['pairs'].items():recenttot[u]+=sum(p['week_counts'][:2])
    for un,u in enumerate(users):
        raw,rank,_=signals(c,u,un);prof=profiles.get(u,{});add=additional(rank[g],raw[g],prof,old[u]);pos={k:{i:n for n,i in enumerate(r)} for k,r in rank.items()}
        oldrows={str(z['items'][j]):j for j in index[u]}
        for i in sorted(old[u]|set(add),key=int):
            if i in oldrows:
                j=oldrows[i];x=z['X'][j].copy()
                if a.phase=='early':
                    bit=c['bits'].get((u,i),0);streak=0
                    while bit&1:streak+=1;bit>>=1
                    x=np.r_[x,streak,sum(c['h']['pairs'].get((u,i),{}).get('week_counts',[0]*10)[:2])/max(1,recenttot[u])].astype(np.float32)
                alignment.append((j,len(xs)))
                # Audit reconstructed definitions on one deterministic old row per user.
                if i==min(old[u],key=int):
                    from v11_retrieval import feature
                    ref=feature(c,u,i,raw,pos,allnames);ref[76]=sum(c['h']['pairs'].get((u,i),{}).get('week_counts',[0]*10)[:2])/max(1,recenttot[u])
                    for k,n in enumerate(allnames):
                        delta=abs(float(ref[k])-float(x[k]))
                        if delta>1e-6:diffs[n]+=1;maxdiff[n]=max(maxdiff[n],delta)
            else:
                from v11_retrieval import feature
                x=feature(c,u,i,raw,pos,allnames);x[76]=sum(c['h']['pairs'].get((u,i),{}).get('week_counts',[0]*10)[:2])/max(1,recenttot[u])
            xs.append(x);us.append(u);items.append(i);isold.append(i in oldrows)
            if truth is not None:y.append(i in truth[u])
        counts['additional_pairs']+=len(add)
        if un%1000==0:print(a.phase,'features',un,len(users),flush=True)
    X=np.vstack(xs)
    if not np.isfinite(X).all() or X.dtype!=np.float32:raise ValueError('Invalid new features')
    if a.phase!='early' and not all(np.array_equal(X[b],z['X'][a]) for a,b in alignment):raise ValueError('Original features changed')
    np.savez_compressed(OUT/f'{a.phase}.npz',X=X,users=np.asarray(us),items=np.asarray(items),columns=np.asarray(allnames),original=np.asarray(isold),**({'binary_label':np.asarray(y,dtype=np.int8)} if truth is not None else {}))
    if a.phase=='early':dump('early_labels.json',truth)
    expanded={u:set(old[u]) for u in users}
    for u,i in zip(us,items):expanded[u].add(i)
    meta={'phase':a.phase,'generator':g,'limit':LIMIT,'rows':len(X),'features':77,'users':len(users),'old_rows_preserved_exactly':a.phase!='early','additional_pairs':counts['additional_pairs'],
        'context':c['metadata'],'source_feature_reconstruction_differences_sample':diffs,'source_feature_max_difference_sample':maxdiff,
        'difference_policy':'Original rows preserved byte-exact. Added rows use timestamp-clipped/deduplicated legacy definitions. Source comparisons expose old template metadata-affinity differences rather than silently changing baselines.',
        'data_sha256':sha(OUT/f'{a.phase}.npz'),'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if truth is not None:meta['original_retrieval']=metrics(old,truth,users,profiles);meta['expanded_retrieval']=metrics(expanded,truth,users,profiles)
    dump(a.phase+'_build.json',meta);print(meta,flush=True)

if __name__=='__main__':main()
