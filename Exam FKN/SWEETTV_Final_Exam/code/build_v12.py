"""Refit V11's bounded PPR retrieval through Week30; consistently recompute all features."""
from v12_common import *
from v11_retrieval import context,signals,additional,feature

def main():
    t=time.monotonic()
    if (OUT/'test.npz').exists() or (OUT/'test_build.json').exists():raise FileExistsError('Preserve final candidate cache')
    raw=load(V8/'test.npz');allnames=raw['columns'].tolist();users=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    if tuple(dict.fromkeys(raw['users']))!=users:raise ValueError('Original template order changed')
    pool=tuple(sorted({str(int(r['tv_show_id'])) for r in records(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0},key=int))
    c=context(1583704800,1595797200,validation.WINDOWS['test'][1],'history',users,pool);oldidx=defaultdict(dict)
    for j,(u,i) in enumerate(zip(raw['users'],raw['items'])):oldidx[str(u)][str(i)]=j
    xs=[];us=[];items=[];oldmask=[];newcounts={};changed=Counter();maxdiff=Counter();checks=[]
    for un,u in enumerate(users):
        source,rank,_=signals(c,u,un);old=set(oldidx[u]);add=additional(rank['ppr'],source['ppr'],c['h']['data'].profiles.get(u,{}),old,50);positions={g:{i:k for k,i in enumerate(v)} for g,v in rank.items()}
        if un<3:
            s2,r2,_=signals(c,u,un)
            if source!=s2 or rank!=r2:raise ValueError('Nondeterministic final retrieval')
            checks.append({'user_id':u,'repeat_full_generator_scores_and_ranks':True})
        for i in sorted(old|set(add),key=int):
            # Same full-history generators / trailing10wk behavioral definitions as expanded validation.
            # Recompute old rows as well: old V5 final channel/hour-affinity columns used20wk context.
            x=feature(c,u,i,source,positions,allnames)
            if i in oldidx[u]:
                original=raw['X'][oldidx[u][i]];diff=np.abs(x.astype(float)-original.astype(float))
                for k in np.flatnonzero(diff>1e-6):changed[allnames[k]]+=1;maxdiff[allnames[k]]=max(maxdiff[allnames[k]],float(diff[k]))
            xs.append(x);us.append(u);items.append(i);oldmask.append(i in old)
        newcounts[u]=len(add)
        if un%300==0:print('V12 final candidates',un,len(users),flush=True)
    X=np.vstack(xs)
    if X.dtype!=np.float32 or not np.isfinite(X).all() or len(set(zip(us,items)))!=len(us) or max(newcounts.values())>50:raise ValueError('Invalid final expanded feature matrix')
    z={'X':X,'users':np.asarray(us),'items':np.asarray(items),'columns':np.asarray(allnames),'original':np.asarray(oldmask)}
    np.savez_compressed(OUT/'test.npz',**z);M,names=new_matrix(z,c['metadata']['historical_users'])
    np.savez_compressed(OUT/'test_model_features.npz',X=M,users=z['users'],items=z['items'],columns=np.asarray(names),original=z['original'])
    if names!=read(OUT/'baseline_reproduction.json')['feature_names']:raise ValueError('Native feature order mismatch')
    dump('test_build.json',{'phase':'test','generator':'ppr','limit':50,'rows':len(us),'features':77,'users':1260,'original_rows':int(sum(oldmask)),'additional_pairs':int(len(us)-sum(oldmask)),
        'all_original_candidates_preserved':True,'all_features_recomputed':True,'new_candidates_per_user':newcounts,'context':c['metadata'],'source_cache_path':str(V8/'test.npz'),'source_cache_sha256':sha(V8/'test.npz'),
        'data_sha256':sha(OUT/'test.npz'),'model_feature_sha256':sha(OUT/'test_model_features.npz'),'repeat_generator_checks':checks,
        'original_feature_changes_counts':dict(changed),'original_feature_max_difference':dict(maxdiff),
        'consistency_policy':'All rows use V11 expanded validation feature definitions. Original V5 final first50 channel/hour-affinity signals were full20wk; recomputing aligns them to trailing10wk used in training. Generator/schedule differences also explicitly recorded. Old artifacts are not changed.',
        'target_behavior_read':False,'target_schedule_sha256':sha(SOURCE/'export_arh_31-42-final.csv'),'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'rows':len(us),'additional_pairs':int(len(us)-sum(oldmask)),'changed_original_feature_columns':dict(changed),'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
