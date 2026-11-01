"""Development-only unseen rank/distribution audit before expanded-pool fitting."""
from v12_common import *

def moments(a):
    a=np.asarray(a,dtype=float);good=a[np.isfinite(a)]
    if not len(good):return {'rows':len(a),'missing_rate':1. if len(a) else None}
    return {'rows':len(a),'mean':float(good.mean()),'median':float(np.median(good)),'p10':float(np.quantile(good,.1)),'p90':float(np.quantile(good,.9)),
        'min':float(good.min()),'max':float(good.max()),'zero_rate':float(np.mean(a==0)),'missing_rate':float(np.mean(~np.isfinite(a)))}

def main():
    t=time.monotonic()
    if (OUT/'unseen_diagnosis.json').exists():raise FileExistsError('Preserve diagnostics')
    z,truth=data();base=read(OUT/'baseline_reproduction.json');dev=base['development_users'];dm=np.isin(z['users'],dev);names=z['columns'].tolist();scores=np.load(OUT/'V10_expanded_scores.npy',allow_pickle=False)
    prof=history_profiles();seen=np.asarray([str(i) in prof[str(u)] for u,i in zip(z['users'],z['items'])]);relevant=z['binary_label'].astype(bool)
    ranking=ranks(scores,z,dev);position=np.zeros(len(scores),dtype=np.int32);ix=defaultdict(dict)
    for j,(u,i) in enumerate(zip(z['users'],z['items'])):
        if dm[j]:ix[str(u)][str(i)]=j
    for u,r in ranking.items():
        for k,i in enumerate(r,1):position[ix[u][i]]=k
    masks={'original_all':dm&z['original'],'original_relevant':dm&z['original']&relevant,'original_unseen':dm&z['original']&~seen,
        'original_unseen_relevant':dm&z['original']&~seen&relevant,'original_seen_relevant':dm&z['original']&seen&relevant,
        'added_all':dm&~z['original'],'added_relevant':dm&~z['original']&relevant,'added_irrelevant':dm&~z['original']&~relevant}
    groups={};fields=('ppr_score','ppr_rank','ppr_support_adjusted_score','association_rules_score','association_rules_rank','cbf_similarity','cbf_rank','pop_distinct_users','popularity_rank',
        'personal_frequency_score','user_program_airings','program_active_weeks','completion80_share','completion80_active_week_streak','candidate_source_count','target_airing_count','target_availability_week_fraction','candidate_channel_affinity','candidate_hour_affinity')
    for g,mask in masks.items():
        n=int(mask.sum());pos=position[mask];groups[g]={'pairs':n,'relevant_Top5_labels':int((mask&relevant).sum()),'seen_fraction':float(seen[mask].mean()) if n else None,
            'unseen_fraction':float((~seen[mask]).mean()) if n else None,'score':moments(scores[mask]),'rank':moments(pos),'rank_bins':{},
            'features':{f:moments(z['X'][mask,names.index(f)]) for f in fields},'missing_feature_rate':float(np.mean(~np.isfinite(z['X'][mask]))) if n else None,
            'missing_personal_history_flag_rate':float(z['X'][mask,names.index('missing_personal_history')].mean()) if n else None,
            'missing_target_availability_flag_rate':float(z['X'][mask,names.index('missing_target_availability')].mean()) if n else None}
        for k,m in [('1to5',pos<=5),('6to10',(pos>=6)&(pos<=10)),('11to20',(pos>=11)&(pos<=20)),('below20',pos>20)]:
            groups[g]['rank_bins'][k]={'pairs':int(m.sum()),'fraction':float(m.mean()) if n else None}
    rows=[];old=dm&z['original'];new=dm&~z['original']
    for j,f in enumerate(names):
        a=moments(z['X'][old,j]);b=moments(z['X'][new,j]);lo=z['X'][old,j].min();hi=z['X'][old,j].max()
        rows.append({'feature':f,**{'original_'+k:v for k,v in a.items()},**{'added_'+k:v for k,v in b.items()},'added_outside_original_range_fraction':float(np.mean((z['X'][new,j]<lo)|(z['X'][new,j]>hi)))})
    table('candidate_feature_distributions.csv',rows)
    gain=np.zeros(len(names))
    for f in range(3):gain+=lgb.Booster(model_file=str(V10/f'pairwise_inner{f}.txt')).feature_importance('gain')
    importance=sorted([{'feature':f,'gain':float(v),'gain_share':float(v/gain.sum())} for f,v in zip(names,gain)],key=lambda r:-r['gain']);table('V10_feature_importance.csv',importance)
    source=names.index('candidate_source_count');structural=bool(np.all(z['X'][new,source]==0) and np.all(z['X'][old,source]>=1));counts=load(OUT/'validation_targets.npz')['counts']
    poolstats={}
    for g,mask in [('original',old),('expanded',dm),('added',new)]:
        poolstats[g]={'rows':int(mask.sum()),'local_binary_Top5_positive_count':int(relevant[mask].sum()),'local_binary_positive_rate':float(relevant[mask].mean()),
            'any_qualifying_count_positive':int((counts[mask]>0).sum()),'any_qualifying_count_positive_rate':float((counts[mask]>0).mean()),'grade_counts':dict(Counter(str(int(x)) for x in grades(counts[mask])))}
    dump('unseen_diagnosis.json',{'population':'development_only','users':len(dev),'groups':groups,'training_distributions':poolstats,'importance':importance,
        'structural_support_mismatch_confirmed':structural,
        'interpretation':[
            'Added candidates are positive-PPR historical-unseen items; their absence of personal-repeat features is structural, not a failed feature join. Missingness and zero rates are reported separately.',
            'Candidate-source-count support mismatch exists if original values>=1 and all extras0; original V10 never fitted the expanded distribution. Personalized score and rank distributions must also be considered.',
            'Low scores correlate with absent repeat evidence and weaker/deeper retrieval; feature gain and distributions do not establish causal popularity bias.',
            'V12 addresses measured distribution mismatch by fitting all expanded examples. No unseen bonus, artificial feature fill, extra algorithm, or new hyperparameter sweep is introduced.',
            'Binary supervision remains historical local Top5 membership. Graded supervision uses all qualifying target occasions, including non-Top5 programs; this is an explicit target change, not official competition ground truth.'
        ],'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'added':groups['added_all'],'added_relevant':groups['added_relevant'],'original_unseen_relevant':groups['original_unseen_relevant'],
        'structural_support_mismatch':structural,'training_distributions':poolstats,'top_features':importance[:8],'seconds':time.monotonic()-t},flush=True)

if __name__=='__main__':main()
