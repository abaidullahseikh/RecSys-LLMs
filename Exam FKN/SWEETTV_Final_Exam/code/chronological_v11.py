"""Frozen-strategy reliability check on an earlier chronological window."""
import time
from collections import defaultdict
import numpy as np
import validation
from v11_common import *
from train_v11 import mining_scores

def main():
    t=time.monotonic()
    if (OUT/'chronological_validation.json').exists():raise FileExistsError('Preserve chronological evidence')
    frozen=sha(OUT/'selection.json');z,truth=model_data('early');users=tuple(dict.fromkeys(z['users']));dev=tuple(u for u in users if partition(u)!=0);ass=tuple(u for u in users if partition(u)==0)
    params=read(OUT/'baseline_reproduction.json')['parameters'];fold=np.asarray([inner_fold(str(u)) for u in z['users']]);dm=np.isin(z['users'],dev);scores={n:np.zeros(len(z['users'])) for n in ('E1','E2','E3','E4')};models=[]
    for f in range(3):
        fitusers=tuple(u for u in dev if inner_fold(u)!=f);vu=tuple(u for u in dev if inner_fold(u)==f);ti=np.flatnonzero(np.isin(z['users'],fitusers));vi=np.flatnonzero(dm&(fold==f));oi=ti[z['original'][ti]]
        path=OUT/f'early_E1_inner{f}.txt';bst=fit(z,oi,path,params);scores['E2'][vi]=bst.predict(z['X'][vi],num_threads=4);scores['E1'][vi]=scores['E2'][vi]
        miner,meta=mining_scores(z,fitusers,f'early_outer{f}',params)
        for name in ('E3','E4'):
            ii=oi if name=='E3' else ti;selected=sample_negatives(z,ii,miner);path=OUT/f'early_{name}_inner{f}.txt';bst=fit(z,selected,path,params);scores[name][vi]=bst.predict(z['X'][vi],num_threads=4)
            np.save(OUT/f'early_{name}_inner{f}_training_indices.npy',selected,allow_pickle=False)
        models.append({'fold':f,'fit_users':list(fitusers),'evaluation_users':list(vu),'mining':meta})
        print('early chrono fold',f,flush=True)
    # Historical seen classification uses only the early feature-history interval.
    bounds=read(OUT/'early_build.json')['context']['history'];prof=defaultdict(set)
    for r in records(validation.STEP2/'train_airing_overlaps.csv.gz'):
        if bounds[0]<=int(r['first_overlap_utc'])<int(r['last_overlap_utc'])<=bounds[1] and int(r['overlap_seconds'])>=120:prof[r['user_id']].add(r['tv_show_id'])
    comparison={}
    for n,s in scores.items():
        rr=ranks(s,z,dev,z['original'] if n in ('E1','E3') else None);comparison[n]=measured(rr,truth,dev,prof);save_pred(f'early_{n}_development.csv',rr)
        np.save(OUT/f'early_{n}_scores.npy',s,allow_pickle=False)
    main,maintruth=model_data();mainprof=profiles('train');base=np.load(OUT/'expanded_V10_crossfit_scores.npy',allow_pickle=False)
    matched={}
    for n in scores:
        s=base if n in ('E1','E2') else np.load(OUT/f'{n}_crossfit_scores.npy',allow_pickle=False)
        rr=ranks(s,main,dev,main['original'] if n in ('E1','E3') else None);matched[n]=measured(rr,maintruth,dev,mainprof)
    # User-population and label sensitivity are diagnostics only; no changed labels or tuning.
    baseline=read(OUT/'baseline_reproduction.json');selection=read(OUT/'selection.json');alllabels=read(V8/'labels.json');strata={}
    for v,path in [('V8',V8/'E2_streak_all_crossfit.csv'),('V9',V9/'selected_all_crossfit.csv'),('V10',V10/'selected_all_crossfit.csv'),('V11',OUT/f"{selection['selected']}_all_crossfit.csv")]:
        pred={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(path)};template=set(users);main_dev=baseline['development_users']
        subsets={'all_development':main_dev,'template_development':[u for u in main_dev if u in template],'nontemplate_development':[u for u in main_dev if u not in template]}
        strata[v]={g:{'users':len(uu),'MAP5':map5(pred,alllabels['secondary'],uu),'alternative80_MAP5':map5(pred,alllabels['fraction80'],uu)} for g,uu in subsets.items()}
    chosen=selection['selected'];pred_new={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(OUT/f'{chosen}_all_crossfit.csv')};pred_old={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(V10/'selected_all_crossfit.csv')}
    from v8_common import ap
    diffs=np.asarray([ap(maintruth[u],pred_new[u])-ap(maintruth[u],pred_old[u]) for u in baseline['development_users']]);rng=np.random.default_rng(2026);boot=[]
    for _ in range(20):boot.extend(diffs[rng.integers(0,len(diffs),size=(100,len(diffs)))].mean(axis=1).tolist())
    if sha(OUT/'selection.json')!=frozen:raise ValueError('Selection changed during chronological diagnosis')
    dump('chronological_validation.json',{'selection_frozen_before_this_check':True,'selection_sha256':frozen,'early':{'history':bounds,'target':[bounds[1],1589749200],
        'users':len(dev),'assessment_users_not_used_for_tuning':len(ass),'comparison':comparison,'models':models},
        'main_matched_template_development':{'history':[1583704800,1589749200],'target':[1589749200,1595797200],'users':len(dev),'comparison':matched},
        'population_and_alternative_label_sensitivity':strata,
        'paired_development_bootstrap':{'resamples':2000,'seed':2026,'mean_delta':float(diffs.mean()),'interval95':np.quantile(boot,[.025,.975]).tolist(),'note':'Descriptive user bootstrap on reused development labels; not a Kaggle confidence interval or an independent selection test'},
        'limitations':['Only20 weeks of logged history; earlier fold has5 observed history weeks and5 target weeks, main10/10, final20/12. Original feature-rate denominators retained for matching schema; these windows are not fully comparable.',
            'Earlier cache/cohort was examined during V6. Main development and assessment reused in prior experiments; neither is pristine.',
            'Template users are a selected, more predictable population than broader training cohort. Population mix and time/horizon shift can change model ordering.',
            'Official hidden relevance construction remains unconfirmed; local>=120s frequency Top5 may differ. Alternative80 labels are diagnostic hypotheses only.',
            'One additional historical cutoff is available without inventing data; all fitting and mining at that cutoff exclude evaluation-user labels. No future-trained V10 model is applied to the early fold.'],
        'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'early':comparison,'main_matched':matched,'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'seconds':time.monotonic()-t},flush=True)

if __name__=='__main__':main()
