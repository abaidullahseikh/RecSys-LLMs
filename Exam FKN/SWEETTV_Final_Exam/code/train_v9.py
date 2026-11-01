"""One controlled new binary model and at most two development-selected RRF weights."""
from collections import defaultdict
import csv,time,resource,math
import numpy as np
import lightgbm as lgb
import validation
from v8_common import ROOT,SOURCE,read,load,sha,records,inner_fold,map5

OUT=ROOT/'results/v9_experiment';V8=ROOT/'results/v8_experiment'
FEATURE='ppr_support_adjusted_score'
PRIOR_VIEWERS=20.


def dump(name,data):
    p=OUT/name
    if p.exists():raise FileExistsError('Preserve V9 result')
    p.write_text(__import__('json').dumps(data,indent=2)+'\n')


def new_matrix(z,historical_users):
    if historical_users<=0:raise ValueError('Historical user denominator must be positive')
    names=z['columns'].tolist();p=z['X'][:,names.index('ppr_score')].astype(float)
    support=z['X'][:,names.index('pop_distinct_users')].astype(float)
    if np.any(p<0) or np.any(support<0):raise ValueError('Invalid historical signal')
    # Fixed shrinkage/square-root transform, not a parameter search.
    signal=p*np.sqrt(historical_users/(support+PRIOR_VIEWERS))
    cols=read(V8/'experiment.json')['selected_columns']
    X=np.column_stack([z['X'][:,cols],signal.astype(np.float32)])
    return X,[names[i] for i in cols]+[FEATURE]


def rankings(score,z,indices,cohort):
    names=z['columns'].tolist();pi=names.index('popularity_rank');ci=names.index('user_cold_start');g=defaultdict(list)
    for n,s in zip(indices,score):g[str(z['users'][n])].append((int(n),float(s)))
    result={}
    for u in cohort:
        group=g[u]
        if not group:raise ValueError('Missing user candidate group')
        cold=z['X'][group[0][0],ci]>.5
        group.sort(key=lambda ns:((0 if cold else -ns[1]),float(z['X'][ns[0],pi]),int(z['items'][ns[0]])))
        result[u]=tuple(str(z['items'][n]) for n,s in group)
    return result


def fuse(new,base,weight,cohort):
    out={}
    for u in cohort:
        a={i:k for k,i in enumerate(new[u])};b={i:k for k,i in enumerate(base[u])}
        if set(a)!=set(b):raise ValueError('Fusion must use identical candidates')
        scores={i:weight/(60+a[i])+(1-weight)/(60+b[i]) for i in a}
        out[u]=tuple(sorted(a,key=lambda i:(-scores[i],a[i],int(i)))[:5])
    return out


def top(ranking):return {u:tuple(p[:5]) for u,p in ranking.items()}


def save(name,pred):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(p[:5])) for u,p in pred.items())


def main():
    t=time.monotonic()
    if (OUT/'experiment.json').exists():raise FileExistsError('Preserve completed V9 model run')
    if not (OUT/'analysis_complete.json').exists():raise ValueError('Ranking/drift audit must complete before model changes')
    ex=read(V8/'experiment.json');pop=read(V8/'population.json');dm=read(V8/'data_manifest.json');labels=read(V8/'labels.json')
    z=load(V8/'validation.npz');X,names=new_matrix(z,pop['historical_users']);us=z['users'];items=z['items']
    dev=tuple(ex['development_users']);assessment=tuple(ex['assessment_users']);allusers=tuple(pop['users'])
    mask=np.isin(us,dev);fold=np.asarray([inner_fold(str(u)) for u in us]);y=z['binary_label']
    params=ex['selected_params'];rounds=ex['rounds']
    plan={'targeted_improvement':'PPR mass calibrated by smoothed historical audience support, as an additional feature; original76 V8 features unchanged',
        'formula':'ppr_score * sqrt(number_of_users_with_qualifying_training_history / (distinct_historical_viewers(program)+20))',
        'prior_viewers':PRIOR_VIEWERS,'training_denominator_users':pop['historical_users'],
        'justification':'Development errors leave42.87% of reachable labels below5. PPR/AR candidate rank strongly associated with global popularity; false Top5 programs have more support than relevant6-10 items. Test personalization relative to audience support, not blanket diversity.',
        'caveat':'Descriptive associations are not causal proof of popularity bias; added feature may worsen accuracy',
        'feature_names':names,'parameters':params,'rounds':rounds,'thread_limit':4,'seed':2026,
        'candidate_rows_and_labels':'identical V8 cache and >=120s Top5 labels',
        'analysis_sha256':sha(OUT/'analysis_complete.json'),'selection_rule':'Highest development OOF MAP among new E2 pure and justified E4 fusion; ties prefer pure E2. Baseline E1 cannot be relabeled V9.',
        'fusion_plan':'Only if more than10% development Top5 lists change, test RRF with new-model weights0.50 and0.75; constant60',
        'E3_skipped':'No evidence from this audit that training positives are wrongly defined; avoid changing labels or weighting based on hidden Kaggle outcomes',
        'Kaggle_scores_used_as_labels':False,'assessment_caveat':'All-user error audit includes the reused V8 assessment; no claim of pristine assessment. No assessment scores used to select fusion.'}
    dump('plan.json',plan)
    baselines=np.load(OUT/'V8_crossfit_scores.npy',allow_pickle=False);oof=np.zeros(len(us));meta=[]
    for f in range(3):
        ti=np.flatnonzero(mask&(fold!=f));vi=np.flatnonzero(mask&(fold==f));vu=tuple(u for u in dev if inner_fold(u)==f)
        if set(us[ti])&set(us[vi]) or set(us[ti])&set(assessment):raise ValueError('Group contamination')
        bst=lgb.train(params,lgb.Dataset(X[ti],label=y[ti],feature_name=names),num_boost_round=rounds)
        path=OUT/f'E2_inner{f}.txt';bst.save_model(str(path));oof[vi]=bst.predict(X[vi],num_threads=4)
        meta.append({'fold':f,'training_rows':len(ti),'valid_rows':len(vi),'training_users':len(set(us[ti])),
            'valid_users':len(vu),'model_sha256':sha(path)})
    di=np.flatnonzero(mask);bi=rankings(baselines[mask],z,di,dev);new=rankings(oof[mask],z,di,dev)
    baseline=map5(top(bi),labels['secondary'],dev);new_score=map5(top(new),labels['secondary'],dev)
    if abs(baseline-ex['comparisons'][ex['selected']]['development']['secondary_MAP'])>1e-14:raise ValueError('Baseline mismatch')
    save('E2_development.csv',new);changed=sum(new[u][:5]!=bi[u][:5] for u in dev)/len(dev)
    variants=[{'id':'E2_pure','new_weight':1.,'development_MAP':new_score}]
    if changed>.10:
        for w in (.50,.75):
            pred=fuse(new,bi,w,dev);variants.append({'id':f'E4_rrf_{w:.2f}','new_weight':w,'development_MAP':map5(pred,labels['secondary'],dev)})
            save(f'E4_development_{w:.2f}.csv',pred)
    selected=max(variants,key=lambda r:(r['development_MAP'],r['new_weight']==1.))
    dump('selection.json',{'selected':selected,'variants':variants,'V8_development_MAP':baseline,
        'changed_development_Top5_fraction':changed,'selected_before_assessment_scoring':True,'rule':plan['selection_rule']})
    bt=time.monotonic();bst=lgb.train(params,lgb.Dataset(X[mask],label=y[mask],feature_name=names),num_boost_round=rounds)
    bst.save_model(str(OUT/'assessment_model.txt'));ai=np.flatnonzero(~mask);score=bst.predict(X[~mask],num_threads=4)
    newa=rankings(score,z,ai,assessment);basea=rankings(baselines[~mask],z,ai,assessment)
    chosen=fuse(newa,basea,selected['new_weight'],assessment)
    save('E2_assessment.csv',newa);save('selected_assessment.csv',chosen)
    comparison={
        'E1_V8':{'development_MAP':baseline,'assessment_MAP':map5(top(basea),labels['secondary'],assessment)},
        'E2_pure':{'development_MAP':new_score,'assessment_MAP':map5(top(newa),labels['secondary'],assessment)},
        'selected':{'development_MAP':selected['development_MAP'],'assessment_MAP':map5(chosen,labels['secondary'],assessment)}}
    for variant in variants:
        if variant['new_weight']==1.:continue
        pa=fuse(newa,basea,variant['new_weight'],assessment);save(f"{variant['id']}_assessment.csv",pa)
        comparison[variant['id']]={'development_MAP':variant['development_MAP'],'assessment_MAP':map5(pa,labels['secondary'],assessment)}
    dp=fuse(new,bi,selected['new_weight'],dev);merged={u:(dp[u] if u in dp else chosen[u]) for u in allusers}
    save('selected_all_crossfit.csv',merged)
    final=lgb.train(params,lgb.Dataset(X,label=y,feature_name=names),num_boost_round=rounds);final.save_model(str(OUT/'final_model.txt'))
    importance=sorted([{'feature':n,'gain':float(g),'splits':int(s)} for n,g,s in zip(names,bst.feature_importance('gain'),bst.feature_importance('split'))],key=lambda r:-r['gain'])
    result={'status':'trained_and_evaluated','selected':selected,'comparison':comparison,'variants':variants,'feature_names':names,
        'base_V8_indices':ex['selected_columns'],'original_V8_selected':ex['selected'],'new_feature':FEATURE,
        'parameters':params,'rounds':rounds,'development_users':list(dev),'assessment_users':list(assessment),
        'inner_models':meta,'assessment_model_sha256':sha(OUT/'assessment_model.txt'),'final_model_sha256':sha(OUT/'final_model.txt'),
        'feature_source_sha256':sha(V8/'validation.npz'),'feature_importance':importance,
        'all_crossfit_MAP':map5(merged,labels['secondary'],allusers),
        'assessment_fit_predict_seconds':time.monotonic()-bt,'seconds':time.monotonic()-t,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'source_input':'All V8 historical rows, groups, candidate pools and labels unchanged; only one additional PPR-calibration signal',
        'E3_alternative_labels':'skipped_not_justified','Kaggle_score':None,'mandatory_CSV_regardless_of_delta':True,
        'recommended_final':'submission/final_submission_v8_experimental.csv','private_MAP_best_user_reported':.39908}
    dump('experiment.json',result)
    print(__import__('json').dumps({'selected':selected,'changed_lists':changed,'comparison':comparison,'seconds':result['seconds'],'peak_rss_mib':result['peak_rss_mib']},indent=2))


if __name__=='__main__':main()
