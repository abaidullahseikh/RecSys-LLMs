"""Five locked configurations: comparable binary baseline, two-feature ablation, regression."""
import csv,time,resource,json
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import validation
from v8_common import ROOT,OUT,read,dump,load,sha,partition,inner_fold,rank,map5,save_pred

EXPERIMENTS=('E1_binary75','E2_streak','E2_recent_share','E2_both','E3_logcount75')


def columns_for(name):
    if name in ('E1_binary75','E3_logcount75'):return list(range(75))
    if name=='E2_streak':return list(range(75))+[75]
    if name=='E2_recent_share':return list(range(75))+[76]
    if name=='E2_both':return list(range(77))
    raise ValueError(name)


def population_metrics(pred,labels,users,pop):
    template=set(pop['template_users']);sizes=pop['history_sizes']
    def group(condition):return [u for u in users if condition(u)]
    groups={'all':list(users),'template':group(lambda u:u in template),'nontemplate':group(lambda u:u not in template),
        'history_and_outcome':group(lambda u:sizes[u]>0 and bool(labels['primary'][u])),
        'heavy101plus':group(lambda u:sizes[u]>=101),'light1to20':group(lambda u:1<=sizes[u]<=20),
        'weak1to5':group(lambda u:1<=sizes[u]<=5),'cold0':group(lambda u:sizes[u]==0),
        'empty_target':group(lambda u:not labels['primary'][u])}
    return {g:{'users':len(us),'secondary_MAP':map5(pred,labels['secondary'],us),'fraction80_MAP':map5(pred,labels['fraction80'],us),
        'primary_MAP':map5(pred,labels['primary'],us)} for g,us in groups.items()}


def baseparams(name):
    return {'objective':'regression' if name=='E3_logcount75' else 'binary','metric':'None','num_leaves':15,
        'min_data_in_leaf':100,'learning_rate':.05,'lambda_l2':5.,'feature_fraction':1.,'bagging_fraction':1.,
        'seed':2026,'num_threads':4,'deterministic':True,'force_col_wise':True,'verbosity':-1}


def main():
    start=time.monotonic()
    if (OUT/'experiment.json').exists():raise FileExistsError('Preserve V8 experiment')
    z=load(OUT/'validation.npz');X=z['X'];us=z['users'];items=z['items'];names=z['columns'].tolist()
    pop=read(OUT/'population.json');labels=read(OUT/'labels.json');allusers=tuple(pop['users'])
    assessment=tuple(u for u in allusers if partition(u)==0);dev=tuple(u for u in allusers if partition(u)!=0)
    dm=np.isin(us,dev);am=~dm;fold=np.asarray([inner_fold(str(u)) for u in us],dtype=np.int8)
    plan={'experiments':{n:{'params':baseparams(n),'columns':[names[i] for i in columns_for(n)]} for n in EXPERIMENTS},
        'boosting_rounds':78,'no_hyperparameter_search':True,'seed':2026,'threads':4,
        'development_users':list(dev),'assessment_users':list(assessment),
        'grouping':'SHA256(user_id) first32bit mod5==0 assessment; development inner SHA256(v8-inner:+user) mod3',
        'label':'Top5 by >=120s qualifying-airing frequency; numeric ID frequency tie-break is our convention',
        'regression_target':'log1p(actual count of >=120s qualifying held-out airings), including zero counts; reduces domination by extreme broadcast counts while preserving monotone ordering',
        'selection':'Maximize development OOF secondary Top5-120 MAP among E2/E3 valid new configurations; tie by primary MAP then fewer features. E1 cannot be relabeled V8. Outer assessment revealed only after selection.',
        'generation':'Mandatory new selected configuration CSV regardless of development-vs-E1 gain, if correctness checks pass',
        'broader_metrics':'All-user cross-fitted predictions: development OOF (2/3 development fit) plus assessment (all development fit). All users evaluated without their own target outcomes in their model.',
        'reuse_caveat':'Template outer256 previously examined V4-V7; broader non-template outcomes not used for previous model selection. No pristine claim for the template subset.'}
    dump('plan.json',plan)
    cv={};oofs={};targets={n:(np.log1p(z['future_counts'].astype(np.float64)) if n=='E3_logcount75' else z['binary_label']) for n in EXPERIMENTS}
    for name in EXPERIMENTS:
        t=time.monotonic();cols=columns_for(name);target=targets[name];pred={};folds=[]
        for f in range(3):
            ti=np.flatnonzero(dm&(fold!=f));vi=np.flatnonzero(dm&(fold==f));vu=tuple(u for u in dev if inner_fold(u)==f)
            if set(us[ti])&set(us[vi]) or set(us[ti])&set(assessment):raise ValueError('Supervised user contamination')
            bst=lgb.train(baseparams(name),lgb.Dataset(X[ti][:,cols],label=target[ti],feature_name=[names[i] for i in cols]),num_boost_round=78)
            bst.save_model(str(OUT/f'{name}_inner{f}.txt'));s=bst.predict(X[vi][:,cols],num_threads=4)
            pred.update(rank(s,z,vi,vu));folds.append({'fold':f,'training_users':len(set(us[ti])),'validation_users':len(vu),
                'training_rows':len(ti),'validation_rows':len(vi),'model_sha256':sha(OUT/f'{name}_inner{f}.txt')})
        pred={u:pred[u] for u in dev};save_pred(f'{name}_development.csv',pred);oofs[name]=pred
        cv[name]={'secondary_MAP':map5(pred,labels['secondary'],dev),'fraction80_MAP':map5(pred,labels['fraction80'],dev),
            'primary_MAP':map5(pred,labels['primary'],dev),'folds':folds,'feature_count':len(cols),
            'seconds':time.monotonic()-t,'peak_rss_mib_so_far':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        print(name,'development',cv[name]['secondary_MAP'],'seconds',cv[name]['seconds'],flush=True)
    new=EXPERIMENTS[1:];selected=max(new,key=lambda n:(cv[n]['secondary_MAP'],cv[n]['primary_MAP'],-cv[n]['feature_count']))
    dump('selection.json',{'selected':selected,'development':cv,'frozen_before_assessment':True,'rule':plan['selection']})
    comparisons={};importance={};allpred={}
    for name in EXPERIMENTS:
        t=time.monotonic();cols=columns_for(name)
        bst=lgb.train(baseparams(name),lgb.Dataset(X[dm][:,cols],label=targets[name][dm],feature_name=[names[i] for i in cols]),num_boost_round=78)
        bst.save_model(str(OUT/f'{name}_assessment_model.txt'))
        it=time.monotonic();ai=np.flatnonzero(am);s=bst.predict(X[am][:,cols],num_threads=4);pred=rank(s,z,ai,assessment)
        seconds_inference=time.monotonic()-it
        save_pred(f'{name}_assessment.csv',pred)
        merged={u:(oofs[name][u] if u in oofs[name] else pred[u]) for u in allusers};save_pred(f'{name}_all_crossfit.csv',merged)
        allpred[name]=merged
        comparisons[name]={'development':cv[name],
            'assessment':population_metrics(pred,labels,assessment,pop),
            'all_users_crossfit':population_metrics(merged,labels,allusers,pop),
            'assessment_fit_inference_seconds':time.monotonic()-t,'inference_seconds':seconds_inference,
            'model_sha256':sha(OUT/f'{name}_assessment_model.txt'),'candidate_recall':{},'feature_count':len(cols)}
        importance[name]=sorted([{'feature':names[i],'gain':float(g),'splits':int(k)} for i,g,k in zip(cols,bst.feature_importance('gain'),bst.feature_importance('split'))],key=lambda r:-r['gain'])
    # Candidate recall denominators count relevant labels, not only positives captured by generators.
    candidates=defaultdict(set)
    for u,i in zip(us,items):candidates[str(u)].add(str(i))
    recall={}
    for group,population in [('all',allusers),('development',dev),('assessment',assessment),('template',pop['template_users'])]:
        denominator=sum(len(labels['secondary'][u]) for u in population)
        recall[group]={'users':len(population),'average_candidates':float(np.mean([len(candidates[u]) for u in population])),
            'micro_recall':sum(len(candidates[u]&set(labels['secondary'][u])) for u in population)/denominator,
            'users_under5':sum(len(candidates[u])<5 for u in population)}
    cols=columns_for(selected);bst=lgb.train(baseparams(selected),lgb.Dataset(X[:,cols],label=targets[selected],feature_name=[names[i] for i in cols]),num_boost_round=78)
    bst.save_model(str(OUT/'final_model.txt'))
    result={'status':'trained_and_evaluated','selected':selected,'comparisons':comparisons,'candidate_recall':recall,
        'feature_importance':importance,'feature_names':names,'selected_columns':cols,'selected_params':baseparams(selected),'rounds':78,
        'development_users':list(dev),'assessment_users':list(assessment),'final_fit_users':len(allusers),'final_fit_rows':len(us),
        'final_model_sha256':sha(OUT/'final_model.txt'),'feature_sha256':sha(OUT/'validation.npz'),
        'selected_development_delta_vs_E1':cv[selected]['secondary_MAP']-cv['E1_binary75']['secondary_MAP'],
        'seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'versions':{'lightgbm':lgb.__version__,'numpy':np.__version__},'kaggle_score':None,'submission_generated':False,
        'final_recommendation':'Keep genuinely scored V5 until a new submission receives a better genuine private score'}
    dump('experiment.json',result)
    print(json.dumps({'selected':selected,'development_delta':result['selected_development_delta_vs_E1'],
        'assessment':{n:comparisons[n]['assessment']['all'] for n in EXPERIMENTS},
        'all_crossfit':{n:comparisons[n]['all_users_crossfit']['all'] for n in EXPERIMENTS},
        'seconds':result['seconds'],'peak_rss_mib':result['peak_rss_mib']},indent=2))


if __name__=='__main__':main()
