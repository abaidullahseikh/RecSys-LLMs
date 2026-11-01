"""Locked-budget candidate/temporal experiments; OOF selection precedes outer scoring."""
from collections import defaultdict
import csv
import json
import time
import resource
import numpy as np
import lightgbm as lgb
import evaluation
import validation
from features_v5 import ROOT,sha,records
from train_v5 import user_hash,parameters
from v6_data import OUT,corrected_ap,dump

RECIPES=[{'id':'fixes50','k':50,'early_weight':0.},
         {'id':'expanded100','k':100,'early_weight':0.},
         {'id':'temporal50','k':50,'early_weight':.5},
         {'id':'temporal100','k':100,'early_weight':.5}]
WEIGHTS=(.5,.75,1.)


def load(name):
    with np.load(OUT/f'{name}_data.npz') as z:return {k:z[k] for k in z.files}


def matrix_rows(z,allowed,k):
    return np.flatnonzero(np.isin(z['users'],list(allowed))&z[f'mask{k}'])


def dataset(z,target,ix):
    return z['X'][ix],np.asarray([str(z['items'][n]) in target[str(z['users'][n])] for n in ix],dtype=np.int8)


def rankings(score,z,ix,cohort):
    names=z['columns'].tolist();cold=names.index('user_cold_start');pop=names.index('popularity_rank')
    grouped=defaultdict(list)
    for n,s in zip(ix,score):grouped[str(z['users'][n])].append((int(n),float(s)))
    out={}
    for u in cohort:
        group=grouped[u]
        if z['X'][group[0][0],cold]>.5:order=sorted(group,key=lambda ns:(z['X'][ns[0],pop],int(z['items'][ns[0]])))
        else:order=sorted(group,key=lambda ns:(-ns[1],z['X'][ns[0],pop],int(z['items'][ns[0]])))
        out[u]=tuple(str(z['items'][n]) for n,s in order)
    return out


def fuse(new,base,weight,cohort):
    out={}
    for u in cohort:
        a={i:k for k,i in enumerate(new[u])};b={i:k for k,i in enumerate(base[u])}
        # Frozen V5's original union is retained; expansion-only items have no baseline vote.
        score={i:weight/(60+a[i])+(1-weight)/(60+b[i]) if i in b else weight/(60+a[i]) for i in a}
        out[u]=tuple(sorted(a,key=lambda i:(-score[i],a[i],int(i)))[:5])
    return out


def scores(pred,target,cohort):
    return {name:sum(corrected_ap(set(truth[u]),pred[u]) for u in cohort)/len(cohort) for name,truth in target.items()}


def write_predictions(name,pred):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(p[:5])) for u,p in pred.items())


def fit(recipe,main,early,truth,etruth,allowed,params,rounds):
    ix=matrix_rows(main,allowed,recipe['k']);X,y=dataset(main,truth,ix);weights=np.ones(len(y))
    if recipe['early_weight']:
        ei=matrix_rows(early,allowed,recipe['k']);EX,ey=dataset(early,etruth,ei)
        X=np.vstack([X,EX]);y=np.concatenate([y,ey]);weights=np.concatenate([weights,np.full(len(ey),recipe['early_weight'])])
    model=lgb.train(params,lgb.Dataset(X,label=y,weight=weights,feature_name=main['columns'].tolist()),num_boost_round=rounds)
    return model,{'rows':len(y),'positive_rows':int(y.sum()),'users':len(allowed),'early_weight':recipe['early_weight']}


def main():
    started=time.monotonic()
    if (OUT/'experiment.json').exists():raise FileExistsError('Preserve V6 run')
    main=load('main');early=load('early');targets=json.loads((OUT/'main_labels.json').read_text());etargets=json.loads((OUT/'early_labels.json').read_text())
    ex=json.loads((ROOT/'results/v5_experiment/experiment.json').read_text());dev=ex['development_users'];outer=ex['outer_users']
    names=main['columns'].tolist();params=ex['selection']['parameters'];rounds=ex['selection']['rounds']
    v5model=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/evaluation_model.txt'))
    outer50=matrix_rows(main,outer,50)
    v5rank=rankings(v5model.predict(main['X'][outer50],num_threads=4),main,outer50,outer)
    # Fixes do not change validation values; assert exact reference replay before experimenting.
    saved=evaluation.parse_prediction_records(list(records(ROOT/'results/v5_experiment/heldout_v5.csv')))
    if any(tuple(v5rank[u][:5])!=saved[u] for u in outer):raise ValueError('V5 fix-only validation changed unexpectedly')
    base_oof={};oof_by_recipe={r['id']:{} for r in RECIPES};candidate_metrics=[]
    for k in (50,100):
        groups=defaultdict(set)
        for n in np.flatnonzero(main[f'mask{k}']):groups[str(main['users'][n])].add(str(main['items'][n]))
        for population,us in [('development',dev),('outer',outer)]:
            denom=sum(len(targets['secondary'][u]) for u in us)
            candidate_metrics.append({'k':k,'population':population,'users':len(us),
                'avg_candidates':float(np.mean([len(groups[u]) for u in us])),
                'recall':sum(len(groups[u]&set(targets['secondary'][u])) for u in us)/denom,
                'insufficient_users':sum(len(groups[u])<5 for u in us)})
    for f in range(3):
        train=[u for u in dev if user_hash(u,'v6-inner:')%3!=f];valid=[u for u in dev if user_hash(u,'v6-inner:')%3==f]
        for recipe in RECIPES:
            model,meta=fit(recipe,main,early,targets['secondary'],etargets['secondary'],train,params,rounds)
            vi=matrix_rows(main,valid,recipe['k']);pred=rankings(model.predict(main['X'][vi],num_threads=4),main,vi,valid)
            oof_by_recipe[recipe['id']].update(pred)
            if recipe['id']=='fixes50':base_oof.update(pred)
        print('Completed locked-budget inner fold',f,flush=True)
    devscores=[]
    for recipe in RECIPES:
        pure={u:oof_by_recipe[recipe['id']][u][:5] for u in dev}
        devscores.append({'recipe':recipe['id'],'weight':1.,'metrics':scores(pure,targets,dev)})
        if recipe['id']!='fixes50':
            for w in WEIGHTS[:-1]:devscores.append({'recipe':recipe['id'],'weight':w,'metrics':scores(fuse(oof_by_recipe[recipe['id']],base_oof,w,dev),targets,dev)})
    selected=max(devscores,key=lambda r:(r['metrics']['secondary'],r['metrics']['primary'],-next(x['k'] for x in RECIPES if x['id']==r['recipe'])))
    dump('selection.json',{'selected':selected,'recipes':RECIPES,'params':params,'rounds':rounds,
        'rule':'Highest development-only OOF secondary120 MAP, then primary MAP, then smaller pool. Fixed V5 parameters and 78-round budget; no outer tuning.',
        'ensemble':'Not attempted: deterministic fixed-budget models use no bagging or feature subsampling; seed-only ensemble would be redundant.'})
    comparison={'reproduced_V5':scores(saved,targets,outer)};models={};meta={};pred_by_recipe={}
    for recipe in RECIPES:
        model,m=fit(recipe,main,early,targets['secondary'],etargets['secondary'],dev,params,rounds)
        model.save_model(str(OUT/f"evaluation_{recipe['id']}.txt"));models[recipe['id']]=model;meta[recipe['id']]=m
        vi=matrix_rows(main,outer,recipe['k']);ranks=rankings(model.predict(main['X'][vi],num_threads=4),main,vi,outer)
        pred_by_recipe[recipe['id']]=ranks;comparison[recipe['id']]=scores({u:r[:5] for u,r in ranks.items()},targets,outer)
        write_predictions(f"heldout_{recipe['id']}.csv",ranks)
    chosen_recipe=next(r for r in RECIPES if r['id']==selected['recipe'])
    chosen=fuse(pred_by_recipe[selected['recipe']],v5rank,selected['weight'],outer)
    comparison['selected']=scores(chosen,targets,outer);write_predictions('heldout_selected.csv',chosen)
    # Earlier chronological evaluation, with outer user IDs absent from all supervised examples.
    ei=matrix_rows(early,dev,50);EX,ey=dataset(early,etargets['secondary'],ei)
    em=lgb.train(params,lgb.Dataset(EX,label=ey,feature_name=names),num_boost_round=rounds)
    evi=matrix_rows(early,outer,50);erank=rankings(em.predict(early['X'][evi],num_threads=4),early,evi,outer)
    early_score=scores({u:r[:5] for u,r in erank.items()},etargets,outer)
    write_predictions('earlier_fold_heldout.csv',erank)
    # Final fitting only after selection and outer results have been frozen.
    allusers=validation.EvaluationProtocol().cohort
    final,fmeta=fit(chosen_recipe,main,early,targets['secondary'],etargets['secondary'],allusers,params,rounds)
    final.save_model(str(OUT/'final_model.txt'))
    gain=final.feature_importance('gain');importance=[{'feature':n,'gain':float(g),'splits':int(s)} for n,g,s in zip(names,gain,final.feature_importance('split'))]
    importance.sort(key=lambda r:-r['gain'])
    result={'status':'trained','development_oof_results':devscores,'selected':selected,'selected_recipe':chosen_recipe,
        'outer_comparisons':comparison,'earlier_fold':{'history':'Weeks11-15','target':'Weeks16-20','outer_metrics':early_score,
            'training_users':len(dev),'evaluation_users':len(outer),'overlap_with_main_target':False},
        'outer_users':outer,'development_users':dev,'candidate_recall':candidate_metrics,'fit_metadata':meta,
        'final_fit_metadata':fmeta,'feature_columns':names,'parameters':params,'rounds':rounds,
        'feature_importance':importance,'final_model_sha256':sha(OUT/'final_model.txt'),
        'v5_recommended_until_genuine_better_score':True,'private_score_v5_user_reported':.39706,
        'private_scores_used_as_labels':False,'private_scores_used_for_hyperparameters':False,
        'outer_population_caveat':'Reused project outer256 population. Excluded from supervised training for all periods and inner tuning, but previously examined in earlier project audits.',
        'seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    dump('experiment.json',result)
    print(json.dumps({'selected':selected,'outer_comparisons':comparison,'earlier_fold':early_score,
        'seconds':result['seconds'],'peak_rss_mib':result['peak_rss_mib']},indent=2))


if __name__=='__main__':main()
