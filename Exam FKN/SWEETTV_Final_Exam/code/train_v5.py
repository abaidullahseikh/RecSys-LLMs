"""Bounded native LightGBM experiment. Outer users never select models/blends."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import resource
import time
import lightgbm as lgb
import numpy as np
import evaluation
import validation
from features_v5 import OUT, ROOT, records, sha

WEIGHTS=(.25,.5,.75,1.)
SPECS=({'id':'leaves15','num_leaves':15,'min_data_in_leaf':100},
       {'id':'leaves31','num_leaves':31,'min_data_in_leaf':100},
       {'id':'leaves31_no_cbf','num_leaves':31,'min_data_in_leaf':100})


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def user_hash(u,prefix=''):
    return int(hashlib.sha256((prefix+str(u)).encode()).hexdigest()[:8],16)


def predictions(scores,X,users,items,cohort,names,weight=1.):
    """RRF between reranker candidate rank and original V2 candidate rank."""
    grouped=defaultdict(list)
    for n,u in enumerate(users):grouped[str(u)].append(n)
    ppr=names.index('ppr_rank');pf=names.index('personal_frequency_rank')
    pop=names.index('popularity_rank');cold=names.index('user_cold_start')
    result={}
    for u in cohort:
        ix=np.asarray(grouped[u]);numeric=np.array([int(items[i]) for i in ix],dtype=np.int64)
        popularity=X[ix,pop]
        if X[ix[0],cold]>.5:
            order=np.lexsort((numeric,popularity))
        else:
            base=.5/(60+X[ix,ppr].astype(float))+.5/(60+X[ix,pf].astype(float))
            base_order=np.lexsort((numeric,popularity,-base));lgb_order=np.lexsort((numeric,popularity,-scores[ix]))
            bpos=np.empty(len(ix),int);lpos=np.empty(len(ix),int)
            bpos[base_order]=np.arange(len(ix));lpos[lgb_order]=np.arange(len(ix))
            fused=weight/(60+lpos)+(1-weight)/(60+bpos)
            order=np.lexsort((numeric,popularity,-fused))
        result[u]=tuple(str(items[ix[j]]) for j in order[:5])
    return result


def ap(labels,items):
    hits=0;terms=[]
    for k,i in enumerate(items,1):
        if i in labels:hits+=1;terms.append(hits/k)
    return math.fsum(terms)/min(5,len(labels)) if labels else 0.


def metrics(pred,targets,cohort):
    return {name:math.fsum(ap(truth[u],pred[u]) for u in cohort)/len(cohort) for name,truth in targets.items()}


def save_pred(path,pred):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id'])
        w.writerows((u,' '.join(v)) for u,v in pred.items())


def parameters(spec):
    return {'objective':'binary','metric':'None','learning_rate':.05,'num_leaves':spec['num_leaves'],
        'min_data_in_leaf':spec['min_data_in_leaf'],'lambda_l2':5.,'feature_fraction':1.,
        'bagging_fraction':1.,'num_threads':4,'seed':2026,'deterministic':True,'force_col_wise':True,'verbosity':-1}


def selected_columns(spec,names):
    excluded={'cbf_similarity','cbf_rank','source_cbf','user_category_affinity','user_genre_affinity'} if spec['id'].endswith('no_cbf') else set()
    return [i for i,c in enumerate(names) if c not in excluded]


def main():
    started=time.monotonic()
    if (OUT/'experiment.json').exists():raise FileExistsError('Preserve completed V5 experiment')
    z=np.load(OUT/'validation_features.npz');X=z['X'];us=z['users'];it=z['items'];names=z['columns'].tolist()
    protocol=validation.EvaluationProtocol();cohort=protocol.cohort
    tr=ROOT/'results/step10_lightgbm/candidate_features_top50.csv.gz'
    original=list(records(tr))
    y=np.array([int(r['label_top5_freq120']) for r in original],dtype=np.int8)
    y80=np.array([int(r['label_top5_fraction80']) for r in original],dtype=np.int8)
    if [r['user_id'] for r in original]!=us.tolist() or [r['tv_show_id'] for r in original]!=it.tolist():raise ValueError('Feature identity misalignment')
    secondary={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    from run_cbf_step9 import load_fraction80_top5_truth
    frac,_=load_fraction80_top5_truth(protocol)
    targets={'primary_all120':protocol.truth,'secondary_top5_120':secondary,'hypothesis_top5_80':frac}
    outer=tuple(u for u in cohort if user_hash(u)%5==0);dev=tuple(u for u in cohort if u not in set(outer))
    devmask=np.array([u in set(dev) for u in us]);outermask=~devmask
    fold=np.array([user_hash(u,'v5-inner:')%3 for u in us]);spec_results=[];oofs={}
    v2path=ROOT/'results/step8/validation_predictions/rrf_ppr_personal_frequency_ppr_weight_0.50/predictions.csv'
    v2=evaluation.parse_prediction_records(list(records(v2path)))
    basepred=predictions(np.zeros(len(us)),X,us,it,dev,names,0.)
    base_diff=sum(basepred[u]!=v2[u] for u in dev)
    for spec in SPECS:
        cols=selected_columns(spec,names);oof=np.zeros(len(y));rounds=[];fold_metrics=[]
        for f in range(3):
            ti=np.flatnonzero(devmask&(fold!=f));vi=np.flatnonzero(devmask&(fold==f));fold_users=tuple(u for u in dev if user_hash(u,'v5-inner:')%3==f)
            dtrain=lgb.Dataset(X[ti][:,cols],label=y[ti],feature_name=[names[i] for i in cols])
            dval=lgb.Dataset(X[vi][:,cols],label=y[vi],reference=dtrain)
            def feval(s,data):
                pred=predictions(s,X[vi],us[vi],it[vi],fold_users,names,1.)
                return 'map5',metrics(pred,{'label':secondary},fold_users)['label'],True
            booster=lgb.train(parameters(spec),dtrain,num_boost_round=160,valid_sets=[dval],feval=feval,
                callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(0)])
            oof[vi]=booster.predict(X[vi][:,cols],num_threads=4);rounds.append(booster.best_iteration)
            fold_metrics.append(metrics(predictions(oof[vi],X[vi],us[vi],it[vi],fold_users,names),targets,fold_users))
        variants=[]
        for w in WEIGHTS:
            pred=predictions(oof[devmask],X[devmask],us[devmask],it[devmask],dev,names,w)
            variants.append({'lgb_weight':w,**metrics(pred,targets,dev)})
        spec_results.append({'spec':spec,'rounds':rounds,'fold_metrics':fold_metrics,'blend_oof_metrics':variants})
        oofs[spec['id']]=oof
        print('Inner user-CV',spec['id'],rounds,variants,flush=True)
    choices=[(r,v) for r in spec_results for v in r['blend_oof_metrics']]
    best,bestblend=max(choices,key=lambda rv:(rv[1]['secondary_top5_120'],rv[1]['primary_all120'],-rv[0]['spec']['num_leaves']))
    spec=best['spec'];cols=selected_columns(spec,names);rounds=int(np.median(best['rounds']));weight=bestblend['lgb_weight']
    selection={'spec':spec,'parameters':parameters(spec),'rounds':rounds,'lgb_weight':weight,
        'feature_indices':cols,'feature_names':[names[i] for i in cols],
        'selection_rule':'Maximize development-user OOF Top5-120 MAP; primary MAP then fewer leaves break ties. Outer256 users excluded from tuning and stopping.',
        'frozen_before_outer_evaluation':True}
    dump('selection.json',selection)
    booster=lgb.train(parameters(spec),lgb.Dataset(X[devmask][:,cols],label=y[devmask],feature_name=[names[i] for i in cols]),num_boost_round=rounds)
    booster.save_model(str(OUT/'evaluation_model.txt'))
    os=booster.predict(X[outermask][:,cols],num_threads=4)
    pred=predictions(os,X[outermask],us[outermask],it[outermask],outer,names,weight)
    pure=predictions(os,X[outermask],us[outermask],it[outermask],outer,names,1.)
    save_pred(OUT/'heldout_v5.csv',pred);save_pred(OUT/'heldout_lightgbm_only.csv',pure)
    comparisons={'V5_selected_blend':metrics(pred,targets,outer),'V5_lightgbm_only':metrics(pure,targets,outer),'V2':metrics(v2,targets,outer)}
    baseline_paths={'PPR':ROOT/'results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv',
        'PersonalFrequency':ROOT/'results/step4/personal_frequency_qualifying_airings_predictions.csv',
        'V1_PPR':ROOT/'results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv',
        'V3':ROOT/'results/step9_cbf/best_validation_candidate.csv'}
    for name,path in baseline_paths.items():comparisons[name]=metrics(evaluation.parse_prediction_records(list(records(path))),targets,outer)
    # Independently documented alternative label fit. It cannot replace the frozen main selection.
    alt=lgb.train(parameters(spec),lgb.Dataset(X[devmask][:,cols],label=y80[devmask],feature_name=[names[i] for i in cols]),num_boost_round=rounds)
    altpred=predictions(alt.predict(X[outermask][:,cols],num_threads=4),X[outermask],us[outermask],it[outermask],outer,names,1.)
    save_pred(OUT/'heldout_80label_diagnostic.csv',altpred);comparisons['LightGBM_80label_diagnostic']=metrics(altpred,targets,outer)
    # Source-based corrected feature ablation uses same fixed model configuration; diagnostic only.
    importance=[{'feature':names[i],'gain':float(g),'splits':int(s)} for i,g,s in zip(cols,booster.feature_importance('gain'),booster.feature_importance('split'))]
    importance.sort(key=lambda r:-r['gain'])
    paired=np.array([ap(secondary[u],pred[u])-ap(secondary[u],v2[u]) for u in outer]);rng=np.random.default_rng(2026)
    ci=np.quantile([np.mean(paired[rng.integers(0,len(outer),len(outer))]) for _ in range(2000)],[.025,.975]).tolist()
    final=lgb.train(parameters(spec),lgb.Dataset(X[:,cols],label=y,feature_name=[names[i] for i in cols]),num_boost_round=rounds)
    final.save_model(str(OUT/'final_model.txt'))
    result={'status':'trained','selection':selection,'inner_cv':spec_results,'outer_comparisons':comparisons,
        'outer_users':list(outer),'development_users':list(dev),'rows':len(y),'columns':names,
        'base_candidate_rank_v2_top5_differences_development':base_diff,'feature_importance':importance,
        'paired_secondary_delta_ci95':ci,'outer_improved_users':int(np.sum(paired>1e-12)),'outer_worsened_users':int(np.sum(paired< -1e-12)),
        'v2_recommendation':'Remain designated final pending independent verification and a genuine superior Kaggle score',
        'evaluation_model_sha256':sha(OUT/'evaluation_model.txt'),'final_model_sha256':sha(OUT/'final_model.txt'),
        'feature_file_sha256':sha(OUT/'validation_features.npz'),'target':'Top5 by >=120s qualifying-airing frequency',
        'kaggle_scores_used_for_selection':False,'heldout_caveat':'Outer users were excluded from this supervised training/search; previous project model choices and V4 audit used this population. Not a pristine new holdout.',
        'training_seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'versions':{'lightgbm':lgb.__version__,'numpy':np.__version__}}
    dump('experiment.json',result)
    print(json.dumps({'selection':selection,'comparisons':comparisons,'ci95':ci,'seconds':result['training_seconds']},indent=2))


if __name__=='__main__':main()
