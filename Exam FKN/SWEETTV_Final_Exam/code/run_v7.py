"""Fast E1/V5, E2/LambdaRank, E3/one new repeat feature experiment.

Same frozen candidates, labels and user-group folds as V5; no broad search.
Creates only new V7 artifacts. Existing V1--V6 evidence is read-only.
"""
from collections import defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import lightgbm as lgb
import numpy as np
import evaluation
import validation
from train_v5 import predictions,parameters,user_hash
from verify_v5 import rank_reference,ap

ROOT=validation.ROOT
OUT=ROOT/'results/v7_experiment'
V5=ROOT/'results/v5_experiment'
ZONE=ZoneInfo('Europe/Kyiv')
FEATURE='trailing_consecutive_active_weeks'
IDS=('E1_V5','E2_LambdaRank','E3_repeat_streak')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def grouping(users):
    """LightGBM groups must be contiguous and cannot mix user identities."""
    if not len(users):raise ValueError('Empty ranking groups')
    starts=np.r_[0,np.flatnonzero(users[1:]!=users[:-1])+1]
    keys=users[starts]
    if len(set(keys.tolist()))!=len(keys):raise ValueError('User groups are not contiguous')
    return np.diff(np.r_[starts,len(users)]).astype(np.int32)


def streak(mask):
    n=0
    while mask&1:n+=1;mask>>=1
    return n


def streak_feature(z,period,end):
    """Use only qualifying historical airings within trailing 10 calendar weeks."""
    t=time.monotonic();endlocal=datetime.fromtimestamp(end,ZONE);lo=int((endlocal-timedelta(weeks=10)).timestamp())
    wanted=set(zip(z['users'].tolist(),z['items'].tolist()));bits=defaultdict(int);cross=0;used=0
    source=validation.STEP2/f'{period}_airing_overlaps.csv.gz'
    expected=read(validation.STEP2/'join_manifest.json')['artifacts'][source.name]['sha256']
    if sha(source)!=expected:raise ValueError('Historical overlap source hash mismatch')
    for r in records(source):
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not first<last<=end:raise ValueError('Future behavior found in repeat feature source')
        if first<lo:cross+=last>lo;continue
        if int(r['overlap_seconds'])<120:continue
        key=r['user_id'],r['tv_show_id']
        if key not in wanted:continue
        date=datetime.fromtimestamp(first,ZONE).date()
        age=((endlocal.date()-date).days-1)//7
        if not 0<=age<10:raise ValueError('Historical week age outside window')
        bits[key]|=1<<age;used+=1
    values=np.asarray([streak(bits.get((str(u),str(i)),0)) for u,i in zip(z['users'],z['items'])],dtype=np.float32)
    return values,{'feature':FEATURE,'definition':'Consecutive completed local calendar weeks active immediately before cutoff, starting with the latest completed week; stop at first gap; >=120s qualifying airing; maximum10.',
        'history_window_utc':[lo,end],'source_path':str(source.relative_to(ROOT)),'source_sha256':expected,
        'qualifying_candidate_airings_used':used,'boundary_crossing_rows_excluded':cross,
        'nonzero_rows':int(np.count_nonzero(values)),'max':float(values.max()),'seconds':time.monotonic()-t}


def model_params(name,base):
    result=dict(base)
    if name=='E2_LambdaRank':
        result.update(objective='lambdarank',label_gain=[0,1],lambdarank_truncation_level=8,lambdarank_norm=True,eval_at=[5])
    result.update(seed=2026,num_threads=4,metric='None')
    return result


def matrix(name,X,repeat):
    return np.column_stack((X,repeat)) if name=='E3_repeat_streak' else X


def map5(pred,truth,cohort):return math.fsum(ap(truth[u],pred[u]) for u in cohort)/len(cohort)


def save_preds(name,pred):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(p)) for u,p in pred.items())


def fit(name,X,y,us,names,params,rounds,valid=None,feval=None):
    kwargs={'group':grouping(us)} if name=='E2_LambdaRank' else {}
    dtrain=lgb.Dataset(X,label=y,feature_name=names,**kwargs)
    if valid is None:return lgb.train(params,dtrain,num_boost_round=rounds)
    VX,VY,VU=valid;vkwargs={'group':grouping(VU)} if name=='E2_LambdaRank' else {}
    dval=lgb.Dataset(VX,label=VY,reference=dtrain,**vkwargs)
    return lgb.train(params,dtrain,num_boost_round=rounds,valid_sets=[dval],feval=feval,
        callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(0)])


def main():
    start=time.monotonic()
    if OUT.exists():raise FileExistsError('Preserve completed/partial V7 run; investigate before rerunning')
    OUT.mkdir()
    protected=list((ROOT/'submission').glob('*'))+list((ROOT/'evidence').glob('*session*.json'))
    for d in ('v5_experiment','v6_experiment'):protected+=list((ROOT/'results'/d).glob('*'))
    protected+=list((ROOT/'code').glob('*v5*.py'))+list((ROOT/'code').glob('*v6*.py'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()};dump('preservation_before.json',before)
    v5=read(V5/'experiment.json');sel=v5['selection'];fm=read(V5/'feature_manifest.json');protocol=validation.EvaluationProtocol()
    z=archive(V5/'validation_features.npz');X=z['X'];us=z['users'];it=z['items'];names=z['columns'].tolist()
    dev=tuple(v5['development_users']);outer=tuple(v5['outer_users']);devset=set(dev);outerset=set(outer)
    if devset&outerset or devset|outerset!=set(protocol.cohort):raise ValueError('Bad user groups')
    if names!=sel['feature_names'] or names!=fm['features']:raise ValueError('V5 feature schema differs')
    if sha(V5/'validation_features.npz')!=fm['validation']['features_sha256'] or sha(V5/'evaluation_model.txt')!=v5['evaluation_model_sha256']:raise ValueError('V5 checksum mismatch')
    if fm['validation']['trailing_history_utc']!=[1583704800,1589749200] or fm['validation']['target_window']!=[1589749200,1595797200]:raise ValueError('V5 temporal boundaries changed')
    truth={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
    y=np.asarray([str(i) in truth[str(u)] for u,i in zip(us,it)],dtype=np.int8)
    eligible=set(protocol.candidates)
    if not set(it)<=eligible:raise ValueError('Invalid V5 candidate')
    dm=np.isin(us,dev);om=np.isin(us,outer);grouping(us)
    m=lgb.Booster(model_file=str(V5/'evaluation_model.txt'));os=m.predict(X[om],num_threads=4)
    reproduced=rank_reference(os,X[om],us[om],it[om],outer,names,1.)
    saved=evaluation.parse_prediction_records(list(records(V5/'heldout_v5.csv')))
    value=map5(reproduced,truth,outer)
    if reproduced!=saved or not math.isclose(value,.4886328125,rel_tol=0,abs_tol=1e-15):raise ValueError('V5 reproduction failed; stopping')
    print('V5 reproduced exactly:',value,flush=True)
    dump('v5_reproduction.json',{'passed':True,'MAP':value,'users':len(outer),'feature_count':len(names),
        'same_users':True,'same_candidates':True,'no_feature_rebuild':True,'training_window':fm['validation']['trailing_history_utc'],
        'original_model_sha256':sha(V5/'evaluation_model.txt'),'original_feature_sha256':sha(V5/'validation_features.npz')})
    required=['program_airings_recent_2w','program_airings_recent_4w','program_airings_recent_8w','program_active_weeks','user_program_recency_days']
    if not all(c in names for c in required) or FEATURE in names:raise ValueError('Feature presence audit changed')
    repeat,rmeta=streak_feature(z,'train',1589749200)
    np.save(OUT/'repeat_validation.npy',repeat,allow_pickle=False)
    dump('repeat_feature.json',{'existing_requested_features':required,'new':rmeta,'feature_sha256':sha(OUT/'repeat_validation.npy')})
    spec={'num_leaves':15,'min_data_in_leaf':100};base=parameters(spec)
    plan={'experiments':list(IDS),'configs':{n:model_params(n,base) for n in IDS},
        'seed':2026,'threads':4,'inner_folds':'same as V5: SHA256(v5-inner:+user) modulo3, development users only',
        'max_rounds':160,'early_stopping_patience':20,'metric_for_stopping':'Top5-120 MAP@5',
        'LambdaRank_note':'Pairwise LambdaRank optimizes truncated NDCG; binary label gains [0,1], eval_at5, truncation8 (K+3). Selection/stopping evaluated by MAP@5.',
        'selection_rule':'maximum inner-development OOF MAP@5; ties retain E1. Outer scores reported separately; no alternate model selected from outer scores.',
        'submission_gate':'Only inner-selected E2/E3 with both positive OOF gain and positive outer gain versus reproduced E1; otherwise no CSV.',
        'new_feature':FEATURE,'no_hyperparameter_grid':True,'Kaggle_scores_used_for_selection':False}
    dump('plan.json',plan)
    fold=np.asarray([user_hash(u,'v5-inner:')%3 for u in us],dtype=np.int8)
    cv={};cvtime=time.monotonic()
    for name in IDS:
        t=time.monotonic();M=matrix(name,X,repeat);fn=names+[FEATURE] if name=='E3_repeat_streak' else names
        oof=np.zeros(len(us));best_rounds=[];fold_results=[]
        for f in range(3):
            ti=np.flatnonzero(dm&(fold!=f));vi=np.flatnonzero(dm&(fold==f));vu=tuple(u for u in dev if user_hash(u,'v5-inner:')%3==f)
            def metric_callback(score,data):
                pred=predictions(score,M[vi],us[vi],it[vi],vu,fn,1.)
                return 'map5',map5(pred,truth,vu),True
            model=fit(name,M[ti],y[ti],us[ti],fn,plan['configs'][name],160,(M[vi],y[vi],us[vi]),metric_callback)
            oof[vi]=model.predict(M[vi],num_threads=4);best_rounds.append(model.best_iteration)
            model.save_model(str(OUT/f'{name}_inner{f}.txt'))
            fold_results.append({'fold':f,'train_users':len(set(us[ti])),'valid_users':len(vu),'train_rows':len(ti),'valid_rows':len(vi),'rounds':model.best_iteration})
        preds=predictions(oof[dm],M[dm],us[dm],it[dm],dev,fn,1.);score=map5(preds,truth,dev)
        save_preds(f'{name}_oof.csv',preds)
        cv[name]={'oof_MAP':score,'rounds':best_rounds,'final_rounds':int(np.median(best_rounds)),
            'folds':fold_results,'seconds':time.monotonic()-t,'feature_count':len(fn),
            'peak_rss_mib_so_far':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        print(name,'OOF',score,'rounds',best_rounds,flush=True)
    expected=next(row for row in v5['inner_cv'] if row['spec']['id']=='leaves15')['blend_oof_metrics']
    expected=next(row['secondary_top5_120'] for row in expected if row['lgb_weight']==1.)
    if not math.isclose(cv['E1_V5']['oof_MAP'],expected,rel_tol=0,abs_tol=1e-14):raise ValueError('E1 OOF baseline does not reproduce V5')
    selected=max(IDS,key=lambda n:cv[n]['oof_MAP'])
    dump('selection.json',{'selected':selected,'cv':cv,'frozen_before_outer_scoring':True,'rule':plan['selection_rule']})
    comparison={};models={};outer_pred={}
    for name in IDS:
        t=time.monotonic();M=matrix(name,X,repeat);fn=names+[FEATURE] if name=='E3_repeat_streak' else names
        model=fit(name,M[dm],y[dm],us[dm],fn,plan['configs'][name],cv[name]['final_rounds'])
        model.save_model(str(OUT/f'{name}_evaluation.txt'));models[name]=model
        preds=predictions(model.predict(M[om],num_threads=4),M[om],us[om],it[om],outer,fn,1.)
        if name=='E1_V5' and preds!=saved:raise ValueError('Retrained E1 outer predictions differ')
        save_preds(f'{name}_outer.csv',preds);outer_pred[name]=preds
        comparison[name]={'development_oof_MAP':cv[name]['oof_MAP'],'outer_MAP':map5(preds,truth,outer),
            'outer_primary_MAP':map5(preds,protocol.truth,outer),'rounds':cv[name]['final_rounds'],
            'outer_fit_predict_seconds':time.monotonic()-t,'feature_count':len(fn),'cv_seconds':cv[name]['seconds']}
    gate=selected!='E1_V5' and cv[selected]['oof_MAP']>cv['E1_V5']['oof_MAP']+1e-12 and comparison[selected]['outer_MAP']>value+1e-12
    importance={n:sorted([{'feature':c,'gain':float(g),'splits':int(s)} for c,g,s in zip(models[n].feature_name(),models[n].feature_importance('gain'),models[n].feature_importance('split'))],key=lambda r:-r['gain']) for n in IDS}
    paired=np.array([ap(truth[u],outer_pred[selected][u])-ap(truth[u],saved[u]) for u in outer]);rng=np.random.default_rng(2026)
    ci=np.quantile([paired[rng.integers(0,len(outer),len(outer))].mean() for _ in range(2000)],[.025,.975]).tolist()
    result={'status':'trained_and_evaluated','comparison':comparison,'selected':selected,'submission_gate_passed':gate,
        'outer_users':list(outer),'development_users':list(dev),'feature_schema':names,
        'selected_delta_ci95':ci,'feature_importance':importance,'candidate_policy':'Frozen V5 Top50 union, unchanged',
        'outer_reuse_caveat':'Same 256 users previously examined in V4-V6. Excluded from this training/selection; not a pristine new research holdout.',
        'seconds':time.monotonic()-start,'cv_seconds':time.monotonic()-cvtime,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'versions':{'lightgbm':lgb.__version__,'numpy':np.__version__},'submission_created':False,'Kaggle_score':None}
    # Final refit and feature schema only when the predeclared gate passes.
    if gate:
        name=selected;M=matrix(name,X,repeat);fn=names+[FEATURE] if name=='E3_repeat_streak' else names
        model=fit(name,M,y,us,fn,plan['configs'][name],cv[name]['final_rounds'])
        model.save_model(str(OUT/'final_model.txt'))
        result['final_model_sha256']=sha(OUT/'final_model.txt');result['final_feature_schema']=fn
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Protected prior evidence changed')
    dump('experiment.json',result)
    print(json.dumps({k:result[k] for k in ('selected','submission_gate_passed','comparison','seconds','peak_rss_mib')},indent=2))


if __name__=='__main__':main()
