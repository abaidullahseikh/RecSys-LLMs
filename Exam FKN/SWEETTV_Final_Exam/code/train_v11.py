"""One controlled nested-OOF hard-negative strategy on original/expanded pools."""
import time
import numpy as np
import lightgbm as lgb
from v11_common import *

def mining_scores(z,fit_users,prefix,params):
    """Nested miners never fit outer validation or assessment labels."""
    fitmask=np.isin(z['users'],fit_users);splits=np.asarray([digest_split(str(u)) for u in z['users']]);score=np.full(len(z['users']),np.nan);meta=[]
    for k in range(2):
        ti=np.flatnonzero(fitmask&(splits!=k)&z['original']);si=np.flatnonzero(fitmask&(splits==k));mineusers=set(z['users'][si]);trainusers=set(z['users'][ti])
        if trainusers&mineusers or not trainusers<=set(fit_users) or not mineusers<=set(fit_users):raise ValueError('Nested mining leakage')
        path=OUT/f'{prefix}_miner{k}.txt';bst=fit(z,ti,path,params);score[si]=bst.predict(z['X'][si],num_threads=4)
        meta.append({'split':k,'fit_users':sorted(str(u) for u in trainusers),'score_users':sorted(str(u) for u in mineusers),'model_sha256':sha(path)})
    if not np.isfinite(score[fitmask]).all():raise ValueError('Mining scores missing')
    return score,meta

def main():
    t=time.monotonic()
    if (OUT/'hard_negative_experiment.json').exists():raise FileExistsError('Preserve controlled experiment')
    z,truth=model_data();ex=read(OUT/'baseline_reproduction.json');params=ex['parameters'];dev=tuple(ex['development_users']);ass=tuple(ex['assessment_users']);users=tuple(dict.fromkeys(z['users']));dm=np.isin(z['users'],dev)
    folds=np.asarray([inner_fold(str(u)) for u in z['users']]);prof=profiles('train');oof={n:np.zeros(len(z['users'])) for n in ('E3','E4')};models={n:[] for n in oof};mining=[]
    dump('hard_negative_plan.json',{'strategy':'Keep every available positive; top20 OOF-miner negative examples plus20 deterministic hash-sampled remaining negatives per user. Evaluation retains entire candidate pools.',
        'nested_mining':'For each development outer fold, two miners fit only other outer-training users split by v11-miner SHA256 mod2. No outer evaluation or assessment target labels enter a miner or sampling.',
        'E3':'Original pool at fitting and evaluation','E4':'Expanded pool at fitting and evaluation; mining baseline trained only original rows',
        'gate':read(OUT/'E1_E2.json')['E4_gate'],'parameters':params,'rounds':78,'seed':2026,'max_threads':4,'search':False,
        'selection':'Best development MAP among genuine new E2/E3/E4; E1 reported as scored V10 baseline, not renamed. Ties favor E3, E2, E4 in that order. Freeze before assessment evaluation.'})
    for f in range(4):
        ts=time.monotonic();fitusers=tuple(u for u in dev if inner_fold(u)!=f) if f<3 else dev
        validusers=tuple(u for u in dev if inner_fold(u)==f) if f<3 else ass
        ti=np.flatnonzero(np.isin(z['users'],fitusers));vi=np.flatnonzero(np.isin(z['users'],validusers))
        if set(fitusers)&set(validusers) or set(fitusers)&set(ass):raise ValueError('Outer supervised leakage')
        # Reuse frozen V10 development OOF mining for the assessment fit. All its miners exclude assessment.
        if f==3:
            score=np.load(OUT/'expanded_V10_crossfit_scores.npy',allow_pickle=False);mm={'source':'Verified V10 development OOF models; each training user absent from its miner; all assessment users excluded from fitting'}
        else:score,mm=mining_scores(z,fitusers,f'outer{f}',params)
        mining.append({'outer_fold':f,'metadata':mm})
        for name in ('E3','E4'):
            ii=ti[z['original'][ti]] if name=='E3' else ti;vv=vi[z['original'][vi]] if name=='E3' else vi
            selected=sample_negatives(z,ii,score)
            if not set(ii[z['binary_label'][ii]>0])<=set(selected):raise ValueError('Positive dropped')
            path=OUT/f'{name}_inner{f}.txt';bst=fit(z,selected,path,params);oof[name][vv]=bst.predict(z['X'][vv],num_threads=4)
            np.save(OUT/f'{name}_inner{f}_training_indices.npy',selected,allow_pickle=False)
            models[name].append({'fold':f,'fit_users':len(fitusers),'valid_users':len(validusers),'available_rows':len(ii),'sampled_rows':len(selected),'positives':int(z['binary_label'][selected].sum()),
                'model_path':str(path),'model_sha256':sha(path),'sample_indices_sha256':sha(OUT/f'{name}_inner{f}_training_indices.npy'),'outer_seconds_so_far':time.monotonic()-ts})
        if f==2:
            devresult={name:measured(ranks(oof[name],z,dev,z['original'] if name=='E3' else None),truth,dev,prof) for name in oof}
            candidates={'E2':read(OUT/'E1_E2.json')['comparison']['E2']['development'],**devresult};tie={'E3':2,'E2':1,'E4':0};selected=max(candidates,key=lambda n:(candidates[n]['MAP5'],tie[n]))
            dump('selection.json',{'selected':selected,'development':candidates,'frozen_before_assessment_scoring':True,'tie_preference':['E3','E2','E4'],'baseline_V10_development_MAP':ex['development_MAP']})
        print('hard negative outer',f,'seconds',time.monotonic()-ts,flush=True)
    result={}
    for name in oof:
        rr=ranks(oof[name],z,users,z['original'] if name=='E3' else None);save_pred(name+'_all_crossfit.csv',rr);np.save(OUT/f'{name}_crossfit_scores.npy',oof[name],allow_pickle=False)
        result[name]={'development':measured(rr,truth,dev,prof),'assessment':measured(rr,truth,ass,prof),'objective':'lambdarank','features':77,'parameters':params,'rounds':78,'seed':2026,
            'models':models[name],'runtime_seconds_shared':time.monotonic()-t,'peak_rss_mib':rss(),'verification_status':'Nested user-separated mining and positive-retention assertions passed; final independent verification pending'}
    dump('hard_negative_experiment.json',{'comparison':result,'mining':mining,'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'selected':read(OUT/'selection.json')['selected'],'comparison':{n:{s:m[s] for s in ('development','assessment')} for n,m in result.items()},'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
