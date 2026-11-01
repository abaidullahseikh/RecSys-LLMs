"""E1/E2: replay V10 across original and label-free expanded candidate pools."""
import time
import numpy as np
import lightgbm as lgb
from v11_common import *

def main():
    t=time.monotonic()
    if (OUT/'E1_E2.json').exists():raise FileExistsError('Preserve E1/E2 evaluation')
    z,truth=model_data();ex=read(OUT/'baseline_reproduction.json');dev=tuple(ex['development_users']);ass=tuple(ex['assessment_users']);users=tuple(dict.fromkeys(z['users']));dm=np.isin(z['users'],dev)
    fold=np.asarray([inner_fold(str(u)) for u in z['users']]);scores=np.zeros(len(z['users']));times=[]
    for f in range(4):
        ti=time.monotonic();ix=np.flatnonzero(dm&(fold==f)) if f<3 else np.flatnonzero(~dm);path=V10/(f'pairwise_inner{f}.txt' if f<3 else 'pairwise_assessment.txt')
        bst=lgb.Booster(model_file=str(path))
        if bst.feature_name()!=z['columns'].tolist():raise ValueError('Schema mismatch')
        scores[ix]=bst.predict(z['X'][ix],num_threads=4);times.append(time.monotonic()-ti)
    r1=ranks(scores,z,users,z['original']);r2=ranks(scores,z,users);saved={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(V10/'selected_all_crossfit.csv')}
    if {u:r[:5] for u,r in r1.items()}!=saved:raise ValueError('E1 changed on expanded feature matrix')
    prof=profiles('train');result={}
    for name,rr in [('E1',r1),('E2',r2)]:
        result[name]={'development':measured(rr,truth,dev,prof),'assessment':measured(rr,truth,ass,prof),'objective':'lambdarank','features':77,'parameters':ex['parameters'],'rounds':78,'seed':2026,
            'inference_seconds':sum(times),'retrieval_build_seconds':read(OUT/'validation_build.json')['seconds'],'peak_rss_mib':rss(),'verification_status':'E1 exact V10 reproduction; source rows preserved; full schema checked'}
        save_pred(name+'_all_crossfit.csv',rr)
    np.save(OUT/'expanded_V10_crossfit_scores.npy',scores,allow_pickle=False)
    delta=result['E2']['development']['MAP5']-result['E1']['development']['MAP5']
    dump('E1_E2.json',{'comparison':result,'development_MAP_delta':delta,'seconds':time.monotonic()-t,'peak_rss_mib':rss(),
        'interpretation':'Additional unseen candidates compete for five slots; original model learned truncated top50-union source-rank distribution. Extra candidates can displace correct old items without sufficiently reliable calibration. Recall gain is not ranking gain.' if delta<0 else 'Expanded unseen PPR retrieval improves development MAP with original model; assess hard-negative correction separately.',
        'E4_gate':'Expanded development recall increased and all extra features are available from the same historical generator pipeline; E4 permitted independent of E2 MAP outcome.'})
    print({'comparison':{k:{g:v[g] for g in ('development','assessment')} for k,v in result.items()},'delta':delta},flush=True)

if __name__=='__main__':main()
