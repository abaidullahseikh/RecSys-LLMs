"""Exactly two full-expanded-pool rankers; development-only selection and earlier check."""
import argparse,hashlib
from v12_common import *
from v8_common import ap

def bootstrap(new,old,truth,users):
    d=np.asarray([ap(truth[u],new[u])-ap(truth[u],old[u]) for u in users]);rng=np.random.default_rng(2026);means=[]
    for _ in range(20):means.extend(d[rng.integers(0,len(d),size=(100,len(d)))].mean(axis=1).tolist())
    return {'mean_delta':float(d.mean()),'interval95':np.quantile(means,[.025,.975]).tolist(),'resamples':2000,'seed':2026,'scope':'Descriptive reused-development-user bootstrap; not a Kaggle score interval or an independent confirmation'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['main','early']);a=parser.parse_args();phase='validation' if a.phase=='main' else 'early';t=time.monotonic();report='experiment.json' if a.phase=='main' else 'chronological_validation.json'
    if (OUT/report).exists():raise FileExistsError('Preserve completed experiment')
    if not read(OUT/'baseline_reproduction.json')['exact'] or not (OUT/'unseen_diagnosis.json').exists():raise ValueError('Reproduction/diagnosis required before fitting')
    z,truth=data(phase);base=read(OUT/'baseline_reproduction.json');users=tuple(dict.fromkeys(z['users']));selection_before=sha(OUT/'selection.json') if a.phase=='early' else None
    dev=tuple(base['development_users']) if a.phase=='main' else tuple(u for u in users if partition(u)!=0);ass=tuple(base['assessment_users']) if a.phase=='main' else tuple(u for u in users if partition(u)==0)
    dm=np.isin(z['users'],dev);fold=np.asarray([inner_fold(str(u)) for u in z['users']]);prof=history_profiles(phase)
    if a.phase=='main':targets=load(OUT/'validation_targets.npz')
    else:
        count,check,targetmeta=target_counts(z,'early')
        if check!=truth:raise ValueError('Earlier target labels differ')
        targets={'counts':count,'graded':grades(count),'binary':z['binary_label']};np.savez_compressed(OUT/'early_targets.npz',**targets);dump('early_target_manifest.json',targetmeta)
    if not np.array_equal(targets['binary'],z['binary_label']):raise ValueError('Binary baseline label changed')
    prefix='' if a.phase=='main' else 'early_'
    if a.phase=='main':
        dump('plan.json',{'experiments':{'E3':{'target':'Original local>=120s frequency Top5 binary membership','parameters':parameters('E3')},
            'E4':{'target':'All qualifying target-airing counts graded0/1/2/3 for0/1/2-3/4+; not official relevance','parameters':parameters('E4')}},
            'training_distribution':'All expanded rows; no hard-negative or random subsampling, weighting, forced positives, or unseen bonus',
            'candidate_policy':'Verified original five-generator Top50 union plus first50 positive-score PPR historical-unseen programs absent from original pool',
            'features':z['columns'].tolist(),'rounds':78,'seed':2026,'max_threads':4,'selection':'Maximum development OOF MAP among newly trained E3/E4; tie prefer binary E3. Freeze before either new assessment fit. Earlier chronological check is descriptive after selection.',
            'binary_vs_graded_scope_caveat':'E4 expands positive scope to all count-positive items, not only Top5. Differences therefore test this entire frequency-grade target, not pure reweighting of identical positive IDs.',
            'original_expanded_cache_sha256':sha(V11/'validation.npz'),'target_manifest':base['target_provenance'],'no_parameter_search':True,'no_new_algorithms':True})
    scores={n:np.zeros(len(z['users'])) for n in ('E3','E4')};meta={n:[] for n in scores};timings={n:{'training_seconds':0.,'inference_seconds':0.} for n in scores}
    for f in range(4 if a.phase=='main' else 3):
        ti=np.flatnonzero(dm&(fold!=f)) if f<3 else np.flatnonzero(dm);vi=np.flatnonzero(dm&(fold==f)) if f<3 else np.flatnonzero(~dm)
        fitusers=set(str(u) for u in z['users'][ti]);validusers=set(str(u) for u in z['users'][vi])
        if fitusers&validusers or fitusers&set(ass):raise ValueError('Supervised user leakage')
        groups=grouping(z['users'][ti]);gp=OUT/f'{prefix}inner{f}_groups.npy';np.save(gp,groups,allow_pickle=False)
        for n in scores:
            y=targets['binary'] if n=='E3' else targets['graded'];path=OUT/f'{prefix}{n}_inner{f}.txt';bst,fit_seconds=fit(z,y,ti,path,n)
            t0=time.monotonic();scores[n][vi]=bst.predict(z['X'][vi],num_threads=4);infer=time.monotonic()-t0;timings[n]['training_seconds']+=fit_seconds;timings[n]['inference_seconds']+=infer
            meta[n].append({'fold':f,'fit_users':sorted(fitusers),'evaluation_users':sorted(validusers),'training_rows':len(ti),'validation_rows':len(vi),'query_groups':len(groups),
                'group_size_sum':int(groups.sum()),'groups_sha256':sha(gp),'model_path':str(path),'model_sha256':sha(path),'label_histogram':dict(Counter(str(int(x)) for x in y[ti])),
                'training_seconds':fit_seconds,'inference_seconds':infer,'peak_rss_mib':rss()})
        if f==2 and a.phase=='main':
            ds={n:metrics(ranks(s,z,dev),truth,dev,prof) for n,s in scores.items()};selected=max(ds,key=lambda n:(ds[n]['MAP5'],n=='E3'))
            dump('selection.json',{'selected':selected,'development':ds,'baseline_V10_development_MAP':base['comparison']['V10']['development']['MAP5'],
                'baseline_V11_development_MAP':base['comparison']['V11']['development']['MAP5'],'frozen_before_new_assessment_fits':True,'rule':'Highest development MAP among genuine new E3/E4 only; tie E3'})
        print(a.phase,'fold',f,'trained full expanded rows',len(ti),flush=True)
    result={};ranking={}
    for n,s in scores.items():
        rr=ranks(s,z,users if a.phase=='main' else dev);ranking[n]=rr;np.save(OUT/f'{prefix}{n}_scores.npy',s,allow_pickle=False);save_pred(f'{prefix}{n}_crossfit.csv',rr)
        result[n]={'development':metrics(rr,truth,dev,prof),'assessment':metrics(rr,truth,ass,prof) if a.phase=='main' else None,'models':meta[n],
            'objective':'lambdarank','parameters':parameters(n),'rounds':78,'features':77,'seed':2026,**timings[n],'peak_rss_mib':max(m['peak_rss_mib'] for m in meta[n]),'verification_status':'Runtime schema/group/separation assertions passed; independent verification pending'}
    # Reuse original V10 models for E1/E2, replayed on the appropriate historical cutoff.
    if a.phase=='main':bs=np.load(OUT/'V10_expanded_scores.npy',allow_pickle=False)
    else:
        bs=np.zeros(len(z['users']))
        for f in range(3):
            vi=np.flatnonzero(dm&(fold==f));bs[vi]=lgb.Booster(model_file=str(V11/f'early_E1_inner{f}.txt')).predict(z['X'][vi],num_threads=4)
    for n in ('E1','E2'):
        rr=ranks(bs,z,users if a.phase=='main' else dev,z['original'] if n=='E1' else None);ranking[n]=rr;save_pred(f'{prefix}{n}_crossfit.csv',rr)
        result[n]={'development':metrics(rr,truth,dev,prof),'assessment':metrics(rr,truth,ass,prof) if a.phase=='main' else None,'objective':'lambdarank','parameters':parameters('E3'),'rounds':78,'features':77,'seed':2026,
            'training_seconds':0.,'training_runtime_note':'Reused baseline models, no V12 baseline retraining','inference_seconds':base['comparison']['V10']['replay_inference_seconds'] if a.phase=='main' else None,
            'peak_rss_mib':rss(),'verification_status':'Baseline exact reproduction passed; new comparison verification pending'}
    foldrows=[]
    for n in result:
        for f in range(3):
            uu=tuple(u for u in dev if inner_fold(u)==f);foldrows.append({'id':n,'fold':f,'users':len(uu),'MAP5':map5(ranking[n],truth,uu),'phase':a.phase})
    table(prefix+'fold_scores.csv',foldrows)
    if a.phase=='main':
        old11={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(V11/'E3_all_crossfit.csv')};uncertainty={n:{'versus_V10':bootstrap(ranking[n],ranking['E1'],truth,dev),'versus_V11':bootstrap(ranking[n],old11,truth,dev)} for n in ('E3','E4')}
        dump(report,{'comparison':result,'selected':read(OUT/'selection.json')['selected'],'uncertainty':uncertainty,'folds':foldrows,
            'prior_V11_E4_not_duplicated':{'development_MAP':read(V11/'hard_negative_experiment.json')['comparison']['E4']['development']['MAP5'],'fitting_distribution':'Sampled negatives, versus all rows in V12'},
            'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    else:
        if selection_before!=sha(OUT/'selection.json'):raise ValueError('Selection changed during reliability check')
        mainz,maintruth=data();mainprof=history_profiles();mainmatched={}
        for n in result:
            ss=np.load(OUT/('V10_expanded_scores.npy' if n in ('E1','E2') else f'{n}_scores.npy'),allow_pickle=False);rr=ranks(ss,mainz,dev,mainz['original'] if n=='E1' else None);mainmatched[n]=metrics(rr,maintruth,dev,mainprof)
        dump(report,{'comparison':result,'main_matched_template_development':mainmatched,'history':[1583704800,1586725200],'target':[1586725200,1589749200],
            'users':len(dev),'selection_frozen_sha256':selection_before,'folds':foldrows,
            'limitations':['Earlier5/5wk and main10/10wk history/target horizons differ; original schema normalization retained. Earlier cohort was examined in V6/V11.',
                'Main development and assessment are repeatedly examined. Earlier results describe reliability after frozen main-development selection; no chronological or assessment retuning.',
                'Only20wk logged behavior; final20wk history/12wk target differs from local folds. Official hidden target definition is not confirmed.'],
            'seconds':time.monotonic()-t,'peak_rss_mib':rss()})
    print({'phase':a.phase,'selected':read(OUT/'selection.json')['selected'],'comparison':{n:{g:r[g] for g in ('development','assessment')} for n,r in result.items()},'seconds':time.monotonic()-t,'peak_rss_mib':rss()},flush=True)

if __name__=='__main__':main()
