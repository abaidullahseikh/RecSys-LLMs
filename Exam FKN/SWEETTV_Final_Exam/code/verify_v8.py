"""Independent target, new-feature, model replay, metric and optional CSV checks.

Reads shared saved75 features; independent reconstruction applies to all new
signals and target counts. It is not a second implementation of every generator.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo
import csv,gzip,json,math,time,traceback
import numpy as np
import lightgbm as lgb
import evaluation,validation

ROOT=validation.ROOT;OUT=ROOT/'results/v8_experiment';ZONE=ZoneInfo('Europe/Kyiv')


def read(p):return json.loads(Path(p).read_text())


def rows(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def sha(p):
    with Path(p).open('rb') as f:return __import__('hashlib').file_digest(f,'sha256').hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def hashvalue(u,prefix=''):return int(__import__('hashlib').sha256((prefix+u).encode()).hexdigest()[:8],16)


def ap(labels,ranking):
    labels=set(labels);r=list(ranking[:5]);result=Fraction(0)
    for i in labels:
        if i in r:
            k=r.index(i)+1;result+=Fraction(len(set(r[:k])&labels),k)
    return float(result/min(5,len(labels))) if labels else 0.


def measured(pred,truth,users):return math.fsum(ap(truth[u],pred[u]) for u in users)/len(users) if users else None


def reference_rank(scores,z,ix,users):
    names=z['columns'].tolist();cold=names.index('user_cold_start');pop=names.index('popularity_rank');g=defaultdict(list)
    for n,s in zip(ix,scores):g[str(z['users'][n])].append((int(n),float(s)))
    result={}
    for u in users:
        if not g[u]:raise ValueError('No candidates for user')
        c=z['X'][g[u][0][0],cold]>.5
        ls=sorted(g[u],key=lambda ns:((0 if c else -ns[1]),float(z['X'][ns[0],pop]),int(z['items'][ns[0]])))
        result[u]=tuple(str(z['items'][n]) for n,s in ls[:5])
    return result


def source_reference(z,period,end,cohort):
    """Independent Monday-date >=80% streak and last2-week frequency-share scan."""
    wanted=set(zip(z['users'].tolist(),z['items'].tolist()));start_dt=datetime.fromtimestamp(end,ZONE)-timedelta(weeks=10)
    lo=int(start_dt.timestamp());latest=datetime.fromtimestamp(end,ZONE).date()-timedelta(weeks=1)
    active=defaultdict(set);recent=Counter();user_total=Counter()
    for r in rows(validation.STEP2/f'{period}_airing_overlaps.csv.gz'):
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not first<last<=end:raise ValueError('Future history row')
        if first<lo or r['user_id'] not in cohort:continue
        secs=int(r['overlap_seconds']);dur=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        d=datetime.fromtimestamp(first,ZONE).date();monday=d-timedelta(days=d.weekday())
        if secs>=120 and monday>=latest-timedelta(weeks=1):user_total[r['user_id']]+=1
        key=r['user_id'],r['tv_show_id']
        if key not in wanted:continue
        if Fraction(secs,dur)>=Fraction(4,5):active[key].add(monday)
        if secs>=120 and monday>=latest-timedelta(weeks=1):recent[key]+=1
    values=np.empty((len(z['users']),2),dtype=np.float32)
    for n,(u,i) in enumerate(zip(z['users'],z['items'])):
        key=str(u),str(i);p=latest;streak=0
        while p in active[key] and streak<10:streak+=1;p-=timedelta(weeks=1)
        values[n]=[streak,recent[key]/max(1,user_total[str(u)])]
    return values


def targets_reference(cohort):
    a={u:Counter() for u in cohort};b={u:Counter() for u in cohort}
    for r in rows(validation.STEP2/'validation_airing_overlaps.csv.gz'):
        u=r['user_id']
        if u not in a:continue
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not 1589749200<=first<last<=1595797200:raise ValueError('Target window corruption')
        i=r['tv_show_id'];s=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if int(i)<=0 or not 0<s<=d:raise ValueError('Invalid normalized overlap')
        if s>=120:a[u][i]+=1
        if Fraction(s,d)>=Fraction(4,5):b[u][i]+=1
    top=lambda c:{u:sorted(v,key=lambda i:(-v[i],int(i)))[:5] for u,v in c.items()}
    return a,b,{'secondary':top(a),'fraction80':top(b),'primary':{u:sorted(v,key=int) for u,v in a.items()}}


def main():
    start=time.monotonic();checks=[];detail={};submission=ROOT/'submission/final_submission_v8_experimental.csv'
    phase='submission' if submission.exists() else 'validation';resultpath=OUT/f'{phase}_verification.json'
    if resultpath.exists():resultpath=OUT/f'{phase}_verification_recheck_{time.time_ns()}.json'
    def check(name,ok,info=None):
        checks.append({'name':name,'passed':bool(ok),'detail':info})
        if not ok:raise ValueError(name)
    try:
        ex=read(OUT/'experiment.json');pop=read(OUT/'population.json');dm=read(OUT/'data_manifest.json');plan=read(OUT/'plan.json');selection=read(OUT/'selection.json')
        z=load(OUT/'validation.npz');cohort=tuple(pop['users']);dev=tuple(ex['development_users']);assessment=tuple(ex['assessment_users']);eligible=set(dm['candidate_pool'])
        names=z['columns'].tolist();us=z['users'];items=z['items'];X=z['X']
        check('feature_archive_hash',sha(OUT/'validation.npz')==dm['feature_file_sha256']==ex['feature_sha256'])
        check('string_ID_schema_and_unique_pairs',us.dtype.kind in 'US' and items.dtype.kind in 'US' and names==ex['feature_names'] and X.shape==(dm['rows'],77) and len(set(zip(us,items)))==len(us))
        check('finite_features_and_eligible_candidates',np.isfinite(X).all() and set(items)<=eligible)
        check('disjoint_user_grouping_all_population',not set(dev)&set(assessment) and set(dev)|set(assessment)==set(cohort))
        check('assessment_hash_partition',all(hashvalue(u)%5==0 for u in assessment) and all(hashvalue(u)%5!=0 for u in dev))
        check('source_checksums',all(sha(ROOT/p)==h for p,h in dm['source_files'].items()))
        check('historical_feature_boundary',dm['history_provenance']['lo']==1583704800 and dm['history_provenance']['hi']==1589749200)
        a,b,truth=targets_reference(cohort)
        stored=read(OUT/'labels.json');counts=read(OUT/'future_airing_counts.json')
        check('all_targets_counts_independently_reconstructed',truth==stored and a==counts['threshold120'] and b==counts['threshold80'])
        check('binary_and_count_row_alignment',np.array_equal(z['binary_label'],[str(i) in truth['secondary'][str(u)] for u,i in zip(us,items)])
            and np.array_equal(z['future_counts'],[a[str(u)].get(str(i),0) for u,i in zip(us,items)])
            and np.array_equal(z['future_counts80'],[b[str(u)].get(str(i),0) for u,i in zip(us,items)]))
        old=load(ROOT/'results/v5_experiment/validation_features.npz');index={(str(u),str(i)):n for n,(u,i) in enumerate(zip(us,items))}
        idx=np.asarray([index[str(u),str(i)] for u,i in zip(old['users'],old['items'])])
        check('V5_template75_features_reused_exactly',np.array_equal(X[idx,:75],old['X']))
        check('new_signals_all_rows_independent',np.array_equal(X[:,75:],source_reference(z,'train',1589749200,set(cohort))))
        print('Counts, hypothesis labels and both new features independently reconstructed',flush=True)
        folds=np.asarray([hashvalue(str(u),'v8-inner:')%3 for u in us]);devmask=np.isin(us,dev);assessmask=~devmask
        recomputed={}
        for name,entry in plan['experiments'].items():
            cols=[names.index(c) for c in entry['columns']];fp=ex['comparisons'][name]
            oof={}
            for f in range(3):
                ti=np.flatnonzero(devmask&(folds!=f));vi=np.flatnonzero(devmask&(folds==f));vu=tuple(u for u in dev if hashvalue(u,'v8-inner:')%3==f)
                check(f'{name}_fold{f}_no_user_leakage',not set(us[ti])&set(us[vi]) and not set(us[ti])&set(assessment))
                path=OUT/f'{name}_inner{f}.txt';bst=lgb.Booster(model_file=str(path))
                check(f'{name}_fold{f}_model_integrity',sha(path)==fp['development']['folds'][f]['model_sha256'] and bst.feature_name()==entry['columns'] and bst.current_iteration()==78)
                pred=reference_rank(bst.predict(X[vi][:,cols],num_threads=4),z,vi,vu);oof.update(pred)
            saved=evaluation.parse_prediction_records(list(rows(OUT/f'{name}_development.csv')))
            check(name+'_development_model_replay',oof==saved)
            dscore=measured(oof,truth['secondary'],dev)
            check(name+'_development_MAP',math.isclose(dscore,fp['development']['secondary_MAP'],rel_tol=0,abs_tol=1e-14))
            modelpath=OUT/f'{name}_assessment_model.txt';bst=lgb.Booster(model_file=str(modelpath));ai=np.flatnonzero(assessmask)
            check(name+'_assessment_model_hash',sha(modelpath)==fp['model_sha256'])
            pred=reference_rank(bst.predict(X[ai][:,cols],num_threads=4),z,ai,assessment)
            saved=evaluation.parse_prediction_records(list(rows(OUT/f'{name}_assessment.csv')))
            check(name+'_assessment_model_replay',pred==saved)
            allpred=evaluation.parse_prediction_records(list(rows(OUT/f'{name}_all_crossfit.csv')))
            check(name+'_crossfit_prediction_identity',all(allpred[u]==(oof[u] if u in oof else pred[u]) for u in cohort))
            for population,values in [('assessment',assessment),('all_users_crossfit',cohort)]:
                pp=pred if population=='assessment' else allpred
                template=set(pop['template_users']);sizes=pop['history_sizes']
                subsets={'all':list(values),'template':[u for u in values if u in template],'nontemplate':[u for u in values if u not in template],
                    'history_and_outcome':[u for u in values if sizes[u]>0 and truth['primary'][u]],
                    'heavy101plus':[u for u in values if sizes[u]>=101],'light1to20':[u for u in values if 1<=sizes[u]<=20],
                    'weak1to5':[u for u in values if 1<=sizes[u]<=5],'cold0':[u for u in values if sizes[u]==0],
                    'empty_target':[u for u in values if not truth['primary'][u]]}
                for label,populationusers in subsets.items():
                    storedrow=fp[population][label]
                    if storedrow['users']!=len(populationusers):raise ValueError('Stratum user count differs')
                    for key,tname in [('secondary_MAP','secondary'),('fraction80_MAP','fraction80'),('primary_MAP','primary')]:
                        val=measured(pp,truth[tname],populationusers)
                        if val is None:
                            if storedrow[key] is not None:raise ValueError('Empty population score not null')
                        elif not math.isclose(val,storedrow[key],rel_tol=0,abs_tol=1e-14):raise ValueError(f'{name}/{population}/{label}/{key}')
            check(name+'_all_strata_and_labels_independent_MAP',True)
            prod=evaluation.evaluate_predictions(pred,{u:truth['secondary'][u] for u in assessment},assessment,eligible)['primary_map_at_5']
            oscore=measured(pred,truth['secondary'],assessment)
            check(name+'_production_evaluator_agreement',math.isclose(prod,oscore,rel_tol=0,abs_tol=1e-14))
            recomputed[name]={'development_MAP':dscore,'assessment_MAP':oscore}
        best=max([n for n in plan['experiments'] if n!='E1_binary75'],key=lambda n:(recomputed[n]['development_MAP'],
            ex['comparisons'][n]['development']['primary_MAP'],-ex['comparisons'][n]['feature_count']))
        check('new_model_development_only_selection',best==ex['selected']==selection['selected'])
        check('valid_model_even_when_development_regresses',best!='E1_binary75' and (OUT/'final_model.txt').is_file())
        check('final_model_hash',sha(OUT/'final_model.txt')==ex['final_model_sha256'])
        original=evaluation.parse_prediction_records(list(rows(ROOT/'results/v5_experiment/heldout_v5.csv')))
        expected=read(ROOT/'results/v5_experiment/experiment.json');vu=tuple(expected['outer_users']);om=np.isin(old['users'],vu)
        bm=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/evaluation_model.txt'))
        replay=reference_rank(bm.predict(old['X'][om],num_threads=4),old,np.flatnonzero(om),vu)
        check('historical_V5_exact_predictions_and_MAP',replay==original and math.isclose(measured(replay,truth['secondary'],vu),.4886328125,rel_tol=0,abs_tol=1e-15))
        detail['V5_original_MAP']=measured(replay,truth['secondary'],vu)
        detail['V5_MAP_under_unverified80_hypothesis']=measured(replay,truth['fraction80'],vu)
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=h]
        check('all_V1_V7_and_original_data_and_exports_preserved',not changed,changed)
        if submission.exists():
            sm=read(OUT/'submission_manifest.json');tm=read(OUT/'test_feature_manifest.json');tz=load(OUT/'test.npz')
            template=[r['user_id'] for r in rows(ROOT.parent/'sweettv-tv-program-recommender/submission.csv')]
            check('test_feature_integrity_schema',sha(OUT/'test.npz')==tm['test_npz_sha256'] and tz['columns'].tolist()==names and np.isfinite(tz['X']).all())
            check('test_new_features_independent_reconstruction',np.array_equal(tz['X'][:,75:],source_reference(tz,'history',1595797200,set(template))))
            originaltest=load(ROOT/'results/v5_experiment/test_features.npz')
            check('final75_schema_and_V5_context_unchanged',np.array_equal(tz['X'][:,:75],originaltest['X']) and np.array_equal(tz['users'],originaltest['users']) and np.array_equal(tz['items'],originaltest['items']))
            with submission.open(newline='',encoding='utf-8') as f:
                rr=csv.reader(f);check('exact_submission_header',next(rr)==['user_id','tv_show_id']);rlist=list(rr)
            check('exact1260_template_users_and_order',len(rlist)==1260 and len({r[0] for r in rlist})==1260 and [r[0] for r in rlist]==template)
            pool={str(int(r['tv_show_id'])) for r in rows(ROOT.parent/'sweettv-tv-program-recommender/export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
            pred={}
            for row in rlist:
                if len(row)!=2:raise ValueError('Invalid CSV row width')
                p=row[1].split(' ')
                if len(p)!=5 or len(set(p))!=5 or not set(p)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in p):raise ValueError('Invalid CSV recommendations')
                pred[row[0]]=tuple(p)
            check('five_distinct_canonical_positive_scheduled_ids',True)
            model=lgb.Booster(model_file=str(OUT/'final_model.txt'));cols=ex['selected_columns'];ss=model.predict(tz['X'][:,cols],num_threads=4)
            finalpred=reference_rank(ss,tz,np.arange(len(ss)),template)
            check('all1260_rankings_final_model_replay',finalpred==pred)
            check('exact_selected_feature_order',model.feature_name()==[names[i] for i in cols]==sm['selected_feature_names'])
            check('submission_model_feature_manifest_hashes',sha(submission)==sm['submission_sha256'] and sha(OUT/'final_model.txt')==sm['model_sha256'] and sha(OUT/'experiment.json')==sm['experiment_sha256'])
            detail['submission_path']=str(submission.relative_to(ROOT));detail['submission_sha256']=sha(submission)
            checksum=OUT/'submission_sha256.txt'
            if checksum.exists() and checksum.read_text().strip()!=sha(submission):raise ValueError('Existing checksum differs')
            if not checksum.exists():checksum.write_text(sha(submission)+'\n')
        detail.update(selected=best,independent_scores=recomputed,population_users=len(cohort),development_users=len(dev),assessment_users=len(assessment))
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':f'{type(error).__name__}: {error}'})
        detail['traceback']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),
        'checks':checks,'details':detail,'seconds':time.monotonic()-start,
        'scope':'Independent all-user held-out count/label reconstruction and all new signal values; every inner/assessment model replayed, all subgroup MAPs rationally checked; original75 feature generators shared/reused, not independently rebuilt.'}
    resultpath.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_passed':result['all_passed'],'passed_checks':result['passed_checks'],'failed_checks':result['failed_checks'],'report':str(resultpath.relative_to(ROOT)),
        'seconds':result['seconds'],'details':detail},indent=2))
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
