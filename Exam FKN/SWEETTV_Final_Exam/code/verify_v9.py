"""Independent V9 transformation, replay, MAP, analysis and optional CSV verifier."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import csv,json,hashlib,math,time,traceback
import numpy as np
import lightgbm as lgb
import evaluation,validation

ROOT=validation.ROOT;OUT=ROOT/'results/v9_experiment';V8=ROOT/'results/v8_experiment'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def rows(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def ap(labels,pred):
    labels=set(labels);pred=list(pred[:5]);value=Fraction(0)
    for i in labels:
        if i in pred:
            r=pred.index(i)+1;value+=Fraction(len(set(pred[:r])&labels),r)
    return float(value/min(5,len(labels))) if labels else 0.


def map5(pred,truth,users):return math.fsum(ap(truth[u],pred[u]) for u in users)/len(users)


def reference_matrix(z,ex,denominator):
    columns=z['columns'].tolist();p=columns.index('ppr_score');s=columns.index('pop_distinct_users')
    values=z['X'][:,p].astype(float)/np.sqrt((z['X'][:,s].astype(float)+20.)/denominator)
    return np.column_stack([z['X'][:,ex['base_V8_indices']],values.astype(np.float32)])


def reference_rank(scores,z,ix,cohort):
    columns=z['columns'].tolist();pop=columns.index('popularity_rank');cold=columns.index('user_cold_start');g=defaultdict(list)
    for n,s in zip(ix,scores):g[str(z['users'][n])].append((int(n),float(s)))
    result={}
    for u in cohort:
        if not g[u]:raise ValueError('Missing candidate user')
        coldflag=z['X'][g[u][0][0],cold]>.5
        ls=sorted(g[u],key=lambda p:((0 if coldflag else -p[1]),float(z['X'][p[0],pop]),int(z['items'][p[0]])))
        result[u]=tuple(str(z['items'][n]) for n,s in ls)
    return result


def reference_blend(new,base,w,cohort):
    out={}
    for u in cohort:
        a={i:j for j,i in enumerate(new[u])};b={i:j for j,i in enumerate(base[u])}
        if a.keys()!=b.keys():raise ValueError('Fusion candidate mismatch')
        values={i:w/(60+j)+(1-w)/(60+b[i]) for i,j in a.items()}
        out[u]=tuple(sorted(a,key=lambda i:(-values[i],a[i],int(i)))[:5])
    return out


def main():
    t=time.monotonic();checks=[];detail={};submission=ROOT/'submission/final_submission_v9_experimental.csv'
    path=OUT/('submission_verification.json' if submission.exists() else 'validation_verification.json')
    if path.exists():path=OUT/f'verification_recheck_{time.time_ns()}.json'
    def check(name,ok,info=None):
        checks.append({'name':name,'passed':bool(ok),'detail':info})
        if not ok:raise ValueError(name)
    try:
        ex=read(OUT/'experiment.json');plan=read(OUT/'plan.json');selection=read(OUT/'selection.json');v8=read(V8/'experiment.json')
        analysis=read(OUT/'analysis_complete.json');labels=read(V8/'labels.json');pop=read(V8/'population.json')
        z=load(V8/'validation.npz');names=z['columns'].tolist();allusers=tuple(pop['users']);dev=tuple(ex['development_users']);assessment=tuple(ex['assessment_users'])
        check('V8_submission_original_identity',sha(ROOT/'submission/final_submission_v8_experimental.csv')==read(V8/'submission_manifest.json')['submission_sha256'])
        check('analysis_precedes_model_plan',sha(OUT/'analysis_complete.json')==plan['analysis_sha256'] and sha(OUT/'ranking_analysis.json')==analysis['ranking_analysis_sha256'])
        check('same_feature_cache_and_labels',sha(V8/'validation.npz')==analysis['validation_source_sha256']==ex['feature_source_sha256'])
        check('same_user_partition',list(dev)==v8['development_users'] and list(assessment)==v8['assessment_users'] and not set(dev)&set(assessment))
        check('same_model_settings_78rounds_fourthreads',ex['parameters']==v8['selected_params'] and ex['rounds']==78 and ex['parameters']['num_threads']<=4)
        check('new_not_relabelled_baseline',ex['new_feature']=='ppr_support_adjusted_score' and len(ex['feature_names'])==77 and ex['feature_names'][:-1]==read(V8/'submission_manifest.json')['selected_feature_names'])
        check('history_context_valid',read(V8/'data_manifest.json')['history_provenance']['hi']==1589749200 and read(V8/'test_feature_manifest.json')['new_signal_provenance']['window'][1]==1595797200)
        M=reference_matrix(z,ex,pop['historical_users']);check('calibrated_feature_is_finite_nonnegative',np.isfinite(M).all() and np.all(M[:,-1]>=0))
        us=z['users'];it=z['items'];dm=np.isin(us,dev);am=~dm
        folds=np.asarray([int(hashlib.sha256(('v8-inner:'+str(u)).encode()).hexdigest()[:8],16)%3 for u in us])
        bs=np.load(OUT/'V8_crossfit_scores.npy',allow_pickle=False)
        check('baseline_score_cache_integrity',sha(OUT/'V8_crossfit_scores.npy')==analysis['baseline_scores_sha256'])
        oof={};baseline={};score=np.zeros(len(us))
        for f in range(3):
            ti=np.flatnonzero(dm&(folds!=f));vi=np.flatnonzero(dm&(folds==f));vu=tuple(u for u in dev if int(hashlib.sha256(('v8-inner:'+u).encode()).hexdigest()[:8],16)%3==f)
            check(f'fold{f}_no_user_contamination',not set(us[ti])&set(us[vi]) and not set(us[ti])&set(assessment))
            model_path=OUT/f'E2_inner{f}.txt';m=lgb.Booster(model_file=str(model_path))
            check(f'fold{f}_model_hash_schema',sha(model_path)==ex['inner_models'][f]['model_sha256'] and m.feature_name()==ex['feature_names'] and m.current_iteration()==78)
            score[vi]=m.predict(M[vi],num_threads=4);oof.update(reference_rank(score[vi],z,vi,vu))
            original=lgb.Booster(model_file=str(V8/f'E2_streak_inner{f}.txt'));origscore=original.predict(z['X'][vi][:,v8['selected_columns']],num_threads=4)
            check(f'fold{f}_V8_score_replay',np.array_equal(origscore,bs[vi]))
            baseline.update(reference_rank(origscore,z,vi,vu))
        pred={u:oof[u][:5] for u in dev};saved=evaluation.parse_prediction_records(rows(OUT/'E2_development.csv'))
        check('all_development_new_predictions_replay',pred==saved)
        basepred={u:baseline[u][:5] for u in dev};old=evaluation.parse_prediction_records(rows(V8/'E2_streak_development.csv'))
        check('all_development_V8_predictions_replay',basepred==old)
        computed=[]
        for v in ex['variants']:
            p=reference_blend(oof,baseline,v['new_weight'],dev);val=map5(p,labels['secondary'],dev)
            check(v['id']+'_development_MAP_independent',math.isclose(val,v['development_MAP'],rel_tol=0,abs_tol=1e-14))
            computed.append({'id':v['id'],'new_weight':v['new_weight'],'development_MAP':val})
        best=max(computed,key=lambda r:(r['development_MAP'],r['new_weight']==1.))
        check('development_only_selection_recomputed',best['id']==ex['selected']['id']==selection['selected']['id'] and best['new_weight']==ex['selected']['new_weight'])
        changed=sum(oof[u][:5]!=baseline[u][:5] for u in dev)/len(dev)
        check('fusion_difference_gate',math.isclose(changed,selection['changed_development_Top5_fraction'],abs_tol=1e-15) and ((len(computed)>1)==(changed>.1)))
        model=lgb.Booster(model_file=str(OUT/'assessment_model.txt'));ai=np.flatnonzero(am)
        check('assessment_model_integrity',sha(OUT/'assessment_model.txt')==ex['assessment_model_sha256'] and model.feature_name()==ex['feature_names'])
        sr=model.predict(M[ai],num_threads=4);ranked=reference_rank(sr,z,ai,assessment);br=reference_rank(bs[ai],z,ai,assessment)
        p=reference_blend(ranked,br,best['new_weight'],assessment);saved=evaluation.parse_prediction_records(rows(OUT/'selected_assessment.csv'))
        check('selected_assessment_prediction_replay',p==saved)
        for name,pp in [('E1_V8',{u:br[u][:5] for u in assessment}),('E2_pure',{u:ranked[u][:5] for u in assessment}),('selected',p)]:
            val=map5(pp,labels['secondary'],assessment)
            check(name+'_assessment_MAP_independent',math.isclose(val,ex['comparison'][name]['assessment_MAP'],rel_tol=0,abs_tol=1e-14))
            production=evaluation.evaluate_predictions(pp,{u:labels['secondary'][u] for u in assessment},assessment,validation.EvaluationProtocol().candidates)['primary_map_at_5']
            check(name+'_production_metric_parity',math.isclose(val,production,rel_tol=0,abs_tol=1e-14))
        # Independently check the ranking-error totals from the original full scores.
        ri=reference_rank(bs,z,np.arange(len(bs)),allusers);missing=reachable=hits5=hits10=hits20=den=0
        for u in allusers:
            truth=set(labels['secondary'][u]);den+=len(truth);r=ri[u]
            reachable+=len(set(r)&truth);missing+=len(truth-set(r));hits5+=len(set(r[:5])&truth);hits10+=len(set(r[:10])&truth);hits20+=len(set(r[:20])&truth)
        expected=read(OUT/'ranking_analysis.json')['groups']['all']
        check('ranking_error_recall_independent',math.isclose(reachable/den,expected['candidate_recall'],abs_tol=1e-14) and math.isclose(hits5/den,expected['Recall5'],abs_tol=1e-14)
            and math.isclose(hits10/den,expected['Recall10'],abs_tol=1e-14) and math.isclose(hits20/den,expected['Recall20'],abs_tol=1e-14)
            and math.isclose((reachable-hits5)/reachable,expected['reachable_true_below5_fraction'],abs_tol=1e-14))
        check('final_new_model_integrity',sha(OUT/'final_model.txt')==ex['final_model_sha256'])
        protected=read(OUT/'preservation_before.json');changed_files=[s for s,h in protected.items() if sha(ROOT/s if not Path(s).is_absolute() else Path(s))!=h]
        check('all_previous_results_submissions_data_and_exports_unchanged',not changed_files,changed_files)
        if submission.exists():
            sm=read(OUT/'submission_manifest.json');tm=read(OUT/'test_feature_manifest.json');test=load(OUT/'test_features.npz');original=load(V8/'test.npz')
            n=read(ROOT/'data/step2/threshold_120s/interaction_manifest.json')['periods']['history']['unique_users']
            check('test_exact_transformation_and_historical_denominator',tm['historical_users_denominator']==n and np.allclose(test['X'],reference_matrix(original,ex,n),rtol=0,atol=1e-7))
            check('test_source_identity_and_schema',np.array_equal(test['users'],original['users']) and np.array_equal(test['items'],original['items']) and test['columns'].tolist()==ex['feature_names'])
            check('test_feature_manifest_integrity',sha(OUT/'test_features.npz')==tm['feature_file_sha256'] and sha(OUT/'test_feature_manifest.json')==sm['feature_manifest_sha256'])
            template=[r['user_id'] for r in rows(ROOT.parent/'sweettv-tv-program-recommender/submission.csv')]
            with submission.open(newline='',encoding='utf-8') as f:
                reader=csv.reader(f);check('exact_header',next(reader)==['user_id','tv_show_id']);data=list(reader)
            check('exact1260_template_IDs_order_no_duplicates',len(data)==1260 and len(set(r[0] for r in data))==1260 and [r[0] for r in data]==template)
            pool={str(int(r['tv_show_id'])) for r in rows(ROOT.parent/'sweettv-tv-program-recommender/export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
            out={}
            for row in data:
                if len(row)!=2:raise ValueError('Unexpected index column/missing field')
                ids=row[1].split(' ')
                if len(ids)!=5 or len(set(ids))!=5 or not set(ids)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in ids):raise ValueError('Invalid Kaggle Top5')
                out[row[0]]=tuple(ids)
            check('five_distinct_canonical_positive_eligible_ids',True)
            model=lgb.Booster(model_file=str(OUT/'final_model.txt'));scores=model.predict(test['X'],num_threads=4)
            r=reference_rank(scores,original,np.arange(len(scores)),template)
            b=reference_rank(np.load(OUT/'V8_test_scores.npy',allow_pickle=False),original,np.arange(len(scores)),template)
            check('all1260_final_rankings_reproduced',reference_blend(r,b,best['new_weight'],template)==out)
            # Repeat inference deterministically, not another training run.
            check('deterministic_inference_repeat',np.array_equal(scores,model.predict(test['X'],num_threads=4)))
            check('final_schema_model_and_csv_hashes',model.feature_name()==ex['feature_names']==sm['feature_names'] and sha(submission)==sm['submission_sha256'] and sha(OUT/'final_model.txt')==sm['model_sha256'])
            checksum=OUT/'submission_sha256.txt'
            if checksum.exists() and checksum.read_text().strip()!=sha(submission):raise ValueError('Checksum file differs')
            if not checksum.exists():checksum.write_text(sha(submission)+'\n')
            detail['submission_sha256']=sha(submission);detail['submission_path']=str(submission.relative_to(ROOT))
        detail.update(selected=best,comparison=ex['comparison'],ranking_recall={'candidate':reachable/den,'at5':hits5/den,'at10':hits10/den,'at20':hits20/den})
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':f'{type(error).__name__}: {error}'})
        detail['traceback']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),
        'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':detail,'seconds':time.monotonic()-t,
        'scope':'All new transformations, saved OOF/assessment rankings and rational MAP replayed independently; baseline full-ranking recall checked; original V8 generators/features shared/reused. No hidden labels available.'}
    path=OUT/('submission_verification.json' if submission.exists() else 'validation_verification.json')
    if path.exists():path=OUT/f'verification_recheck_{time.time_ns()}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
