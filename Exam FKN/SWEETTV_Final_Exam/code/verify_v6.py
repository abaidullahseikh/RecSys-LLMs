"""Verify actual V6 artifacts without fitting models or modifying old evidence."""
from collections import Counter,defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import json
import math
import time
import traceback
import lightgbm as lgb
import numpy as np
import evaluation
import validation
from create_v6_test_features import validate_existing

ROOT=validation.ROOT;OUT=ROOT/'results/v6_experiment';ZONE=ZoneInfo('Europe/Kyiv')
RECIPES=('fixes50','expanded100','temporal50','temporal100')


def read(path):return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def reference_ap(labels,ranking):
    labels=set(labels);ranking=list(ranking[:5]);result=Fraction(0)
    for item in labels:
        if item in ranking:
            rank=ranking.index(item)+1;result+=Fraction(len(set(ranking[:rank])&labels),rank)
    return float(result/min(5,len(labels))) if labels else 0.


def metric(pred,truth,users):return math.fsum(reference_ap(truth[u],pred[u]) for u in users)/len(users)


def choose_inner(ex,selection):
    sizes={r['id']:r['k'] for r in selection['recipes']}
    return max(ex['development_oof_results'],key=lambda r:(r['metrics']['secondary'],r['metrics']['primary'],-sizes[r['recipe']]))


def ranked(scores,z,indices,users):
    """Independent full ranking implementation; no training-module helpers."""
    names=z['columns'].tolist();cold=names.index('user_cold_start');pop=names.index('popularity_rank')
    groups=defaultdict(list)
    for n,s in zip(indices,scores):groups[str(z['users'][n])].append((int(n),float(s)))
    result={}
    for u in users:
        group=groups[u]
        if not group:raise ValueError('Missing candidate user')
        is_cold=z['X'][group[0][0],cold]>.5
        group.sort(key=lambda ns:((0 if is_cold else -ns[1]),float(z['X'][ns[0],pop]),int(z['items'][ns[0]])))
        result[u]=tuple(str(z['items'][n]) for n,s in group)
    return result


def blend_reference(new,base,weight,users):
    result={}
    for u in users:
        b={item:rank for rank,item in enumerate(base[u])}
        values=[]
        for rank,item in enumerate(new[u]):
            value=weight/(60+rank)+((1-weight)/(60+b[item]) if item in b else 0.)
            values.append((item,value,rank))
        result[u]=tuple(item for item,value,rank in sorted(values,key=lambda r:(-r[1],r[2],int(r[0])))[:5])
    return result


def prediction_file(path,users,eligible):
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['user_id','tv_show_id']:raise ValueError(f'Bad header: {path}')
        pred=evaluation.parse_prediction_records(list(reader))
    if tuple(pred)!=tuple(users):raise ValueError(f'User population/order mismatch: {path}')
    if any(not set(p)<=eligible for p in pred.values()):raise ValueError(f'Ineligible program: {path}')
    return pred


def scan_chronology(build,z,labels,cohort):
    """Recalculate labels, accepted-history counts and sample feature arithmetic.

    Reads existing overlaps; never refits generators or creates features.
    """
    lo,hi=build['history']['lo'],build['history']['hi'];targetlo,targethi=build['target_window']
    if not lo<hi==targetlo<targethi<=1595797200:raise ValueError('Invalid historical fold chronology')
    if build['metadata_fit_end']!=hi:raise ValueError('Metadata fitted after feature cutoff')
    names=z['columns'].tolist();positive=np.flatnonzero(z['X'][:,names.index('user_program_airings')]>0)
    sample=positive[np.linspace(0,len(positive)-1,12,dtype=int)]
    wanted={(str(z['users'][n]),str(z['items'][n])):int(n) for n in sample}
    refs={key:Counter() for key in wanted};weeksets={key:set() for key in wanted}
    counts={u:Counter() for u in cohort};f80={u:Counter() for u in cohort};accepted=excluded=0
    history_path=validation.STEP2/f"{build['history']['period']}_airing_overlaps.csv.gz"
    if sha(history_path)!=build['history']['source_sha256']:raise ValueError('History overlap checksum mismatch')
    target_path=validation.STEP2/('train_airing_overlaps.csv.gz' if build['name']=='early' else 'validation_airing_overlaps.csv.gz')
    for r in records(history_path):
        first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if not lo<=first<last<=hi:
            excluded+=first<hi and last>lo
            continue
        accepted+=1;key=(r['user_id'],r['tv_show_id'])
        if key not in wanted:continue
        secs=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if not 0<secs<=d:raise ValueError('Bad overlap duration')
        ref=refs[key]
        for cutoff,numerator,denominator in [('completion50_count',1,2),('completion80_count',4,5),('completion95_count',19,20)]:
            ref[cutoff]+=secs*denominator>=d*numerator
        if secs>=120:
            ref['user_program_airings']+=1
            age=((datetime.fromtimestamp(hi,ZONE).date()-datetime.fromtimestamp(first,ZONE).date()).days-1)//7
            weeksets[key].add(age)
            for w in (2,4,8):ref[f'program_airings_recent_{w}w']+=age<w
    if accepted!=build['history']['accepted_airing_rows']:raise ValueError('History accepted-row count mismatch')
    if excluded!=build['history'].get('source_boundary_rows_excluded',0):raise ValueError('History boundary exclusion mismatch')
    for key,n in wanted.items():
        for feature in ('completion50_count','completion80_count','completion95_count','user_program_airings',
                        'program_airings_recent_2w','program_airings_recent_4w','program_airings_recent_8w'):
            if z['X'][n,names.index(feature)]!=refs[key][feature]:raise ValueError(f'Sample feature mismatch {key}/{feature}')
        if z['X'][n,names.index('program_active_weeks')]!=len(weeksets[key]):raise ValueError('Sample calendar-week mismatch')
    for r in records(target_path):
        u=r['user_id'];first,last=int(r['first_overlap_utc']),int(r['last_overlap_utc'])
        if u not in counts or not targetlo<=first<last<=targethi:continue
        i=r['tv_show_id'];secs=int(r['overlap_seconds']);d=int(r['broadcast_end_utc'])-int(r['broadcast_start_utc'])
        if secs>=120:counts[u][i]+=1
        if 5*secs>=4*d:f80[u][i]+=1
    top=lambda values:sorted(values,key=lambda i:(-values[i],int(i)))[:5]
    expected={'secondary':{u:top(v) for u,v in counts.items()},'fraction80':{u:top(v) for u,v in f80.items()},
              'primary':{u:sorted(v,key=int) for u,v in counts.items()}}
    if expected!=labels:raise ValueError('Independently reconstructed historical target mismatch')
    return {'history_window':[lo,hi],'target_window':[targetlo,targethi],'accepted_history_rows':accepted,
            'boundary_rows_excluded':excluded,'feature_pairs_independently_checked':len(wanted),'all_user_labels_reconstructed':True}


def main():
    started=time.monotonic();checks=[];details={};limitations=[]
    # Hash all existing V6 result files, submissions, and exported sessions before this read-only run.
    protected=[p for p in OUT.iterdir() if p.is_file()]+list((ROOT/'submission').glob('*.csv'))+list((ROOT/'evidence').glob('*session*.json'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    output=OUT/'verification.json'
    if output.exists():output=OUT/f'verification_recheck_{time.time_ns()}.json'
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
        if not ok:raise ValueError(name)
    try:
        ex=read(OUT/'experiment.json');selection=read(OUT/'selection.json');v5ex=read(ROOT/'results/v5_experiment/experiment.json')
        details['experiment_keys']=list(ex)
        protocol=validation.EvaluationProtocol();cohort=protocol.cohort;eligible=set(protocol.candidates)
        outer=tuple(ex['outer_users']);dev=tuple(ex['development_users']);outerset=set(outer)
        check('exact_disjoint_user_partition',len(outer)==256 and len(dev)==1004 and not outerset&set(dev) and outerset|set(dev)==set(cohort))
        check('outer_hash_assignment',all(int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5==0 for u in outer))
        check('same_V5_validation_users',list(outer)==v5ex['outer_users'] and list(dev)==v5ex['development_users'])
        check('historical_selection_not_Kaggle_labels',ex['private_scores_used_as_labels'] is False and ex['private_scores_used_for_hyperparameters'] is False)
        # Exactly match the saved training selection key, including smaller-pool tie break.
        best=choose_inner(ex,selection)
        check('inner_OOF_argmax_matches_selection',best==selection['selected']==ex['selected'])
        recipe_by_id={r['id']:r for r in selection['recipes']}
        check('recipe_parameters_match',recipe_by_id[best['recipe']]==ex['selected_recipe'] and selection['params']==ex['parameters'] and selection['rounds']==ex['rounds'])
        details['selected']=best
        limitations.append('Inner fold models and raw OOF predictions were not persisted. Selection is recomputed from all 10 recorded OOF summaries; those summaries cannot be independently rescored without retraining, which this audit does not do.')
        mainz=load_npz(OUT/'main_data.npz');early=load_npz(OUT/'early_data.npz')
        data={'main':mainz,'early':early};labels={};builds={}
        for name,z in data.items():
            build=read(OUT/f'{name}_build.json');builds[name]=build;labels[name]=read(OUT/f'{name}_labels.json')
            check(name+'_archive_recorded_hash',sha(OUT/f'{name}_data.npz')==build['data_sha256'])
            check(name+'_schema',set(z)=={'X','users','items','columns','mask20','mask50','mask100'} and z['X'].shape==(build['rows'],len(ex['feature_columns'])) and z['columns'].tolist()==ex['feature_columns'])
            check(name+'_finite_features_string_ids',np.isfinite(z['X']).all() and z['users'].dtype.kind in 'US' and z['items'].dtype.kind in 'US')
            check(name+'_unique_pairs_and_user_order',len(set(zip(z['users'],z['items'])))==len(z['users']) and tuple(dict.fromkeys(z['users']))==cohort)
            check(name+'_nested_candidate_masks',all(z[k].dtype.kind=='b' for k in ('mask20','mask50','mask100')) and np.all(~z['mask20']|z['mask50']) and np.all(~z['mask50']|z['mask100']))
            details[name+'_chronology']=scan_chronology(build,z,labels[name],cohort)
            check(name+'_chronology_labels_and_sample_features',True,details[name+'_chronology'])
        check('earlier_targets_end_before_main_targets_start',builds['early']['target_window'][1]==builds['main']['history']['hi']==builds['main']['target_window'][0])
        check('main_primary_truth_matches_frozen_protocol',all(set(labels['main']['primary'][u])==protocol.truth[u] for u in cohort))
        secondary={r['user_id']:set(r['top5_validation_program_ids'].split()) for r in records(ROOT/'results/step5/secondary_top5_truth.csv')}
        check('main_secondary_truth_matches_frozen_protocol',all(set(labels['main']['secondary'][u])==secondary[u] for u in cohort))
        print('Chronology and labels verified; replaying saved models',flush=True)
        names=ex['feature_columns'];main_outer=np.isin(mainz['users'],outer)
        main_dev=np.isin(mainz['users'],dev);early_dev=np.isin(early['users'],dev)
        pred_by_recipe={};full_ranks={};metrics={}
        def compare_metrics(name,pred,truth,expected):
            computed={key:metric(pred,t,outer) for key,t in truth.items()}
            check(name+'_independent_rational_MAP',all(math.isclose(v,expected[k],rel_tol=0,abs_tol=1e-14) for k,v in computed.items()),computed)
            prod=evaluation.evaluate_predictions(pred,{u:truth['secondary'][u] for u in outer},outer,eligible)['primary_map_at_5']
            check(name+'_production_MAP_agreement',math.isclose(prod,computed['secondary'],rel_tol=0,abs_tol=1e-14))
            return computed
        for recipe in selection['recipes']:
            name=recipe['id'];k=recipe['k'];ix=np.flatnonzero(main_outer&mainz[f'mask{k}'])
            model=lgb.Booster(model_file=str(OUT/f'evaluation_{name}.txt'))
            check(name+'_model_columns_rounds',model.feature_name()==names and model.current_iteration()==ex['rounds'])
            ranked_full=ranked(model.predict(mainz['X'][ix],num_threads=4),mainz,ix,outer)
            saved=prediction_file(OUT/f'heldout_{name}.csv',outer,eligible)
            check(name+'_model_replays_saved_predictions',{u:r[:5] for u,r in ranked_full.items()}==saved)
            pred_by_recipe[name]=saved;full_ranks[name]=ranked_full
            metrics[name]=compare_metrics(name,saved,labels['main'],ex['outer_comparisons'][name])
            mi=np.flatnonzero(main_dev&mainz[f'mask{k}']);ei=np.flatnonzero(early_dev&early[f'mask{k}']) if recipe['early_weight'] else []
            positives=sum(str(mainz['items'][n]) in labels['main']['secondary'][str(mainz['users'][n])] for n in mi)
            positives+=sum(str(early['items'][n]) in labels['early']['secondary'][str(early['users'][n])] for n in ei)
            meta=ex['fit_metadata'][name]
            check(name+'_supervised_fit_rows_and_labels',meta['rows']==len(mi)+len(ei) and meta['positive_rows']==positives and meta['users']==len(dev) and meta['early_weight']==recipe['early_weight'])
        v5z=load_npz(ROOT/'results/v5_experiment/validation_features.npz');v5outer=np.isin(v5z['users'],outer);vi=np.flatnonzero(v5outer)
        v5model=lgb.Booster(model_file=str(ROOT/'results/v5_experiment/evaluation_model.txt'))
        check('V5_evaluation_model_hash',sha(ROOT/'results/v5_experiment/evaluation_model.txt')==v5ex['evaluation_model_sha256'])
        v5rank=ranked(v5model.predict(v5z['X'][vi],num_threads=4),v5z,vi,outer)
        v5pred=prediction_file(ROOT/'results/v5_experiment/heldout_v5.csv',outer,eligible)
        check('V5_validation_predictions_unchanged',{u:r[:5] for u,r in v5rank.items()}==v5pred)
        metrics['reproduced_V5']=compare_metrics('V5',v5pred,labels['main'],ex['outer_comparisons']['reproduced_V5'])
        selected=prediction_file(OUT/'heldout_selected.csv',outer,eligible)
        check('selected_blend_replays_with_V5_not_V2',blend_reference(full_ranks[best['recipe']],v5rank,best['weight'],outer)==selected)
        metrics['selected']=compare_metrics('selected',selected,labels['main'],ex['outer_comparisons']['selected'])
        early_pool=set(early['items'])|set(i for t in labels['early'].values() for u in t for i in t[u])
        earlypred=prediction_file(OUT/'earlier_fold_heldout.csv',outer,early_pool)
        computed={k:metric(earlypred,t,outer) for k,t in labels['early'].items()}
        check('earlier_fold_saved_predictions_metrics',all(math.isclose(v,ex['earlier_fold']['outer_metrics'][k],rel_tol=0,abs_tol=1e-14) for k,v in computed.items()),computed)
        limitations.append('Earlier-fold-only model was not saved. Its prediction metrics and independently reconstructed targets are checked; its model replay is unavailable.')
        candidate_sets={}
        for k in (20,50,100):
            grouped=defaultdict(set)
            for u,i in zip(mainz['users'][mainz[f'mask{k}']],mainz['items'][mainz[f'mask{k}']]):grouped[str(u)].add(str(i))
            check(f'candidate{k}_eligibility_and_minimum5',all(len(grouped[u])>=5 and grouped[u]<=eligible for u in cohort))
            candidate_sets[k]=grouped
        recalls=[]
        for row in ex['candidate_recall']:
            population=outer if row['population']=='outer' else dev;k=row['k'];groups=candidate_sets[k]
            value=sum(len(groups[u]&secondary[u]) for u in population)/sum(len(secondary[u]) for u in population)
            avg=sum(len(groups[u]) for u in population)/len(population)
            check(f'recall{k}_{row["population"]}',math.isclose(value,row['recall'],rel_tol=0,abs_tol=1e-14) and math.isclose(avg,row['avg_candidates'],abs_tol=1e-12) and row['users']==len(population))
            recalls.append({'k':k,'population':row['population'],'recall':value,'avg_candidates':avg})
        k20=sum(len(candidate_sets[20][u]&secondary[u]) for u in outer)/sum(len(secondary[u]) for u in outer)
        details['top20_outer_recall_recomputed_not_a_recorded_selection_result']=k20
        check('final_model_recorded_hash',sha(OUT/'final_model.txt')==ex['final_model_sha256'])
        check('saved_test_cache_validation',validate_existing()['all_passed'])
        same_test=sha(OUT/'test_features.npz')==sha(ROOT/'results/v5_experiment/test_features.npz')
        check('test_cache_is_exact_V5_copy_as_source_predicts',same_test)
        limitations.append('test_features.npz is byte-identical to V5 test features. Its existing manifest is verified but this is not a completed v6_data test build or validated V6 inference configuration.')
        # Check all V5-era submissions and exports. One new audit script was
        # accidentally included in that historical glob: report it explicitly.
        historical=read(OUT/'preservation_before.json');expected_change={'code/audit_v5_source.py'}
        modifications=[p for p,h in historical.items() if sha(ROOT/p)!=h]
        check('V1_to_V5_and_original_exports_historical_preservation',not set(modifications)-expected_change,{'historical_snapshot_differences':modifications,'known_difference':'New V6 audit script fixed after its snapshot; no V5 code or artifact is exempted.'})
        check('all_existing_result_files_unchanged_during_verification',all(sha(ROOT/p)==h for p,h in before.items()))
        check('selected_outer_score_does_not_beat_V5',metrics['selected']['secondary']<metrics['reproduced_V5']['secondary'])
        check('no_V6_submission_created',not (ROOT/'submission/final_submission_v6.csv').exists())
        details.update({'outer_metrics':metrics,'candidate_recall':recalls,'V5_submission_sha256':sha(ROOT/'submission/final_submission_v5.csv'),
            'selection_explanation':{'Top100_pure_inner':next(r['metrics']['secondary'] for r in ex['development_oof_results'] if r['recipe']=='expanded100' and r['weight']==1.),
                'selected_inner':best['metrics']['secondary'],'Top100_outer':metrics['expanded100']['secondary'],'selected_outer':metrics['selected']['secondary'],
                'baseline_outer':metrics['reproduced_V5']['secondary']}})
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'uncaught_verification_error','passed':False,'detail':f'{type(error).__name__}: {error}'})
        details['exception']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),
        'failed_checks':sum(not c['passed'] for c in checks),'checks':checks,'details':details,'limitations':limitations,
        'training_restarted':False,'feature_tables_rebuilt':False,'submission_generated':False,
        'recommendation':'Keep V5 as designated final; no new V6 CSV','seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'all_passed':result['all_passed'],'passed_checks':result['passed_checks'],'failed_checks':result['failed_checks'],
        'report':str(output.relative_to(ROOT)),'seconds':result['seconds'],'limitations':limitations},indent=2))
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
