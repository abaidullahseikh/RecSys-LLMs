"""Independent historical classification, score calibration, model replay and CSV audit."""
from collections import defaultdict,Counter
from fractions import Fraction
from pathlib import Path
import csv,json,time,math,traceback,hashlib,gzip
import numpy as np
import lightgbm as lgb
import evaluation,validation
from v8_common import SOURCE

ROOT=validation.ROOT;OUT=ROOT/'results/v10_experiment';V8=ROOT/'results/v8_experiment';V9=ROOT/'results/v9_experiment'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',newline='',encoding='utf-8') as f:yield from csv.DictReader(f)


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def actual_flags(z,period):
    bounds=validation.WINDOWS['train' if period=='train' else 'history'];pairs=set();sizes=Counter();air=Counter()
    p=validation.STEP2/'threshold_120s'/f'{period}_interactions.csv.gz'
    meta=read(validation.STEP2/'threshold_120s/interaction_manifest.json')
    if sha(p)!=meta['artifacts'][p.name]['sha256']:raise ValueError('Historical source checksum differs')
    for r in rows(p):
        f,l=int(r['first_watch_utc']),int(r['last_watch_utc'])
        if not bounds[0]<=f<l<=bounds[1] or int(r['watched_seconds'])<120*int(r['qualifying_airings']):raise ValueError('Historical leakage or threshold inconsistency')
        pairs.add((r['user_id'],r['tv_show_id']));sizes[r['user_id']]+=1;air[r['user_id']]+=int(r['qualifying_airings'])
    seen=np.asarray([(str(u),str(i)) in pairs for u,i in zip(z['users'],z['items'])]);size=np.asarray([sizes[str(u)] for u in z['users']],dtype=np.int32)
    return {'seen':seen,'sizes':size,'cold':size==0,'light':(size>=1)&(size<=20)},pairs,sizes,air


def ap(truth,pred):
    labels=set(truth);pred=list(pred[:5]);result=Fraction(0)
    for i in labels:
        if i in pred:
            k=pred.index(i)+1;result+=Fraction(len(set(pred[:k])&labels),k)
    return float(result/min(5,len(labels))) if labels else 0.


def map5(pred,truth,users):return math.fsum(ap(truth[u],pred[u]) for u in users)/len(users)


def rank_ref(scores,z,users,flags,bonus=0.,policy='global',full=False):
    pi=z['columns'].tolist().index('popularity_rank');ix=defaultdict(list)
    for n,u in enumerate(z['users']):ix[str(u)].append(n)
    pred={}
    for u in users:
        indices=ix[u];cold=flags['cold'][indices[0]];enabled=policy=='global' or (policy=='light1to20' and flags['light'][indices[0]])
        def key(n):
            if cold:value=0.
            else:value=float(scores[n])+(float(bonus) if enabled and not flags['seen'][n] else 0.)
            return (-value,float(z['X'][n,pi]),int(z['items'][n]))
        order=sorted(indices,key=key)
        pred[u]=tuple(str(z['items'][n]) for n in (order if full else order[:5]))
    return pred


def metric_ref(pred,truth,users,z,flags):
    seenpairs={(str(u),str(i)) for u,i,s in zip(z['users'],z['items'],flags['seen']) if s};stats=Counter();den=sum(len(truth[u]) for u in users)
    for u in users:
        for i in pred[u][:5]:
            name='seen' if (u,i) in seenpairs else 'unseen';stats[name+'_slots']+=1;stats[name+'_hits']+=i in truth[u]
    return {'MAP5':map5(pred,truth,users),'Recall5':(stats['seen_hits']+stats['unseen_hits'])/den if den else None,
        'seen_precision':stats['seen_hits']/stats['seen_slots'] if stats['seen_slots'] else None,
        'unseen_precision':stats['unseen_hits']/stats['unseen_slots'] if stats['unseen_slots'] else None}


def main():
    t=time.monotonic();checks=[];detail={};submission=ROOT/'submission/final_submission_v10_experimental.csv'
    resultpath=OUT/('submission_verification.json' if submission.exists() else 'validation_verification.json')
    if resultpath.exists():resultpath=OUT/f'verification_recheck_{time.time_ns()}.json'
    def check(name,ok,info=None):
        checks.append({'name':name,'passed':bool(ok),'detail':info})
        if not ok:raise ValueError(name)
    try:
        ex=read(OUT/'experiment.json');plan=read(OUT/'plan.json');selection=read(OUT/'selection.json');e9=read(V9/'experiment.json');e8=read(V8/'experiment.json')
        z=load(V8/'validation.npz');flags,pairs,sizes,air=actual_flags(z,'train');savedflags=load(OUT/'validation_flags.npz')
        check('all_seen_unseen_user_group_flags_independent',all(np.array_equal(v,savedflags[k]) for k,v in flags.items()))
        check('schema_and_original_feature_hash',z['columns'].tolist()==e8['feature_names'] and sha(V8/'validation.npz')==e9['feature_source_sha256'])
        labels=read(V8/'labels.json');pop=read(V8/'population.json');cohort=tuple(pop['users']);truth=labels['secondary']
        dev=tuple(ex['development_users']);assessment=tuple(ex['assessment_users']);dm=np.isin(z['users'],dev);am=~dm
        check('same_disjoint_user_partition',list(dev)==e9['development_users'] and list(assessment)==e9['assessment_users'] and not set(dev)&set(assessment))
        names=z['columns'].tolist();p=names.index('ppr_score');s=names.index('pop_distinct_users');n=pop['historical_users']
        M=np.column_stack([z['X'][:,e9['base_V8_indices']],(z['X'][:,p].astype(float)/np.sqrt((z['X'][:,s].astype(float)+20.)/n)).astype(np.float32)])
        scores9=np.zeros(len(z['users']));scores8=np.zeros_like(scores9)
        for f in range(3):
            fold=np.asarray([int(hashlib.sha256(('v8-inner:'+str(u)).encode()).hexdigest()[:8],16)%3 for u in z['users']]);vi=np.flatnonzero(dm&(fold==f));ti=np.flatnonzero(dm&(fold!=f))
            check(f'fold{f}_supervised_user_separation',not set(z['users'][ti])&set(z['users'][vi]) and not set(z['users'][ti])&set(assessment))
            for version,path,X in [('V9',V9/f'E2_inner{f}.txt',M),('V8',V8/f'E2_streak_inner{f}.txt',z['X'][:,e8['selected_columns']])]:
                bst=lgb.Booster(model_file=str(path));score=bst.predict(X[vi],num_threads=4)
                if version=='V9':scores9[vi]=score
                else:scores8[vi]=score
        ai=np.flatnonzero(am)
        scores9[ai]=lgb.Booster(model_file=str(V9/'assessment_model.txt')).predict(M[ai],num_threads=4)
        scores8[ai]=lgb.Booster(model_file=str(V8/'E2_streak_assessment_model.txt')).predict(z['X'][ai][:,e8['selected_columns']],num_threads=4)
        check('V9_saved_scores_model_replay',np.array_equal(scores9,np.load(OUT/'V9_crossfit_scores.npy',allow_pickle=False)))
        check('V8_saved_scores_model_replay',np.array_equal(scores8,np.load(V9/'V8_crossfit_scores.npy',allow_pickle=False)))
        r9=rank_ref(scores9,z,cohort,flags,full=True);r8=rank_ref(scores8,z,cohort,flags)
        top9={u:r[:5] for u,r in r9.items()}
        check('V8_and_V9_exact_crossfit_predictions',r8==evaluation.parse_prediction_records(list(rows(V8/'E2_streak_all_crossfit.csv')))
            and top9==evaluation.parse_prediction_records(list(rows(V9/'selected_all_crossfit.csv'))))
        for name,pred in [('V8',r8),('V9',top9)]:
            for stage,users in [('development',dev),('assessment',assessment)]:
                computed=map5(pred,truth,users)
                check(name+'_'+stage+'_independent_MAP',math.isclose(computed,ex['baseline'][name][stage]['MAP5'],rel_tol=0,abs_tol=1e-14))
        curve=list(rows(OUT/'development_calibration_grid.csv'));best={}
        for policy in ('global','light1to20'):
            local=[]
            for beta in plan['global_grid']:
                pred=rank_ref(scores9,z,dev,flags,beta,policy);m=metric_ref(pred,truth,dev,z,flags)
                saved=next(r for r in curve if r['policy']==policy and float(r['unseen_bonus'])==beta)
                for k,v in m.items():
                    if v is None:
                        if saved[k]!='':raise ValueError('Null precision differs')
                    elif not math.isclose(v,float(saved[k]),rel_tol=0,abs_tol=1e-14):raise ValueError(f'Calibration grid mismatch {policy}/{beta}/{k}')
                local.append({'beta':beta,'MAP':m['MAP5']})
            chosen=max(local,key=lambda r:(r['MAP'],r['beta']==0.,-abs(r['beta'])))
            check(policy+'_development_selection',chosen['beta']==selection['family_best'][policy]['unseen_bonus'])
            best[policy]=chosen
        check('no_forced_nonzero_adjustment',all(v['beta']==0. or v['MAP']>ex['baseline']['V9']['development']['MAP5']+1e-12 for v in best.values()))
        # Conditional pairwise path must be genuine and only when both families selected zero.
        if ex['strategy']['type']=='pairwise_V9_features':
            check('pairwise_conditional_gate',all(r['beta']==0. for r in best.values()) and ex['pairwise_info'] is not None)
            pscores=np.zeros(len(z['users']))
            for f in range(3):
                vi=np.flatnonzero(dm&(fold==f));modelpath=OUT/f'pairwise_inner{f}.txt';bst=lgb.Booster(model_file=str(modelpath))
                check(f'pairwise_fold{f}_hash_schema_objective',sha(modelpath)==ex['pairwise_info']['folds'][f]['model_sha256'] and bst.feature_name()==e9['feature_names'] and bst.params['objective']=='lambdarank')
                # Query groups remain contiguous after filtering by user.
                us=z['users'][vi];blocks=[str(us[0])]+[str(us[k]) for k in range(1,len(us)) if us[k]!=us[k-1]]
                if len(set(blocks))!=len(blocks):raise ValueError('Interleaved ranking query group')
                pscores[vi]=bst.predict(M[vi],num_threads=4)
            bst=lgb.Booster(model_file=str(OUT/'pairwise_assessment.txt'));pscores[ai]=bst.predict(M[ai],num_threads=4)
            selected=rank_ref(pscores,z,cohort,flags)
            check('genuine_new_selected_pairwise_model',sha(OUT/'final_pairwise.txt')==ex['pairwise_info']['final_model_sha256'])
        else:
            st=ex['strategy'];selected=rank_ref(scores9,z,cohort,flags,st['unseen_bonus'],st['policy'])
        check('selected_validation_prediction_replay',selected==evaluation.parse_prediction_records(list(rows(OUT/'selected_all_crossfit.csv'))))
        for stage,users in [('development',dev),('assessment',assessment)]:
            actual=map5(selected,truth,users)
            check('selected_'+stage+'_MAP_recomputed',math.isclose(actual,ex['selected_'+stage+'_MAP'],rel_tol=0,abs_tol=1e-14))
            prod=evaluation.evaluate_predictions({u:selected[u] for u in users},{u:truth[u] for u in users},users,validation.EvaluationProtocol().candidates)['primary_map_at_5']
            check('selected_'+stage+'_production_evaluator_parity',math.isclose(actual,prod,rel_tol=0,abs_tol=1e-14))
        # Complete missing-label seen/unseen split from actual historical pairs, not a rolling flag.
        audit=read(OUT/'seen_unseen_audit.json');counts={flag:Counter() for flag in ('seen','unseen')};rowmap={(str(u),str(i)):j for j,(u,i) in enumerate(zip(z['users'],z['items']))}
        for key,j in rowmap.items():
            u,i=key;flag='seen' if key in pairs else 'unseen';counts[flag]['candidate_pairs']+=1;counts[flag]['relevant_candidate_pairs']+=i in truth[u]
        for u in cohort:
            pos={i:k+1 for k,i in enumerate(r9[u])}
            for i in truth[u]:
                flag='seen' if (u,i) in pairs else 'unseen';counts[flag]['all_relevant_labels']+=1;k=pos.get(i)
                if k is None:counts[flag]['missing_labels']+=1
                elif k<=5:counts[flag]['true_1to5']+=1
                elif k<=10:counts[flag]['true_6to10']+=1
                elif k<=20:counts[flag]['true_11to20']+=1
                else:counts[flag]['true_below20']+=1
        for flag in counts:
            for k in ('candidate_pairs','relevant_candidate_pairs','true_1to5','true_6to10','true_11to20','true_below20'):
                if counts[flag][k]!=audit['counts'][flag][k]:raise ValueError('Seen/unseen audit mismatch')
            denom=counts[flag]['all_relevant_labels']
            for k in ('true_1to5','true_6to10','true_11to20','missing_labels'):counts[flag][k+'_fraction_all_relevant']=counts[flag][k]/denom if denom else 0.
        check('seen_unseen_analysis_independently_completed',True)
        detail['seen_unseen_all_label_counts']=counts
        protected=read(OUT/'preservation_before.json');changed=[p for p,h in protected.items() if sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=h]
        check('V1_V9_all_results_sources_and_exports_preserved',not changed,changed)
        if submission.exists():
            sm=read(OUT/'submission_manifest.json');tz=load(V9/'test_features.npz');original=load(V8/'test.npz');template=tuple(r['user_id'] for r in rows(SOURCE/'submission.csv'))
            tf,_,_,_=actual_flags(original,'history');final9=lgb.Booster(model_file=str(V9/'final_model.txt'));fs9=final9.predict(tz['X'],num_threads=4)
            check('V9_final_scores_reproduced',np.array_equal(fs9,np.load(V9/'V9_test_scores.npy',allow_pickle=False)))
            check('V9_final_original_CSV_reproduced',rank_ref(fs9,original,template,tf)==evaluation.parse_prediction_records(list(rows(ROOT/'submission/final_submission_v9_experimental.csv'))))
            if ex['strategy']['type']=='pairwise_V9_features':
                model=lgb.Booster(model_file=str(OUT/'final_pairwise.txt'));score=model.predict(tz['X'],num_threads=4)
                check('V10_final_model_feature_schema',model.feature_name()==e9['feature_names'] and model.params['objective']=='lambdarank')
                expected=rank_ref(score,original,template,tf)
                check('deterministic_model_inference',np.array_equal(score,model.predict(tz['X'],num_threads=4)))
            else:
                st=ex['strategy'];expected=rank_ref(fs9,original,template,tf,st['unseen_bonus'],st['policy'])
            with submission.open(newline='',encoding='utf-8') as f:
                rr=csv.reader(f);check('exact_Kaggle_header',next(rr)==['user_id','tv_show_id']);rlist=list(rr)
            check('1260_unique_template_users_exact_order',len(rlist)==1260 and len({r[0] for r in rlist})==1260 and [r[0] for r in rlist]==list(template))
            pool={str(int(r['tv_show_id'])) for r in rows(SOURCE/'export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
            pred={}
            for row in rlist:
                if len(row)!=2:raise ValueError('Bad row width/index column')
                pp=row[1].split(' ')
                if len(pp)!=5 or len(set(pp))!=5 or not set(pp)<=pool or any(not i.isascii() or not i.isdecimal() or i.startswith('0') for i in pp):raise ValueError('Invalid Top5 formatting/eligibility')
                pred[row[0]]=tuple(pp)
            check('five_distinct_positive_canonical_scheduled_IDs',True)
            check('all_final_rankings_replayed',pred==expected)
            check('not_historical_submission_renamed',pred!=evaluation.parse_prediction_records(list(rows(ROOT/'submission/final_submission_v9_experimental.csv'))))
            check('manifest_and_CSV_SHA',sha(submission)==sm['submission_sha256'])
            (OUT/'submission_sha256.txt').write_text(sha(submission)+'\n')
            detail['submission_sha256']=sha(submission);detail['submission_path']=str(submission.relative_to(ROOT))
        detail.update(strategy=ex['strategy'],calibration_selection=best,selected_development_MAP=ex['selected_development_MAP'],selected_assessment_MAP=ex['selected_assessment_MAP'])
    except Exception as error:
        if not checks or checks[-1]['passed']:checks.append({'name':'verification_exception','passed':False,'detail':f'{type(error).__name__}: {error}'})
        detail['traceback']=traceback.format_exc()
    result={'all_passed':all(c['passed'] for c in checks),'passed_checks':sum(c['passed'] for c in checks),'failed_checks':sum(not c['passed'] for c in checks),
        'checks':checks,'details':detail,'seconds':time.monotonic()-t,
        'scope':'Historical seen/group flags and label-rate audit independently reconstructed; all seven grid corrections/two families MAP rechecked; saved models replayed and original feature generators shared, not independently rebuilt.'}
    path=OUT/('submission_verification.json' if submission.exists() else 'validation_verification.json')
    if path.exists():path=OUT/f'verification_recheck_{time.time_ns()}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
