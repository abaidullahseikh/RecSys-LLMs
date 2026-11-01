"""Replay V8, audit full ranking errors and distribution shift before any V9 fit."""
from collections import defaultdict,Counter
from pathlib import Path
import csv,json,time,math,resource
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
import evaluation,validation
from v8_common import ROOT,SOURCE,sha,load,read,records,inner_fold,map5

OUT=ROOT/'results/v9_experiment';V8=ROOT/'results/v8_experiment'


def dump(name,data):
    p=OUT/name
    if p.exists():
        if read(p)!=data:raise ValueError(f'Existing evidence differs: {name}')
        return
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def rank_all(score,z,cohort):
    names=z['columns'].tolist();pop=names.index('popularity_rank');cold=names.index('user_cold_start');groups=defaultdict(list)
    for n,u in enumerate(z['users']):groups[str(u)].append(n)
    indices={};pred={}
    for u in cohort:
        ix=np.asarray(groups[u]);num=np.asarray([int(z['items'][i]) for i in ix],dtype=np.int64)
        s=np.zeros(len(ix)) if z['X'][ix[0],cold]>.5 else score[ix]
        order=np.lexsort((num,z['X'][ix,pop],-s));indices[u]=ix[order]
        pred[u]=tuple(str(z['items'][n]) for n in ix[order[:5]])
    return indices,pred


def moments(x):
    finite=x[np.isfinite(x)]
    return {'min':float(finite.min()),'max':float(finite.max()),'mean':float(finite.mean()),'std':float(finite.std()),
        'missing_rate':float(np.mean(~np.isfinite(x))),'zero_rate':float(np.mean(x==0)),
        'median':float(np.median(finite)),'p95':float(np.quantile(finite,.95))}


def main():
    start=time.monotonic()
    if (OUT/'analysis_complete.json').exists():raise FileExistsError('Preserve completed V9 analysis')
    OUT.mkdir(exist_ok=True)
    prior=[]
    for folder in ('code','results','submission','evidence'):
        prior += [p for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in str(p)
            and 'v9' not in p.name and p.suffix not in ('.sqlite','.pyc')]
    prior += [ROOT/'exam_quiz_FKN.md']+list(SOURCE.glob('*.csv'))
    dump('preservation_before.json',{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in prior})
    ex=read(V8/'experiment.json');sm=read(V8/'submission_manifest.json');fm=read(V8/'test_feature_manifest.json');dm=read(V8/'data_manifest.json')
    if not read(V8/'submission_verification.json')['all_passed']:raise ValueError('V8 prior verification is not passing')
    if sha(ROOT/sm['submission_path'])!=sm['submission_sha256'] or sha(V8/'final_model.txt')!=sm['model_sha256']:raise ValueError('V8 submission/model identity mismatch')
    z=load(V8/'validation.npz');tz=load(V8/'test.npz');names=z['columns'].tolist();cols=ex['selected_columns'];selected=ex['selected']
    if selected!='E2_streak' or names!=ex['feature_names'] or tz['columns'].tolist()!=names:raise ValueError('Unexpected V8 schema')
    if sha(V8/'validation.npz')!=dm['feature_file_sha256'] or sha(V8/'test.npz')!=fm['test_npz_sha256']:raise ValueError('V8 feature cache identity mismatch')
    pop=read(V8/'population.json');labels=read(V8/'labels.json');cohort=tuple(pop['users']);dev=tuple(ex['development_users']);assessment=tuple(ex['assessment_users'])
    us=z['users'];items=z['items'];X=z['X'];dmask=np.isin(us,dev);amask=~dmask;fold=np.asarray([inner_fold(str(u)) for u in us]);score=np.zeros(len(us))
    model_paths=[]
    for f in range(3):
        path=V8/f'{selected}_inner{f}.txt';model=lgb.Booster(model_file=str(path));ix=np.flatnonzero(dmask&(fold==f))
        if model.feature_name()!=sm['selected_feature_names']:raise ValueError('Inner feature schema mismatch')
        score[ix]=model.predict(X[ix][:,cols],num_threads=4);model_paths.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    path=V8/f'{selected}_assessment_model.txt';model=lgb.Booster(model_file=str(path));ix=np.flatnonzero(amask)
    score[ix]=model.predict(X[ix][:,cols],num_threads=4);model_paths.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    ranks,pred=rank_all(score,z,cohort)
    saved=evaluation.parse_prediction_records(list(records(V8/f'{selected}_all_crossfit.csv')))
    if pred!=saved:raise ValueError('V8 grouped validation predictions do not reproduce')
    devmetric=map5(pred,labels['secondary'],dev)
    if not math.isclose(devmetric,sm['selected_development_metrics']['secondary_MAP'],rel_tol=0,abs_tol=1e-15):raise ValueError('V8 development MAP mismatch')
    template=tuple(r['user_id'] for r in records(SOURCE/'submission.csv'))
    final=lgb.Booster(model_file=str(V8/'final_model.txt'));ts=final.predict(tz['X'][:,cols],num_threads=4)
    _,finalpred=rank_all(ts,tz,template)
    if finalpred!=evaluation.parse_prediction_records(list(records(ROOT/sm['submission_path']))):raise ValueError('V8 final submission does not replay')
    dump('V8_reproduction.json',{'exact_development_predictions':True,'exact_all_crossfit_predictions':True,'exact_submission_predictions':True,
        'development_MAP':devmetric,'assessment_MAP':map5(pred,labels['secondary'],assessment),'model_paths':model_paths,
        'final_model_path':str((V8/'final_model.txt').relative_to(ROOT)),'final_model_sha256':sha(V8/'final_model.txt'),
        'submission_path':sm['submission_path'],'submission_sha256':sm['submission_sha256'],'selected_feature_names':sm['selected_feature_names'],
        'candidate_method':'Frozen V8 Top50 union PPR/PF/AR/CBF/Popularity; no label-forced positives',
        'history':[1583704800,1589749200],'target':[1589749200,1595797200],'user_count':len(cohort),'development_users':len(dev),
        'Kaggle_private_user_reported':.39908,'Kaggle_public_user_reported':.41811})
    print('V8 exact reproduction; now auditing all candidate ranks',flush=True)
    for filename,values in [('V8_crossfit_scores.npy',score),('V8_test_scores.npy',ts)]:
        path=OUT/filename
        if path.exists():
            if not np.array_equal(np.load(path,allow_pickle=False),values):raise ValueError('Saved baseline scores differ')
        else:np.save(path,values,allow_pickle=False)
    fields=['personal_frequency_score','user_program_airings','user_program_recency_days','pop_distinct_users','target_airing_count',
        'program_airings_recent_2w','program_active_weeks','completion80_share','personal_to_scheduled_airings','ppr_score']
    fi={c:names.index(c) for c in fields};seen_idx=names.index('user_program_seen');pr=names.index('popularity_rank')
    pp=names.index('ppr_rank');pf=names.index('personal_frequency_rank');ar=names.index('association_rules_rank');cb=names.index('cbf_rank')
    pref={g:[] for g in ('model','ppr','personal_frequency','association_rules','cbf','popularity')}
    group_rows=defaultdict(list);peruser=[];itemgroups=defaultdict(Counter);feature_acc={};feature_num=Counter()
    for bucket in ('true_top5','false_top5','true_6to10','true_11to20','true_below20','all_candidates'):
        feature_acc[bucket]=np.zeros(len(fields))
    for u in cohort:
        ix=ranks[u];ranking=[str(items[j]) for j in ix];pos={i:k+1 for k,i in enumerate(ranking)};truth=set(labels['secondary'][u])
        partitions={b:[] for b in ('true_top5','true_6to10','true_11to20','true_below20','missing')}
        rowindex={str(items[j]):j for j in ix}
        for i in sorted(truth,key=int):
            k=pos.get(i)
            b='missing' if k is None else 'true_top5' if k<=5 else 'true_6to10' if k<=10 else 'true_11to20' if k<=20 else 'true_below20'
            partitions[b].append(i)
            if k is not None:
                j=rowindex[i];feature_acc[b]+=X[j,[fi[c] for c in fields]];feature_num[b]+=1
                for tag in ('previously_watched' if X[j,seen_idx]>.5 else 'previously_unseen',
                            'popular_top50' if X[j,pr]<50 else 'less_popular_rank51plus'):
                    itemgroups[tag]['relevant_candidates']+=1;itemgroups[tag]['relevant_in_top5']+=k<=5
                    itemgroups[tag]['relevant_in_top10']+=k<=10;itemgroups[tag]['relevant_in_top20']+=k<=20
        wrong=[i for i in ranking[:5] if i not in truth]
        for i in wrong:
            j=rowindex[i];feature_acc['false_top5']+=X[j,[fi[c] for c in fields]];feature_num['false_top5']+=1
            for tag in ('previously_watched' if X[j,seen_idx]>.5 else 'previously_unseen','popular_top50' if X[j,pr]<50 else 'less_popular_rank51plus'):
                itemgroups[tag]['false_top5']+=1
        feature_acc['all_candidates']+=X[ix][:,[fi[c] for c in fields]].sum(axis=0);feature_num['all_candidates']+=len(ix)
        base={'user_id':u,'is_development':u in set(dev),'history_programs':pop['history_sizes'][u],
            'candidates':len(ix),'relevant':len(truth),'AP5':map5(pred,labels['secondary'],[u]),
            'hits5':len(partitions['true_top5']),'hits10':len(partitions['true_top5'])+len(partitions['true_6to10']),
            'hits20':len(partitions['true_top5'])+len(partitions['true_6to10'])+len(partitions['true_11to20']),
            'reachable':len(truth)-len(partitions['missing']),'missing':len(partitions['missing']),
            'wrong_top5':' '.join(wrong),**{('missing_program_ids' if b=='missing' else b):' '.join(v) for b,v in partitions.items()}}
        peruser.append(base)
        size=pop['history_sizes'][u];repeat_ratio=float(X[ix[0],names.index('user_history_airings_total')])/max(1,size)
        tags=['all','development' if u in set(dev) else 'assessment','template' if u in set(template) else 'nontemplate',
              'heavy101plus' if size>=101 else 'light1to20' if 1<=size<=20 else 'cold0' if size==0 else 'medium21to100',
              'strong_repeat3plus' if repeat_ratio>=3 else 'weak_repeat_under3']
        for tag in tags:group_rows[tag].append(base)
        # Within-user score/rank correlations: descriptive associations, not causal bias proof.
        gen={'model':score[ix], 'ppr':-X[ix,pp], 'personal_frequency':-X[ix,pf], 'association_rules':-X[ix,ar],
             'cbf':-X[ix,cb], 'popularity':-X[ix,pr]}
        for g,gs in gen.items():
            vals=[]
            for c in ('personal_frequency_score','user_program_recency_days','pop_distinct_users','target_airing_count'):
                target=X[ix,fi[c]].astype(float);target=-target if c=='user_program_recency_days' else target
                # Missing personal history recency sentinel71 means stale/missing, not recent.
                vals.append(float(spearmanr(gs,target).statistic) if np.ptp(gs)>0 and np.ptp(target)>0 else None)
            pref[g].append(vals)
    summaries={}
    for tag,rr in group_rows.items():
        denominator=sum(r['relevant'] for r in rr);reachable=sum(r['reachable'] for r in rr)
        summaries[tag]={'users':len(rr),'MAP5':sum(r['AP5'] for r in rr)/len(rr),
            'candidate_recall':sum(r['reachable'] for r in rr)/denominator if denominator else None,
            'Recall5':sum(r['hits5'] for r in rr)/denominator if denominator else None,
            'Recall10':sum(r['hits10'] for r in rr)/denominator if denominator else None,
            'Recall20':sum(r['hits20'] for r in rr)/denominator if denominator else None,
            'true_missing_fraction':sum(r['missing'] for r in rr)/denominator if denominator else None,
            'reachable_true_below5_fraction':(reachable-sum(r['hits5'] for r in rr))/reachable if reachable else None,
            'relevant_labels':denominator,'reachable_labels':reachable,'hits5':sum(r['hits5'] for r in rr)}
    table('ranking_errors_by_user.csv',peruser)
    table('ranking_error_groups.csv',[{'group':g,**v} for g,v in summaries.items()])
    preference={g:{c:float(np.mean([r[j] for r in v if r[j] is not None])) for j,c in enumerate(('frequency','recency','popularity','future_airing_opportunities')) if any(r[j] is not None for r in v)} for g,v in pref.items()}
    bucket_means={b:{c:float(feature_acc[b][j]/feature_num[b]) for j,c in enumerate(fields)} for b in feature_acc if feature_num[b]}
    # Train/assessment/test distribution comparisons include matched-template summaries to distinguish population from time shift.
    groups={'development':np.isin(us,dev),'assessment':np.isin(us,assessment),'template_validation':np.isin(us,pop['template_users'])}
    drift=[]
    for j,c in enumerate(names):
        row={'feature':c}
        for period,mask in groups.items():row.update({period+'_'+k:v for k,v in moments(X[mask,j].astype(float)).items()})
        row.update({'test_'+k:v for k,v in moments(tz['X'][:,j].astype(float)).items()})
        row['test_fraction_outside_development_range']=float(np.mean((tz['X'][:,j]<X[groups['development'],j].min())|(tz['X'][:,j]>X[groups['development'],j].max())))
        drift.append(row)
    table('feature_distribution_shift.csv',drift)
    dump('ranking_analysis.json',{'groups':summaries,'item_group_counts':itemgroups,'rank_score_spearman_mean_within_user':preference,
        'feature_means_by_error_bucket':bucket_means,'model_not_changed_before_analysis':True,
        'definitions':{'popular':'training popularity rank<50','strong_repeat':'historical qualifying-airing count per distinct historical program>=3; behavior only',
            'seen':'user_program_seen flag in historical10wk feature window','recall':'micro against all relevant Top5 labels including labels missing from candidates'},
        'remaining_official_target_uncertainty':'Selection uses same local>=120s top5 target as V8.>=80% rule remains an unverified hypothesis.'})
    dump('analysis_complete.json',{'ranking_analysis_sha256':sha(OUT/'ranking_analysis.json'),'drift_sha256':sha(OUT/'feature_distribution_shift.csv'),
        'validation_source_sha256':sha(V8/'validation.npz'),'test_source_sha256':sha(V8/'test.npz'),
        'baseline_scores_sha256':sha(OUT/'V8_crossfit_scores.npy'),'baseline_test_scores_sha256':sha(OUT/'V8_test_scores.npy'),
        'seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    print(json.dumps({'groups':summaries,'itemgroups':itemgroups,'bucket_means':bucket_means,'seconds':time.monotonic()-start},indent=2))


if __name__=='__main__':main()
