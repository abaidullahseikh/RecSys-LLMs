"""Seen/unseen audit and small development-only score calibration experiment.

Reuses saved leakage-safe V9 scores/models; no candidate regeneration.
"""
from collections import defaultdict,Counter
from pathlib import Path
import csv,gzip,json,time,resource,math
import numpy as np
import lightgbm as lgb
import evaluation,validation
from v8_common import ROOT,SOURCE,read,load,sha,records,inner_fold,map5
from train_v9 import new_matrix

OUT=ROOT/'results/v10_experiment';V8=ROOT/'results/v8_experiment';V9=ROOT/'results/v9_experiment'
GRID=(-.04,-.02,-.01,0.,.01,.02,.04)


def dump(name,obj):
    p=OUT/name
    if p.exists():
        if read(p)!=obj:raise ValueError(f'Existing evidence differs: {name}')
        return
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def history_flags(z,period):
    path=validation.STEP2/'threshold_120s'/f'{period}_interactions.csv.gz'
    im=read(validation.STEP2/'threshold_120s/interaction_manifest.json')
    if sha(path)!=im['artifacts'][path.name]['sha256']:raise ValueError('Historical interaction hash changed')
    bounds=validation.WINDOWS['train' if period=='train' else 'history']
    pairs=set();counts=Counter();airings=Counter()
    for r in records(path):
        u=r['user_id'];i=r['tv_show_id'];f,l=int(r['first_watch_utc']),int(r['last_watch_utc'])
        if not bounds[0]<=f<l<=bounds[1] or int(i)<=0:raise ValueError('Future or invalid historical identity')
        pairs.add((u,i));counts[u]+=1;airings[u]+=int(r['qualifying_airings'])
    us=z['users'];it=z['items'];seen=np.asarray([(str(u),str(i)) in pairs for u,i in zip(us,it)],dtype=bool)
    sizes=np.asarray([counts[str(u)] for u in us],dtype=np.int32)
    cold=sizes==0
    light=(sizes>=1)&(sizes<=20)
    return seen,sizes,cold,light,{'source_path':str(path.relative_to(ROOT)),'source_sha256':sha(path),'window':list(bounds),
        'definition':'Seen means at least one >=120-second qualifying historical airing before the prediction cutoff; no claim of ever-seen outside this log',
        'user_sizes':dict(counts),'user_airings':dict(airings)}


def group_indices(z):
    result=defaultdict(list)
    for n,u in enumerate(z['users']):result[str(u)].append(n)
    return {u:np.asarray(ix,dtype=np.int32) for u,ix in result.items()}


def rank(scores,z,indices,cohort,seen,cold,light,bonus=0.,policy='global',full=False):
    pop=z['columns'].tolist().index('popularity_rank');out={}
    for u in cohort:
        ix=indices[u];num=np.asarray([int(z['items'][n]) for n in ix],dtype=np.int64)
        if cold[ix[0]]:adjust=np.zeros(len(ix))
        else:
            enabled=(policy=='global') or (policy=='light1to20' and light[ix[0]])
            adjust=scores[ix]+(bonus if enabled else 0.)*(~seen[ix])
        order=np.lexsort((num,z['X'][ix,pop],-adjust))
        out[u]=tuple(str(z['items'][n]) for n in ix[order if full else order[:5]])
    return out


def metric_pack(pred,truth,users,z,indices,seen):
    ix_item={u:{str(z['items'][n]):int(n) for n in indices[u]} for u in users}
    s=Counter();den=sum(len(truth[u]) for u in users)
    for u in users:
        for i in pred[u][:5]:
            flag='seen' if seen[ix_item[u][i]] else 'unseen'
            s[flag+'_slots']+=1;s[flag+'_hits']+=i in truth[u]
    return {'users':len(users),'MAP5':map5(pred,truth,users),'Recall5':sum(s[k] for k in ('seen_hits','unseen_hits'))/den if den else None,
        'seen_precision':s['seen_hits']/s['seen_slots'] if s['seen_slots'] else None,
        'unseen_precision':s['unseen_hits']/s['unseen_slots'] if s['unseen_slots'] else None,**s}


def save(name,pred):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['user_id','tv_show_id']);w.writerows((u,' '.join(p[:5])) for u,p in pred.items())


def moments(v):
    v=np.asarray(v,dtype=float)
    if not len(v):return {'rows':0}
    return {'rows':len(v),'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max()),
        'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'p99':float(np.quantile(v,.99))}


def evaluate_audit(scores,z,ix,cohort,labels,seen,cold,light,provenance):
    full=rank(scores,z,ix,cohort,seen,cold,light,full=True);truth=labels['secondary'];names=z['columns'].tolist()
    counts={'seen':Counter(),'unseen':Counter()};peruser=[];groups=defaultdict(list);disp=Counter()
    candidate_labels=np.asarray([str(i) in truth[str(u)] for u,i in zip(z['users'],z['items'])])
    for flag,mask in [('seen',seen),('unseen',~seen)]:
        counts[flag]['candidate_pairs']=int(mask.sum());counts[flag]['relevant_candidate_pairs']=int((mask&candidate_labels).sum())
    for u in cohort:
        rs=full[u];pos={i:k+1 for k,i in enumerate(rs)};byitem={str(z['items'][n]):int(n) for n in ix[u]};relevant=set(truth[u])
        seen_set={i for i,n in byitem.items() if seen[n]};total=provenance['user_airings'].get(u,0);size=provenance['user_sizes'].get(u,0)
        wrongseen=[i for i in rs[:5] if i in seen_set and i not in relevant]
        unseentruebelow=[i for i in relevant if i in pos and i not in seen_set and pos[i]>5]
        disp['users_with_wrong_seen_top5_and_reachable_unseen_true_below5']+=bool(wrongseen and unseentruebelow)
        disp['wrong_seen_slots_in_those_users']+=len(wrongseen) if unseentruebelow else 0
        disp['unseen_true_below5_in_those_users']+=len(unseentruebelow) if wrongseen else 0
        for i in relevant:
            flag='seen' if i in seen_set else 'unseen'
            # Relevant items missing from candidate pool require exact historical pair lookup.
            if i not in pos:continue
            counts[flag]['true_1to5']+=pos[i]<=5;counts[flag]['true_6to10']+=6<=pos[i]<=10
            counts[flag]['true_11to20']+=11<=pos[i]<=20;counts[flag]['true_below20']+=pos[i]>20
        slots=Counter()
        for i in rs[:5]:
            flag='seen' if i in seen_set else 'unseen';slots[flag+'_slots']+=1
            slots[flag+'_hits']+=i in relevant
            counts[flag]['top5_slots']+=1;counts[flag]['top5_hits']+=i in relevant
        row={'user_id':u,'history_programs':size,'repeat_airings_per_program':total/max(1,size),'MAP5':map5(full,truth,[u]),
            'wrong_seen_top5':' '.join(wrongseen),'reachable_unseen_true_below5':' '.join(sorted(unseentruebelow,key=int)),
            **{k:slots[k] for k in ('seen_slots','seen_hits','unseen_slots','unseen_hits')}}
        peruser.append(row)
        tags=['all','heavy101plus' if size>=101 else 'light1to20' if 1<=size<=20 else 'cold0' if size==0 else 'medium21to100',
            'strong_repeat3plus' if total/max(1,size)>=3 else 'weak_repeat_under3']
        for g in tags:groups[g].append(row)
    dist={}
    for flag,mask in [('seen',seen),('unseen',~seen)]:
        for rel,rm in [('all',np.ones(len(seen),bool)),('relevant',candidate_labels),('irrelevant',~candidate_labels)]:
            k=mask&rm;dist[flag+'_'+rel]={'model_score':moments(scores[k]),**{c:moments(z['X'][k,names.index(c)]) for c in
                ('ppr_score','personal_frequency_score','association_rules_score','cbf_similarity','pop_distinct_users','target_airing_count')}}
        d=counts[flag];d['candidate_relevance_rate']=d['relevant_candidate_pairs']/max(1,d['candidate_pairs'])
        d['Top5_precision']=d['top5_hits']/max(1,d['top5_slots'])
        for k in ('true_1to5','true_6to10','true_11to20','true_below20'):d[k+'_fraction_of_reachable_relevant']=d[k]/max(1,d['relevant_candidate_pairs'])
    groupmetrics={g:{'users':len(rr),'MAP5':sum(r['MAP5'] for r in rr)/len(rr),
        'seen_precision':sum(r.get('seen_hits',0) for r in rr)/max(1,sum(r.get('seen_slots',0) for r in rr)),
        'unseen_precision':sum(r.get('unseen_hits',0) for r in rr)/max(1,sum(r.get('unseen_slots',0) for r in rr))} for g,rr in groups.items()}
    table('seen_unseen_errors_by_user.csv',peruser)
    dump('seen_unseen_audit.json',{'counts':counts,'groups':groupmetrics,'score_distributions':dist,'displacement':disp,
        'definition':provenance['definition'],'denominators':'Group relevance rates are conditional on candidate pairs. Top5 capture fractions use reachable relevant labels. Displacement is co-occurrence of a wrong seen item and true unseen candidate below5, not evidence swapping would improve MAP.',
        'previous_V9_full_ranking_analysis':'results/v9_experiment/ranking_analysis.json'})


def main():
    t=time.monotonic()
    if (OUT/'experiment.json').exists():raise FileExistsError('Preserve completed V10 work')
    OUT.mkdir(exist_ok=True)
    prior=[]
    for folder in ('code','results','submission','evidence'):
        prior += [p for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in str(p)
            and 'v10' not in p.name and p.suffix not in ('.sqlite','.pyc')]
    prior+=[ROOT/'exam_quiz_FKN.md']+list(SOURCE.glob('*.csv'))
    dump('preservation_before.json',{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in prior})
    e8=read(V8/'experiment.json');e9=read(V9/'experiment.json');z=load(V8/'validation.npz');labels=read(V8/'labels.json');pop=read(V8/'population.json')
    cohort=tuple(pop['users']);dev=tuple(e8['development_users']);assessment=tuple(e8['assessment_users']);us=z['users'];names=z['columns'].tolist()
    if e9['development_users']!=list(dev) or e9['assessment_users']!=list(assessment) or e9['selected']['new_weight']!=1.:raise ValueError('Unexpected V9 baseline')
    dm=np.isin(us,dev);am=~dm;fold=np.asarray([inner_fold(str(u)) for u in us]);source=read(V9/'analysis_complete.json')
    if sha(V8/'validation.npz')!=e9['feature_source_sha256']:raise ValueError('Validation cache integrity')
    baseline8=np.load(V9/'V8_crossfit_scores.npy',allow_pickle=False)
    M,fn=new_matrix(z,pop['historical_users']);s9=np.zeros(len(us))
    for f in range(3):
        vi=np.flatnonzero(dm&(fold==f));model=lgb.Booster(model_file=str(V9/f'E2_inner{f}.txt'))
        if sha(V9/f'E2_inner{f}.txt')!=e9['inner_models'][f]['model_sha256'] or model.feature_name()!=fn:raise ValueError('V9 inner model integrity')
        s9[vi]=model.predict(M[vi],num_threads=4)
    model=lgb.Booster(model_file=str(V9/'assessment_model.txt'));s9[am]=model.predict(M[am],num_threads=4)
    seen,sizes,cold,light,provenance=history_flags(z,'train');ix=group_indices(z)
    flags_path=OUT/'validation_flags.npz'
    if flags_path.exists():
        previous=load(flags_path)
        if any(not np.array_equal(previous[k],v) for k,v in {'seen':seen,'sizes':sizes,'cold':cold,'light':light}.items()):raise ValueError('Historical flags changed')
    else:np.savez_compressed(flags_path,seen=seen,sizes=sizes,cold=cold,light=light)
    scores_path=OUT/'V9_crossfit_scores.npy'
    if scores_path.exists():
        if not np.array_equal(np.load(scores_path,allow_pickle=False),s9):raise ValueError('Baseline scores changed')
    else:np.save(scores_path,s9,allow_pickle=False)
    b8=rank(baseline8,z,ix,cohort,seen,cold,light);b9=rank(s9,z,ix,cohort,seen,cold,light)
    expected8=evaluation.parse_prediction_records(list(records(V8/'E2_streak_all_crossfit.csv')))
    expected9=evaluation.parse_prediction_records(list(records(V9/'selected_all_crossfit.csv')))
    if b8!=expected8 or b9!=expected9:raise ValueError('Baseline prediction replay failed; stop')
    baseline={'V8':{'development':metric_pack(b8,labels['secondary'],dev,z,ix,seen),'assessment':metric_pack(b8,labels['secondary'],assessment,z,ix,seen)},
        'V9':{'development':metric_pack(b9,labels['secondary'],dev,z,ix,seen),'assessment':metric_pack(b9,labels['secondary'],assessment,z,ix,seen)}}
    for name,expected in [('V8',.4096515850636589),('V9',.40947940698966073)]:
        if abs(baseline[name]['development']['MAP5']-expected)>1e-14:raise ValueError('Baseline metric changed')
    dump('baseline_reproduction.json',{'exact':True,'metrics':baseline,'V8_model_and_CSV':read(V8/'submission_manifest.json'),
        'V9_model_and_CSV':read(V9/'submission_manifest.json'),'history_flags':provenance,'seen_flags_sha256':sha(OUT/'validation_flags.npz'),
        'V9_scores_sha256':sha(OUT/'V9_crossfit_scores.npy')})
    print('V8/V9 rankings and metrics reproduced exactly',flush=True)
    evaluate_audit(s9,z,ix,cohort,labels,seen,cold,light,provenance)
    dump('plan.json',{'base':'V9 pure binary selected model; raw scores are native probability outputs',
        'global_grid':GRID,'user_specific_policy':'Apply same bonus only to noncold users with1-20 distinct qualifying historical programs; all others unchanged',
        'history_definition':provenance['definition'],'group_thresholds':'Light1-20,heavy101+,strong repeat airing count/program>=3; fixed before calibration',
        'selection':'Within each calibration family maximize development MAP including zero; tie prefer zero then smallest absolute correction. Nonzero usable only if it improves development MAP. Between valid improving rules select highest development MAP.',
        'mandatory_fallback':'If both calibration families select zero, train one modest LambdaRank on existing broader V9 features/groups. Prior V7 ranker used different template-only data; this fallback is experimental, not an assumed improvement.',
        'max_threads':4,'seed':2026,'assessment_selection_forbidden':True,'Kaggle_scores_used_as_labels':False})
    curve=[];best={}
    for policy in ('global','light1to20'):
        choices=[]
        for bonus in GRID:
            p=rank(s9,z,ix,dev,seen,cold,light,bonus,policy)
            row={'policy':policy,'unseen_bonus':bonus,**metric_pack(p,labels['secondary'],dev,z,ix,seen)};choices.append(row);curve.append(row)
        selected=max(choices,key=lambda r:(r['MAP5'],r['unseen_bonus']==0.,-abs(r['unseen_bonus'])))
        if selected['unseen_bonus']!=0. and selected['MAP5']<=baseline['V9']['development']['MAP5']+1e-12:raise ValueError('Nonzero without genuine development gain')
        best[policy]=selected
    table('development_calibration_grid.csv',curve)
    improving=[r for r in best.values() if r['unseen_bonus']!=0.]
    chosen=max(improving,key=lambda r:(r['MAP5'],-abs(r['unseen_bonus']))) if improving else None
    dump('selection.json',{'family_best':best,'selected_calibration':chosen,'frozen_before_assessment_scoring':True,
        'calibration_improvement_found':bool(improving),'development_grid_sha256':sha(OUT/'development_calibration_grid.csv')})
    comparisons={'E1_V8':baseline['V8'],'E2_V9':baseline['V9']};selected_pred=None
    for name,policy in [('E3_global','global'),('E4_light_viewers','light1to20')]:
        bonus=best[policy]['unseen_bonus'];dp=rank(s9,z,ix,dev,seen,cold,light,bonus,policy);ap=rank(s9,z,ix,assessment,seen,cold,light,bonus,policy)
        comparisons[name]={'development':metric_pack(dp,labels['secondary'],dev,z,ix,seen),'assessment':metric_pack(ap,labels['secondary'],assessment,z,ix,seen),
            'unseen_bonus':bonus,'policy':policy}
        save(name+'_development.csv',dp);save(name+'_assessment.csv',ap)
    model_info=None
    if chosen:
        strategy={'type':'calibrated_V9','policy':chosen['policy'],'unseen_bonus':chosen['unseen_bonus']}
        selected_pred=rank(s9,z,ix,cohort,seen,cold,light,chosen['unseen_bonus'],chosen['policy'])
        pairwise_status='Skipped: a development-supported simple calibration exists; no added model cost is warranted'
    else:
        # One optional new broad-population pairwise model; not an arbitrary forced bonus.
        from run_v7 import grouping
        params={**e9['parameters'],'objective':'lambdarank','label_gain':[0,1],'eval_at':[5],'lambdarank_truncation_level':8}
        oof=np.zeros(len(us));fold_meta=[]
        for f in range(3):
            ti=np.flatnonzero(dm&(fold!=f));vi=np.flatnonzero(dm&(fold==f));vu=tuple(u for u in dev if inner_fold(u)==f)
            if set(us[ti])&set(us[vi]):raise ValueError('Pairwise contamination')
            bst=lgb.train(params,lgb.Dataset(M[ti],label=z['binary_label'][ti],group=grouping(us[ti]),feature_name=fn),num_boost_round=78)
            path=OUT/f'pairwise_inner{f}.txt';bst.save_model(str(path));oof[vi]=bst.predict(M[vi],num_threads=4)
            fold_meta.append({'fold':f,'model_sha256':sha(path),'training_users':len(set(us[ti])),'validation_users':len(vu)})
        bst=lgb.train(params,lgb.Dataset(M[dm],label=z['binary_label'][dm],group=grouping(us[dm]),feature_name=fn),num_boost_round=78)
        bst.save_model(str(OUT/'pairwise_assessment.txt'));oof[am]=bst.predict(M[am],num_threads=4)
        selected_pred=rank(oof,z,ix,cohort,seen,cold,light)
        comparisons['E5_pairwise']={'development':metric_pack(selected_pred,labels['secondary'],dev,z,ix,seen),'assessment':metric_pack(selected_pred,labels['secondary'],assessment,z,ix,seen)}
        save('E5_development.csv',{u:selected_pred[u] for u in dev});save('E5_assessment.csv',{u:selected_pred[u] for u in assessment})
        final=lgb.train(params,lgb.Dataset(M,label=z['binary_label'],group=grouping(us),feature_name=fn),num_boost_round=78)
        final.save_model(str(OUT/'final_pairwise.txt'))
        model_info={'parameters':params,'rounds':78,'feature_names':fn,'folds':fold_meta,'final_model_sha256':sha(OUT/'final_pairwise.txt')}
        strategy={'type':'pairwise_V9_features','policy':'global','unseen_bonus':0.}
        pairwise_status='One conditional LambdaRank run: simple calibration failed; broad V9 feature context differs from V7, but no superiority assumed'
    save('selected_all_crossfit.csv',selected_pred)
    dump('experiment.json',{'status':'evaluated','baseline':baseline,'comparison':comparisons,'strategy':strategy,'family_best':best,
        'pairwise_status':pairwise_status,'pairwise_info':model_info,'development_users':list(dev),'assessment_users':list(assessment),
        'selected_development_MAP':map5(selected_pred,labels['secondary'],dev),'selected_assessment_MAP':map5(selected_pred,labels['secondary'],assessment),
        'historical_flags_provenance':provenance,'seconds':time.monotonic()-t,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'V8_still_recommended':True,'Kaggle_score':None,'mandatory_experimental_submission':True,
        'assessment_caveat':'Previously examined assessment population and descriptive audit included assessment. No assessment metric selected calibration or model settings.'})
    print(json.dumps({'family_best':best,'strategy':strategy,'comparison':comparisons,'seconds':time.monotonic()-t},indent=2))


if __name__=='__main__':main()
