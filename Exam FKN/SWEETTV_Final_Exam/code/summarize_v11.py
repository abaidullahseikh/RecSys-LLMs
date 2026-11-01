"""Export measured V11 comparisons and fold-level MAP without changing old work."""
import time,csv,json
from v8_common import ROOT,read,records,sha,inner_fold,map5
from prepare_v11 import OUT,V8,dump

def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    vr=read(OUT/'validation_verification.json');sr=read(OUT/'submission_verification.json')
    if not vr['all_passed'] or not sr['all_passed']:raise ValueError('Verification did not pass')
    result={**read(OUT/'E1_E2.json')['comparison'],**read(OUT/'hard_negative_experiment.json')['comparison']};chrono=read(OUT/'chronological_validation.json');selection=read(OUT/'selection.json');base=read(OUT/'baseline_reproduction.json');rows=[];folds=[]
    earlytruth=read(OUT/'early_labels.json');truth=read(V8/'labels.json')['secondary'];dev=base['development_users'];earlydev={u for m in chrono['early']['models'] for u in m['evaluation_users']}
    for n,m in result.items():
        for stage in ('development','assessment'):
            rows.append({'id':n,'configuration':{'E1':'Original V10','E2':'PPR expansion + V10','E3':'Hard negatives + original pool','E4':'Hard negatives + PPR expansion'}[n],
                'population':stage,**{k:v for k,v in m[stage].items() if k!='counts'},'objective':m['objective'],'features':77,'rounds':78,'seed':2026,
                'parameters':json.dumps(m['parameters'],sort_keys=True),'main_model_inference_seconds_shared':m.get('inference_seconds'),
                'hard_negative_training_seconds_shared':m.get('runtime_seconds_shared'),'peak_rss_mib':m['peak_rss_mib'],
                'early_development_MAP':chrono['early']['comparison'][n]['MAP5'],'early_development_users':chrono['early']['users'],
                'main_matched_template_MAP':chrono['main_matched_template_development']['comparison'][n]['MAP5'],'selected':n==selection['selected'],
                'verification':f"passed; validation {vr['passed_checks']}/{vr['failed_checks']}, submission {sr['passed_checks']}/{sr['failed_checks']}"})
        for period,path,labels,pop in [('main',OUT/f'{n}_all_crossfit.csv',truth,dev),('early',OUT/f'early_{n}_development.csv',earlytruth,earlydev)]:
            pred={r['user_id']:tuple(r['tv_show_id'].split()) for r in records(path)}
            for f in range(3):
                uu=sorted(u for u in pop if inner_fold(u)==f);folds.append({'configuration':n,'period':period,'inner_user_fold':f,'users':len(uu),'MAP5':map5(pred,labels,uu),
                    'history_start':1583704800,'history_end':1589749200 if period=='main' else 1586725200,'target_end':1595797200 if period=='main' else 1589749200})
    table('comparison.csv',rows);table('chronological_fold_scores.csv',folds)
    timingfiles={'reproduction':'baseline_reproduction.json','diagnosis':'diagnose_build.json','expanded_feature_build':'validation_build.json','E1_E2_evaluation':'E1_E2.json',
        'hard_negative_experiment':'hard_negative_experiment.json','early_feature_build':'early_build.json','chronological_evaluation':'chronological_validation.json',
        'validation_verification':'validation_verification.json','final_fit_and_generation':'submission_manifest.json','submission_verification':'submission_verification.json'}
    timing={k:{field:read(OUT/p)[field] for field in ('seconds','peak_rss_mib')} for k,p in timingfiles.items()}
    dump('execution_summary.json',{'timing':timing,'total_recorded_stage_seconds':sum(r['seconds'] for r in timing.values()),'maximum_recorded_peak_rss_mib':max(r['peak_rss_mib'] for r in timing.values()),
        'max_CPU_threads':4,'RAM_limit_GiB':11,'selected':selection['selected'],'development_MAP_delta_vs_V10':result[selection['selected']]['development']['MAP5']-result['E1']['development']['MAP5'],
        'assessment_MAP_delta_vs_V10':result[selection['selected']]['assessment']['MAP5']-result['E1']['assessment']['MAP5'],
        'verification':{'validation':{'passed':vr['passed_checks'],'failed':vr['failed_checks']},'submission':{'passed':sr['passed_checks'],'failed':sr['failed_checks']},'regression_tests':vr['details']['regression_tests']},
        'Kaggle_private_V10_user_reported':.40637,'Kaggle_public_V10_user_reported':.42051,'Kaggle_private_V11':None,
        'submission_sha256':read(OUT/'submission_manifest.json')['submission_sha256']})
    dump('implementation_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').glob('*v11*.py'))})
    print({'selected':selection['selected'],'timing':timing,'fold_scores':folds},flush=True)

if __name__=='__main__':main()
