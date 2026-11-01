"""Export measured V12 comparisons, added-item rank outcomes, resources and provenance."""
from v12_common import *

def main():
    vr=read(OUT/'validation_verification.json');sr=read(OUT/'submission_verification.json')
    if not vr['all_passed'] or not sr['all_passed']:raise ValueError('Final verification did not pass')
    ex=read(OUT/'experiment.json');chrono=read(OUT/'chronological_validation.json');sm=read(OUT/'submission_manifest.json');selection=read(OUT/'selection.json');base=read(OUT/'baseline_reproduction.json');rows=[]
    for n in ('E1','E2','E3','E4'):
        m=ex['comparison'][n]
        for stage in ('development','assessment'):
            rows.append({'id':n,'configuration':{'E1':'Original V10','E2':'Original V10 + expanded PPR pool','E3':'All expanded rows + binary LambdaRank','E4':'All expanded rows + graded LambdaRank'}[n],
                'population':stage,**{k:v for k,v in m[stage].items() if k!='counts'},'features':77,'objective':'lambdarank','rounds':78,'seed':2026,
                'training_seconds':m['training_seconds'],'inference_seconds':m['inference_seconds'],'peak_rss_mib':m['peak_rss_mib'],'parameters':json.dumps(m['parameters'],sort_keys=True),
                'early_MAP5':chrono['comparison'][n]['development']['MAP5'],'early_users':chrono['comparison'][n]['development']['users'],
                'matched_template_main_MAP5':chrono['main_matched_template_development'][n]['MAP5'],'selected':n==selection['selected'],
                'verification':f"validation {vr['passed_checks']} passed/{vr['failed_checks']} failed; submission {sr['passed_checks']} passed/{sr['failed_checks']} failed"})
    table('comparison.csv',rows)
    z,truth=data();dev=base['development_users'];index=defaultdict(dict)
    for j,(u,i) in enumerate(zip(z['users'],z['items'])):index[str(u)][str(i)]=j
    added=~z['original'];newrelevant=added&z['binary_label'].astype(bool);after={}
    for n in ('E2','E3','E4'):
        s=np.load(OUT/('V10_expanded_scores.npy' if n=='E2' else f'{n}_scores.npy'),allow_pickle=False);rr=ranks(s,z,dev);bins=Counter();count=0;positions=[]
        for u,r in rr.items():
            for k,i in enumerate(r,1):
                j=index[u][i]
                if not newrelevant[j]:continue
                count+=1;positions.append(k);bins['Top5' if k<=5 else '6to10' if k<=10 else '11to20' if k<=20 else 'below20']+=1
        after[n]={'additional_relevant_pairs':count,'rank_bins':dict(bins),'rank_bin_fractions':{k:bins[k]/count for k in ('Top5','6to10','11to20','below20')},'median_rank':float(np.median(positions))}
    dump('post_training_unseen_audit.json',after)
    phases={'reproduction':'baseline_reproduction.json','diagnosis':'unseen_diagnosis.json','main_training_evaluation':'experiment.json','early_training_evaluation':'chronological_validation.json',
        'final_expanded_features':'test_build.json','validation_verification':'validation_verification.json','final_refit_generation':'submission_manifest.json','submission_verification':'submission_verification.json'}
    timing={n:{k:read(OUT/p)[k] for k in ('seconds','peak_rss_mib')} for n,p in phases.items()}
    failures={p.name:{'seconds':read(p)['seconds'],'passed_checks':read(p)['passed_checks'],'failed_checks':read(p)['failed_checks']} for p in sorted(OUT.glob('validation_verification_failed_attempt*.json'))}
    dump('execution_summary.json',{'selected':selection['selected'],'timing':timing,'recorded_successful_stage_seconds':sum(v['seconds'] for v in timing.values()),'failed_verifier_attempts':failures,
        'recorded_seconds_including_failed_attempts':sum(v['seconds'] for v in timing.values())+sum(v['seconds'] for v in failures.values()),'maximum_recorded_peak_rss_mib':max(v['peak_rss_mib'] for v in timing.values()),
        'max_threads':4,'RAM_limit_GiB':11,'model_training_runtime_note':'E3/E4 main times include three user-disjoint OOF fits plus one development-only assessment fit each; early fits and final selected refit separately measured. Baselines reuse prior models, so their new-training time is0.',
        'validation':{'passed':vr['passed_checks'],'failed':vr['failed_checks']},'submission':{'passed':sr['passed_checks'],'failed':sr['failed_checks']},'regression_tests':vr['details']['regression_tests'],
        'protected_files':sr['details']['protected_files'],'submission_sha256':sm['submission_sha256'],
        'development_delta_vs_V10':sm['local_development_MAP']-base['comparison']['V10']['development']['MAP5'],
        'development_delta_vs_V11':sm['local_development_MAP']-base['comparison']['V11']['development']['MAP5'],
        'assessment_delta_vs_V10':sm['local_assessment_MAP']-base['comparison']['V10']['assessment']['MAP5'],
        'assessment_delta_vs_V11':sm['local_assessment_MAP']-base['comparison']['V11']['assessment']['MAP5'],
        'Kaggle_private_V10_user_reported':.40637,'Kaggle_private_V11_user_reported':.40634,'Kaggle_private_V12':None,'automatic_upload':False})
    dump('implementation_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').glob('*v12*.py'))})
    print({'selected':selection['selected'],'post_training_unseen_audit':after,'main_matched_MAP':{n:m['MAP5'] for n,m in chrono['main_matched_template_development'].items()},
        'uncertainty':ex['uncertainty'],'model_times':{n:{k:m[k] for k in ('training_seconds','inference_seconds','peak_rss_mib')} for n,m in ex['comparison'].items()},'timing':timing,
        'checksum':sm['submission_sha256']},flush=True)

if __name__=='__main__':main()
