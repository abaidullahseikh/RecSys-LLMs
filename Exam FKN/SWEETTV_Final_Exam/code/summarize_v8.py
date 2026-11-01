"""Export measured V8 tables and test evidence; does not alter old experiments."""
import csv,json,io,time,unittest
import numpy as np
from v8_common import OUT,ROOT,read,dump,records


def table(name,rows):
    path=OUT/name
    if path.exists():raise FileExistsError('Preserve table evidence')
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    ex=read(OUT/'experiment.json');rows=[];strata=[]
    for name,entry in ex['comparisons'].items():
        rows.append({'model':name,'features':entry['feature_count'],'rounds':78,
            'development_users':3901,'development_MAP':entry['development']['secondary_MAP'],
            'assessment_users':992,'assessment_MAP':entry['assessment']['all']['secondary_MAP'],
            'assessment80_MAP':entry['assessment']['all']['fraction80_MAP'],
            'crossfit_users':4893,'crossfit_MAP':entry['all_users_crossfit']['all']['secondary_MAP'],
            'crossfit80_MAP':entry['all_users_crossfit']['all']['fraction80_MAP'],
            'cv_training_seconds':entry['development']['seconds'],'assessment_fit_predict_seconds':entry['assessment_fit_inference_seconds'],
            'inference_seconds':entry['inference_seconds'],'selected':name==ex['selected']})
        for population in ('assessment','all_users_crossfit'):
            for stratum,m in entry[population].items():strata.append({'model':name,'population':population,'stratum':stratum,**m})
    table('comparison.csv',rows);table('validation_strata.csv',strata)
    table('selected_feature_importance.csv',ex['feature_importance'][ex['selected']])
    labels=read(OUT/'labels.json');pop=read(OUT/'population.json');v5=read(ROOT/'results/v5_experiment/experiment.json')
    differences={}
    for name,us in [('broader',pop['users']),('template',pop['template_users']),('historical_V5_outer',v5['outer_users'])]:
        added=sum(len(set(labels['fraction80'][u])-set(labels['secondary'][u])) for u in us)
        removed=sum(len(set(labels['secondary'][u])-set(labels['fraction80'][u])) for u in us)
        differences[name]={'users':len(us),'Top5_changed_users':sum(set(labels['secondary'][u])!=set(labels['fraction80'][u]) for u in us),
            'added80_labels':added,'removed120_labels':removed,'symmetric_user_program_changes':added+removed,
            'status':'Alternative hypothesis; not an official correction'}
    dump('label_definition_differences.json',differences)
    feature_audit=[
        ('historical_frequency','personal_frequency_score, user_program_airings','Present; full-history generator frequency and trailing10-week personal frequency have explicit distinct scopes'),
        ('recent_counts','program_airings_recent_2w/4w/8w','Present and populated in V5; not duplicated'),
        ('distinct_weeks','program_active_weeks','Present'),
        ('weekly_consistency','program_week_consistency, active_week_fraction','Present; duplicate-valued consistency columns retained for exact original schema'),
        ('recency','user_program_recency_days, last_watched_week_age','Present; missing sentinel documented in V5'),
        ('completion','completion50_count/80_count/95_count and shares','Present'),
        ('qualifying_airings','airings_ge_120/300/600','Present'),
        ('catch_rate','personal_to_scheduled_airings','Present; opportunity denominator is all historical labeled scheduled airings, not a user-availability model'),
        ('channel_loyalty','candidate_channel_affinity, recent_channel_affinity','Present; original75 template rows preserved rather than silently repaired'),
        ('hour_preference','candidate_hour_affinity','Present'),
        ('stickiness','viewing_stickiness, repeat_airing_share','Present'),
        ('history_confidence','user_recent_active_weeks, user_recent_airings_per_week and history counts','Present'),
        ('future_opportunities','target_airing_count, target_airings_per_week, target_availability_week_fraction','Present; schedule only'),
        ('new_completion_streak','completion80_active_week_streak','New: consecutive latest weeks with >=80% completed airing; differs from V7 >=120s streak'),
        ('new_recent_share','recent2w_personal_airing_share','New: recent2w program count divided by user recent2w count; differs from V5 ten-week personal_airing_share')]
    table('feature_audit.csv',[{'signal':a,'columns':b,'finding':c} for a,b,c in feature_audit])
    modules=('test_data_pipeline','test_step3','test_step4','test_step5','test_step6','test_step8','test_content_based',
        'test_lightgbm_v4','test_v5','test_v6','test_v6_verification','test_v7','test_v8')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules)
    log=io.StringIO();t=time.monotonic();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/'regression_tests.txt').write_text(log.getvalue())
    dump('regression_tests.json',{'all_passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'modules':modules,'seconds':time.monotonic()-t})
    print({'tests':result.testsRun,'passed':result.wasSuccessful(),'label_differences':differences})
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
