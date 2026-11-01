"""Export actual V9 results and run applicable regressions without old writes."""
import csv,io,json,time,unittest
from v8_common import ROOT,read
from train_v9 import OUT,dump


def table(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    ex=read(OUT/'experiment.json');repro=read(OUT/'V8_reproduction.json')
    table('comparison.csv',[{'model':n,'development_users':len(ex['development_users']),'development_MAP':m['development_MAP'],
        'assessment_users':len(ex['assessment_users']),'assessment_MAP':m['assessment_MAP'],
        'selected':n=='selected'} for n,m in ex['comparison'].items()])
    table('feature_importance.csv',ex['feature_importance'])
    modules=('test_data_pipeline','test_step3','test_step4','test_step5','test_step6','test_step8','test_content_based',
        'test_lightgbm_v4','test_v5','test_v6','test_v6_verification','test_v7','test_v8','test_v9')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
    t=time.monotonic();log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/'regression_tests.txt').write_text(log.getvalue())
    dump('regression_tests.json',{'all_passed':r.wasSuccessful(),'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),
        'modules':modules,'seconds':time.monotonic()-t})
    print({'tests_run':r.testsRun,'all_passed':r.wasSuccessful(),'failed':len(r.failures),'errors':len(r.errors)})
    if not r.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
