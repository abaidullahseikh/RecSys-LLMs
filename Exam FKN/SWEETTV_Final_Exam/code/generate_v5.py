"""Write experimental V5 only after saved validation verification succeeds."""
import csv
import json
import time
import resource
import sys
import lightgbm as lgb
import numpy as np
from train_v5 import predictions
from features_v5 import ROOT,OUT,sha,records


def main():
    start=time.monotonic();destination=ROOT/'submission/final_submission_v5.csv'
    if destination.exists():raise FileExistsError('Never overwrite a submission')
    verification=json.loads((OUT/'validation_verification.json').read_text())
    if verification['all_passed'] is not True:raise ValueError('Validation checks did not pass')
    ex=json.loads((OUT/'experiment.json').read_text());selection=ex['selection']
    if sha(OUT/'final_model.txt')!=ex['final_model_sha256']:raise ValueError('Final model checksum changed')
    z=np.load(OUT/'test_features.npz');X=z['X'];columns=z['columns'].tolist()
    if columns!=ex['columns']:raise ValueError('Feature columns differ')
    template=ROOT.parent/'sweettv-tv-program-recommender/submission.csv'
    users=tuple(r['user_id'] for r in records(template))
    if len(users)!=1260 or len(set(users))!=1260:raise ValueError('Invalid original user template')
    model=lgb.Booster(model_file=str(OUT/'final_model.txt'))
    scores=model.predict(X[:,selection['feature_indices']],num_threads=4)
    pred=predictions(scores,X,z['users'],z['items'],users,columns,selection['lgb_weight'])
    pool={str(int(r['tv_show_id'])) for r in records(ROOT.parent/'sweettv-tv-program-recommender/export_arh_31-42-final.csv') if int(r['tv_show_id'])>0 and int(r['duration'])>0}
    for u,p in pred.items():
        if len(p)!=5 or len(set(p))!=5 or not set(p)<=pool:raise ValueError('Invalid Top5')
    with destination.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['user_id','tv_show_id']);writer.writerows((u,' '.join(pred[u])) for u in users)
    audit=json.loads((ROOT/'results/v2_v4_diagnosis/audit.json').read_text())
    if any(sha(ROOT/p)!=h for p,h in audit['protected_hashes'].items()):raise ValueError('Prior evidence changed')
    manifest={'version':5,'status':'experimental_validation_improvement_Kaggle_unscored',
        'recommended_designated_submission':'submission/final_submission_v2.csv',
        'model':'Native LightGBM binary candidate reranker, corrected calendar/completion features',
        'selection':selection,'validation':ex['outer_comparisons'],
        'history':'Permitted behavior through Week30; candidate signals from Weeks11-30; new behavioral features use trailing Weeks21-30',
        'supervised_training':'All original historical examples: Weeks11-20 features -> Weeks21-30 top-five labels, after held-out evaluation was frozen',
        'target':'Weeks31-42 schedule; all valid positive IDs in named schedule remain eligible; schedule feature counts exclude spillovers',
        'future_viewing_used':False,'test_labels_used':False,'kaggle_score':None,'uploaded_to_kaggle':False,
        'user_count':len(users),'predictions_per_user':5,'eligible_programs':len(pool),'template_sha256':sha(template),
        'repeat_programs_allowed':True,'duplicate_top5_ids_allowed':False,
        'candidate_policy':'Reuse frozen Top50 union PPR, Personal Frequency, Association Rules, CBF, Popularity; no label-forced candidates',
        'submission_path':str(destination.relative_to(ROOT)),'submission_sha256':sha(destination),
        'model_path':str((OUT/'final_model.txt').relative_to(ROOT)),'model_sha256':sha(OUT/'final_model.txt'),
        'feature_path':str((OUT/'test_features.npz').relative_to(ROOT)),'feature_sha256':sha(OUT/'test_features.npz'),
        'feature_manifest_sha256':sha(OUT/'feature_manifest.json'),'experiment_sha256':sha(OUT/'experiment.json'),
        'environment':{'python':sys.version,'lightgbm':lgb.__version__,'numpy':np.__version__},
        'generation_seconds':time.monotonic()-start,'generation_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'previous_submissions_preserved':True,'session_exports_preserved':True}
    (ROOT/'results/final_model_v5_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('submission_path','submission_sha256','user_count','eligible_programs','generation_seconds')},indent=2))


if __name__=='__main__':main()
