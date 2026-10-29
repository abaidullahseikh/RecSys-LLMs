"""Validate Step 4B measurements and create separately named exports.

Never executes training/scoring code or overwrites any different existing file.
Independent checks use raw records, saved factors, saved lists and hashes.
"""
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct


def digest(data):return hashlib.sha256(data).hexdigest()


def write_new(path,data):
    if isinstance(data,str):data=data.encode('utf-8')
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError('Refusing overwrite: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)
    return digest(data)


def as_json(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)


def csv_text(fields,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows);return stream.getvalue()


def finalize(root):
    root=Path(root);out=root/'evidence/hw5_mf_reproduction'
    load=lambda name:json.loads((out/name).read_text())
    baseline=load('protected_before.json')
    changed=[name for name,expected in baseline.items() if not Path(name).exists() or digest(Path(name).read_bytes())!=expected]
    if changed:raise AssertionError('Protected files changed: '+repr(changed))
    metadata=load('model_metadata.json');weights=load('model_weights.json');init=load('initial_weights.json')
    audit=load('dataset_audit_verified.json');training=load('training_history.json');verification=load('verification.json')
    summary=load('summary.json');users=load('two_users.json');provenance=load('provenance.json')
    restored=load('restoration_verification.json')
    old=json.loads((root/'evidence/bundle.json').read_text())['evidence']
    assert metadata['label']==summary['label']=='Corrected HW5 MF reproduction'
    assert metadata['originalCheckpointRecovered'] is False
    assert summary['selectedEpoch']==metadata['selectedEpoch']==min(training['history'],key=lambda r:(r['valRMSE'],r['epoch']))['epoch']
    assert len(training['history'])==20 and metadata['config']=={'factors':16,'epochs':20,'lr':.01,'reg':.05,'batchSize':64}
    assert summary['validationRMSE']==metadata['validationRMSE']==metadata['restoredValidationRMSE']==restored['validationRMSE']
    for name,expected in metadata['sourceHashes'].items():assert digest((root/name).read_bytes())==expected
    for name,expected in metadata['implementationHashes'].items():assert digest((root/name).read_bytes())==expected
    assert digest((out/'model_weights.json').read_bytes())==metadata['modelArtifact']['sha256']
    for k,shape,h in [('userFactors','userShape','users'),('movieFactors','movieShape','movies')]:
        assert len(weights[k])==math.prod(weights[shape]) and all(math.isfinite(x) for x in weights[k])
        assert weights[shape]==init[shape]
        actual=digest(struct.pack('<'+'f'*len(weights[k]),*weights[k]))
        assert actual==metadata['modelFloat32Hashes'][h]==restored['float32Hashes'][h]
        assert actual==training['epochHashes'][summary['selectedEpoch']-1][h]
    assert weights['userShape']==[944,16] and weights['movieShape']==[1683,16]

    # Reconstruct all relevance sets and exclusions from the original raw records.
    rows=[tuple(map(int,line.split())) for line in (root/'u.data').read_text().splitlines() if line.strip()]
    by_user=collections.defaultdict(list)
    for row in rows:by_user[row[0]].append(row)
    splits={n:[] for n in ['train','val','test']}
    for u,rs in by_user.items():
        rs.sort(key=lambda r:(r[3],r[1]));a=int(.8*len(rs));b=a+int(.1*len(rs))
        for name,part in zip(splits,[rs[:a],rs[a:b],rs[b:]]):splits[name].extend(part)
    assert {k:len(v) for k,v in splits.items()}==audit['counts']['splits']
    assert weights['globalMean']==sum(r[2] for r in splits['train'])/len(splits['train'])
    candidates=sorted({r[1] for r in splits['train']});pool=set(candidates)
    assert candidates==audit['candidateIds'] and len(pool)==1611
    positives={(u,i) for u,i,r,t in splits['train'] if r>=4};positive_users={u for u,i in positives}
    popularity=collections.Counter(i for u,i in positives)
    seen={name:collections.defaultdict(set) for name in splits};grades={name:collections.defaultdict(dict) for name in splits}
    for name,rs in splits.items():
        for u,i,r,t in rs:seen[name][u].add(i);grades[name][u][i]=r
    evaluations={};max_score_error=0;checked_lists=0
    for name,filename in [('val','validation_evaluation.json'),('test','test_evaluation.json')]:
        ev=load(filename);s=ev['summary'];evaluations[name]=ev
        relevant={};no_eligible=[];unseen=set();unseen_count=0
        for u in sorted(by_user):
            ban=seen['train'][u]|(seen['val'][u] if name=='test' else set())
            positive={i for i,r in grades[name][u].items() if r>=4}
            unseen|=positive-pool;unseen_count+=len(positive-pool)
            relevant[u]=(positive&pool)-ban
            if not relevant[u]:no_eligible.append(u)
        cohort=[u for u in sorted(by_user) if u in positive_users and relevant[u]]
        assert cohort==ev['cohort']==audit['cohorts'][name]==old['cohorts'][name]
        assert [r['userId'] for r in ev['perUser']]==cohort
        assert s['excludedUnseen']==unseen_count and s['unseenIds']==sorted(unseen)
        assert s['excludedNoEligible']==len(no_eligible) and s['noEligibleUserIds']==no_eligible
        assert s['checkedCandidateScores']==len(cohort)*len(pool)
        for row in ev['perUser']:
            u=row['userId'];rel=relevant[u];recs=row['recommendations'];ids=[r['movieId'] for r in recs]
            assert row['relevant']==len(rel) and row['relevantIds']==sorted(rel)
            assert len(ids)==len(set(ids))==10 and set(ids)<=pool
            assert not set(ids)&seen['train'][u]
            if name=='test':assert not set(ids)&seen['val'][u]
            assert recs==sorted(recs,key=lambda r:(-r['score'],r['movieId']))
            assert row['hit']==len(set(ids)&rel) and set(row['hitIds'])==set(ids)&rel
            assert row['recall']==row['hit']/row['relevant']
            assert row['meanPopularity']==sum(popularity[i] for i in ids)/len(ids)
            for rec in recs:
                i=rec['movieId'];assert rec['popularity']==popularity[i]
                prediction=weights['globalMean']+sum(weights['userFactors'][u*16+f]*weights['movieFactors'][i*16+f] for f in range(16))
                error=abs(prediction-rec['score']);max_score_error=max(max_score_error,error);assert error<2e-5
            checked_lists+=1
        assert abs(s['recall']-sum(r['recall'] for r in ev['perUser'])/len(cohort))<1e-14
        assert abs(s['meanPopularity']-sum(r['meanPopularity'] for r in ev['perUser'])/len(cohort))<1e-10
    assert summary['testRecall']==evaluations['test']['summary']['recall']
    assert summary['validationRecall']==evaluations['val']['summary']['recall']
    assert users['users']==old['fixedUsers']==[1,2]
    for item in users['lists']:
        assert item['recommendations']==next(r['recommendations'] for r in evaluations['test']['perUser'] if r['userId']==item['userId'])
    assert users['controlledPprUnchanged']==old['controlled']
    assert verification['originalAfter']['passed']==12 and verification['originalAfter']['failed']==verification['originalAfter']['pending']==0
    assert all(c['status']=='PASS' for c in verification['checks'])

    model_label=metadata['label'];model_sha=metadata['modelArtifact']['sha256']
    cohort_hash=digest(json.dumps(audit['cohorts']['test'],separators=(',',':')).encode())
    fields=['method','model_label','split','alpha','recall_at_10','cohort_size','candidate_count',
      'excluded_unseen_relevant','excluded_no_eligible_users','excluded_no_positive_edge_users','excluded_users_union',
      'mean_training_popularity','selected_ppr_alpha','selected_mf_epoch','mf_validation_rmse','mf_test_rmse',
      'validation_recall_at_10','validation_cohort_size','backend','metric_origin','evidence_source','model_artifact_sha256','test_cohort_sha256']

    def mf_row(name):
        s=evaluations[name]['summary']
        return dict(method='mf_hw5_reproduction',model_label=model_label,split=name,alpha='',recall_at_10=s['recall'],
          cohort_size=s['cohortSize'],candidate_count=1611,excluded_unseen_relevant=s['excludedUnseen'],
          excluded_no_eligible_users=s['excludedNoEligible'],excluded_no_positive_edge_users=s['excludedNoPositiveEdge'],
          excluded_users_union=len(s['excludedUserIds']),mean_training_popularity=s['meanPopularity'],selected_ppr_alpha=.5,
          selected_mf_epoch=summary['selectedEpoch'],mf_validation_rmse=summary['validationRMSE'],mf_test_rmse=summary['testRMSE'] if name=='test' else '',
          validation_recall_at_10=summary['validationRecall'],validation_cohort_size=len(audit['cohorts']['val']),backend=metadata['backend'],
          metric_origin='new corrected-HW5 reproduction',evidence_source='evidence/hw5_mf_reproduction/'+('test' if name=='test' else 'validation')+'_evaluation.json',
          model_artifact_sha256=model_sha,test_cohort_sha256=cohort_hash)
    mf_rows=[mf_row('val'),mf_row('test')]
    write_new(root/'results_hw5_mf_reproduction.csv',csv_text(fields,mf_rows))
    combined=[]
    labels={'popularity':'Popularity','pagerank':'Global PageRank','ppr':'Personalized PageRank'}
    for method in ['popularity','pagerank','ppr']:
        a=next(a for a in old['test']['audits'] if a['summary']['method']==method);s=a['summary']
        assert a['cohort']==audit['cohorts']['test']
        row=dict.fromkeys(fields,'')
        row.update(method=method,model_label=labels[method],split='test',alpha=.5 if method!='popularity' else '',
          recall_at_10=s['recall'],cohort_size=s['cohortSize'],candidate_count=1611,excluded_unseen_relevant=s['excludedUnseen'],
          excluded_no_eligible_users=s['excludedNoEligible'],excluded_no_positive_edge_users=s['excludedNoPositiveEdge'],
          excluded_users_union=len(s['excludedUserUnion']),mean_training_popularity=s['meanPopularity'],selected_ppr_alpha=.5,
          backend=s['backend'],metric_origin='reused previous measurement; exact cohort IDs verified',evidence_source='evidence/bundle.json',test_cohort_sha256=cohort_hash)
        if method=='ppr':
            v=next(a['summary'] for a in old['validation'] if a['summary']['alpha']==.5)
            row.update(validation_recall_at_10=v['recall'],validation_cohort_size=v['cohortSize'])
        combined.append(row)
    combined.append(mf_row('test'))
    write_new(root/'results_four_method_corrected_mf.csv',csv_text(fields,combined))

    rec_fields=['user_id','method','model_label','alpha','selected_mf_epoch','rank','movie_id','title','genres',
                'score','score_definition','training_popularity','mean_list_popularity','split','evidence_origin']
    def recommendation_rows(items,new):
        result=[]
        for item in items:
            for rank,r in enumerate(item['recommendations'],1):
                result.append(dict(user_id=item['userId'],method='mf_hw5_reproduction' if new else item['method'],
                  model_label=model_label if new else labels[item['method']],alpha='' if new or item['alpha'] is None else item['alpha'],
                  selected_mf_epoch=summary['selectedEpoch'] if new else '',rank=rank,movie_id=r['movieId'],title=r['title'],genres=';'.join(r['genres']),
                  score=r['score'],score_definition='raw mean-plus-dot rating-scale ranking score (unclipped)' if new else ('training positive-user count' if item['method']=='popularity' else 'graph-walk probability'),
                  training_popularity=r['popularity'],mean_list_popularity=item['meanPopularity'],split='test',
                  evidence_origin='new reproduction' if new else 'reused evidence/bundle.json'))
        return result
    mf_recs=recommendation_rows(users['lists'],True)
    write_new(root/'recommendations_hw5_mf_reproduction.csv',csv_text(rec_fields,mf_recs))
    all_recs=recommendation_rows([r for r in old['finalLists'] if r['method']!='mf'],False)+mf_recs
    write_new(out/'recommendations_four_method_corrected_mf.csv',csv_text(rec_fields,all_recs))
    historical={'original_hw5':provenance['historical_hw5'],
      'original_hw5_protocol_warning':'Historical Recall ranks the full catalog; not substituted into the HW6 table.',
      'supplied_hw6_reference':{'label':'Supplied HW6 MF reference (historical)','metadata':old['mf'],
        'test':next(r for r in old['test']['rows'] if r['method']=='mf')},
      'corrected_hw5_reproduction':summary,'limitation':summary['limitation'],'instructor_approval_documented':False}
    write_new(out/'three_model_provenance.json',as_json(historical))

    for name,expected_rows in [('results_hw5_mf_reproduction.csv',2),('results_four_method_corrected_mf.csv',4),('recommendations_hw5_mf_reproduction.csv',20)]:
        with (root/name).open(newline='',encoding='utf-8') as stream:parsed=list(csv.DictReader(stream))
        assert len(parsed)==expected_rows
        if 'recommendations' in name:
            for r,e in zip(parsed,mf_recs):assert r['title']==e['title'] and r['genres']==e['genres'] and float(r['score'])==e['score']
        else:
            for r in parsed:
                if r['method']=='mf_hw5_reproduction':assert float(r['recall_at_10'])==evaluations[r['split']]['summary']['recall']
    preservation={'existing_files_checked':len(baseline),'changed_existing_files':changed,'hw5_unchanged':True,
      'original_hw6_sources_and_evidence_unchanged':True,'both_original_session_exports_unchanged':True}
    write_new(out/'preservation_verification.json',as_json(preservation))
    report={'status':'COMPLETE','independent_saved_list_checks':checked_lists,'max_saved_score_error_against_double_precision_factors':max_score_error,
      'model_hashes_verified':True,'source_parity_verified':True,'shared_cohort_ids_identical':True,'shared_test_users':861,
      'results_rows':2,'four_method_rows':4,'new_mf_recommendation_rows':20,'combined_two_user_rows':80,
      'csv_roundtrip_passed':True,'original_hw6_tests':summary['originalSelfChecks'],**preservation}
    write_new(out/'artifact_verification.json',as_json(report))
    manifest={str(p.relative_to(root)):digest(p.read_bytes()) for p in out.rglob('*') if p.is_file() and p.suffix!='.log' and p.name!='manifest.json'}
    for name in ['hw5_mf_reproduction.js','hw5_mf_reproduction_experiment.js','run_hw5_mf_reproduction.py','export_hw5_mf_reproduction.py',
                 'results_hw5_mf_reproduction.csv','results_four_method_corrected_mf.csv','recommendations_hw5_mf_reproduction.csv']:
        manifest[name]=digest((root/name).read_bytes())
    write_new(out/'manifest.json',as_json(manifest))
    return {**summary,'verification':report}


if __name__=='__main__':
    print(as_json(finalize(Path(__file__).resolve().parent)))
