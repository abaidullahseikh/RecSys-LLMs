"""Build submission CSV/JSON exports solely from saved browser measurements."""
import csv
import hashlib
import json
from pathlib import Path


def export_and_validate(root):
    root = Path(root)
    bundle = json.loads((root / 'evidence/bundle.json').read_text())
    evidence = bundle['evidence']
    assert evidence['selection']['testNotYetRun']
    assert evidence['test']['complete']
    for name, expected in evidence['sourceHashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    for name, text in bundle['originalExporterOutputs'].items():
        suffix = '.json' if name == 'verification' else '.csv'
        (root / 'evidence' / ('original-' + name + suffix)).write_text(text, encoding='utf-8')

    verification = json.loads(bundle['originalExporterOutputs']['verification'])
    assert verification['failed'] == verification['pending'] == 0
    audits = evidence['validation'] + evidence['test']['audits']
    verification['experiment_audits'] = [{
        'method': a['summary']['method'], 'split': a['summary']['split'],
        'alpha': a['summary']['alpha'], 'cohort_size': len(a['cohort']),
        'ranking_and_filtering_passed': a['summary']['allRankingAndFilteringChecksPassed'],
        'nonconverged': a['summary']['nonconverged'],
        'short_lists': a['summary']['shortLists'], 'zero_score_lists': a['summary']['zeroScoreLists'],
    } for a in audits]
    verification['nonconverged_runs'] = [
        {'split': a['summary']['split'], 'alpha': a['summary']['alpha'],
         'method': a['summary']['method'], 'user_id': r['userId'], **r['propagation']}
        for a in audits for r in a['perUser']
        if r['propagation'] and not r['propagation']['converged']]
    verification['selected_alpha'] = evidence['selection']['alpha']
    verification['source_sha256'] = evidence['sourceHashes']
    verification['mf_checkpoint_hash_unchanged'] = evidence['mf']['frozenModelHash'] == evidence['test']['modelHashAfterTest']
    verification['evidence_file'] = 'evidence/bundle.json'
    diagnostics=[r['propagation'] for a in audits for r in a['perUser'] if r['propagation']]
    for d in diagnostics:
        assert d['minProbability']>=0 and abs(d['probabilitySum']-1)<1e-5
        if d['converged']:
            assert d['residual']<1e-6 and d['iterations']<500 and d['stationaryL1']<2e-6
        else:
            assert d['iterations']==500
    verification['probability_audit']={
        'diagnostic_records':len(diagnostics),'nonnegative_and_mass_within_1e_5':True,
        'max_mass_error':max(abs(d['probabilitySum']-1) for d in diagnostics),
        'converged_runs_have_residual_below_1e_6_and_stationary_l1_below_2e_6':True}
    before=evidence['test']['memoryBefore'];after=evidence['test']['memoryAfter']
    assert before['numTensors']==after['numTensors'] and before['numDataBuffers']==after['numDataBuffers']
    verification['test_comparison_tensor_counts_unchanged']={
        'before':before['numTensors'],'after':after['numTensors'],
        'data_buffers_before':before['numDataBuffers'],'data_buffers_after':after['numDataBuffers']}
    diagnostic_path = root/'evidence/diagnostic-webgl-validation.json'
    if diagnostic_path.exists():
        dense = json.loads(diagnostic_path.read_text())
        sparse = next(a for a in evidence['validation'] if a['summary']['alpha']==0.5)
        assert dense['summary']['backend']=='webgl' and sparse['summary']['backend']=='wasm'
        assert dense['cohort']==sparse['cohort']
        rank_differences=[]
        max_error=0
        for left,right in zip(dense['perUser'],sparse['perUser']):
            assert left['userId']==right['userId']
            if [r['movieId'] for r in left['recommendations']] != [r['movieId'] for r in right['recommendations']]:
                rank_differences.append(left['userId'])
            scores={r['movieId']:r['score'] for r in left['recommendations']}
            for r in right['recommendations']:
                if r['movieId'] in scores:
                    max_error=max(max_error,abs(scores[r['movieId']]-r['score']))
        equivalence={'alpha':0.5,'split':'val','users':len(dense['cohort']),
                     'rank_different_user_ids':rank_differences,'max_shared_recommendation_score_error':max_error,
                     'dense_recall':dense['summary']['recall'],'sparse_recall':sparse['summary']['recall'],
                     'dense_backend':'webgl','optimized_backend':'wasm'}
        assert max_error<1e-6
        verification['backend_equivalence']=equivalence
        (root/'evidence/backend-equivalence.json').write_text(json.dumps(equivalence,indent=2),encoding='utf-8')
    (root / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')

    fields = ['method','alpha','recall_at_10','cohort_size','excluded_unseen_ids',
              'excluded_no_eligible_user','mean_popularity','iterations','residual','converged',
              'split','excluded_unseen_id_list','excluded_no_positive_edge_user',
              'excluded_users_union','selected_alpha','mf_checkpoint_epoch',
              'nonconverged_users','recall_runtime_ms','audit_runtime_ms','comparison_runtime_ms',
              'backend','mf_training_backend','convergence_scope']
    with (root / 'results.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            s = audit['summary']
            uses_graph = s['method'] in ['pagerank', 'ppr']
            writer.writerow(dict(method=s['method'], alpha=s['alpha'] if uses_graph else '',
                recall_at_10=s['recall'], cohort_size=s['cohortSize'],
                excluded_unseen_ids=s['excludedUnseen'], excluded_no_eligible_user=s['excludedNoEligible'],
                mean_popularity=s['meanPopularity'], iterations=s['maxIterations'] if uses_graph else '',
                residual=s['maxResidual'] if uses_graph else '',
                converged=str(s['nonconverged']==0).lower() if uses_graph else '', split=s['split'],
                excluded_unseen_id_list=';'.join(map(str,s['unseenIds'])),
                excluded_no_positive_edge_user=s['excludedNoPositiveEdge'], excluded_users_union=len(s['excludedUserUnion']),
                selected_alpha=evidence['selection']['alpha'],
                mf_checkpoint_epoch=evidence['mf']['info']['checkpoint'] if s['method']=='mf' else '',
                nonconverged_users=s['nonconverged'] if uses_graph else '',
                recall_runtime_ms=s.get('recallRuntimeMs',''), audit_runtime_ms=s['auditRuntimeMs'],
                comparison_runtime_ms=evidence['test']['comparisonRuntimeMs'] if s['split']=='test' else '',
                backend=s['backend'],mf_training_backend=evidence['mf']['backend'] if s['method']=='mf' else '',
                convergence_scope='maxima across cohort; nonconverged_users is a count' if uses_graph else ''))

    fields = ['user_id','method','alpha','rank','movie_id','title','genres','score','training_popularity',
              'setting','mean_list_popularity','split']
    lists = evidence['finalLists'] + evidence['controlled']
    expected = []
    with (root / 'recommendations.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for item in lists:
            for rank, rec in enumerate(item['recommendations'],1):
                row = dict(user_id=item['userId'],method=item['method'],alpha=item['alpha'] if item['alpha'] is not None else '',
                    rank=rank,movie_id=rec['movieId'],title=rec['title'],genres=';'.join(rec['genres']),score=rec['score'],
                    training_popularity=rec['popularity'],setting=item['setting'],mean_list_popularity=item['meanPopularity'],split='test')
                expected.append(row)
                writer.writerow(row)

    with (root / 'recommendations.csv').open(newline='',encoding='utf-8') as source:
        actual = list(csv.DictReader(source))
    assert len(actual)==len(expected)==110
    for row, exp in zip(actual,expected):
        assert row['title']==exp['title'] and row['genres']==exp['genres']
        assert int(row['movie_id'])==exp['movie_id'] and float(row['score'])==exp['score']
    with (root / 'results.csv').open(newline='',encoding='utf-8') as source:
        results=list(csv.DictReader(source))
    assert len(results)==8
    for row,audit in zip(results,audits):
        assert float(row['recall_at_10'])==audit['summary']['recall']
        assert int(row['cohort_size'])==len(audit['cohort'])
    assert verification['mf_checkpoint_hash_unchanged']
    assert all(a['ranking_and_filtering_passed'] for a in verification['experiment_audits'])
    report={'exports_validated':True,'results_rows':len(results),'recommendation_rows':len(actual),
            'original_tests':{'passed':verification['passed'],'failed':verification['failed'],'pending':verification['pending']},
            'audited_user_lists':sum(len(a['cohort']) for a in audits),
            'nonconverged_runs':len(verification['nonconverged_runs']),
            'source_hashes_match':True,'mf_checkpoint_hash_unchanged':True}
    (root / 'evidence/export-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print(json.dumps(export_and_validate(Path(__file__).resolve().parent),indent=2))
