"""Check preserved starter/data bytes and record the final source manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'data.js': 'eb33f8441f652938cce07e7fc7fe1e36f38acd0b296234074ac90d367e2678cd',
    'index.html': '3a9317bf06f9ef3fee8da60000b1f96182eaf87d947b54c436bb1ceb73f92db1',
    'style.css': 'ea006b0f213cdd988916c9fea5f4789d89c7348c5627eaad2ef64b899729986b',
    'readme.md': '56f2b68e111cad2fed437cf9d375b54722d165bf4446b41138cfc0f59866c486',
    'u.data': '06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490',
    'u.item': '553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701',
    'vendor/d3.min.js': 'f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {name: {'expected': expected, 'actual': digest(ROOT / name)}
                 for name, expected in EXPECTED.items()}
    assert all(r['expected']==r['actual'] for r in preserved.values())
    assert digest(ROOT/'script.js')=='edcaf385714f03ebdd11cb9413740e33759aeb7de182e2476340cc6b9dbf990c'
    prior = ROOT.parent / 'Week 5'
    continuity = {name: digest(prior/name)==digest(ROOT/name) for name in ['data.js','u.data','u.item']}
    assert all(continuity.values())
    manifest = {p.name:digest(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.js','.py','.md']}
    report = {'preserved_files':preserved,'week5_data_continuity':continuity,'source_sha256':manifest,
        'week5_script_sha256':digest(prior/'script.js'),
        'mf_continuity_limitation':{
            'week5_actual_implementation':'Batch-average L2 on gathered rows, script.js:513; unseeded tf.randomNormal; ordered batches.',
            'week6_provided_reference':'Whole-matrix L2 per batch, script.js:953; seeded normal initialization and shuffled batches.',
            'previous_selected_factor_checkpoint_available':False,
            'comparison_label':'HW6 supplied MF reference, not a reproduced HW5 selected checkpoint'},
        'authorized_adapter_corrections':[
            'hw6AdapterPPR primary attempt passes seedNode, not rawUserId.',
            'hw6UserNodeIndex returns -1 when mappings and userIds do not contain the raw ID.'],
        'propagation_optimization':'Student helper only: WASM sparseSegmentSum, unchanged restart/update/tolerance/cap, copied plain-array memoization by graph/backend/alpha/tolerance/cap/seed.'}
    (ROOT/'evidence/source-integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'preserved_files_match':True,'week5_data_match':True,'graph_sha256':manifest['graph.js'],
                      'script_sha256':manifest['script.js']},indent=2))


if __name__=='__main__':
    main()
