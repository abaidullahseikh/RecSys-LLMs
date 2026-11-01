"""Validate/reuse the existing V6 test-feature cache without rewriting evidence.

Despite the historical filename, this repair-mode command never builds data or
creates a replacement manifest. Missing artifacts require explicit investigation.
The existing cache is V5-compatible, not the output of v6_data.build('test').
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import validation

ROOT=validation.ROOT
V6=ROOT/'results/v6_experiment'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def validate_existing(directory=V6,root=ROOT):
    path=directory/'test_features.npz';manifest_path=directory/'test_feature_manifest.json'
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('Both test_features.npz and its existing test_feature_manifest.json are required; nothing was generated')
    meta=json.loads(manifest_path.read_text())
    if digest(path)!=meta['feature_file_sha256']:raise ValueError('Test feature checksum mismatch')
    source=root/meta['source_path']
    if not source.is_file() or digest(source)!=meta['source_sha256']:raise ValueError('Test source checksum mismatch')
    # Eagerly materialize once: repeated NPZ indexing decompresses whole arrays.
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    if set(z)!={'X','users','items','columns'}:raise ValueError('Unexpected test archive schema')
    if z['X'].shape!=(meta['rows'],meta['features']):raise ValueError('Test matrix dimensions differ from manifest')
    if len(z['users'])!=meta['rows'] or len(z['items'])!=meta['rows']:raise ValueError('Test row identity count mismatch')
    if z['columns'].tolist()!=meta['columns']:raise ValueError('Feature order mismatch')
    if z['users'].dtype.kind not in 'US' or z['items'].dtype.kind not in 'US':raise ValueError('IDs must be exact strings')
    if not np.isfinite(z['X']).all():raise ValueError('Nonfinite test features')
    if len(set(zip(z['users'].tolist(),z['items'].tolist())))!=meta['rows']:raise ValueError('Duplicate test pair')
    j=z['columns'].tolist().index('program_airings_recent_2w')
    if int(np.count_nonzero(z['X'][:,j]))!=meta['recent_2w_nonzero_rows']:raise ValueError('Recent counter differs from manifest')
    return {'all_passed':True,'rows':meta['rows'],'features':meta['features'],
        'archive_sha256':meta['feature_file_sha256'],'manifest_sha256':digest(manifest_path),
        'source_verified':True,'features_rebuilt':False,'manifest_rewritten':False,
        'scope':'Existing V5-compatible cache integrity, schema, identities, finite values and recorded counts; not a completed corrected V6 inference build.'}


if __name__=='__main__':print(json.dumps(validate_existing(),indent=2))
