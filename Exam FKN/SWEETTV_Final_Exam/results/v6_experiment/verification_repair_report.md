# V6 verification repair — completed without retraining

## Final recommendation

**Keep V5 as the designated final submission**, with the user's genuine reported private MAP@5 of **0.39706**. The locked inner-validation selection did not outperform V5 on the outer population. No V6 submission was generated or uploaded, and no new Kaggle score is claimed.

Final verification completed in **51.91 seconds: 63 checks passed, zero failed**. All **13 relevant regression tests passed**. Existing model training, completed feature tables, experiment results, V1–V5 submissions and original exported sessions were preserved.

## What was actually on disk

`experiment.json` uses these real top-level structures:

- `development_oof_results`: a list of 10 recipe/blend summaries, each with `recipe`, `weight`, and `metrics` (`secondary`, `fraction80`, `primary`). There is no `inner_cv` or `blend_oof_metrics` key.
- `selected` and `selected_recipe`: the recorded inner-validation selection.
- `outer_comparisons`: `reproduced_V5`, `fixes50`, `expanded100`, `temporal50`, `temporal100`, `selected`.
- `candidate_recall`: a list of four records with `k` (50 or 100), `population` (development or outer), `users`, `avg_candidates`, `recall`, and `insufficient_users`.
- `earlier_fold`, `fit_metadata`, `final_fit_metadata`, `parameters`, `rounds`, `feature_columns`, and `final_model_sha256`.

Actual feature artifacts are `main_data.npz` with `main_build.json`, `early_data.npz` with `early_build.json`, and `test_features.npz` with `test_feature_manifest.json`. **There is no V6 `feature_manifest.json`.** The main and early archives include Boolean `mask20`, `mask50`, and `mask100`; the test cache does not.

The V5 held-out prediction file is `results/v5_experiment/heldout_v5.csv`, containing 256 users. V2's saved validation CSV contains the full 1,260 users. A final Kaggle submission must never be substituted for a local validation prediction file.

## Code repaired

### `code/verify_v6.py`

- Removed references to nonexistent files, functions and JSON fields; loads the real manifests and schemas above.
- Caches each NPZ array once. Repeated archive access was repeatedly decompressing large arrays and causing timeouts. The evaluation protocol/candidate pool is also initialized once, rather than reloaded for every user.
- Uses a separately implemented rational AP reference, then cross-checks the project's production evaluator on the same users and labels.
- Replays all four saved V6 evaluation models on the **main temporal target**, including the temporal-augmentation models. Earlier training examples do not change the evaluation target for those comparisons.
- Independently rebuilds full rankings and the selected reciprocal-rank blend with the saved **V5 evaluation model**, rather than incorrectly substituting V2 or a truncated baseline list.
- Verifies actual recorded candidate recall for both development and outer users at limits 50/100. Top-20 outer recall is additionally recomputed and explicitly identified as diagnostic, not a recorded selection result.
- Checks model columns, boosting rounds, reconstructed supervised row/positive-label counts, group separation, chronology, source checksums, feature hashes and preserved submissions.
- Writes explicit named pass/fail checks and catches exceptions into a failed report. Subsequent runs preserve earlier reports by writing uniquely named rechecks.

### `code/create_v6_test_features.py`

This legacy-named command now **validates and reuses the existing cache only**. It does not rebuild feature tables, mutate V5 module paths, overwrite a manifest, or invent missing provenance. Missing cache/manifest/source inputs cause a clear failure. Its test validates that a corrupted source is rejected and that the existing manifest remains unchanged.

The existing test cache is byte-identical to V5's test archive. It is therefore accurately described as a **V5-compatible cache**, not a completed corrected V6 inference build. This distinction was previously obscured by the filename and resumed-build metadata.

### Regression tests

`code/test_v6_verification.py` adds tests for inner-only selection despite attractive outer scores, complete-rank blending with no vote for baseline-missing items, cold-start/tie handling, independent AP truncation/duplicates, and read-only cache/provenance validation. Existing V5 and V6 regression suites were rerun.

## Independently recomputed results

All main comparisons below use the same 256 outer users, target window and eligible validation catalog.

| Configuration | All-program 120s MAP@5 | Top5-120s MAP@5 | Top5-80% hypothesis MAP@5 |
|---|---:|---:|---:|
| Reproduced V5 | 0.827474 | **0.488633** | 0.378711 |
| Confirmed feature fixes, Top-50 | 0.827474 | **0.488633** | 0.378711 |
| Top-100 only | 0.826445 | **0.492617** | 0.369197 |
| Temporal examples, Top-50 | 0.830339 | 0.486354 | 0.377272 |
| Temporal examples, Top-100 | 0.822552 | 0.490091 | 0.375247 |
| **Inner-selected temporal50 + V5 blend** | 0.830339 | **0.485260** | 0.377272 |

The earlier-fold-only saved predictions were also rescored against independently reconstructed Weeks16–20 labels: secondary MAP@5 **0.494948**, primary **0.804102**, and 80% hypothesis **0.402901**. These scores concern a different target period and must not be compared directly as improvements over the main fold.

Candidate recall on the main outer users:

| Limit | Average candidates/user | Secondary recall |
|---|---:|---:|
| Top-50 | 108.758 | **0.895276** |
| Top-100 | 252.117 | **0.928346** |

Recall and average counts were independently recalculated from the actual mask-selected user/program IDs. No user has fewer than five candidates.

## Why Top-100 was not selected

The exact code-defined selection key is maximum **development-only OOF secondary MAP**, then primary MAP, then smaller candidate pool. Recomputing this key over all 10 saved summaries selects:

- Recipe: **`temporal50`** — earlier historical rows at weight 0.5, Top-50 generators.
- Rank blend: **75% temporal-model ranking + 25% V5-style baseline ranking**.
- Inner OOF secondary MAP: **0.4639674635**.

The pure Top-100 recipe's inner OOF MAP was **0.4591633466**. It subsequently scored **0.4926171875** on the outer users, but promoting it because of that result would change the selection rule after observing the holdout. The selected blend's outer result, **0.4852604167**, is **0.0033723958 below V5**. Even the best displayed Top-100 outer result is only **0.003984375 above V5** and is not an independent selection result.

The recorded selected weight beats pure temporal50 on inner OOF by only about **0.00000996**. This tiny selection margin and the reused project holdout are additional reasons not to claim a robust improvement.

## Chronology and leakage evidence

- Main features: Weeks11–20, UTC `[1583704800,1589749200)`; target Weeks21–30, `[1589749200,1595797200)`.
- Earlier features: Weeks11–15, `[1583704800,1586725200)`; target Weeks16–20, `[1586725200,1589749200)`.
- Independent overlap scans reconstructed accepted feature-period row counts (**1,823,893 main**, **924,319 early**) and confirmed **463** earlier-fold boundary-crossing rows were excluded conservatively.
- All three relevance definitions for all users in both target periods were reconstructed from the historical overlap cache and matched the saved label JSONs.
- Completion and recent-week arithmetic for **24 deterministic positive user/program pairs** (12 per period) was independently checked against existing overlap records.
- Outer users are absent from the reconstructed supervised fitting rows in both periods. The 1,004 development users and 256 outer users are disjoint and match the original SHA-256 split.
- Source code separates historical features from target labels. The recorded feature cutoff equals the target start; no Week31–42 viewing source is used. The final all-user refit legitimately includes historical labels after evaluation, but is never used to report outer validation accuracy.

This verifies chronology against stored sources, code paths and sampled feature arithmetic. It is not an independent reimplementation of every generator. The outer population has been examined in earlier project work, so it is not a new pristine research holdout.

## Preserved evidence and limitations

All current V6 result files were hashed before and after verification and remained unchanged. Historical V1–V5 submissions and native session exports matched their original snapshot hashes. The snapshot accidentally included the then-new `code/audit_v5_source.py`; its documented prior reporting repair is explicitly listed as the sole permitted snapshot difference. No V5 code/model/prediction file is exempted.

V5 submission SHA-256 remains:

`bb8aa11ae3d5276d70deddc697752c5d8bce358b17fe8b08c235cba8265e4d40`

Remaining verification limitations are recorded, not silently skipped:

1. Raw inner OOF predictions and inner models were not persisted. The selection argmax can be verified from recorded summaries; those summaries cannot be independently rescored without retraining.
2. The earlier-fold-only model was not persisted. Its saved predictions, labels and metrics were verified, but a model replay is unavailable.
3. The existing test cache is an unchanged V5-compatible copy, not a completed corrected V6 inference table. No V6 CSV is produced from it.

## Passed and failed attempts

Final: **63 verification checks passed / 0 failed**, **13 regression tests passed / 0 failed**, and Python compilation checks passed.

During this repair, the initial test command omitted `PYTHONPATH=code` and produced three test-module import errors; no tests ran in that attempt. The next run had 12 passing tests and one incorrect synthetic reciprocal-rank expectation: `.5/62 + .5/60` is slightly greater than `1/61`. Correcting that hand expectation yielded 13 passing tests. The repaired full verifier then completed on its first execution, with no failed checks. Earlier inherited verifier crashes/timeouts were caused by stale names, wrong schema/population assumptions, and repeated decompression; they were not evidence of successful verification and are not presented as such.

## Reproduction — verification only

```bash
.venv-lightgbm/bin/python -m py_compile code/verify_v6.py code/create_v6_test_features.py code/test_v6_verification.py code/train_v6.py code/v6_data.py
PYTHONPATH=code .venv-lightgbm/bin/python -m unittest -v test_v6 test_v5 test_v6_verification
.venv-lightgbm/bin/python code/create_v6_test_features.py
.venv-lightgbm/bin/python -u code/verify_v6.py
```

Completed report: `results/v6_experiment/verification.json`. No training command was run, no completed feature table was rebuilt, no new submission was generated, and nothing was submitted to Kaggle.
