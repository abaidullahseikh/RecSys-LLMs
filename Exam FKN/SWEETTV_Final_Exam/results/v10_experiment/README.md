# V10: seen/unseen calibration and conditional pairwise experiment

## Outcome

Generated `submission/final_submission_v10_experimental.csv`. It is a genuine new
LambdaRank prediction file, but regresses locally. **V8 remains the recommended
final submission**, with user-reported private Kaggle MAP 0.39908. V10 has not been
uploaded or Kaggle-scored.

| Strategy | Development MAP@5 (3,901 users) | Assessment MAP@5 (992 users) |
| --- | ---: | ---: |
| V8 binary baseline | 0.4096515851 | 0.4086357527 |
| V9 binary baseline | 0.4094794070 | 0.4105012321 |
| V9 + selected global correction (0) | 0.4094794070 | 0.4105012321 |
| V9 + selected light-viewer correction (0) | 0.4094794070 | 0.4105012321 |
| V10 conditional LambdaRank | 0.4084981771 | 0.4091364247 |

V10 development delta versus V9: **-0.0009812299**. Assessment delta versus V9:
**-0.0013648073**. The assessment population has been examined previously; it is
not an untouched holdout. Its metrics did not select calibration coefficients or
pairwise settings. The descriptive seen/unseen audit also included those users.

## Calibration and fallback

Both families tested corrections `[-0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04]`
added to V9's native probability scores for historically unseen candidates:

- Global: all noncold users.
- Light viewers: only users with 1–20 distinct qualifying historical programs.

Cold users always retain the common popularity fallback. Ranking ties use
popularity rank, then numeric program ID. Zero was included in each family;
ties prefer zero and then the smallest absolute correction. **Both families
selected zero on development MAP.** No arbitrary unseen bonus was forced.

The conditional fallback was one modest LambdaRank configuration on the same
broader-population V9 features: 77 features, 78 boosting rounds, 15 leaves,
learning rate 0.05, minimum 100 rows per leaf, L2 5, binary gains `[0, 1]`,
truncation level 8, seed 2026, and four threads. Three inner fits produced
development cross-fitted scores, one development-only fit produced assessment
scores, and one all-population fit produced final inference scores. No parameter
search was added. The configuration and feature context differ from the older
template-only V7 ranker.

This fallback supplies the required new experimental V10 artifact; its weaker
development result is recorded rather than presented as an improvement. It
changes 899 of 1,260 final user lists relative to V9.

## Seen/unseen findings

Seen status was reconstructed from exact string user/program IDs in the
checksum-verified >=120-second interaction tables. Validation uses Weeks 11–20
history; final inference uses history through Week 30. Future user viewing is
not used. Target broadcast metadata is used for availability and candidates.

The V9 baseline audit over all 4,893 evaluation users found:

| Quantity | Seen | Unseen |
| --- | ---: | ---: |
| Candidate pairs | 245,604 | 293,902 |
| Relevant candidate pairs | 17,934 | 2,329 |
| All true labels, including absent candidates | 18,057 | 5,774 |
| True labels in Top-5 | 11,458 | 126 |
| True labels ranked 6–10 | 3,414 | 239 |
| True labels ranked 11–20 | 2,061 | 746 |
| True labels ranked below 20 | 1,001 | 1,218 |
| True labels missing from candidate pool | 123 | 3,445 |

Thus **59.66% of all true unseen labels are absent from the candidate pool**.
Only **2.18% of all true unseen labels**, or **5.41% of reachable true unseen
labels**, reach Top-5. Missing candidate recall and poor ranking of reachable
unseen items both contribute. Calibration alone cannot recover missing items.

`seen_unseen_audit.json` contains conditional candidate relevance rates, Top-5
precision, score/feature distributions, history-size/repetition groups, and
wrong-seen/true-unseen co-occurrence. Co-occurrence does not prove that swapping
items would improve MAP. Full-label counts, independently reconstructed from
historical pairs, are in `submission_verification.json` under
`details.seen_unseen_all_label_counts`.

## Verification and reproducibility

- 28 initial validation verification checks passed.
- All 38 final verification checks passed, with zero failures.
- Four targeted regression tests passed: zero correction, signed corrections
  and light-viewer gating, cold fallback, deterministic ties/exact large IDs.
- V8/V9 cross-fitted predictions and local baseline metrics reproduced exactly.
- V9 final model scores and original final CSV reproduced exactly.
- All 14 calibration-grid results were independently recomputed.
- V10 validation and final predictions replayed from saved models; local MAP
  also agrees with the production evaluator.
- Final CSV has the exact header, 1,260 unique users in template order, and five
  distinct positive canonical program IDs per row, all in the 6,636-ID schedule
  pool.
- Protected V1–V9 results, code, submissions, source CSVs, and authentic evidence
  exports match the pre-experiment hash snapshot.
- Feature generation was reused from the verified V8/V9 caches; it was not
  independently reimplemented in this verification.

Final CSV SHA-256:

```text
39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e
```

The experiment run took about 90 seconds and peaked at about 962 MiB RSS;
generation took about 23 seconds and 422 MiB RSS. Verification was separate.

Commands, run from `SWEETTV_Final_Exam`:

```bash
.venv-lightgbm/bin/python -u code/run_v10.py
.venv-lightgbm/bin/python -u code/verify_v10.py
.venv-lightgbm/bin/python -m unittest discover -s code -p 'test_v10.py' -v
.venv-lightgbm/bin/python -u code/generate_v10.py
.venv-lightgbm/bin/python -u code/verify_v10.py
```

The completed experiment and generator reject overwriting their final outputs.
The verifier can be rerun and writes a separately named recheck report.

## Attempt history

The first audit-writing attempt stopped when sparse per-user counters produced
inconsistent CSV field names. Its partial eight-row CSV is preserved as
`seen_unseen_errors_by_user_failed_attempt1.csv`. The fix explicitly initializes
all four seen/unseen slot/hit columns. Resume compared the original preservation
snapshot, baseline evidence, historical flags, and baseline scores before
continuing; it did not overwrite the failed CSV or prior-version artifacts.

Primary evidence: `experiment.json`, `plan.json`, `selection.json`,
`development_calibration_grid.csv`, `seen_unseen_audit.json`,
`submission_manifest.json`, `validation_verification.json`,
`submission_verification.json`, and `preservation_before.json`.
