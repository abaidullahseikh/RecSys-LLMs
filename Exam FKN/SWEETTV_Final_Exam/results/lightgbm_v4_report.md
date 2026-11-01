# LightGBM V4: completed training and experimental submission

## Decision

LightGBM 4.6.0 installed successfully from a prebuilt wheel in `.venv-lightgbm` (Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3). System Python and the original environment were not replaced.

**V2 remains the designated final recommendation.** On the same 256 held-out users and the existing Top-5 qualifying-airing relevance target, LightGBM scored **0.45255208333333335**, versus **0.4609765625** for V2: a decrease of 0.00842448. These are local validation scores, not Kaggle scores. The user reports genuine Kaggle private/public scores of 0.37880/0.40074 for V2 and 0.35282/0.37196 for the earlier CBF-based V3.

The requested V4 was produced as an **experimental LightGBM submission**, not as a demonstrated improvement. No upload was performed and its Kaggle score is unknown.

## Environment and genuine training

Commands run from the project root:

```bash
/home/dastan/.local/bin/python3.12 -m venv .venv-lightgbm
.venv-lightgbm/bin/python -m pip install --only-binary=:all: -r requirements-step10-lightgbm.txt
.venv-lightgbm/bin/python code/train_lightgbm_step10.py
.venv-lightgbm/bin/python code/train_final_submission_v4.py
.venv-lightgbm/bin/python code/verify_submission_v4.py
.venv-lightgbm/bin/python code/test_lightgbm_v4.py
```

The saved Step 10 feature table was reused without rebuilding it: 136,955 candidate rows and 50 numeric model features (plus separate IDs, labels and grouping columns). Candidate generation uses Top-50 unions of the five original generators. Recorded secondary candidate recall is 0.893012 micro / 0.892536 macro among nonempty users, with 108.69 candidates per user on average. Top-20 recall is 0.828112 micro, with 45.98 candidates per user. No user has fewer than five candidates.

Training used Week 11–20 features and Week 21–30 historical labels, with 1,004 supervised training users and 256 disjoint evaluation users. Two leaf-count settings (15/31) were evaluated using deterministic three-fold user grouping inside the 1,004-user set. The final choice was 31 leaves, minimum child samples 100, learning rate 0.05, row/feature sampling 0.8, L2 penalty 1, seed 2026, four CPU threads, binary objective and training-fold-derived class weighting. Early stopping on inner-fold binary logloss selected four rounds (median inner-fold best iteration). Top-50 was retained as the prepared inference candidate limit; Top-20 was diagnostic only. No blend or new hyperparameter search was selected using evaluation-user outcomes.

The stored evaluation model fits 109,113 rows with 4,422 positives. After evaluation, the future-inference model was refit with the same four-round configuration on all 136,955 historical candidate-label rows, including labels through Week 30. For inference, the candidate generators, CBF vocabulary, personal profiles and popularity were refit using permitted history through Week 30; the target schedule supplies availability/metadata only.

The 256 users are held out from supervised fitting and parameter selection in this run. They are **not a pristine previously unseen research holdout**: earlier project-wide generator/hybrid selection used the full 1,260-user local split. This inherited selection optimism and reuse of one temporal window limit the strength of the comparison.

## Same-user comparison

| Method | Primary all-program MAP@5 | Secondary Top-5 MAP@5 |
|---|---:|---:|
| Popularity | 0.744297 | 0.305560 |
| Personal Frequency | 0.777057 | 0.451081 |
| PPR | 0.804440 | 0.443932 |
| Association Rules | 0.748555 | 0.315638 |
| V2 PPR + PF | 0.796693 | **0.460977** |
| LightGBM, four rounds, common cold fallback | **0.817891** | 0.452552 |

The original all-1,260-user V2 sensitivity score of 0.440405 is a different population and is not the baseline for this table. The broad primary target and Top-5 target are different relevance definitions. Neither is claimed to reproduce hidden Kaggle labels. The reported 80% rule remains unconfirmed.

## Feature importance and diagnostics

Top gain features include personal-frequency rank (568,037), PPR rank (160,204), target airing count (70,579), target active weeks (29,743), channel affinity (22,144), target total duration (19,763), historical popularity by watched seconds (15,106), and PPR score (10,699). Gain establishes model usage, not a causal accuracy contribution.

Fixed-configuration held-out diagnostic ablations:

| Feature subset | Secondary MAP@5 | Primary MAP@5 |
|---|---:|---:|
| Full | 0.452552 | 0.817891 |
| Remove CBF score/rank and category/genre affinity | 0.450352 | 0.817773 |
| Remove PPR score/rank | 0.438763 | 0.799883 |

These are signal ablations, not complete generator ablations: candidate sets and source-membership flags still incorporate CBF/PPR. They were not used to select an alternate final configuration. The saved `program_airings_recent_2w/4w/8w` features are zero placeholders: the earlier builder never incremented those counters. They therefore provide no usable recency-frequency signal; they were kept unchanged to reuse the existing table. Actual last-watch recency is populated. V4 retains these placeholders for schema consistency rather than silently changing their meaning.

## Runtime and verification

Latest training/CV/ablations/final refit: **21.873 seconds**, peak RSS **598.25 MiB**. Full-history V4 feature generation and inference: approximately **265.35 seconds**, peak RSS **3,635.86 MiB**. Manifest stage timings overlap; the end-to-end estimate is history load + graph stage + remaining stage, not a sum of every field.

`verify_submission_v4.py` passed **23 recorded checks, zero failures** in 28.26 seconds. It checks exact CSV headers, original string user IDs/order, 1,260 rows, five distinct canonical positive scheduled IDs, original source hashes, all historical interaction time bounds, cold-user fallback, model/feature hashes, saved-model replay of all output rankings, disjoint supervised users, hash-based splits, and both held-out MAP scores using the frozen evaluator. It also confirms V1/V2/V3 hashes unchanged. This is an independent verification routine that replays saved feature rows; it is **not** an independent reconstruction of feature engineering or a second implementation of LightGBM.

Three new regression tests passed: gzip candidate loading with a large string ID; cold-user common fallback; CBF similarity extraction across all candidates. Compilation checks passed.

Earlier attempts failed on gzip parsing, the scikit-learn wrapper dependency, stale variable names, missing V4 constants/imports, channel-preference initialization, candidate rank-map indexing, and CBF matrix orientation. These were corrected before the verified output. An early exploratory 20-round result was superseded: it did not respect the four-round inner-CV choice. Likewise earlier scores without the explicit cold-user fallback are not the final metrics. The final report uses only the final four-round common-fallback run.

## Files and status

- CSV: `submission/final_submission_v4.csv`
- SHA-256: `2e994325a168f30dec88730ace90831867e0f2df17fb4175a35e4bbb8c21bef7`
- Model configuration: `results/final_model_v4_manifest.json`
- Training/CV/importance/ablations: `results/step10_lightgbm/lightgbm_training_result.json`
- Evaluation model: `results/step10_lightgbm/models/lightgbm_top50_binary.txt`
- Full historical refit: `results/step10_lightgbm/models/lightgbm_v4_final_refit_all_week21_30.txt`
- Test features: `results/step10_lightgbm/v4_test_candidate_features.csv.gz`
- Verification: `results/final_submission_v4_verification.json`
- Checksum: `results/final_submission_v4_sha256.txt`

V4 contains 1,260 users and uses a schedule pool of 6,636 IDs. Its Top-50-generator union averages 110.73 candidates per user (139,516 candidate rows). V4 is format-ready for manual experimental Kaggle submission. **V2 remains recommended until stronger evidence, such as a genuine better returned Kaggle score, supports a change.** Original session exports were not edited.
