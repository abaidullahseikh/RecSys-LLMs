# Fast V7 experiment — retain V5

**V5 reproduced exactly at 0.4886328125 local Top-5 MAP@5. Neither new configuration beat it on independent development OOF selection, so no V7 submission was generated.** V5 remains the designated final submission with user-reported genuine Kaggle private MAP@5 **0.39706**. No new Kaggle score or achievement of the 0.44801 aspiration is claimed.

| Experiment | Development OOF MAP@5 (1,004 users) | Outer MAP@5 (256 users) | Final rounds | CV time | Outer fit/predict |
|---|---:|---:|---:|---:|---:|
| E1 — V5 binary baseline | **0.468416** | 0.488633 | 78 | 31.99 s | 2.06 s |
| E2 — LambdaRank | 0.467497 | **0.492005** | 56 | 20.63 s | 1.83 s |
| E3 — binary + repeat streak | 0.468154 | 0.489323 | 79 | 34.33 s | 2.12 s |

**Selection:** maximize development-user OOF MAP@5 using V5's same three user-group folds; outer users are excluded from fitting, stopping and selection. E1 wins. E2's outer gain of 0.003372 and E3's outer gain of 0.000690 do not justify selecting them after viewing the repeatedly examined outer holdout. That outer set is not a pristine new research holdout.

## Precisely what was tested

- The original 136,955 candidate rows and 75 V5 features were reused with source-hash, feature-order, candidate-eligibility and temporal-boundary checks. V5's saved outer predictions and rerun binary baseline matched exactly; its development OOF score also reproduced exactly.
- **E2:** one native LightGBM LambdaRank configuration, correctly contiguous user query groups, binary gains `[0,1]`, `eval_at=[5]`, truncation level 8 (K+3), 15 leaves, minimum leaf data 100, learning rate .05 and L2=5. LambdaRank optimizes a truncated NDCG surrogate; early stopping and selection used actual MAP@5, not a claim of direct MAP optimization.
- **E3:** the requested recent counts, active-week counts and meaningful-view recency already exist. No duplicates were added. The single new feature is **trailing consecutive active weeks** immediately before the cutoff, requiring an airing watched for at least 120 seconds in each week, stopping at the first inactive week (maximum10). It distinguishes consecutive behavior from total active weeks and total recent views.
- Both new configurations use seed2026, at most four LightGBM CPU threads, the same candidates, labels, cold-start popularity order, repeat eligibility and deterministic ties. Maximum160 rounds with patience20 on inner MAP; final rounds are the median of inner best iterations. No hyperparameter grid, blend search or ensemble was run.
- Candidate-generator signals use only permitted history; the new feature uses Weeks11–20 for validation. Target labels are the existing Top5-by-120-second-airing-frequency hypothesis from Weeks21–30. The hidden Kaggle relevance definition remains unconfirmed. No future test viewing was accessed.

## Verification and artifacts

**4 regression tests passed; 51 independent checks passed; zero failures.** Verification reloaded every saved inner/outer model, replayed all rankings, recomputed MAP with rational arithmetic and checked the frozen production evaluator. Every new streak value was independently reconstructed using local Monday-date sets rather than the feature builder's bitmask algorithm. User/query grouping, schema, eligibility, hashes and prior evidence preservation passed. Compilation checks passed.

Total experiment time: **101.53 s**; peak RSS **361.49 MiB**. Independent verification: **13.80 s**. Memory is the whole-process peak, not isolated per-model memory. Python project environment: `.venv-lightgbm`; LightGBM4.6.0. Original feature generation was reused, not independently rebuilt.

Evidence in this directory: `plan.json`, `v5_reproduction.json`, `repeat_feature.json`, `selection.json`, `experiment.json`, all inner/outer models and predictions, and `verification.json`. Original V1–V6 result files, submissions and exported sessions match the before-run hashes. **No `submission/final_submission_v7_experimental.csv` exists, and no upload was attempted.**

Commands used:
```bash
.venv-lightgbm/bin/python code/test_v7.py
.venv-lightgbm/bin/python -u code/run_v7.py
.venv-lightgbm/bin/python -u code/verify_v7.py
```
Creation scripts protect completed artifacts. Reproduction requires a fresh output location rather than overwriting evidence.
