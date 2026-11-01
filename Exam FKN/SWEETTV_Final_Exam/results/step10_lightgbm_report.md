# Step 10 — Leakage Audit, Candidate Generation, and LightGBM Attempt

## Executive result

The existing pipeline and prior submissions were preserved. The Step 10 audit and candidate/feature preparation completed and independently passed. A bounded LightGBM reranker could **not** be trained because LightGBM is unavailable in the project and system Python environments; installing it from PyPI timed out and no wheel/source was cached. The trainer recorded that status without fitting a model or inventing scores.

V2 remains the recommended final submission. Its supplied Kaggle scores are private **0.37880** and public **0.40074** (user-reported). The pre-existing CBF-only `submission/final_submission_v3.csv` remains untouched. **No new LightGBM V3 was generated**, and no Kaggle submission was made.

## Official requirements and relevance audit

Read `exam_quiz_FKN.md` and the authoritative `../LLM4Rec. Midterm Quiz.pdf`. The official Kaggle CompetitionService record for competition 24201 confirms **MAP@{K}**, higher is better. The exam fixes K=5. The Midterm Quiz's official Kaggle overview screenshot describes predicting each user's five most-preferred programs from weeks 31–42 viewing history; its Data-tab screenshot identifies `tv_show_id=0` as irrelevant airtime. Neither the accessible official record nor the course documents specifies how hidden preferences are aggregated or gives an 80%-of-airing-duration threshold.

The Kaggle Data API returned HTTP 401 in this environment. The exam's embedded original Data-tab screenshot was reviewed instead. The 80% rule remains an **unverified reported rule**, not an official rule. It was retained only as a separate validation sensitivity definition. The Step 3 primary protocol and historical Step 2 **120-second per-airing** interaction definition remain unchanged.

The data/pipeline audit checked these existing implementation properties against source code and stored Step 2–9 verification evidence:

- Same-channel, half-open viewing/airing overlap; overlapping/touching view fragments are unioned, exact duplicate events are removed, and fragments for one user/broadcast are accumulated before thresholding.
- Actual UTC timestamps determine training/validation/test windows; schedule filenames do not assign events to weeks.
- ID zero is unlabelled airtime and excluded from interactions and candidates. `user_id` remains a string; numeric program IDs are canonicalized exactly without float conversion.
- Validation uses the frozen 6,384-item catalog; the test pool is built from the named Weeks 31–42 schedule.
- Repeat consumption is permitted; within-list duplicates are rejected. Cold-start completion uses the shared popularity order.
- MAP@5 follows Step 3: per-user AP denominator `min(5, |R|)`, macro-average over all 1,260 users, and local empty truth contributes zero. The exact hidden Kaggle label construction remains undocumented.
- Training feature profiles use Week 11–20 only. The test schedule contributes availability/metadata, never user viewing behavior.

Details and official-source evidence are in `results/step10_official_audit.json`. V1/V2 and the existing CBF V3 hashes were rechecked and remain unchanged. The original OpenCode session export was not modified.

## Candidate generation and recall

For each user, candidates are the union of Top-K lists from Personal Frequency, PPR, Association Rules, CBF, and global popularity. Candidate limits were K=20 and K=50 per generator; the union is commonly larger than K because it combines five sources. All candidate lists use only the frozen validation schedule pool and training-fitted model signals. No user has fewer than five candidates.

| Generator limit | Mean union candidates | Median | P95 | Users with <5 | Secondary Top-5 recall (macro / micro) | Users with all five labels covered |
|---:|---:|---:|---:|---:|---:|---:|
| Top-20 | 45.98 | 45 | 57 | 0 | 0.8277 / 0.8281 | 631 |
| Top-50 | 108.69 | 107 | 138 | 0 | 0.8925 / 0.8930 | 823 |

The broad primary target includes every distinct qualifying validation program, so union recall is lower: Top-20 macro/micro 0.1638/0.1279; Top-50 0.2422/0.1994. Under the unverified 80%-fraction Top-5 sensitivity labels, micro recall is 0.7336/0.8175. These are candidate recall figures, not MAP scores.

The feature pipeline wrote `results/step10_lightgbm/candidate_features_top50.csv.gz`: **136,955 user-program rows and 50 columns**, including labels and a deterministic, disjoint user-group split (1,004 training users / 256 evaluation users). `verify_step10_candidates.py` independently checked the candidate sets, split, labels, canonical IDs, and recalls in **10 checks, zero failures**.

## Historical feature table

Feature definitions and source boundaries are recorded in `results/step10_lightgbm/candidate_build_manifest.json`. Features cover:

- User-program qualifying airing count, watched seconds, recency, logged frequency, active weeks, recent 2/4/8-week counts, and airing counts at 120/300/600 seconds and 80% of broadcast duration.
- Program schedule availability in the target period: airing count, total/mean/max duration, channel count, weeks, hours, and hour entropy.
- User channel and viewing-hour affinity; category/genre preference match; cold-start/history-size indicators.
- PPR score/rank, Personal Frequency score/rank, Association Rules score/rank, CBF similarity/rank, popularity scores/rank, and generator-membership flags.

The feature rows are built with Week 11–20 interaction/profile statistics. Validation and test schedules supply only schedule metadata/availability features. `user_id` is stored as text and excluded from numeric model features. The supervised label column is explicitly the per-user Top-5 qualifying-airing validation sensitivity target; an additional all-program primary label and separate 80%-fraction sensitivity label are stored for auditing. The assigned evaluation-user group is disjoint from the candidate training-user group.

## LightGBM attempt and model comparison

`code/train_lightgbm_step10.py` contains a bounded binary-objective reranking design: two small leaf-count settings, deterministic three-fold user-group CV inside the training-user groups, early stopping, Top-20/Top-50 evaluation, CBF/PPR ablations, and feature-importance output. However, the LightGBM import guard found no installation. `.venv/bin/python3 -m pip install lightgbm` failed after PyPI connection timeouts; `lightgbm`, scikit-learn, XGBoost, and CatBoost were absent in both checked Python environments. Therefore **no LightGBM fit, CV score, held-out MAP@5, ablation, feature importance, or LightGBM hybrid result exists**. `results/step10_lightgbm/lightgbm_training_result.json` records the dependency blocker.

| Method | Primary MAP@5 (all 1,260) | Secondary Top-5 MAP@5 | 80%-fraction Top-5 sensitivity MAP@5 |
|---|---:|---:|---:|
| Popularity, distinct viewers | 0.768418 | 0.306627 | 0.266837 |
| Personal Frequency, qualifying airings | 0.787627 | 0.434132 | 0.344587 |
| PPR, cosine/frequency, α=.35, k=30 | 0.815151 | 0.429489 | 0.354977 |
| Association Rules, 14-day confidence × log-lift | 0.771331 | 0.313643 | 0.273966 |
| CBF, channel-context/airing frequency | 0.495029 | 0.200193 | 0.179984 |
| V2, PPR + Personal Frequency RRF .50/.50 | 0.807598 | **0.440405** | 0.352790 |
| LightGBM reranker | Not measured | Not measured | Not measured |

These existing scores are under the same frozen 120-second Step 3 protocol and the separately named Step 9 sensitivity targets. They are not Kaggle private scores. `results/step10_lightgbm/model_comparison.csv` preserves the comparison and marks LightGBM as unavailable/not trained. Since no LightGBM results exist, no claim of improvement or new V3 is made.

## Runtime, memory, verification, and remaining work

- Candidate/feature build: **217.41 seconds**, peak RSS **2,317.73 MiB**.
- Candidate feature file: 136,955 rows; 50 columns.
- Automated tests run: Step 2 26, Step 3 34, Step 4 10, Step 5 22, Step 6 14, Step 8 4, CBF 4 — **114 passed across the suites**.
- Step 10 candidate/feature verification: **10 passed, zero failed**.
- LightGBM: dependency attempt failed; no training, CV, ablation, or model verification was possible.
- No new V3 or later version was produced. The existing CBF V3 remains unchanged; V2 remains recommended.

Reproduction commands:

```bash
sh tests/run_step2_tests.sh
sh tests/run_step3_tests.sh
sh tests/run_step4_tests.sh
sh tests/run_step5_tests.sh
sh tests/run_step6_tests.sh
.venv/bin/python3 code/test_step8.py
.venv/bin/python3 code/test_content_based.py
.venv/bin/python3 code/build_step10_candidates.py
.venv/bin/python3 code/verify_step10_candidates.py
.venv/bin/python3 code/train_lightgbm_step10.py
```

Remaining exam work: make LightGBM available, run the grouped temporal/user-safe reranker experiment and ablations, consider a new submission only if validation evidence supports it, submit manually to obtain a genuine Kaggle score/screenshot if desired, and complete the course PDF deliverable. No new Kaggle score is claimed here.
