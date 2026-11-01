# V9 — targeted PPR calibration and experimental submission

## Result and recommendation

**V9 was trained, generated and verified. Keep V8 as the designated genuinely scored submission.** The user reports V8 private/public MAP@5 **0.39908 / 0.41811**, versus V5 private **0.39706**. These are user-supplied Kaggle results, not locally generated scores. V9 has not been uploaded and has no Kaggle score; the 0.44801 target remains aspirational.

**Experimental CSV:** `submission/final_submission_v9_experimental.csv`  
**SHA-256:** `8576b1739333eab9cab2d84d2b157492f3224c82f148cc535366a52af05694df`

The selected pure new binary model scored slightly **worse on development validation** but slightly better on assessment. Per the mandatory instruction, that development regression did not block a valid experimental CSV. No existing model or submission was relabelled as V9.

## 1. Exact baseline reproduction

Read all of `exam_quiz_FKN.md`, the actual V5–V8 source, V8 summary/configuration and its successful verification record. Project root is `SWEETTV_Final_Exam`; original CSVs remain at its sibling `sweettv-tv-program-recommender/`.

The actual scored V8 file matches its saved SHA-256 `62c8b3f41f6c4e6966bcdd605a10d61c1e4d977c5888c9d5217a7eba338e6f5a`. Its model is `results/v8_experiment/final_model.txt`, selected configuration **E2_streak**, with76 ordered features. Historical features and candidate signals remain as recorded: Weeks11–20 before the Weeks21–30 target; final candidate context refit through Week30; no test viewing behavior.

Replayed all three development models and the assessment model, then the final model and all1,260 submission rankings. **All saved development/all-crossfit/final predictions match exactly.** V8 development Top5-120 MAP@5 is **0.4096515850636589** over3,901 users; assessment is **0.40863575268817204** over992 users. These broader-population metrics must not be confused with V5's historical256-user local score of0.488633 or Kaggle private scores.

Candidate sets, historical label definition (Top5 by >=120-second qualifying-airing frequency), string user IDs, feature order, cold fallback, deterministic score/popularity/ID ranking, and the AP denominator `min(5, |R|)` were preserved. The 80% rule remains unverified as official; no target definition was changed in V9.

## 2. Full-ranking error analysis — completed before model changes

Saved `ranking_errors_by_user.csv` identifies, for **each of4,893 users**, relevant IDs in ranks1–5,6–10,11–20,below20, missing candidate IDs, and false Top5 promotions. Predictions are cross-fitted: development models exclude the user's labels; assessment models fit only development users.

| All-user error component | Relevant programs |
|---|---:|
| Ranks1–5 | 11,566 |
| Ranks6–10 | 3,699 |
| Ranks11–20 | 2,767 |
| Reachable but below20 | 2,231 |
| Missing from candidates | 3,568 |
| Total relevant labels | 23,831 |

All-user micro candidate recall is **85.03%**; Recall@5/10/20 is **48.53% / 64.06% / 75.67%**. **42.92% of reachable true programs rank below five.** Development-only figures are nearly the same: candidate recall85.03%,42.87% of reachable labels below five. There are12,899 false Top5 slots. Candidate omission and reranking are separate constraints; the one targeted feature cannot recover absent candidates.

### User groups (all-user crossfit)

| Group | Users | MAP@5 | Candidate recall | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| All | 4,893 | 0.409446 | 0.8503 | 0.4853 | 0.6406 | 0.7567 |
| Template | 1,260 | 0.469331 | 0.8930 | 0.5399 | 0.7025 | 0.8122 |
| Heavy:101+ programs | 3,118 | 0.473479 | 0.8987 | 0.5399 | 0.7084 | 0.8212 |
| Light:1–20 | 227 | 0.130283 | 0.6317 | 0.2179 | 0.3392 | 0.4604 |
| Strong repeat:airings/program>=3 | 498 | 0.551486 | 0.9419 | 0.6144 | 0.7941 | 0.8907 |
| Weak repeat:<3 (includes cold) | 4,395 | 0.393351 | 0.8398 | 0.4706 | 0.6230 | 0.7414 |
| Cold | 56 | 0.166012 | 0.6008 | 0.2379 | 0.3629 | 0.4315 |

Group definitions use historical behavior only. Light/heavy are by distinct qualifying training programs, not held-out activity.

### Item groups and signal preferences

- Previously watched relevant candidates:17,934 reachable;11,440 rank in Top5 (63.79%). Previously unseen:2,329 reachable;only126 rank in Top5 (5.41%). There is no seen-item exclusion policy; this is a score tendency.
- Popular programs (historical popularity rank1–50):16,797 relevant candidates;10,223 Top5. Rank51+ programs:3,466 reachable relevant candidates;1,343 Top5.
- False Top5 promotions include12,119 watched items and780 unseen items;10,788 popular items and2,111 less-popular items. Popularity/seen status alone is not evidence of an error: many true Top5s also belong to these groups.
- Mean within-user Spearman associations of model scores with historical frequency / recency / popularity / target airing opportunities are **0.485 / 0.419 / 0.728 / 0.788**. PPR candidate order correlates0.871 with popularity; Association Rules order0.970. Personal Frequency order correlates0.889 with historical frequency. CBF does not favor these same signals (negative mean associations in the saved analysis).
- False Top5 promotions average3,026 historical viewers, versus2,654 for true ranks6–10; PPR mass averages0.02710 versus0.01660. This motivates examining personal support relative to global support, but does **not** prove a causal popularity-bias effect.

Ranked failure priorities: **(1)** reachable true items stuck at6–20/below5; **(2)** weak-history/unseen-program under-ranking; **(3)** missing candidates, especially light/cold users; **(4)** inherited time-scope/population shift; **(5)** uncertain local-to-Kaggle relevance/period transfer. Reranking cannot solve all of these.

## 3. Distribution shift

`feature_distribution_shift.csv` contains min/max/mean/std, missing/zero rates,median/p95 and out-of-development-range fraction for every feature across development,assessment,matched template validation and final inference. Interpret population and time changes separately.

| Feature mean | Matched-template validation | Final inference | Interpretation |
|---|---:|---:|---|
| Full-history personal frequency | 2.4583 | 3.7652 | Candidate signal uses10-week history in validation,20 weeks at inference |
| Trailing-window user-program airings | 2.4583 | 1.4472 | Same10-week behavioral window; real time/candidate changes |
| User history program count (candidate-row weighted) | 187.26 | 156.64 | Rolling behavior differs over time, despite longer full-history generator scope |
| User history airing total | 403.36 | 285.77 | Rolling-week behavior change |
| Recency days (71 sentinel for missing) | 48.28 | 53.45 | More stale/missing recent interactions; explicit sentinel not NaN |
| Global distinct viewers | 1,373.11 | 1,652.55 |20-week support versus10-week support plus candidate changes |
| Normalized target airings | 203.81 | 206.07 | Mean close after horizon scaling; p95 changes801 to1,049.17 |
| PPR score | 0.005430 | 0.005406 | Stable marginal mass does not guarantee identical personalized ranks |
| Channel affinity | 0.07790 | 0.06647 | Historical preference/target-channel distribution changes |
| Completion80 share | 0.06794 | 0.07673 | Completion patterns differ |
| Recent2-week count | 0.42213 | 0.41467 | Similar recent-intensity marginal distribution |

Mean candidates per user are110.26 for broader validation,108.69 for template validation,110.73 for final inference. CBF similarity mean is0.26124/0.25677 and AR score0.13107/0.11853 for template validation/final inference. Missing-value rate is zero in the supplied numeric caches; zero/sentinel rates remain separately recorded.

These are observed shifts, not newly proven implementation bugs. Some arise from legitimate later schedules/behavior, others from V8's documented mix of full-history generator signals and rolling10-week features. A raw full-period popularity denominator can therefore have a different scale even when personalized PPR mass looks stable. No final-period user viewing was used to measure the shift—only saved historical features and target schedule metadata.

## 4. One targeted change and controlled comparisons

**E2 adds one PPR calibration feature**:

`ppr_support_adjusted_score = ppr_score × sqrt(historical_qualifying_users / (program_distinct_historical_viewers + 20))`

The fixed square-root scaling and20-viewer smoothing are prespecified, not tuned. Denominators use4,837 qualifying historical users in validation and4,891 through Week30 for final inference. Smoothing avoids exploding rare/zero-support values. The model may learn whether that signal is useful; it does not blanket-penalize all popular items or force diversity.

All76 V8 features, candidate sets,labels,user splits and LightGBM settings are unchanged:binary objective,15 leaves,minimum leaf data100,learning rate.05,L2=5,78 rounds,seed2026,deterministic column-wise fitting,at most4 threads. E2 has77 selected features. One model configuration was fit across the three existing OOF folds,then assessment and final all-user refit.

The alternative target/weighting experiment **E3 was skipped**: the ranking audit did not establish incorrectly labeled positives. Introducing label/weight changes after examining hidden aggregate scores would not be justified.

E2 changes44.12% of development Top5 lists, satisfying the predeclared>10% gate for **one RRF fusion family**, weights.50/.75 new model againstV8;constant60. Raw probabilities were not averaged.

| Configuration | Development OOF MAP@5 (3,901) | Assessment MAP@5 (992) |
|---|---:|---:|
| E1 original V8 (exact reproduction) | **0.4096515851** | 0.4086357527 |
| **E2 pure new model (selected V9)** | **0.4094794070** | 0.4105012321 |
| E4 RRF:new model weight.50 | 0.4094537725 | **0.4106154794** |
| E4 RRF:new model weight.75 | 0.4094324105 | 0.4104743504 |

Best valid **new** configuration is pureE2 by development MAP. Development delta vsV8 is **−0.0001721781**;assessment delta is **+0.0018654794**. E4's higher assessment score was not used to override selection. No proven local improvement is claimed. Assessment users were excluded from supervised fitting/tuning,but were included in descriptive error analysis and had been examined during earlier project experiments; this is not pristine holdout evidence.

The calibrated feature had gain21,437 over42 splits (assessment model),behind the existing frequency/PPR signals. This confirms model use,not causal accuracy benefit. Gain and split counts are in `feature_importance.csv`.

## 5. Final training, CSV and verification

The selected new model was refit on all539,506 historical candidate examples from4,893 users. Feature history precedes the Weeks21–30 labels; those labels are permitted history for future Weeks31–42 inference. Final candidate context reuses V8's139,516 rows with generators/profiles fitted through Week30; only the same new calibration formula is applied. Future schedules are metadata/availability only. No test user viewing or hidden label was accessed.

Verified CSV:exactly1,260 original string IDs,original order,columns `user_id,tv_show_id`,five distinct canonical positive IDs per user,all in the6,636-ID named Weeks31–42 schedule pool. Cold users retain the common historical popularity fallback. Previously watched programs remain eligible. Every ranking is deterministic and replayed from the saved model/features.

- **144 regression tests passed**,including3 focused new calibration/fusion/tie/cold-ID tests;0 failures/errors in the final suite.
- **36 validation checks passed**,followed by **45 final checks passed**,0 failures. These independently replayed all new OOF/assessment models and MAP values,original V8 score caches,all-user recall metrics,final feature arithmetic/model order and1,260 submission rankings. Deterministic inference was repeated successfully.
- Prior models,submissions,source files,original datasets and session exports match pre-run hashes. No source or report for V8 was edited to reflect its newly reported Kaggle score;that score is recorded only in V9's new evidence.
- One audit-writing attempt failed before training because a per-user missing-label count was overwritten by a string field with the same key. It was renamed `missing_program_ids`;baseline score files were checked identical and reused on resume. No feature/model or validation result was changed by this report repair. Final tests/verification did not fail.
- Scope:original V8 features/generators are reused and their hashes checked,not independently reimplemented. Hidden Kaggle label rules and causality remain unverified.

| Stage | Seconds | Peak RSS |
|---|---:|---:|
| Successful reproduction/error/shift audit | 72.08 |557.25 MiB |
| New-model CV,small fusion comparison,assessment and refit |36.54 |913.30 MiB |
| Final augmentation and CSV |13.80 |211.28 MiB |
| Independent final verification |40.81 |not separately measured |

## Files and reproducibility

- CSV:`submission/final_submission_v9_experimental.csv`
- SHA-256:`8576b1739333eab9cab2d84d2b157492f3224c82f148cc535366a52af05694df`
- Model/config:`results/v9_experiment/final_model.txt`, `plan.json`, `selection.json`, `experiment.json`, `submission_manifest.json`
- Analysis:`ranking_errors_by_user.csv`, `ranking_error_groups.csv`, `ranking_analysis.json`, `feature_distribution_shift.csv`
- Results:`comparison.csv`, `feature_importance.csv`, `submission_verification.json`, `submission_sha256.txt`, `regression_tests.json`

```bash
.venv-lightgbm/bin/python code/analyze_v9.py
.venv-lightgbm/bin/python code/train_v9.py
.venv-lightgbm/bin/python code/verify_v9.py
.venv-lightgbm/bin/python code/generate_v9.py
.venv-lightgbm/bin/python code/verify_v9.py
.venv-lightgbm/bin/python code/summarize_v9.py
```

Creation commands protect completed outputs;reproduce into a fresh explicit directory rather than deleting evidence. No automatic upload was attempted. V9 is ready for manual **experimental** upload,not a demonstrated improvement over V8. Remaining weaknesses are limited candidate recall,unseen/light-user ranking,shared temporal-window assumptions and the uncertain transfer from our local Top5 labels to hidden Kaggle preferences.
