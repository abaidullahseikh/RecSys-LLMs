# V8 — broader validation, repeat-feature ablation and count regression

## Outcome

**A genuine new V8 experimental CSV was generated and verified, as requested, despite a small development-validation regression. V5 remains the designated final submission.** Its user-reported genuine private MAP@5 is **0.39706**. V8 has not been uploaded and has no Kaggle score. The 0.44801 private-score aspiration is not claimed as achieved.

- **CSV:** `submission/final_submission_v8_experimental.csv`
- **SHA-256:** `62c8b3f41f6c4e6966bcdd605a10d61c1e4d977c5888c9d5217a7eba338e6f5a`
- **Selected new configuration:** `E2_streak`, binary LightGBM with 75 original features plus one >=80%-completion active-week-streak feature.
- **Final verification:** **90 checks passed, 0 failed**. **141 regression tests passed, 0 failed**. No old training run, model, feature file, submission, original dataset or native session export was modified.

## Official labels and V5 reproduction

The official Kaggle Data URL returned only a JavaScript shell. The public CompetitionService record confirms **MAP@{K}**, higher is better; the exam fixes K=5. Retrieved official material does **not** confirm `screen_time >= .80`, frequency-ranked hidden Top-5 construction, or the hidden frequency tie-break. The 80% rule is **unverified**, not a corrected official target. No other student's report was used as official proof.

Sources: [Kaggle Data](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), [official competition metadata](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionId=24201), and the completely reread `exam_quiz_FKN.md`. Evidence: `official_labels.json`.

Held-out counts were independently reconstructed from verified same-channel user/airing overlap caches. They aggregate unioned fragments per airing before applying either >=120 seconds or >=80% of the **full broadcast duration**. Zero program IDs are excluded. The 80% alternative ranks by qualifying-airing count and uses numeric program ID for ties; that tie-break is our convention.

**V5 replay is exact:** original model, 75 feature columns, 256 original outer users, frozen candidates, original ordering, original labels, MAP convention and saved predictions all agree. Historical V5 Top5-120 MAP@5 remains **0.4886328125**. On the **different, unverified Top5-80 target**, the same V5 predictions score **0.3787109375**. This is not a corrected official score or an improvement comparison.

| Label-definition comparison | Users | Users whose Top-5 set changes | Added 80%-only labels | Removed 120s-only labels | Symmetric user/program differences |
|---|---:|---:|---:|---:|---:|
| Historical V5 outer group | 256 | 238 | 481 | 515 | 996 |
| Entire template cohort | 1,260 | 1,146 | 2,252 | 2,472 | 4,724 |
| Broader V8 population | 4,893 | 4,400 | 8,071 | 10,213 | 18,284 |

These are **alternative-definition differences**, not confirmed label-construction bugs. The old 120-second relevance/interaction artifacts remain untouched. Earlier audits/tests and this run found no new confirmed matching or label implementation defect requiring a historical-baseline correction. Whole-source count and new-feature scans were independently verified; raw matching and union logic were covered by existing second-by-second oracle and integration regression tests, not a newly repeated all-user raw-log join.

## Broader population and leakage controls

Calculated from this project's own data:

- 4,837 users have qualifying training history (Weeks11–20).
- 4,804 have qualifying held-out outcomes (Weeks21–30).
- **4,750 have both**; their metrics are reported separately.
- The history/outcome union is 4,891. Including the two template users absent from both yields **4,893 users**.
- This union includes 56 cold users and 89 empty-target users; they are not discarded to raise MAP.
- **3,901 development users** and **992 assessment users**, assigned deterministically by SHA256(user_id) modulo5; development has three user-grouped OOF folds. A user's target outcomes are never used to fit the model making that user's cross-fitted prediction.

The candidate models use only Weeks11–20 behavior during validation. Target schedules provide metadata/availability. Supervised models are trained on other users' historical Weeks21–30 labels. The graph can use all users' **pre-cutoff** history; that is not future-label leakage. Neither user nor program ID is a numeric model feature. Identity arrays stay strings throughout.

The all-user crossfit report combines development OOF predictions (fit on two-thirds of development users) and assessment predictions (fit on all development users). This is a valid no-own-label evaluation but the fitting sample sizes differ; it is not a single-model in-sample score. The template assessment subset is the previously examined 256-user population and is **not pristine**. The broader nontemplate assessment group was not used for V5–V7 tuning. No assessment results selected V8.

## Features and candidates

The feature audit is `feature_audit.csv`. Historical frequency, recent2/4/8-week frequency, active weeks, weekly consistency, recency, completion50/80/95, qualifying-airing counts, catch rate, channel/hour affinity, stickiness, history-confidence proxies and future airing opportunities **already exist**. They were not duplicated. Duplicate-valued original consistency columns were retained for baseline-schema reproducibility rather than silently changed. Catch rate uses total historical scheduled opportunities; it is not a user-online-availability estimate. Original template rows retain V5's explicit mix of full-history candidate signals and rolling behavioral features.

Exactly two new feature signals were tested:

1. **`completion80_active_week_streak`:** number of consecutive immediately preceding completed local calendar weeks with an airing watched >=80%, stopping at the first inactive week, maximum10. Unlike V7's streak, this specifically requires near-complete airings rather than any >=120s watch.
2. **`recent2w_personal_airing_share`:** a program's qualifying >=120s airing count in the latest two completed weeks divided by the user's qualifying airing count for **all programs in those two weeks**. Unlike V5's `personal_airing_share`, both numerator and denominator are recent, not whole-ten-week totals.

Both use only preceding historical behavior. The final new signals use Weeks21–30; the saved 75-feature final candidate context was refit through Week30 by V5 and reused with verified hashes. Vocabulary/IDF is historical; later schedules supply candidate metadata only. No future test viewing is read.

Five-generator Top-50 unions (PPR, Personal Frequency, Association Rules, CBF, Popularity) were used without target-label-forced positives. Template candidate rows and original75 features were reused **exactly** from V5. Nontemplate rows use the same implemented generators and feature definitions. The resulting table contains **539,506 rows × 77 columns**.

| Population | Users | Mean candidates | Micro Top5-120 recall | Users with <5 candidates |
|---|---:|---:|---:|---:|
| All | 4,893 | 110.26 | 0.850279 | 0 |
| Development | 3,901 | 110.14 | 0.850345 | 0 |
| Assessment | 992 | 110.72 | 0.850021 | 0 |
| Template | 1,260 | 108.69 | 0.893012 | 0 |

## Controlled experiments and ablation

No hyperparameter search. Every configuration uses 15 leaves, minimum leaf data100, learning rate .05, L2=5, 78 rounds, seed2026, deterministic column-wise fitting, at most four threads, and no row/feature sampling. These are V5's selected binary model settings, held fixed for all comparisons. Binary targets are the original Top5-120 relevance; regression predicts `log1p(actual qualifying-airing count)` for every candidate, including zeros. Log scaling dampens extreme broadcast counts and preserves monotone ranking of target counts.

**Important:** E1 below is the original **V5 binary design refitted under the broader grouped protocol**, not the original template-trained V5 model relabelled. Historical V5 replay is reported separately above. Only broader E1/E2/E3 scores are directly comparable in this table.

| Configuration | Features | Development OOF MAP@5 (3,901) | Assessment MAP@5 (992) | All-user crossfit MAP@5 (4,893) | Crossfit Top5-80 MAP@5 |
|---|---:|---:|---:|---:|---:|
| E1 original binary design | 75 | **0.409803** | 0.409022 | **0.409645** | 0.334149 |
| **E2 completion-streak only (selected new model)** | 76 | **0.409652** | 0.408636 | 0.409446 | 0.334279 |
| E2 recent-share only | 76 | 0.409020 | 0.410198 | 0.409259 | **0.334490** |
| E2 both signals | 77 | 0.408776 | **0.411380** | 0.409304 | 0.334374 |
| E3 log-count regression | 75 | 0.398465 | 0.401663 | 0.399113 | 0.317952 |

The feature ablation does **not** support an accuracy-improvement claim: neither feature nor their combination beats E1 development MAP. Count regression is also worse. The best valid **new** configuration by the predeclared development rule is E2_streak; E1 was ineligible for experimental relabelling. Its development delta is **−0.00015124** and its assessment delta is **−0.00038642**. The model was selected before assessment scoring, despite E2_both's later higher assessment value.

### Broader selected-model strata (all-user crossfit)

| Stratum | Users | E1 MAP@5 | Selected E2 MAP@5 |
|---|---:|---:|---:|
| Both qualifying history and held-out outcomes | 4,750 | 0.420020 | 0.419815 |
| Template users | 1,260 | 0.469659 | 0.469331 |
| Nontemplate users | 3,633 | 0.388830 | 0.388676 |
| Heavy: 101+ historical programs | 3,118 | 0.473938 | 0.473479 |
| Light: 1–20 | 227 | 0.130518 | 0.130283 |
| Weak: 1–5 (subset of light) | 47 | 0.128706 | 0.129415 |
| Cold: 0 | 56 | 0.166012 | 0.166012 |
| Empty held-out labels | 89 | 0.000000 | 0.000000 |

Within the old 256 template assessment users, broader E1 scores **0.491497** and selected E2 **0.490768**; original V5 was **0.488633**. The fitting population changed, so this is not an isolated feature improvement or evidence that more users automatically improve quality. `validation_strata.csv` contains every group's three label-definition scores, counts and model comparison.

Selected model importance is dominated by personal-frequency rank, PPR score/rank, target active weeks and airing counts, recent-four-week counts, channel affinity and recency. Gain importance shows model usage, not causal usefulness; measured ablations above determine whether the new signals helped. All gains/split counts are in `selected_feature_importance.csv`.

## Final inference, verification and preservation

The selected **new** binary configuration was refit on all539,506 eligible historical candidate examples from 4,893 users after validation was frozen. Labels come only from historical Weeks21–30. Final inference uses V5's verified 139,516 candidate rows for the original1,260 users with context refit through Week30, plus the exact same completion-streak transformation. Final model names/order match the selected76-column schema. Five candidates are ranked deterministically by score, then common popularity order, then numeric ID. Cold users receive the shared history-only popularity list. Repeated previously viewed programs remain allowed; duplicates within one list do not.

**Mandatory experimental CSV was created despite local regression**, without bypassing correctness. Independently verified: original template IDs/order, 1,260 rows, exact two headers, no index/missing/duplicate users, five unique canonical positive IDs, named target schedule eligibility (6,636 items), model and feature hashes/order, all1,260 saved-model prediction replays, and preservation of prior evidence.

- **141 regression tests passed**, including all applicable Step2–7 suites plus7 V8-focused tests for aggregation, double-counting, completion thresholds, chronology/channel/zero handling, exact IDs, MAP conventions and feature-order expectations.
- **81 validation checks passed**, then **90 final submission checks passed**, all with **zero failures**.
- All new feature values for both historical validation and final inference were reconstructed using a separately implemented local Monday-date algorithm; all target counts and relevance sets were independently reconstructed; all15 inner and five assessment models were replayed; every saved subgroup metric was checked using rational AP, with production-evaluator parity.
- Scope limit: the original75 feature/generator implementation is reused, not independently reimplemented. The tests do not establish official Kaggle label construction or hidden-test accuracy.
- No failed V8 experiment or verification run occurred. Original protected evidence (603 files at start) and datasets retain their snapshot hashes. No V5 artifact or exported session was edited.

## Runtime, files and commands

| Stage | Runtime | Peak process RSS |
|---|---:|---:|
| Broader candidate/feature preparation | 240.85 s | 2,625.18 MiB |
| All CV, ablations, assessment fits and selected final fit | 139.91 s | 823.18 MiB |
| Final feature augmentation and CSV | 24.68 s | 269.76 MiB |
| Independent final verification | 89.02 s | not separately measured |

LightGBM4.6.0, NumPy2.2.6, Python3.12.14 in the existing dedicated environment. Per-configuration CV and inference timings are in `comparison.csv`; inference/fit timings are not confused with unrelated historical V5 measurements.

Key evidence: `official_labels.json`, `V5_reproduction.json`, `population.json`, `data_manifest.json`, `label_definition_differences.json`, `feature_audit.csv`, `plan.json`, `selection.json`, `experiment.json`, `comparison.csv`, `validation_strata.csv`, `final_model.txt`, `submission_manifest.json`, `test_feature_manifest.json`, `regression_tests.json`, `submission_verification.json` and `submission_sha256.txt` (all under `results/v8_experiment/`).

```bash
.venv-lightgbm/bin/python code/prepare_v8.py
.venv-lightgbm/bin/python code/build_v8.py
.venv-lightgbm/bin/python code/train_v8.py
.venv-lightgbm/bin/python code/verify_v8.py
.venv-lightgbm/bin/python code/generate_v8.py
.venv-lightgbm/bin/python code/verify_v8.py
.venv-lightgbm/bin/python code/summarize_v8.py
```

Creation scripts protect completed artifacts; reproduce into an explicit fresh output location rather than deleting evidence. Final verification is read-only with respect to previous experiments and writes new recheck reports.

V8 is ready for **manual experimental upload**, but has no genuine Kaggle score. Keep V5 as the designated final until a genuinely better private score is returned. Remaining exam deliverables include the final PDF/repository link, authentic private-score screenshot for the designated submission, consistent two-user business analysis, and the authentic unmodified full-session export. No online lift, satisfaction or revenue effect is claimed.
