# V11 targeted performance experiment

## Outcome and score status

**Generated and verified:** `submission/final_submission_v11_experimental.csv`.
Selected configuration: **E3 — hard-negative LambdaRank, original V10 candidate
pool**. It changes 718 of 1,260 final lists relative to V10.

V10 remains the best genuinely Kaggle-scored submission: **private 0.40637,
public 0.42051**, as reported by the user. V11 has **no Kaggle score** and was
not uploaded automatically. The aspirational private target is 0.44801; the
verified V10-to-target gap remains 0.04164. Local gains do not establish that
V11 closes any of that private-score gap.

## 1. Exact V10 reproduction

Read the full exam, V8–V10 Python implementation, V10 report, and verification
report before improvements. V10 reproduction passed all 28 checks:

- Development MAP@5: **0.40849817710558545**, 3,901 users.
- Secondary assessment MAP@5: **0.4091364247311828**, 992 users.
- All 4,893 local cross-fitted Top-5 lists and all 1,260 final lists reproduced.
- Final model: `results/v10_experiment/final_pairwise.txt`.
- Inference: `code/generate_v10.py`, using verified V9 test features and V10's
  historical-cold-user popularity fallback.
- Model SHA-256:
  `5bd830d88cb808a2617a982bba93b304e600738071b2ce367c7911ff79d92a65`.
- Existing V10 CSV SHA-256 matched its manifest:
  `39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e`.

Configuration: LambdaRank, 77 ordered V9 features, 78 rounds, 15 leaves,
learning rate 0.05, minimum leaf rows 100, L2 5, binary gains `[0,1]`, truncation
level 8, seed 2026, four threads. Contiguous user query groups are checked.
The pool is the union of the existing Top-50 PPR, historical frequency,
association-rule, CBF, and popularity lists. Repeated programs are allowed.

Historical features use Weeks 11–20, with targets strictly in Weeks 21–30.
The unchanged local target is Top-5 by >=120-second qualifying-airing count,
numeric program-ID tie-break; AP@5 divides by `min(5, number of relevant IDs)`.
The official hidden label construction remains unconfirmed. Ranking uses model
score descending, popularity rank ascending, numeric ID ascending. Cold users
ignore model scores and use the common popularity fallback.

The old V10 report predates the newly reported Kaggle results. It remains
unmodified; its recommendation and unscored status are superseded here.

## 2. Missing-candidate diagnosis: development users only

There were **2,843 missing relevant development pairs**: **101 previously
watched**, **2,742 unseen within the historical log**. All 2,843 were eligible
and had target-window schedule availability; none should have been excluded.

Overlapping diagnostic categories among those missing pairs:

| Historical/metadata reachability | Missing relevant pairs |
| --- | ---: |
| Matching historical category | 2,679 |
| Matching historical genre | 2,600 |
| Positive association-rule score | 769 |
| Positive personalized PPR score | 2,151 |
| Positive CBF similarity | 2,772 |
| Eligible and scheduled in the target period | 2,843 |

These are overlapping categories, not mutually exclusive partitions.
Metadata matching alone does not prove a good recommendation.

Audited the existing generators without changing their fitted algorithms.
For each, measured a bounded expansion: its first **50 positive-score unseen
items absent from the original pool**, in its unchanged native ranking order.
No future labels enter this candidate construction.

| Diagnostic pool | Unique user-item pairs | Added pairs | Additional relevant recovered | Candidate recall | Unseen recall | Seen recall | Avg/user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 429,669 | 0 | 0 | 85.0345% | 40.4819% | 99.2981% | 110.14 |
| PPR expansion | 622,619 | 192,950 | **881** | **89.6721%** | **59.6049%** | 99.2981% | 159.60 |
| Association-rule expansion | 618,663 | 188,994 | 283 | 86.5242% | 46.6247% | 99.2981% | 158.59 |
| CBF expansion | 622,655 | 192,986 | 239 | 86.2926% | 45.6696% | 99.2981% | 159.61 |

Generator score-computation time across development users: PPR 40.32 s,
association rules 49.02 s, CBF 26.32 s. Shared historical/metadata setup took
115.03 s. Complete diagnosis, including ranking, filtering, and reporting,
took 304.70 s; process peak RSS was 2,528.90 MiB. Per-generator times are
score computation, not separate end-to-end process measurements.

**Chosen retrieval hypothesis: deeper unseen PPR retrieval**, because it
recovered substantially more relevant pairs under the same bounded budget.
Only this expansion was carried into ranking experiments. Existing graph,
restart, rules, content vectors, and source-score definitions were reused;
no new CBF/AR algorithms or ALS were introduced. Final selection used MAP,
not this retrieval-recall result alone.

## 3. Complete expanded features and original-model ranking

The all-user validation pool increased from **539,506 to 781,306 pairs**.
Its original rows and all original features were preserved exactly.
Additional rows have complete historical, schedule, generator-score/rank,
completion-streak, and adjusted-PPR features. No unavailable feature was
filled with an arbitrary constant. The source-membership indicators retain
their original Top-50 meaning even for newly retrieved candidates.

Every row uses float32 values, finite-value checks, original transformations,
and the exact native 77-feature ordering. A deterministic one-original-row
per user reconstruction found zero feature differences. Fresh replay for
three deterministic development users reproduced all added candidates and
every added feature, as well as repeat generator outputs.

E2 scored the expanded pool with the original, user-separated V10 models.
**Top-5 MAP and Recall@5 were exactly unchanged**, despite better candidate
recall. Development Recall@10 rose from 63.0626% to 63.0942%; Recall@20 rose
from 74.7381% to 74.9118%. Added candidates reached deeper ranks but did not
enter Top-5. This is evidence of a ranking bottleneck, not evidence that
retrieval improvements guarantee useful Top-5 discovery.

## 4. Controlled hard-negative strategy

Used one fixed strategy: keep **every positive**, then at most **20 highest
OOF-miner-scored negatives plus 20 hash-sampled remaining negatives per user**.
This changes the training negative distribution; it does not duplicate rows
or claim that existing training negatives are new unique candidates.
Evaluation always scores the full configuration-specific candidate pool.

For each main outer development fold, two nested miners fit disjoint subsets
of that fold's training users. No outer evaluation-user or assessment-user
target labels enter mining or sampling. A training user's own labels are
absent from its miner. This avoids reusing a historical V10 miner that might
have fitted the current outer evaluation users' labels.

Assessment fitting uses V10 development OOF scores, whose models exclude all
assessment users. Final refitting uses verified user-excluded V10 cross-fits
on the permitted historical target table. All seeds, sampling counts,
features, rounds, and LambdaRank settings are fixed. E3 and E4 differ only
in original versus expanded candidate/training pools. There was no search.

## 5. E1–E4 comparison

Development has 3,901 users; secondary assessment has 992 users.

| ID | Configuration | Development MAP@5 | Assessment MAP@5 | Dev candidate recall | Dev unseen recall | Dev seen recall | Dev Recall@5 | Avg candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | Original V10 | 0.4084981771 | 0.4091364247 | 85.0345% | 40.4819% | 99.2981% | 48.3603% | 110.14 |
| E2 | PPR expansion + V10 | 0.4084981771 | 0.4091364247 | 89.6721% | 59.6049% | 99.2981% | 48.3603% | 159.60 |
| **E3** | **Hard negatives + original pool** | **0.4096236008** | **0.4118077957** | 85.0345% | 40.4819% | 99.2981% | 48.4761% | 110.14 |
| E4 | Hard negatives + expanded pool | 0.4087853542 | 0.4128351815 | 89.6721% | 59.6049% | 99.2981% | 48.4287% | 159.60 |

All use the same 77-feature, seed-2026, 78-round LambdaRank configuration.
`comparison.csv` contains both populations' full recall metrics, settings,
sample sizes, runtime fields, peak memory, and verification status.

**Selected E3 using development MAP**, frozen before E3/E4 assessment scoring.
E4's higher assessment MAP did not change selection. The assessment was
previously examined and is secondary, not a pristine holdout.

E3's local delta versus V10 is **+0.0011254237 development**,
**+0.0026713710 assessment**. A 2,000-resample paired development-user bootstrap
gave a descriptive 95% delta interval **[-0.0004059020, +0.0026228015]**.
It includes zero and is neither a Kaggle interval nor an independent test.

## 6. Chronological reliability

Ran a separate earlier cutoff, retraining all configurations and nested
miners using only that cutoff's permitted data. Did not apply a future-trained
V10 model to the earlier period. Selection was already frozen.

| Window / evaluated population | Users | E1 MAP | E2 MAP | E3 MAP | E4 MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| History Weeks 11–15, target Weeks 16–20; template development users | 1,004 | 0.4912361664 | 0.4912361664 | 0.4897687030 | 0.4947388225 |
| History Weeks 11–20, target Weeks 21–30; same template users | 1,004 | 0.4656706507 | 0.4656706507 | 0.4646513944 | 0.4647045153 |
| Main broader development | 3,901 | 0.4084981771 | 0.4084981771 | 0.4096236008 | 0.4087853542 |

All inner-fold scores and sample sizes are in `chronological_fold_scores.csv`.
For example, main E3 MAPs are 0.4065376756 / 0.4110597755 / 0.4112863280 on
1,305 / 1,277 / 1,319 users. Earlier E3 MAPs are 0.5001781665 / 0.4801340996 /
0.4890734824 on 343 / 348 / 313 users.

**Reliability findings:** E3's broader-population gain does not reproduce on
the matched template cohort or earlier cutoff. V10 previously lost against
V8 on broader development (0.408498 versus 0.409652), but won on template
development (0.465671 versus 0.463865). Population mix is a measurable reason
that broader local ordering can disagree with Kaggle ordering. Alternative
80%-completion label sensitivity also changes relative scores, but that
definition is not confirmed official and was not used for fitting/selection.

Only 20 logged weeks exist. Earlier history/target lengths are 5/5 weeks,
main 10/10, final 20/12; identical schema/rate conventions were retained,
but these are not fully comparable horizons. The earlier template cohort
was examined during V6. Main development/assessment were also reused.
Chronological shifts, sampling noise, population mix, and target-definition
uncertainty prevent claiming a dependable Kaggle gain from these local gains.

## 7. Final inference and verification

Refitted E3 on **215,983 sampled historical training rows across 4,893 users**,
preserving every original-pool positive. Supervised examples use pre-Week-21
features and Weeks 21–30 targets; all information is permitted through Week 30.
Final feature/PPR context uses history through Week 30. The selected original
candidate pool and exact final transformations reuse the verified V8/V9 cache.
The selected E3 strategy does not use PPR-expanded final candidates.

- **Validation verification: 84 passed, 0 failed.**
- **Submission verification: 89 passed, 0 failed.**
- **Regression tests: 152 run, 0 failures, 0 errors.**
- Independent held-out labels, MAP/recall, negative samples, model scores,
  deterministic ranking, and earlier-fold metric replay passed.
- Exactly 1,260 original test users in original order; correct two-column
  schema; five distinct eligible positive canonical IDs per user.
- Eligibility uses the 6,636 nonzero positive-duration programs in the final
  schedule. Future user viewing was not used. Schedule use is metadata and
  availability only.
- All **790 protected files** match the pre-V11 snapshot, including V1–V10
  sources/models/results/submissions/data and five original session JSON exports.
- Original joining behavior is covered by source/cache hashes and regression
  tests; it was not independently reimplemented. Fresh feature/generator replay
  shares the audited historical generator implementations.

**CSV:** `submission/final_submission_v11_experimental.csv`

**SHA-256:**

```text
9eda8006701038f3c2188387e3a0f3600fb026d6417b5be05e79aba9117f348b
```

Final model: `results/v11_experiment/final_model.txt`, SHA-256
`60c4eedc2432ef80f7c27a7049cd156e26624d751c3857669e0015d524c603d4`.

## 8. Cost and reproduction

Measured stage times: diagnosis 304.70 s; expanded feature build 449.95 s;
shared E1/E2 evaluation 14.40 s; complete E3/E4 nested mining/training 102.95 s;
earlier feature build 168.66 s; earlier evaluation 50.99 s; final fit/inference
28.86 s. Recorded stages including reproduction and verification total about
**26.4 minutes**, excluding interactive analysis and editing.

Maximum recorded process RSS was **3,317.93 MiB (3.24 GiB)**, below 11 GiB.
Final fitting/generation peaked at 857.84 MiB. Every stage capped CPU/BLAS/
LightGBM threads at four. Shared work is reported as shared, not attributed
as separately measured per-model runtime. See `execution_summary.json`.

Run from `SWEETTV_Final_Exam`, with existing verified V8–V10 artifacts:

```bash
.venv-lightgbm/bin/python -u code/prepare_v11.py
.venv-lightgbm/bin/python -u code/build_v11.py diagnose
.venv-lightgbm/bin/python -u code/build_v11.py validation
.venv-lightgbm/bin/python -u code/evaluate_v11.py
.venv-lightgbm/bin/python -u code/train_v11.py
.venv-lightgbm/bin/python -u code/build_v11.py early
.venv-lightgbm/bin/python -u code/chronological_v11.py
.venv-lightgbm/bin/python -u code/verify_v11.py validation
.venv-lightgbm/bin/python -u code/generate_v11.py
.venv-lightgbm/bin/python -u code/verify_v11.py submission
.venv-lightgbm/bin/python -u code/summarize_v11.py
```

Builders/trainers/generator reject overwriting completed outputs; verification
rechecks receive separate report names. For a clean rerun, use a separate
workspace while preserving original artifacts. `build_v11.py test` implements
the same expansion for inference if an expanded configuration is selected;
it was unnecessary for selected E3, which uses the verified original pool.

Primary evidence: `baseline_reproduction.json`, `retrieval_diagnosis.json`,
`missing_development_pairs.csv`, `validation_build.json`, `E1_E2.json`,
`hard_negative_plan.json`, `hard_negative_experiment.json`, `selection.json`,
`chronological_validation.json`, `comparison.csv`,
`chronological_fold_scores.csv`, `submission_manifest.json`,
`validation_verification.json`, `submission_verification.json`,
`regression_tests.txt`, `preservation_before.json`, and
`implementation_sha256.json`.
