# V12: full expanded-pool LambdaRank and frequency grades

## Outcome

**Generated and verified:** `submission/final_submission_v12_experimental.csv`.
Selected **E3: binary LambdaRank fitted on every expanded candidate row**.
Development MAP@5 is **0.4088470193**, secondary assessment **0.4109173387**.
This is a small local improvement over V10 and a local regression versus V11.
It is not evidence of improved genuine Kaggle private performance.

Best genuinely scored submission remains **V10, private 0.40637**. User-reported
V11 private is **0.40634**; V5/V8 private are 0.39706/0.39908. V12's genuine
private/public scores are **unknown**. No automatic upload was performed. The
aspirational private target remains 0.44801, with a verified-best gap of 0.04164.

## 1. Requirements, preservation, and exact reproduction

Read the complete exam, V10/V11 implementation, V11 summary, and verification
report before changes. Preserved the earlier reports as historical records;
their unscored status for V11 predates the newly user-reported Kaggle result.

Both baselines reproduced exactly, including all local Top-5 predictions,
saved score arrays, and all 1,260 final lists. Reproduction passed 51 checks.

| Baseline | Development MAP@5; 3,901 users | Assessment MAP@5; 992 users |
| --- | ---: | ---: |
| V10 | 0.40849817710558545 | 0.4091364247311828 |
| V11 | 0.4096236007861232 | 0.41180779569892473 |

V10's exact model is `results/v10_experiment/final_pairwise.txt`, SHA-256
`5bd830d88cb808a2617a982bba93b304e600738071b2ce367c7911ff79d92a65`.
It uses the ordered 77-feature V9 schema: the original 75 features, completion
streak, and adjusted PPR. The unchanged adjusted-PPR transformation is
`ppr_score * sqrt(historical_users / (distinct_program_viewers + 20))`.

Existing submission checksums matched their recorded manifests:

- V10: `39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e`
- V11: `9eda8006701038f3c2188387e3a0f3600fb026d6417b5be05e79aba9117f348b`

Verified historical feature boundaries, held-out user separation, complete
query groups, label alignment, deterministic ordering and cold fallback.
Main history is Weeks 11–20, targets Weeks 21–30. Training-user targets in
this historical window are permitted supervision; each evaluation user's
own target labels are excluded from its fitted model. Final context uses
permitted behavior through Week 30, never Weeks 31–42 user viewing.

## 2. Why added relevant unseen programs rank poorly

Diagnosis used development users only and the existing V10 OOF rankers.
There are **192,950 added candidate pairs**, all unseen within the historical
log, including **881 relevant local Top-5 pairs**. They have **zero missing
numeric features**, and all have timestamp-clipped target availability.

| Statistic | Added relevant unseen pairs | Original relevant unseen pairs |
| --- | ---: | ---: |
| Pairs | 881 | 1,865 |
| Mean native LambdaRank score | -2.1288 | -1.6051 |
| Median rank in expanded pool | 39 | 24 |
| Median PPR score | 0.001409 | 0.005243 |
| Median association-rule score | 0.004714 | 0.229711 |
| Mean historical distinct program viewers | 684.40 | 2,282.91 |
| Mean target airings, ten-week equivalent | 914.48 | 606.11 |
| Mean target-week availability fraction | 0.9723 | 0.9397 |

Native LambdaRank scores are unbounded ranking scores, not probabilities.

For the 881 newly recovered relevant pairs under V10:

- Top-5: **0 / 881 = 0%**.
- Ranks 6–10: **8 / 881 = 0.91%**.
- Ranks 11–20: **88 / 881 = 9.99%**.
- Below rank 20: **785 / 881 = 89.10%**.

Across all added pairs, only four entered Top-5, all irrelevant to the local
Top-5 truth; 99.17% ranked below 20. This distinction matters: unchanged MAP
does not imply that no new candidate ever entered a prediction list.

**Strongest measured explanation:** strong repeat-frequency preference,
weaker/deeper personalized retrieval signals, and a training-support mismatch.
Historical-frequency rank accounts for **73.55% of V10 split gain**. Added
programs have no personal repeat history. Every added row has original
`candidate_source_count = 0`, whereas original training rows have 1–5.
Their original Top-50 source flags correctly remain zero; they are not
fabricated missing features. Their schedule availability is strong, so
missing schedule information does not explain these low ranks.

These are descriptive associations, not proof of causal popularity bias.
The raw-score calculation is unchanged across old/new candidates; no score
scale conversion was introduced. V12 addresses the measured mismatch by
training on the full expanded distribution, without an automatic unseen bonus.

Full distributions, missing/zero rates, all-feature comparisons, and model
importance are saved in `unseen_diagnosis.json`,
`candidate_feature_distributions.csv`, and `V10_feature_importance.csv`.

## 3. Matched expanded candidates and label construction

Reused checksum-verified V11 expansion caches. The unchanged retrieval rule
is the original five-generator Top-50 union plus the first **50 positive-PPR
historical-unseen programs absent from that union**, in native PPR order.
No relevance labels select candidates or force positives into the pool.

| Development statistic | Original | Expanded |
| --- | ---: | ---: |
| Candidate rows | 429,669 | 622,619 |
| Average candidates/user | 110.14 | 159.60 |
| Candidate recall | 85.0345% | 89.6721% |
| Relevant unseen candidate recall | 40.4819% | 59.6049% |
| Relevant seen candidate recall | 99.2981% | 99.2981% |
| Local binary positives | 16,154 | 17,035 |
| Local binary-positive rate | 3.7596% | 2.7360% |
| Any qualifying-count positives | 99,532 | 114,474 |

All-user training data increased from **539,506 to 781,306 rows** across
4,893 users. The expanded table contains **21,354 binary positives**.
Every V12 training query includes every expanded candidate; no negative
subsampling or per-item weights are used. Group-size sums equal row counts.

This is **not V11 E4 repeated**: V11 E4 kept all available positives and at
most 20 hard plus 20 sampled other negatives per user. V12 E3 fits every
expanded row. No additional algorithm or hyperparameter search was added.

Reconstructed target counts only after the existing candidate table was
available. One qualifying occasion is one matched user/broadcast airing
with >=120 total overlapping seconds after fragment union. Repeated
broadcasts count separately; fragmented views of one airing are not separate
occasions. ID0 is excluded. Calendar/overlap rules remain unchanged.

Binary E3 keeps the V10/V11 local frequency-Top-5 membership labels. E4 uses:

| Qualifying target-airing count | Grade | LightGBM gain |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2–3 | 2 | 3 |
| 4+ | 3 | 7 |

These grades are experimental, not confirmed official rules. **E4 applies
grades to all count-positive target programs, including those outside the
local Top-5.** Thus it changes positive coverage as well as relative gains;
the result is not an isolated test of reweighting identical positive IDs.
Target counts/grades are never feature columns or retrieval inputs.

## 4. Fixed LambdaRank configurations and main comparison

Both newly trained models use 77 float32 features, 78 rounds, 15 leaves,
learning rate 0.05, minimum leaf rows 100, L2 5, truncation level 8,
seed 2026, deterministic column-wise training and four threads. Only label
targets/gains differ between E3/E4. Three user-disjoint development fits
produce OOF predictions; a development-only fit produces assessment scores.

| Experiment | Development MAP@5 | Assessment MAP@5 | Dev candidate recall | Dev unseen candidate recall | Dev Recall@5 | Dev unseen Top-5 precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1: Original V10/original pool | 0.4084981771 | 0.4091364247 | 85.0345% | 40.4819% | 48.3603% | 14.1227% |
| E2: V10/expanded pool | 0.4084981771 | 0.4091364247 | 89.6721% | 59.6049% | 48.3603% | 14.0825% |
| **E3: All expanded rows/binary** | **0.4088470193** | **0.4109173387** | 89.6721% | 59.6049% | 48.3076% | 13.4529% |
| E4: All expanded rows/graded | 0.4041872454 | 0.4058809364 | 89.6721% | 59.6049% | 47.9286% | 14.1631% |

Development uses 3,901 users; secondary assessment uses 992. Full metrics,
including seen/unseen candidate and Top-5 recall, are in `comparison.csv`.

**Selected E3 solely using development MAP**, frozen before either new
assessment fit. It improves over V10 by **+0.0003488422 development** and
**+0.0017809140 assessment**, but regresses versus V11 by
**-0.0007765815 development** and **-0.0008904570 assessment**.
Graded E4 regresses against binary E3 by **0.0046597738 development MAP**.

The E3 paired development-user bootstrap 95% interval versus V10 is
**[-0.0010017891, +0.0017091237]**, including zero. This uses 2,000 resamples
with seed 2026 and is descriptive on reused development data, not a Kaggle
confidence interval.

**Unseen ranking was not solved.** E3 still puts none of the newly recovered
881 relevant pairs in Top-5, although ranks 6–10 increase from 8 to 20.
Its overall unseen Top-5 relevant recall falls from E2's 2.1489% to 1.9535%.
Graded E4 puts one recovered relevant item in Top-5, but worsens overall MAP.
Full post-training bins are in `post_training_unseen_audit.json`.

## 5. Chronological reliability

After freezing main-development selection, retrained the same two fixed
configurations on the earlier cutoff. Baselines at that cutoff use only
earlier-fitted models, not V10 models trained on later targets.

| Population/window | Users | E1 | E2 | E3 | E4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| History Weeks 11–15, target Weeks 16–20 | 1,004 | 0.4912361664 | 0.4912361664 | 0.4911249447 | 0.4872111554 |
| Main window, same template development users | 1,004 | 0.4656706507 | 0.4656706507 | 0.4643592297 | 0.4604216467 |
| Main broader development | 3,901 | 0.4084981771 | 0.4084981771 | 0.4088470193 | 0.4041872454 |

E3's small broader-population gain does not reproduce on the matched
template cohort or earlier cutoff. Earlier5/5-week and main10/10-week
history/target horizons differ, and final history/target is20/12 weeks.
Earlier and main cohorts were already examined in prior experiments; none
is pristine. Inner-fold scores and sample sizes are saved in
`fold_scores.csv` and `early_fold_scores.csv`. No chronological or assessment
result changed the frozen selection.

## 6. Final expanded inference and consistency

Refitted selected E3 on **all 781,306 historical expanded rows**, grouped
into 4,893 users, using targets only through Week30. Refit retrieval on all
permitted history through Week30. Final original139,516 candidates plus
62,900 extra unseen pairs give **202,416 final candidate rows**.

Recomputed complete features for **every final row**, including originals,
using the same V11 expanded-training definitions and exact native order.
This exposed older V5 final-cache definition differences: full-history versus
trailing10-week channel/hour affinities, timestamp-clipped/deduplicated
schedule counters, and associated metadata-derived scores/ranks. The exact
changed-column counts and maximum differences are in `test_build.json`.
This consistency work does change final feature values relative to V10/V11's
old cache; it is not hidden or presented as a separately validated gain.
Original candidate IDs are preserved, and old caches remain unchanged.

All behavioral features precede the cutoff. Schedule rows are used only for
metadata/availability, timestamp-clipped to the relevant target window;
final eligibility is the 6,636 nonzero positive-duration programs in the
Weeks31–42 file. Every final recommendation also has clipped availability.
The original cold fallback is preserved: ignore learned scores for users
without qualifying historical profiles, then rank by historical popularity
and numeric program ID. All predictions are deterministic.

V12 changes **710 user lists versus V10**, **759 versus V11**.

**CSV:** `submission/final_submission_v12_experimental.csv`

**SHA-256:**

```text
189b3079a331e1f30ac6acf28424fbffb36e54a8828481380305fabaff041b8a
```

Final model: `results/v12_experiment/final_model.txt`, SHA-256
`b9e243a9a88f49ee0416ed54d44d052599e667fc1845e16d6275baa2213b4a78`.

## 7. Verification, resources, and attempt history

- **Latest validation verification: 112 passed, 0 failed.**
- **Final submission verification: 118 passed, 0 failed.**
- **Regression tests: 157 run, 0 failures, 0 errors.**
- Independent count/grade reconstruction, label alignment, group sums,
  model-score replay at both cutoffs, rational MAP and production-evaluator
  parity, recall/precision, deterministic candidates/features and final
  ranking all passed.
- Exactly1,260 original integer user-ID strings in exact template order;
  exact original CSV header; five distinct positive canonical eligible IDs.
- All **889 protected V1–V11 files** match their pre-V12 hashes, including
  models, code, submissions, experiments/reports, input data, and the five
  original session JSON exports.
- Original viewing-to-program joining is protected by hashes and regression
  tests, not independently reimplemented. Fresh sampled feature/generator
  replay shares the audited implementations.

Three initial verifier-wrapper attempts stopped on strict NumPy-string type
handling or on incorrectly passing only retrieved IDs as the evaluator's
eligibility universe. These reports are retained as
`validation_verification_failed_attempt1/2/3.json`. The wrapper now preserves
canonical string IDs and includes unretrieved eligible truth in the metric
universe, without adding any truth IDs to candidates. A regression covers
that denominator contract. No model, candidate table, target or metric
definition was changed to obtain the passing reports.

The generation/verification command was interrupted after generation had
finished. Existing final artifacts were inspected, preserved, and submission
verification resumed without refitting/regeneration.

| Measured stage | Time | Peak process RSS |
| --- | ---: | ---: |
| Exact reproduction | 40.88 s | 1,156.82 MiB |
| Development diagnosis | 15.32 s | 857.79 MiB |
| Main training and evaluation | 163.73 s | 1,069.66 MiB |
| Earlier training and evaluation | 41.10 s | 1,058.72 MiB |
| Final expanded features | 223.42 s | 2,580.68 MiB |
| Passing validation verification/tests | 564.03 s | 2,761.10 MiB |
| Selected final fit/generation | 29.88 s | 857.18 MiB |
| Final submission verification | 97.00 s | 1,642.83 MiB |

E3's four main fits took79.11 s and inference2.04 s; E4's four main fits
took56.02 s and inference1.74 s. Final selected refitting took9.99 s,
inference0.22 s. Baseline models were reused, with zero new baseline fitting.
Peak process figures include shared setup and are not isolated model heaps.

Successful recorded stages total **19.6 minutes**, or **22.5 minutes including
the three failed verifier attempts**, excluding editing, the interruption,
and incomplete interrupted work. Maximum recorded RSS was **2.70 GiB**;
all CPU/BLAS/LightGBM stages used at most four threads, below11 GiB RAM.

## 8. Reproduction and remaining limitations

Run from `SWEETTV_Final_Exam` with the verified earlier artifacts:

```bash
.venv-lightgbm/bin/python -u code/prepare_v12.py
.venv-lightgbm/bin/python -u code/diagnose_v12.py
.venv-lightgbm/bin/python -u code/run_v12.py main
.venv-lightgbm/bin/python -u code/run_v12.py early
.venv-lightgbm/bin/python -u code/build_v12.py
.venv-lightgbm/bin/python -u code/verify_v12.py validation
.venv-lightgbm/bin/python -u code/generate_v12.py
.venv-lightgbm/bin/python -u code/verify_v12.py submission
.venv-lightgbm/bin/python -u code/summarize_v12.py
```

Builders/trainers/generator refuse to overwrite final outputs; verifier
rechecks use separate filenames. `implementation_sha256.json` pins V12 code.

Limitations: official hidden labels remain unconfirmed; grade targets broaden
positive coverage and may not align with MAP's local Top-5 target; unseen
Top-5 capture remains weak; private-user/time/horizon shifts are unobserved;
the small local E3 gain is uncertain and absent on matched/earlier cohorts;
final feature consistency changes cannot be evaluated against future labels.
No genuine private-score improvement is claimed before a Kaggle result.
