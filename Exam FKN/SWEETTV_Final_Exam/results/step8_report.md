# Step 8 — Kaggle Score Diagnosis and Validation-Selected Update

## Outcome

The existing v1 submission was reproduced exactly from its recorded full-history PPR training path. It was not a stale-model, fallback, or program-ID corruption failure. The audit found one narrow candidate-pool defect: v1 time-clipped the named test schedule and therefore omitted four positive-duration, nonzero IDs present in `export_arh_31-42-final.csv`. The v2 candidate pool now includes every valid positive ID in that named file: **6,636 candidates**.

The most likely reason for the large gap between the supplied Kaggle scores and the original local primary score is a relevance-target mismatch, amplified by concentrated co-view recommendations. The local primary target treats **every distinct program with a qualifying Week 21–30 airing as relevant**; some users have hundreds of relevant IDs. The existing secondary target is narrower, keeping each user's five most frequently watched qualifying validation programs. Its PPR score (0.436048) is much nearer the user-reported Kaggle public/private values (0.39316 / 0.37269) than the all-program primary score (0.815151). This supports a target-mismatch explanation, but neither local metric is known to reproduce the hidden Kaggle labels. Kaggle results here are the values supplied by the user; no new Kaggle score was obtained.

The Step 8 selection rule was set in code before running its 16-variant search: maximize the fixed secondary Top-5 sensitivity MAP@5; break ties by higher primary MAP@5, lower prediction runtime, then config ID. It uses only the historical validation split. This selected **PPR + personal qualifying-airing rank fusion**, with equal reciprocal-rank weights and constant 60. The resulting local secondary MAP@5 is 0.440405, compared with 0.436048 for the best saved pure-PPR sensitivity configuration. Its primary MAP@5 is 0.807598, separately retained and lower than original PPR's 0.815151. It is therefore a validation-supported trade-off under the declared selection rule, not a guaranteed Kaggle-score improvement.

## V1 audit and concentration diagnosis

The exact v1 generator is `code/train_final_submission.py`; its final configuration and source hashes are in `results/final_model_manifest.json`. `code/run_step8.py` independently rebuilt the Weeks 11–30 PPR graph and reproduced all **1,260/1,260** v1 lists byte-for-byte at the row-value level. The configuration was `cosine_pfrequency_a0.35_k30`: cosine graph weights, frequency restart vector, restart probability 0.35, 30 strongest neighbors before symmetrization, 40-history cap, at least 3 common viewers, L1 tolerance `1e-8`, at most 250 iterations. It refit from 1,164,296 history interaction rows and 4,325 positive program nodes. No stale graph was loaded; the submission is reproducible from the verified historical input.

| V1 diagnostic | Result |
|---|---:|
| Unique recommendations across 6,300 slots | 96 (1.4475% of the 6,632-item v1 pool) |
| Program 2400480 frequency | 1,242 users (98.57%) |
| Program 240081 frequency | 1,142 users (90.63%) |
| PPR-ranked slots | 6,290 / 6,300 |
| Popularity-fallback slots | 10 / 6,300, all for the 2 users without Weeks 11–30 history |
| Warm users with 5 PPR-ranked suggestions | 1,258 / 1,258 |
| Distinct complete ordered lists | 859 |
| Identical user-list pairs | 1,738 / 793,170 (0.219%) |
| User pairs sharing at least 4 of 5 IDs | 32.91% |
| User pairs sharing at least 3 of 5 IDs | 69.77% |

The five most common programs have nonconstant PPR scores across users, confirming the model is personalized even though its output is concentrated. Across the cohort, score coefficient of variation was 0.344 for 2400480, 0.382 for 240081, 0.352 for 2400508, 0.198 for 6500479, and 0.249 for 2400467. Their high graph support and co-view centrality make them attractive for many restart vectors; per-user score differences often do not move them out of the top five. Popularity concentration is therefore not, by itself, proof of a broken PPR implementation.

No incorrect ID canonicalization or ineligible v1 prediction was found. All v1 program tokens were canonical positive IDs in its own pool. The pool discrepancy was specifically the four IDs `90040970`, `90041043`, `90066394`, and `90075056`: v1 clipped candidates by interval overlap with the test UTC period, although the task calls for valid nonzero IDs from the named 31–42 schedule. The v2 verifier reconstructs and checks the full named-file pool instead.

## Validation-target audit

The frozen Step 3 primary metric is MAP@5 over all 1,260 template users, with each user's relevant set containing **all distinct positive programs with at least one qualifying Week 21–30 airing**. Per-user AP is divided by `min(5, number_of_relevant_programs)`; a user with no validation labels contributes zero. Previously watched programs are permitted; duplicate predictions within one list are rejected. The validation candidate set contains 6,384 programs.

The secondary sensitivity target is the five most frequently watched qualifying validation programs per user, with deterministic numeric-ID tie-breaking. This was already defined and used in Steps 5–6. It is a helpful narrower diagnostic, not a proven reconstruction of Kaggle's hidden relevance labels. A broad primary target can reward predictions that match any five among many future programs; it can therefore yield much higher MAP@5 than a target centered on only a small Top-5 subset.

The Step 5 and Step 6 independent verifiers passed their saved 196 and 147 checks respectively. They and the Step 8 verifier found no validation/test viewing leakage in the completed methods. The final training input is the verified Step 2 Weeks 11–30 history artifact, ending before the Week 31–42 test period. The test schedule is used only to form candidate availability. The project consistently allows recommending previously watched programs, with five unique IDs per user. The four-ID v1 catalog restriction is documented above and corrected in v2; it does not explain the much larger metric-target gap.

## Validation-only model comparison

All variants use the same frozen cohort, validation candidates, primary truth, and existing secondary truth. The complete 16-row table, prediction files, and metric outputs are in `results/step8/validation_model_comparison.csv`, `results/step8/validation_predictions/`, and `results/step8/step8_experiment_results.json`. Candidate coverage is distinct recommended items divided by 6,384. Runtime below is prediction/ranking time for the validation cohort; common model-fitting time is recorded in prior Step 4–6 evidence and final v2's manifest.

| Validation model/configuration | Primary MAP@5 | Secondary MAP@5 | Unique / coverage | Prediction seconds |
|---|---:|---:|---:|---:|
| Original PPR — cosine/frequency, α=.35, k=30 | 0.815151 | 0.429489 | 117 / 1.833% | 13.02 |
| PPR sensitivity winner — cosine/frequency, α=.35, k=50 | 0.813021 | 0.436048 | 139 / 2.177% | 14.95 |
| Personal frequency — qualifying airings | 0.787627 | 0.434132 | 216 / 3.383% | 0.64 |
| Association Rules — 14-day, confidence × log-lift | 0.771331 | 0.313643 | 24 / 0.376% | 17.42 |
| PPR personalization — uniform, α=.35, k=30 | 0.771354 | 0.310873 | 63 / 0.987% | 13.05 |
| PPR personalization — duration, α=.35, k=30 | 0.786913 | 0.393481 | 207 / 3.242% | 14.56 |
| PPR personalization — recency, α=.35, k=30 | 0.774328 | 0.316101 | 93 / 1.457% | 16.56 |
| PPR + distinct-user popularity score blend, β=.05 | 0.814974 | 0.428743 | 117 / 1.833% | 12.61 |
| PPR + distinct-user popularity score blend, β=.15 | 0.814571 | 0.427669 | 113 / 1.770% | 13.03 |
| PPR + distinct-user popularity score blend, β=.30 | 0.814354 | 0.425437 | 112 / 1.754% | 12.65 |
| **PPR + personal-frequency RRF, PPR weight .50** | **0.807598** | **0.440405** | **174 / 2.726%** | **1.97** |
| PPR + personal-frequency RRF, PPR weight .75 | 0.813111 | 0.435942 | 149 / 2.334% | 2.06 |
| PPR + Association Rules RRF, PPR weight .50 | 0.790011 | 0.357333 | 30 / 0.470% | 2.01 |
| PPR + Association Rules RRF, PPR weight .75 | 0.802894 | 0.386783 | 51 / 0.799% | 2.04 |
| PPR + global popularity RRF, PPR weight .25 | 0.771413 | 0.311167 | 8 / 0.125% | 1.99 |
| PPR + global popularity RRF, PPR weight .50 | 0.778106 | 0.327439 | 17 / 0.266% | 1.98 |

The selected personal-frequency fusion improves secondary MAP@5 by **0.004357** over the best saved pure-PPR sensitivity model and by **0.006272** over personal frequency alone. It retains strong primary MAP@5, while scoring 0.007553 below primary-best PPR. On validation it recommends 174 distinct programs rather than PPR's 117; mean pairwise Top-5 item overlap is 1.870 rather than 2.751. These diversity changes describe output variation and do not establish that diversity causes better accuracy or user experience.

The fusion uses full deterministic rankings from each model. For candidate `i` and zero-based ranks `r_ppr`, `r_freq`, its fused score is `0.5/(60+r_ppr) + 0.5/(60+r_freq)`. Exact ties use the common full-history popularity rank, then numeric program ID. Rank fusion is selected from the predeclared secondary metric; no actual Kaggle score or hidden label was used to select its weights.

## Final v2 refit and independent verification

The v2 model is refit on all permitted historical interactions through Week 30 using the 120-second same-channel, half-open overlap join. Split-period overlap seconds are combined per user/broadcast before thresholding; ID zero is excluded. Full-history PPR uses cosine co-view edges, a frequency restart vector, restart α=.35, top-30 neighbors before symmetric union, 40-program graph-history cap, minimum 3 common viewers, tolerance `1e-8`, and 250 maximum iterations. Personal frequency ranks full-history eligible programs by qualifying-airing count then ID. Both complete rankings use the same distinct-user/airings/ID popularity order for completion and deterministic RRF ties. Repeats from history are allowed; duplicates within an output list are not.

- **New CSV:** `submission/final_submission_v2.csv`
- **Users:** 1,260, in original template order; five unique canonical eligible IDs per user.
- **Test candidate pool:** 6,636 positive, nonzero, valid-duration program IDs from `export_arh_31-42-final.csv`.
- **Training source:** `data/step2/threshold_120s/history_interactions.csv.gz`, SHA-256 `7093e882ca1e1f9763340681b7f0009e32e9d2803be90513c68c6dcd8fb01834`.
- **v2 output unique programs:** 149 / 6,636 candidates; top frequencies: 2400480 for 1,110 users, 240081 for 923, 2400508 for 588.
- **SHA-256:** `e7df8177af1aeda8cb0b7fdf543e762acda7f375b30cacdf38e4115cc823b5b6`.
- **Refit runtime:** 50.54 seconds end to end on Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, Linux x86-64.

`code/verify_kaggle_submission_v2.py` independently passed **11 checks, zero failures**. It re-derived the schedule candidate pool; streamed and checked all 1,164,296 training interactions for hash, 120-second threshold and Weeks 11–30 bounds; rechecked all 16 validation configurations and their saved MAP/diversity metrics under the declared selection rule; independently rebuilt the full-history PPR graph and compared all graph arrays with the saved artifact; independently refit the frequency profiles and RRF rankings and reproduced all 1,260 output rows; checked schema, order, ID canonicalization/eligibility, missing values and duplicates; and verified the SHA-256. Its final runtime was 317.70 seconds.

The focused Step 8 helper suite passed **4/4 tests** (`.venv/bin/python3 code/test_step8.py`). Earlier diagnostic/training attempts and their corrections are recorded in `results/step8_failures.json`; v1 remains preserved. No Kaggle submission was performed. The v2 CSV is ready for manual Kaggle upload, but its Kaggle public/private score is unknown.

## Reproduction and evidence

```bash
.venv/bin/python3 code/test_step8.py
.venv/bin/python3 code/run_step8.py
.venv/bin/python3 code/train_final_submission_v2.py
.venv/bin/python3 code/verify_kaggle_submission_v2.py
```

Key evidence: `results/step8/v1_submission_diagnostics.json`, `results/step8/validation_model_comparison.csv`, `results/step8/step8_experiment_results.json`, `results/step8_failures.json`, `results/final_model_v2_manifest.json`, `results/final_submission_v2_verification.json`, and `results/final_submission_v2_sha256.txt`.
