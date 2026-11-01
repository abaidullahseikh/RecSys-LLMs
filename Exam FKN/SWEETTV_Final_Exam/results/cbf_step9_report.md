# Step 9 — Content-Based Filtering Evaluation

## Summary and V3 decision

Sparse metadata CBF and 24 CBF-containing rank-fusion configurations were evaluated on the frozen Step 3 validation split. The best individual CBF reached **0.495029 primary MAP@5** but only **0.200193** on the existing Top-5 sensitivity target. The best new hybrid under the predeclared validation selection rule was **PPR + personal frequency + CBF**, weights 0.40/0.40/0.20, with **0.798844 primary MAP@5** and **0.413601 secondary MAP@5**.

The existing PPR + personal-frequency model scores **0.807598 primary** and **0.440405 secondary** under those same targets. Since no new CBF or hybrid candidate improves the selected secondary validation metric, the evidence does **not** support replacing v2. **No `final_submission_v3.csv` was generated.** The user-reported Kaggle private MAP@5 of 0.37880 and aspirational 0.44801 target were not used to select configurations; neither was independently queried or achieved in this work.

## Official metric and the reported 80% rule

The official Kaggle CompetitionService record for competition **24201** reports the evaluation algorithm as **MAP@{K}**, higher is better, and describes the task as predicting TV program preferences. The accessible official competition metadata does not specify how hidden relevance labels are constructed and does not state that watched fraction must be at least 80%.

The course quiz fixes `K=5`; the project’s Step 3 report explicitly says the live hidden-label construction was not established. The reported **80%-of-airing-duration then frequency-ranked Top-5** rule is therefore **unverified**, not an official rule. I evaluated it only as a separately labeled sensitivity target: for each validation user-airing, compute the existing unioned same-channel overlap divided by full broadcast duration; count an airing if `5 × overlap_seconds >= 4 × duration`; rank each user’s programs by such airing counts and retain five. It produced nonempty labels for 1,238 users. It did not replace or tune the frozen 120-second primary protocol.

Sources: [official Kaggle competition record](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionId=24201), [official competition overview](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/), `exam_quiz_FKN.md`, `results/step3_evaluation_report.md`, and `results/cbf_step9_official_evidence.json`.

The historic interaction rule remains unchanged for model fitting and the primary comparison: same-channel, half-open view/airing overlap; merge user/broadcast overlap fragments; count an airing at **120 or more unioned seconds**; drop ID zero. The alternative 80% target is constructed from the same verified overlaps but is reported separately.

## CBF implementation

`code/content_based.py` implements a sparse TF-IDF-equivalent space without adding scikit-learn or dense item matrices. It creates one metadata document per positive-ID program in the Week 11–20 schedule, fits vocabulary and document frequency/IDF **on those training schedule documents only**, then transforms validation candidates using that fixed vocabulary. Term values use binary term frequency, smoothed IDF `log((1+N)/(1+df))+1`, field-specific multipliers, and L2 row normalization.

Fields included were category, three genres, title terms, director names, actor names, production year/decade, channel ID, and channel title. Metadata rows are combined per program. Candidate metadata comes from schedule records only; user viewing is used only for the Weeks 11–20 personal profile. User profiles weight historical program vectors by qualifying airing count, log-frequency, or log watched seconds. Candidates are ranked by sparse cosine similarity; unseen/zero-similarity and cold-start slots use the shared Step 4 popularity fallback (distinct viewers, then qualifying airings, then numeric ID). Recommending a previously watched program remains allowed, with five unique IDs per list.

The 12 individual CBF variants used four metadata field-weight settings and three user-profile weights. The best was `cbf_channel_context_emphasis_airings`: category 1.5, genre 1.25, title 0.8, director 0.3, actors 0.2, year 0.1, channel 1.0; raw qualifying-airing user weights. Its feature space had 5,842 historical program documents, 24,309 vocabulary terms, 118,965 training-matrix nonzeros, and 119,665 validation-candidate-matrix nonzeros.

## Results

All methods use the same 1,260 users, 6,384 validation candidates, 120-second Step 3 primary labels, and popularity fallback. Primary MAP is over all distinct qualifying validation programs; the existing secondary MAP is the frozen per-user top-five-by-qualifying-airing sensitivity target. Candidate coverage is distinct recommended programs / 6,384.

| Model | Primary MAP@5 | Secondary Top-5 MAP@5 | Fraction-80 sensitivity MAP@5 | Unique programs / coverage |
|---|---:|---:|---:|---:|
| Popularity — distinct viewers | 0.768418 | 0.306627 | 0.266837 | 5 / 0.078% |
| Personal frequency — qualifying airings | 0.787627 | 0.434132 | 0.344587 | 216 / 3.383% |
| Personalized PageRank — cosine/frequency, α=.35, k=30 | 0.815151 | 0.429489 | 0.354977 | 117 / 1.833% |
| Association Rules — 14-day, confidence × log-lift | 0.771331 | 0.313643 | 0.273966 | 24 / 0.376% |
| Existing PPR + personal-frequency RRF (.50/.50) | **0.807598** | **0.440405** | **0.352790** | 174 / 2.726% |
| **Best individual CBF — channel-context/airing frequency** | **0.495029** | **0.200193** | **0.179984** | 272 / 4.261% |

### Best new hybrid in each requested family

| Hybrid | Selected validation weights | Primary MAP@5 | Secondary Top-5 MAP@5 | Fraction-80 sensitivity MAP@5 | Unique programs / coverage |
|---|---|---:|---:|---:|---:|
| PPR + CBF | PPR/CBF = .75/.25 | 0.799646 | 0.401746 | 0.330444 | 190 / 2.976% |
| Personal Frequency + CBF | PF/CBF = .75/.25 | 0.767579 | 0.391857 | 0.323641 | 277 / 4.339% |
| PPR + PF + CBF | PPR/PF/CBF = .40/.40/.20 | 0.798844 | **0.413601** | 0.337353 | 232 / 3.634% |
| PPR + PF + Association Rules + CBF | PPR/PF/AR/CBF = .30/.30/.20/.20 | 0.809201 | 0.408706 | 0.331532 | 170 / 2.663% |

The four-way hybrid slightly exceeds v2 on the **primary** local proxy, but falls below it on the predeclared secondary selection metric. No CBF hybrid improves the existing v2 secondary MAP@5 of 0.440405. The selected three-way hybrid is 0.026805 below v2 on that metric. It is not selected for a final submission.

The complete table has all 12 CBF configurations, 12 new hybrids, and five historical baselines in `results/step9_cbf/model_comparison.csv`; predictions are in `results/step9_cbf/predictions/`. CBF+PPR/PF/AR fusion uses full candidate rankings and reciprocal-rank fusion with constant 60. Weight grids are saved in `code/run_cbf_step9.py` and the result JSON. Configurations are selected on the frozen secondary validation MAP, tie-broken by primary MAP, prediction time, and ID. The fraction-80 sensitivity target does not select models.

## Runtime, memory, tests, and verification

- Full 12-CBF / 12-hybrid comparison run: **297.31 seconds**, peak RSS **2,063.10 MiB** (Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, Linux x86-64).
- Individual CBF ranking for 1,260 users: approximately **12.88–14.27 seconds** per profile/field configuration; feature-space fit/transform was about **0.5–0.6 seconds** per field-weight configuration.
- New hybrid ranking/fusion for 1,260 users: approximately **2.09–2.43 seconds** per tested weight configuration.
- `code/test_content_based.py`: **4 tests passed**. They cover train-only vocabulary behavior, sparse transform, multilingual token handling, cold-start fallback, repeat-allowed candidates, unique Top-5, and agreement between the fast metric calculation and Step 3 MAP@5.
- `code/verify_cbf_step9.py`: **9 independent checks passed, zero failed**. It rechecked all 29 prediction files against primary, secondary, and fraction-80 targets; checked the candidate coverage and diversity summaries; cross-checked the best individual CBF and selected hybrid with the frozen Step 3 evaluator; and confirmed the model-selection/V3 decision.
- Initial fixture expectation and runner performance/reporting issues were corrected; the actual attempts are preserved in `results/cbf_step9_failures.json`.
- Optional LightGBM was not implemented. The requested CBF and hybrid search completed without it.

## Submission status

**No V3 was generated.** The existing `submission/final_submission_v2.csv` and all prior experiments remain unchanged. The best CBF-containing candidate does not beat v2 on the specified validation selection metric, and no prediction has been submitted to Kaggle. The user-supplied 0.37880 private score and 0.44801 aspiration are not treated as locally reproducible scores; no new Kaggle score is claimed.

Key evidence: `results/cbf_step9_official_evidence.json`, `results/step9_cbf/step9_validation_results.json`, `results/step9_cbf/step9_verification.json`, `results/cbf_step9_verification.json`, and `results/cbf_step9_failures.json`.
