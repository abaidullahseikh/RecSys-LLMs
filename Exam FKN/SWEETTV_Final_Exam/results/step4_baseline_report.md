# Step 4 — baseline recommenders

**Student:** Abaidullah Seikh · **Course:** LLM4Rec Final Exam (FKN)  
**Scope:** global popularity and personal-frequency references only. Personalized PageRank, Association Rules, other models, hybrids, Kaggle predictions and submissions are not part of this stage.

## Summary

Both baseline families were fitted using only the verified Week 11–20 interaction file and evaluated through the unchanged Step 3 MAP@5 protocol for all **1,260** original cohort users. Five score definitions were run: three global popularity scores and two personal-frequency scores.

| Model variant | Primary MAP@5, n=1,260 | Nonempty-label diagnostic MAP@5, n=1,246 | Warm-user primary MAP, n=1,250 | Cold-user primary MAP, n=10 | Users with ≥1 relevant hit |
|---|---:|---:|---:|---:|---:|
| Popularity — distinct users | **0.768418** | 0.777052 | 0.771037 | 0.441000 | 1,210 |
| Popularity — qualifying airings | 0.649291 | 0.656586 | 0.651528 | 0.369667 | 1,220 |
| Popularity — watched seconds | 0.732974 | 0.741209 | 0.735565 | 0.409000 | 1,207 |
| Personal frequency — qualifying airings | **0.787627** | 0.796477 | 0.790400 | 0.441000 | 1,227 |
| Personal frequency — watched seconds | 0.692394 | 0.700174 | 0.694405 | 0.441000 | 1,216 |

The best tested score within the popularity family is **distinct-user count**. The best tested personal-frequency score is **qualifying-airing count**. Their primary-MAP difference is **+0.019209** for personal frequency in this fixed local validation protocol. This is a descriptive hold-out difference, not a customer-satisfaction or causal claim. The nonempty diagnostic is displayed separately and must only be compared with other nonempty diagnostics.

## Frozen inputs and shared evaluation contract

Read before baseline implementation: complete `exam_quiz_FKN.md`, Step 2 preprocessing report, Step 3 evaluation report and verification, Step 3 `evaluation.py`/`validation.py`, and `data/step3/evaluation_config.json`.

- Hold-out training: Week 11–20, `[1583704800,1589749200)` UTC, from `data/step2/threshold_120s/train_interactions.csv.gz` only.
- Validation labels: Week 21–30, from the frozen `data/step3/validation_truth.csv`.
- Cohort: 1,260 users in original template order; 1,250 warm and 10 cold.
- Validation candidate pool: exactly 6,384 distinct positive IDs from the frozen schedule-only candidate file.
- Relevance: all distinct programs with a qualifying validation airing under Step 2's 120-second per-airing rule; no Top-5 truncation.
- MAP@5/AP denominator: the Step 3 implementation, `min(5, number of relevant programs)`; empty truth contributes AP=0. Primary and nonempty diagnostic are separate.
- Previously watched programs are permitted in recommendations when present in the validation pool. Duplicate IDs within one recommendation list remain prohibited.
- Personal-frequency fallback uses the Step 3 contract: global popularity sorted by distinct training users descending, qualifying airing count descending, then numeric program ID ascending. The popularity baseline itself is a global ranking for every user.

The Step 3 configuration hash remains `400b00eb6d660a663222e2e083ee59b2132469b2a06da28918b11323c88ec758`; protocol ID remains `40f6d1c93367106b22d3b66a76605afd12a1ec929f4fe7929bd137749880f239`. No evaluation policy was changed.

### Trained score definitions

Popularity statistics were accumulated over all **716,796** distinct training user–program rows and all **4,837** users with a qualifying program interaction:

1. **Distinct users:** number of unique exact-string training user IDs with an interaction for the program.
2. **Qualifying airings:** sum of `qualifying_airings` over training user–program rows.
3. **Watched seconds:** sum of `watched_seconds` over training user–program rows.

The ranking's primary key is that variant's descending score. Deterministic ties use distinct training users descending, qualifying airing count descending, and numeric ID ascending. All counts are training-only, including IDs outside the current candidate pool; only eligible validation IDs are ranked. Popularity emits the same global Top-5 for every user.

Personal-frequency scores use each user's Weeks 11–20 rows:

1. Sum of qualifying broadcast counts for that user/program.
2. Sum of watched seconds on qualifying broadcasts for that user/program.

The user-specific score ranks historical programs that are in the common validation catalog; ties use numeric ID ascending. Previously watched eligible programs are retained. If fewer than five eligible personal programs exist, fill from the distinct-user global popularity order, skipping duplicates. A user without any qualifying training history receives the full same popularity fallback list. Training history outside the target schedule is not recommended and causes fallback slots rather than an ineligible ID.

## Runtime, memory, and catalogue behavior

| Model family / stage | Training runtime | Prediction runtime for 1,260 users | RSS observed after fit |
|---|---:|---:|---:|
| Popularity statistics, shared by 3 variants | 4.315 s | 0.0046–0.0076 s per variant | 105.61 MiB |
| Personal-frequency tables, shared by 2 variants | 4.174 s | 1.603–1.671 s per variant | 185.68 MiB |

Whole baseline run peak RSS: **188.32 MiB**. Independent full verification peak RSS: **237.94 MiB**. Measurements are from `time.monotonic`, `/proc/self/statm` current RSS checkpoints, and Linux `resource.getrusage(...).ru_maxrss / 1024`. The personal model stores per-user/program count and seconds tables. Times include reading/hash-validating training inputs via the guarded Step 3 handle. They exclude the later independent-verification pass and report drafting. Python was 3.14.4; no third-party packages were installed.

| Model variant | Validation candidates | Distinct programs recommended across cohort | Candidate coverage | Users getting ≥1 popularity backfill slot | Backfill slots |
|---|---:|---:|---:|---:|---:|
| All popularity variants | 6,384 | 5 | 0.0783% | N/A: this is the global baseline itself | N/A |
| Personal — qualifying airings | 6,384 | 216 | 3.3835% | 14 | 61 |
| Personal — watched seconds | 6,384 | 533 | 8.3490% | 14 | 61 |

Ten cold users use the full popularity fallback in either personal-frequency variant. Four additional warm users have fewer than five distinct personal programs available in the validation catalog; together that makes 14 users with at least one popularity-completed slot. Global popularity recommendations are the popularity baseline itself for all 1,260 users, not a personalized fallback event.

## Recommendation-list overlap and training familiarity

The global distinct-user popularity variant makes the same Top-5 for every user. Personal-frequency lists vary substantially more: the event-count variant recommends 216 distinct programs across the cohort and averages **2.124 of five** IDs in common with the distinct-user popularity list; the seconds variant recommends 533 distinct programs and averages **1.575 of five** in common. Only 17 event-count lists and 13 seconds lists are identical to the distinct-user popularity Top-5.

Across all 6,300 personal recommendation slots, **6,239 (99.03%)** are programs in that user's qualifying training history. This is expected for a personal-history method, while popularity's five global items match each recipient's training history in 5,577 slots (**88.52%**). This describes logged familiarity only; it does not measure whether a program was enjoyed. A schedule may contain multiple titles for one ID, so displayed title/category use the first encountered validation-period schedule metadata for that ID; metadata was not a recommender feature.

## Provisional two-user error analysis

Selection was fixed without looking at validation labels or recommendations: among the original cohort with training history, the **heavy** example has the maximum number of distinct qualifying training programs; the **light** example has the minimum positive number among the remaining warm users. Ties follow template order. These IDs are saved in `provisional_two_users.json` for reuse across later methods.

### Heavy viewer — `13571199046623737884`

- Training history: **757** distinct programs, **2,358** qualifying broadcasts, **5,118,834** watched seconds. Leading examples by broadcast count include SpongeBob (`12001682`, Children, 118 broadcasts), `ТСН. Неделя` (`2400480`, Info, 93), and `Факты. Утро` (`240081`, Info, 88).
- Validation ground truth: **333 distinct programs**.

| Method | Top-5 (ID — title [category]) | Validation hits |
|---|---|---:|
| Popularity — distinct users | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги с Олегом Панютой [Info]; `6500479` — Мир наизнанку [Cognitive]; `2400467` — Викна-Новости [Info] | 5 |
| Popularity — qualifying airings | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `12001682` — SpongeBob [Children]; `2400508` — Сегодня. Итоги [Info]; `20088` — Завтрак с 1+1 [Entertainment] | 4 |
| Popularity — watched seconds | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `20088` — Завтрак с 1+1 [Entertainment]; `6500479` — Мир наизнанку [Cognitive] | 4 |
| Personal — qualifying airings | `12001682` — SpongeBob [Children]; `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `20088` — Завтрак с 1+1 [Entertainment] | 4 |
| Personal — watched seconds | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `12001682` — SpongeBob [Children]; `10003159` — Ничто не случается дважды [Series] | 4 |

### Light viewer — `8150816946177670084`

- Training history: **1** distinct program/broadcast, **2,753** watched seconds: `90053098`, Мужество в бою [Films].
- Validation ground truth: **15 distinct programs**.

| Method | Top-5 (ID — title [category]) | Validation hits |
|---|---|---:|
| Popularity — distinct users | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `6500479` — Мир наизнанку [Cognitive]; `2400467` — Викна-Новости [Info] | 1 |
| Popularity — qualifying airings | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `12001682` — SpongeBob [Children]; `2400508` — Сегодня. Итоги [Info]; `20088` — Завтрак с 1+1 [Entertainment] | 1 |
| Popularity — watched seconds | `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `20088` — Завтрак с 1+1 [Entertainment]; `6500479` — Мир наизнанку [Cognitive] | 1 |
| Personal — qualifying airings | `90053098` — Мужество в бою [Films]; `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `6500479` — Мир наизнанку [Cognitive] | 1 |
| Personal — watched seconds | `90053098` — Мужество в бою [Films]; `2400480` — ТСН. Неделя [Info]; `240081` — Факты. Утро [Info]; `2400508` — Сегодня. Итоги [Info]; `6500479` — Мир наизнанку [Cognitive] | 1 |

These lists illustrate that personal baselines keep previously watched items when eligible, then fill from globally common programs; global popularity reflects broad audience support. They do not support claims about user satisfaction. Full histories, all 333/15 validation labels, and metadata-labeled lists are in `results/step4/provisional_two_users.json`.

## Verification, recorded failures, and preservation

- `sh tests/run_step4_tests.sh`: **10/10 tests passed**, independently checking popularity aggregates/ranking, personal event/seconds ranking, cold and candidate-missing users, fallback, source provenance, deterministic tie behavior, unchanged string IDs, and the Step 3 evaluator. Test log: `results/step4_test_runs/step4_test_run_001.json`.
- Final `python3 code/verify_step4.py`: **2,651 checks passed, zero failed**. The verifier independently streams all Week 11–20 rows into `set`/`Counter` reference aggregates, compares all popularity score tables and all **2,520** per-user personal Top-5 rankings, validates every saved list through `evaluation.parse_prediction_records` and `EvaluationProtocol.evaluate`, recomputes all user APs/MAPs, fallback counts, list coverage, user hits and warm/cold MAP, and checks original/Step 1–3 source integrity and provenance. Final record: `results/step4/step4_verification_attempt_005.json` and required `results/step4_verification.json`.
- The original five data CSV hashes and frozen Step 3 configuration/truth/cohort hashes were rechecked; no Kaggle output exists in `submission/`.
- No future user-view file is registered or accessed. Validation interactions are accessed only by the Step 3 evaluator for scoring, never by fit functions. Validation-period schedule metadata is used only to display titles/categories in the two-user example.

**Verifier correction retained:** verification attempt 001 reported 2,492 failures because its reference-ranking loop appended fallback IDs even when a personal ranking already had five slots. Saved predictions correctly used only five. The reference was corrected to append fallback only if fewer than five eligible personal items exist. Attempt 002 passed the corrected ranking check; attempt 003 added list-familiarity checks; attempt 004 added full Step 1–3 preservation and OpenCode-export hash checks; attempt 005 checks that the requested root-level comparison CSV matches the run copy. Attempt 005 passed all 2,651 checks. All attempts are preserved.

A later attempt to rerun `python3 code/run_step4.py` stopped with `FileExistsError` because `results/step4/` already contained the successful baseline run. It did not change any outputs. The runner intentionally protects completed artifacts against overwrite; this event is recorded in `results/step4_failures.json`.

## Reproduction commands and artifacts

Environment: Python 3.14.4, standard library only, same Linux/i7 environment recorded in Step 1. No random seed is used by these deterministic count/ranking baselines. Run from the project root:

```bash
sh tests/run_step4_tests.sh
python3 code/run_step4.py
python3 code/export_step4_comparison.py
python3 code/verify_step4.py
```

The completed outputs already exist. `run_step4.py` refuses to overwrite `results/step4/`; use a deliberately selected new output directory before rerunning. Do not rerun into the existing evidence directory.

Important files:
- `code/popularity.py`, `code/personal_frequency.py`, `code/run_step4.py`, `code/test_step4.py`, `code/verify_step4.py`, and `code/export_step4_comparison.py`.
- `results/step4_baseline_comparison.csv`: required comparison deliverable, one row per score variant; an identical archived run copy is `results/step4/step4_baseline_comparison.csv`.
- `results/step4_verification.json`: required latest verification summary; detailed numbered attempts are retained under `results/step4/`.
- `results/step4/popularity_{distinct_users,qualifying_airings,watched_seconds}_predictions.csv` and per-user AP CSVs.
- `results/step4/personal_frequency_{qualifying_airings,watched_seconds}_predictions.csv` and per-user AP CSVs.
- `results/step4/step4_run.json`, `step4_outputs.json`, verification attempts, and `provisional_two_users.json`.
- `results/step4_test_runs/step4_test_run_001.json` and `results/step4_failures.json`.

`step4_outputs.json` stores byte sizes and SHA-256 hashes for the baseline predictions, per-user APs, run summary, comparison CSV and two-user evidence. Prediction hashes are also repeated in the comparison table. The baseline run SHA-256 is `cecd0a27b0f80d4bde01ff85ea8d22740911e3719004410ba410aa6431d9870d`; the root-level comparison CSV SHA-256 is `18a3caea4d8b1ec777b8e99a83ead081449832a555654d3808a15560139c3f9e`; the output manifest SHA-256 is `e7f14ee5c4744c0bbc0bc8027a30738f6bf5cf32804e0e74ec44d52a291b6093`.

Step 1, Step 2, Step 3, original datasets, the frozen Step 3 configuration/truth, and earlier completed reports remain intact.

## Interpretation and readiness

On this local threshold-defined hold-out, distinct-user global popularity provides a strong reference. Personal qualifying-airing frequency is higher by **0.019209 primary MAP points** and produces more varied lists; personal watched-seconds frequency is lower than the best popularity variant. Personal-frequency lists are more closely tied to logged history: 99.03% of personal recommendation slots were previously in that user's qualifying training history. List variation and MAP do not measure enjoyment, satisfaction, or business impact.

**Step 4 is ready for Step 5.** The required popularity baseline and additional personal-frequency reference have been individually evaluated, their predictions and per-user APs are saved, and independent count/ranking/metric checks pass. Personalized PageRank and Association Rules remain the next approved methods. No hybrid, final Kaggle file, or Kaggle submission was produced.
