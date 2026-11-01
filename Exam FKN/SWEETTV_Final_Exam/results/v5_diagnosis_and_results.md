# SWEET.TV: V1–V4 diagnosis and experimental V5

## Decision

**V5 is generated and independently checked. V2 remains the designated final submission until a genuine better Kaggle result is returned.**

User-reported private/public scores: V1 **0.37269/0.39316**, V2 **0.37880/0.40074**, V3 **0.35282/0.37196**, V4 **0.35210/0.36523**. These motivated an audit, not label reconstruction or hyperparameter selection. V5 has no Kaggle score; the 0.44801 private-score objective is not claimed as achieved.

## 1. Independent audit

Evidence: `results/v2_v4_diagnosis/audit.json`, `paired_users.csv`, `paired_programs.csv`, `feature_distribution_shift.csv`, and `existing_test_output.txt`. Reproduce with `.venv-lightgbm/bin/python code/audit_v2_v4.py` in a fresh output location (the script protects its completed audit).

### Matching, identity, metric and chronology

- **Join correctness:** independently read all raw views for four selected users (including the same heavy/light example users), localized timestamps separately, merged overlapping view fragments and directly matched same-channel airings. All **5,448** resulting overlaps matched the saved history cache exactly, from **20,707** historical viewing rows. This is a four-user raw-data audit, not a fresh raw join of every user.
- **Whole-cache bounds:** scanned 1,823,893 train, 1,191,015 validation and 3,014,157 history overlap rows. No invalid positive-ID, time-window or overlap-duration bounds were found. ID zero is excluded; fragments are summed per broadcast after union, rather than pooling different broadcasts to cross a threshold.
- **Calendar-week defect:** fixed 604,800-second buckets are not calendar weeks across DST. **5,063 training rows** (4,831 user-program pairs) and **12,496 full-history rows** (11,617 pairs) differ from local calendar assignment. The ten-week temporal split boundaries themselves are correct. Week features and a fixed-window association transaction are distinct policies; the AR 14-day UTC window is intentionally fixed, not claimed to be an ISO week.
- **Airing versus viewing week:** 3,921 training and 6,734 history rows have different broadcast-start and first-view weeks because an airing can cross a week boundary. V5 explicitly assigns behavioral aggregates by first observed overlap's local calendar date. Schedule statistics use the airing start/target-overlap date. Neither should be silently conflated with the filename.
- **Test schedule spillover:** the named Week31–42 file includes 2,399 rows starting in Week43, including **257 positive-duration labeled airings**. V4's schedule features counted them and clamped their week bucket to the final target week. V5 clips schedule feature counts to the actual target interval. Candidate IDs still come from the named schedule file, preserving the common eligible pool.
- **IDs:** V1–V4 retain every original string user ID in original order. Converting through float64 would alter **1,256 of 1,260** IDs, but no such conversion was found in the submitted identity paths. IDs remain excluded from numeric features.
- **MAP@5:** an independent rational relevant-item/prefix implementation agrees with the production evaluator on 1,000 additional randomized cases and all paired validation comparisons (within floating-point tolerance). The 60 existing join/evaluation tests pass. AP denominator remains `min(5, |R|)` and empty-label users remain in the average.

### Official relevance rule versus hypotheses

The live Kaggle Data URL returned only its JavaScript page shell. The accessible official CompetitionService record confirms **MAP@{K}**, not how hidden relevance is constructed. Attempts to retrieve additional unauthenticated competition-data/page endpoints returned 404. Prior Data-list access returned 401. Thus **the 80% rule is unconfirmed**, not disproved. No hidden labels or other participant predictions were accessed.

Both local definitions were reconstructed from the same verified validation-airing cache:

1. Historical **120-second** rule: all distinct qualifying programs (original primary target), and separately the five most frequent qualifying programs (existing secondary target).
2. **80%-completion hypothesis:** count an airing only when `5 × unioned_overlap_seconds >= 4 × full_airing_duration`, then rank programs by count and retain five; numeric ID breaks frequency ties.

There are 1,246 nonempty Top5-120 users versus 1,238 Top5-80 users. Their relevant sets overlap by **2.979 programs per user on average**. The original all-120 target averages **157.55 relevant programs/user**, making it much easier to obtain five relevant hits.

### Same 256 users, candidate pool and labels

| Model strategy | All-distinct 120s MAP@5 | Top5-120 MAP@5 | Top5-80 hypothesis MAP@5 |
|---|---:|---:|---:|
| V1 strategy / PPR | 0.804440 | 0.443932 | 0.355436 |
| V2 PPR + PF | 0.796693 | **0.460977** | 0.349505 |
| V3 PPR + PF + CBF | 0.787930 | 0.430078 | 0.329792 |
| V4 LightGBM | **0.817891** | 0.452552 | 0.346058 |

The broad all-120 target favors V4 despite its worse Kaggle result. Narrower Top-5 definitions are more plausible diagnostics of the task than the broad proxy, but **neither reproduces the Kaggle ordering**: Top5-120 ranks V2 > V4 > V1 > V3; Top5-80 ranks V1 > V2 > V4 > V3; Kaggle ranks V2 > V1 > V3 > V4. With only four aggregate scores and different target weeks, it would be unjustified to select the label definition by how closely it fits Kaggle. V5 therefore retains Top5-120 as the main predeclared selection target and treats 80% as a separate experiment.

### Where V4 loses

V4 worsens **98** of the 256 users, improves 72 and ties 86 on Top5-120. The paired delta is **−0.008424**, with a descriptive user-bootstrap 95% interval **[−0.023008, +0.006432]**; the local loss is not statistically conclusive. Five cold users have identical fallback results. Heavy users (101+ historical programs) account for most losses: 84 worsen and 56 improve; average delta −0.012381.

Examples (all local validation, not hidden Kaggle user outcomes):

| User | V2 AP | V4 AP | Relevant programs dropped |
|---|---:|---:|---|
| `15896072116603972829` | 0.800000 | 0.400000 | `12002896`, `12002955` |
| `3469248566758145871` | 0.483333 | 0.100000 | `5200303`, `6500479` |
| `15337315789178217621` | 0.710000 | 0.383333 | `400357` |

Largest negative AP-contribution changes include `2400480` (−0.009258 MAP contribution), `2400508` (−0.005273), `12001682` (−0.004271), `200432` (−0.003008), and `400361` (−0.002396). These contributions depend on preceding ranks; they are not isolated causal item effects. For `2400480`, no relevant recommendation was dropped and one was gained, but its AP contribution still fell: ordering matters.

**Candidate exclusion is ruled out for V2 recommendations:** all **1,280/1,280** V2 validation slots for these users and all **6,300/6,300** V2 final-submission slots are present in V4's candidate unions. The loss is reranking within available candidates.

### Feature and training defects/risks

- Recent-2/4/8-week counters are **zero for every row** in both saved V4 feature tables: the builder never increments them.
- V4 learns on 10-week counts but receives 20-week behavior and 12-week availability at final inference. **51.95%** of inference rows exceed the training maximum for target active weeks; **21.02%** exceed the training maximum for viewing recency. This is a measured distribution shift, not proof of its share of the Kaggle loss.
- V4 stopped after **four trees** using class-weighted binary training and inner-fold binary logloss rather than MAP. Coarse scores tie at the fifth/sixth position for 88 held-out users. Validation breaks ties by numeric ID, final inference by popularity then ID; applying the latter to validation changes 218 rankings and yields 0.449831 instead of 0.452552 Top5-120 MAP.
- No future-view feature or supervised outer-user label leakage was found. All-user Week21–30 labels in the final reranker are legitimate historical labels for Week31–42 inference. However, prior generator choices and repeated project-wide experiments used this validation population. The 256-user outer group is disjoint for the current supervised fit, **not a pristine never-before-used research holdout**.
- V3 already underperformed V2 locally when adding CBF. Complexity was not evidence of improvement. Its live outcome supports retaining V2, but cannot identify individual hidden-test losses.

## 2. V5 corrections and controlled experiment

New code and outputs are separate from all old models/submissions. Existing candidate sets and upstream PPR/PF/AR/CBF signals are reused; the expensive original matching and model generators were not rerun.

**Corrected features:** local-calendar week ages; real recent 2/4/8-week counts; 50%, 80%, 95% completion counts/shares; average completion; last qualifying watch week; recent-versus-old frequency trend; active-week consistency; repeat-airing share; user-relative week stickiness; weekly count variation; personal airing share; historical schedule airing frequency; target-to-history airing-rate ratio; missing-history/availability flags; recent channel affinity. There are 75 numeric columns, including 25 additions.

Behavioral additions use **trailing ten calendar weeks** both at validation and final prediction: Weeks11–20 for the former and Weeks21–30 for the latter. Candidate generators retain full permitted history through Week30 for final inference. Full-history signals and corrected rolling behavioral features have distinct documented roles. Target schedule identities are deduplicated and counts normalized to a common ten-week equivalent. V5 conservatively excludes 209 full-history overlap rows straddling the rolling cutoff because their exact within-airing fragment locations cannot be reconstructed from the aggregate alone.

This is not a complete independent attribution experiment: several defects and the training objective/stopping policy were changed together. Their individual accuracy contributions remain unmeasured.

### Selection and measured outcomes

- Binary LightGBM, seed 2026, four CPU threads, deterministic column-wise training; no class weighting; learning rate 0.05; L2=5; minimum leaf data=100.
- Three configurations: 15 leaves; 31 leaves; 31 leaves with direct CBF signals removed. No-CBF still uses the existing shared candidate union, so it is not a complete generator ablation.
- Three inner user-group folds across 1,004 development users; early stopping on **Top5-120 MAP**. Maximum 160 rounds, patience 20.
- RRF blend weights LightGBM/V2 = .25/.75, .50/.50, .75/.25, 1/0, selected using development OOF predictions only. Tie-breaking is popularity then numeric ID in both validation and final inference.
- Winner: **15 leaves, 78 rounds, LightGBM weight 1.0** (no blend). OOF Top5-120 MAP=0.468416. No multi-seed ensemble was run.

| Same 256 outer users | All-120 MAP@5 | Top5-120 MAP@5 | Top5-80 MAP@5 |
|---|---:|---:|---:|
| Personal Frequency | 0.777057 | 0.451081 | 0.337897 |
| PPR | 0.804440 | 0.443932 | 0.355436 |
| V2 RRF | 0.796693 | 0.460977 | 0.349505 |
| V4 before corrections | 0.817891 | 0.452552 | 0.346058 |
| **V5 selected model** | **0.827474** | **0.488633** | 0.378711 |
| Separate 80%-label LightGBM diagnostic | 0.797565 | 0.443646 | **0.404772** |

The 80%-label model uses the main selected hyperparameters but different historical labels; it was not eligible to replace the frozen main winner after seeing outer outcomes. This illustrates that the chosen relevance definition strongly changes measured accuracy.

V5 improves Top5-120 by **+0.027656 over V2**, with 94 users better and 44 worse. Paired bootstrap interval: **[+0.016406, +0.039378]**. This is encouraging local evidence, subject to the inherited/reused-holdout caveat, and does not imply a private Kaggle score above 0.44801.

**Candidate recall:** unchanged Top50 unions cover 0.893012 of secondary labels across all original users (micro), and **0.895276 on the 256 outer users**. Top20 was 0.828112 over the original cohort. Candidate counts average 108.69 for validation and 110.73 for final inference; no user has fewer than five.

**Importance by gain:** personal-frequency rank 95,137; PPR score 33,169; PPR rank 26,971; target airing count 8,454; target active weeks 5,062; target/history airing-rate ratio 4,024; recency 3,312; channel affinity 3,274; corrected recent-two-week count 2,399. Nonzero gain confirms model use, not causal benefit.

## 3. Verification, artifacts and reproducibility

- **60 existing join/evaluator tests passed**, plus **4 new V5 regression tests passed**.
- **16 validation checks passed** before generation; **21 final checks passed**, zero failed. Checks include 48 independently recalculated user-program completion/weekly feature samples, independent rational AP, frozen production-evaluator agreement, model replay, disjoint supervised groups, inner-only selection, exact template order, schedule eligibility, five unique canonical IDs, and checksums of protected submissions/evidence.
- Scope limit: shared candidate generators are not independently reimplemented; feature arithmetic is sampled, while prediction/format/metric checks cover all saved rows/users. No synthetic checks are claimed as full raw-data verification.
- No V5 test failures occurred. Existing defects documented above remain in historical experiment code/results; corrections are isolated in the V5 modules.
- Feature preparation: **85.39 s**. Training, inner CV, blend comparisons, diagnostic fit and final refit: **97.27 s**, peak RSS **789.07 MiB**. Submission generation: **5.55 s**. Final verification: **23.27 s**.

**Experimental CSV:** `submission/final_submission_v5.csv`  
**SHA-256:** `bb8aa11ae3d5276d70deddc697752c5d8bce358b17fe8b08c235cba8265e4d40`

1,260 original users; five distinct positive IDs each; eligible pool 6,636; exact columns `user_id,tv_show_id`. Configuration: `results/final_model_v5_manifest.json`. Models, comparisons, feature provenance and verification: `results/v5_experiment/`. V1–V4 and native session exports match their pre-audit hashes. No upload was made.

```bash
.venv-lightgbm/bin/python code/test_v5.py
.venv-lightgbm/bin/python code/features_v5.py
.venv-lightgbm/bin/python code/train_v5.py
.venv-lightgbm/bin/python code/verify_v5.py
.venv-lightgbm/bin/python code/generate_v5.py
.venv-lightgbm/bin/python code/verify_v5.py
```

Creation commands protect their completed artifacts; reproduce into fresh output locations rather than deleting evidence.

## 4. Three independently testable next improvements

1. **Calendar/recency feature ablation:** on fresh historical rolling-origin folds, compare original V4 features with only calendar/real-recent-count corrections. Keep model objective, candidate pool and parameters fixed to measure their contribution separately.
2. **Horizon-shift ablation:** compare full-history raw counts against trailing-ten-week rates and target-per-week availability, holding all other features fixed. Evaluate on a second later historical cutoff to test whether normalization transfers beyond this reused split.
3. **Ranking-oriented stopping and conservative fusion:** compare four-round logloss stopping against MAP-based stopping and V2 blend weights using inner folds only, reserving an untouched chronological evaluation window. Do not select the 80% label model using Kaggle scores; assess both hypotheses independently.

Remaining exam deliverables: manually obtain a genuine V5 Kaggle result if desired; retain the best genuine scored submission; finalize the required PDF with repository URL, authentic private-score screenshot, consistent two-user business analysis and limitations; export the authentic full OpenCode session without editing it. No A/B, satisfaction or revenue improvement is claimed.
