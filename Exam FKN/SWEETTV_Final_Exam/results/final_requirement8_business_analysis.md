# Requirement 8 — selection of a recommendation strategy for SWEET.TV

## 1. Direct answer and scope of the recommendation

**I select V10 LightGBM LambdaRank as the final competition strategy for this
project.** It retrieves candidates through popularity, personal frequency,
Personalized PageRank, Association Rules and Content-Based Filtering, then
learns their ordering using behavioral, collaborative, content and schedule
features. Its genuine Kaggle private MAP@5 is **0.40637**, and its genuine
Kaggle public MAP@5 is **0.42051**, as confirmed by the user. These returned
competition scores are distinct from every local score discussed below.
The designated production model and submission are preserved unchanged. [S1–S3]

The justification is a combination of the scored competition artifact,
documented historical evaluation, complementary information sources and
feasibility on the recorded batch workload. It is not a claim that V10 wins
for every user or that adding more methods necessarily improves accuracy.
The two-user evidence contains important counterexamples: popularity and
other simpler methods match more of the heavy user's validation programs
than the historical V10 proxy. V10 is the selected **competition model**;
deployment as a service would require further evaluation of reliability,
contemporary user cohorts and operational cost. [S2, S3]

### Evaluation definitions

The original individual methods and the Requirement 7 examples share
Weeks 11–20 historical features and Weeks 21–30 validation viewing. A
qualifying occasion requires at least **120 unioned overlap seconds** on the
same channel and broadcast; `tv_show_id = 0` is excluded. The original
Step 3 primary target contains all distinct qualifying validation programs.
Its AP@5 denominator is `min(5, number of relevant programs)`. The separate
frequency-Top-5 target retains the programs with the greatest qualifying
airing counts. Neither local construction is asserted to reproduce the
hidden Kaggle relevance definition. [S1, S2, S4–S7]

The **V10-configuration temporal evaluation proxy** in Requirement 7 is not
the production V10 model. It has different learned weights and an earlier
fitting population and horizon: features from Weeks 11–15, supervised targets
from Weeks 16–20, and fitting strictly before Week 21. Its prediction inputs
use Weeks 11–20 features for the same Weeks 21–30 comparison. This avoids
scoring the production model on outcomes used in its historical fitting.
The saved V10 user-cross-fitted results exclude the evaluated user's labels
but fit other users' Weeks 21–30 targets; those results are reported separately
as user-held-out development/assessment evidence, not as the strict temporal
two-user proxy. [S2, S3]

## 2. Evidence from the same heavy and light viewers

The fixed users are **`13571199046623737884`** and
**`8150816946177670084`**, preserved as exact strings. They were originally
selected by historical activity, not recommendation accuracy: the heavy
user had the largest distinct-program history in the frozen cohort, while
the light user had the smallest positive history, with original cohort
order breaking ties. [S2]

| Method | Heavy matches | Heavy AP@5 | Light matches | Light AP@5 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 5/5 | 1.000000 | 1/5 | 0.100000 |
| Personal Frequency | 4/5 | 0.800000 | 1/5 | 0.066667 |
| Personalized PageRank | 5/5 | 1.000000 | 1/5 | 0.100000 |
| Association Rules | 5/5 | 1.000000 | 0/5 | 0.000000 |
| Content-Based Filtering | 0/5 | 0.000000 | 0/5 | 0.000000 |
| V10 temporal proxy | 4/5 | 0.800000 | 1/5 | 0.200000 |

Matches in this table use **all distinct qualifying validation programs**.
These are individual AP@5 values, not aggregate model MAP@5 or Kaggle scores.
The complete ordered lists, metadata and per-item match flags are in
`results/final_requirement7_examples.md`; no replacement examples were
generated for this analysis. [S2]

### 2.1 Heavy viewer: extensive history and familiar recommendations

User `13571199046623737884` has **757 distinct historical programs, 2,358
qualifying airings and 5,118,834 watched seconds** on qualifying airings. The
validation truth contains **333 distinct programs**. This breadth gives
several general-purpose lists many opportunities to match observed viewing;
it does not make the user representative of the whole platform. Summed
viewing seconds are log aggregates, not verified continuous attention. [S2]

Popularity recommends familiar news and cognitive programs, including
`240081` — **Факты. Утро.** and `6500479` — **'Мир наизнанку. Китай', 16 эп.**
All popularity recommendations occur in this user's history and validation
truth. Its success here is real, but it is not personalization: the same
global ordering is offered to every user. Personal Frequency instead leads
with `12001682` — **М/с 'Губка Боб Квадратные Штаны'.**, reflecting a strong
historical repeat signal. It also recommends `20088` — **'Завтрак с 1+1'.
Информационно-развлекательная программа.**, which has no qualifying validation
match. Past frequency is therefore useful evidence, not a guarantee that
viewing will continue. [S2, S4]

PPR moves news ahead of the cartoon and includes `6500479` instead of
Personal Frequency's unmatched `20088`. Its graph-based ordering matches
the complete list, although all recommended IDs were already seen in the
history. Association Rules produces the same set as the popularity list
in a different order and also matches the complete list. This is successful
familiar-content recommendation, not evidence of discovery. Many historical
antecedents support the rule scores; for example, `20088 → 2400480` has
support **0.363621**, confidence **0.961761** and lift **1.270394** in the saved
training-window transactions. [S2, S5, S6]

CBF selects series and entertainment, including `10003148` — **Т/с 'Стоматолог',
1 с.**, but none of its recommendations occurs in qualifying validation
viewing. Some were historically familiar and others were unseen. This
shows that the computed metadata similarity was insufficient to predict
this user's held-out choices; it does not show that the user disliked the
programs. The V10 temporal proxy reorders familiar items but retains the
unmatched `20088`, so its match count is below the simpler successful lists.
The different fitting horizon, Top-5 training objective and learned ordering
are relevant qualifications; the examples do not establish one exclusive
cause for the error. [S2, S7]

### 2.2 Light viewer: weak personal evidence and different kinds of extension

User `8150816946177670084` has **one qualifying historical program and one
airing, totaling 2,753 watched seconds**: `90053098` — **Х/ф 'Мужество в бою'.**
The validation truth contains **15 distinct programs**. This is a warm user
with sparse evidence, not a cold-start user. A single film cannot support a
confidently estimated broad preference profile. [S2]

Personal Frequency repeats that historical film, which has no qualifying
validation match, and uses the documented popularity fallback at **ranks
2–5**. Its matching ID, `240081`, is placed below the corresponding global
popularity position. PPR also retains the film, then assigns positive graph
scores to news and cognitive programs. Its `240081` recommendation is a
historically unseen ID that does match subsequent logged viewing. These
popular-looking graph outputs are not fallback fills: the saved computation
has enough positive-score candidates. [S2, S4, S5]

Association Rules recommends films beyond the user's direct history, such
as `90075859` — **Х/ф 'Экипаж'.** The saved rule `90053098 → 90075859` has
support **0.005877**, confidence **0.485816** and lift **23.158204**. Its strong
relative association can raise it above common news under confidence–log-lift
scoring, but the recommended films have no qualifying validation matches.
CBF repeats the historical film and retrieves metadata-similar films,
including `90082673` — **Премьера. Х/ф 'Кролик Джоджо'.** These also fail to
match the observed validation set. Both methods extend a limited profile
in algorithmically plausible ways; plausibility is not observed accuracy.
The evidence does not identify why any particular film was not watched. [S2, S6, S7]

The V10 temporal proxy places `240081` first. It does not increase the number
of matching programs relative to popularity, but its higher position produces
a higher AP@5. This separates **candidate generation**—making an eligible
program available for recommendation—from **ranking quality**, which
determines its position. **Historical novelty** means absence from the
qualifying training log; an **actual validation match** requires subsequent
qualifying viewing. An unseen film with no match cannot be counted as a
successful discovery. Even an unseen matching item does not establish that
the recommendation caused viewing or that the user enjoyed it. [S2]

## 3. Algorithmic comparison

### Popularity and Personal Frequency

Popularity aggregates historical audience support and precomputes one global
ranking over eligible programs. Counting is linear in the interaction rows;
sorting the catalog is inexpensive relative to pair-mining or iterative
graph scoring, and the shared list can be cached. Its strengths are simple,
deterministic coverage and a defensible fallback when personal evidence is
absent. Its central limitation is lack of individual preference information,
visible in the light user's largely unmatched global list. [S4]

Personal Frequency sorts the user's eligible historical programs by qualifying
airing count and keeps repeated programs when allowed. It directly models
repeat-viewing preference, but cannot retrieve a new program from personal
counts alone. Sparse histories require backfill; absent histories require the
complete shared popularity fallback. The heavy user's unmatched repeat and
the light user's mostly backfilled list demonstrate these limitations. [S2, S4]

### Personalized PageRank

PPR builds a program graph from historical co-view relationships. Cosine
normalization accounts for historical audience sizes; a frequency-weighted
restart distribution anchors diffusion to the user's own history. Iterative
transitions can reach programs the user did not directly watch. Hubs and
widely connected programs can still receive substantial mass, so personalized
graph computation need not produce niche recommendations. The implementation
uses sparse transitions, capped history for graph pair generation and a
bounded neighborhood to control cost. It is a collaborative signal, not a
content-similarity calculation or a calibrated probability of enjoyment. [S5]

### Association Rules and the conceptual contrast with Matrix Factorization

A rule has an **antecedent** program and a **consequent** program. In the
historical user-window baskets, support is the fraction containing both;
confidence is the conditional fraction containing the consequent among
baskets containing the antecedent; lift divides that confidence by the
consequent's marginal basket frequency. These describe explicit co-view
relationships, not causal influence. The implemented scorer aggregates
confidence multiplied by log-adjusted lift, with historical log-frequency
antecedent weights. The light-user film rule illustrates why an item can
have high lift without being the next program this particular user watches.
The saved rule table makes that relationship inspectable. [S6]

**Matrix Factorization was not implemented as an independently evaluated method**
in this project, and there is **no empirical comparison with it**. Conceptually,
latent-factor methods learn low-dimensional user and item representations
from an interaction matrix and score their compatibility. They can share
information across many interactions without exposing a specific antecedent–
consequent rule. Association Rules instead exposes local co-occurrence
statistics directly. This difference in representation does not establish
that either approach would outperform the other on this dataset.

### Content-Based Filtering

CBF represents program categories, genres, titles, directors, actors,
production-year information and channels in a sparse, normalized TF-IDF-style
space. Vocabulary and IDF are fitted only on historical schedule metadata;
the user profile weights historical item vectors by qualifying airings in
the selected configuration. Metadata may provide a candidate-generation
route when collaborative support is weak, provided the program has usable
attributes in the fitted vocabulary. It does not solve absence of a user
profile. For these two users, the positive similarities did not translate
into validation matches. That observation limits claims for standalone CBF;
it does not isolate its marginal contribution within a learned hybrid. [S2, S7]

### V10 LambdaRank: retrieval followed by learned ordering

V10 preserves the original **Top-50 union from five generators**: popularity,
Personal Frequency, PPR, Association Rules and CBF. Source lists and training-
only popularity completion are deduplicated and restricted to eligible
program IDs. Generator scores/ranks, source-membership indicators, historical
frequency and recency, completion and weekly consistency, popularity,
metadata affinity, and target-schedule availability form the feature groups.
The **77-feature** schema also contains an audience-support-adjusted PPR
feature and the recent completion-week streak. Schedule features describe
availability, not future user viewing. [S3, S8]

The ranker learns tree-based score interactions within user query groups;
there are **no manually assigned fusion weights** for these components in
V10. LambdaRank uses a pairwise ranking surrogate rather than directly
optimizing MAP@5 as a smooth loss. The fixed configuration is **78 boosting
rounds, 15 leaves, learning rate 0.05, minimum leaf size 100, L2 regularization
5.0, binary gains [0, 1], truncation level 8, seed 2026 and four threads**.
Final ties use historical popularity rank and numeric program ID. Users
without a qualifying full historical profile ignore learned scores and use
the shared popularity fallback. [S3, S8]

The configuration's provenance matters. Both V10 score-calibration families
retained **zero unseen adjustment** on development validation. A single fixed
LambdaRank configuration was then run as the conditional experimental
fallback; it was not the best model on development MAP. The original
assessment was repeatedly examined and is secondary, not pristine. The
current final designation follows the subsequently supplied genuine Kaggle
result, not a claim that V10 won every original local comparison. [S3]

## 4. Historical validation and final-model justification

For the individual methods, the following scores were recomputed from saved
predictions on the common **1,260-user** cohort. Both columns are local
MAP@5; their different relevance definitions must not be mixed. [S4–S7]

| Method | All-distinct qualifying-program MAP@5 | Frequency-Top-5 MAP@5 |
| --- | ---: | ---: |
| Popularity | 0.768418 | 0.306627 |
| Personal Frequency | 0.787627 | 0.434132 |
| Personalized PageRank | 0.815151 | 0.429489 |
| Association Rules | 0.771331 | 0.313643 |
| Content-Based Filtering | 0.495029 | 0.200193 |

PPR is strongest among these selected standalone methods on all-distinct
truth, while Personal Frequency is stronger on the frequency-Top-5 target.
That ordering difference is direct evidence that target definition matters.
It is inappropriate to compare the larger all-distinct values directly with
Kaggle private MAP or with the broader-population ranker experiment. [S4–S7]

V10's saved user-cross-fitted frequency-Top-5 MAP@5 is **0.408498 on 3,901
development users** and **0.409136 on 992 assessment users**. V8's recorded
development result in that comparison is **0.409652**, above V10. These
broader-cohort results are **not directly comparable** with the standalone
table or the strict temporal two-user proxy. They confirm that offline
validation did not consistently predict the model ordering subsequently
observed on genuine Kaggle evaluation. [S3]

Multiple generators give the ranker access to different forms of evidence:
direct repetition, audience co-view structure, explicit rules and metadata
similarity, together with globally supported candidates. The rationale is
flexible learned ordering over that combined pool, rather than a proven
incremental gain from every component. The experiments and examples do not
provide a causal component-by-component attribution. In particular, the
standalone CBF results and the two-user outcomes do not justify claiming
that metadata necessarily improved V10. [S2, S3, S7, S8]

## 5. Computational cost and operational trade-offs

The recorded hardware context is an **Intel Core i7-6600U at 2.60 GHz, two
physical cores and four logical CPUs, with approximately 11 GiB RAM**.
Historical Step 4 used Python 3.14.4 and the standard library; Steps 5, 6
and 9 recorded Python 3.14.4, NumPy 2.5.4 and SciPy 1.18.1. The surrounding
V8/V9 LightGBM pipeline records Python 3.12.14, LightGBM 4.6.0 and NumPy
2.2.6; V10's manifest does not separately repeat a full environment record.
Those are provenance records, not a new matched-environment benchmark. [S3–S9]

| Method | Recorded preparation/training | Recorded prediction/generation | Recorded process peak and scope | Sources |
| --- | --- | --- | --- | --- |
| Popularity | 4.315 s, shared popularity statistics | 0.006307 s, selected variant, 1,260 users | 188.32 MiB, entire shared Step 4 run | [S4] |
| Personal Frequency | 4.174 s, personal tables; shared popularity preparation separate | 1.603169 s, selected variant, 1,260 users | 188.32 MiB, same shared Step 4 run | [S4] |
| Personalized PageRank | 6.738 s, selected profile/pair/graph preparation; plus 4.102 s popularity fallback preparation | 14.665256 s, 1,260 users | 468.60 MiB, entire staged Step 5 search | [S5] |
| Association Rules | 17.882 s, selected model preparation | 21.337538 s, 1,260 users | 3,103.86 MiB, entire Step 6 comparison | [S6] |
| Content-Based Filtering | 0.600195 s, selected metadata-space fit/transform, not full pipeline preparation | 13.725104 s, selected profile construction and ranking, 1,260 users | 2,063.10 MiB, entire Step 9 comparison | [S7] |
| V10 LambdaRank | Fit-only not separately recorded; complete V10 experiment 90.271 s | Score-only not separately recorded; cached-feature final generation 22.834 s, 1,260 users | 961.61 MiB, experiment; 421.60 MiB, final generation | [S3] |

These figures are **not strictly comparable** and are **not per-request service
latencies**. The shared and multi-configuration peaks are not isolated model
heaps. V10's complete experiment includes replay, diagnostics, calibration,
multiple fits and evaluation, while final generation reuses verified feature
caches and performs additional checks. It does not measure an end-to-end
uncached retrieval/feature-building service. Missing isolated V10 fit and
scoring times are not inferred from these aggregate records. [S3–S7]

The implementation trade-offs follow the algorithms. Popularity stores
small global aggregates; Personal Frequency stores user–program counts.
PPR additionally needs sparse graph construction and iterative scoring.
Rule preparation expands co-occurrence pairs in historical baskets and
retains rules per antecedent, explaining its larger recorded comparison
footprint. CBF requires metadata normalization, sparse item/profile vectors
and similarity calculations. V10 adds candidate-feature construction and
supervised tree fitting, but scores a bounded union rather than every
catalog item. All were feasible for the recorded offline workloads on the
available machine. Caching, incremental updates and serving architecture
would need separate engineering and measurement before any latency or
capacity commitment. [S3–S8]

## 6. Business recommendation and evidence-supported limitations

For the competition, I would retain **V10** with its documented candidate
generators and common fallback. Popularity provides coverage when personal
evidence is absent; historical frequency preserves repeat preferences;
PPR supplies collaborative graph evidence; rules provide inspectable
co-view associations; CBF supplies metadata-based candidates; LambdaRank
provides learned ordering across these signals. For a platform service,
this is a candidate design to evaluate further, not a production-readiness
claim. Contemporary chronological evaluation, sparse/cold-user monitoring,
catalog-coverage checks and operational measurements would be required.

The user-experience implications must remain conditional. Familiar programs
may make lists recognizable, but can repeat programs whose viewing has
stopped. Graph/rule/content candidates may broaden the list, but novelty
alone is not a successful outcome. Popularity can provide eligible coverage
yet underserve specific interests. The light example illustrates that moving
a single genuine match upward can improve ranking accuracy without increasing
the number of matches. None of these observations establishes that the
recommendations caused viewing.

The following limitations constrain the interpretation:

1. **Incomplete item labeling.** Only 115,613 of 1,074,235 rows in the first
   historical schedule contain positive program IDs: approximately **10.8%**.
   Unlabeled airtime cannot be reconstructed as a known program recommendation
   from these inputs, and `tv_show_id = 0` cannot be recommended. [S1, S9]
2. **Assumed viewing threshold.** The 120-second rule is a local modeling
   assumption, not a demonstrated preference or satisfaction threshold.
   Shorter viewing is excluded, and the hidden competition target remains
   unconfirmed. [S1, S4–S7]
3. **Behavioral change.** Historical repeats need not continue. The previously
   frequent `20088` and the light user's historical `90053098` have no
   qualifying validation matches in the examples. [S2]
4. **Sparse and cold histories.** The light user has limited personal evidence;
   truly cold users have none and receive the common popularity fallback.
   Metadata without a user profile does not by itself identify preferences. [S2, S8]
5. **Retrieval ceiling.** In the original broader V10 candidate pool, **20,263
   of 23,831 relevant Top-5 labels** are reachable, giving **85.03% candidate
   recall**. A relevant item outside the pool cannot be recovered by reranking
   alone. This is a local candidate diagnostic, not Kaggle test recall. [S3, S8]
6. **Novelty versus relevance.** Previously unseen programs need not be useful
   recommendations; both users' CBF lists illustrate unmatched alternatives.
   The rule-generated films for the light user are also unmatched. [S2]
7. **Log coverage.** Absence from this platform's qualifying history does not
   prove a program was never watched elsewhere or below the threshold. [S1, S2]
8. **No enjoyment inference.** Viewing does not prove enjoyment or satisfaction;
   an unmatched item is not evidence of dislike. [S1, S2]
9. **Illustrative sampling.** Two deliberately contrasting users are not a
   statistically representative experiment. The heavy user's much broader
   validation set also makes raw match opportunities different. [S2]
10. **Evaluation mismatch and reuse.** Offline target definitions, populations
    and previously examined validation outcomes differ from genuine Kaggle
    evaluation. A higher local score does not guarantee a higher private score. [S3–S7]
11. **Proxy limitations.** The strict historical V10 proxy has different learned
    weights, an earlier fitting horizon and a different fitting population
    from production V10. Its per-user results cannot be presented as the
    production model's historical performance. [S2]
12. **Operational evidence is incomplete.** Recorded batch times and shared
    process peaks do not establish online latency, reliability or scalability.
    **No live A/B test was performed. No revenue, retention, satisfaction or
    causal business improvement was measured.** [S1, S3–S7]

## 7. Conclusion

V10 is the justified final **competition** strategy because it is the
designated, genuinely scored learned reranker with a transparent retrieval
pipeline, verified artifacts and manageable recorded offline execution.
Its choice coexists with stronger simpler-method results for particular
users and some local targets. The appropriate platform conclusion is to
retain reliable popularity coverage, exploit repeat and collaborative
signals, and evaluate learned ordering carefully. The evidence supports
continued testing of that design, not universal superiority or measured
business impact. No production model was retrained or modified for this
report, and no new Kaggle submission was created.

## 8. Evidence references and verification notes

The references below are project files, not external studies. Numerical
comparisons were checked against actual prediction files and machine-readable
records before writing. Full ordered examples remain in Requirement 7.

- **[S1] Exam requirements:** `exam_quiz_FKN.md`, especially §§3, 5, 7 and 8.
- **[S2] Verified examples:** `results/final_requirement7_examples.md`;
  `results/final_requirement7_evidence/examples.json`, `verification.json`,
  `report_verification.json` and `v10_temporal_evaluation_config.json`.
  These identify actual predictions, source schedule rows, viewing matches,
  rule evidence, fallback positions and proxy provenance.
- **[S3] V10:** `results/v10_experiment/experiment.json`, `plan.json`,
  `selection.json`, `submission_manifest.json`, `selected_all_crossfit.csv`,
  and `README.md`; implementation in `code/run_v10.py` and
  `code/generate_v10.py`. Experiment timing is `seconds`/`peak_rss_mib` in
  `experiment.json`; generation timing uses the same fields in
  `submission_manifest.json`. Genuine Kaggle scores are user-confirmed
  returned scores, not locally calculated or newly queried here. The old
  V10 README predates those results; its former recommendation is preserved
  as historical evidence, not adopted as the current designation.
- **[S4] Baselines:** `results/step4_baseline_report.md` and
  `results/step4/step4_run.json`, specifically `training_runtime_seconds`,
  `prediction_runtime_seconds`, `peak_rss_mib` and `variants`;
  `results/step4/popularity_distinct_users_predictions.csv` and
  `personal_frequency_qualifying_airings_predictions.csv`;
  `code/popularity.py` and `code/personal_frequency.py`.
- **[S5] PPR:** `results/step5_ppr_report.md` and
  `results/step5/ppr_model_config.json`, particularly `best_metrics` and
  `peak_process_rss_mib`; selected predictions in
  `results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv`;
  implementation in `code/ppr.py`.
- **[S6] Association Rules:** `results/step6_association_report.md`,
  `results/step6/association_best_config.json` and `association_run.json`.
  Preparation and prediction times are in the best-config record; the run
  record supplies the whole-comparison peak. Predictions and actual rule
  statistics are in `results/step6/experiments/time_window_14d_lift_log/`;
  implementation in `code/association_rules.py`.
- **[S7] CBF:** `results/cbf_step9_report.md` and
  `results/step9_cbf/step9_validation_results.json`. Preparation uses
  `feature_spaces.channel_context_emphasis.fit_and_transform_seconds`;
  prediction uses the selected experiment's `prediction_runtime_seconds`;
  the run-level `peak_rss_mib` covers the whole comparison. Saved selected
  predictions are `results/step9_cbf/predictions/cbf_channel_context_emphasis_airings.csv`;
  implementation in `code/content_based.py` and `code/run_cbf_step9.py`.
- **[S8] Candidate/features and targets:** `code/build_v8.py`,
  `code/v6_data.py`, `code/train_v9.py`, `results/v8_experiment/validation.npz`,
  `labels.json` and `results/v9_experiment/test_feature_manifest.json`.
  Candidate recall was recomputed from original candidate IDs and Top-5
  labels. Original-method evaluation uses `data/step3/evaluation_config.json`
  and `validation_truth.csv`.
- **[S9] Hardware/data provenance:** `results/step1_audit.md`,
  `results/step1_profile.json`, the original
  `../sweettv-tv-program-recommender/export_arh_11-20-final.csv`, and
  `results/v8_experiment/submission_manifest.json` for the recorded
  LightGBM-pipeline software environment. Positive-ID schedule rows were
  counted directly for this analysis.

`results/final_requirement6_verification.md` was **not available**. This
section instead verifies its cited source artifacts directly and makes no
claim that a separate Requirement 6 verification document exists.

Audit outputs are `results/final_requirement8_evidence/verified_measurements.json`,
`report_claims.json`, `preparation_verification.json`, `report_verification.json`
and `preservation_before.json`. The verification checks user identities,
reported values and titles, actual viewing matches, proxy/production
separation, measurement scope, unmeasured-outcome statements and preservation
of the existing evidence and original session exports. No additional model
fit, new recommendation example or Kaggle submission forms part of this task.
