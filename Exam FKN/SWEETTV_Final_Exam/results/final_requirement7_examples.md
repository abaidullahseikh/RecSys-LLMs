# Requirement 7 — verified recommendation examples for the same two users

**Designated final model:** V10 LightGBM LambdaRank. **Genuine Kaggle private
MAP@5: 0.40637**, as supplied by the user. This section compares historical
recommendations; its per-user hits and AP@5 values are not Kaggle scores.
The production V10 model and its Kaggle submission remain unchanged.

## 1. Common historical protocol and comparison scope

All six lists for each user predict the **same Weeks 21–30 validation window**
using recommendation features available by the end of Week 20:

- Historical features: Weeks 11–20, UTC `[1583704800, 1589749200)`.
- Validation: Weeks 21–30, UTC `[1589749200, 1595797200)`.
- Frozen Step 3 protocol:
  `40f6d1c93367106b22d3b66a76605afd12a1ec929f4fe7929bd137749880f239`.
- Common eligible universe: **6,384 positive program IDs**, from the frozen
  schedule-only validation pool. Previously watched programs are allowed.
- A qualifying viewing is at least **120 unioned overlap seconds per aired
  program**, matched on channel and time. Fragmented views of one broadcast
  are combined before applying the threshold. `tv_show_id = 0` is not an item.
- **Actual validation match** means the ID occurs among *all distinct*
  qualifying validation programs in `data/step3/validation_truth.csv`.
  The truth is not truncated to five for the main comparison.
- AP@5 uses denominator `min(5, number of relevant IDs)`. A separate
  frequency-Top-5 diagnostic is explicitly labeled below.

One previously selected configuration represents each method family. These
configurations were selected in the original experiments over their complete
validation cohort, not chosen to make these two examples look favorable.

| Family | Exact configuration and saved recommendation source |
| --- | --- |
| Popularity | Distinct historical viewers, ties by qualifying airings then numeric ID; `results/step4/popularity_distinct_users_predictions.csv` |
| Personal frequency | Qualifying-airing count, numeric-ID tie break; `results/step4/personal_frequency_qualifying_airings_predictions.csv` |
| Personalized PageRank | `cosine_pfrequency_a0.35_k30`: cosine co-view graph, history cap40, minimum3 common viewers, top30 neighbors before symmetrization, frequency restart, restart probability0.35; `results/step5/experiments/cosine_pfrequency_a0.35_k30/predictions.csv` |
| Association Rules | `time_window_14d_lift_log`: 14-day historical baskets, confidence×log(1+min(lift,100)), log-frequency antecedent weights; support≥0.0005, confidence≥0.01, cooccurrence≥3, lift≥1, top50 rules per antecedent; `results/step6/experiments/time_window_14d_lift_log/predictions.csv` |
| Content-Based Filtering | `cbf_channel_context_emphasis_airings`, historical-airing-weighted user profile; `results/step9_cbf/predictions/cbf_channel_context_emphasis_airings.csv` |
| V10 configuration — temporal evaluation proxy | Separate evaluation-only LambdaRank with the fixed V10 configuration, fitted strictly before Week21; details below. It is **not** the production model or an identical-weights reproduction. |

### Strict temporal handling of V10

The production V10 model was fitted using historical target outcomes through
Week30. Its production predictions must not be scored against Weeks21–30 as
unseen data. The saved V10 user-cross-fitted predictions exclude each user's
own labels, but their rankers learned from other users' Weeks21–30 targets.
They therefore do not satisfy a strict global pre-Week21 fitting cutoff.

For this comparison, a **separate historical evaluation proxy** was constructed:

1. Training features: Weeks11–15, `[1583704800, 1586725200)`.
2. Supervised training targets: Weeks16–20, `[1586725200, 1589749200)`.
3. Prediction features: Weeks11–20; evaluation targets: Weeks21–30.
4. Original, unexpanded five-generator candidate union; no new retrieval rule.
5. The same **77 feature names/order and fixed V10 parameters**: LambdaRank,
   binary frequency-Top-5 labels, gains`[0,1]`, 78 rounds, 15 leaves,
   learning rate0.05, minimum leaf rows100, L2=5, truncation level8,
   deterministic training, seed2026 and four threads.
6. Fitted **133,605 rows grouped into 1,260 users**, using the verified earlier
   feature cache. Every fitting target precedes the common validation cutoff.

Its model is saved separately at
`results/final_requirement7_evidence/v10_temporal_evaluation_model.txt`.
The earlier five-week training horizon and template-user fitting population
differ from production V10's broader training population. Consequently this
is a **V10-configuration temporal proxy**, not evidence that the unchanged
production V10 would attain these exact historical lists or scores. No
hyperparameter selection or optimization was performed for this task.

### Metadata display policy

Titles and categories below are copied verbatim from the **first in-file
positive-duration airing overlapping Weeks21–30** in
`export_arh_21-30-final.csv`. They are not invented translations or narrative
abbreviations. All60 displayed positions have a nonempty title and category.
One program ID may have many episode/title variants: the display is a
representative schedule entry, not proof that the user watched that exact
episode or subtitle. Source row numbers and all observed title/category
variants are retained in `final_requirement7_evidence/examples.json`.

## 2. Why these exact users were selected

Both IDs remain exact strings throughout loading, scoring, and export.
The existing Step4 selection rule used training activity only: the largest
number of distinct qualifying historical programs for the heavy example;
the smallest positive number among remaining warm users for the light
example. Ties follow original template order. That rule was reproduced
over the frozen1,260-user cohort and selected these same IDs.

| Verified statistic | Heavy: `13571199046623737884` | Light: `8150816946177670084` |
| --- | ---: | ---: |
| Distinct historical programs | 757 | 1 |
| Qualifying historical airings | 2,358 | 1 |
| Historical watched seconds on qualifying airings | 5,118,834 | 2,753 |
| Historical programs eligible in validation catalog | 575 | 1 |
| Active historical calendar weeks | 10 | 1 |
| Channels with qualifying historical airings | 50 | 1 |
| Actual distinct validation programs | 333 | 15 |
| Actual qualifying validation airings | 437 | 17 |

The heavy user's leading historical programs include `12001682` (118
airings), `2400480` (93), and `240081` (88). The light user's sole qualifying
historical program is `90053098`, **Х/ф 'Мужество в бою'.** in the validation
metadata: one airing totaling2,753 seconds. These are aggregate logged
activities, not measures of enjoyment or continuous individual attention.
Neither example is cold-start; the light example is a **warm, sparse-history**
user. No user was reselected based on recommendation accuracy.

## 3. Heavy viewer: `13571199046623737884`

“Seen in training” means a qualifying viewing in Weeks11–20. “Popularity
fallback” identifies an actual procedural backfill position, not simply an
item that happens to be popular. Popularity itself is the global baseline.

| Method | Rank | tv_show_id | Title from validation schedule | Category | Actual validation match | Seen in training | Popularity fallback |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Popularity | 1 | `2400480` | ТСН. | Инфо | Yes | Yes | No |
| Popularity | 2 | `240081` | Факты. Утро. | Инфо | Yes | Yes | No |
| Popularity | 3 | `2400508` | Сегодня. | Инфо | Yes | Yes | No |
| Popularity | 4 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | Yes | Yes | No |
| Popularity | 5 | `2400467` | 'Викна-Новости'. | Инфо | Yes | Yes | No |
| Personal frequency | 1 | `12001682` | М/с 'Губка Боб Квадратные Штаны'. | Детям | Yes | Yes | No |
| Personal frequency | 2 | `2400480` | ТСН. | Инфо | Yes | Yes | No |
| Personal frequency | 3 | `240081` | Факты. Утро. | Инфо | Yes | Yes | No |
| Personal frequency | 4 | `2400508` | Сегодня. | Инфо | Yes | Yes | No |
| Personal frequency | 5 | `20088` | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | No | Yes | No |
| Personalized PageRank | 1 | `2400480` | ТСН. | Инфо | Yes | Yes | No |
| Personalized PageRank | 2 | `240081` | Факты. Утро. | Инфо | Yes | Yes | No |
| Personalized PageRank | 3 | `12001682` | М/с 'Губка Боб Квадратные Штаны'. | Детям | Yes | Yes | No |
| Personalized PageRank | 4 | `2400508` | Сегодня. | Инфо | Yes | Yes | No |
| Personalized PageRank | 5 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | Yes | Yes | No |
| Association Rules | 1 | `240081` | Факты. Утро. | Инфо | Yes | Yes | No |
| Association Rules | 2 | `2400480` | ТСН. | Инфо | Yes | Yes | No |
| Association Rules | 3 | `2400508` | Сегодня. | Инфо | Yes | Yes | No |
| Association Rules | 4 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | Yes | Yes | No |
| Association Rules | 5 | `2400467` | 'Викна-Новости'. | Инфо | Yes | Yes | No |
| Content-Based Filtering | 1 | `10003148` | Т/с 'Стоматолог', 1 с. | Сериалы | No | No | No |
| Content-Based Filtering | 2 | `10003102` | Т/с 'Опер по вызову', 2 сезон, 9 с. | Сериалы | No | Yes | No |
| Content-Based Filtering | 3 | `400471` | Заробітчани, 4 сезон, 1 эп. Турция. | Развлечения | No | No | No |
| Content-Based Filtering | 4 | `10003080` | Т/с 'Школа', 1 с. | Сериалы | No | Yes | No |
| Content-Based Filtering | 5 | `1000608` | Т/с 'Приключения Геркулеса', 5 сезон, 1 и 2 с. | Сериалы | No | No | No |
| V10 configuration — temporal evaluation proxy | 1 | `2400480` | ТСН. | Инфо | Yes | Yes | No |
| V10 configuration — temporal evaluation proxy | 2 | `240081` | Факты. Утро. | Инфо | Yes | Yes | No |
| V10 configuration — temporal evaluation proxy | 3 | `12001682` | М/с 'Губка Боб Квадратные Штаны'. | Детям | Yes | Yes | No |
| V10 configuration — temporal evaluation proxy | 4 | `2400508` | Сегодня. | Инфо | Yes | Yes | No |
| V10 configuration — temporal evaluation proxy | 5 | `20088` | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | No | Yes | No |

### Heavy-viewer interpretation

- Global popularity matches all five items, and all five were already in
  this user's history. This is a useful reference for a broad, active viewer;
  it does not demonstrate personalization, because every user gets that list.
- Personal frequency preserves the historical leader`12001682`, but also
  repeats`20088`, which has no qualifying validation match. High past counts
  therefore do not guarantee continued viewing in the hold-out.
- PPR retains familiar programs while moving news ahead of the cartoon and
  replacing personal frequency's`20088` with`6500479`. The result matches
  five actual validation IDs. Its co-view diffusion uses other users' **past**
  behavior, rather than only this user's raw counts.
- Association Rules largely reproduces the popular news/cognitive set in a
  different order. Its match count is5/5, but this example does not show new
  discovery: all recommendations were seen historically. For instance,
  training rule`20088 → 2400480` has support0.363621, confidence0.961761,
  lift1.270394; many historical antecedents contribute to the news scores.
- CBF recommends four series and one entertainment item; three IDs are unseen
  in this history. None matches qualifying validation viewing. Metadata
  similarity, even with a rich history, is not enough to establish a useful
  recommendation here. No claim is made that unseen recommendations are
  inherently better or that these programs were disliked.
- The strict temporal V10 proxy reorders familiar programs and matches4/5,
  retaining the unsuccessful`20088` repeat. It does not beat popularity on
  this example. These two users are illustrations, not a basis for replacing
  the designated production model.

## 4. Light viewer: `8150816946177670084`

| Method | Rank | tv_show_id | Title from validation schedule | Category | Actual validation match | Seen in training | Popularity fallback |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Popularity | 1 | `2400480` | ТСН. | Инфо | No | No | No |
| Popularity | 2 | `240081` | Факты. Утро. | Инфо | Yes | No | No |
| Popularity | 3 | `2400508` | Сегодня. | Инфо | No | No | No |
| Popularity | 4 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | No | No | No |
| Popularity | 5 | `2400467` | 'Викна-Новости'. | Инфо | No | No | No |
| Personal frequency | 1 | `90053098` | Х/ф 'Мужество в бою'. | Фильмы | No | Yes | No |
| Personal frequency | 2 | `2400480` | ТСН. | Инфо | No | No | Yes |
| Personal frequency | 3 | `240081` | Факты. Утро. | Инфо | Yes | No | Yes |
| Personal frequency | 4 | `2400508` | Сегодня. | Инфо | No | No | Yes |
| Personal frequency | 5 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | No | No | Yes |
| Personalized PageRank | 1 | `90053098` | Х/ф 'Мужество в бою'. | Фильмы | No | Yes | No |
| Personalized PageRank | 2 | `240081` | Факты. Утро. | Инфо | Yes | No | No |
| Personalized PageRank | 3 | `2400480` | ТСН. | Инфо | No | No | No |
| Personalized PageRank | 4 | `6500479` | 'Мир наизнанку. Китай', 16 эп. | Познавательное | No | No | No |
| Personalized PageRank | 5 | `2400467` | 'Викна-Новости'. | Инфо | No | No | No |
| Association Rules | 1 | `90075859` | Х/ф 'Экипаж'. | Фильмы | No | No | No |
| Association Rules | 2 | `90081867` | Х/ф 'Миссия невыполнима: Последствия'. | Фильмы | No | No | No |
| Association Rules | 3 | `90077612` | Х/ф 'Джек Райан: Теория хаоса'. | Фильмы | No | No | No |
| Association Rules | 4 | `90057593` | Х/ф 'Сокровище Амазонки'. | Фильмы | No | No | No |
| Association Rules | 5 | `90076096` | Х/ф 'После нашей эры'. | Фильмы | No | No | No |
| Content-Based Filtering | 1 | `90053098` | Х/ф 'Мужество в бою'. | Фильмы | No | Yes | No |
| Content-Based Filtering | 2 | `90082673` | Премьера. Х/ф 'Кролик Джоджо'. | Фильмы | No | No | No |
| Content-Based Filtering | 3 | `90064040` | Х/ф 'Всемирный торговый центр', 1 ч. | Фильмы | No | No | No |
| Content-Based Filtering | 4 | `90045696` | Х/ф 'Пуленепробиваемый'. | Фильмы | No | No | No |
| Content-Based Filtering | 5 | `90063567` | Х/ф 'Второй в команде'. | Фильмы | No | No | No |
| V10 configuration — temporal evaluation proxy | 1 | `240081` | Факты. Утро. | Инфо | Yes | No | No |
| V10 configuration — temporal evaluation proxy | 2 | `2400480` | ТСН. | Инфо | No | No | No |
| V10 configuration — temporal evaluation proxy | 3 | `2500430` | 'Затерянный мир'. | Познавательное | No | No | No |
| V10 configuration — temporal evaluation proxy | 4 | `2400508` | Сегодня. | Инфо | No | No | No |
| V10 configuration — temporal evaluation proxy | 5 | `20088` | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | No | No | No |

### Light-viewer interpretation

- Only one historical film is available as a personal-frequency signal.
  It is eligible, but it was not watched in qualifying validation viewing.
  Ranks2–5 are the actual popularity completion, not fabricated personal
  recommendations. The sole hit,`240081`, arrives at rank3.
- Popularity matches`240081` at rank2 and misses four items. It gives usable
  coverage with little personal information, but does not recover the
  user's broader validation activity of15 distinct IDs.
- PPR retains the old film first, then puts`240081` second. The news items
  have positive graph scores and are **not fallback fills**, despite their
  similarity to the global popularity list. A personalized computation can
  still produce popular recommendations.
- Association Rules supplies five previously unseen films, all with real
  training-derived rule scores, but no validation matches. The only
  historical antecedent`90053098` supports`90075859` with confidence0.485816
  and lift23.158204; the confidence×log-lift score can favor that film over
  broadly popular news. A strong association is not proof of this user's
  future choice.
- CBF repeats the historical film and retrieves four other films using the
  historical metadata profile. None matches the actual validation set.
  A one-program profile is narrow; plausible category/content matches do
  not demonstrate accurate future prediction.
- The strict temporal V10 proxy puts the sole matching ID,`240081`, first.
  All five recommendations are unseen in this user's one-item history, and
  none is a procedural cold/popularity fallback. It improves rank placement
  for this single hit but still matches only1/5. Its parameters were fixed
  before inspecting these example outcomes.

## 5. Comparable per-user results and target-definition sensitivity

These are **per-user AP@5 values**, not aggregate model MAP or Kaggle scores.
The main “hits” column counts matches against all333/15 actual validation IDs.
The frequency-Top-5 columns are a separate diagnostic using the same lists.

| User | Method | Actual-viewing hits/5 | AP@5: all distinct truth | Frequency-Top-5 hits/5 | AP@5: frequency-Top-5 truth |
| --- | --- | ---: | ---: | ---: | ---: |
| Heavy | Popularity | 5 | 1.000000 | 4 | 0.710000 |
| Heavy | Personal frequency | 4 | 0.800000 | 2 | 0.233333 |
| Heavy | Personalized PageRank | 5 | 1.000000 | 3 | 0.520000 |
| Heavy | Association Rules | 5 | 1.000000 | 4 | 0.710000 |
| Heavy | Content-Based Filtering | 0 | 0.000000 | 0 | 0.000000 |
| Heavy | V10 configuration — temporal evaluation proxy | 4 | 0.800000 | 2 | 0.400000 |
| Light | Popularity | 1 | 0.100000 | 1 | 0.100000 |
| Light | Personal frequency | 1 | 0.066667 | 1 | 0.066667 |
| Light | Personalized PageRank | 1 | 0.100000 | 1 | 0.100000 |
| Light | Association Rules | 0 | 0.000000 | 0 | 0.000000 |
| Light | Content-Based Filtering | 0 | 0.000000 | 0 | 0.000000 |
| Light | V10 configuration — temporal evaluation proxy | 1 | 0.200000 | 1 | 0.200000 |

The heavy user's frequency-Top-5 truth is
`240081 2400480 6500479 2400467 2500413`; the light user's is
`240081 200432 500281 10002514 90034524`. This explains why an item can be
an actual viewing match without being a Top-5-target hit. Neither local
target construction is asserted to reproduce the hidden Kaggle labels.

## 6. Actual fallback and cold-start verification

| Method | Heavy scored/eligible personal candidates | Light scored/eligible personal candidates | Actual fallback ranks: heavy / light |
| --- | ---: | ---: | --- |
| Popularity | Not personalized | Not personalized | None / none: it is the global baseline itself |
| Personal frequency | 575 | 1 | None / **2,3,4,5** |
| PPR | 1,700 positive-score eligible candidates | 1,675 | None / none |
| Association Rules | 529 eligible rule-scored candidates | 46 | None / none |
| CBF | 6,384 positive-similarity candidates | 6,191 | None / none |
| V10 configuration proxy | 108 multigenerator candidates | 114 | None / none: both users have history |

The shared fallback sorts eligible programs by distinct **training** viewers
descending, qualifying historical airings descending, then numeric ID
ascending, skipping already selected IDs. Its first five are
`2400480 240081 2400508 6500479 2400467`.

Personal frequency uses backfill when fewer than five eligible historical
programs exist. PPR uses it when fewer than five eligible programs have
positive graph scores or when no profile exists. Association Rules fills
after its rule-scored candidates. CBF fills after positive similarities or
when the user vector is empty. The LambdaRank policy ignores learned scores
for a user with no qualifying full historical profile and uses the common
popularity order. **Neither fixed user is cold**, so no cold-start event is
claimed from these two examples. No missing recommendation was invented.

CBF uses training-only vocabulary/IDF fitted on5,842 historical item documents
and24,309 terms. Its fixed field weights are category1.5, genre1.25, title0.8,
director0.3, actors0.2, year0.1 and channel1.0; user weights are historical
qualifying airings. Its many positive similarities are real scores, not
evidence that all those programs are strong personal recommendations.

## 7. Business interpretation and limitations

V10 remains the designated final strategy, with the supplied genuine private
MAP@5 of0.40637. These two examples do not establish its overall superiority;
in particular the temporal proxy loses to popularity for the heavy user.
The production ranker combines repeat-viewing, graph, rule, content and
availability signals, while a common popularity fallback ensures coverage.

The practical distinction visible here is between **coverage**, **familiarity**
and **correct ordering**. Popularity provides inexpensive coverage and can
match a broad heavy viewer well. Personal frequency strongly repeats the
past and needs four fills for the sparse user. PPR and rules can extend
one-item histories into many scored candidates, but their resulting lists
may still be popular or inaccurate. CBF's metadata matching increases
historical novelty in these examples without producing a validation hit.
The proxy illustrates that moving an existing hit upward changes AP even
when the number of hits is unchanged.

Limitations supported by this evidence:

- Two deliberately contrasting users are illustrative, not a representative
  experiment or an A/B test. The333-label heavy user has many more possible
  matches than the15-label light user.
- “Unseen” means absent from this platform's qualifying historical log,
  not never watched elsewhere. No match does not prove dislike; a match
  does not prove enjoyment.
- Only labeled airtime can become an item; the exam notes that only10.8%
  of schedule rows carry nonzero program IDs. The120-second rule misses
  shorter viewing and is a chosen local definition.
- Episode/title variants share IDs. Display metadata is an observed schedule
  representative, not a claim about the exact watched episode. No displayed
  metadata field is missing; the general variant ambiguity remains.
- The V10 temporal proxy uses the same configuration/schema but different,
  earlier fitting data and learned weights. It is not the final production
  model, whose use on its own historical target period would be invalid.
- Original configurations were selected on previously examined validation
  data. This report performs no retuning and claims no untouched holdout.

## 8. Verification and source evidence

The executed extraction/replay passed **472 checks, 0 failed**. It verified:

- The exact two user strings and the original history-only selection rule.
- Historical activity independently from matched training-airing rows.
- Saved prediction and configuration hashes, the PPR graph and AR rules.
- Exact replay of the two users' popularity, frequency, PPR, AR and CBF lists.
- The strict pre-validation fitting boundary, labels, groups and schema for
  the separate V10-configuration evaluation proxy.
- All60 ranked positions: exactly five distinct positive eligible IDs per
  method/user, metadata provenance, actual-viewing flags, AP, and true fallback.
- **966 previously existing files unchanged**, including every earlier
  experiment/report/submission and the original five session JSON exports.

The final report/table consistency and independent overlap-label checks are
recorded separately in
`results/final_requirement7_evidence/report_verification.json`.

Production V10 CSV SHA-256 remains:

```text
39ebef90f28783caa7960bdda8c28347116d7837b3af4fd73c6a9891ea49fd7e
```

Production V10 model SHA-256 remains:

```text
5bd830d88cb808a2617a982bba93b304e600738071b2ce367c7911ff79d92a65
```

Primary source references, relative to the project root:

1. `exam_quiz_FKN.md`, §7 and §8.
2. `data/step3/evaluation_config.json`, `validation_truth.csv`, and
   `data/step2/validation_candidates.csv`.
3. `data/step2/threshold_120s/train_interactions.csv.gz`,
   `validation_interactions.csv.gz`, `train_airing_overlaps.csv.gz`, and
   `validation_airing_overlaps.csv.gz` for source counts and matching.
4. `results/step4_baseline_report.md`, `results/step4/step4_run.json`,
   `step4_outputs.json`, `provisional_two_users.json`, and the saved baseline
   prediction files listed in §1.
5. `results/step5/ppr_model_config.json`, `graph_cosine_top30.npz`,
   `ppr_user_examples.md`, and selected PPR predictions.
6. `results/step6/association_best_config.json`,
   `experiments/time_window_14d_lift_log/rules.csv`, selected predictions,
   and `association_user_examples.md`.
7. `results/cbf_step9_report.md`,
   `results/step9_cbf/step9_validation_results.json`, and selected CBF predictions.
8. `results/v10_experiment/README.md`, `experiment.json`,
   `submission_manifest.json`, and `selected_all_crossfit.csv`. The old V10
   README predates the user's genuine0.40637 private score; it is preserved
   as a historical record, not treated as the current model designation.
9. `results/v11_experiment/early.npz` and `early_build.json` for the verified
   pre-validation evaluation-proxy feature source; original V8 validation
   features for Weeks11–20 prediction inputs.
10. `../sweettv-tv-program-recommender/export_arh_21-30-final.csv` for every
    displayed title/category. CBF uses schedule-only metadata transformed
    with historical vocabulary; no target viewing is a feature.

Machine-readable evidence: `results/final_requirement7_evidence/examples.json`,
`recommendations.csv`, `v10_temporal_evaluation_config.json`, `verification.json`,
`report_verification.json`, and `preservation_before.json`. These retain the
complete validation counts, per-rank scores/familiarity/fallback, metadata
source rows/variants, and source hashes.

Reproduction uses `code/final_requirement7.py` followed by
`code/verify_final_requirement7.py`. The evidence builder refuses to overwrite
completed outputs. No Kaggle submission or production-model modification
was performed.
