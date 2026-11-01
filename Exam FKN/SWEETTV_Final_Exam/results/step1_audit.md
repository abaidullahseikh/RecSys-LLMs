# SWEET.TV — Step 1 inspection audit

**Student:** Abaidullah Seikh  
**Instructor:** Seungmin Jin  
**Deadline:** 1 November 2026, 19:00 MSK  
**Inspection date:** 30 October 2026  
**Stage:** inspection and design only; no recommendation algorithms implemented, no models fitted, no MAP experiment run, and no prediction/submission CSV generated.

**Finding:** all five required CSVs are present, parse successfully, and are processable with streaming/disk-backed methods on this machine. Important corrections to a naive interpretation are daylight-saving time, schedule spillovers, unlabelled airtime, repeated viewing intervals, and large user IDs. The data supports the proposed chronological split. Some competition evaluator details and report-format instructions remain unresolved.

Evidence labels used below:
- **Requirement:** stated in the primary exam specification or the identified course materials.
- **Observed:** measured from the supplied files or the current environment.
- **Proposal/inference:** a documented implementation decision, not an asserted competition rule or an experimental performance result.

## 1. Official requirements

### Sources actually read

1. **Primary:** `/home/dastan/Recommender Systems/FKN/exam_quiz_FKN.md`, all 125 lines, read before any other action. Section/line references below refer to this original.
2. **Presentation:** `/home/dastan/Recommender Systems/FKN/LLM4Rec. Midterm Quiz.pdf`, all 10 pages, including image-only metric and Kaggle screenshots.
3. **Homework rules located during inspection:** `/home/dastan/Recommender Systems/Week 2/homework_guidelines (2).pdf`, all 12 pages. Its applicability follows from the exam's reference to existing homework guidelines; whether a newer revision exists is unknown.
4. Live Kaggle evaluation, rules, and data pages were fetched on the inspection date. Each returned only the page title through the available web-fetch tool. An additional unauthenticated GET to the competition service endpoint returned HTTP 400. Neither result verifies the live evaluator details, account acceptance, or current submission quota.

### Mandatory exam checklist

| Requirement | Source | Implication |
|---|---|---|
| Individual work; deadline above | §1, header | Attribute assistance and adapted sources; do not use another participant's predictions. |
| Accept competition rules with the student's own account | §2, lines 11–16 | Instructor acceptance does not establish student access. Confirm before a later submission. |
| Exactly five ranked program IDs per test user; use K=5 everywhere | §§1–3, lines 7, 13–14, 40 | All local methods and final output must share this list length. Actual template has 1,260 users. |
| At least two approved methods | §4, lines 46–51 | Approved: CBF, Association Rules, implicit MF, program co-view Personalized PageRank. |
| Popularity baseline, measured alongside each method | §4, line 53 | It does not count toward the two approved methods. Personal viewing frequency is also an extra reference, not an approved-method substitute. |
| Channel/time-interval join and justified watched threshold | §5, line 57 | Distinguish session duration from broadcast duration; explicitly construct any watch fraction. |
| Explicit treatment of program ID 0 | §3 and §5, lines 36, 58 | Unlabelled airtime is not a recommendable item. |
| One common temporal hold-out, no later than Weeks 21–30 | §5, lines 59–60; slides p.2 | Same boundaries, users, candidate rules, relevance labels, and metric for all methods. Fit on training only. |
| Explicit candidate pool and already-watched policy | §5, line 61 | Pool bounded by labelled schedule programs. Recommending previously watched programs is not forbidden. |
| Same cold-start fallback and exact-five completion policy | §5, lines 62–63 | Explain ranking, ties, deduplication, and fallback per method. Scores are not interchangeable probabilities. |
| Hyperparameters, random seeds, runtime, environment | §5, line 64 | Record actual execution settings and measurements, not estimated performance as measured results. |
| No future viewing behavior | §5, line 65 | Test schedule may establish program/channel/time availability, never user consumption. Use conservative availability-only use of future schedule. |
| Final strategy justified; hybrids optional | §6 | If combining methods, implement and evaluate each component individually; select combination weights on the hold-out. |
| Same two justified users across all methods | §7 | Show ordered IDs, titles, categories for each method; reproduce the IDs in the PDF; explain fallback. |
| Business and algorithmic analysis | §8 | Compare recommendations with the same users' observed viewing; discuss familiar/new items, method mechanisms, validation, cost, user experience, and evidenced failure cases. |
| Qualify discovery/enjoyment claims | §8 | Absence from these logs does not mean never watched; unfamiliarity does not mean usefulness. No A/B-test, online-experiment, or revenue claim. |
| Exactly two course-channel files | §9, lines 92–96 | PDF report and original unmodified AI-session export. The Kaggle CSV belongs in GitHub, not as a third course attachment. |
| PDF contents | §9, lines 98–101 | GitHub link; screenshot of returned private score for designated final submission; method comparison, validation, business and algorithmic analysis. |
| GitHub contents | §9, line 103 | Implementation code and exact final scored submission file. Reproduction instructions are explicitly optional in the exam, but recommended and planned. |
| Homework format, records, disclosure | §9, lines 105–107 | External code/models/data/AI assistance must be disclosed; do not fabricate sources, experiments, or verification. |
| Original, corresponding session export | §9 and homework §§4,9 | Native unmodified OpenCode JSON containing actual work. A manually reconstructed transcript is not acceptable. |

**Grading:** 50 points from the returned **private score**, ranked within the exam cohort: successive 20% bands receive 50/40/30/20/10. The frozen public leaderboard position is irrelevant. The other 50 points use the existing homework criteria, with no invented replacement breakdown. Gate 0 automatically fails the exam. The template's 0.00000 example and late-submission acceptance are reported by the instructor, not verified by a submission from this project.

### Homework requirements and discrepancies that must not be silently discarded

The located binding guidelines require a title block, 50–150-word four-sentence abstract, Introduction, Related Work (3–7 consulted entries, IEEE numbering, alternatives), Method, Experiments with verification, substantive Discussion (failure/root cause/fix/verification/next improvement), AI Usage Disclosure, and References. Index terms are optional. Record tools/versions and what was personally verified versus trusted. Follow BUILD → ASK WHY → TEST → EXPLAIN → IMPROVE; personally verify at least a number, derivation, and assumption as described in §7. This AI audit does not satisfy personal engagement on the student's behalf. Course lateness is not excused by Kaggle allowing late competition submissions.

**Conflict:** homework p.3 says a five-page hard cap; pp.4–5 and p.12 say six pages excluding references. The exam introduces no replacement cap or language requirement. Resolve this with the instructor; a ≤5-page body is a conservative planning target, not a claimed resolution. The presentation p.7 asks for the student's full name in Russian plus Kaggle name in the submission description; obtain the student's preferred Russian rendering rather than inventing it. Slide p.8 says submissions are “usually” limited to five, while screenshots show other values; verify the actual account quota later.

## 2. Available project files

### Original inputs found

All five Kaggle files are at the exact paths below; sizes are bytes, not rounded MB:

| File | Actual absolute path | Bytes |
|---|---|---:|
| View log | `/home/dastan/Recommender Systems/FKN/sweettv-tv-program-recommender/dataset11-30.csv` | 396,216,555 |
| Early schedule | `/home/dastan/Recommender Systems/FKN/sweettv-tv-program-recommender/export_arh_11-20-final.csv` | 153,322,643 |
| Later historical schedule | `/home/dastan/Recommender Systems/FKN/sweettv-tv-program-recommender/export_arh_21-30-final.csv` | 154,366,907 |
| Future schedule | `/home/dastan/Recommender Systems/FKN/sweettv-tv-program-recommender/export_arh_31-42-final.csv` | 185,952,991 |
| Submission template | `/home/dastan/Recommender Systems/FKN/sweettv-tv-program-recommender/submission.csv` | 38,348 |

Total CSV size: **889,897,444 bytes**. No essential dataset is missing. The original competition archive was not found in FKN and is unnecessary because all five extracted files are present. No test-period user-view log is supplied or needed.

Also found: original exam (11,694 bytes), presentation (816,689 bytes), and homework guidelines in the sibling Week 2 folder (691,834 bytes). No applicable `AGENTS.md` was found in the searched course tree or inspected parent locations. The initial FKN folder contained no implementation, experiment artifacts, repository, or session export.

### Original SHA-256 fingerprints

| Original | SHA-256 |
|---|---|
| `dataset11-30.csv` | `8c9e5f058157041e5c33f13e64dc1625a5177cc6d5e434ba18cd8a69afafb53b` |
| `export_arh_11-20-final.csv` | `7888aa00c20db3db23f95000c721eccf2878e5f58bf5383bb25b99f078b1033c` |
| `export_arh_21-30-final.csv` | `bdc9fa9850239aaed608bece9cdb200011e690300470fe85b980457d8c0cb2d4` |
| `export_arh_31-42-final.csv` | `5cc01f4619bd94886d83744f00bbc15cc5fa23384d297244a7dc2207450c4088` |
| `submission.csv` | `d8c209dce3c7c9d815d8e2a6226e1be63052ee7afa7e8ba8b75dc2f703967a02` |
| `exam_quiz_FKN.md` | `49dd3b0db2a0b056448b394b4e7e0454e793e884ff31d040f576ed829cbda161` |
| `LLM4Rec. Midterm Quiz.pdf` | `e9f6589ab2ac477585320ab2f9a61ada4670d9510859f0acf26b158ff1a428d8` |
| `homework_guidelines (2).pdf` | `307495a7f0a88d2f4d05817898199aa2f3c4ec6025f3c184e59f8f7a40880736` |

The first seven originals were hashed before full profiling. The additional guidelines were hashed after discovery. `results/original_sha256_after.json` records the closing comparison, including the guidelines check; `results/step1_verification.json` records the local exam-copy comparison. Original inputs were opened read-only, never rewritten. Fingerprints establish local preservation, not a claim of comparison with a publisher-supplied checksum.

### Generated inspection evidence

- `results/step1_profile.json`: complete CSV profiles, column-level missingness, channel inventories, ISO-week counts, exact duplicate statistics, environment and timings.
- `results/step1_intervals.json`: initial naive-wall-clock overlap diagnostics, retained as the initial interpretation.
- `results/step1_clock.json`: DST hypothesis checks and absent-channel IDs.
- `results/step1_intervals_kyiv.json`: corrected-clock descriptive overlap diagnostics used for the main recommendations below.
- Hash manifests, `results/homework_guidelines_sha256.json`, and `results/step1_verification.json`.
- `results/step1_inspection.sqlite`: disposable disk-backed viewing cache, ignored by Git. It is not a model, prediction artifact, or edited original.
- Three executed inspection utilities under `code/`, project scaffolding, this audit, and the native checkpoint described in §13.

## 3. Dataset statistics

All counts are **CSV records excluding headers**, obtained with UTF-8 `csv.reader`, not newline counting. No malformed field counts or numeric/timestamp parse failures were observed. Exact duplicate checks compare every parsed field (including text and whitespace) using canonical row serialization and SQLite grouping; duplicate “excess” means occurrences beyond the first.

### Counts and identities

| File | Records | Unique users | Channels | Distinct IDs incl. 0 | Positive IDs | Duplicate excess |
|---|---:|---:|---:|---:|---:|---:|
| `dataset11-30.csv` | 5,821,987 | 4,914 | 202 | N/A: no program column | N/A | 17 (17 groups) |
| `export_arh_11-20-final.csv` | 1,074,235 | N/A: no user column | 400 | 5,848 | 5,847 | 0 |
| `export_arh_21-30-final.csv` | 1,074,394 | N/A | 403 | 6,364 | 6,363 | 0 |
| `export_arh_31-42-final.csv` | 1,295,378 | N/A | 404 | 6,637 | 6,636 | 0 |
| `submission.csv` | 1,260 | 1,260 | N/A | 1, the placeholder 0 | 0 | 0 |

The four main files total **9,265,994 records**, matching the exam. Union of positive IDs across the historical schedule files: **8,234**; across all three schedule files: **9,671**. File-level overlap: 3,976 later-historical IDs also appear in the early file, and 2,387 do not; 1,437 future-file IDs are absent from both historical files. These file-level counts differ from timestamp-clipped candidate counts in §9.

### Columns and storage types

CSV has no embedded type schema. The following are inspected semantic types and proposed safe storage, not a claim that pandas inferred them:

**View log**, in original order:

| Column | Observed representation | Safe interpretation |
|---|---|---|
| `user_id` | Unsigned decimal integer text | Exact string key; never float or signed int64 |
| `vsetv_id` | Integer text | Channel integer, joins `channel_id` |
| `start_time` | `YYYY-MM-DD HH:MM:SS` | Explicitly parsed local timestamp; timezone inference discussed below |
| `stop_time` | Same format | Viewing stop timestamp |
| `duraton` | Integer text | Elapsed viewing seconds; preserve the original misspelling |

**All schedules**, identical order/schema:

`channel_id, channel_title, start_time, duration, tv_show_title, tv_show_id, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3, year_of_production, director, actors`.

- `channel_id`, `duration`, `tv_show_id`: integer-form values; safe integer conversion after validation. `duration` is broadcast elapsed seconds.
- `start_time`: `DD.MM.YYYY HH:MM:SS`; explicit day-first parse, not automatic month/day inference.
- All title/category/genre/director/actor fields: UTF-8 strings, nullable where indicated below. Quoted commas in titles/cast require a real CSV parser.
- `year_of_production`: nullable **text**, not an integer column. Nonnumeric nonempty rows: 53,616 / 51,935 / 64,162. Examples include `1996-`, `2012-2012`, `2014-`. Any later feature extraction must preserve raw text and document start/end-year parsing.

**Template:** `user_id` is exact decimal-string identity; `tv_show_id` is a **single string containing five space-separated integer tokens**, not five CSV columns. All 1,260 rows contain `0 0 0 0 0`. Header is exactly `user_id,tv_show_id`. No duplicate users, empty cells, or malformed token counts were found. Preserve original row order as well as original IDs in a later output; write no DataFrame index.

**ID precision:** view-user range is `2244466330591177`–`18443832178892490378`; 4,912 unique IDs exceed 2^53 and 2,483 exceed signed int64. Template range is `36938861791891792`–`18440717181607000261`; **all 1,260 exceed 2^53**, and **655 exceed signed int64**. String keys avoid rounding, overflow, and spreadsheet scientific-notation conversions. All template users occur somewhere in the view log. No leading-zero user IDs were observed, but strings also preserve them if future files contain any.

### Missing values

Empty/whitespace-only values were counted in every field. No whitespace-only nonempty cells or literal `null`, `none`, `nan`, `na`, `n/a` markers were found. Zero is **not** treated as an empty CSV cell.

- View log: **0 missing** in all five columns.
- Template: **0 missing** in both columns.
- Schedule columns `channel_id`, `channel_title`, `start_time`, `duration`, `tv_show_title`, `tv_show_id`, `tv_show_category`: **0 missing in each file**.

| Nullable schedule column | Weeks 11–20 file | Weeks 21–30 file | Weeks 31–42 file |
|---|---:|---:|---:|
| `tv_show_genre_1` | 620,089 | 616,370 | 737,871 |
| `tv_show_genre_2` | 928,009 | 924,020 | 1,113,598 |
| `tv_show_genre_3` | 1,054,059 | 1,050,668 | 1,266,621 |
| `year_of_production` | 959,500 | 954,382 | 1,151,706 |
| `director` | 974,050 | 969,487 | 1,170,189 |
| `actors` | 973,861 | 968,834 | 1,169,519 |

### Actual timestamp coverage (as written, local wall clock)

| File | Earliest start | Latest start | Latest end |
|---|---|---|---|
| Views | 2020-03-09 00:00:02 | 2020-07-26 23:58:42 | 2020-07-26 23:59:58 |
| Schedule 11–20 | 2020-03-08 23:00:00 | 2020-05-18 11:45:00 | 2020-05-18 12:00:00 |
| Schedule 21–30 | 2020-05-18 00:00:00 | 2020-07-27 11:55:00 | 2020-07-27 12:00:00 |
| Schedule 31–42 | 2020-07-27 00:00:00 | 2020-10-19 11:55:00 | 2020-10-19 12:00:00 |
| Template | N/A | N/A | N/A |

Schedule ends are inferred from start plus broadcast duration; extrema above are unchanged by the DST correction. The first profile's naive end calculations around March 29 are superseded by the timezone-aware diagnostic, not by modifying the raw CSV.

Viewing starts occupy ISO weeks 11–30 of **2020**, not 2026. The early schedule has 86 starts in Week 10 and 2,431 in Week 21; the next has 2,429 in Week 31; the future file has 2,399 in Week 43. Filenames are not sufficient filters.

### Duration distributions (seconds; exact nearest-rank quantiles)

| Input duration | Minimum | P25 | Median | P75 | P90 | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Viewing `duraton` | 61 | 133 | 396 | 1,452 | 3,911 | 6,344 | 14,080 | 342,145 |
| Broadcast 11–20 | 0 | 600 | 1,500 | 3,000 | 5,400 | 6,900 | 12,600 | 86,400 |
| Broadcast 21–30 | 60 | 600 | 1,500 | 3,000 | 5,400 | 7,200 | 13,200 | 86,400 |
| Broadcast 31–42 | 60 | 600 | 1,500 | 3,000 | 5,400 | 7,200 | 13,500 | 43,200 |

Viewing sessions can span several broadcasts, and the longest lasts nearly four days. Their duration is not a watch percentage or a program length.

## 4. Data-quality issues

1. **DST rather than an unexplained duration error.** Naive subtraction disagrees with `duraton` in 117 rows, always by exactly +3,600 seconds. All 117 cross `2020-03-29 04:00:00`; no viewing start or stop occurs in local hour 03 that day. `Europe/Kyiv` localization followed by UTC subtraction matches every checked row near the transition (135,487 rows), and the full corrected diagnostic has zero duration mismatches. Across 394 schedule channel transitions, next-start minus current-start minus duration is also exactly 3,600 naive seconds. Example: channel 3, March 29 01:25, duration 6,000, next start 04:05. This strongly supports a skipped local hour, not arbitrary corrupted durations. The precise timezone name remains an inference, since several zones share this transition.
2. **Do not add elapsed seconds on a naive/local wall clock across DST.** Localize schedule start, convert to UTC, then add duration on that UTC timeline; localize both viewing endpoints separately. Keep local calendar boundaries for the week split. Audit both parsed duration consistency and nonexistent/ambiguous times in Step 2.
3. **17 exact duplicate viewing records.** Deduplicate derived data or union intervals; leave originals intact. The 17 rows do not explain all overlap inflation.
4. **Substantial same-user/channel overlap and adjacency.** Unioning 5,821,987 raw intervals yields 5,441,773 segments (this reduction includes touching fragments). Same-channel duplicate/overlap time inflates naive summed elapsed viewing by **116,225,303 seconds**. Fragment counts must not be used as watch-frequency counts.
5. **Concurrent different-channel viewing.** Summed per-user/channel union time is 8,502,989,921 seconds; union across all channels per user is 7,872,180,150 seconds. The 630,809,771-second excess demonstrates concurrency. The files do not identify devices or prove fraud. Keep channel-specific evidence for matching; do not invent a single-device sequence or sum this as unique human attention. Binary user support and qualifying-airing counts reduce duration inflation sensitivity.
6. **Schedule gaps and missing channels.** Five viewed channels (`149, 1174, 1508, 1509, 1510`) are absent from both historical schedules, accounting for 113,579 view rows. Overall, 413,297 view rows have no schedule overlap under the DST-aware interpretation; some are gaps/coverage limitations on otherwise present channels. Do not label those rows as negative feedback or infer a program from a nearby title.
7. **One zero-duration schedule row.** Early file, channel 644, `29.03.2020 04:00:00`, ID 0. Quarantine/drop from derived interval calculations. No negative IDs or other nonpositive intervals were observed; all view intervals are positive.
8. **Schedule overlaps.** The naive per-file check found 2 / 2 / 8 rows overlapping earlier same-channel coverage, with 0 / 0 / 4 labelled-versus-labelled overlaps. Exact duplicate rows within each schedule are zero. Boundary raw-row comparisons across successive files found zero exact duplicates; on historically viewed channels, combined/deduplicated labelled airings had no overlaps under either tested clock interpretation. Still assert this property in the production join; do not silently double-credit conflicting program assignments if encountered. Future-schedule overlaps cannot supply future user labels.
9. **IDs identify recurring programs, not necessarily a unique episode/title string.** Positive IDs with multiple titles: 1,685 / 2,508 / 1,679 per schedule file. Positive IDs with multiple categories within a file: zero. Use IDs as items; choose a documented representative historical title for display. Do not create new items solely from episode title variations.
10. **Nine categories overall, eight among positive IDs.** All categories include `Музыка`; it does not occur among positive-ID rows in these files. Metadata is sparse and some years are ranges. CBF feasibility does not imply useful metadata for every item.
11. **Selection and drift.** All recorded view durations exceed 60 seconds; this is an observed truncation, not a verified competition watched threshold. Weekly volume changes sharply: 105,223 starts in Week 27 versus 271,172 in Week 28. No causal explanation is established. The validation interval has fewer rows and longer overlap durations than training, so use a chronological evaluation and report limitations.

## 5. Missing program ID analysis

| Schedule file | ID 0 rows | Labelled rows | Labelled row share | Labelled share of summed broadcast seconds |
|---|---:|---:|---:|---:|
| 11–20 | 958,622 | 115,613 | 10.7624% | 15.9663% |
| 21–30 | 952,889 | 121,505 | 11.3092% | 16.9000% |
| 31–42 | 1,150,604 | 144,774 | 11.1762% | 16.6721% |

**Clarification of the exam's shorthand:** “10.8%” is the early-file **row fraction**, not literally the proportion of elapsed airtime. The duration-weighted quantities above are summed row durations, not overlap-corrected channel exposure. The published unique counts include sentinel 0, so recommendable distinct counts are one lower. This explanation refines measured quantities without relaxing the missing-ID requirement.

DST-aware matching of the original view rows gives:

| Match status | Raw view rows |
|---|---:|
| At least one labelled-airing overlap | 2,964,378 |
| Some schedule coverage, but no labelled overlap | 2,444,312 |
| No schedule overlap | 413,297 |
| Total | 5,821,987 |

Thus 50.9169% of view rows touch a labelled airing; 4,497,257,030 of 8,502,989,921 per-user/channel union seconds (52.8903%) intersect labelled airings. User exposure is not uniformly distributed across schedule rows, so 10.8% schedule coverage does not imply only 10.8% of viewing is identifiable.

**Proposed ID policy:** retain ID-0 rows only for coverage/context diagnostics; exclude them from all item catalogs, training edges, support counts, relevance sets, and recommendations. Do not infer a missing ID from title similarity in the mandatory implementation. Missing labels are unknowns, not evidence of dislike. Retain unmatched/no-ID counts so failures can be discussed honestly. Template zeros are placeholders to replace later, not valid output items.

## 6. Viewing-to-program matching design

### Exact rule and double-count prevention

For view interval `V=[v_start,v_stop)` and airing `A=[a_start,a_start+duration)` on the same channel:

```text
vsetv_id == channel_id
overlap_seconds = max(0, min(v_stop, a_end) - max(v_start, a_start))
```

Use half-open intervals: touching endpoints have zero overlap. All interval arithmetic should use UTC elapsed seconds under the documented local-clock interpretation. Merely joining on channel/date or choosing the first broadcast after a view starts is incorrect.

Proposed processing order:

1. Read exact string user IDs; parse both timestamp formats explicitly; check/record the timezone assumption. Reject/quarantine invalid spans without editing originals.
2. Establish training/validation boundaries before any behavior aggregation. Clip viewing time to its period. The 1,007 raw intervals crossing the training cutoff contribute only their pre-cutoff portion to training; remaining time is validation evidence only.
3. Deduplicate schedule airing identity `(channel_id, start_utc, end_utc, tv_show_id)` after comparing metadata; preserve conflict diagnostics. Boundary spillovers from the adjacent historical file can complete availability, but not expand behavior cutoffs.
4. Group view fragments by `(user_id, channel_id)` within the period, union overlapping/touching fragments, then intersect with all eligible labelled airings. This avoids duplicate-time credit while allowing one long session to match multiple programs.
5. Accumulate disjoint overlap pieces for each `(user_id, channel, airing_start, airing_end, tv_show_id, period)` before applying the watched threshold. Two short fragments of the same airing can jointly qualify. Do not add separate repeat broadcasts merely to pass a per-airing threshold.
6. Count qualifying distinct airings for frequency, retain union seconds as an optional weight, and deduplicate `(user_id, program_id)` for binary relevance/support. Check total credited time per user-airing does not exceed its broadcast duration. If overlapping different labelled IDs exist on one channel, quarantine ambiguous time or resolve it explicitly and identically for all methods; historical diagnostics did not encounter this on viewed channels.

Avoid a Cartesian DataFrame join. Channel-sorted interval sweeps or start-time/prefix-max-end indexes plus bounded search are feasible. The inspection used SQLite ordering and per-channel interval indexes, not an exported interaction training set.

### Observed overlap distribution

The following are full-data descriptive join measurements under the supported `Europe/Kyiv` hypothesis, not samples or model results. “User-airing union” sums only disjoint fragments of that specific broadcast within the period. Quantiles are nearest-rank.

| Population | Count | P10 (s) | P25 | Median | P75 | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train raw-view × labelled-airing overlaps | 2,383,336 | 77 | 146 | 465 | 1,322 | 2,782 | 6,577 | 22,800 |
| Train user-airing union overlaps | 1,823,893 | 86 | 197 | 621 | 1,767 | 3,599 | 7,253 | 28,387 |
| Validation raw-view × airing overlaps | 1,465,129 | 83 | 174 | 600 | 1,803 | 4,262 | 8,100 | 30,000 |
| Validation user-airing union overlaps | 1,191,015 | 95 | 245 | 888 | 2,508 | 5,134 | 8,700 | 30,000 |

Constructed user-airing watched fractions, `union_overlap / full_broadcast_duration`: training P25 **4.10%**, median **17.27%**, P75 **57.74%**; validation P25 **4.89%**, median **23.83%**, P75 **79.25%**. Fractions were stored to one basis point for diagnostics; a rounded zero does not mean an actual zero-length intersection. At least the 90th percentile is 100% in each period. There is no pre-existing watch-percentage column.

### Proposed watched threshold: 120 elapsed seconds per user-airing

| Absolute cutoff | Qualifying train user-airings | Train users | Qualifying validation user-airings | Validation users |
|---|---:|---:|---:|---:|
| ≥60 s | 1,773,566 | 4,841 | 1,165,285 | 4,806 |
| **≥120 s** | **1,531,876** | **4,837** | **1,028,093** | **4,804** |
| ≥300 s | 1,224,147 | 4,833 | 855,420 | 4,799 |
| ≥600 s | 958,577 | 4,824 | 707,790 | 4,792 |

**Rationale from training data:** 120 seconds excludes very brief/boundary overlaps while retaining **83.99%** of positive user-airing overlaps. A 300-second rule retains only **67.12%** and excludes all programs shorter than five minutes. The 120-second rule still excludes broadcasts shorter than two minutes; this is an explicit limitation. A high percentage threshold would also penalize long broadcasts even where absolute viewing is substantial. Choose the simple absolute cutoff as the initial frozen protocol; do not claim it reproduces Kaggle's hidden label construction or is empirically optimal.

The validation threshold table is a coverage audit, not label tuning for a favorable score. If later comparing thresholds or a fraction rule, name each protocol separately and rerun every method with that protocol; never compare methods evaluated against different relevance definitions as if the scores were commensurate.

## 7. MAP@5 definition

**Confirmed from the exam:** metric is MAP@K, higher is better, and this exam fixes K=5. **Confirmed from presentation pp.4–5:** normalize each user's AP by `min(K, |R_u|)`, then macro-average across users. The slides explicitly say to match the evaluator convention. The detailed live Kaggle scorer implementation was not retrievable, so empty-label/duplicate-prediction behavior and hidden relevance construction remain unconfirmed by the platform.

For the proposed local protocol, let `R_u` be the set of distinct positive program IDs with a qualifying validation airing, and `p_u[1:5]` the ordered predictions. Define `hit_u(r)=1` only for a relevant ID's first appearance; otherwise 0. Then:

```text
P_u@r  = (sum_{j=1..r} hit_u(j)) / r
AP_u@5 = (sum_{r=1..5} P_u@r * hit_u(r)) / min(5, |R_u|), when |R_u| > 0
MAP@5  = (1 / |U|) * sum_{u in U} AP_u@5
```

**Proposed edge policies:** empty `R_u` gives AP=0 for the primary all-user metric; duplicate predicted IDs cannot earn repeated hits and are rejected by the output validator; exactly five unique positive eligible IDs are required. Macro-average users, not views, and do not divide by the number of retrieved hits or by five when fewer than five relevant items exist.

The presentation's example is a useful future unit-test target: relevant `{A,B,C}`, ranking `A,B,D,C,E` gives `(1+1+3/4)/3 = 11/12`; `D,A,E,B,C` gives `(1/2+2/4+3/5)/3 = 8/15`. These are illustrative arithmetic examples, **not project experiment results or student personal verification**.

The slide's competition screenshot describes “five most preferred” programs from future viewing. It does not reveal how preference was computed or how many ground-truth IDs the scorer retains. Our proposed “any ≥120-second qualifying airing” relevance set is therefore a transparent offline proxy, not a verified recreation of private labels. Do not claim public/private MAP equivalence based on the metric formula alone.

## 8. Temporal validation protocol

### Fixed chronological periods

Use local calendar ISO-week boundaries, then convert to UTC for arithmetic:

| Role | Local half-open interval | Actual evidence and permitted behavior |
|---|---|---|
| Training, Weeks 11–20 | `[2020-03-09 00:00:00, 2020-05-18 00:00:00)` | 3,945,173 view rows touch training; fit all model statistics only here. |
| Validation, Weeks 21–30 | `[2020-05-18 00:00:00, 2020-07-27 00:00:00)` | 1,877,821 view rows touch validation, including the post-cutoff portion of 1,007 boundary-crossing rows. Labels/selection only. |
| Final fitting, Weeks 11–30 | `[2020-03-09 00:00:00, 2020-07-27 00:00:00)` | Refit chosen, frozen strategy using all available historical behavior only after comparison/selection. |
| Test prediction, Weeks 31–42 | `[2020-07-27 00:00:00, 2020-10-19 00:00:00)` | Use schedule for availability; no test viewing is available or permitted. Exclude Week 43 spillover. |

Under the Kyiv interpretation the UTC boundaries are March 8 22:00, May 17 21:00, July 26 21:00, and October 18 21:00 respectively. The training interval crosses DST, so fixed-offset conversion for the entire period is wrong. The two raw-row period counts sum to the full view count plus 1,007 because clipping splits crossing intervals, not because those seconds should be counted twice.

### Shared comparison contract (proposal)

- **Fit population:** all users with permitted training history, not just the test-template users. No validation history may create popularity counts, co-view edges, rules, user profiles, recency features, or hyperparameter-specific fitted statistics.
- **Primary evaluation users:** exactly the **1,260 original template users**, preserved as strings and in original order. This mirrors the known target cohort and keeps users fixed independently of model capability. Do not drop cold users to improve a score.
- **Relevance:** distinct positive IDs from ≥120 seconds of unioned overlap with at least one validation airing. No artificial truncation to five relevant items without confirmation of the hidden label rule.
- **Primary MAP:** all 1,260 users, empty relevance → AP=0. There are **1,246 nonempty-label users** and **14 empty-label users**. Report the count and also a clearly labelled nonempty-only MAP diagnostic for those same 1,246 users. The all-user theoretical ceiling from empty rows alone is 1,246/1,260 = 0.988889, not an achieved score.
- **Train history coverage:** 4,876 users have any training viewing; 4,837 have a ≥120-second labelled airing. Among template users, 1,256 have any training viewing and 1,250 have qualifying history: **10 cold users**, comprising four with no training viewing and six with viewing but no usable labelled overlap. Eight of the ten have nonempty validation relevance.
- **Final cold cases:** two template users have no ≥120-second labelled airing in either historical partition. Their histories require the shared fallback. This is a coverage diagnostic, not a final-model fitting result.
- **Common candidates, repeats, tie-breaking, fallback:** as in §9 for every method.
- **Selection:** a small predeclared hyperparameter grid, one common hold-out, all attempted comparisons retained. Select with MAP@5 and documented compute/coverage tradeoffs. Validation guides selection, never the fitted model's user-item statistics. Multiple trials on one hold-out are subject to selection optimism; disclose that limitation.
- **Final refit:** rebuild interactions over the entire history window, including recombining any airing fragments split by the now-removed validation boundary. Do not blindly concatenate thresholded train/validation tables, which could miss a boundary airing qualifying only in combination.
- Choose the same heavy and light/cold users from training-only history descriptors, with a deterministic rule and justification before looking for flattering recommendation examples. Compare their eventual lists against validation behavior, not inaccessible test outcomes.

## 9. Candidate-pool and cold-start policy

**Proposal:** include all positive program IDs having a valid airing with positive overlap with the prediction window. Use actual airing timestamps across adjacent schedule files, not merely the filename of the target schedule. Only the availability catalog may look ahead; learned user behavior may not.

| Timestamp-clipped catalog | Positive IDs |
|---|---:|
| Training-window schedule | 5,842 |
| Validation-window schedule | 6,384 |
| Test-window schedule | 6,649 |

The validation catalog has **2,391** IDs absent from the training-window schedule; test has **1,436** absent from the historical-window catalogs. Counts exceed the corresponding single-file counts where preceding-file spillovers genuinely air within the window. This is a proposed inclusive availability policy, not a claim Kaggle enforces this exact catalog. The scan used schedule metadata only; no future behavior was consulted.

**Allow previously watched programs for every method.** The exam does not forbid them. Among template users' 198,517 distinct validation user-program pairs at 120 seconds, 42,934 (21.63%) also occur in the same user's training history. This supports treating repeat consumption as possible; it is not a measured recommendation gain. “Allow repeats” means across time, not duplicate IDs inside one Top-5.

**Common fallback order, determined from training only:** candidate programs by descending distinct training-user support, then descending qualifying-airing count, then ascending numeric program ID. Cold items receive zero learned support and appear only after supported items unless a later individually evaluated method can score them. If all candidate supports are zero, deterministic ID ordering still provides five eligible IDs; no unrecorded random or future-popularity fallback is allowed.

**For every method:** filter IDs to the shared candidate catalog; remove 0 and duplicates; rank by method score with the same fallback ordering for ties; keep up to five positively supported method recommendations; fill missing slots from the common fallback without duplication. If a user lacks usable seeds/rules/graph reachability, return that fallback. Assert that at least five candidates exist; observed catalogs comfortably satisfy this. Do not treat independently scaled model scores as probabilities.

**Cold-item limitation:** the training inspection sees only 2,948 watched program IDs despite 5,842 scheduled candidates. Pure history, PPR, and rules cannot learn behavior for unseen items. This limits coverage and motivates later CBF exploration, but does not justify silently narrowing other methods' candidate pools.

## 10. Recommended algorithms

Implement the four planned methods in this order. The two qualifying exam methods are **program co-view PPR and Association Rules**. No method has been evaluated, and no performance claim is made.

| Priority | Method | Proposed definition | Feasibility and limitations |
|---|---|---|---|
| 1 | Required popularity baseline | Distinct training users with qualifying program viewing; common tie/fallback rule | One pass over deduplicated user-item presence. Essential reference; not one of the two approved methods. |
| 2 | Personal viewing frequency | Per-user count of distinct qualifying broadcasts of each ID, then shared fallback | Sparse lookup and ranking. Do not count raw fragments or conflate it with approved CF/MF. |
| 3 | Personalized PageRank | Nodes are program IDs; weighted **program–program co-view** edges built from training users; restart distribution from that user's qualifying history | Meets the specified graph requirement. Avoid replacing it with user-user PageRank or a popularity-only random walk. |
| 4 | Association Rules | Training user baskets, singleton antecedent/consequent rules `i→j` with support, confidence, lift | Approved method. Length-two itemsets keep computation bounded; this is a documented restricted rule-mining design, not a claim of full Apriori/FP-Growth enumeration. |

### PPR specification to implement, not an executed experiment

- Binary training incidence `X[u,i]`; integer co-view counts from `X.T @ X`, diagonal removed. Use at least int32 for counts, not uint8 multiplication that can overflow.
- Initial edge weight proposal: cosine-normalized co-user support `co_count(i,j)/sqrt(support(i)*support(j))`; require at least three co-view users; retain top 100 neighbors per item, symmetrize explicitly, then row-normalize. Deterministic tie handling.
- Personalized restart `s_u(i) ∝ log1p(qualifying_airing_count[u,i])` over training history. Initial damping α=0.85; dangling mass returns to `s_u`; iterate `p_next = α P.T p + (1−α)s_u` with the corresponding dangling correction.
- Initial convergence settings: L1 tolerance `1e-8`, maximum 200 iterations, record convergence and wall time. Candidate restriction applies at ranking time, not by deleting all historical seed nodes from the graph. If seeds lack graph edges, use the documented dangling/fallback policy.
- Small planned sensitivity grid: α in `{0.70,0.85}`, neighbor cap in `{50,100}` with other preprocessing fixed. All are untested configuration proposals.

### Association-rule specification to implement

- A basket is a user's distinct qualifying programs in the training window; each user contributes once to a pair's support. Define `N` as the 4,837 nonempty training baskets under the proposed threshold, recomputed by the production pipeline.
- For pair count `n_ij`, `support=n_ij/N`, `confidence(i→j)=n_ij/n_i`, `lift=confidence/(n_j/N)`.
- Initial rule filters: pair count ≥3, confidence ≥0.05, lift ≥1. Score a candidate by the sum over watched antecedents of normalized user-history weight × confidence × `log1p(lift)`; retain explanatory antecedents. A simple confidence-only alternative may be a separately logged ablation.
- Candidate repeats remain permitted. Missing rules or fewer than five supported consequents use the exact shared fallback, not a special model-only policy. A small support-count grid `{3,5,10}` is sufficient initially; avoid exhaustive hyperparameter search before correctness checks.

### Scale, memory, runtime

The DST-aware 120-second audit found **716,796 training user-program pairs**, 4,837 nonempty users, and 2,948 watched items. Median user basket: 132 items; maximum: 1,171. Enumerating every user's unordered pair would emit **76,705,132 pair occurrences** before aggregation. A Python dictionary of every event/pair is unnecessary; sparse integer multiplication or streamed pair accumulation is preferable. Both PPR and rules may reuse training-only co-view statistics while retaining different, independently evaluated scoring mechanisms.

Array-storage estimates, not measured model memory:
- Training CSR with float32 values/int32 indices is roughly **5.49 MiB** excluding maps, intermediates, and overhead.
- A dense 2,948² float32 item matrix is **33.15 MiB**; an 8,234² matrix is **258.63 MiB**. Dense co-occurrence intermediate storage is possible at this scale but sparse/pruned graphs give better headroom.
- A directed top-100 graph over 8,234 items needs about 6.3 MiB for value/index arrays before symmetrization and overhead. Batch PPR by 32–64 users to bound workspace. A Python-object graph can consume much more memory than arrays.
- Preprocessing diagnostics actually took minutes (§11). PPR cost is approximately `O(users × iterations × graph_edges)` without batching/reuse; rules' main cost is pair aggregation. Model wall times are **not measured yet**. Benchmark the initial configuration and cap grids before expanding scope.

**Dependencies:** NumPy/SciPy for sparse arrays and numerical operations, pytest for meaningful verification; NetworkX could be a tiny-graph reference but is not installed and is not needed for the production graph. `mlxtend` is optional for a small rule-check reference, not necessary for pair rules. All are currently uninstalled in the inspected interpreter.

**Defer until the mandatory comparison is verified:** CBF (sparse metadata and missing years), implicit MF/ALS (additional library/Python compatibility and hyperparameters), hybrids (evaluate individual components first; rank fusion avoids pretending raw scores are comparable). Do not create unused algorithm files now. The current diagnostic code uses only the standard library and no copied participant code.

## 11. Environment and dependencies

Observed environment:

| Property | Observed value |
|---|---|
| OpenCode | 1.18.30 |
| AI model reported by this harness | `openai/gpt-6-astra` |
| OS/platform | Linux 7.0.0-42-generic, x86_64, glibc 2.43 |
| CPU | Intel Core i7-6600U @ 2.60 GHz; 2 cores / 4 logical CPUs |
| RAM | `free -h`: 11 GiB total, approximately 3.4 GiB available at inspection |
| Swap | 4.0 GiB |
| Disk | Approximately 157 GiB available on the workspace filesystem |
| Python | `/usr/bin/python3`, 3.14.4, GCC 15.2.0; `python` command absent |
| pip | 25.1.1 for Python 3.14 |
| SQLite | 3.46.1 |
| Installed package observed | psutil 7.1.0 (not required by these scripts) |
| Not installed in inspected interpreter | numpy, scipy, pandas, polars, duckdb, networkx, scikit-learn, implicit, mlxtend, pytest, kaggle, pypdf |
| Other tools | Git and OpenCode executables present; `uv` not found in PATH |

This is the current interpreter, not a claim that no other environment exists on the computer. No packages were installed and no models were trained. Python 3.14 wheel compatibility must be tested before choosing later numerical-library versions; prefer an isolated compatible environment and pin versions actually run. Do not put invented tested versions in `requirements.txt`.

### Actual diagnostic execution

| Diagnostic | Wall time | Peak resident memory |
|---|---:|---:|
| Completed full CSV profiles + disk-backed exact duplicates + view index | 249.374 s | 324.74 MiB |
| Initial naive-clock interval audit | 130.181 s | 514.48 MiB |
| DST-aware full interval audit | 215.769 s | 515.22 MiB |

The first profiling attempt exceeded the tool's 120-second limit and was terminated; it was rerun successfully with a 600-second limit. Initial `python --version` failed because only `python3` is present. These are actual execution issues, not hidden successes. The profile then completed all five files; later diagnostics ran successfully. The exam's 4.6-second/12.6-MB reading benchmark is an instructor measurement on a different workload/environment, not this project's result.

Memory control: streaming CSV reads; 10,000-record SQL batches; 32-MiB SQLite page cache and file-backed temporary operations; channel-at-a-time interval processing; compact identity sets and counters; no full CSV DataFrames. The SQLite cache uses disk space and is excluded from Git. The diagnostic runtime includes substantially more than a raw read.

Reproduction commands are in the root README. Run from the project root with `python3`; preserve the original hash manifests before re-running `profile`. `integrity` and `verify` are inspection checks, not evidence that future recommenders or an evaluator have passed tests.

## 12. Project folder structure

The requested scaffold is created at:

`/home/dastan/Recommender Systems/FKN/SWEETTV_Final_Exam/`

```text
SWEETTV_Final_Exam/
    exam_quiz_FKN.md             # checked copy; original remains one level above
    README.md
    requirements.txt            # Step 1: standard library only
    .gitignore
    code/
        inspect_step1.py
        inspect_intervals_step1.py
        inspect_clock_step1.py
    data/
        README.md               # original paths; no moved/rewritten CSVs
    tests/
        README.md               # planned fixtures/check records
    results/
        step1_audit.md
        step1_profile.json
        step1_intervals.json
        step1_intervals_kyiv.json
        step1_clock.json
        original_sha256_before.json
        original_sha256_after.json
        homework_guidelines_sha256.json
        step1_verification.json
        step1_inspection.sqlite  # derived, ignored by Git
    submission/
        README.md               # no generated prediction file
    report/
        README.md               # no final PDF yet
    evidence/
        README.md
        step1_opencode_session.json  # native live-session checkpoint
```

All Python source, including future Python test modules, belongs under `code/`. `tests/` can hold non-Python fixtures and verification records, satisfying both directory and Python-location requirements. No unused model modules were created. Original Kaggle files remain at the inspected external sibling path; `data/README.md` records that location. Later reproduction instructions should parameterize paths for a clean clone. No Git initialization, commit, push, or public repository was performed in Step 1.

## 13. Gate 0 risks

The primary exam makes Gate 0 an automatic exam failure. The located homework rules explicitly identify invalid/fabricated/misattributed references (including broken URLs), claim–evidence mismatch, fabricated benchmarks, non-running solutions claimed to run, hallucinated APIs, materially wrong reasoning/root causes, claimed verification not performed, missing/noncorresponding session evidence, cherry-picked metrics, and a generic discussion without substantive 5-Loop engagement as major failures.

| Risk in this project | Required resolution/evidence |
|---|---|
| Counting popularity or personal frequency as an approved second method | Implement and individually evaluate both program co-view PPR and Association Rules. |
| Float/signed-int64 user-ID damage | Keep identity strings; validate exact template equality and ordering on output. |
| Wrong local clock or filename-only split | Preserve DST evidence and explicit timestamp boundaries; test clipping and elapsed-duration invariants. |
| Duplicate fragments inflate support/frequency | Union per-user/channel time and aggregate per airing; binary support per user/item. |
| Future user behavior leaks into models | Assert training maximum timestamp and provenance for all fitted statistics; hold validation labels outside fitting. |
| Claiming a local score is identical to Kaggle's hidden metric/labels | Distinguish course-confirmed formula from unconfirmed hidden relevance/edge-case behavior. |
| Dropping the 14 empty-label users or 10 cold users selectively | Freeze and report the all-1,260 protocol, plus labelled secondary diagnostics consistently. |
| Treating ID 0 as a program or missing labels as dislike | Exclude 0 from items and predictions; document missingness/coverage limitations. |
| Instructor/other participant's score presented as this student's | No project Kaggle score exists; later retain the exact CSV, submission ID, returned private score, and screenshot from the student's account. |
| Report asserts experiments, performance, business lift, or verification not done | Trace each claim to executed outputs; no model results exist at this stage. Distinguish this agent's checks from the student's own checks. |
| External code or citations falsely attributed or unverified | Record any actual adaptations and license/source details. No participant code/predictions were consulted or adapted here. Verify sources before citing their claims. |
| Edited or incomplete “original session” | Preserve native exports; a checkpoint is not the completed project's full session. Never reconstruct, sanitize, or manually merge the required original export. |
| Omitted course deliverable | Later PDF + native session only via course channel; GitHub must contain exact final scored CSV and implementation; PDF needs private-score screenshot, not just a written number. |
| Thin AI-generated reflection or invented personal work | Student must actually engage in ASK WHY/TEST/EXPLAIN/IMPROVE, verify the required quantities, and write truthful personal reasoning. |

### Session preservation

The native OpenCode listing identified session **`ses_e7371ba5affeMuqWtLibI0NrzK`**, titled **“SWEET.TV Step 1: Requirements and Data Audit”**, in the FKN directory. `opencode export --help` confirms a native JSON export command. A direct, unmodified live-session checkpoint is saved as `evidence/step1_opencode_session.json`; its hash/session ID/message count are recorded by the closing verifier. No sanitization or transcript rewriting is used.

The checkpoint necessarily predates later verification calls and the final response. Preserve it unchanged and export a new full-project session after all later work, using a new filename. Do not claim the checkpoint includes future Step 2 work. If multiple sessions are necessary, keep them natively intact and clarify packaging with the instructor rather than fabricating one log. Session evidence is ignored by Git by default but must be supplied through the required course channel at the end.

## 14. Missing information

**No required dataset is missing.** The primary exam, presentation, and a copy of the homework guidelines were found and read. Remaining items are information or future deliverables, not missing source CSVs:

1. **Live official evaluator details:** competition-specific empty-ground-truth policy, handling of repeated predictions, and exact construction of hidden preferred-program labels. The available course formula is confirmed; Kaggle's full live definition is not.
2. **Explicit timezone declaration:** strong DST evidence supports Europe/Kyiv-compatible local time, but no supplied source names the timezone. Use and disclose the supported inference; seek confirmation if possible.
3. **Report cap/current guideline version:** five versus six pages conflict; whether the located Week 2 guideline is the latest binding version. No new language rule appears in the exam.
4. **Student account details/access:** Kaggle username, accepted rules/current quota, preferred Russian spelling for the submission description, GitHub account/repository URL, student email for the report, and final course submission channel. Do not invent any of these.
5. **Future evidence:** no trained models, method-comparison scores, final strategy, recommendation example lists, Kaggle submission ID/private-score screenshot, final PDF, or full-project final session export exists yet. They must be created from genuine later work.
6. **Dependency compatibility and model timings:** library versions and numerical runtime/memory must be measured after installation in the later implementation environment, not inferred from another machine.

The failed live-page retrieval does not prevent a local implementation under the explicitly documented course convention. It does prevent claiming a fully verified Kaggle scorer specification. All unresolved matters should be checked before final report/submission claims are made.

## 15. Exact Step 2 implementation plan

**This is a plan only. Step 2 has not begun.** Keep the four mandatory/reference implementations and the shared evaluation as the next milestone; optional CBF/MF/hybrids follow only after that milestone is verified.

1. **Freeze a protocol configuration.** Record the 120-second per-airing rule, Kyiv-compatible clock inference, the exact four boundaries in §8, all 1,260 evaluation users, empty-label AP=0, shared candidate/fallback/tie policies, allowed repeat consumption, and a fixed seed (proposed 42). Confirm unresolved instructor/Kaggle items when possible; version any changed protocol rather than silently replacing it.
2. **Create an isolated numerical environment.** Install NumPy, SciPy, pytest in a compatible Python environment; record actual versions, CPU/thread settings, and package lock. Keep all execution inside OpenCode and all Python files inside `code/`. Update `requirements.txt` only to versions actually verified.
3. **Implement safe loaders and preprocessing (`code/data_pipeline.py`).** Hash-check originals; exact string users; explicit timestamp formats; localize then use UTC; exclude nonpositive airings; timestamp-filter schedules; clip views at split boundaries; union same-user/channel fragments; join labelled intervals; aggregate per-airing seconds before thresholding. Produce separately named derived interactions and coverage summaries under `data/`/`results/`; never edit source files.
4. **Verify preprocessing (`code/test_data_pipeline.py`).** Include a manual interval-overlap fixture, touching endpoints, duplicate and overlapping fragments, multiple fragments of one airing, a session spanning programs, zero IDs, schedule gaps, DST crossing, and the training cutoff. Assert all template IDs remain exact strings, all credited seconds are bounded, no train view extends beyond cutoff, and actual row/user/catalog coverage reconciles with this audit or has a documented reason for differences. Rebuild full-history airings afresh for any eventual final fit.
5. **Implement the common evaluator/ranker (`code/evaluation.py`, `code/test_evaluation.py`).** Test the slide examples, empty relevance, repeated predictions, more/fewer than five relevant IDs, macro-averaging, candidate filtering, and common fallback. Student separately hand-computes and records at least one metric and one real join example; do not label an AI-generated test as personal verification.
6. **Implement/evaluate popularity and personal frequency (`code/baselines.py`).** Use training-only distinct-user support and qualifying-airing counts. Save Top-5 validation diagnostics, primary/all-user MAP, nonempty-user MAP, cold/warm coverage, and timing with the protocol identifier. These are not the two approved methods.
7. **Implement/evaluate program co-view PPR (`code/ppr.py`, `code/test_ppr.py`).** Build the specified item graph and personalized restart, verify a tiny hand-checkable graph, stochastic normalization, dangling handling, convergence, disconnected seeds, and candidate restriction. Run the initial configuration before the small predeclared grid; record failures and all attempts.
8. **Implement/evaluate Association Rules (`code/association_rules.py`, `code/test_association_rules.py`).** Compute training support/confidence/lift without duplicated-user contributions; verify a tiny basket example independently; rank and use the identical fallback. Run the initial filters, then the limited support grid. Compare it individually with both baselines and PPR.
9. **Create one common comparison runner (`code/run_validation.py`).** Assert identical user IDs, labels, candidates, cutoff, threshold, and K across all methods. Save actual metrics, settings, run times, memory measurements, seeds, dependency versions, and output hashes in `results/`. Do not invent a success if a method fails or does not converge.
10. **Produce the required two-user evidence (`code/analyze_users.py`).** Select a heavy and light/cold template user with a documented training-history-based rule; use the same IDs for every method. Show each ordered five-ID list with historical title/category lookup, fallback flags, and validation hits. Explain direct rules versus graph diffusion, popularity/frequency behavior, familiarity/discovery limits, and observed failure cases. Discuss CBF/MF mechanisms only as untested alternatives unless actually implemented later.
11. **Mandatory-comparison checkpoint.** Confirm two approved methods plus popularity are implemented and separately evaluated with meaningful verification. Select a provisional strategy from real results and cost; preserve all evidence and a new native session checkpoint. Stop for review before extending scope to CBF/MF/hybrids or producing a Kaggle artifact.

**Later, after approval of the implementation/evaluation milestone:** optionally evaluate justified extensions; freeze the final strategy; refit with all behavior through Week 30; generate and validate exactly five ranked eligible positive IDs for each original template user for Weeks 31–42; submit through the student's accepted Kaggle account; retain exact submitted CSV/hash and returned private-score screenshot; publish required GitHub contents; prepare the truthful PDF and original final session export. These actions were not performed in Step 1.
