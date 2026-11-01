# Step 2 — viewing-to-program preprocessing

**Student:** Abaidullah Seikh · **Instructor:** Seungmin Jin  
**Stage:** preprocessing and independent verification only  
**Source of requirements:** completed `results/step1_audit.md` and all 125 lines of `exam_quiz_FKN.md`, read before implementation.

**Outcome:** the complete historical viewing-to-broadcast pipeline ran successfully. The selected 120-second rule produces **716,796 training** and **569,104 validation user–program pairs**, exactly matching Step 1. All **26 current tests** and **100 full-data verification checks** pass. One initial timestamp-validation test failed, was fixed, and remains recorded in the evidence. The original five CSVs and the original Step 1 OpenCode export have unchanged SHA-256 hashes.

The current Step 2 request restricts this milestone to preprocessing; it supersedes the broader implementation sequence proposed in Step 1 §15. Recommendation methods and evaluation remain later milestones. No recommendation models, MAP@5 scores, or Kaggle prediction file were produced.

## Scope and protocol

### Exact schemas and identity preservation

The loader requires the inspected column order and fails on an unexpected header or record width:

- Views: `user_id,vsetv_id,start_time,stop_time,duraton`.
- Schedules: `channel_id,channel_title,start_time,duration,tv_show_title,tv_show_id,tv_show_category,tv_show_genre_1,tv_show_genre_2,tv_show_genre_3,year_of_production,director,actors`.
- Original template: `user_id,tv_show_id`.

User IDs remain original strings from ingestion through SQLite, CSV output, grouping, and coverage checks. No conversion to float or integer is used for user identity. Verification includes IDs above signed-int64 range and distinct strings `000123` and `123`. Consumers of derived CSVs must continue to load `user_id` as text.

### Time interpretation and boundaries

**Europe/Kyiv is an evidence-supported inference, not an officially declared competition timezone.** Viewing timestamps are explicitly parsed as `YYYY-MM-DD HH:MM:SS`; schedule starts as `DD.MM.YYYY HH:MM:SS`. Dates and clock fields are validated. Transition-day local times are round-tripped through UTC: nonexistent times are rejected, and ambiguous autumn times require an explicit fold. The supplied periods precede the autumn transition, but that behavior is tested.

Elapsed-time arithmetic uses UTC integer seconds. Broadcast end is **UTC-localized start + elapsed schedule duration**, so adding an hour across the spring clock jump is correct. Viewing endpoints are localized separately, and their elapsed difference must equal the supplied `duraton`. This check succeeded for every source viewing record, including the 117 DST-crossing records identified in Step 1. A cached day offset accelerates ordinary dates; transition dates take the explicit fold/round-trip path.

| Partition | Local half-open interval | UTC Unix-second interval |
|---|---|---|
| Train | `[2020-03-09 00:00, 2020-05-18 00:00)` | `[1583704800, 1589749200)` |
| Validation | `[2020-05-18 00:00, 2020-07-27 00:00)` | `[1589749200, 1595797200)` |
| Full history | `[2020-03-09 00:00, 2020-07-27 00:00)` | `[1583704800, 1595797200)` |
| Test availability only | `[2020-07-27 00:00, 2020-10-19 00:00)` | `[1595797200, 1603054800)` |

Training/validation viewing is clipped separately before matching. A future-only viewing record cannot enter either partition; the integration test inserts one explicitly and verifies its exclusion. Test schedule rows contribute availability IDs only and are never joined to user behavior. Full-history outputs are prepared data for possible later fitting; **they must not be used to train a model being evaluated on the Week 21–30 hold-out**.

### Join, union, and configurable threshold

1. Stream and validate original files; check their hashes against Step 1 before processing and again afterward.
2. Use a disk-backed sort by `(channel,user,start,end,original_record)`. Remove exact duplicate viewing records by their complete original parsed CSV field values. Preserve original files.
3. Within each user/channel and partition, union overlapping or touching viewing intervals. A continuous session may match several broadcasts; distinct channels are not merged into one artificial viewing sequence.
4. Deduplicate broadcast identities `(channel,start_utc,end_utc,program_id)` in the channel index. Invalid zero-duration broadcasts are excluded. Conflicting overlapping positive-ID airings on a viewed historical channel cause a clear failure rather than silent double credit; none occurred in this run.
5. Match only the same channel and positive half-open interval overlap:

   `max(0, min(view_end,broadcast_end) - max(view_start,broadcast_start))`.

6. Sum disjoint matching fragments **per user and broadcast** before deciding whether it was watched. Keep the threshold-independent seconds and first/last matched timestamps on disk, including overlaps shorter than 120 seconds.
7. Combine the two disjoint partitions' raw per-broadcast overlaps to form full history **before thresholding**. This is not concatenation of thresholded training and validation tables.
8. Derive user–program pairs by keeping broadcasts whose summed overlap is at least the requested threshold, then aggregating qualifying-broadcast counts, credited seconds, and first/last qualifying watch times for each user and program.

The selected threshold is **120 elapsed seconds per user/broadcast**, configured with `--threshold-seconds 120`. It is the Step 1 evidence-based starting rule, not a universal watched definition or a verified hidden Kaggle label rule. A second **300-second** dataset was actually derived from the same saved overlap cache. No raw interval join was rebuilt for that alternative. Different airings of the same program cannot pool subthreshold seconds to pass the per-broadcast cutoff.

## Implementation and derived outputs

### Source and verification files

| File | Purpose |
|---|---|
| `code/data_pipeline.py` | Production ingestion, explicit timezone parsing, exact deduplication, union, interval index, threshold-free join, and separate threshold derivation CLI |
| `code/test_data_pipeline.py` | Unit/integration suite and independent brute-force reference |
| `code/verify_step2.py` | Full-output invariants, independent SQL aggregation, history-merge check, Step 1 comparison, source/session integrity |
| `tests/interval_fixtures.json` | Eleven hand-specified fixtures with exact expected overlap records |
| `tests/run_step2_tests.sh` | Test entry point: `sh tests/run_step2_tests.sh` |
| `results/step2_test_runs/run_001.json` | Initial test attempt, including the real failed assertion |
| `results/step2_test_runs/run_002.json` | Corrected parser suite, all passing |
| `results/step2_test_runs/run_003.json` | Final suite after moving temporary test files under the project, all passing |
| `results/step2_verification.json` | Passed/failed full checks, all test attempts, output statistics, hashes, comparison deltas, timing and memory |
| `results/step2_preprocessing_report.md` | This report |

All Python implementation and test code is inside `code/`. `tests/` contains the executable shell entry point, fixtures, and documentation, satisfying the requested test directory while preserving the all-Python-in-`code/` requirement. The root, data, tests, and dependency documentation were updated for this stage.

### Derived data layout

```text
data/step2/
    join_manifest.json
    train_airing_overlaps.csv.gz
    validation_airing_overlaps.csv.gz
    history_airing_overlaps.csv.gz
    train_candidates.csv
    validation_candidates.csv
    history_candidates.csv
    test_candidates.csv
    template_users.csv
    user_coverage.csv
    threshold_120s/
        interaction_manifest.json
        train_interactions.csv.gz
        validation_interactions.csv.gz
        history_interactions.csv.gz
        template_user_coverage.csv
    threshold_300s/
        interaction_manifest.json
        train_interactions.csv.gz
        validation_interactions.csv.gz
        history_interactions.csv.gz
        template_user_coverage.csv
```

**Threshold-free overlap schema:**

`user_id,channel_id,broadcast_start_utc,broadcast_end_utc,tv_show_id,overlap_seconds,first_overlap_utc,last_overlap_utc`.

One row represents a user and one airing within one partition, with positive unioned seconds. First/last timestamps bound the fragments; their difference can exceed credited seconds when there are gaps. Both seconds and full broadcast duration remain available for later absolute- or fraction-threshold exploration.

**Thresholded interaction schema:**

`user_id,tv_show_id,qualifying_airings,watched_seconds,first_watch_utc,last_watch_utc`.

One row represents one user–program pair. `watched_seconds` sums only qualifying airings, and `qualifying_airings` counts broadcasts, not raw viewing fragments. Repeated programs across broadcasts remain meaningful. Interaction rows are sorted deterministically by exact user string and numeric program ID. Gzip files use fixed timestamps and no embedded filename so identical table contents produce stable compressed bytes.

Candidate files contain only positive program IDs airing within the named timestamp window. `template_users.csv` and each threshold-specific coverage table retain all 1,260 template users in original order, including users with zero qualifying interactions. `user_coverage.csv` records every raw-view/template user and whether each historical period contains any viewing.

The manifests include code hashes, input/output SHA-256 hashes, UTC boundaries, threshold configuration, environment version, real counts, time, and memory. Disposable SQLite workspaces were removed after successful generation. Derived data is excluded from Git by the existing ignore rules; it can be regenerated from the recorded originals.

## Measured interaction counts and coverage

### Selected 120-second protocol

| Quantity | Train | Validation | Full history |
|---|---:|---:|---:|
| Raw viewing rows touching partition, before duplicate removal | 3,945,173 | 1,877,821 | 5,821,987 |
| Users with any viewing | 4,876 | 4,820 | 4,914 |
| Positive user–broadcast overlaps, before threshold | 1,823,893 | 1,191,015 | 3,014,157 |
| Users with any positive labelled overlap | 4,841 | 4,806 | 4,895 |
| **Qualifying user–broadcast interactions, ≥120 s** | **1,531,876** | **1,028,093** | **2,559,270** |
| **Distinct qualifying user–program pairs** | **716,796** | **569,104** | **1,164,296** |
| **Users with qualifying interactions** | **4,837** | **4,804** | **4,891** |
| **Programs with qualifying interactions** | **2,948** | **3,200** | **4,325** |
| Credited seconds in qualifying broadcasts | 2,378,091,661 | 2,084,440,507 | 4,462,534,914 |
| Available positive program IDs | 5,842 | 6,384 | 8,233 |

The test-window availability catalog contains **6,649** positive IDs; no test-view interaction file exists. Full-history pair counts are not the sum of training and validation pair counts because user/program identities overlap across periods. Per-airing history counts also account for broadcasts spanning the partition boundary.

### Fixed template-user coverage

| Coverage among 1,260 template users | Train | Validation | Full history |
|---|---:|---:|---:|
| Any viewing | 1,256 | 1,246 | 1,260 |
| ≥120-second labelled interaction | 1,250 | 1,246 | 1,258 |
| No qualifying interaction | **10** | **14** | **2** |
| Distinct qualifying user–program pairs | 238,142 | 198,517 | 393,725 |

The training cold cases include four users with no training viewing and six with viewing but no qualifying labelled history. Empty-history users remain explicitly represented in coverage files. Later recommenders will need the common fallback; this pipeline does not generate any recommendation or fallback list. Validation labels are local observed-overlap labels, not verified reproductions of hidden Kaggle preferences.

### Alternative 300-second derivation actually executed

| Quantity | Train | Validation | Full history |
|---|---:|---:|---:|
| Qualifying user–broadcast interactions | 1,224,147 | 855,420 | 2,078,922 |
| Distinct user–program pairs | 573,058 | 474,141 | 949,493 |
| Users | 4,833 | 4,799 | 4,890 |
| Programs | 2,901 | 3,160 | 4,276 |
| Template users with qualifying history | 1,250 | 1,246 | 1,258 |

Both derivations reference the **same join-manifest SHA-256**. This demonstrates threshold reuse and records coverage sensitivity; it is not a recommendation-quality comparison or threshold selection based on MAP.

## Missing IDs and data quality

### ID-zero policy

`tv_show_id=0` represents unlabelled airtime. Such rows participate only in schedule-coverage diagnostics. They are excluded from the positive-airing index, overlap artifacts, user–program interactions, and candidate catalogs. Unlabelled or unmatched viewing is not treated as dislike and no missing ID is inferred from titles.

| Schedule | ID-zero rows | Zero-duration rows excluded |
|---|---:|---:|
| `export_arh_11-20-final.csv` | 958,622 | 1 |
| `export_arh_21-30-final.csv` | 952,889 | 0 |
| `export_arh_31-42-final.csv` | 1,150,604 | 0 |

The zero-duration row has ID 0 and is excluded before interval indexing. All other verified source numeric/timestamp records passed the loader. The pipeline fails on invalid duration or ambiguous timestamps instead of silently changing their interpretation.

### Viewing-record accounting

- Source viewing records: **5,821,987**.
- Exact duplicate records removed from derived processing: **17**.
- Unique records after exact deduplication: **5,821,970**.
- Unioned historical user/channel segments: **5,441,773**.
- Raw elapsed viewing seconds: **8,619,215,224**.
- User/channel union seconds: **8,502,989,921**.
- Duplicate/overlap seconds eliminated: **116,225,303**.
- Of unioned seconds, **7,996,509,692** have schedule coverage and **4,497,257,030** have labelled schedule coverage.

Diagnostic match categories retain raw-record counting to compare directly with Step 1:

| Raw-record category | Records |
|---|---:|
| At least one labelled overlap | 2,964,378 |
| Schedule coverage but no labelled overlap | 2,444,312 |
| No schedule overlap | 413,297 |
| **Total** | **5,821,987** |

Of the unmatched/gap population, **113,579** records use channels absent from both historical schedules. The pipeline keeps channel-specific evidence and does not collapse concurrent different-channel sessions into purported single-person attention. These logged elapsed seconds should not be interpreted as verified unique human attention or enjoyment.

## Independent verification and fixes

### Current test result: 26 tests passed

The suite contains eleven hand-specified fixtures, 100 seeded random comparisons within one test, and additional parsing, threshold, leakage, and end-to-end tests. Random fixture generation uses seed **42**. The brute-force reference enumerates each integer second in a short broadcast and asks whether **any** same-user/same-channel raw view covers it. It uses no production timestamp parser, interval-union helper, binary-search index, or matching function.

| Mandatory requirement | Executed verification |
|---|---|
| Correct channel matching | A same-time wrong-channel broadcast receives no credit; randomized multi-channel cases also compare against the reference. |
| Exact overlap | `[30,180)` viewing intersected with `[0,100)` airing credits exactly 70 seconds. |
| Touching boundaries | Views touching only a start/end endpoint credit zero seconds. |
| Duplicate viewing records | Repeated views do not multiply seconds; the miniature CSV run records one exact duplicate removal. |
| Overlapping fragments | Nested and overlapping fragments credit the union, not their duration sum. |
| Multiple fragments of one broadcast | Two disjoint 70-second pieces aggregate to 140 seconds and pass 120 only after aggregation. |
| One view spanning programs | Three consecutive airings receive their respective 50/100/50-second overlaps. |
| Missing program IDs | ID-zero airings never become interactions; unlabelled-only users retain empty coverage. |
| Cold-start users | No-view, wrong-channel, and unlabelled-only cases are empty; template coverage preserves them. |
| DST | Spring `02:30→04:30` is 3,600 elapsed seconds; broadcast end and 1,800-second partial overlap are verified. Nonexistent spring and ambiguous autumn local times are checked. |
| Train/validation boundaries | Each side receives only its clipped 80 seconds; full history recombines to 160. Appended future-only behavior does not alter training output. |
| Exact user IDs | Large unsigned-looking strings and leading-zero strings survive parser, SQLite, and CSV round trips unchanged. |

Additional checks cover explicit formats and invalid dates/clock fields; correct UTC week boundaries; repeated broadcasts not pooling subthreshold seconds; duplicate schedule identities; conflicting labelled airings; unioned schedule coverage; threshold edge values; and corrupted-cache rejection.

The miniature CSV integration test executes complete ingestion and derivation, hides the source directory, and successfully derives a different threshold **without raw inputs**. It repeats the 120-second derivation and separately rebuilds the miniature raw join, comparing output-file hashes byte-for-byte. That is actual fixture-level reproducibility verification; a second full production raw-join run was not claimed or required.

### Recorded failure and fix

1. `run_001.json`: **26 tests, one failed subtest, zero errors.** The parser accepted `2020-05-18 24:00:00` because this Python 3.14 `datetime.fromisoformat` accepts that notation. The intended verified schema allows hours 00–23 only.
2. **Fix:** explicit hour/minute/second bounds were added before parsing; invalid folds are also rejected. The format/date checks remain in place.
3. `run_002.json`: **26 passed, zero failures/errors** after the fix.
4. The integration test's temporary directory was then moved under `tests/` so temporary work stays in the project. `run_003.json`: **26 passed, zero failures/errors**, including the fixed timestamp test. This final source hash is checked by the full verifier.

There were no failed full-dataset preprocessing commands. The original failed test evidence was retained rather than overwritten or omitted.

### Full production-output verification: 100 checks passed, zero failed

`code/verify_step2.py` independently:

- Checks hashes of every generated artifact, original CSV, and the existing Step 1 session export.
- Scans **every overlap row** in all three partitions, checking exact original user membership, positive eligible program IDs, unique sorted broadcast identities, positive/bounded seconds, and temporal cutoffs.
- Computes expected user–program aggregates using **SQLite `GROUP BY`, `COUNT`, `SUM`, `MIN`, and `MAX`**, separately from the production Python per-user aggregator. Every output row for **both thresholds and all three partitions** matches, with zero mismatches.
- Reconciles independently measured row/user/program/seconds counts with the manifests.
- Streams and merges the unthresholded train/validation files, checking **every full-history overlap row** against the expected recombination.
- Checks the same audited Step 1 quantities, original template order, complete cold-user coverage, pipeline/test source hashes, and all-Python-in-`code/` placement.

These are agent-executed checks through OpenCode. They do not claim that the student personally hand-verified any result. The final exam's personal-verification/disclosure requirement remains applicable.

## Comparison with Step 1

**All directly comparable Step 1 quantities matched exactly**, including positive user–broadcast counts, 120- and 300-second qualifying-airing/user/template counts, 120-second user–program pairs and unique programs, temporal raw-view counts, missing-ID match categories, schedule/labelled union seconds, 17 duplicate events, and timestamp-clipped train/validation/test catalogs. Each comparison and zero delta is saved in `step2_verification.json`.

Two scope-dependent differences were explicitly investigated:

### 1. Historical file union 8,234 versus history-window catalog 8,233

Step 1 reported **8,234** distinct positive IDs across the two entire historical schedule files. Timestamp-clipped history has **8,233**. The independent scan found the sole excluded ID:

- Program **90015141**, title `Х/ф 'Большой белый обман'.`
- File: `export_arh_21-30-final.csv`.
- Start: **27 July 2020, 03:10 local**, duration 5,700 seconds.
- This is after the **27 July 00:00** historical cutoff, despite the file's name.

The discrepancy is therefore an intended time filter, not lost historical interaction data. The train, validation, and test catalogs remain exactly 5,842 / 6,384 / 6,649 as in the Step 1 timestamp-based audit.

### 2. Full-history broadcast recombination

**751 user–broadcast identities** have positive overlap in both historical partitions. Therefore:

`1,823,893 + 1,191,015 − 751 = 3,014,157` full-history positive user–broadcast rows.

At the selected 120-second threshold:
- **700** cross-boundary user–broadcasts qualify separately on both sides and must count only once in full history.
- **One** cross-boundary user–broadcast qualifies only after the two individually subthreshold portions are combined.
- Hence `1,531,876 + 1,028,093 − 700 + 1 = 2,559,270` qualifying history airings.

At 300 seconds, the corresponding numbers are **653** qualifying on both sides and **eight** qualifying only after recombination, giving `1,224,147 + 855,420 − 653 + 8 = 2,078,922`.

This confirms the Step 1 warning against concatenating already-thresholded partitions. There is no unresolved coverage discrepancy.

## Runtime, memory, and reproducibility

### Actual measurements

| Executed stage | Wall time | Peak process RSS |
|---|---:|---:|
| Full raw ingestion, sorting, union, join, manifests/hashes | **214.184 s** | **99.11 MiB** |
| Selected 120-second derivation | **56.120 s** | **96.96 MiB** |
| Alternate 300-second derivation | **48.125 s** | **96.91 MiB** |
| Final unit/integration suite | **1.089 s** | Not separately reported by the suite |
| Independent full-output verification | **142.549 s** | **93.36 MiB** |

Primary preprocessing plus selected derivation took **270.304 seconds**, approximately **4 minutes 30 seconds**. The additional 300-second derivation and independent full-data verification are reported separately. Within the raw build: schedule ingestion 27.530 s, viewing ingestion/indexing 110.032 s, interval join 69.897 s; remaining time includes integrity and artifact work.

Measurements come from `time.monotonic()` and Linux `resource.getrusage(...).ru_maxrss / 1024`. They are process peak resident memory, not estimated model memory or filesystem-cache/disk-usage measurements. The standard library and SQLite disk-backed sorting keep the full view log out of RAM; buffers contain at most 10,000 insertion records, the SQLite cache is configured at 32 MiB, and matching is per channel/user. The workload differs from Step 1's broader inspection, whose higher peak memory is not a model benchmark comparison.

Environment: Python **3.14.4**, system `zoneinfo` data, standard-library CSV/gzip/SQLite/unittest, on the CPU/Linux environment documented in Step 1. No numerical libraries or new packages were required or installed for this stage. Model dependencies remain a later concern.

### Reproduction commands

Run from `/home/dastan/Recommender Systems/FKN/SWEETTV_Final_Exam/`:

```bash
# Tests append a new numbered JSON attempt; earlier failures remain available.
sh tests/run_step2_tests.sh

# Fresh output directory required. The original-hash baseline is checked first.
python3 code/data_pipeline.py build

# Threshold derivations use only data/step2/ overlaps and their manifest.
python3 code/data_pipeline.py derive --threshold-seconds 120
python3 code/data_pipeline.py derive --threshold-seconds 300

# Verify the saved production outputs and comparison with Step 1.
python3 code/verify_step2.py
```

The production outputs above already exist. Both `build` and `derive` refuse to overwrite an existing output directory. To regenerate without changing current evidence:

```bash
python3 code/data_pipeline.py build --output-dir data/step2_replay
python3 code/data_pipeline.py derive --join-dir data/step2_replay --threshold-seconds 120
```

The full-data verifier currently targets the canonical `data/step2/` outputs. For a fresh clone/workspace, use the first command sequence. For an additional threshold on the existing cache:

```bash
python3 code/data_pipeline.py derive --threshold-seconds 60
```

The 60-second production command is an instruction, **not a claimed full-data run**; a 60-second derivation was executed only on the integration fixture. The actual production alternatives retained here are 120 and 300 seconds. For another source location use `build --source-dir PATH`; the default expected-hash manifest still requires the original five CSV contents. Derived-output manifests contain all relevant table hashes; performance timings in JSON need not be identical between machines/runs.

## Integrity and unresolved questions

- **Original datasets unchanged:** the raw build hashes all five before and after processing, and the independent verifier rehashes them against Step 1. Every comparison passed. Hash values remain those in `results/original_sha256_before.json` and are repeated in `results/step2_verification.json`.
- **Original OpenCode export unchanged:** `evidence/step1_opencode_session.json` retains SHA-256 `797d19d6f36a03402f1bfeb69a40b7329c7c69b63bc540105fae2607943a7c6f`. It was read for verification only, never edited or replaced.
- **Code provenance:** new production and verification code was written and executed in this OpenCode session. The production implementation follows the documented Step 1 design; the brute-force and SQL references use separate computational approaches. No other participant's code or predictions were used.
- **Current implementation/test status:** no unresolved failing test, output-invariant violation, or unexplained Step 1 coverage mismatch remains.
- **External uncertainty remains:** Europe/Kyiv is an inference; official hidden Kaggle relevance construction and detailed scorer edge cases remain unverified. Local threshold labels must not be represented as the private labels. The report page-cap/account questions from Step 1 also remain outside this preprocessing milestone.
- **Operational limitation:** production matching deliberately fails on conflicting positive-ID airings on a viewed historical channel. None occurred in the actual run; any newly supplied conflicting schedules would require an explicit resolution policy and new verification.
- **Next stage:** proceed only on the user's instruction. Step 2 ends with validated preprocessing artifacts; no recommendation algorithm or MAP@5 result is part of this deliverable.
