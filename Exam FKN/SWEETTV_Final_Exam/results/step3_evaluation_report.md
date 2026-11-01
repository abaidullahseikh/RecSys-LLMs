# Step 3 — MAP@5 and temporal validation

**Student:** Abaidullah Seikh · **Course:** LLM4Rec — Final Exam (FKN)  
**Scope:** evaluator, fixed validation protocol, candidate verification, and reusable fallback interface. No recommendation model was implemented or fitted, and no Kaggle output was generated.

**Outcome:** **34 automated tests passed**, followed by **110 full-data/contract/integrity checks passed, zero current failures**. The frozen cohort contains **1,260 users**, including **1,246 with labels**, **14 without labels**, and **10 without qualifying training history**. Validation has **6,384 eligible programs**. The framework is ready for Step 4 under the explicitly documented local-label and timezone assumptions.

The seven required documents/code/manifests were read completely before implementation. The compressed files' real headers and sample records were inspected and saved before use. Completed Step 2 preprocessing was neither rerun nor modified.

## Fixed evaluation protocol

The primary protocol is selected before any recommendation-method evaluation:

| Decision | Frozen policy |
|---|---|
| Ranking length | Exactly five unique eligible IDs in production evaluation |
| Primary population | All 1,260 original template users, in original order |
| Relevance | All distinct positive program IDs with at least one qualifying validation-period airing |
| Watched threshold | Existing Step 2 rule: at least 120 elapsed unioned viewing seconds per user/broadcast |
| AP normalization | `min(5, number of distinct relevant programs)` |
| Empty-label users | AP@5 = 0, retained in primary MAP |
| Secondary diagnostic | Macro MAP over the same cohort's 1,246 nonempty-label users |
| Previously watched programs | Allowed, identically for every method |
| Duplicate IDs within one ranking | Invalid in strict production evaluation |
| Unknown/zero/malformed predicted IDs | Rejected in strict production evaluation |
| Candidate availability | Positive-ID broadcasts overlapping the target period; schedule metadata only |
| Cold-start fallback | Future training-only popularity ranking; interface and synthetic fixtures only in Step 3 |
| Timezone | Europe/Kyiv-compatible clock, evidence-supported inference rather than an official rule |
| Randomness | None in production evaluator/protocol construction; test seed 42 |

**Do not compare one model's primary all-user MAP with another model's nonempty-only diagnostic.** Results returned by `EvaluationProtocol.evaluate` include the protocol ID, configuration hash, truth hash, cohort-order hash, and candidate-ID hash so later comparisons can identify a mismatched protocol.

Machine-readable policy and source provenance are frozen in `data/step3/evaluation_config.json`, with its SHA-256 in `data/step3/evaluation_config.sha256`. The configuration includes the complete ordered cohort ID list, source roles, candidate restrictions/counts, the boundaries below, ground-truth rule, repeat/fallback rules, and file hashes.

### Temporal boundaries

| Role | Local half-open interval | UTC Unix-second interval |
|---|---|---|
| Hold-out training, Weeks 11–20 | `[2020-03-09 00:00, 2020-05-18 00:00)` | `[1583704800, 1589749200)` |
| Validation, Weeks 21–30 | `[2020-05-18 00:00, 2020-07-27 00:00)` | `[1589749200, 1595797200)` |
| Later final fitting only, Weeks 11–30 | `[2020-03-09 00:00, 2020-07-27 00:00)` | `[1583704800, 1595797200)` |
| Later test availability, Weeks 31–42 | `[2020-07-27 00:00, 2020-10-19 00:00)` | `[1595797200, 1603054800)` |

These match the verified Step 2 manifest. A recorded last-watch endpoint equal to a window's exclusive end is valid because it closes a half-open viewing interval; a first-watch timestamp at that end belongs to the next period. Step 3 does not parse user wall-clock timestamps again or reconstruct the join: it verifies the persisted UTC first/last timestamps and Step 2 provenance. Metadata-only candidate verification uses the established timezone interpretation and UTC elapsed broadcast durations.

## Inputs, schema inspection, and integrity

### Actual input schema

All three existing interaction files have the exact inspected header:

```text
user_id,tv_show_id,qualifying_airings,watched_seconds,first_watch_utc,last_watch_utc
```

Candidate files have `tv_show_id`; the cohort input has `user_id`. Headers, record widths, canonical numeric program IDs, exact string user IDs, sorted distinct user/program pairs, period membership, and minimum qualifying-duration consistency are validated. Unexpected records raise errors with source/row details instead of being silently dropped.

For example, the first inspected training record is user `10008352999832730058`, program `20088`, nine qualifying airings and 8,581 seconds. Its first/last UTC endpoints are `1585032071` and `1589436000`. The history input has the same schema but is forbidden as hold-out training input; it was inspected and checked as an existing artifact only.

### Input and output SHA-256 fingerprints

Paths below are relative to `SWEETTV_Final_Exam/`:

| File | SHA-256 |
|---|---|
| `data/step2/threshold_120s/train_interactions.csv.gz` | `3af4022f659c7c804ba791114963394b321baa813debe7aec6fac1b9947f1a93` |
| `data/step2/threshold_120s/validation_interactions.csv.gz` | `e85ff75d56a13f44784ee8fa765e754050e3fc2e7719fcf6b627df16e4f62912` |
| `data/step2/threshold_120s/history_interactions.csv.gz` | `7093e882ca1e1f9763340681b7f0009e32e9d2803be90513c68c6dcd8fb01834` |
| `data/step2/train_candidates.csv` | `93ec316b22aa0d2791f3393dc984ec6262c66f3b92e75ea3466d5867119b0470` |
| `data/step2/validation_candidates.csv` | `6f6ae055d0af1a70aa4e1b691cbc12cd054f2b80679643351e81dd8bcf22cd19` |
| `data/step2/test_candidates.csv` | `28fc2ed785ab57cb70e90609e0edd4e9c18b9268f1794212356b9883a2112bfa` |
| `data/step2/template_users.csv` | `5192399554230bf4feee964b0c55e29ed12117e50170272a052b49a3c59e9c61` |
| `data/step2/join_manifest.json` | `3758a9c343afc6a67f9bf1b17d61714f820fc22dc614b392e7def06aae379df3` |
| `data/step2/threshold_120s/interaction_manifest.json` | `8795ccb204122b4694fd18c7934fd0fec972d9487223e502476e7e4d75af3c87` |
| `data/step3/validation_truth.csv` | `166778fecc198e1acf73db2c3860580c3678275d5e5ccc180a17405b8181773d` |
| `data/step3/evaluation_cohort.csv` | `5e8309d71cdbea025cd87c33899fba4b2c37cde666512a9cb336dfa1e6a191c9` |
| `data/step3/evaluation_config.json` | `400b00eb6d660a663222e2e083ee59b2132469b2a06da28918b11323c88ec758` |

Ordered cohort identity hash (canonical JSON array of unchanged strings):

`bdbc14629f88df8079ad7bd5106e33e619cde024c0efa5e6f4b272068a22303e`.

Semantic protocol ID:

`40f6d1c93367106b22d3b66a76605afd12a1ec929f4fe7929bd137749880f239`.

The initial snapshot protects **63 existing files**, including all five original Kaggle CSVs, completed Step 2 products/code/reports/test evidence, earlier audit materials, and both original session exports. All 63 before/after byte sizes and SHA-256 values matched. These checks are in `results/step3_inputs_before.json` and `results/step3_verification.json`. The initial snapshot also compares originals to Step 1 fingerprints and Step 2 products to their own manifests.

No existing completed source/data/evidence file was edited in this stage. New files were added under `code/`, `tests/`, `data/step3/`, and `results/`; session checkpoints, if exported, use new filenames. Python-generated bytecode is excluded from the protected-source snapshot.

## Metric definition and API

Let `R_u` be the distinct relevant-program set and `p_u(r)` the predicted program at rank `r`. Define `hit_u(r)` to be one only when the ID is relevant and has not appeared at an earlier predicted rank. Then:

```text
P_u@r  = sum(hit_u(j), j=1..r) / r
AP_u@5 = sum(P_u@r * hit_u(r), r=1..5) / min(5, |R_u|), if |R_u| > 0
AP_u@5 = 0, if |R_u| = 0
MAP@5  = mean(AP_u@5 over all 1,260 cohort users)
```

The formula follows the official FKN course convention established in the exam and presentation. The hidden Kaggle ground-truth construction and scorer edge cases have not been independently established. The explicit local policies below must not be presented as newly verified platform rules.

### `code/evaluation.py`

- `average_precision_at_5(relevant, predictions, strict=False, candidates=None)` is the mathematical primitive. It respects rank, uses distinct relevance, awards no repeated duplicate credit, considers only ranks 1–5, and returns zero for empty relevance.
- In its deliberately **non-strict mathematical mode**, short/empty predictions are supported, duplicates get no repeated credit, and malformed/nonpositive/ineligible tokens are nonhits that still consume their original rank. This mode is for edge-case diagnostics/tests, not production comparison.
- `evaluate_predictions(..., strict=True)` is the default cohort evaluator. Strict mode requires every cohort user, exactly five ordered predictions, five distinct IDs, and eligibility in the common candidate pool. Missing/extra users and malformed records raise `ValueError`. Empty-label users' predictions are still validated before assigning AP=0.
- `parse_prediction_records(records)` reads exact two-field CSV-style records (`user_id`, `tv_show_id` containing five single-space-separated IDs), detects duplicate user rows before a dictionary could hide them, and rejects malformed tokens/fields. Eligibility is then checked by the evaluator.
- User IDs must be unchanged ASCII decimal **strings**. Floats and integer user keys are rejected; leading zeros in user identity are retained. Canonical positive program strings or exact Python integers are accepted; floats, booleans, zero, negatives, scientific notation, whitespace-normalized tokens, and noncanonical leading-zero prediction strings are rejected.
- Deterministic sums use `math.fsum`. The result contains `primary_map_at_5`, separately named `diagnostic_map_at_5_nonempty`, `per_user_ap_at_5`, primary/nonempty/empty population counts, and identity hashes. If an entirely empty-label synthetic population is evaluated, its primary score is zero and diagnostic is `None` rather than a division by zero.

### `code/validation.py`

- `EvaluationProtocol()` loads and verifies the frozen configuration, candidates, and ground truth. Its `.evaluate(predictions)` always uses strict production mode and the fixed cohort, with protocol/configuration/truth hashes attached to the result.
- `training_input(config)` returns an immutable training-only data handle. It allows only the registered Weeks 11–20 source path, hash, role, and UTC interval. Its `.rows()` iterator validates the original six-column interaction schema and temporal/candidate invariants. It exposes no validation truth or validation-history features.
- `training_input(config, proposed_path=...)` explicitly rejects validation, full-history, and future-test source paths as hold-out training input.
- `load_candidates`, `candidates_from_airings`, `schedule_catalogs`, `load_truth`, and configuration/hash utilities support deterministic candidate construction, validation-only target creation, and reuse by all methods.

Per-user AP values are returned in cohort insertion order. The configuration's explicit cohort array and `evaluation_cohort.csv` are the authoritative user order; JSON object key sorting in verification files does not redefine the cohort.

## Ground truth and evaluation cohort

### Selected primary target: all qualifying programs, not a Top-5 target

The ground truth is built **only** from `data/step2/threshold_120s/validation_interactions.csv.gz`. Step 2 already:

1. Unions overlapping fragments within user/channel and clips time to the partition.
2. Accumulates elapsed seconds per user/broadcast.
3. Calls a broadcast qualifying at ≥120 seconds.
4. Aggregates repeat broadcasts into one user–program pair with a qualifying-airing count and credited seconds.

Step 3 treats each distinct qualifying validation program as one binary relevant item. Nine qualifying broadcasts of one program still produce one relevant program. No training frequency, validation popularity, title similarity, candidate popularity, or future viewing is used to select labels.

**No Top-5 target truncation is performed.** Consequently no “most-watched” selection rule or relevance tie-break is needed. Program IDs are serialized in ascending numeric-ID order for reproducibility only; this is not an ordering of their relevance. A future Top-5-only target would be a separately named sensitivity protocol and must not be mixed with this primary protocol.

### Saved representation

`data/step3/validation_truth.csv` has exactly 1,260 records plus its header:

```text
user_id,relevant_program_ids,relevant_count
```

`relevant_program_ids` is a space-separated field of all relevant IDs, not necessarily five. An empty-label user has an empty field and count `0`; no sentinel program ID is inserted. Original user strings and original template row order are preserved. A separate `data/step3/evaluation_cohort.csv` records `user_id,has_training_history,validation_relevant_count` for all users. The truth loader requires exact cohort/order, consistent counts, positive distinct eligible IDs, and deterministic serialization.

### Actual coverage

| Quantity | Verified value |
|---|---:|
| Primary evaluation cohort | **1,260 users** |
| Users with qualifying training history | **1,250** |
| Users without qualifying training history | **10** |
| Users with relevant validation labels | **1,246** |
| Users without relevant validation labels | **14** |
| Relevant user–program pairs in this cohort | **198,517** |
| Distinct programs in this cohort's truth | **3,140** |
| Minimum / maximum relevant programs per cohort user | **0 / 714** |
| All-user training source pairs / users / programs | **716,796 / 4,837 / 2,948** |
| All-user validation source pairs / users / programs | **569,104 / 4,804 / 3,200** |

The 3,140 truth programs are for the fixed template cohort; the 3,200 programs in the full validation interaction file also include other users. This expected population difference is not missing data. All source validation rows are validated before only the predetermined template users' labels are retained. No user is selected or excluded on the basis of model success or label availability.

## Candidates, repeat policy, and fallback interface

### Common availability pools

| Purpose | Count | Source |
|---|---:|---|
| Training catalog validation | 5,842 | Existing `data/step2/train_candidates.csv` |
| **Local validation predictions** | **6,384** | Existing `data/step2/validation_candidates.csv` |
| Later test predictions | 6,649 | Existing `data/step2/test_candidates.csv` |

Candidate files must have distinct positive IDs in ascending numeric order and match their recorded hashes. Step 3 rechecked **exact ID-set equality**, not merely counts, against broadcast schedules clipped by actual timestamps. The independent verifier uses a separate `datetime.strptime`/aware-UTC calculation. Availability includes legitimate adjacent-file spillover airings overlapping the target window; schedule filenames alone do not determine eligibility.

Candidate construction accepts program/time metadata, not user behavior. The test-period schedule supplies availability only. Validation consumption never selects or prunes candidate availability. All ground-truth IDs must be eligible; unexpected out-of-catalog labels cause a failure rather than being removed to improve a metric.

**Previously watched programs remain allowed.** No method-specific “seen-item” filter is permitted under this protocol. This differs from duplicate IDs within one five-item list, which are invalid.

### Common cold-start contract, not a fitted popularity model

The eventual fallback must rank eligible candidates by:

1. Descending number of **distinct training users** with qualifying interaction.
2. Descending total qualifying **training-airing count**.
3. Ascending numeric program ID.

Zero-support items follow supported items using the same deterministic tie rule. Statistics must come only from the registered training source and then be restricted to the target catalog. The exact ranked fallback for the real data is **not computed in Step 3**.

`evaluation.FallbackRanking` defines an immutable interface containing an already-built candidate ranking, training-file SHA-256, training interval, candidate-ID hash, and the fixed policy name. `complete_top5` validates those tags and the complete distinct candidate inventory, then fills an insufficient/empty method ranking without duplicate IDs. The helper filters well-formed method IDs to the common pool; it does not learn or calculate popularity statistics. Strict production evaluation separately rejects any invalid final list.

Synthetic fixtures verify identical fallback for cold users, allowed repeat consumption, deterministic completion, duplicate removal, candidate restriction, and rejection of validation-period or wrong-source provenance. The future Step 4 fitting implementation must still prove that its counts truly came from training; a declared provenance tag alone is not proof of a model's computation.

## Independent tests and leakage checks

### Mathematical verification

The independent reference uses exact `fractions.Fraction` arithmetic. It iterates each relevant item's first predicted position and recomputes the distinct relevant prefix there. This differs from production's forward running hit counter; no production AP helper is used in the reference.

| Hand-computed fixture | Exact expected value | Actual AP@5 |
|---|---|---:|
| A: relevant `[10,20,30]`, predicted `[10,99,20,88,30]` | `(1 + 2/3 + 3/5)/3 = 34/45` | **0.7555555555555555** |
| B: relevant `[1,2]`, predicted `[3,1,4,5,2]` | `(1/2 + 2/5)/2 = 9/20` | **0.45** |
| All five correct, even in a different relevant-item order | `1` | Passed |
| No predictions correct | `0` | Passed |
| Single relevant item at rank five | `1/5` | Passed |
| More than five relevant items, first five all hits | `1` using denominator 5 | Passed |
| Duplicate `[1,1,2,9,8]`, relevant `{1,2}` | `5/6` in lenient mode; strict rejection | Passed |
| Empty relevance | `0`; malformed production list still rejected | Passed |

The suite also covers shifted relevant ranks, short/empty predictions, items below rank five, invalid/zero/unknown IDs, floats/bools/scientific notation, exact large IDs, duplicate/missing/extra user records, missing truth users, and malformed ground-truth fields. **500 seeded random cases** agree with the independent rational reference within 1e-15. All 34 current test methods pass.

### Synthetic full-cohort API verification — not model performance

The full verifier creates in-memory five-ID lists by cyclic positions in the sorted candidate catalog using each cohort row index. There is no fitting, learned statistic, or ground-truth-based prediction construction. These are evaluator fixtures only and are not saved as a Kaggle prediction file.

- All 1,260 per-user AP values agree with the independent rational calculation.
- Synthetic primary MAP: **13/900 = 0.014444444444444444**.
- Synthetic nonempty diagnostic MAP: **13/890 = 0.014606741573033707**.
- All 14 empty-label AP values are zero, and repeated evaluation with reordered input-mapping insertion order gives identical results.

These two numbers are explicitly **synthetic verification outputs, not the performance of Popularity, PPR, rules, or any other recommendation method**. They are saved with that designation in `results/step3_synthetic_checks.json` along with per-user AP values and exact reference fractions. No actual recommendation-model score exists at this stage.

### Full-data and leakage checks

The independent verifier completed **110 checks, all passing**:

- All 63 protected original/completed files retain their exact contents.
- Original template IDs, complete cohort order, cohort manifest, candidate files, and configuration hashes agree.
- Every row of the training, validation, and full-history interaction files was independently checked for schema, distinct sorted pair identity, positive exact program IDs, and appropriate UTC boundaries. **Zero boundary/identity violations** were found.
- Independent validation-source sets exactly equal the saved primary truth for every cohort user.
- Independent metadata-only schedule sets exactly equal each saved training/validation/test candidate pool.
- The real training-only accessor was consumed across **716,796 rows** while a guard recorded its CSV reads. It opened only the registered training interaction file, never the validation or history interaction files.
- Explicit attempts to supply validation, history, or future-test interaction paths as hold-out training input were rejected. No test-view source is registered or accessed.
- Dropping a cold user or an empty-label user is rejected in strict full-cohort evaluation; malformed, duplicate, unknown, and zero prediction IDs are also rejected.
- All Python source remains in `code/`; no prediction CSV appears in `submission/`.

Observed training endpoints span `1583709360` through exclusive end `1589749200`; validation endpoints span `1589749200` through `1595797197`. These are inside the frozen windows. The history file was read for verification, not supplied to any fitting operation. The framework currently has **zero fitted models**.

These checks establish the behavior of the implemented API and data contract. They are not an OS sandbox: future model code must use the training-only interface and must itself be verified not to bypass the contract. Permitted future schedule availability is distinct from prohibited future viewing behavior. Agent-executed verification is not claimed as the student's personal hand verification required by the course.

### Actual failure, fix, and reruns

1. Initial unit run: **33 passed, zero failures/errors**.
2. First full `python3 code/validation.py build`: **failed before creating `data/step3/`** with `ValueError: Not a canonical positive program ID: '00052641'` during metadata-only candidate verification.
3. Root cause: original schedules contain a few leading-zero numeric spellings, whereas the completed Step 2 pipeline has always represented program IDs through exact integer conversion. The new strict processed/prediction-ID validator was incorrectly applied directly to raw schedule text.
4. Inspection found **zero** affected rows in the early schedule, **one** in the later historical schedule (`00052641 → 52641`), and **three** in the test schedule (`00078499 → 78499`, and two `00079865 → 79865`). Evidence and examples are in `results/step3_schedule_id_inspection.json`.
5. Fix: validate raw schedule IDs as ASCII decimal strings and convert through **exact Python integers, never floats**, solely to match Step 2 numeric identity. Strict processed/prediction tokens remain canonical. Original CSV strings and Step 2 files remain unchanged.
6. Added a regression test exercising a real-style leading-zero schedule row, zero sentinel, and an integer larger than 2^53. Second unit run: **34 passed, zero failures/errors**. Corrected full build then succeeded, followed by **110 passing verification checks**.

The production failure is preserved in `results/step3_failures.json`, and both unit attempts remain under `results/step3_test_runs/`. The failed build's duration was not measured separately and is not invented. No temporal-boundary violation was detected or silently discarded.

## Executed results, runtime, and memory

| Executed stage | Wall time | Peak process RSS |
|---|---:|---:|
| Final 34-test suite | **0.088 s** | **27.47 MiB** |
| Corrected full protocol/truth construction, including candidate recheck | **26.594 s** | **58.51 MiB** |
| Full independent verification | **66.420 s** | **84.64 MiB** |
| One synthetic 1,260-user evaluator call, included in verification | **6.761 s** | Included in verifier peak |

The successful build plus full verification took **93.014 seconds**. This excludes initial reading, the integrity snapshot, failed build, ID investigation, and development time; it is not a claimed total session duration. Timings use `time.monotonic()`; memory is Linux `resource.getrusage(...).ru_maxrss / 1024`, measured per process. No model timing or memory benchmark is claimed.

Python **3.14.4** was used. There are **no new third-party dependencies**: CSV/gzip/JSON, dataclasses, hashing, unittest, exact fractions, and zoneinfo are from the standard library; timezone rules come from the system IANA database. The existing machine has approximately 11 GiB RAM and four logical CPUs per Step 1. Streaming scans avoid loading millions of interactions into DataFrames; only cohort relevance sets, small candidate sets, and summary counters are retained.

### Files added in this stage

```text
code/evaluation.py
code/validation.py
code/test_step3.py
code/verify_step3.py
tests/run_step3_tests.sh                 # executable
data/step3/validation_truth.csv
data/step3/evaluation_cohort.csv
data/step3/evaluation_config.json
data/step3/evaluation_config.sha256
results/step3_inputs_before.json
results/step3_input_inspection.json
results/step3_schedule_id_inspection.json
results/step3_failures.json
results/step3_build_stats.json
results/step3_test_runs/run_001.json
results/step3_test_runs/run_002.json
results/step3_synthetic_checks.json
results/step3_verification.json
results/step3_evaluation_report.md
```

No completed Step 2 implementation, dataset, test evidence, report, original CSV, or original session export was modified.

## Reproduction commands

Run from `/home/dastan/Recommender Systems/FKN/SWEETTV_Final_Exam/` with `python3`:

```bash
# Initial schema/hash inspection (already executed; refuses to replace the baseline).
python3 code/verify_step3.py snapshot

# Tests; each invocation appends a numbered attempt rather than erasing earlier ones.
sh tests/run_step3_tests.sh

# Raw leading-zero-ID investigation, executed after the first build failed.
python3 code/verify_step3.py inspect-ids

# Build truth and fixed protocol from existing Step 2 artifacts; no preprocessing.
python3 code/validation.py build

# Full-data, numerical, strict-record, leakage and preservation checks.
python3 code/verify_step3.py verify
```

The actual sequence was snapshot → initial 33-test run → failed build → raw-ID inspection and corrected 34-test run → successful build → full verification. The test runner was made executable with `chmod +x tests/run_step3_tests.sh`.

On the already-completed workspace, use only the test and full-verification commands to recheck the frozen artifacts. The snapshot refuses replacement; the builder refuses to overwrite an existing output directory. A separate truth/config reconstruction can use:

```bash
python3 code/validation.py build --output-dir data/step3_replay
```

That replay command is documented, **not claimed as executed**. It reuses Step 2 artifacts and does not rebuild the raw join. It writes a new configuration whose artifact paths/hash differ because of the new directory; the policy/cohort/source-derived semantic protocol ID remains the comparable identifier. The builder's `results/step3_build_stats.json` and verifier's latest summary files are rewritten on a future invocation, so archive those run summaries first if retaining their original timings. Unit-attempt logs and original session exports remain separately preserved.

For later model integration, initialize `EvaluationProtocol`, obtain the training handle with `training_input(protocol.config)`, train only from that handle's `.rows()`, and pass a user-keyed mapping of exactly five eligible IDs per cohort user to `protocol.evaluate`. Compare `primary_map_at_5` only against primary values with matching protocol/configuration/truth/catalog/cohort identity. Do not pass the full-history file during local hold-out comparison.

## Uncertainties and Step 4 readiness

**Step 3 is ready for Step 4 under the frozen local protocol.** Mathematical tests, production-record validation, actual-file temporal checks, candidate provenance, cohort coverage, ground-truth consistency, and preservation checks all pass. The fallback interface is ready to receive a training-only popularity ranking; fitting that ranking is Step 4 work and has not started.

Remaining assumptions and obligations:

- Europe/Kyiv-compatible time is strongly supported by the observed DST behavior but remains an inference, not an official competition declaration.
- The primary relevance set is all qualifying validation programs under the existing 120-second threshold. Hidden Kaggle relevance construction has not been established, so no equivalence is claimed. The empty-label zero convention is an explicit local policy.
- Every future method must use this same cohort, truth, candidate pool, repeat policy, fallback policy, and primary/diagnostic distinction. Any alternative threshold or Top-5 target requires a separately named protocol and an independently reported comparison.
- A future fallback or model's self-reported provenance is insufficient on its own; verify the actual fitting code and data accesses in Step 4.
- No current failing check or unexplained data discrepancy remains. Course report-cap/account issues noted in Step 1 remain outside this evaluator milestone.
- The native OpenCode work history must continue to be preserved, and the student must honestly perform and disclose personal verification before the final exam submission.

**Stopped at Step 3: no popularity/PPR/rules/CBF/MF implementation, no model training, no model-performance claims, no final predictions, and no Kaggle submission.**
