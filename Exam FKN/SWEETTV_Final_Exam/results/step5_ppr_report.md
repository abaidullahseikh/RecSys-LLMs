# Step 5 — Personalized PageRank on a program co-view graph

**Student:** Abaidullah Seikh · **Course:** LLM4Rec Final Exam (FKN)  
**Scope:** first approved method, genuine sparse program-graph Personalized PageRank. Association Rules remain for Step 6. No Kaggle submission or final prediction file was generated.

## Summary

The frozen Step 3 protocol was preserved exactly. A weighted, sparse, undirected program co-view graph was built only from the **Weeks 11–20 training interactions**, then scored for all **1,260** validation users against the same **6,384 validation-period candidates** and unchanged validation labels.

The staged search tested **16 PPR configurations**. Best primary configuration:

```text
Cosine co-view weights · frequency restart vector · restart probability 0.35
· top-30 directed neighbors per graph node before symmetric union
· user history capped at 40 programs for graph construction
· minimum common-viewer edge count 3
· L1 convergence tolerance 1e-8 · maximum 250 iterations
```

| Evaluation | Best PPR result |
|---|---:|
| **Primary MAP@5, all 1,260 users** | **0.815151** |
| Diagnostic MAP@5, 1,246 users with validation labels | 0.824310 |
| Hit Rate@5, all users | 0.975397 (1,229 users with at least one hit) |
| Top-1 hit rate | 0.919841 |
| Secondary Top-5 sensitivity MAP@5, same cohort | 0.429489 |
| Distinct programs recommended in best PPR Top-5s | 117 / 6,384 (1.8327% candidate coverage) |
| Training-warm / cold users | 1,250 / 10 |
| Cold-start users using the common fallback | 10; 50 total fallback slots |
| Users not converged at configured tolerance | 0 / 1,260 |

Under the **primary, all-qualifying-program** local validation target, PPR exceeds the Step 4 best popularity reference (0.768418) by **0.046733** and the best personal-frequency reference (0.787627) by **0.027524**. These are actual offline results under the frozen local proxy, not customer-satisfaction, causal, online-test, or Kaggle-private-score claims.

The separate **secondary Top-5 sensitivity** target gives a different selection: the primary-best PPR's sensitivity MAP is 0.429489, while `cosine_pfrequency_a0.35_k50` reaches 0.436048. The best personal qualifying-airing baseline is 0.434132 under the same sensitivity target. This alternative target does not replace the primary evaluation or claim to reproduce hidden Kaggle labels.

## Frozen protocol, inputs, and no-leakage design

Verified inputs were re-read from the actual Step 3 configuration and data manifests. The Step 4 baseline run, all saved baseline predictions, comparison table, and independent verification were checked before comparison. The Step 3 protocol ID is:

`40f6d1c93367106b22d3b66a76605afd12a1ec929f4fe7929bd137749880f239`

The evaluation-config SHA-256 is `400b00eb6d660a663222e2e083ee59b2132469b2a06da28918b11323c88ec758`. Training uses only `data/step2/threshold_120s/train_interactions.csv.gz` (716,796 distinct user–program interactions; SHA-256 `3af4022f659c7c804ba791114963394b321baa813debe7aec6fac1b9947f1a93`). The source contains 4,837 users with qualifying training history and 2,948 positive program IDs.

The frozen periods remain:

- Training, Weeks 11–20: `[1583704800, 1589749200)` UTC.
- Validation labels, Weeks 21–30: `[1589749200, 1595797200)` UTC.
- Test schedule IDs are not PPR model input. No test-period viewing behavior is present or accessed.

The underlying Step 2 rule remains the same-channel, half-open viewing/broadcast interval join with **at least 120 unioned seconds per user/broadcast**. `tv_show_id=0` remains unlabelled airtime excluded from items. Step 3 allows recommending previously watched programs and uses the exact same 1,260-user population, all-distinct-validation-program relevance set, empty-label AP=0, AP denominator `min(5, |R_u|)`, strict five-item list, and validation candidate pool. The `Europe/Kyiv` time interpretation remains an evidence-supported inference, not an official competition timezone.

Training PPR receives only the Step 3 `training_input` handle. Validation labels are used after each ranking is generated to evaluate/select configurations. Validation-period interactions are read after all PPR rankings have been frozen for the separately labeled Top-5 sensitivity analysis. Test schedule user behavior is never used. Schedule metadata is used only for legitimate candidate availability and, in the fixed user examples, title/category display.

## Graph construction

### Nodes and user-basket truncation

Each of the **2,948** distinct positive-ID programs appearing in training interactions is one node. For each training user, the graph builder forms distinct program history and keeps at most **40** programs for pair generation, sorted by:

1. Qualifying training-airing count, descending.
2. Watched training seconds, descending.
3. Numeric program ID, ascending.

This explicit cap controls pair expansion for long histories. It generates **3,524,267** within-user unordered pair occurrences and **592,769** distinct raw undirected co-view pairs. It truncates **532,492 user–program history entries (74.29% of training pairs)** from graph pair-generation only. Full training profiles are still retained for each user's PPR restart vector; they are not truncated to the graph seeds. The cap can reduce long-tail/niche co-view connections; this is a supported methodological limitation, and a higher-cap sensitivity run is not included in this staged search.

Pairs with fewer than **three distinct common training users** are removed. The common-viewer count counts a user once per program pair because Step 2's training data is one aggregated row per user/program. No validation/user future co-occurrence enters pair counts.

### Edge weight variants and sparse neighbor truncation

Both requested graph weightings were built from the same training-only pair counts:

- **G1 Common-viewer count:** `w(i,j) = common training users(i,j)`.
- **G2 Cosine-normalized:** `w(i,j) = common(i,j) / sqrt(training_viewers(i) * training_viewers(j))`, where each program's denominator is its number of distinct training users.

For each node, retain its strongest `k` neighbors, sorting weight descending and neighbor numeric ID ascending. The directed top-`k` selections are symmetrized by the **union**: if either endpoint retains the pair, include it with the shared symmetric raw weight. Row-normalize the resulting adjacency to construct transition matrix `P`. The union preserves undirected co-view links but can produce in-degree above `k` for highly selected hub programs; this is why the measured maximum degree can exceed the per-node outgoing-neighbor cap.

Because pair generation applies the 40-program user cap, an edge's common-viewer numerator counts users who retained **both** programs in that capped graph profile. G2's program denominators use distinct training viewers in the full training interactions. Thus G2 is training-support-normalized, capped co-view similarity; cap-based pair omission can make its numerator lower than the uncapped co-view count.

| Graph artifact | Nodes | Directed nonzero edges | Undirected edges | Avg. degree | Max. degree | Directed density | Dangling nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common-viewer count, top-30 | 2,948 | 103,932 | 51,966 | 35.255 | 2,139 | 1.1959% | 754 |
| Cosine, top-20 | 2,948 | 71,138 | 35,569 | 24.131 | 2,128 | 0.8186% | 754 |
| Cosine, top-30 | 2,948 | 99,960 | 49,980 | 33.908 | 2,136 | 1.1502% | 754 |
| Cosine, top-50 | 2,948 | 148,500 | 74,250 | 50.373 | 2,139 | 1.7087% | 754 |

Non-dangling transition-row sums were within floating-point tolerance of 1.0. Dangling rows have zero transitions; their rank mass is redistributed to that user's restart vector at each iteration. The stored graph artifacts are compressed SciPy CSR-derived arrays in `.npz`, not a dense item-by-item matrix. SciPy/NumPy are pinned for reproducibility.

## Personalized PageRank

For user restart vector `p`, the implementation solves:

```text
r_next = alpha * p + (1-alpha) * (P.T @ r + dangling_mass(r) * p)
```

Here `alpha` is explicitly the **restart probability** requested by Step 5, not the conventional continuation/damping probability. Restart vectors are normalized and computed solely from that user's qualifying Weeks 11–20 history:

- **Uniform:** equal mass per distinct program.
- **Frequency:** mass proportional to qualifying-airing count.
- **Duration:** mass proportional to watched seconds.
- **Recency:** mass `2^(-age_seconds / 2,592,000)`, using the last training watch and a fixed 30-day half-life.

Deterministic convergence: L1 residual ≤ `1e-8`, maximum 250 iterations; rank scores break ties by the training-only Step 4 distinct-user popularity order, then numeric ID. The measured best-configuration mean was **19.34 iterations**, maximum **23**; the maximum across all configurations was 45, with **zero non-converged users**. All predictions score only validation candidates with positive graph PPR mass; zero-score/unseen candidates and insufficient personal graph reachability are filled from the shared training-only popularity fallback. Lists contain exactly five unique positive eligible IDs, and repeat consumption remains allowed.

## Staged configurations and primary results

**Stage 1** fixed restart probability 0.20 and top-30 neighbors, comparing both graph weights and all four restart vectors. **Stage 2** selected the best Stage 1 graph/personalization pair (`cosine` + `frequency`) and tested restart probabilities 0.10/0.20/0.35 with neighborhood sizes 20/30/50. The already-run center configuration was reused; total unique configs: **16**. Config selection uses primary MAP@5 only, with deterministic config-ID tie-breaking.

| Stage | Graph | Personalization | Restart | Top-k | Primary MAP@5 | Diagnostic MAP@5 | Secondary Top-5 sensitivity MAP@5 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Common count | Uniform | 0.20 | 30 | 0.769106 | 0.777747 | 0.308061 |
| 1 | Common count | Frequency | 0.20 | 30 | 0.793127 | 0.802039 | 0.370939 |
| 1 | Common count | Duration | 0.20 | 30 | 0.791016 | 0.799904 | 0.365053 |
| 1 | Common count | Recency | 0.20 | 30 | 0.770410 | 0.779066 | 0.309775 |
| 1 | Cosine | Uniform | 0.20 | 30 | 0.769307 | 0.777951 | 0.308876 |
| 1 | Cosine | Frequency | 0.20 | 30 | 0.801222 | 0.810225 | 0.391767 |
| 1 | Cosine | Duration | 0.20 | 30 | 0.797169 | 0.806126 | 0.382370 |
| 1 | Cosine | Recency | 0.20 | 30 | 0.771066 | 0.779730 | 0.311907 |
| 2 | Cosine | Frequency | 0.10 | 20 | 0.781005 | 0.789781 | 0.337712 |
| 2 | Cosine | Frequency | 0.20 | 20 | 0.795093 | 0.804026 | 0.374765 |
| 2 | Cosine | Frequency | 0.35 | 20 | 0.811929 | 0.821051 | 0.415159 |
| 2 | Cosine | Frequency | 0.10 | 30 | 0.785616 | 0.794444 | 0.350185 |
| 2 | Cosine | Frequency | **0.35** | **30** | **0.815151** | **0.824310** | 0.429489 |
| 2 | Cosine | Frequency | 0.10 | 50 | 0.791556 | 0.800449 | 0.368638 |
| 2 | Cosine | Frequency | 0.20 | 50 | 0.810190 | 0.819294 | 0.412897 |
| 2 | Cosine | Frequency | 0.35 | 50 | 0.813021 | 0.822156 | **0.436048** |

All values are from `results/step5/ppr_experiments.csv`; the table rounds to six decimals. The best primary configuration is `cosine_pfrequency_a0.35_k30`. The different best configuration under secondary Top-5 sensitivity is `cosine_pfrequency_a0.35_k50`; sensitivity did not select or replace the primary model.

## Comparison against Step 4 baselines

| Method | Primary MAP@5, all users | Diagnostic MAP@5, labeled users | Secondary Top-5 sensitivity MAP@5 |
|---|---:|---:|---:|
| Best popularity — distinct viewers | 0.768418 | 0.777052 | 0.306627 |
| Best personal frequency — qualifying airings | 0.787627 | 0.796477 | 0.434132 |
| Best PPR primary — cosine/frequency, alpha .35, top-30 | **0.815151** | **0.824310** | 0.429489 |
| PPR sensitivity winner — cosine/frequency, alpha .35, top-50 | 0.813021 | 0.822156 | **0.436048** |

PPR's primary MAP improvement over the best Step 4 personal-frequency baseline is **0.027524** in this held-out local protocol. On the secondary Top-5 sensitivity target, the PPR primary winner is **0.004643 below** personal frequency, while the separately selected sensitivity PPR setting is **0.001915 above** it. The primary winner is not changed based on this alternate relevance definition. All five saved Step 4 baselines were recalculated under the same secondary target and are in `results/step5_ppr_comparison.csv`.

The primary target contains all distinct qualifying validation programs, with as many as 714 for a cohort user. For sensitivity only, each user's five validation programs with the greatest qualifying-airing count are treated as relevant; ties use ascending numeric ID. This construction is transparent and useful for diagnosis but is **not verified as the hidden Kaggle target**.

## Cold-start, diversity, and two-user examples

There are **10 cold-start users** without qualifying training history. All 10 receive the shared Step 4 global popularity list; total fallback slots are 50. Their primary MAP is **0.441**, equal to the same frozen fallback's measured cold-user score. No validation label enters the fallback. Warm-user primary MAP for the best PPR model is 0.818144.

For the best primary PPR model:

- 1,229/1,260 users receive at least one validation-relevant Top-5 item (Hit Rate@5 0.975397).
- 1,159/1,260 have a relevant item at rank 1 (Top-1 hit rate 0.919841).
- 117 distinct candidate programs appear in all cohort lists (1.8327% candidate coverage); high overlap is expected under a small Top-5 list and popularity-tied fallback, not evidence of satisfaction.

The exact Step 4 user IDs were retained:

| User | Training-history pattern | Validation relevant programs |
|---|---|---:|
| Heavy `13571199046623737884` | 757 distinct training programs; 2,358 qualifying airings | 333 |
| Light `8150816946177670084` | 1 training program: `90053098`, a film, 2,753 seconds | 15 |

| User | Method | Top-5 IDs | Validation hits |
|---|---|---|---:|
| Heavy | Best popularity (distinct viewers) | `[2400480, 240081, 2400508, 6500479, 2400467]` | 5 |
| Heavy | Best personal frequency (airings) | `[12001682, 2400480, 240081, 2400508, 20088]` | 4 |
| Heavy | Best PPR | `[2400480, 240081, 12001682, 2400508, 6500479]` | 5 |
| Light | Best popularity (distinct viewers) | `[2400480, 240081, 2400508, 6500479, 2400467]` | 1 |
| Light | Best personal frequency (airings) | `[90053098, 2400480, 240081, 2400508, 6500479]` | 1 |
| Light | Best PPR | `[90053098, 240081, 2400480, 6500479, 2400467]` | 1 |

For the heavy user, the frequency-personalized graph places SpongeBob (`12001682`, Children) among high-ranked programs connected through shared training audiences; all PPR Top-5 items happen to be in that user's validation labels. For the light user, PPR retains the one familiar film and adds globally supported/coview-linked news and documentary programs; only `240081` is in the observed validation labels. This is evidence of logged overlap only: unfamiliar items are not assumed useful discoveries and no viewing indicates enjoyment.

`results/step5/ppr_user_examples.md` contains both complete training histories, complete validation label sets, metadata for lists and relevant items, all three Top-5s and per-item validation-hit indicators. Metadata is schedule display information only.

## Runtime, memory, environment, and dependencies

| Stage | Runtime | Peak process RSS |
|---|---:|---:|
| Load/validate training profiles (716,796 pairs) | 3.884 s | Included below |
| Deterministic capped co-pair generation (3,524,267 occurrences) | 2.300 s | Included below |
| Individual graph materialization (four graph artifacts) | 0.51–0.62 s each | Included below |
| Popularity fallback fitting | 4.102 s | Included below |
| PPR + prediction/evaluation/output staged search (16 configurations) | about 489.4 s | **468.60 MiB total PPR run peak** |
| Independent verification | 208.240 s | **386.73 MiB** |
| Step 5 automated tests | 0.041 s | 82.00 MiB |

The selected configuration's PPR training time is **6.738 s** (profile load + shared pair generation + cosine top-30 graph build), with the separately reported shared popularity fallback fit taking **4.102 s**. Its 1,260-user prediction generation took **14.665 s**. Stage-two alpha .10 configurations take more iterations and about 18 seconds to generate predictions; the exact configuration runtimes are in the experiment CSV. The end-to-end run from profiles through all 16 searches, metrics, and artifacts took about **495.6 s**. Timings use `time.monotonic`; RSS uses Linux `/proc/self/statm` checkpoints and `resource.getrusage(...).ru_maxrss/1024`.

Environment: Python **3.14.4**, NumPy **2.5.4**, SciPy **1.18.1**, system IANA timezone data; four logical CPUs and approximately 11 GiB RAM per Step 1. `requirements-step5.txt` pins NumPy/SciPy. The project `.venv` is ignored by Git. No external source code, participant predictions, or training data were used.

## Verification, failure corrections, and reproducibility

### Tests and independent verification

- `code/test_step5.py`: **22 tests passed**, including manual G1/G2 calculations, sparse normalization, dangling redistribution, uniform/frequency/duration/recency vectors, deterministic capped training pairs, PPR power iteration against a separate dense pure-Python reference, convergence/non-convergence reporting, candidate filtering, exact-five fallback, ID-zero rejection, cold starts, exact user IDs, and Step 3 MAP@5 consistency. Final test run: `results/step5_test_runs/run_009.json` (**0.039 s**, peak RSS **58.27 MiB**).
- `code/verify_step5.py`: **196 checks passed, zero failed**. The independent verifier streams training rows, independently rebuilds capped co-view pair counts, reconstructs each full top-k edge set, verifies common-viewer/cosine weights and sparse transitions, re-evaluates all 16 PPR predictions/metrics, recomputes secondary labels, re-scores all five Step 4 baseline predictions, checks cold fallback, fixed users, session hashes, and preserved Step 1–4 outputs. Final verifier runtime: **213.251 s**, peak RSS: **386.99 MiB**. Verification is recorded in `results/step5_verification.json`.
- The dense toy reference independently implements the recurrence and dangling correction without calling SciPy's sparse PageRank update. Current production convergence had zero non-converged users in any searched configuration.

### Failures that were preserved and corrected

`results/step5_failures.json` records actual test, runner, verification, and artifact corrections. Highlights:

1. Early synthetic graph fixtures had a constructor arity error and incorrectly expected symmetric transition probabilities after row normalization; fixtures were corrected to test symmetric raw edge weight and separately normalized directed transitions.
2. A toy PageRank test initially used an iteration cap too low for its stringent tolerance; the implemented deterministic production limit/tolerance were set to 250 and `1e-8` for the alpha .10 grid, and the independent calculation compares convergence/residual rather than requiring identical iteration counts under different tolerances.
3. A PPR list already containing five IDs was accidentally extended by the fallback loop. The loop now stops before any additional append at length five; a regression test covers full PPR completion and cold fallback inventory.
4. Staged-search metadata, a Step 4 sensitivity lookup, graph-build timing, duplicate experiment-table column issues, and the final test/verifier contract were fixed. Earlier output snapshots are retained under `results/step5_failed_001` through `_005`; the final rerun carries the corrected measurements and schemas.
5. The independent verifier had a test-count expectation and diagnostic issues, all fixed. Its initial complete pass had 172 checks; the expanded final verifier added whole retained-edge/statistics, Step 1–4 preservation, and output-manifest checks and passed **196** checks. Prior failed records remain in numbered attempt files.

### Reproduction

From the project root, using Python 3.14:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements-step5.txt
sh tests/run_step5_tests.sh
.venv/bin/python3 code/run_step5.py
.venv/bin/python3 code/verify_step5.py
```

`run_step5.py` refuses to overwrite an existing `results/step5/` directory. The completed run and evidence already exist; a deliberate future rerun must use an explicit fresh output path after parameterizing the runner, not overwrite these artifacts. The run used no random seed. The frozen Step 3 interaction/truth/candidate and Step 4 baseline inputs are SHA-256 validated; every experiment stores prediction and per-user AP hashes. `ppr_model_config.json` records the selected parameters, graph hashes/statistics, protocol identity, dependency versions, runtime, and output hashes.

`results/step5_sha256.json` records hashes for the report, comparison, best prediction, model config, test run, verifier, artifacts, dependency pins, source files, and preserved Step 1–4 session exports. The new native Step 5 checkpoint is saved separately at `evidence/step5_opencode_session.json`; it is not reconstructed or substituted for the eventual full-project session export.

## Limitations and Step 6 readiness

- Only 10.8% of schedule rows contain a program ID in the first historical schedule; unlabelled airtime remains absent from item nodes and cannot be inferred as a title-level interaction.
- The 40-program per-user cap retains strongest observed training programs by airing count/duration/ID and discards 532,492 user-program history entries from graph pair construction. This reduces pair expansion but may remove weak/niche associations.
- Graph construction uses at least three common viewers; low-support edges are deliberately omitted. Symmetric union of directed top-k selections yields some very high-in-degree hubs (observed max degree 2,139); this is measured and can concentrate diffusion.
- A small candidate coverage fraction (117 distinct recommendations in the best model) reflects repeated Top-5 outputs and fallback, not all 6,384 candidates being surfaced. Discovery/diversity is not optimized or user-validated here.
- The primary ground truth is the Step 3 local threshold proxy, not verified hidden Kaggle labels. The secondary Top-5 target is a sensitivity definition only.
- Europe/Kyiv remains an evidence-supported inference rather than an official timezone statement.
- These are offline MAP results, not evidence of enjoyment, satisfaction, revenue or online lift. No Kaggle private score/screenshot has been generated.

**Step 5 is ready for Step 6.** Personalized PageRank is implemented and independently evaluated as the first approved method; the second approved method, Association Rules, has not yet been implemented. No CBF, Matrix Factorization, hybrid, Kaggle submission file or Kaggle submission was produced.
