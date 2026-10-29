# HW6: PageRank and Personalized PageRank

This is the submission run guide. The original assignment specification is
[`readme.md`](readme.md). The application uses MovieLens 100K and TensorFlow.js
to compare popularity, global PageRank, user-seeded PPR, and the supplied HW6
MF reference under one retrieval protocol.

## Run the application

From this directory:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/index.html>. HTTP is required for dataset loading.
Internet access is required for the TensorFlow.js CDN dependencies. d3 is
vendored locally. Build the graph, then run the self-checks.

For the measured experiment, use the ordered driver below. The starter's
combined recall button performs its comparison before its alpha sweep, so it
does not itself enforce validation-first selection.

## Reproduce the complete measured experiment

The optional runner requires `google-chrome` on `PATH`, Python, and `websockets`:

```bash
python3 -m pip install websockets==15.0.1
python3 run_experiment.py start
```

The runner starts a separate headless Chrome profile and a loopback-only server
at <http://127.0.0.1:8766/index.html>. It uses remote-debugging port 9333 and
temporary browser files under `/tmp/opencode`, which the runner creates as needed.

On a fresh experiment page, initialize the driver:

```bash
python3 run_experiment.py eval '(async()=>{for(let i=0;i<120&&!HW6_STATE.loaded;i++)await new Promise(r=>setTimeout(r,500));await import("./experiment.js");return await HW6_EXPERIMENT.setup();})()' --out evidence/setup.json
```

Optionally measure backend feasibility on disposable synthetic factors:

```bash
python3 run_experiment.py eval 'HW6_EXPERIMENT.benchmarkMf()' --out evidence/mf-backend-benchmark.json
```

Then run the full sequence:

```bash
python3 -u run_experiment.py pipeline
python3 export_evidence.py
python3 audit_sources.py
```

The pipeline explicitly switches to WASM for validation; runs each of the four
prescribed alphas on the entire validation cohort; locks the winner; trains MF
on WebGL; freezes its best-validation checkpoint; evaluates all four methods on
the common test cohort; collects the fixed users' final and controlled lists;
and writes the exports. Progress is printed by stage and MF epoch. The 30-epoch
training can take several minutes; CPU training is substantially slower on the
measured machine. Keep the runner and browser alive until completion.

Every completed stage is saved in `evidence/`, including the trained factor
arrays in `mf-checkpoint.json`. These files survive a machine restart. An
interrupted training run must be restarted; a partial epoch is not a completed
checkpoint. To repeat the whole experiment, first use
`python3 run_experiment.py reset`, then initialize and run the pipeline again.
Completed evidence files are overwritten only by the corresponding new stage.

You can also import `experiment.js` from the browser console and invoke, in order:

```javascript
await import('./experiment.js');
await HW6_EXPERIMENT.setup();
for (const alpha of [0.50, 0.70, 0.85, 0.95]) {
  HW6_EXPERIMENT.validationAlpha(alpha);
  await tf.nextFrame();
}
HW6_EXPERIMENT.selectAlpha();
await HW6_EXPERIMENT.trainMf('webgl');
await HW6_EXPERIMENT.testComparison();
HW6_EXPERIMENT.collectUsers();
const bundle = HW6_EXPERIMENT.bundle();
```

## Dependency versions used

| Component | Measured version |
|---|---|
| Chrome | 153.0.8010.36, headless Linux x86_64 |
| TensorFlow.js | 4.22.0 |
| TensorFlow.js WASM backend | 4.22.0, SIMD enabled |
| d3 | 7.9.0, local vendor file |
| Python | 3.14.4 |
| websockets (optional runner) | 15.0.1 |

TF.js main bundle:
`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js`.
The experiment driver loads the WASM backend from
`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/`.
The page's default backend is WebGL; the driver deliberately uses WASM for
graph propagation and WebGL for MF training. WASM lacks the
`UnsortedSegmentSum` kernel needed by the supplied gather gradient.

## Implementation and protocol

- Graph nodes are the 943 users and 1,611 movies observed in training: 2,554
  nodes, with explicit ascending-ID maps and no padding node.
- Positive edges come only from distinct training user/movie pairs rated at
  least 4. There are 45,602 edges, one zero-degree user (685), and 253 zero-degree
  movies.
- Each user is split chronologically 80/10/10, with ascending movie ID breaking
  timestamp ties: 79,619 training, 9,596 validation, 10,785 test records.
- Both walks compute
  `p_next = alpha * (Q^T p + dangling_mass * v) + (1 - alpha) * v`.
  Global restart is uniform; PPR restart is one-hot at the mapped user node.
  Initial `p = v`; tolerance is L1 `< 1e-6`, cap 500 iterations. A cap is not
  treated as convergence. The implementation conservatively requires the
  tolerance to be reached before the cap.
- CPU/WebGL propagation uses the rectangular off-diagonal blocks. The WASM
  student helper uses tensor `sparseSegmentSum` on preprocessed incoming-edge
  indices; propagation is not a JavaScript per-edge update. Explicit small
  transition fixtures use `tf.matMul(Q, p, true, false)`. All per-call tensors
  are disposed. A weak, graph-identity cache stores copied plain probability
  arrays for one backend/alpha/tolerance/cap tuple at a time.
- Validation excludes all training-rated movies; test additionally excludes
  all validation-rated movies, regardless of rating. Candidates are the same
  1,611 training movies. Scores sort descending, ties by ascending movie ID.
- Relevance is rating at least 4. Recall@10 is a macro mean over 846 validation
  users or 861 test users. All four methods share exactly the same test IDs.
- Validation has 4 relevant records outside the candidate pool, 97 users with
  no eligible relevant movie, and one without a positive training edge. Test
  has 13, 82, and one respectively. These user exclusions overlap: the excluded
  user unions are 97 and 82, not 98 and 83. Full IDs are in the JSON evidence.
- Alpha is selected only by validation PPR Recall@10; exact ties choose the
  smaller alpha. Global PageRank uses that same alpha, not a separately tuned
  value. MF uses its own validation-RMSE checkpoint objective.
- Users 1 and 2 are fixed by choosing the two smallest common-test-cohort IDs
  before final test metrics. They have 217/49 training ratings and 131/32
  positive edges. Controlled settings change only the seed (1 to 2 at 0.85),
  or only alpha (0.85 to 0.50 for user 1).

Two authorized scaffold corrections remain in `script.js`: the PPR adapter
passes the mapped seed index, and unknown raw user IDs map to `-1` rather than
aliasing a graph node. Dataset, loader, HTML, CSS, original assignment text,
vendor file, and MF training/scoring code retain their audited bytes.

## Measured validation results

All rows use the same 846 validation users, WASM backend, and training-only
exclusions. These values were reproduced after restarting the experiment.

| Alpha | Macro Recall@10 | Nonconverged users |
|---|---:|---:|
| **0.50 (selected)** | **0.1144831591** | **0** |
| 0.70 | 0.1047969508 | 0 |
| 0.85 | 0.0900207716 | 0 |
| 0.95 | 0.0737503039 | 27 |

At 0.95 the affected walks reached the 500-iteration cap; its row reports the
lists produced by those capped walks, not a claim that every walk converged.
No tolerance or iteration limit was relaxed. The selected alpha was applied
before any final test comparison.

## Final test results

All methods use exactly the same **861 users** and the same candidate/filtering
rules. PageRank and PPR use the selected alpha, 0.50.

| Method | Macro Recall@10 | Mean recommended training popularity |
|---|---:|---:|
| Popularity | 0.0760229134 | 271.9969 |
| Global PageRank | 0.0796932415 | 257.8203 |
| PPR | **0.1121885322** | 244.2067 |
| Supplied HW6 MF reference, epoch 1 | 0.0211374889 | 96.7609 |

The four-method comparison took 39,222 ms; including the independent audits,
47,605 ms. All selected-alpha PageRank/PPR runs converged. Every test list had
10 items, and no method needed zero-score items in these Top-10 lists. TF.js
tensor and data-buffer counts stayed at 10 before and after comparison: the
existing MF allocations were retained, with no additional propagation tensors.
The MF factor fingerprints were unchanged by test evaluation.

The supplied MF is a **degenerate reference in this run**: every candidate
prediction for every test-cohort user was the identical float32 value
`3.5806026458740234`. Its Top-10 lists therefore follow eligible ascending movie
IDs, not distinguishable factor scores. Its lower mean popularity is not
evidence of useful discovery or superior personalization. The MF continuity
limitation below is essential when interpreting this table.

### Fixed two-user evidence

User 1 has 217 training ratings / 131 positive edges, including *Jean de
Florette* (5), *Reservoir Dogs* (4), and *The Empire Strikes Back* (5). User 2 has
49 / 32, including *The English Patient*, *The Full Monty*, and *Rosewood* (all
4). Their leading positive genre counts are Drama/Comedy/Action = 61/39/36 for
user 1, and Drama/Comedy/Romance = 18/11/11 for user 2; genres are multi-label.
Full histories, timestamps, lower-rated examples, and title/genre lists are in
`evidence/users.json` and `recommendations.csv`.

| Method | User 1 list mean popularity | User 2 list mean popularity | Common movie IDs | Test hits, user 1 / user 2 |
|---|---:|---:|---|---|
| Popularity | 246.8 | 262.1 | 318 (1 item) | 2/19 / 0/2 |
| Global PageRank | 239.4 | 247.4 | 222 (1 item) | 3/19 / 0/2 |
| PPR, alpha 0.50 | 228.4 | 226.7 | 475 (1 item) | 3/19 / 0/2 |
| Supplied MF | 72.2 | 111.4 | 5, 6 (2 items) | 7/19 / 0/2 |

These fixed users are descriptive examples, not a favorable subset: MF
retrieved more of user 1's held-out relevant movies despite its poor cohort
average, and all four methods missed both of user 2's eligible relevant movies.
User-specific exclusions also make global/popularity lists differ; list overlap
alone is not a personalization-quality measure.

### Controlled PPR settings

| Setting | User | Alpha | Mean popularity | Common items with baseline | Iterations |
|---|---:|---:|---:|---:|---:|
| Baseline A | 1 | 0.85 | 245.5 | 10 | 91 |
| Change seed/user only | 2 | 0.85 | 254.9 | 0 | 89 |
| Change alpha only | 1 | 0.50 | 228.4 | 7 | 22 |

All three converged. The user/seed setting keeps alpha fixed and applies the
new user's exclusions. The alpha-only setting keeps user 1 and the exclusions
fixed: *Scream*, *Air Force One*, and *Titanic* leave the list; *Trainspotting*,
*Sense and Sensibility*, and *E.T. the Extra-Terrestrial* enter. This particular
lower-alpha list has lower mean popularity; that observation is not a general
novelty guarantee.

## MF continuity limitation

The measured MF is the **provided HW6 reference**, not a recovered Week 5
selected factor checkpoint. It uses 16 factors; seed 20261013; normal
initialization with standard deviation 0.05; Adam learning rate 0.01; batch
size 512; regularization 0.05; 30 shuffled epochs; dot product plus the training
global mean; and the earliest strictly best validation-RMSE epoch.

Week 5's actual implementation averages L2 norms of the gathered batch rows
(`Week 5/script.js:513`), uses unseeded `tf.randomNormal`, and visits ordered
batches. HW6 instead penalizes the sums of **all** factor entries at every batch
(`script.js:953`) and uses seeded shuffling. These objectives are materially
different despite their similar score formulas. No saved Week 5 selected factor
arrays were available. Thus the four-method comparison does not establish that
PPR beats the previous selected MF model. Week 5 files remain untouched.

The completed WebGL run took 659,973 ms (11.00 minutes). All 30 validation RMSEs
were exactly 1.1829944264078285 at the recorded precision, so the earliest-tie
rule selected **epoch 1**. Its restored validation RMSE was checked again. The
checkpoint's user/movie factor L2 norms were 0.00111852 and 0.00148788; both
factor arrays and their SHA-256 fingerprints are saved. Training used no test
records or test metrics.

## Evidence files and interpretation

- `verification.json`: the original checks plus full-cohort ranking/filtering
  audit summaries, nonconverged-user records, source hashes, and checkpoint hash
  verification.
- `results.csv`: four validation and four final test rows. `iterations` and
  `residual` are cohort maxima for graph methods, not a single representative
  user's pair. `recall_runtime_ms` is available for validation;
  `audit_runtime_ms` measures the independent audit; test
  `comparison_runtime_ms` is the shared total for the four-method comparison.
- `recommendations.csv`: 80 final-list rows plus 30 controlled-setting rows.
  The `setting` column distinguishes repeated user/method/alpha combinations.
  Each list includes training popularity and its mean.
- `evidence/bundle.json`: complete per-user recall/hit/relevance lists,
  convergence diagnostics, training history, model fingerprints, raw exporter
  strings, two-user histories, titles, genres, and overlaps.
- `evidence/original-*`: exact outputs of the starter's three serializers.
  Root CSVs add complete exclusion, setting, runtime and convergence fields.
- `evidence/export-validation.json`: CSV round-trip and source/metric checks.
- `evidence/source-integrity.json`: preserved-file hashes and Week 5 continuity.

The saved verification contains **12 original checks passed, 0 failed, 0
pending**, plus audits of 6,828 user/method/alpha lists. Graph diagnostics check
nonnegativity, probability mass, residuals and an independent extra update.
`evidence/backend-equivalence.json` also compares the dense WebGL diagnostic
with optimized WASM at the selected alpha: all 846 validation users have
identical ordered Top-10 IDs and Recall@10; the largest score difference among
shared recommended movies is 2.794e-9.

## Known limitations

The split is chronological within each user, not globally time-consistent;
MovieLens times are rating timestamps rather than viewing sessions. This is a
warm-user evaluation, not a cold-start result. The original UTF-8 loader produces
12 replacement characters from `u.item`; metadata is exported as actually
loaded. True sparsity is 93.6953306358% over 943 × 1,682; the starter's padded
display reports 93.71%.

PPR scores are walk probabilities, not predicted ratings, purchases, or viewing
completion. MF scores and walk scores have different scales. List differences
or lower popularity do not by themselves establish better recommendations.
Runtime measurements apply only to this machine/backend; browser warm-up,
memoization and factor migration affect timing. The supplied MF reference keeps
optimizer-related tensors after training, so post-training TF.js memory is not
expected to return to zero; graph-only runs retain no propagation tensors.

The report and the single deployment/online-experiment analysis belong to the
next reporting step. The course's AI-session export must be obtained through
the course-required mechanism; these scripts do not manufacture a session log.
