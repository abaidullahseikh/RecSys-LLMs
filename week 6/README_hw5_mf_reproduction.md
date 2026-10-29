# Step 4B — Corrected HW5 MF reproduction

The original HW5 selected epoch-19 checkpoint weights were unavailable. A new
model was trained using the corrected HW5 MF implementation and selected by
validation RMSE. This reproduction is not asserted to be numerically identical
to the original HW5 model.

This separate supplement does not replace `README.md`, the authoritative
`readme.md`, or any previous experiment. No instructor approval of the
substitution is claimed. It restores the practical comparison with the
corrected algorithm; it cannot recover the original stochastic realization.

## Algorithm and compatibility boundary

`hw5_mf_reproduction.js` creates the isolated `HW5_MF_REPRODUCTION` namespace.
It does not assign to the seven HW6 student functions, the supplied HW6 MF
reference, or any instructor test function.

The original corrected `Week 5/script.js` has SHA-256:

```text
1475668aec8c1b008b8b9c6c0b3302f16e9c7bca326e32ea5a0cd7a11e8d51e1
```

The runner verifies source parity for `initializeFactors`, `predictRating`,
`computeRMSEFor`, `selectBestCheckpoint`, `trainMF`, and their scalar/model
helpers. Whitespace aside, the training changes are only the `async` signature
and an awaited epoch observer after the original best-checkpoint snapshot.
The observer persists progress and lets the page paint; it changes neither
the optimizer updates nor their order.

| Setting | Reproduction |
|---|---|
| Factors | 16 |
| Factor indexing | Raw user/movie IDs; 944×16 and 1,683×16 matrices |
| Initialization | Original unseeded `tf.randomNormal(shape,0,0.05)` |
| Mean | Mean of the 79,619 observed training ratings |
| Optimizer | `tf.train.adam(0.01)` with original backend defaults |
| Measured WebGL Adam defaults | beta1 0.9; beta2 0.999; epsilon 0.0001 |
| Batch size | 64 |
| Order | Original split-array order; no shuffle |
| Maximum epochs | 20 |
| Regularization | `0.05 * mean_batch(||u||² + ||i||²)` on gathered rows |
| Checkpoint | Minimum validation RMSE; exact ties choose earliest epoch |
| Rating prediction/RMSE | `clamp(trainingMean + dot, 1, 5)` |
| Recommendation score | Raw `trainingMean + dot`, exactly as HW5's recommender |

The original HW5 `predictRating` clips ratings, while its `recommendTopK`
deliberately ranks **unclipped** mean-plus-dot scores. Both behaviors are
preserved. Recommendation CSV scores can therefore exceed five or be negative;
they are rating-scale ranking scores, not bounded predictions or probabilities.
Clipping them before ranking would change the original scorer and create ties.

The HW6 compatibility adapter changes only retrieval eligibility/order:
training-movie candidates; training exclusions for validation; training plus
validation exclusions for test; descending scores and ascending movie ID ties.
It does not copy HW5's catalog-wide retrieval metric into the HW6 comparison.

## Dataset and reused experiments

HW5/HW6 `u.data`, `u.item`, and `data.js` are byte-identical. The loader and
chronological per-user 80/10/10 split are unchanged. Missing ratings are never
training targets. The split contains 79,619 / 9,596 / 10,785 records, and the
candidate pool contains 1,611 training movies. Graph construction has 45,602
positive training edges.

The existing validation-selected PPR alpha remains **0.50**. User IDs **1 and 2**
remain fixed. Previous popularity/PageRank/PPR measurements may be reused only
after exact cohort-ID equality and full MF candidate scoreability have passed.
The new evaluation stops if the shared user set changes. Such a change would
require a separately authorized/reported common-cohort recalculation, not a
mixed-population comparison.

The existing PPR controlled demonstrations remain historical measured evidence:
user 1 at 0.85, user 2 at 0.85, and user 1 at 0.50. No graph experiment is rerun
by the reproduction pipeline, apart from the required small original self-checks.

## Reproducibility and model restoration

The original initialization has **no explicit seed**, which this reproduction
preserves. `initial_weights.json` archives all actually generated float32
values. A same-backend replay can start from those exact arrays instead of
drawing a new initialization. The independent short training replay verifies
that identical initial arrays and update order produce identical results in
the tested fixture. A second full training run and cross-device bitwise identity
are not claimed.

The selected weights, full history, epoch fingerprints, dataset/source hashes,
runtime versions, configuration, initialization hashes and artifact hashes are
saved under `evidence/hw5_mf_reproduction/`. Selection happens before the test
RMSE or test Recall@10 is computed. Validation Recall@10 is descriptive and is
measured after the validation-RMSE checkpoint is frozen; it is not a second
selection objective.

To restore the saved selected model without training, serve this folder over
HTTP, wait for the existing application to load, and use the browser console:

```javascript
await import('./hw5_mf_reproduction.js');
await tf.setBackend('webgl');
await tf.ready();
const weights = await (await fetch(
  'evidence/hw5_mf_reproduction/model_weights.json'
)).json();
HW5_MF_REPRODUCTION.restoreModel(weights);
HW5_MF_REPRODUCTION.predictRating(1, 100); // clipped rating prediction
HW5_MF_REPRODUCTION.scoreAll(1)[100];     // original raw ranking score
```

Row zero exists only to retain HW5's raw-ID indexing. It is never a graph node
or retrieval candidate. Restoring a model disposes the previous reproduction
variables, not variables owned by the original HW6 application.

## Runner and dependencies

The optional isolated runner uses Python 3, `websockets` 15.0.1, and
`google-chrome` on `PATH`. TensorFlow.js is pinned to 4.22.0. The native page
uses jsDelivr; during this run that host timed out, so the runner loaded the
same version from:

```text
https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js
```

The actual downloaded bundle hash, browser version and GPU renderer are
recorded in `model_metadata.json`. This mirror fallback runs from the separate
driver; `index.html` remains unchanged.

For a **fresh evidence destination**, the execution sequence is:

```bash
python3 -B run_hw5_mf_reproduction.py prepare
python3 -B run_hw5_mf_reproduction.py start
python3 -B -u run_hw5_mf_reproduction.py run
```

The completed destination is intentionally protected: `prepare` refuses an
existing baseline, and `run` refuses an existing initialization. Do not delete
the completed evidence to replay a model. Use a separate project copy/destination
and load the archived initialization for an exact-start replay.

The dedicated runner uses loopback port 8767, remote-debugging port 9334, and a
separate temporary Chrome profile under `/tmp/opencode/hw5-mf-reproduction`.
It saves each epoch separately and never replaces an existing file with
different bytes. To check an active job:

```bash
python3 -B run_hw5_mf_reproduction.py status
```

After browser completion, the following command validates the saved artifacts
and creates the separate exports, without training or changing original outputs:

```bash
python3 -B run_hw5_mf_reproduction.py finalize
```

## New files and evidence

- `results_hw5_mf_reproduction.csv`: reproduction validation/test retrieval rows.
- `results_four_method_corrected_mf.csv`: one shared-cohort test row per method,
  with validation recall where previously/newly available. Blank graph-method
  RMSE fields mean not applicable, not zero.
- `recommendations_hw5_mf_reproduction.csv`: the two fixed users' new MF lists.
- `evidence/hw5_mf_reproduction/recommendations_four_method_corrected_mf.csv`:
  existing three-method lists plus the new MF lists for both fixed users.
- `model_weights.json`, `model_metadata.json`, `training_history.json`:
  complete selected model and training provenance.
- `initial_weights.json`, `initialization.json`: actual starting arrays and their
  distribution/memory checks.
- `restoration_verification.json`: factor equality, 40 clipped predictions,
  complete raw movie-score vectors for both users, and restored validation RMSE.
- `validation_evaluation.json`, `test_evaluation.json`: all evaluated user IDs,
  relevant IDs, hits, Top-10 items, score variation, popularity and exclusions.
- `two_users.json`: new MF lists, overlap, old lists, unchanged PPR controls,
  and the existing training histories.
- `provenance.json`, `three_model_provenance.json`: separate original HW5,
  supplied HW6 reference, and new reproduction records.
- `verification.json`, `artifact_verification.json`, `preservation_verification.json`:
  browser checks, independent saved-evidence checks, and protected-file hashes.
- `manifest.json`: hashes of completed evidence and implementation/export files.

### Preserved unsuccessful attempts

Startup and verification failures are retained in `failure_*.json` and the
separate pipeline logs. No real training began in these attempts:

1. jsDelivr did not load, leaving the dataset uninitialized.
2. The first mirror fallback attempted the wrong bootstrap entry point; the
   driver was corrected to call the existing `hw6Bootstrap` function.
3. The independent Adam reference assumed epsilon 1e-7. TF.js WebGL actually
   supplied epsilon **0.0001**. The reference now reads the actual optimizer
   configuration; neither training code nor the test tolerance was changed.
   The corrected reference's maximum update error was about **1.89e-7**, and
   the fixture replay and untouched-row checks passed.

`dataset_audit.json` belongs to the unsuccessful pretraining attempt.
`dataset_audit_verified.json` records the successful preflight and final
implementation hashes. Earlier evidence has not been overwritten.

## Scope and limitations

The original HW5 metrics are historical and are not expected targets for this
new unseeded run. The supplied HW6 MF reference remains separately available in
the original evidence. The original HW5 source, all seven HW6 student functions,
the reference MF, datasets, original README files, original exports and both
original AI-session exports are protected by a before/after hash inventory.

This remains warm-user, per-user-chronological MovieLens evaluation. It does
not measure cold start, session intent, viewing completion, retention or
revenue. The existing metadata-decoding limitation and graph nonconvergence
disclosures remain in the original README/evidence. No instructor acceptance of
the missing-original-checkpoint substitution is asserted. Final report writing,
citations and course submission confirmations remain a separate step.
