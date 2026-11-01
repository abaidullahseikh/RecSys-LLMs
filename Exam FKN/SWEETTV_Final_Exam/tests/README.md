# Step 2 preprocessing verification

Run from the project root:

```bash
sh tests/run_step2_tests.sh
python3 code/verify_step2.py
```

All Python code is in `../code/`, including `test_data_pipeline.py` and its independent brute-force oracle. This directory contains `interval_fixtures.json` with eleven deterministic hand-specified cases and `run_step2_tests.sh` as the test entry point.

The current 26-test suite includes 100 seeded random oracle comparisons, explicit channel/overlap/duplicate/fragment/missing-ID/cold-user cases, DST and boundary checks, exact string IDs, threshold reuse, and miniature end-to-end reproducibility. The full-data verifier independently recomputes all output aggregates using SQL and checks raw history recombination. It currently targets `data/step2/`.

Each suite run appends a numbered JSON record under `results/step2_test_runs/`; the initial real failure is preserved. `results/step2_verification.json` records all attempts plus the current 100 full-data checks. No model or recommendation-metric tests are part of Step 2.
