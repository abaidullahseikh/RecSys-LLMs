# SWEET.TV Final Exam — preprocessing stage

Student: Abaidullah Seikh · Instructor: Seungmin Jin  
Deadline: 1 November 2026, 19:00 MSK

Start with [the Step 1 audit](results/step1_audit.md) and [the Step 2 preprocessing report](results/step2_preprocessing_report.md). The viewing-to-program pipeline and its independent checks have now run. Recommendation algorithms, fitted models, recommendation validation scores, and a competition submission have not been produced.

## Step 2: preprocessing and verification

All Python implementation, including tests, is in `code/`; fixtures and the shell test runner are in `tests/`. No third-party packages are required.

```bash
sh tests/run_step2_tests.sh
python3 code/data_pipeline.py build
python3 code/data_pipeline.py derive --threshold-seconds 120
python3 code/data_pipeline.py derive --threshold-seconds 300
python3 code/verify_step2.py
```

The build and derivation commands require **new output directories** and refuse to overwrite current artifacts. Existing outputs are under `data/step2/`; use `--output-dir data/step2_replay` for a fresh build, then `derive --join-dir data/step2_replay`. Threshold changes reuse cached unthresholded per-user/per-airing overlap seconds and do not rebuild the raw join. The verifier targets the canonical `data/step2/` outputs.

At the selected 120-second threshold there are 716,796 training, 569,104 validation, and 1,164,296 full-history user–program pairs. Full-history data includes validation-period behavior and must not be used to fit models evaluated on that hold-out. Preserve IDs as strings when reading all CSVs.

The completed verification records 26 passing tests and 100 passing full-data checks. An initial timestamp-format failure and its fix are retained in `results/step2_test_runs/` and summarized in `results/step2_verification.json`.

## Original inputs

The five original Kaggle CSVs remain in `../sweettv-tv-program-recommender/`. Their before/after SHA-256 manifests are under `results/`, with additional Step 2 integrity checks in the preprocessing manifests and verification file. The authoritative specification is `../exam_quiz_FKN.md`; `exam_quiz_FKN.md` here is a checked copy. Course slides remain at `../LLM4Rec. Midterm Quiz.pdf`.

## Reproduce the inspection

Run from `SWEETTV_Final_Exam/` with Python 3.14 and system timezone data:

```bash
python3 code/inspect_step1.py profile
python3 code/inspect_intervals_step1.py --clock naive
python3 code/inspect_clock_step1.py
python3 code/inspect_intervals_step1.py --clock kyiv
python3 code/inspect_step1.py reference
python3 code/inspect_step1.py integrity
```

These commands overwrite generated diagnostic JSON/cache files. Archive the existing integrity manifests before repeating `profile`, which establishes a new baseline. CSV inputs are opened read-only. The SQLite cache is derived, contains viewing records, and is ignored by Git. The inspection uses the standard library; no third-party packages were installed for Step 1.

The `Europe/Kyiv` interpretation is supported by the observed daylight-saving transition but is not an officially stated competition timezone. Both clock interpretations are retained for comparison.

## Evidence

Keep the native OpenCode session intact. A Step 1 checkpoint is not the final full-project session export. See `evidence/README.md`. No Git repository has been initialized or published during Step 1.
