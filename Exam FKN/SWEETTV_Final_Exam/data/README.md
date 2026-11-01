# Data location

Original Kaggle files are retained unchanged at `../../sweettv-tv-program-recommender/` relative to this directory. See `../results/original_sha256_before.json` for exact absolute paths, byte sizes, and hashes.

Step 2 derived data is now in `step2/`, separately from all originals. `step2/join_manifest.json` records source/output hashes, time boundaries, and overlap coverage. Unthresholded per-user/per-broadcast overlaps support alternative thresholds without rebuilding the raw join.

`step2/threshold_120s/` is the selected interaction dataset; `step2/threshold_300s/` is an actually executed alternative. `history_*` artifacts include Weeks 21–30 and must not be used to train a model evaluated on that validation window. `test_candidates.csv` is an availability catalog, not user behavior or predictions.

Keep `user_id` as text when reading CSVs. Original template order and cold users are retained in the coverage files. See `../results/step2_preprocessing_report.md` for schemas and reproduction instructions. Do not overwrite or edit the five originals. Raw/derived data and inspection SQLite caches are excluded from Git.
