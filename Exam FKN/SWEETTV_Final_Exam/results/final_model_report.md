# Final Kaggle Prediction Preparation

## Submission

- **CSV:** `submission/final_submission.csv`
- **Rows/users:** 1,260, in the exact order of the original Kaggle template; each row contains five unique program IDs.
- **Columns:** `user_id,tv_show_id`; the five IDs are space-separated in `tv_show_id`.
- **Test candidate pool:** 6,632 canonical positive program IDs from valid, nonzero entries in `export_arh_31-42-final.csv` that overlap the Weeks 31–42 test period.
- **SHA-256:** `b8d38a1caecc55f72094af7668224f4f5c398a8fba5159361cc2852948106188`.

The file is prepared and verified for Kaggle upload; it has **not** been submitted. No Kaggle public or private score is known. The reported **0.815151 MAP@5** below is the project's local Weeks 21–30 validation result, not a Kaggle private score.

## Model selection

The best verified primary validation result among all completed method sweeps is Personalized PageRank configuration `cosine_pfrequency_a0.35_k30`, with local all-user MAP@5 **0.8151507936507936**. The 16-configuration PPR sweep and 196-check independent Step 5 verification are recorded in `results/step5/` and `results/step5_verification.json`. The best association-rule configuration scored 0.7713306878306878 across its 11-configuration sweep; its independent verification passed 147 checks. The cross-method comparison likewise selects PPR. The selection checks revalidated saved prediction hashes and scores against both experiment tables and reports.

The 1,260-user frozen local validation uses Weeks 11–20 for fitting and Weeks 21–30 for evaluation. It retains the project's primary all-qualifying-program relevance definition. The local validation score is used only to choose the model configuration; hidden Kaggle relevance and the resulting Kaggle score are unknown.

## Final refit and inference

The selected PPR strategy was refit using the verified Step 2 `threshold_120s/history_interactions.csv.gz` data, containing **1,164,296 user-program interaction rows** from Weeks 11–30. This historical input is built from the same-channel, half-open viewing/broadcast overlap join. Viewing fragments are accumulated for each user/broadcast across Weeks 11–20 and Weeks 21–30 before thresholding; a broadcast counts as watched at **120 or more unioned overlap seconds**. Program ID zero is unlabelled airtime and excluded. IDs are retained as strings and canonical positive schedule IDs use their integer decimal representation.

PPR is trained from these history-only user/program profiles:

- Cosine-normalized undirected program co-view graph.
- Per-user graph construction cap: 40 programs, ordered by qualifying airing count descending, watched seconds descending, then numeric program ID ascending.
- Minimum common viewers per co-view edge: 3.
- Keep up to 30 strongest neighbors per node before symmetric union.
- Frequency restart vector, weighted by each user's qualifying airing counts.
- Restart probability 0.35; L1 tolerance `1e-8`; maximum 250 iterations.
- Rank ties: common full-history popularity order, then numeric program ID.
- The shared popularity fallback ranks by distinct historical viewers, qualifying airings, then numeric program ID. It completes lists when PPR supplies too few eligible items and serves users with no historical profile.
- Recommending a previously watched program remains allowed. Duplicate IDs within one five-item list are not allowed.

The full history contained 4,891 users and 4,325 positive program nodes; two template users had no qualifying history and used the common popularity fallback. The test-period schedule supplied only candidate availability. No Weeks 31–42 user-viewing behavior was used.

## Runtime and environment

The final command `.venv/bin/python3 code/train_final_submission.py` completed in **34.18 seconds** end-to-end, including saved-experiment checks. It took 9.35 seconds to load and validate the history profiles, 3.30 seconds to fit the graph, and 21.13 seconds for fallback preparation and cohort prediction. It wrote a 1.59 MB final graph artifact.

Environment: Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, Linux x86-64, four logical CPUs. Detailed runtime, graph statistics, dependency versions, data hashes and configuration are in `results/final_model_manifest.json`.

## Verification evidence

The independent command `.venv/bin/python3 code/verify_kaggle_submission.py` passed **14 checks, zero failures**, in 9.83 seconds. It independently parsed the CSV; confirmed exact schema, row count, user IDs/order, unique users, five canonical unique nonzero IDs per row, candidate eligibility, missing/index-column checks and checksum; reconstructed candidates from the raw Week 31–42 schedule; streamed and checked all 1,164,296 historical interaction rows against time bounds and threshold; verified the model configuration against the saved PPR experiments; confirmed the Step 5/Step 6 verification results and that no evaluated alternative had higher primary validation MAP@5; and checked the saved PPR graph hash.

Evidence files:

- `results/final_submission_verification.json` — independent check results and checksum.
- `results/final_submission_sha256.txt` — SHA-256 checksum line.
- `results/final_model_manifest.json` — final model, input provenance, parameters, runtime, environment and graph hash.
- `results/final_ppr_graph.npz` — refit graph artifact.

## Reproduction

From the project root with the existing environment:

```bash
.venv/bin/python3 code/train_final_submission.py
.venv/bin/python3 code/verify_kaggle_submission.py
```

The training script refuses to overwrite the final CSV, graph, or model manifest. The independent verifier writes the checksum and final verification JSON after the checks pass.
