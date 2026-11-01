# Experimental V3 Submission — PPR + Personal Frequency + CBF

## Status

`submission/final_submission_v3.csv` was generated and independently verified as a **separate experimental candidate**. It is **not** recommended over v2: its frozen secondary validation MAP@5 is **0.4136005**, below v2's **0.4404048**. V2 remains the recommended final submission. V3 has not been submitted to Kaggle and has no Kaggle score.

The selection was based only on saved validation results. `ppr_pf_cbf_0.40_0.40_0.20` was the highest-secondary-scoring new CBF-containing candidate in the 12-CBF / 12-hybrid Step 9 grid. Its primary local MAP@5 is 0.7988439, also below v2's 0.8075979. The aspirational Kaggle private score of 0.44801 was not reached or claimed; a local validation metric cannot establish that outcome.

## V3 file and integrity

- **File:** `submission/final_submission_v3.csv`
- **Rows:** 1,260 users, in exact original Kaggle template order
- **Format:** `user_id,tv_show_id`; each `tv_show_id` cell contains five space-separated IDs
- **Predictions:** five distinct, positive, schedule-eligible program IDs per user
- **Test candidate pool:** 6,636 valid nonzero program IDs from the Weeks 31–42 schedule
- **SHA-256:** `94a2e6c0fa39ebb6ffad62cd5abd560e538253902168ad8adad1aa8d6d6642ea`
- **Checksum:** `results/final_submission_v3_sha256.txt`
- **Verification:** `results/final_submission_v3_verification.json` — **12 checks passed, zero failed**

The independent verifier refit the PPR graph, CBF feature space, historical frequency profiles, and RRF rankings from the saved historical inputs and reproduced every v3 row. It also checked the raw schedule candidate set, original template order, CSV schema, ID canonicalization, eligibility, duplicates, training-period boundaries and preservation of v1/v2. Verification runtime was 236.76 seconds.

## Model and data

The model fuses complete candidate rankings using reciprocal-rank fusion with constant 60 and zero-based ranks:

```text
score(i) = 0.40 / (60 + PPR_rank(i))
         + 0.40 / (60 + personal_frequency_rank(i))
         + 0.20 / (60 + CBF_rank(i))
```

Deterministic ties use the shared full-history popularity rank, then numeric program ID. Recommending previously watched programs is allowed; duplicates within a list are not. Popularity is fitted on the permitted history and supplies completion/cold-start rankings for the component models.

- **Training behavior:** 1,164,296 verified user/program interaction rows from Weeks 11–30, using the established same-channel half-open overlap join, combined overlap fragments and 120-second airing threshold. Program ID zero is excluded. No Week 31–42 viewing behavior was used.
- **PPR:** cosine co-view graph; qualifying-airing frequency personalization; restart probability 0.35; 30 neighbors per node before symmetrization; history cap 40; minimum 3 common viewers; L1 tolerance `1e-8`; maximum 250 iterations.
- **Personal frequency:** qualifying-airing counts, descending, then numeric program ID.
- **CBF:** sparse TF-IDF-equivalent metadata vectors. IDF vocabulary was fitted only on Weeks 11–30 schedule metadata; the Weeks 31–42 schedule supplied candidate availability and candidate metadata. The validated channel-context feature weights were category 1.5, genre 1.25, title 0.8, director 0.3, actors 0.2, year 0.1, channel 1.0; user profiles use qualifying-airing counts.
- **Runtime:** 71.41 seconds end-to-end for the full-history refit and v3 generation. Environment: Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1, Linux x86-64; peak RSS 1,147.29 MiB.

Configuration and provenance are saved in `results/final_model_v3_manifest.json`; the refit PPR graph is `results/final_ppr_graph_v3.npz`.

## Preservation and submission status

V1 and v2 remain unchanged; their recorded SHA-256 values were checked during v3 generation and independent verification. The original OpenCode session export was not altered. No Kaggle submission was made automatically.

V3 is suitable for a **manual Kaggle upload as an experimental submission** if you want to test it, but the validation evidence says to keep **`submission/final_submission_v2.csv` as the recommended final model**. Its user-reported current Kaggle private MAP@5 of 0.37880 remains the only actual Kaggle score cited here.
