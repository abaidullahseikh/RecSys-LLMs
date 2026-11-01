# Step 6 — Association-Rule Recommendations

## Summary

Step 6 adds association-rule recommendations using training-period viewing only. It preserves the Step 3 evaluation protocol and compares 11 rule variants with the saved Step 4 and Step 5 baselines. The selected configuration is **14-day time-window baskets, confidence × log-lift scoring, and log-frequency antecedent weights**.

The selected association-rule model reaches **MAP@5 = 0.771331** on the 1,260-user primary cohort. This is slightly above the Step 4 distinct-viewer popularity baseline (0.768418), but below Step 4 personal frequency by qualifying airings (0.787627) and Step 5 Personalized PageRank (0.815151). The secondary top-five truth check is a sensitivity comparison, not the primary model-selection metric.

## Evaluation results

| Method / variant | Primary MAP@5 | Nonempty-user diagnostic MAP@5 | Secondary top-five sensitivity MAP@5 |
|---|---:|---:|---:|
| Association rules — 14-day, confidence × log-lift | 0.771331 | 0.779997 | 0.313643 |
| Step 4 — popularity, distinct viewers | 0.768418 | 0.774551 | 0.306627 |
| Step 4 — personal frequency, qualifying airings | 0.787627 | 0.796477 | 0.434132 |
| Step 5 — Personalized PageRank, cosine / qualifying frequency, α=0.35, k=30 | 0.815151 | 0.824310 | 0.429489 |

All Step 6 candidates contain five unique eligible program IDs. The selected model has 99.21% rule coverage of cohort users; its 10 users without applicable rules use the shared popularity fallback. The model generated 114,613 directed rules from 1,563,107 distinct co-occurring item pairs.

## Selected configuration

- **Transactions:** For each user, distinct eligible programs watched within each 14-day window in the training period.
- **Rule filters:** minimum co-occurrence count 3; support ≥ 0.0005; confidence ≥ 0.01; lift ≥ 1.0; up to 50 neighbors per antecedent.
- **Scoring:** confidence multiplied by log-lift, aggregated with log-frequency antecedent weights.
- **Fallback:** distinct training users, then qualifying airings, then numeric program ID; repeat recommendations are not used to pad the list.
- **Training data:** 716,796 training interactions; 23,310 user-window baskets; 28,718,290 pair occurrences; no validation/test behavior used to fit rules.
- **Recorded model preparation time:** 17.88 seconds. Prediction generation took 21.34 seconds.

Across the tested association-rule variants, the best primary MAP@5 was 0.771331. Changing the lift threshold to 1.2 lowered it to 0.707262. The 7-day and history-based baselines scored 0.768743 and 0.765944 respectively.

## Verification and artifacts

The independent verification completed **147 checks with zero failures**, including transaction statistics reconstructed from training inputs, support/confidence/lift for a sample of 250 saved rules, all 11 prediction and AP files, primary and secondary metrics, cold-start fallback behavior, baseline references, input preservation, and artifact hashes. The Step 6 unit suite passed all 14 tests.

Key files:

- `results/step6/association_predictions.csv` — selected model’s predictions.
- `results/step6/association_best_config.json` — selected configuration and metrics.
- `results/step6/association_experiments.csv` — per-variant results and artifact references.
- `results/step6_association_comparison.csv` — association-rule, Step 4, and Step 5 comparison.
- `results/step6/association_user_examples.md` — recommendations for the fixed heavy/light example users.
- `results/step6_verification.json` — detailed independent verification results.
- `results/step6/association_artifacts.json` — hashes and sizes of Step 6 outputs.

No Kaggle submission was created.
