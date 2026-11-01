"""Finalize Step 9 coverage fields from saved predictions without refitting."""

import csv
import json
from pathlib import Path

import run_cbf_step9 as step9
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step9_cbf"

BASELINE_PATHS={
    "popularity_distinct_users":ROOT/"results"/"step4"/"popularity_distinct_users_predictions.csv",
    "personal_frequency_qualifying_airings":ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv",
    "ppr_cosine_pfrequency_a0.35_k30":ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv",
    "association_rules_time_window_14d_lift_log":ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv",
    "existing_ppr_personal_frequency_rrf_050":ROOT/"results"/"step8"/"validation_predictions"/"rrf_ppr_personal_frequency_ppr_weight_0.50"/"predictions.csv",
}


def read_predictions(path,cohort):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        records=list(csv.DictReader(stream))
    predictions=validation.evaluation.parse_prediction_records(records) if hasattr(validation,"evaluation") else None
    if predictions is None:
        import evaluation
        predictions=evaluation.parse_prediction_records(records)
    if tuple(predictions)!=tuple(cohort):raise ValueError(f"Prediction cohort/order mismatch: {path}")
    return predictions


def main():
    result=json.loads((OUT/"step9_validation_results.json").read_text())
    protocol=validation.EvaluationProtocol();n_candidates=len(protocol.candidates)
    comparison_path=OUT/"model_comparison.csv"
    with comparison_path.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=len(result["experiments"]):raise ValueError("Comparison/JSON row count mismatch")
    by_id={row["config_id"]:row for row in result["experiments"]}
    for row in rows:
        cid=row["config_id"]
        path=BASELINE_PATHS[cid] if cid in BASELINE_PATHS else OUT/"predictions"/f"{cid}.csv"
        preds=read_predictions(path,protocol.cohort)
        unique={item for user in protocol.cohort for item in preds[user]}
        coverage=len(unique)/n_candidates
        row["candidate_coverage_fraction"]=format(coverage,".17g")
        by_id[cid]["candidate_coverage_fraction"]=coverage
        by_id[cid]["unique_recommended_programs"]=len(unique)
    with comparison_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    _,fraction_details=step9.load_fraction80_top5_truth(protocol)
    result["unverified_fraction80_sensitivity"]=fraction_details
    result["experiments"]=[by_id[r["config_id"]] for r in result["experiments"]]
    result["coverage_rebuilt_from_saved_predictions"]=True
    (OUT/"step9_validation_results.json").write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows_finalized":len(rows),"candidate_count":n_candidates,
        "cbf_best_coverage":by_id[result["cbf_best_individual_config_id"]]["candidate_coverage_fraction"],
        "selected_config_id":result["selected_candidate_config_id"],
        "selected_coverage":by_id[result["selected_candidate_config_id"]]["candidate_coverage_fraction"]},indent=2))


if __name__=="__main__":main()
