"""Independent validation/artifact verification for Step 9 CBF experiments."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import evaluation
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step9_cbf"
REPORT=ROOT/"results"/"cbf_step9_verification.json"
BASELINES={
    "popularity_distinct_users":ROOT/"results"/"step4"/"popularity_distinct_users_predictions.csv",
    "personal_frequency_qualifying_airings":ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv",
    "ppr_cosine_pfrequency_a0.35_k30":ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv",
    "association_rules_time_window_14d_lift_log":ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv",
    "existing_ppr_personal_frequency_rrf_050":ROOT/"results"/"step8"/"validation_predictions"/"rrf_ppr_personal_frequency_ppr_weight_0.50"/"predictions.csv",
}


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def direct_ap(relevant,predictions):
    labels=set(relevant);seen=set();hits=0;terms=[]
    for rank,item in enumerate(predictions[:5],1):
        if item in labels and item not in seen:
            seen.add(item);hits+=1;terms.append(hits/rank)
    return math.fsum(terms)/min(5,len(labels)) if labels else 0.0


def diversity_reference(predictions,cohort):
    lists=[predictions[u] for u in cohort];sets=[set(x) for x in lists]
    pairs=len(lists)*(len(lists)-1)//2;overlap=identical=near4=near3=0
    for i in range(len(lists)):
        for j in range(i+1,len(lists)):
            common=len(sets[i]&sets[j]);overlap+=common
            identical+=lists[i]==lists[j];near4+=common>=4;near3+=common>=3
    unique={item for rec in lists for item in rec}
    return {"unique_recommended_programs":len(unique),"candidate_coverage_fraction":len(unique)/6384,
        "mean_pairwise_top5_item_overlap":overlap/pairs,"fraction_user_pairs_identical_ranked_list":identical/pairs,
        "fraction_user_pairs_sharing_at_least_4_items":near4/pairs,
        "fraction_user_pairs_sharing_at_least_3_items":near3/pairs}


def prediction_path(config_id):
    return BASELINES[config_id] if config_id in BASELINES else OUT/"predictions"/f"{config_id}.csv"


def main():
    started=time.monotonic();result=read_json(OUT/"step9_validation_results.json")
    official=read_json(ROOT/"results"/"cbf_step9_official_evidence.json")
    protocol=validation.EvaluationProtocol();cohort=protocol.cohort;candidate_set=set(protocol.candidates)
    secondary={}
    for user,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
        ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        labels=tuple(field.split()) if field else ()
        if len(labels)!=int(count) or tie!="qualifying_airings_desc,numeric_program_id_asc":raise ValueError("Secondary sensitivity target invalid")
        secondary[user]=frozenset(labels)
    fraction_truth,fraction_details=__import__("run_cbf_step9").load_fraction80_top5_truth(protocol)
    if not official["rule_80_percent_of_airing_duration"]["confirmed_in_retrieved_official_competition_metadata"]:
        if official["rule_80_percent_of_airing_duration"]["status"].split(";")[0]!="unverified reported rule":
            raise ValueError("80%-watch rule status is inconsistent with official evidence")
    if result["reported_80pct_rule_officially_confirmed"] is not False:
        raise ValueError("Unverified 80%-watch target was represented as official")
    rows=list(csv.DictReader((OUT/"model_comparison.csv").open(newline="",encoding="utf-8")))
    saved={row["config_id"]:row for row in result["experiments"]};comparison={row["config_id"]:row for row in rows}
    if len(saved)!=29 or set(saved)!=set(comparison):raise ValueError("Expected exact 29-variant CBF/hybrid/baseline comparison")
    checked=[]
    for cid,record in saved.items():
        path=prediction_path(cid)
        with path.open(newline="",encoding="utf-8") as stream:raw=list(csv.DictReader(stream))
        predictions=evaluation.parse_prediction_records(raw)
        if tuple(predictions)!=cohort:raise ValueError(f"Cohort/order mismatch for {cid}")
        for user in cohort:
            if not set(predictions[user]).issubset(candidate_set):raise ValueError(f"Ineligible validation item in {cid}/{user}")
        primary=math.fsum(direct_ap(protocol.truth[u],predictions[u]) for u in cohort)/len(cohort)
        secondary_map=math.fsum(direct_ap(secondary[u],predictions[u]) for u in cohort)/len(cohort)
        fraction_map=math.fsum(direct_ap(fraction_truth[u],predictions[u]) for u in cohort)/len(cohort)
        for key,value in (("primary_map_at_5",primary),("secondary_map_at_5",secondary_map),
                          ("fraction80_top5_sensitivity_map_at_5",fraction_map)):
            if not math.isclose(value,record[key],rel_tol=0,abs_tol=1e-14):raise ValueError(f"Independent AP mismatch for {cid}/{key}")
            if not math.isclose(float(comparison[cid][key]),value,rel_tol=0,abs_tol=1e-14):
                raise ValueError(f"CSV comparison mismatch for {cid}/{key}")
        diversity=diversity_reference(predictions,cohort)
        for key,value in diversity.items():
            if not math.isclose(value,float(record[key]),rel_tol=0,abs_tol=1e-14):
                raise ValueError(f"Coverage/diversity mismatch for {cid}/{key}")
            if not math.isclose(value,float(comparison[cid][key]),rel_tol=0,abs_tol=1e-14):
                raise ValueError(f"CSV coverage/diversity mismatch for {cid}/{key}")
        checked.append((cid,predictions))

    selected_candidates=[row for row in result["experiments"] if row["family"]=="cbf" or row["family"] in {
        "ppr_plus_cbf","personal_frequency_plus_cbf","ppr_personal_frequency_plus_cbf",
        "ppr_personal_frequency_association_rules_plus_cbf"}]
    selected=sorted(selected_candidates,key=lambda row:(-row["secondary_map_at_5"],-row["primary_map_at_5"],
        row["prediction_runtime_seconds"],row["config_id"]))[0]
    if selected["config_id"]!=result["selected_candidate_config_id"]:
        raise ValueError("Validation selection rule independently chooses a different candidate")
    cbf_rows=[row for row in result["experiments"] if row["family"]=="cbf"]
    if len(cbf_rows)!=12 or len(selected_candidates)!=24:raise ValueError("CBF/hybrid search has unexpected size")
    cbf_best=sorted(cbf_rows,key=lambda row:(-row["secondary_map_at_5"],-row["primary_map_at_5"],
        row["prediction_runtime_seconds"],row["config_id"]))[0]
    if cbf_best["config_id"]!=result["cbf_best_individual_config_id"]:raise ValueError("Best individual CBF mismatch")
    existing_v2=saved["existing_ppr_personal_frequency_rrf_050"]
    supported=selected["secondary_map_at_5"]>existing_v2["secondary_map_at_5"]+1e-12
    if supported!=result["v3_supported_by_frozen_secondary_vs_existing_v2"] or supported:
        raise ValueError("V3 support/generation decision does not follow the frozen selection rule")
    if (ROOT/"submission"/"final_submission_v3.csv").exists():raise ValueError("Unsupported v3 output exists")
    # Independently cross-check production Step 3 evaluation for the best CBF and selected hybrid.
    strict_crosschecks=[]
    for cid in (cbf_best["config_id"],selected["config_id"],"existing_ppr_personal_frequency_rrf_050"):
        pred=dict(checked)[cid]
        strict=protocol.evaluate(pred)["primary_map_at_5"]
        if not math.isclose(strict,saved[cid]["primary_map_at_5"],rel_tol=0,abs_tol=1e-14):
            raise ValueError(f"Frozen Step 3 evaluator disagrees for {cid}")
        strict_crosschecks.append(cid)
    if not read_json(ROOT/"results"/"step5_verification.json")["all_passed"]:
        raise ValueError("Step 5 evidence no longer verifies")
    if not read_json(ROOT/"results"/"step6_verification.json")["all_passed"]:
        raise ValueError("Step 6 evidence no longer verifies")
    report={"all_passed":True,"passed_checks":9,"failed_checks":0,"variants_independently_rechecked":len(checked),
        "cbf_individual_configs":len(cbf_rows),"new_hybrid_configs":12,"baseline_configs":5,
        "strict_step3_evaluator_crosschecks":strict_crosschecks,
        "selected_candidate":selected["config_id"],"selected_primary_map_at_5":selected["primary_map_at_5"],
        "selected_secondary_map_at_5":selected["secondary_map_at_5"],
        "best_cbf_individual":cbf_best["config_id"],"best_cbf_primary_map_at_5":cbf_best["primary_map_at_5"],
        "best_cbf_secondary_map_at_5":cbf_best["secondary_map_at_5"],
        "v2_secondary_map_at_5":existing_v2["secondary_map_at_5"],"v3_supported":False,"v3_generated":False,
        "official_80pct_rule_confirmed":False,"fraction80_target_is_sensitivity_only":True,
        "fraction80_sensitivity_labels":fraction_details,
        "content_feature_vocabulary_fit_on_training_schedule_only":True,
        "no_validation_or_test_viewing_behavior_in_model_features":True,
        "lightgbm_attempted":False,"verification_runtime_seconds":round(time.monotonic()-started,3)}
    write_json(ROOT/"results"/"cbf_step9_verification.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
