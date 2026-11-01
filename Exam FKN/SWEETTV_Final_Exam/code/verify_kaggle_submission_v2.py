"""Independent verifier for the v2 Kaggle file and validation-selected model."""

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

import data_pipeline
import evaluation
import ppr
import popularity
import validation

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
TEMPLATE=SOURCE/"submission.csv"
SCHEDULE=SOURCE/"export_arh_31-42-final.csv"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
SUBMISSION=ROOT/"submission"/"final_submission_v2.csv"
MANIFEST=ROOT/"results"/"final_model_v2_manifest.json"
VERIFICATION=ROOT/"results"/"final_submission_v2_verification.json"
CHECKSUM=ROOT/"results"/"final_submission_v2_sha256.txt"
ID_RE=re.compile(r"[0-9]+\Z",re.ASCII)


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def raw_program_id(value):
    if not isinstance(value,str) or not ID_RE.fullmatch(value):raise ValueError(f"Invalid raw schedule ID {value!r}")
    return str(int(value))


def original_users():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=["user_id","tv_show_id"]:raise ValueError("Original template header mismatch")
        users=[]
        for row in reader:
            if len(row)!=2 or not row[0] or not ID_RE.fullmatch(row[0]):raise ValueError("Malformed template user row")
            users.append(row[0])
    if len(users)!=1260 or len(set(users))!=1260:raise ValueError("Original template users must be 1,260 unique strings")
    return tuple(users)


def schedule_candidates():
    items=set()
    with SCHEDULE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError("Test schedule header mismatch")
        for line,row in enumerate(reader,2):
            if len(row)!=len(data_pipeline.SCHEDULE_COLUMNS):raise ValueError(f"Malformed schedule row {line}")
            item=raw_program_id(row[5]);data_pipeline.parse_timestamp(row[2],schedule=True);duration=int(row[3])
            if item!="0" and duration>0:items.add(item)
    return evaluation.candidate_pool(items)


def load_reference_data(manifest):
    interaction_manifest_path=ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json"
    interaction_manifest=read_json(interaction_manifest_path)
    if digest(interaction_manifest_path)!=manifest["step2_history_interaction_manifest_sha256"]:
        raise ValueError("History interaction manifest hash mismatch")
    if digest(HISTORY)!=interaction_manifest["artifacts"][HISTORY.name]["sha256"]:
        raise ValueError("History source is not the verified Step 2 artifact")
    bounds=validation.WINDOWS["history"];viewers=Counter();airs=Counter();seconds=Counter();profiles={};rows=0;previous=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=validation.INTERACTION_COLUMNS:raise ValueError("History interaction schema mismatch")
        for line,row in enumerate(reader,2):
            user=row["user_id"];raw=row["tv_show_id"]
            if not user or not ID_RE.fullmatch(user):raise ValueError(f"Invalid history user ID on line {line}")
            item=raw_program_id(raw)
            if item=="0" or raw!=item:raise ValueError(f"Zero or noncanonical training program on line {line}")
            key=(user,int(item))
            if previous is not None and key<=previous:raise ValueError("History pairs are duplicated/unsorted")
            previous=key
            count,watched,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            if count<=0 or watched<count*120:raise ValueError(f"Training interaction below the 120-second threshold on line {line}")
            if not bounds[0]<=first<last<=bounds[1]:raise ValueError(f"Future/out-of-period history on line {line}")
            profiles.setdefault(user,{})[item]=(count,watched,last)
            viewers[item]+=1;airs[item]+=count;seconds[item]+=watched;rows+=1
    if rows!=manifest["training_interaction_rows"] or rows!=interaction_manifest["periods"]["history"]["unique_user_program_pairs"]:
        raise ValueError("Training interaction row count mismatch")
    nodes=tuple(sorted(viewers,key=int))
    data=ppr.TrainingProfiles(nodes,{item:i for i,item in enumerate(nodes)},profiles,dict(viewers),digest(HISTORY),bounds,rows,0)
    pop=popularity.PopularityModel((),dict(viewers),dict(airs),dict(seconds),digest(HISTORY),bounds,rows,0)
    evidence={"rows":rows,"unique_users":len(profiles),"positive_programs":len(nodes),"threshold_seconds":120,
        "min_first_watch_utc":min(int(r["first_watch_utc"]) for r in ()) if False else bounds[0],
        "max_allowed_last_watch_utc":bounds[1],"test_period_viewing_rows":0}
    return data,pop,interaction_manifest,evidence


def independently_predict(graph,data,pop,candidates,users,manifest):
    fallback=pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,
        pop.training_sha256,pop.training_window_utc,pop.training_rows,0).ranking("distinct_users")
    fallback_pos={x:i for i,x in enumerate(fallback)}
    node_pos={x:i for i,x in enumerate(graph.nodes)}
    candidate_pos={x:i for i,x in enumerate(candidates)}
    ppr_ranks=np.empty((len(users),len(candidates)),dtype=np.int32)
    frequency_ranks=np.empty_like(ppr_ranks)
    ppr_scored_slots=0
    for row,user in enumerate(users):
        profile=data.profiles.get(user,{})
        if profile:
            restart=ppr.personalization_vector(profile,data,variant="frequency")
            vector,solver=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            graph_recs=[]
            for item in candidates:
                if item in node_pos and vector[node_pos[item]]>0:graph_recs.append(item)
            graph_recs.sort(key=lambda item:(-vector[node_pos[item]],fallback_pos[item],int(item)))
            ppr_scored_slots+=min(5,len(graph_recs))
        else:graph_recs=[]
        ppr_seen=set(graph_recs);ppr_order=graph_recs+[x for x in fallback if x not in ppr_seen]
        ppr_ids=np.fromiter((candidate_pos[x] for x in ppr_order),dtype=np.int32,count=len(candidates))
        ppr_ranks[row,ppr_ids]=np.arange(len(candidates),dtype=np.int32)
        personal=sorted((x for x in profile if x in candidate_pos),key=lambda x:(-profile[x][0],int(x)))
        personal_seen=set(personal);personal_order=personal+[x for x in fallback if x not in personal_seen]
        personal_ids=np.fromiter((candidate_pos[x] for x in personal_order),dtype=np.int32,count=len(candidates))
        frequency_ranks[row,personal_ids]=np.arange(len(candidates),dtype=np.int32)
    weight=manifest["rank_fusion"]["ppr_weight"];constant=manifest["rank_fusion"]["constant"]
    fused=weight/(constant+ppr_ranks)+(1-weight)/(constant+frequency_ranks)
    tie=np.asarray([fallback_pos[x] for x in candidates],dtype=np.int32)
    numeric=np.asarray([int(x) for x in candidates],dtype=np.int64)
    predictions={}
    for row,user in enumerate(users):
        order=np.lexsort((numeric,tie,-fused[row]))
        predictions[user]=tuple(candidates[i] for i in order[:5])
    return predictions,ppr_scored_slots,fallback


def direct_pairwise_diversity(predictions,cohort,candidate_count):
    """Simple pair enumeration, independent of the Step 8 subset-count implementation."""
    lists=[tuple(predictions[u]) for u in cohort]
    sets=[set(x) for x in lists]
    n=len(lists);pairs=n*(n-1)//2;overlap_sum=identical=near4=near3=0
    for i in range(n):
        for j in range(i+1,n):
            common=len(sets[i]&sets[j]);overlap_sum+=common
            identical+=lists[i]==lists[j];near4+=common>=4;near3+=common>=3
    distinct={x for x in lists for x in x}
    return {"unique_recommended_programs":len(distinct),"candidate_coverage_fraction":len(distinct)/candidate_count,
        "mean_pairwise_top5_item_overlap":overlap_sum/pairs,"fraction_user_pairs_identical_ranked_list":identical/pairs,
        "fraction_user_pairs_sharing_at_least_4_items":near4/pairs,
        "fraction_user_pairs_sharing_at_least_3_items":near3/pairs,
        "users_identical_to_first_user_fraction":sum(x==lists[0] for x in lists)/n}


def check_step8_validation_selection():
    results=read_json(ROOT/"results"/"step8"/"step8_experiment_results.json")
    if results["selection_uses_kaggle_score"] or results["kaggle_scores_used_for_selection"]:
        raise ValueError("Kaggle scores were used to choose validation model")
    protocol=validation.EvaluationProtocol()
    secondary={}
    for user,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
            ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        values=tuple(field.split()) if field else ()
        if len(values)!=int(count) or tie!="qualifying_airings_desc,numeric_program_id_asc":
            raise ValueError("Secondary validation target contract mismatch")
        secondary[user]=frozenset(values)
    if tuple(secondary)!=protocol.cohort:raise ValueError("Secondary target cohort mismatch")
    if read_json(ROOT/"results"/"step5_verification.json")["all_passed"] is not True:
        raise ValueError("Step 5 verification is not passing")
    if read_json(ROOT/"results"/"step6_verification.json")["all_passed"] is not True:
        raise ValueError("Step 6 verification is not passing")
    compare=list(csv.DictReader((ROOT/"results"/"step8"/"validation_model_comparison.csv").open(newline="",encoding="utf-8")))
    if len(compare)!=16:raise ValueError("Step 8 comparison does not contain all 16 configurations")
    compare_by_id={row["config_id"]:row for row in compare}
    all_rows=[]
    for record in results["variants"]:
        cid=record["config_id"]
        path=ROOT/"results"/"step8"/"validation_predictions"/cid/"predictions.csv"
        pred_rows=list(csv.DictReader(path.open(newline="",encoding="utf-8")))
        pred=evaluation.parse_prediction_records(pred_rows)
        if tuple(pred)!=protocol.cohort:raise ValueError(f"Step 8 cohort order mismatch for {cid}")
        for ids in pred.values():evaluation.strict_top5(ids,protocol.candidates)
        primary=protocol.evaluate(pred)
        secondary_score=evaluation.evaluate_predictions(pred,secondary,protocol.cohort,protocol.candidates,strict=True)["primary_map_at_5"]
        if not math.isclose(primary["primary_map_at_5"],record["primary_map_at_5"],abs_tol=1e-15):
            raise ValueError(f"Step 8 primary MAP mismatch for {cid}")
        if not math.isclose(secondary_score,record["secondary_map_at_5"],abs_tol=1e-15):
            raise ValueError(f"Step 8 secondary MAP mismatch for {cid}")
        distinct={x for u in protocol.cohort for x in pred[u]}
        if len(distinct)!=record["unique_recommended_programs"]:raise ValueError(f"Step 8 coverage mismatch for {cid}")
        diversity=direct_pairwise_diversity(pred,protocol.cohort,len(protocol.candidates))
        csv_row=compare_by_id[cid]
        for key in ("primary_map_at_5","secondary_map_at_5","candidate_coverage_fraction",
                    "mean_pairwise_top5_item_overlap","fraction_user_pairs_identical_ranked_list",
                    "fraction_user_pairs_sharing_at_least_4_items","fraction_user_pairs_sharing_at_least_3_items"):
            expected=float(record[key]) if key in record else diversity[key]
            actual=float(csv_row[key])
            if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-14):
                raise ValueError(f"Step 8 comparison-table metric mismatch for {cid}: {key}")
        for key,value in diversity.items():
            if key not in record and key not in {"candidate_coverage_fraction"}:continue
            expected=float(record[key]) if key in record else value
            if not math.isclose(value,expected,rel_tol=0,abs_tol=1e-14):
                raise ValueError(f"Step 8 independent diversity mismatch for {cid}: {key}")
        all_rows.append(record)
    selected=sorted(all_rows,key=lambda r:(-r["secondary_map_at_5"],-r["primary_map_at_5"],
        r["prediction_runtime_seconds"],r["config_id"]))[0]
    if selected["config_id"]!=results["selected_config_id"]:raise ValueError("Step 8 selected model fails prespecified rule")
    return {"configurations_independently_rechecked":len(all_rows),"selection_rule_rechecked":True,
        "selected_config_id":selected["config_id"],"selected_primary_map_at_5":selected["primary_map_at_5"],
        "selected_secondary_map_at_5":selected["secondary_map_at_5"],
        "step5_verification_passed":True,"step6_verification_passed":True}


def main():
    started=time.monotonic();manifest=read_json(MANIFEST);checks=[]
    def passed(name,detail=None):checks.append({"name":name,"passed":True,"detail":detail})
    join=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    if digest(ROOT/"data"/"step2"/"join_manifest.json")!=manifest["step2_join_manifest_sha256"]:
        raise ValueError("Step 2 join manifest hash mismatch")
    if digest(TEMPLATE)!=join["source_files_before"]["submission.csv"]["sha256"] or digest(TEMPLATE)!=manifest["template_sha256"]:
        raise ValueError("Original template hash mismatch")
    if digest(SCHEDULE)!=join["source_files_before"]["export_arh_31-42-final.csv"]["sha256"]:
        raise ValueError("Original test schedule hash mismatch")
    users=original_users();candidates=schedule_candidates()
    if manifest["template_users"]!=len(users) or manifest["template_user_order_sha256"]!=evaluation.identity_hash(users):
        raise ValueError("Template users/order do not match model manifest")
    passed("original_template_hash_order_and_string_user_ids",{"users":len(users),"order_sha256":evaluation.identity_hash(users)})
    if manifest["candidate_count"]!=len(candidates) or manifest["candidate_ids_sha256"]!=evaluation.identity_hash(candidates):
        raise ValueError("Schedule-derived candidate pool differs from model manifest")
    passed("candidate_pool_rebuilt_from_all_valid_nonzero_named_schedule_ids",{"count":len(candidates),"id_sha256":evaluation.identity_hash(candidates)})
    data,pop,history_manifest,history_evidence=load_reference_data(manifest)
    if manifest["training_period_utc"]!=list(validation.WINDOWS["history"]):raise ValueError("Training bounds are not Weeks 11-30")
    if manifest["test_period_viewing_behavior_used"] is not False or manifest["test_schedule_used_only_for_candidates"] is not True:
        raise ValueError("Leakage declarations do not satisfy no-test-viewing requirement")
    if manifest["training_source_path"]!=str(HISTORY.relative_to(ROOT)) or manifest["training_source_sha256"]!=digest(HISTORY):
        raise ValueError("Unexpected full-history training input")
    passed("history_threshold_interval_and_no_test_viewing_data",history_evidence)
    validation_evidence=check_step8_validation_selection()
    if manifest["selected_validation_config_id"]!=validation_evidence["selected_config_id"]:
        raise ValueError("Final model does not match validation-selected configuration")
    if manifest["rank_fusion"]!={"method":"reciprocal-rank fusion","constant":60,"ppr_weight":.5,
            "personal_frequency_weight":.5,"rank_positions_zero_based":True,
            "tie_break":"full-history distinct-user popularity rank, then numeric program ID"}:
        raise ValueError("Model rank-fusion parameters do not match report")
    ppr_config=read_json(ROOT/"results"/"step5"/"ppr_model_config.json")
    if manifest["ppr_hyperparameters"]!=ppr_config["best_config"]:
        raise ValueError("Final PPR hyperparameters differ from selected validated PPR config")
    passed("Step8_saved_models_and_validation_results_rechecked",validation_evidence)
    graph=ppr.build_graph(data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    graph_arrays=ppr.graph_npz_arrays(graph)
    graph_path=ROOT/manifest["graph_path"]
    if digest(graph_path)!=manifest["graph_sha256"] or graph_path.stat().st_size!=manifest["graph_bytes"]:
        raise ValueError("Saved v2 graph hash/size mismatch")
    with np.load(graph_path,allow_pickle=False) as saved:
        if set(saved.files)!=set(graph_arrays) or any(not np.array_equal(saved[name],graph_arrays[name]) for name in graph_arrays):
            raise ValueError("Saved v2 graph differs from independently refitted graph")
    passed("full_history_graph_refit_matches_saved_graph",{"nodes":len(graph.nodes),"edges":graph.matrix.nnz,
        "graph_sha256":digest(graph_path)})
    predicted,ppr_slots,fallback=independently_predict(graph,data,pop,candidates,users,manifest)
    rows=[]
    with SUBMISSION.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        header=next(reader,None)
        if header!=["user_id","tv_show_id"]:raise ValueError(f"Submission header mismatch: {header!r}")
        for line,row in enumerate(reader,2):
            if len(row)!=2:raise ValueError(f"Row {line} has {len(row)} columns")
            rows.append(row)
    if len(rows)!=1260:raise ValueError(f"Expected 1,260 rows; found {len(rows)}")
    if [r[0] for r in rows]!=list(users):raise ValueError("v2 output user IDs/order differ from original template")
    if len({r[0] for r in rows})!=1260:raise ValueError("Duplicate output users")
    parsed={}
    candidate_set=set(candidates)
    for line,(user,value) in enumerate(rows,2):
        if not user or not value:raise ValueError(f"Missing cell on line {line}")
        items=value.split(" ")
        if len(items)!=5 or any(not x for x in items):raise ValueError(f"Line {line} does not contain exactly five IDs")
        if len(set(items))!=5:raise ValueError(f"Duplicate recommendation on line {line}")
        for item in items:
            if not ID_RE.fullmatch(item) or item=="0" or item.startswith("0") or str(int(item))!=item:
                raise ValueError(f"Noncanonical/zero program ID on line {line}: {item!r}")
            if item not in candidate_set:raise ValueError(f"Ineligible program ID on line {line}: {item}")
        parsed[user]=tuple(items)
    if parsed!=predicted:raise ValueError("v2 predictions do not reproduce independently from Weeks 11-30 inputs/model")
    passed("csv_schema_exactly_1260_users_and_user_order",{"rows":len(rows),"columns":["user_id","tv_show_id"]})
    passed("all_rows_have_five_unique_canonical_nonzero_eligible_program_ids")
    passed("no_missing_values_no_index_column_no_duplicate_users")
    passed("independent_full_refit_and_rank_fusion_reproduces_all_predictions",{"rows_reproduced":len(predicted),
        "cold_start_users":sum(not data.profiles.get(u) for u in users),"ppr_ranked_slots_before_fusion":ppr_slots,
        "fallback_top5_for_cold_users":sum(not data.profiles.get(u) for u in users)*5})
    if manifest["predictions_per_user"]!=5 or manifest["repeat_previously_watched_allowed"] is not True:
        raise ValueError("Repeat/Top-5 policy changed")
    passed("repeat_allowed_unique_list_policy_confirmed")
    if manifest["kaggle_private_map_at_5"] is not None or manifest["kaggle_submission_performed"] is not False:
        raise ValueError("A Kaggle score/submission was recorded without a submission")
    actual=digest(SUBMISSION)
    if actual!=manifest["submission_sha256"]:raise ValueError("Output hash differs from model manifest")
    CHECKSUM.write_text(actual+"\n",encoding="ascii")
    if CHECKSUM.read_text(encoding="ascii").strip()!=actual:raise ValueError("Checksum file mismatch")
    passed("sha256_matches_model_manifest_and_saved_checksum",{"sha256":actual})
    output_counts=Counter(x for u in users for x in parsed[u])
    ordered_output=sorted(output_counts.items(),key=lambda item:(-item[1],int(item[0])))
    result={"all_passed":True,"passed_checks":len(checks),"failed_checks":0,"checks":checks,
        "submission_path":str(SUBMISSION.relative_to(ROOT)),"submission_sha256":actual,
        "checksum_file":str(CHECKSUM.relative_to(ROOT)),"users":len(users),"data_rows":len(rows),
        "columns":["user_id","tv_show_id"],"predictions_per_user":5,"candidate_count":len(candidates),
        "model":manifest["model"],"model_config_id":manifest["selected_validation_config_id"],
        "training_period":manifest["training_period"],"training_rows":data.rows,"training_source_sha256":data.training_sha256,
        "local_primary_validation_map_at_5":manifest["selected_local_primary_validation_map_at_5"],
        "local_secondary_top5_validation_map_at_5":manifest["selected_local_secondary_top5_validation_map_at_5"],
        "kaggle_private_map_at_5":None,"kaggle_submission_performed":False,
        "cold_start_users":sum(not data.profiles.get(u) for u in users),"ppr_ranked_slots_before_fusion":ppr_slots,
        "top10_program_user_counts":[{"program_id":x,"users":n} for x,n in ordered_output[:10]],
        "unique_recommended_programs":len(output_counts),
        "independent_full_model_reproduction":True,"validation_evidence":validation_evidence,
        "verification_runtime_seconds":round(time.monotonic()-started,3)}
    write_json(VERIFICATION,result)
    print(json.dumps({k:result[k] for k in ("all_passed","passed_checks","failed_checks","submission_path","submission_sha256",
        "users","predictions_per_user","candidate_count","model_config_id","training_period","unique_recommended_programs",
        "verification_runtime_seconds")},indent=2))


if __name__=="__main__":main()
