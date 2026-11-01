"""Refit the validation-selected PPR/personal-frequency RRF for Weeks 31-42."""

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import scipy

import data_pipeline
import evaluation
import ppr
import popularity
import validation

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
TEMPLATE=SOURCE/"submission.csv"
TEST_SCHEDULE=SOURCE/"export_arh_31-42-final.csv"
SUBMISSION=ROOT/"submission"/"final_submission_v2.csv"
MANIFEST=ROOT/"results"/"final_model_v2_manifest.json"
GRAPH_PATH=ROOT/"results"/"final_ppr_graph_v2.npz"
STEP8=ROOT/"results"/"step8"/"step8_experiment_results.json"


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def load_original_users():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==["user_id","tv_show_id"],"Original Kaggle template header mismatch")
        users=[]
        for row in reader:
            require(len(row)==2,"Malformed original Kaggle template row")
            users.append(evaluation.user_id(row[0]))
    require(len(users)==1260 and len(set(users))==1260,"Template cohort is not 1,260 unique string IDs")
    return tuple(users)


def candidate_pool_from_named_schedule():
    """Use all valid positive IDs in the named 31-42 file (no time-window clip)."""
    candidates=set()
    with TEST_SCHEDULE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==data_pipeline.SCHEDULE_COLUMNS,"Week 31-42 schedule schema mismatch")
        for line,row in enumerate(reader,2):
            require(len(row)==len(data_pipeline.SCHEDULE_COLUMNS),f"Malformed Week 31-42 row {line}")
            item=validation.schedule_program_id(row[5])
            start=data_pipeline.parse_timestamp(row[2],schedule=True)
            duration=int(row[3])
            if item!="0" and duration>0:candidates.add(item)
    pool=evaluation.candidate_pool(candidates)
    require(len(pool)>=5,"Fewer than five valid test schedule candidates")
    return pool


def load_full_history():
    info=read_json(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    require(digest(HISTORY)==info["artifacts"][HISTORY.name]["sha256"],"Weeks 11-30 history hash mismatch")
    bounds=validation.WINDOWS["history"]
    profiles={};viewers=Counter();airs=Counter();seconds=Counter();rows=0;previous=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==validation.INTERACTION_COLUMNS,"Historical interaction schema mismatch")
        for row in reader:
            user=evaluation.user_id(row["user_id"]);item=validation.program_id(row["tv_show_id"])
            count,watched,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            key=(user,int(item))
            require(previous is None or key>previous,"Historical interactions are duplicated/unsorted")
            previous=key
            require(count>0 and watched>=120*count,"Historical interaction violates 120-second threshold")
            require(bounds[0]<=first<last<=bounds[1],"Weeks 31-42 behavior found in training interactions")
            profiles.setdefault(user,{})[item]=(count,watched,last)
            viewers[item]+=1;airs[item]+=count;seconds[item]+=watched;rows+=1
    require(rows==info["periods"]["history"]["unique_user_program_pairs"],"Historical interaction row count differs from Step 2 manifest")
    nodes=tuple(sorted(viewers,key=int))
    data=ppr.TrainingProfiles(nodes,{item:i for i,item in enumerate(nodes)},profiles,dict(viewers),
        digest(HISTORY),bounds,rows,0.0)
    pop=popularity.PopularityModel((),dict(viewers),dict(airs),dict(seconds),digest(HISTORY),bounds,rows,0.0)
    return data,pop,info


def model_rankings(graph,data,pop,candidates,users,*,ppr_weight=.5,rrf_constant=60):
    fallback=pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,
        pop.training_sha256,pop.training_window_utc,pop.training_rows,pop.training_runtime_seconds).ranking("distinct_users")
    fallback_rank={item:i for i,item in enumerate(fallback)}
    node_index={item:i for i,item in enumerate(graph.nodes)}
    candidate_index={item:i for i,item in enumerate(candidates)}
    ppr_position=np.empty((len(users),len(candidates)),dtype=np.int32)
    personal_position=np.empty_like(ppr_position)
    ids=np.asarray([int(item) for item in candidates],dtype=np.int64)
    ppr_scored_slots=0
    for row,user in enumerate(users):
        profile=data.profiles.get(user,{})
        if profile:
            restart=ppr.personalization_vector(profile,data,variant="frequency")
            scores,info=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            ppr_ranked=[item for item in candidates if item in node_index and scores[node_index[item]]>0]
            ppr_ranked.sort(key=lambda item:(-scores[node_index[item]],fallback_rank.get(item,len(fallback)),int(item)))
            ppr_scored_slots+=min(5,len(ppr_ranked))
        else:
            ppr_ranked=[]
        seen=set(ppr_ranked);ppr_full=ppr_ranked+[item for item in fallback if item not in seen]
        ppr_indices=np.fromiter((candidate_index[item] for item in ppr_full),dtype=np.int32,count=len(candidates))
        ppr_position[row,ppr_indices]=np.arange(len(candidates),dtype=np.int32)

        personal_scores=data.profiles.get(user,{})
        personalized=sorted((item for item in personal_scores if item in candidate_index),
                            key=lambda item:(-personal_scores[item][0],int(item)))
        personal_seen=set(personalized);personal_full=personalized+[item for item in fallback if item not in personal_seen]
        personal_indices=np.fromiter((candidate_index[item] for item in personal_full),dtype=np.int32,count=len(candidates))
        personal_position[row,personal_indices]=np.arange(len(candidates),dtype=np.int32)
    fused=ppr_weight/(rrf_constant+ppr_position)+(1-ppr_weight)/(rrf_constant+personal_position)
    tie=np.asarray([fallback_rank[item] for item in candidates],dtype=np.int32)
    predictions={}
    for row,user in enumerate(users):
        order=np.lexsort((ids,tie,-fused[row]))
        chosen=tuple(candidates[i] for i in order[:5])
        require(len(chosen)==5 and len(set(chosen))==5,"RRF failed to produce five unique recommendations")
        predictions[user]=chosen
    diagnostics={"cold_start_users":sum(not data.profiles.get(u) for u in users),
        "users_with_personal_frequency_history":sum(bool(data.profiles.get(u)) for u in users),
        "fallback_policy":evaluation.FALLBACK_POLICY,"rrf_rank_constant":rrf_constant,
        "rrf_ppr_weight":ppr_weight,"rrf_personal_frequency_weight":1-ppr_weight,
        "ppr_ranked_slots_before_fusion":ppr_scored_slots}
    return predictions,fallback,diagnostics


def main():
    require(not SUBMISSION.exists() and not MANIFEST.exists(),
            "Refusing to overwrite an existing v2 submission or model manifest")
    started=time.monotonic();ppr_config=read_json(ROOT/"results"/"step5"/"ppr_model_config.json")
    step8=read_json(STEP8)
    require(step8["selected_config_id"]=="rrf_ppr_personal_frequency_ppr_weight_0.50",
            "Step 8 did not select the documented rank-fusion configuration")
    require(ppr_config["best_config_id"]=="cosine_pfrequency_a0.35_k30","Unexpected validated PPR parent configuration")
    require(read_json(ROOT/"results"/"step5_verification.json")["all_passed"] is True
            and read_json(ROOT/"results"/"step6_verification.json")["all_passed"] is True,
            "Prior model validation evidence did not pass")
    users=load_original_users()
    require(users==validation.EvaluationProtocol().cohort,"Kaggle template cohort/order differs from frozen local cohort")
    candidates=candidate_pool_from_named_schedule()
    data,pop,history_manifest=load_full_history()
    profiles_done=time.monotonic()
    graph=ppr.build_graph(data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    graph_done=time.monotonic()
    arrays=ppr.graph_npz_arrays(graph)
    if GRAPH_PATH.exists():
        with np.load(GRAPH_PATH,allow_pickle=False) as saved:
            require(set(saved.files)==set(arrays),"Existing interrupted v2 graph schema differs")
            require(all(np.array_equal(saved[name],arrays[name]) for name in arrays),"Existing interrupted v2 graph differs from refit graph")
    else:
        with GRAPH_PATH.open("wb") as stream:np.savez_compressed(stream,**arrays)
    predictions,fallback,diagnostics=model_rankings(graph,data,pop,candidates,users,ppr_weight=.5,rrf_constant=60)
    pred_done=time.monotonic()
    eligible=set(candidates)
    require(tuple(predictions)==users,"Prediction order differs from original template")
    require(all(len(predictions[u])==5 and len(set(predictions[u]))==5 and set(predictions[u]).issubset(eligible)
        and all(item.isascii() and item.isdecimal() and not item.startswith("0") for item in predictions[u]) for u in users),
        "Generated CSV values violate output contract")
    with SUBMISSION.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","tv_show_id"])
        for user in users:writer.writerow([user," ".join(predictions[user])])
    join_manifest=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    interaction_manifest_path=ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json"
    original=join_manifest["source_files_before"]
    require(digest(TEMPLATE)==original["submission.csv"]["sha256"],"Original template hash mismatch")
    require(digest(TEST_SCHEDULE)==original["export_arh_31-42-final.csv"]["sha256"],"Test schedule hash mismatch")
    manifest={"version":2,"model":"Weighted reciprocal-rank fusion of Personalized PageRank and personal qualifying-airing frequency",
        "selected_validation_config_id":step8["selected_config_id"],"selection_rule":step8["selection_rule"],
        "selected_local_primary_validation_map_at_5":step8["selected_primary_map_at_5"],
        "selected_local_secondary_top5_validation_map_at_5":step8["selected_secondary_map_at_5"],
        "original_ppr_best_local_primary_map_at_5":ppr_config["best_metrics"]["primary_map_at_5"],
        "validation_period":"Weeks 21-30; local hold-out proxy only; not a Kaggle score",
        "training_period":"Weeks 11-30 inclusive; UTC interval [1583704800,1595797200)",
        "training_period_utc":list(data.training_window_utc),"training_source_path":str(HISTORY.relative_to(ROOT)),
        "training_source_sha256":data.training_sha256,"training_interaction_rows":data.rows,
        "training_users":len(data.profiles),"training_program_nodes":len(data.nodes),
        "threshold_seconds_per_airing":120,
        "matching_procedure":"same-channel half-open viewing/broadcast overlap; combine disjoint Week 11-20 and Week 21-30 overlap seconds per user/broadcast before threshold; retain >=120 unioned seconds; discard tv_show_id=0",
        "ppr_hyperparameters":ppr_config["best_config"],
        "personal_frequency_variant":"qualifying_airings descending, numeric program ID ascending",
        "rank_fusion":{"method":"reciprocal-rank fusion","constant":60,"ppr_weight":.5,"personal_frequency_weight":.5,
            "rank_positions_zero_based":True,"tie_break":"full-history distinct-user popularity rank, then numeric program ID"},
        "fallback_policy":evaluation.FALLBACK_POLICY,"fallback_variant":"distinct_training_users",
        "repeat_previously_watched_allowed":True,"within_list_duplicates_allowed":False,
        "candidate_source":"all valid, positive-duration, nonzero program IDs in export_arh_31-42-final.csv; no hidden viewing data",
        "candidate_schedule_path":str(TEST_SCHEDULE),"candidate_schedule_sha256":digest(TEST_SCHEDULE),
        "candidate_count":len(candidates),"candidate_ids_sha256":evaluation.identity_hash(candidates),
        "test_schedule_used_only_for_candidates":True,"test_period_viewing_behavior_used":False,
        "template_path":str(TEMPLATE),"template_sha256":digest(TEMPLATE),"template_users":len(users),
        "template_user_order_sha256":evaluation.identity_hash(users),"predictions_per_user":5,
        "submission_path":str(SUBMISSION.relative_to(ROOT)),"submission_sha256":digest(SUBMISSION),
        "graph_path":str(GRAPH_PATH.relative_to(ROOT)),"graph_sha256":digest(GRAPH_PATH),"graph_bytes":GRAPH_PATH.stat().st_size,
        "graph_statistics":graph.statistics(),"diagnostics":diagnostics,
        "step8_experiment_results_sha256":digest(STEP8),
        "step8_comparison_sha256":digest(ROOT/"results"/"step8"/"validation_model_comparison.csv"),
        "step5_verification_passed":True,"step6_verification_passed":True,
        "step2_history_interaction_manifest_sha256":digest(interaction_manifest_path),
        "step2_join_manifest_sha256":digest(ROOT/"data"/"step2"/"join_manifest.json"),
        "step2_history_stats":history_manifest["periods"]["history"],
        "runtime_seconds":{"history_load_and_checks":profiles_done-started,"ppr_graph_fit":graph_done-profiles_done,
            "prediction_and_fusion":pred_done-graph_done,"total_including_prior_sweep_crosschecks":time.monotonic()-started},
        "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,
            "cpu_count":__import__("os").cpu_count(),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},
        "kaggle_private_map_at_5":None,"kaggle_submission_performed":False}
    write_json(MANIFEST,manifest)
    print(json.dumps({"submission":str(SUBMISSION),"sha256":manifest["submission_sha256"],"users":len(users),
        "candidate_count":len(candidates),"training_rows":data.rows,"selected_config":manifest["selected_validation_config_id"],
        "local_primary_MAP@5":manifest["selected_local_primary_validation_map_at_5"],
        "local_secondary_MAP@5":manifest["selected_local_secondary_top5_validation_map_at_5"],
        "end_to_end_seconds":manifest["runtime_seconds"]["total_including_prior_sweep_crosschecks"]},indent=2))


if __name__=="__main__":main()
