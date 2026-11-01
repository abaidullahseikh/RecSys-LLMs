"""Refit the validated PPR strategy on Weeks 11--30 and write Kaggle CSV."""

from collections import Counter
import csv
import gzip
import hashlib
import importlib.metadata
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
SUBMISSION=ROOT/"submission"/"final_submission.csv"
MANIFEST=ROOT/"results"/"final_model_manifest.json"
GRAPH_PATH=ROOT/"results"/"final_ppr_graph.npz"


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def json_read(path):return json.loads(Path(path).read_text())


def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def verify_saved_experiments():
    """Cross-check both completed method sweeps against saved reports/configs."""
    step5=ROOT/"results"/"step5"
    ppr_report=(ROOT/"results"/"step5_ppr_report.md").read_text()
    ppr_verification=json_read(ROOT/"results"/"step5_verification.json")
    ppr_config=json_read(step5/"ppr_model_config.json")
    with (step5/"ppr_experiments.csv").open(newline="",encoding="utf-8") as stream:
        ppr_rows=list(csv.DictReader(stream))
    require(ppr_verification["all_passed"] is True,"Step 5 verifier has not passed")
    require(len(ppr_rows)==16,"Saved PPR sweep is incomplete")
    for row in ppr_rows:
        pred=ROOT/"results"/"step5"/"experiments"/row["config_id"]/"predictions.csv"
        require(digest(pred)==row["prediction_sha256"],f"PPR prediction hash mismatch: {row['config_id']}")
    ppr_best=max(ppr_rows,key=lambda r:(float(r["primary_map_at_5"]),r["config_id"]))
    require(ppr_best["config_id"]==ppr_config["best_config_id"]==ppr_config["best_config"]["config_id"],
            "PPR model config is not the highest saved primary MAP@5 result")
    require(abs(float(ppr_best["primary_map_at_5"])-ppr_config["best_metrics"]["primary_map_at_5"])<1e-15,
            "PPR model config score differs from experiment row")
    require(ppr_best["config_id"] in ppr_report and "0.815151" in ppr_report,
            "PPR report does not agree with selected saved result")
    step6_rows=[]
    with (ROOT/"results"/"step6"/"association_experiments.csv").open(newline="",encoding="utf-8") as stream:
        step6_rows=list(csv.DictReader(stream))
    step6_report=(ROOT/"results"/"step6_association_report.md").read_text()
    step6_verification=json_read(ROOT/"results"/"step6_verification.json")
    require(step6_verification["all_passed"] is True and len(step6_rows)==11,
            "Step 6 sweep/verifier is incomplete")
    for row in step6_rows:
        pred=ROOT/row["prediction_file"]
        require(digest(pred)==row["prediction_sha256"],f"Association prediction hash mismatch: {row['config_id']}")
    association_best=max(float(r["primary_map_at_5"]) for r in step6_rows)
    require(abs(association_best-float(step6_verification["best_primary_map_at_5"]))<1e-15,
            "Step 6 verifier score differs from saved sweep")
    require("0.771331" in step6_report,"Step 6 report does not agree with saved experiment results")
    with (ROOT/"results"/"step6_association_comparison.csv").open(newline="",encoding="utf-8") as stream:
        comparisons=list(csv.DictReader(stream))
    ppr_cmp=[r for r in comparisons if r["family"]=="ppr"]
    require(len(ppr_cmp)==1 and abs(float(ppr_cmp[0]["primary_map_at_5"])-ppr_config["best_metrics"]["primary_map_at_5"])<1e-15,
            "PPR cross-method comparison differs from PPR experiment")
    overall=max(float(r["primary_map_at_5"]) for r in comparisons)
    require(abs(overall-float(ppr_best["primary_map_at_5"]))<1e-15,
            "An already evaluated comparison method exceeds primary-best PPR")
    return ppr_config,ppr_best,{
        "step5_configurations_checked":len(ppr_rows),"step5_verifier_passed":True,
        "step6_configurations_checked":len(step6_rows),"step6_verifier_passed":True,
        "best_ppr_primary_map_at_5":float(ppr_best["primary_map_at_5"]),
        "best_association_rules_primary_map_at_5":association_best,
        "all_saved_comparison_methods_primary_map_at_5":overall,
        "selected_ppr_config_id":ppr_best["config_id"]}


def load_template():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==["user_id","tv_show_id"],"Original Kaggle template schema mismatch")
        users=[]
        for row in reader:
            require(len(row)==2,"Malformed original template row")
            users.append(evaluation.user_id(row[0]))
    require(len(users)==1260 and len(set(users))==1260,"Original template user population invalid")
    return tuple(users)


def candidate_pool_from_test_schedule():
    expected=data_pipeline.SCHEDULE_COLUMNS
    lower,upper=validation.WINDOWS["test"]
    ids=set()
    with TEST_SCHEDULE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==expected,"Test schedule schema mismatch")
        for row in reader:
            require(len(row)==len(expected),"Malformed test schedule row")
            raw=row[5]
            canonical=validation.schedule_program_id(raw)
            if canonical=="0":continue
            start=data_pipeline.parse_timestamp(row[2],schedule=True)
            duration=int(row[3])
            if duration<=0:continue
            if start<upper and start+duration>lower:ids.add(canonical)
    pool=evaluation.candidate_pool(ids)
    require(len(pool)>=5,"Test schedule has fewer than five valid programs")
    return pool


def load_history_profiles():
    interaction_manifest=json_read(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    history_info=interaction_manifest["artifacts"][HISTORY.name]
    actual_hash=digest(HISTORY)
    require(actual_hash==history_info["sha256"],"Full-history interaction hash mismatch")
    bounds=validation.WINDOWS["history"]
    profiles={};viewers=Counter();rows=0
    previous=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==validation.INTERACTION_COLUMNS,"Full-history schema mismatch")
        for row in reader:
            user=evaluation.user_id(row["user_id"])
            item=validation.program_id(row["tv_show_id"])
            airings,seconds,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            key=(user,int(item))
            require(previous is None or key>previous,"History interactions are not unique and sorted")
            previous=key
            require(airings>0 and seconds>=120*airings,"History interaction violates 120-second qualifying threshold")
            require(bounds[0]<=first<last<=bounds[1],"History interaction leaks outside Weeks 11-30")
            profiles.setdefault(user,{})[item]=(airings,seconds,last)
            viewers[item]+=1;rows+=1
    require(rows==interaction_manifest["periods"]["history"]["unique_user_program_pairs"],"History interaction count differs from Step 2 manifest")
    data=ppr.TrainingProfiles(nodes=tuple(sorted(viewers,key=int)),
        index={item:i for i,item in enumerate(sorted(viewers,key=int))},
        profiles=profiles,distinct_viewers=dict(viewers),training_sha256=actual_hash,
        training_window_utc=tuple(bounds),rows=rows,runtime_seconds=0.0)
    return data,interaction_manifest


def make_full_history_popularity(data,candidates):
    airs=Counter();seconds=Counter()
    for profile in data.profiles.values():
        for item,(count,watched,_) in profile.items():
            airs[item]+=count;seconds[item]+=watched
    return popularity.PopularityModel(candidates=candidates,distinct_users=data.distinct_viewers,
        qualifying_airings=dict(airs),watched_seconds=dict(seconds),training_sha256=data.training_sha256,
        training_window_utc=data.training_window_utc,training_rows=data.rows,training_runtime_seconds=0.0)


def main():
    started=time.monotonic()
    require(not SUBMISSION.exists(),"Refusing to overwrite an existing final_submission.csv")
    require(not MANIFEST.exists() and not GRAPH_PATH.exists(),"Refusing to overwrite existing final model evidence")
    ppr_config,ppr_best,sweep_evidence=verify_saved_experiments()
    selected=ppr_config["best_config"]
    require(selected=={"config_id":"cosine_pfrequency_a0.35_k30","weight":"cosine","personalization":"frequency",
        "restart_probability":0.35,"max_neighbors":30,"history_cap":40,"min_common_viewers":3,
        "tolerance":1e-8,"max_iterations":250,"recency_half_life_seconds":2592000,
        "stage":"stage2_restart_and_neighborhood"},"PPR configuration does not match verified report")
    users=load_template();pool=candidate_pool_from_test_schedule()
    step2_test=validation.load_candidates(ROOT/"data"/"step2"/"test_candidates.csv",
        expected_hash=json_read(ROOT/"data"/"step2"/"join_manifest.json")["artifacts"]["test_candidates.csv"]["sha256"])
    # Step 2's catalog also includes boundary-crossing rows from earlier
    # schedules; the Kaggle pool is intentionally restricted to the named
    # Weeks 31-42 schedule in the task.
    require(set(pool).issubset(set(step2_test)),"Week 31-42 schedule IDs are not contained in the verified Step 2 catalog")
    data,interaction_manifest=load_history_profiles()
    train_done=time.monotonic()
    graph=ppr.build_graph(data,weight_type=selected["weight"],max_neighbors=selected["max_neighbors"],
        history_cap=selected["history_cap"],min_common_viewers=selected["min_common_viewers"])
    graph_done=time.monotonic()
    with GRAPH_PATH.open("wb") as stream:np.savez_compressed(stream,**ppr.graph_npz_arrays(graph))
    pop=make_full_history_popularity(data,pool)
    fallback=pop.ranking("distinct_users")
    predictions,infos=ppr.predict_cohort(graph,data,users,pool,fallback,
        personalization=selected["personalization"],restart_probability=selected["restart_probability"],
        tolerance=selected["tolerance"],max_iterations=selected["max_iterations"],
        half_life_seconds=selected["recency_half_life_seconds"])
    pred_done=time.monotonic()
    require(tuple(predictions)==users,"Prediction user order differs from original template")
    require(all(len(predictions[u])==5 and len(set(predictions[u]))==5 and set(predictions[u]).issubset(set(pool))
                and all(int(item)>0 for item in predictions[u]) for u in users),"Generated predictions violate Top-5 constraints")
    with SUBMISSION.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["user_id","tv_show_id"])
        for user in users:writer.writerow([user," ".join(predictions[user])])
    total=time.monotonic()-started
    join_manifest=json_read(ROOT/"data"/"step2"/"join_manifest.json")
    source_manifest=join_manifest["source_files_before"]
    original_submission=source_manifest["submission.csv"]
    original_schedule=source_manifest["export_arh_31-42-final.csv"]
    require(digest(TEMPLATE)==original_submission["sha256"],"Original template hash mismatch")
    require(digest(TEST_SCHEDULE)==original_schedule["sha256"],"Original test schedule hash mismatch")
    manifest={
        "model":"Personalized PageRank on weighted sparse program co-view graph",
        "selected_validation_config_id":selected["config_id"],"selected_validation_primary_map_at_5":ppr_config["best_metrics"]["primary_map_at_5"],
        "validation_protocol_id":ppr_config["protocol_id"],"validation_config_sha256":ppr_config["evaluation_config_sha256"],
        "training_period":"Weeks 11-30 inclusive; UTC interval [1583704800,1595797200)",
        "training_period_utc":list(data.training_window_utc),"training_source_path":str(HISTORY.relative_to(ROOT)),
        "training_source_sha256":data.training_sha256,"training_source_manifest_sha256":digest(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json"),
        "training_threshold_seconds_per_airing":120,"training_interaction_rows":data.rows,
        "training_users":len(data.profiles),"training_program_nodes":len(data.nodes),
        "matching_procedure":"same-channel half-open overlap; accumulate overlap fragments per user/broadcast, merge Weeks 11-20 and 21-30 before applying threshold; retain >=120 unioned overlap seconds; drop tv_show_id=0",
        "model_features":"qualifying-airing counts and co-view structure from Weeks 11-30 only; user frequency restart vector; no schedule metadata features",
        "test_viewing_behavior_used":False,"test_schedule_used_only_for_candidates":True,
        "candidate_schedule_path":str(TEST_SCHEDULE),"candidate_schedule_sha256":digest(TEST_SCHEDULE),
        "candidate_count":len(pool),"candidate_ids_sha256":evaluation.identity_hash(pool),
        "template_path":str(TEMPLATE),"template_sha256":digest(TEMPLATE),"template_users":len(users),
        "template_user_order_sha256":evaluation.identity_hash(users),"submission_path":str(SUBMISSION.relative_to(ROOT)),
        "submission_sha256":digest(SUBMISSION),"predictions_per_user":5,
        "repeat_previously_watched_allowed":True,"within_list_duplicates_allowed":False,
        "fallback_policy":evaluation.FALLBACK_POLICY,"fallback_variant":"distinct_training_users",
        "fallback_training_period":"Weeks 11-30","cold_start_users":sum(not data.profiles.get(u) for u in users),
        "cold_start_users_using_fallback":sum(not data.profiles.get(u) for u in users),
        "fallback_slots":sum(infos[u]["fallback_slots"] for u in users),
        "hyperparameters":selected,"graph_statistics":graph.statistics(),
        "graph_path":str(GRAPH_PATH.relative_to(ROOT)),"graph_sha256":digest(GRAPH_PATH),"graph_bytes":GRAPH_PATH.stat().st_size,
        "saved_experiment_crosschecks":sweep_evidence,
        "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,
            "cpu_count":__import__("os").cpu_count(),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},
        "runtime_seconds":{"history_profile_load_and_checks":train_done-started,"graph_fit":graph_done-train_done,
            "fallback_and_prediction_generation":pred_done-graph_done,"submission_write_and_manifest":time.monotonic()-pred_done,
            "end_to_end_including_saved_experiment_checks":total},
        "saved_ppr_experiment_prediction_sha256":ppr_best["prediction_sha256"],
        "step2_join_manifest_sha256":digest(ROOT/"data"/"step2"/"join_manifest.json"),
        "step2_history_interaction_manifest_sha256":digest(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json"),
        "step2_history_stats":interaction_manifest["periods"]["history"],
        "no_kaggle_submission_performed":True}
    json_write(MANIFEST,manifest)
    print(json.dumps({"submission":str(SUBMISSION),"sha256":manifest["submission_sha256"],
        "users":len(users),"candidate_count":len(pool),"training_rows":data.rows,
        "graph_runtime_seconds":graph.runtime_seconds,"end_to_end_seconds":total,
        "best_validation_MAP_at_5":manifest["selected_validation_primary_map_at_5"]},indent=2))


if __name__=="__main__":main()
