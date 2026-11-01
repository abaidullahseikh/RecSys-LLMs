"""Build the separately labeled experimental PPR+PF+CBF v3 from Weeks 11-30."""

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import scipy

import content_based as cbf
import data_pipeline
import evaluation
import ppr
import popularity
import validation
from run_cbf_step9 import source_schedule_metadata

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
TEMPLATE=SOURCE/"submission.csv"
SCHEDULE_TRAIN=SOURCE/"export_arh_11-20-final.csv"
SCHEDULE_VALID=SOURCE/"export_arh_21-30-final.csv"
SCHEDULE_TEST=SOURCE/"export_arh_31-42-final.csv"
SUBMISSION=ROOT/"submission"/"final_submission_v3.csv"
MANIFEST=ROOT/"results"/"final_model_v3_manifest.json"
GRAPH_PATH=ROOT/"results"/"final_ppr_graph_v3.npz"
VALIDATION_RESULTS=ROOT/"results"/"step9_cbf"/"step9_validation_results.json"

SELECTED_ID="ppr_pf_cbf_0.40_0.40_0.20"
FIELD_CONFIG="channel_context_emphasis"
PROFILE_WEIGHT="airings"
FUSION_WEIGHTS=(.40,.40,.20)
RRF_CONSTANT=60


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def template_users():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==["user_id","tv_show_id"],"Original template schema mismatch")
        users=[]
        for row in reader:
            require(len(row)==2,"Malformed original template row")
            users.append(evaluation.user_id(row[0]))
    require(len(users)==1260 and len(set(users))==1260,"Template user IDs invalid")
    return tuple(users)


def all_schedule_candidates():
    ids=set()
    with SCHEDULE_TEST.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==data_pipeline.SCHEDULE_COLUMNS,"Week 31-42 schedule schema mismatch")
        for line,row in enumerate(reader,2):
            require(len(row)==len(data_pipeline.SCHEDULE_COLUMNS),f"Malformed test schedule row {line}")
            item=validation.schedule_program_id(row[5])
            data_pipeline.parse_timestamp(row[2],schedule=True)
            if item!="0" and int(row[3])>0:ids.add(item)
    return evaluation.candidate_pool(ids)


def all_test_schedule_metadata():
    rows=[]
    with SCHEDULE_TEST.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==data_pipeline.SCHEDULE_COLUMNS,"Test schedule metadata schema mismatch")
        for line,row in enumerate(reader,2):
            require(len(row)==len(data_pipeline.SCHEDULE_COLUMNS),f"Malformed test metadata row {line}")
            item=validation.schedule_program_id(row["tv_show_id"])
            data_pipeline.parse_timestamp(row["start_time"],schedule=True)
            if item!="0" and int(row["duration"])>0:rows.append((item,row))
    return cbf.aggregate_metadata_rows(rows)


def load_full_history():
    interaction_manifest=read_json(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    require(digest(HISTORY)==interaction_manifest["artifacts"][HISTORY.name]["sha256"],"Weeks 11-30 history hash mismatch")
    bounds=validation.WINDOWS["history"]
    profiles={};viewers=Counter();airings=Counter();seconds=Counter();row_count=0;previous=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==validation.INTERACTION_COLUMNS,"Historical interactions schema mismatch")
        for row in reader:
            user=evaluation.user_id(row["user_id"]);item=validation.program_id(row["tv_show_id"])
            count,watched,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            key=(user,int(item))
            require(previous is None or key>previous,"Historical interactions are duplicated or unsorted")
            previous=key
            require(count>0 and watched>=120*count,"Interaction below 120-second threshold")
            require(bounds[0]<=first<last<=bounds[1],"Training interaction outside Weeks 11-30")
            profiles.setdefault(user,{})[item]=(count,watched,last)
            viewers[item]+=1;airings[item]+=count;seconds[item]+=watched;row_count+=1
    require(row_count==interaction_manifest["periods"]["history"]["unique_user_program_pairs"],"History rows differ from Step 2 manifest")
    nodes=tuple(sorted(viewers,key=int))
    model=ppr.TrainingProfiles(nodes,{item:i for i,item in enumerate(nodes)},profiles,dict(viewers),
        digest(HISTORY),bounds,row_count,0.0)
    pop=popularity.PopularityModel((),dict(viewers),dict(airings),dict(seconds),digest(HISTORY),bounds,row_count,0.0)
    return model,pop,interaction_manifest


def full_ppr_rankings(graph,data,candidates,fallback,users):
    idx={item:i for i,item in enumerate(graph.nodes)};fallback_rank={x:i for i,x in enumerate(fallback)}
    result={};scored_slots=0
    for user in users:
        profile=data.profiles.get(user,{})
        if profile:
            restart=ppr.personalization_vector(profile,data,variant="frequency")
            scores,_=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            ranked=[item for item in candidates if item in idx and scores[idx[item]]>0]
            ranked.sort(key=lambda item:(-scores[idx[item]],fallback_rank[item],int(item)))
            scored_slots+=min(5,len(ranked))
        else:ranked=[]
        seen=set(ranked)
        result[user]=tuple(ranked+[item for item in fallback if item not in seen])
    return result,scored_slots


def full_personal_frequency_rankings(data,candidates,fallback,users):
    eligible=set(candidates);result={}
    for user in users:
        profile=data.profiles.get(user,{})
        ranked=sorted((item for item in profile if item in eligible),key=lambda item:(-profile[item][0],int(item)))
        seen=set(ranked);result[user]=tuple(ranked+[item for item in fallback if item not in seen])
    return result


def full_cbf_rankings(data,train_docs,candidate_docs,candidates,fallback,users):
    weights=cbf.FIELD_WEIGHT_CONFIGS[FIELD_CONFIG]
    space=cbf.FeatureSpace.fit(train_docs,weights)
    candidate_matrix=space.transform(candidate_docs,candidates,weights)
    train_index={item:i for i,item in enumerate(space.training_items)}
    user_vectors,profile_stats=cbf.build_user_profiles(data.profiles,space.training_items,train_index,
        space.training_matrix,users,weighting=PROFILE_WEIGHT)
    indexed,diagnostics=cbf.rank_candidates(candidate_matrix,user_vectors,candidates,fallback,batch_size=64)
    return {user:indexed[i] for i,user in enumerate(users)},{
        "field_config":FIELD_CONFIG,"field_weights":weights,"user_profile_weighting":PROFILE_WEIGHT,
        "training_documents":len(space.training_items),"vocabulary_size":len(space.vocabulary),
        "training_matrix_shape":list(space.training_matrix.shape),"training_matrix_nnz":int(space.training_matrix.nnz),
        "candidate_matrix_shape":list(candidate_matrix.shape),"candidate_matrix_nnz":int(candidate_matrix.nnz),
        "user_profile_stats":profile_stats,
        "users_with_cbf_fallback_slots":sum(diagnostics[i]["fallback_slots_top5"]>0 for i in range(len(users))),
        "cbf_popularity_fallback_slots":sum(diagnostics[i]["fallback_slots_top5"] for i in range(len(users)))}


def rrf_top5(rankings,weights,cohort,candidates,fallback):
    if len(rankings)!=len(weights) or not math.isclose(math.fsum(weights),1.0,abs_tol=1e-12):
        raise ValueError("Rank fusion weights must sum to one")
    matrices=[np.empty((len(cohort),len(candidates)),dtype=np.int32) for _ in rankings]
    index={item:i for i,item in enumerate(candidates)};positions=np.arange(len(candidates),dtype=np.int32)
    for n,ranking in enumerate(rankings):
        for row,user in enumerate(cohort):
            order=np.fromiter((index[item] for item in ranking[user]),dtype=np.int32,count=len(candidates))
            matrices[n][row,order]=positions
    scores=np.zeros(matrices[0].shape,dtype=np.float64)
    for weight,pos in zip(weights,matrices):scores+=weight/(RRF_CONSTANT+pos)
    fallback_position={item:i for i,item in enumerate(fallback)}
    tie=np.asarray([fallback_position[item] for item in candidates],dtype=np.int32)
    numeric=np.asarray([int(item) for item in candidates],dtype=np.int64)
    output={}
    for row,user in enumerate(cohort):
        order=np.lexsort((numeric,tie,-scores[row]))
        output[user]=tuple(candidates[i] for i in order[:5])
    return output


def main():
    require(not SUBMISSION.exists() and not MANIFEST.exists(),
            "Refusing to overwrite an existing v3 artifact")
    started=time.monotonic();users=template_users();candidates=all_schedule_candidates()
    validation_result=read_json(VALIDATION_RESULTS)
    require(validation_result["selected_candidate_config_id"]==SELECTED_ID,
        "v3 config does not match selected CBF validation candidate")
    require(validation_result["v3_supported_by_frozen_secondary_vs_existing_v2"] is False,
        "CBF candidate unexpectedly beats v2; selection/status needs review")
    v2_manifest=read_json(ROOT/"results"/"final_model_v2_manifest.json")
    require(v2_manifest["selected_validation_config_id"]=="rrf_ppr_personal_frequency_ppr_weight_0.50",
        "Existing v2 model identity mismatch")
    data,pop,interaction_manifest=load_full_history();profiles_done=time.monotonic()
    ppr_graph=ppr.build_graph(data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    graph_done=time.monotonic()
    arrays=ppr.graph_npz_arrays(ppr_graph)
    if GRAPH_PATH.exists():
        with np.load(GRAPH_PATH,allow_pickle=False) as saved:
            require(set(saved.files)==set(arrays) and all(np.array_equal(saved[name],arrays[name]) for name in arrays),
                "Existing interrupted v3 graph differs from the fresh refit")
    else:
        with GRAPH_PATH.open("wb") as stream:np.savez_compressed(stream,**arrays)
    fallback=pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,
        pop.training_sha256,pop.training_window_utc,pop.training_rows,0.0).ranking("distinct_users")
    train_docs=source_schedule_metadata([SCHEDULE_TRAIN,SCHEDULE_VALID],validation.WINDOWS["history"])
    candidate_docs=all_test_schedule_metadata()
    missing=set(candidates)-set(candidate_docs)
    if missing:raise ValueError(f"Final test candidates missing metadata documents: {len(missing)}")
    feature_started=time.monotonic()
    cbf_rankings,cbf_stats=full_cbf_rankings(data,train_docs,candidate_docs,candidates,fallback,users)
    cbf_done=time.monotonic()
    ppr_rankings,ppr_scored_slots=full_ppr_rankings(ppr_graph,data,candidates,fallback,users)
    pf_rankings=full_personal_frequency_rankings(data,candidates,fallback,users)
    predictions=rrf_top5([ppr_rankings,pf_rankings,cbf_rankings],FUSION_WEIGHTS,users,candidates,fallback)
    predict_done=time.monotonic()
    eligible=set(candidates)
    require(tuple(predictions)==users,"v3 user/order mismatch")
    require(all(len(predictions[u])==5 and len(set(predictions[u]))==5 and set(predictions[u]).issubset(eligible)
        and all(item.isascii() and item.isdecimal() and not item.startswith("0") for item in predictions[u]) for u in users),
        "v3 output contract violation")
    with SUBMISSION.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","tv_show_id"])
        for user in users:writer.writerow([user," ".join(predictions[user])])
    join_manifest=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    original=join_manifest["source_files_before"]
    require(digest(TEMPLATE)==original["submission.csv"]["sha256"],"Original Kaggle template hash mismatch")
    require(digest(SCHEDULE_TEST)==original["export_arh_31-42-final.csv"]["sha256"],"Original Week 31-42 schedule hash mismatch")
    require(digest(ROOT/"submission"/"final_submission.csv")==read_json(ROOT/"results"/"final_model_manifest.json")["submission_sha256"],
        "Original v1 submission changed")
    require(digest(ROOT/"submission"/"final_submission_v2.csv")==v2_manifest["submission_sha256"],"Original v2 submission changed")
    manifest={
        "version":3,"status":"experimental_candidate_not_recommended_final","recommended_final_submission":"submission/final_submission_v2.csv",
        "model":"Experimental weighted RRF: PPR + personal qualifying-airing frequency + sparse metadata CBF",
        "validation_selected_config_id":SELECTED_ID,
        "local_primary_validation_map_at_5":validation_result["selected_primary_map_at_5"],
        "local_secondary_validation_map_at_5":validation_result["selected_secondary_map_at_5"],
        "existing_v2_local_primary_validation_map_at_5":validation_result["v2_existing_primary_map_at_5"],
        "existing_v2_local_secondary_validation_map_at_5":validation_result["v2_existing_secondary_map_at_5"],
        "v3_secondary_improves_v2":False,"kaggle_score":None,
        "selection_note":"Highest-secondary-scoring new CBF/hybrid candidate in the saved Step 9 grid; below v2, so this file is experimental only.",
        "training_period":"Weeks 11-30 inclusive; UTC [1583704800,1595797200)",
        "training_period_utc":list(data.training_window_utc),"training_source_path":str(HISTORY.relative_to(ROOT)),
        "training_source_sha256":data.training_sha256,"training_interaction_rows":data.rows,
        "threshold_seconds_per_airing":120,"training_users":len(data.profiles),"training_program_nodes":len(data.nodes),
        "interaction_match":"Step 2 same-channel half-open overlap; combine disjoint Week 11-20 and Week 21-30 overlap per user/airing; >=120 seconds; ID zero excluded",
        "ppr":{"weight":"cosine","personalization":"qualifying-airing frequency","restart_probability":.35,
            "top_neighbors_per_node":30,"history_cap":40,"min_common_viewers":3,"tolerance":1e-8,"max_iterations":250},
        "personal_frequency":{"score":"qualifying airing count descending","tie_break":"numeric program ID ascending"},
        "cbf":{"method":"sparse TF-IDF-equivalent content cosine","field_config":FIELD_CONFIG,
            "field_weights":cbf.FIELD_WEIGHT_CONFIGS[FIELD_CONFIG],"profile_weighting":PROFILE_WEIGHT,
            "idf_fit_period":"Weeks 11-30 schedules only","training_schedule_sources":[str(SCHEDULE_TRAIN),str(SCHEDULE_VALID)],
            "candidate_metadata_source":str(SCHEDULE_TEST),"candidate_metadata_only":True,"statistics":cbf_stats},
        "rank_fusion":{"method":"reciprocal-rank fusion","weights":{"ppr":.4,"personal_frequency":.4,"cbf":.2},
            "constant":RRF_CONSTANT,"rank_positions_zero_based":True,"ties":"shared full-history popularity order then numeric program ID"},
        "fallback_policy":evaluation.FALLBACK_POLICY,"fallback_variant":"distinct training users desc, qualifying airings desc, numeric ID asc",
        "repeat_previously_watched_allowed":True,"within_list_duplicates_allowed":False,
        "candidate_source":"all positive-duration nonzero IDs from export_arh_31-42-final.csv",
        "candidate_count":len(candidates),"candidate_ids_sha256":evaluation.identity_hash(candidates),
        "test_schedule_used_only_for_candidates_and_metadata":True,"test_user_viewing_behavior_used":False,
        "template_path":str(TEMPLATE),"template_sha256":digest(TEMPLATE),"template_users":len(users),
        "template_order_sha256":evaluation.identity_hash(users),"columns":["user_id","tv_show_id"],"predictions_per_user":5,
        "submission_path":str(SUBMISSION.relative_to(ROOT)),"submission_sha256":digest(SUBMISSION),
        "graph_path":str(GRAPH_PATH.relative_to(ROOT)),"graph_sha256":digest(GRAPH_PATH),"graph_bytes":GRAPH_PATH.stat().st_size,
        "graph_statistics":ppr_graph.statistics(),"ppr_scored_top5_slots_before_fusion":ppr_scored_slots,
        "full_history_cold_start_users":sum(not data.profiles.get(u) for u in users),
        "step9_experiment_results_sha256":digest(VALIDATION_RESULTS),
        "step9_comparison_sha256":digest(ROOT/"results"/"step9_cbf"/"model_comparison.csv"),
        "v2_submission_sha256_preserved":v2_manifest["submission_sha256"],
        "runtime_seconds":{"history_profiles_load":profiles_done-started,"ppr_graph_fit":graph_done-profiles_done,
            "schedule_metadata_and_cbf_fit":cbf_done-feature_started,"prediction_and_rank_fusion":predict_done-cbf_done,
            "end_to_end":time.monotonic()-started},
        "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},
        "kaggle_submission_performed":False}
    write_json(MANIFEST,manifest)
    print(json.dumps({k:manifest[k] for k in ("submission_path","submission_sha256","status","candidate_count","template_users",
        "validation_selected_config_id","local_primary_validation_map_at_5","local_secondary_validation_map_at_5",
        "existing_v2_local_secondary_validation_map_at_5","kaggle_submission_performed")},indent=2))


if __name__=="__main__":main()
