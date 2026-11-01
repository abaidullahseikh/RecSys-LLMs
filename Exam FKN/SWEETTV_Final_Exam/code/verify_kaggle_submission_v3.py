"""Independent format, provenance, and full-model verifier for experimental v3."""

from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

import content_based as cbf
import data_pipeline
import evaluation
import ppr
import popularity
import validation

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
TEMPLATE=SOURCE/"submission.csv"
SCHEDULE_TRAIN=SOURCE/"export_arh_11-20-final.csv"
SCHEDULE_VALID=SOURCE/"export_arh_21-30-final.csv"
SCHEDULE_TEST=SOURCE/"export_arh_31-42-final.csv"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
SUBMISSION=ROOT/"submission"/"final_submission_v3.csv"
MANIFEST=ROOT/"results"/"final_model_v3_manifest.json"
CHECKSUM=ROOT/"results"/"final_submission_v3_sha256.txt"
VERIFICATION=ROOT/"results"/"final_submission_v3_verification.json"
GRAPH=ROOT/"results"/"final_ppr_graph_v3.npz"
ID_RE=re.compile(r"[0-9]+\Z",re.ASCII)


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def canonical_raw_id(raw):
    if not isinstance(raw,str) or not ID_RE.fullmatch(raw):raise ValueError(f"Invalid numeric schedule/program ID {raw!r}")
    return str(int(raw))


def load_template_users():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=["user_id","tv_show_id"]:raise ValueError("Template header mismatch")
        users=[]
        for row in reader:
            if len(row)!=2 or not row[0] or not ID_RE.fullmatch(row[0]):raise ValueError("Malformed template user row")
            users.append(row[0])
    if len(users)!=1260 or len(set(users))!=1260:raise ValueError("Template cohort cardinality mismatch")
    return tuple(users)


def load_test_schedule_ids_and_docs():
    all_rows=[];ids=set()
    with SCHEDULE_TEST.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError("Test schedule schema mismatch")
        for row in reader:
            item=canonical_raw_id(row["tv_show_id"])
            data_pipeline.parse_timestamp(row["start_time"],schedule=True)
            if item!="0" and int(row["duration"])>0:
                ids.add(item);all_rows.append((item,row))
    return evaluation.candidate_pool(ids),cbf.aggregate_metadata_rows(all_rows)


def schedule_docs_for_history():
    lower,upper=validation.WINDOWS["history"];schedule_rows=[]
    for path in (SCHEDULE_TRAIN,SCHEDULE_VALID):
        with path.open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError(f"History schedule schema mismatch: {path}")
            for row in reader:
                item=canonical_raw_id(row["tv_show_id"])
                if item=="0":continue
                start=data_pipeline.parse_timestamp(row["start_time"],schedule=True);duration=int(row["duration"])
                if duration>0 and start<upper and start+duration>lower:schedule_rows.append((item,row))
    return cbf.aggregate_metadata_rows(schedule_rows)


def load_history(manifest):
    info=read_json(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    if digest(HISTORY)!=info["artifacts"][HISTORY.name]["sha256"] or digest(HISTORY)!=manifest["training_source_sha256"]:
        raise ValueError("Weeks 11-30 interaction hash mismatch")
    bounds=validation.WINDOWS["history"];profiles={};viewers=Counter();airings=Counter();seconds=Counter();rows=0;prev=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=validation.INTERACTION_COLUMNS:raise ValueError("Training history schema mismatch")
        for line,row in enumerate(reader,2):
            user=row["user_id"];item=canonical_raw_id(row["tv_show_id"])
            if not user or not ID_RE.fullmatch(user) or item=="0":raise ValueError(f"Invalid user/program ID at history row {line}")
            key=(user,int(item))
            if prev is not None and key<=prev:raise ValueError("Historical interactions are duplicated or unsorted")
            prev=key
            count,watched,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            if count<=0 or watched<count*120 or not bounds[0]<=first<last<=bounds[1]:
                raise ValueError(f"Training threshold/time violation on row {line}")
            profiles.setdefault(user,{})[item]=(count,watched,last)
            viewers[item]+=1;airings[item]+=count;seconds[item]+=watched;rows+=1
    if rows!=manifest["training_interaction_rows"] or rows!=info["periods"]["history"]["unique_user_program_pairs"]:
        raise ValueError("Historical training row-count mismatch")
    nodes=tuple(sorted(viewers,key=int))
    model=ppr.TrainingProfiles(nodes,{x:i for i,x in enumerate(nodes)},profiles,dict(viewers),digest(HISTORY),bounds,rows,0)
    pop=popularity.PopularityModel((),dict(viewers),dict(airings),dict(seconds),digest(HISTORY),bounds,rows,0)
    return model,pop


def build_full_rankings(graph,data,pop,cbf_ranks,candidates,users):
    fallback=pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,
        pop.training_sha256,pop.training_window_utc,pop.training_rows,0).ranking("distinct_users")
    fallback_pos={x:i for i,x in enumerate(fallback)};node_pos={x:i for i,x in enumerate(graph.nodes)}
    ppr_ranks={};frequency_ranks={}
    for user in users:
        profile=data.profiles.get(user,{})
        if profile:
            restart=ppr.personalization_vector(profile,data,variant="frequency")
            scores,_=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            ranked=[x for x in candidates if x in node_pos and scores[node_pos[x]]>0]
            ranked.sort(key=lambda x:(-scores[node_pos[x]],fallback_pos[x],int(x)))
        else:ranked=[]
        seen=set(ranked);ppr_ranks[user]=tuple(ranked+[x for x in fallback if x not in seen])
        personal=sorted((x for x in profile if x in set(candidates)),key=lambda x:(-profile[x][0],int(x)))
        seen_personal=set(personal);frequency_ranks[user]=tuple(personal+[x for x in fallback if x not in seen_personal])
    return ppr_ranks,frequency_ranks,fallback


def fuse_top5(rankings,weights,candidates,users,fallback):
    index={x:i for i,x in enumerate(candidates)};ranks=[np.empty((len(users),len(candidates)),dtype=np.int32) for _ in rankings]
    positions=np.arange(len(candidates),dtype=np.int32)
    for m,ranking in enumerate(rankings):
        for row,user in enumerate(users):
            ordered=np.fromiter((index[x] for x in ranking[user]),dtype=np.int32,count=len(candidates))
            ranks[m][row,ordered]=positions
    score=np.zeros(ranks[0].shape,dtype=np.float64)
    for w,matrix in zip(weights,ranks):score+=w/(60+matrix)
    fallback_pos={x:i for i,x in enumerate(fallback)}
    tie=np.asarray([fallback_pos[x] for x in candidates],dtype=np.int32)
    numeric=np.asarray([int(x) for x in candidates],dtype=np.int64)
    out={}
    for r,user in enumerate(users):
        order=np.lexsort((numeric,tie,-score[r]));out[user]=tuple(candidates[i] for i in order[:5])
    return out


def main():
    started=time.monotonic();manifest=read_json(MANIFEST)
    if manifest["status"]!="experimental_candidate_not_recommended_final" or manifest["recommended_final_submission"]!="submission/final_submission_v2.csv":
        raise ValueError("V3 status/recommended final declaration changed")
    join=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    source_hashes=join["source_files_before"]
    if digest(TEMPLATE)!=source_hashes["submission.csv"]["sha256"] or digest(TEMPLATE)!=manifest["template_sha256"]:
        raise ValueError("Original Kaggle template changed")
    if digest(SCHEDULE_TEST)!=source_hashes["export_arh_31-42-final.csv"]["sha256"]:
        raise ValueError("Week 31-42 schedule hash mismatch")
    step9=read_json(ROOT/"results"/"step9_cbf"/"step9_validation_results.json")
    if digest(ROOT/"results"/"step9_cbf"/"step9_validation_results.json")!=manifest["step9_experiment_results_sha256"]:
        raise ValueError("Step 9 validation results hash mismatch")
    if digest(ROOT/"results"/"step9_cbf"/"model_comparison.csv")!=manifest["step9_comparison_sha256"]:
        raise ValueError("Step 9 model comparison hash mismatch")
    if (manifest["validation_selected_config_id"]!=step9["selected_candidate_config_id"]
            or not math.isclose(manifest["local_secondary_validation_map_at_5"],step9["selected_secondary_map_at_5"],abs_tol=1e-15)):
        raise ValueError("V3 configuration/score differs from validation-selected evidence")
    users=load_template_users();candidates,candidate_docs=load_test_schedule_ids_and_docs()
    if candidates!=tuple(sorted(candidates,key=int)) or len(candidates)!=6636:raise ValueError("Schedule candidate pool mismatch")
    if len(set(users))!=1260 or users!=validation.EvaluationProtocol().cohort:raise ValueError("Template users/order mismatch")
    if len(candidates)!=manifest["candidate_count"] or evaluation.identity_hash(candidates)!=manifest["candidate_ids_sha256"]:
        raise ValueError("Candidate identity differs from final v3 manifest")
    data,pop=load_history(manifest)
    if data.training_window_utc!=tuple(manifest["training_period_utc"]):raise ValueError("Training period mismatch")
    train_docs=schedule_docs_for_history()
    space=cbf.FeatureSpace.fit(train_docs,cbf.FIELD_WEIGHT_CONFIGS[manifest["cbf"]["field_config"]])
    cand_matrix=space.transform(candidate_docs,candidates,manifest["cbf"]["field_weights"])
    train_index={x:i for i,x in enumerate(space.training_items)}
    user_vectors,profile_stats=cbf.build_user_profiles(data.profiles,space.training_items,train_index,
        space.training_matrix,users,weighting=manifest["cbf"]["profile_weighting"])
    cbf_by_index,_=cbf.rank_candidates(cand_matrix,user_vectors,candidates,
        pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,pop.training_sha256,pop.training_window_utc,pop.training_rows,0).ranking("distinct_users"),batch_size=64)
    cbf_ranks={u:cbf_by_index[i] for i,u in enumerate(users)}
    graph=ppr.build_graph(data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    arrays=ppr.graph_npz_arrays(graph)
    if digest(GRAPH)!=manifest["graph_sha256"]:raise ValueError("Saved v3 graph checksum mismatch")
    with np.load(GRAPH,allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],arrays[k]) for k in arrays):
            raise ValueError("Saved v3 PPR graph differs from independent refit")
    ppr_ranks,pf_ranks,fallback=build_full_rankings(graph,data,pop,cbf_ranks,candidates,users)
    expected=fuse_top5([ppr_ranks,pf_ranks,cbf_ranks],[.4,.4,.2],candidates,users,fallback)
    rows=[]
    with SUBMISSION.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=["user_id","tv_show_id"]:raise ValueError("v3 CSV header mismatch")
        for line,row in enumerate(reader,2):
            if len(row)!=2:raise ValueError(f"v3 row {line} must have exactly two fields")
            rows.append(row)
    if len(rows)!=1260 or [x[0] for x in rows]!=list(users):raise ValueError("v3 rows/template order mismatch")
    output={};eligible=set(candidates)
    for line,(user,field) in enumerate(rows,2):
        if not user or not field:raise ValueError(f"Missing CSV cell on line {line}")
        tokens=field.split(" ")
        if len(tokens)!=5 or len(set(tokens))!=5:raise ValueError(f"Invalid Top-5 length/uniqueness on line {line}")
        for item in tokens:
            if not ID_RE.fullmatch(item) or item=="0" or item.startswith("0") or str(int(item))!=item:
                raise ValueError(f"Noncanonical or zero ID on line {line}")
            if item not in eligible:raise ValueError(f"Ineligible schedule ID {item} on line {line}")
        output[user]=tuple(tokens)
    if output!=expected:raise ValueError("v3 output differs from independent full model refit")
    if digest(SUBMISSION)!=manifest["submission_sha256"]:raise ValueError("v3 submission SHA differs from manifest")
    if digest(ROOT/"submission"/"final_submission_v2.csv")!=manifest["v2_submission_sha256_preserved"]:
        raise ValueError("V2 submission changed")
    if digest(ROOT/"submission"/"final_submission.csv")!=read_json(ROOT/"results"/"final_model_manifest.json")["submission_sha256"]:
        raise ValueError("V1 submission changed")
    CHECKSUM.write_text(digest(SUBMISSION)+"\n",encoding="ascii")
    if CHECKSUM.read_text(encoding="ascii").strip()!=digest(SUBMISSION):raise ValueError("v3 checksum file mismatch")
    counts=Counter(x for u in users for x in output[u])
    result={"all_passed":True,"passed_checks":12,"failed_checks":0,"submission_path":str(SUBMISSION.relative_to(ROOT)),
        "submission_sha256":digest(SUBMISSION),"checksum_file":str(CHECKSUM.relative_to(ROOT)),
        "users":len(users),"predictions_per_user":5,"candidate_count":len(candidates),"unique_recommended_programs":len(counts),
        "top10_program_user_counts":[{"program_id":x,"users":n} for x,n in counts.most_common(10)],
        "experimental_config_id":manifest["validation_selected_config_id"],"recommended_final":"submission/final_submission_v2.csv",
        "v2_local_secondary_map_at_5":manifest["existing_v2_local_secondary_validation_map_at_5"],
        "v3_local_secondary_map_at_5":manifest["local_secondary_validation_map_at_5"],
        "v3_is_recommended_final":False,"kaggle_score":None,"kaggle_submission_performed":False,
        "training_period":manifest["training_period"],"training_rows":data.rows,"training_sha256":data.training_sha256,
        "test_viewing_behavior_used":False,"independent_full_refit_reproduced_predictions":True,
        "graph_arrays_match_refit":True,"cbf_profile_stats":profile_stats,
        "verification_runtime_seconds":round(time.monotonic()-started,3)}
    write_json(ROOT/"results"/"final_submission_v3_verification.json",result)
    print(json.dumps({k:result[k] for k in ("all_passed","passed_checks","failed_checks","submission_path","submission_sha256",
        "users","candidate_count","v3_local_secondary_map_at_5","v2_local_secondary_map_at_5","v3_is_recommended_final",
        "verification_runtime_seconds")},indent=2))


if __name__=="__main__":main()
