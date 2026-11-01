"""Train sparse metadata CBF on Week 11-20 and compare CBF/rank-fusion models."""

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

import association_rules as ar
import content_based as cbf
import data_pipeline
import evaluation
import ppr
import personal_frequency
import popularity
import run_step8 as step8
import validation

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
OUT=ROOT/"results"/"step9_cbf"
SCHEDULE_TRAIN=SOURCE/"export_arh_11-20-final.csv"
SCHEDULE_VALID=SOURCE/"export_arh_21-30-final.csv"
SCHEDULE_TEST=SOURCE/"export_arh_31-42-final.csv"

PROFILE_WEIGHTS=("log_frequency","airings","watched_seconds")
FUSION_GRID={
    "ppr_plus_cbf":((.25,.75),(.50,.50),(.75,.25)),
    "personal_frequency_plus_cbf":((.25,.75),(.50,.50),(.75,.25)),
    "ppr_personal_frequency_plus_cbf":((.40,.40,.20),(.30,.30,.40),(.25,.25,.50)),
    "ppr_personal_frequency_association_rules_plus_cbf":((.30,.30,.20,.20),(.25,.25,.25,.25),(.30,.30,.10,.30)),
}


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def source_schedule_metadata(paths,window):
    """Stream allowed schedule metadata only; no viewing log is opened here."""
    rows=[]
    for path in paths:
        with Path(path).open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError(f"Schedule schema mismatch: {path}")
            for row in reader:
                item=validation.schedule_program_id(row["tv_show_id"])
                if item=="0":continue
                start=data_pipeline.parse_timestamp(row["start_time"],schedule=True)
                duration=int(row["duration"])
                if duration>0 and start<window[1] and start+duration>window[0]:rows.append((item,row))
    return cbf.aggregate_metadata_rows(rows)


def load_fraction80_top5_truth(protocol):
    """Sensitivity only: top five programs by count of airings watched >=80%."""
    path=validation.STEP2/"validation_airing_overlaps.csv.gz"
    join=validation.json_read(validation.STEP2/"join_manifest.json")
    if digest(path)!=join["artifacts"][path.name]["sha256"]:raise ValueError("Validation airing overlap hash mismatch")
    cohort=set(protocol.cohort);counts={u:Counter() for u in protocol.cohort}
    with gzip.open(path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        expected=["user_id","channel_id","broadcast_start_utc","broadcast_end_utc","tv_show_id","overlap_seconds","first_overlap_utc","last_overlap_utc"]
        if reader.fieldnames!=expected:raise ValueError("Validation airing overlap schema mismatch")
        for row in reader:
            user=row["user_id"]
            if user not in cohort:continue
            item=validation.program_id(row["tv_show_id"])
            start,end=int(row["broadcast_start_utc"]),int(row["broadcast_end_utc"])
            overlap=int(row["overlap_seconds"])
            if item!="0" and end>start and 5*overlap>=4*(end-start):counts[user][item]+=1
    truth={}
    for user in protocol.cohort:
        ordered=sorted(counts[user],key=lambda item:(-counts[user][item],int(item)))
        truth[user]=frozenset(ordered[:5])
    serialized=json.dumps([[u,sorted(truth[u],key=int)] for u in protocol.cohort],separators=(",",":"),ensure_ascii=True)
    relevance_hash=hashlib.sha256(serialized.encode()).hexdigest()
    details={"definition":"Top five user-programs by count of validation airings whose unioned user overlap is >=80% of broadcast duration",
        "fraction_threshold_numerator":4,"fraction_threshold_denominator":5,"users_with_nonempty_truth":sum(bool(x) for x in truth.values()),
        "relevance_ids_sha256":relevance_hash,"source_sha256":digest(path)}
    return truth,details


def metric_record(config_id,rankings,protocol,secondary,fraction_truth,elapsed,diagnostics=None,extra=None):
    # Exact Step 3 AP formula with one shared candidate check, avoiding rebuilding
    # the candidate set inside strict_top5 once per user and target definition.
    eligible=set(protocol.candidates);preds={u:tuple(rankings[u][:5]) for u in protocol.cohort}
    per_target={"primary":[] ,"secondary":[],"fraction80":[]}
    labelled={"primary":[],"secondary":[],"fraction80":[]}
    for user in protocol.cohort:
        ids=preds[user]
        if len(ids)!=5 or len(set(ids))!=5 or not set(ids).issubset(eligible):
            raise ValueError(f"Invalid strict Top-5 for {config_id}/{user}")
        for name,truth in (("primary",protocol.truth),("secondary",secondary),("fraction80",fraction_truth)):
            relevant=truth[user];seen=set();hits=0;terms=[]
            for rank,item in enumerate(ids,1):
                if item in relevant and item not in seen:
                    seen.add(item);hits+=1;terms.append(hits/rank)
            ap=math.fsum(terms)/min(5,len(relevant)) if relevant else 0.0
            per_target[name].append(ap)
            if relevant:labelled[name].append(ap)
    primary_map=math.fsum(per_target["primary"])/len(protocol.cohort)
    secondary_map=math.fsum(per_target["secondary"])/len(protocol.cohort)
    fraction_map=math.fsum(per_target["fraction80"])/len(protocol.cohort)
    frequencies=Counter(x for u in protocol.cohort for x in preds[u])
    diversity=step8.pairwise_diversity(preds,protocol.cohort)
    diversity["candidate_coverage_fraction"]=len(frequencies)/len(protocol.candidates)
    fallback_slots=sum((diagnostics or {}).get(u,{}).get("fallback_slots",0) for u in protocol.cohort)
    record={"config_id":config_id,"primary_map_at_5":primary_map,"secondary_map_at_5":secondary_map,
        "diagnostic_map_at_5_nonempty":math.fsum(labelled["primary"])/len(labelled["primary"]) if labelled["primary"] else None,
        "fraction80_top5_sensitivity_map_at_5":fraction_map,
        "fraction80_top5_nonempty_map_at_5":math.fsum(labelled["fraction80"])/len(labelled["fraction80"]) if labelled["fraction80"] else None,
        "hit_rate_at_5":sum(bool(set(preds[u])&protocol.truth[u]) for u in protocol.cohort)/len(protocol.cohort),
        "candidate_count":len(protocol.candidates),**diversity,"prediction_runtime_seconds":elapsed,
        "popularity_fallback_slots":fallback_slots,
        "users_with_any_popularity_fallback":sum((diagnostics or {}).get(u,{}).get("fallback_slots",0)>0 for u in protocol.cohort),
        "top10_recommended_program_counts":frequencies.most_common(10),"predictions":preds}
    record.update(extra or {})
    return record


def position_matrix(rankings,cohort,candidates):
    return step8.rank_position_matrix(rankings,cohort,candidates)


def multi_rrf(rankings,weights,cohort,candidates,fallback,constant=60,*,position_matrices=None):
    if len(rankings)!=len(weights) or not math.isclose(math.fsum(weights),1.0,abs_tol=1e-12):
        raise ValueError("RRF weights must be nonnegative and sum to one")
    matrices=position_matrices or [position_matrix(rank,cohort,candidates) for rank in rankings]
    if len(matrices)!=len(weights):raise ValueError("Position matrices/weights mismatch")
    scores=np.zeros_like(matrices[0],dtype=np.float64)
    for weight,positions in zip(weights,matrices):scores+=float(weight)/(constant+positions)
    fallback_rank={item:i for i,item in enumerate(fallback)}
    tie=np.asarray([fallback_rank[item] for item in candidates],dtype=np.int32)
    numeric=np.asarray([int(item) for item in candidates],dtype=np.int64)
    result={}
    for row,user in enumerate(cohort):
        order=np.lexsort((numeric,tie,-scores[row]))
        result[user]=tuple(candidates[i] for i in order)
    return result


def write_predictions(path,cohort,predictions):
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","tv_show_id"])
        for user in cohort:
            ids=tuple(evaluation.program_id(x) for x in predictions[user])
            if len(ids)!=5 or len(set(ids))!=5:raise ValueError("Experiment prediction is not five unique IDs")
            writer.writerow([user," ".join(ids)])


def complete_saved_predictions(predictions,cohort,fallback):
    result={}
    for user in cohort:
        seen=set(predictions[user])
        result[user]=tuple(predictions[user])+tuple(item for item in fallback if item not in seen)
    return result


def main():
    if OUT.exists() and (OUT/"step9_validation_results.json").exists():
        raise FileExistsError("Refusing to overwrite completed CBF/Step 9 results")
    if not OUT.exists():OUT.mkdir(parents=True,exist_ok=False)
    start_all=time.monotonic();protocol=validation.EvaluationProtocol();cohort=protocol.cohort;candidates=protocol.candidates
    secondary={}
    for user,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
        ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        secondary[user]=frozenset(field.split()) if field else frozenset()
    if tuple(secondary)!=cohort:raise ValueError("Secondary target differs from frozen validation cohort")
    fraction_truth,fraction_details=load_fraction80_top5_truth(protocol)
    bounds=validation.WINDOWS
    train_docs=source_schedule_metadata([SCHEDULE_TRAIN],bounds["train"])
    candidate_docs=source_schedule_metadata([SCHEDULE_TRAIN,SCHEDULE_VALID,SCHEDULE_TEST],bounds["validation"])
    print("Read schedule-only train/candidate metadata",len(train_docs),len(candidate_docs),flush=True)
    missing_metadata=set(candidates)-set(candidate_docs)
    if missing_metadata:raise ValueError(f"Validation candidates missing schedule metadata: {len(missing_metadata)}")
    if not train_docs:raise ValueError("No positive training-schedule metadata documents")
    train=ppr.build_training_profiles(protocol.config)
    pop=popularity.fit_popularity(protocol.config,candidates)
    fallback=pop.ranking("distinct_users")
    personal=personal_frequency.fit_personal_frequency(protocol.config,candidates)
    ppr_graph=ppr.build_graph(train,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    ppr_predictions,ppr_rankings,ppr_infos,_,_=step8.build_ppr_predictions(ppr_graph,train,cohort,candidates,fallback,
        "frequency",.35,collect_candidate_scores=False)
    saved_ppr={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv.DictReader(
        (ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv").open(newline="",encoding="utf-8"))}
    if ppr_predictions!=saved_ppr:raise ValueError("Recreated PPR baseline differs from saved Step 5")
    personal_rankings,personal_fallback_slots=step8.full_personal_rankings(personal,cohort,candidates,fallback)
    saved_personal={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv.DictReader(
        (ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv").open(newline="",encoding="utf-8"))}
    if step8.top5_from_full_rankings(personal_rankings)!=saved_personal:raise ValueError("Personal-frequency baseline differs from saved Step 4")

    # Train the documented best Step 6 rule model on the identical Week 11-20 split.
    ar_profiles,_,_=__import__("run_step6").build_training_profiles(protocol.config)
    ar_dataset=ar.build_window_dataset(protocol.config,14)
    ar_model=ar.mine_rules(ar_dataset,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.0,top_neighbors=50)
    ar_rankings,ar_fallback_slots=step8.full_ar_rankings(ar_model,ar_profiles,cohort,candidates,fallback,
        score_type="confidence_log_lift",weighting="log_frequency")
    print("Rebuilt matching baseline rankings",flush=True)
    saved_ar={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv.DictReader(
        (ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv").open(newline="",encoding="utf-8"))}
    if step8.top5_from_full_rankings(ar_rankings)!=saved_ar:raise ValueError("Association Rules baseline differs from saved Step 6")

    saved_hybrid={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv.DictReader(
        (ROOT/"results"/"step8"/"validation_predictions"/"rrf_ppr_personal_frequency_ppr_weight_0.50"/"predictions.csv").open(newline="",encoding="utf-8"))}
    hybrid_rank=multi_rrf([ppr_rankings,personal_rankings],[.5,.5],cohort,candidates,fallback)
    if step8.top5_from_full_rankings(hybrid_rank)!=saved_hybrid:raise ValueError("Existing PPR+PF hybrid differs from saved Step 8")

    experiments=[];inference_seconds={};feature_build_meta={}
    for feature_config,field_weights in cbf.FIELD_WEIGHT_CONFIGS.items():
        fit_started=time.monotonic()
        space=cbf.FeatureSpace.fit(train_docs,field_weights)
        train_matrix=space.training_matrix
        candidate_matrix=space.transform(candidate_docs,candidates,field_weights)
        fit_seconds=time.monotonic()-fit_started
        feature_build_meta[feature_config]={"training_documents":len(space.training_items),"vocabulary_size":len(space.vocabulary),
            "training_matrix_shape":list(train_matrix.shape),"training_matrix_nnz":int(train_matrix.nnz),
            "candidate_matrix_shape":list(candidate_matrix.shape),"candidate_matrix_nnz":int(candidate_matrix.nnz),
            "fit_and_transform_seconds":fit_seconds,"field_weights":field_weights}
        training_index={item:i for i,item in enumerate(space.training_items)}
        for profile_weight in PROFILE_WEIGHTS:
            config_id=f"cbf_{feature_config}_{profile_weight}"
            inference_start=time.monotonic()
            user_vectors,profile_stats=cbf.build_user_profiles(train.profiles,space.training_items,training_index,
                train_matrix,cohort,weighting=profile_weight)
            indexed,diagnostics=cbf.rank_candidates(candidate_matrix,user_vectors,candidates,fallback,batch_size=64)
            rankings={u:indexed[i] for i,u in enumerate(cohort)}
            user_diag={u:diagnostics[i] for i,u in enumerate(cohort)}
            elapsed=time.monotonic()-inference_start
            row=metric_record(config_id,rankings,protocol,secondary,fraction_truth,elapsed,
                {u:{"fallback_slots":user_diag[u]["fallback_slots_top5"]} for u in cohort},
                {"family":"cbf","feature_config":feature_config,"profile_weighting":profile_weight,
                 "fit_and_transform_seconds":fit_seconds,"users_with_profile":profile_stats["users_with_profile"],
                 "profile_program_pairs":profile_stats["profile_program_pairs"],
                 "mean_positive_similarity_candidates":float(np.mean([user_diag[u]["positive_similarity_candidates"] for u in cohort]))})
            experiments.append(row);inference_seconds[config_id]=elapsed
            folder=OUT/"predictions";folder.mkdir(exist_ok=True)
            write_predictions(folder/f"{config_id}.csv",cohort,row["predictions"])
        print("Completed CBF feature weighting",feature_config,flush=True)

    # First-stage selection is only within individually evaluated CBF models.
    cbf_rows=[row for row in experiments if row["family"]=="cbf"]
    best_cbf=sorted(cbf_rows,key=lambda row:(-row["secondary_map_at_5"],-row["primary_map_at_5"],
        row["prediction_runtime_seconds"],row["config_id"]))[0]
    # Refit the selected CBF feature weights and reconstruct its full candidate ranking for fusion.
    best_feature=best_cbf["feature_config"];best_profile=best_cbf["profile_weighting"]
    best_space=cbf.FeatureSpace.fit(train_docs,cbf.FIELD_WEIGHT_CONFIGS[best_feature])
    best_candidate_matrix=best_space.transform(candidate_docs,candidates,cbf.FIELD_WEIGHT_CONFIGS[best_feature])
    best_user_vectors,_=cbf.build_user_profiles(train.profiles,best_space.training_items,
        {item:i for i,item in enumerate(best_space.training_items)},best_space.training_matrix,cohort,weighting=best_profile)
    best_cbf_rows,_=cbf.rank_candidates(best_candidate_matrix,best_user_vectors,candidates,fallback,batch_size=64)
    cbf_full_rank={u:best_cbf_rows[i] for i,u in enumerate(cohort)}
    print("Selected CBF config",best_cbf["config_id"],best_cbf["secondary_map_at_5"],flush=True)
    positions={name:position_matrix(rank,cohort,candidates) for name,rank in (
        ("ppr",ppr_rankings),("personal_frequency",personal_rankings),
        ("association_rules",ar_rankings),("cbf",cbf_full_rank))}

    best_hybrid={"record":None,"ranking":None}
    def add_hybrid(config_id,families,rankings,weights,group):
        t=time.monotonic()
        names=tuple(families)
        fused=multi_rrf(rankings,weights,cohort,candidates,fallback,
            position_matrices=[positions[name] for name in names])
        elapsed=time.monotonic()-t
        record=metric_record(config_id,fused,protocol,secondary,fraction_truth,elapsed,None,
            {"family":group,"components":families,"rrf_weights":list(weights),"rrf_constant":60,
             "cbf_component":best_cbf["config_id"]})
        experiments.append(record);write_predictions(OUT/"predictions"/f"{config_id}.csv",cohort,record["predictions"])
        current=best_hybrid["record"]
        if current is None or (-record["secondary_map_at_5"],-record["primary_map_at_5"],record["prediction_runtime_seconds"],record["config_id"]) < \
                (-current["secondary_map_at_5"],-current["primary_map_at_5"],current["prediction_runtime_seconds"],current["config_id"]):
            best_hybrid.update(record=record,ranking=fused)
        print("Evaluated hybrid",config_id,record["secondary_map_at_5"],flush=True)
        return record

    # Requested 2-way and multi-way rank-fusion grids around the individually selected CBF.
    for w_ppr in (.25,.50,.75):
        add_hybrid(f"ppr_cbf_ppr_weight_{w_ppr:.2f}",["ppr","cbf"],[ppr_rankings,cbf_full_rank],
            [w_ppr,1-w_ppr],"ppr_plus_cbf")
    for w_pf in (.25,.50,.75):
        add_hybrid(f"personal_frequency_cbf_pf_weight_{w_pf:.2f}",["personal_frequency","cbf"],
            [personal_rankings,cbf_full_rank],[w_pf,1-w_pf],"personal_frequency_plus_cbf")
    for weights in FUSION_GRID["ppr_personal_frequency_plus_cbf"]:
        add_hybrid("ppr_pf_cbf_"+"_".join(f"{x:.2f}" for x in weights),
            ["ppr","personal_frequency","cbf"],[ppr_rankings,personal_rankings,cbf_full_rank],weights,"ppr_personal_frequency_plus_cbf")
    for weights in FUSION_GRID["ppr_personal_frequency_association_rules_plus_cbf"]:
        add_hybrid("ppr_pf_ar_cbf_"+"_".join(f"{x:.2f}" for x in weights),
            ["ppr","personal_frequency","association_rules","cbf"],
            [ppr_rankings,personal_rankings,ar_rankings,cbf_full_rank],weights,
            "ppr_personal_frequency_association_rules_plus_cbf")

    # Include prior baselines/hybrid in a common side-by-side table.
    print("Starting saved baseline comparison",flush=True)
    existing_specs=[("popularity_distinct_users",ROOT/"results"/"step4"/"popularity_distinct_users_predictions.csv","popularity"),
        ("personal_frequency_qualifying_airings",ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv","personal_frequency"),
        ("ppr_cosine_pfrequency_a0.35_k30",ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv","ppr"),
        ("association_rules_time_window_14d_lift_log",ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv","association_rules"),
        ("existing_ppr_personal_frequency_rrf_050",ROOT/"results"/"step8"/"validation_predictions"/"rrf_ppr_personal_frequency_ppr_weight_0.50"/"predictions.csv","ppr_personal_frequency_hybrid")]
    for config_id,path,family in existing_specs:
        print("Reading saved baseline",config_id,flush=True)
        records=list(csv.DictReader(path.open(newline="",encoding="utf-8")))
        preds=evaluation.parse_prediction_records(records)
        if family=="personal_frequency":diag={u:{"fallback_slots":personal_fallback_slots[u]} for u in cohort}
        elif family=="ppr":diag=ppr_infos
        elif family=="association_rules":diag={u:{"fallback_slots":ar_fallback_slots[u]} for u in cohort}
        else:diag=None
        row=metric_record(config_id,preds,protocol,secondary,fraction_truth,0.0,diag,{"family":family,"source":str(path.relative_to(ROOT))})
        row["predictions"]=preds
        experiments.append(row)
        print("Re-evaluated baseline",config_id,flush=True)
    print("Re-evaluated historical comparison baselines",flush=True)

    # Cross-check prior baseline rows against their saved measured validation scores.
    previous={row["config_id"]:row for row in experiments}
    expected={
        "personal_frequency_qualifying_airings":(.7876269841269841,.43413227513227515),
        "ppr_cosine_pfrequency_a0.35_k30":(.8151507936507936,.429489417989418),
        "association_rules_time_window_14d_lift_log":(.7713306878306878,.31364285714285717),
        "existing_ppr_personal_frequency_rrf_050":(.8075978835978836,.44040476190476185)}
    for cid,(primary_expected,secondary_expected) in expected.items():
        if not math.isclose(previous[cid]["primary_map_at_5"],primary_expected,abs_tol=1e-15):
            raise ValueError(f"Existing baseline primary score mismatch: {cid}")
        if not math.isclose(previous[cid]["secondary_map_at_5"],secondary_expected,abs_tol=1e-15):
            raise ValueError(f"Existing baseline secondary score mismatch: {cid}")

    # Only validation-selected candidates are eligible for final v3 consideration.
    eligible_new=[row for row in experiments if row["family"]=="cbf" or row["family"] in FUSION_GRID]
    selected=sorted(eligible_new,key=lambda row:(-row["secondary_map_at_5"],-row["primary_map_at_5"],
        row["prediction_runtime_seconds"],row["config_id"]))[0]
    print("Selected validation candidate",selected["config_id"],flush=True)
    if selected["family"]=="cbf":
        selected_feature=selected["feature_config"];selected_profile=selected["profile_weighting"]
        selected_space=cbf.FeatureSpace.fit(train_docs,cbf.FIELD_WEIGHT_CONFIGS[selected_feature])
        selected_candidate_matrix=selected_space.transform(candidate_docs,candidates,cbf.FIELD_WEIGHT_CONFIGS[selected_feature])
        selected_user_vectors,_=cbf.build_user_profiles(train.profiles,selected_space.training_items,
            {item:i for i,item in enumerate(selected_space.training_items)},selected_space.training_matrix,cohort,weighting=selected_profile)
        selected_rows,_=cbf.rank_candidates(selected_candidate_matrix,selected_user_vectors,candidates,fallback,batch_size=64)
        selected_rank={u:selected_rows[i] for i,u in enumerate(cohort)}
    else:
        if best_hybrid["record"]["config_id"]==selected["config_id"]:
            selected_rank=best_hybrid["ranking"]
        else:
            raise ValueError("Selected hybrid is not the retained best validation hybrid")
    require_new_v3=(selected["secondary_map_at_5"]>previous["existing_ppr_personal_frequency_rrf_050"]["secondary_map_at_5"]+1e-12)

    summary_rows=[]
    for row in experiments:
        summary_rows.append({k:row[k] for k in ("config_id","family","primary_map_at_5","secondary_map_at_5",
            "fraction80_top5_sensitivity_map_at_5","diagnostic_map_at_5_nonempty","hit_rate_at_5",
            "unique_recommended_programs","candidate_coverage_fraction","mean_pairwise_top5_item_overlap",
            "fraction_user_pairs_identical_ranked_list","fraction_user_pairs_sharing_at_least_4_items",
            "fraction_user_pairs_sharing_at_least_3_items","prediction_runtime_seconds","popularity_fallback_slots",
            "users_with_any_popularity_fallback","top10_recommended_program_counts")})
    with (OUT/"model_comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=list(summary_rows[0]);writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for row in summary_rows:
            writer.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    # The selected run's prediction file gets a stable config-independent name.
    write_predictions(OUT/"best_validation_candidate.csv",cohort,step8.top5_from_full_rankings(selected_rank))
    validation_results={"stage":"Content-Based Filtering and rank-fusion validation",
        "protocol_id":protocol.config["protocol_id"],"candidate_count":len(candidates),"cohort_users":len(cohort),
        "primary_definition":"frozen Step 3 all-distinct-programs with >=120-second airing truth; denominator min(5,|R|)",
        "secondary_definition":"frozen Step 5 per-user top-five qualifying airing truth",
        "unverified_fraction80_sensitivity":fraction_details,
        "selection_rule":"maximize frozen secondary Top-5 MAP@5; tie by higher frozen primary MAP@5, lower prediction runtime, config ID",
        "selection_uses_kaggle_score":False,"cbf_individual_configurations":len(cbf_rows),
        "new_hybrid_configurations":sum(row["family"] in FUSION_GRID for row in experiments),
        "cbf_best_individual_config_id":best_cbf["config_id"],"cbf_best_individual_primary_map_at_5":best_cbf["primary_map_at_5"],
        "cbf_best_individual_secondary_map_at_5":best_cbf["secondary_map_at_5"],
        "selected_candidate_config_id":selected["config_id"],"selected_candidate_family":selected["family"],
        "selected_primary_map_at_5":selected["primary_map_at_5"],"selected_secondary_map_at_5":selected["secondary_map_at_5"],
        "selected_fraction80_top5_sensitivity_map_at_5":selected["fraction80_top5_sensitivity_map_at_5"],
        "v3_supported_by_frozen_secondary_vs_existing_v2":require_new_v3,
        "v2_existing_primary_map_at_5":previous["existing_ppr_personal_frequency_rrf_050"]["primary_map_at_5"],
        "v2_existing_secondary_map_at_5":previous["existing_ppr_personal_frequency_rrf_050"]["secondary_map_at_5"],
        "feature_spaces":feature_build_meta,"experiments":summary_rows,
        "training_profile_source_sha256":train.training_sha256,
        "train_schedule_sha256":validation.sha256(SCHEDULE_TRAIN),"validation_schedule_sha256":validation.sha256(SCHEDULE_VALID),
        "validation_truth_sha256":protocol.config["truth"]["sha256"],
        "runtime_seconds":{"step9_total":time.monotonic()-start_all,"association_dataset_fit":ar_dataset.build_seconds,
            "association_pair_mining":ar_dataset.pair_mining_seconds,"ppr_graph_fit":ppr_graph.runtime_seconds},
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "environment":{"python":__import__("sys").version,"numpy":np.__version__,"scipy":__import__("scipy").__version__},
        "no_future_viewing_behavior_used":True,"no_other_participant_predictions_or_models_used":True,
        "official_metric":"MAP@{K}, K=5; official label construction not specified in public competition metadata",
        "reported_80pct_rule_officially_confirmed":False,"v3_generated":False}
    write_json(OUT/"step9_validation_results.json",validation_results)
    print(json.dumps({"cbf_best":best_cbf["config_id"],"cbf_primary":best_cbf["primary_map_at_5"],
        "cbf_secondary":best_cbf["secondary_map_at_5"],"selected_candidate":selected["config_id"],
        "selected_primary":selected["primary_map_at_5"],"selected_secondary":selected["secondary_map_at_5"],
        "v2_primary":validation_results["v2_existing_primary_map_at_5"],"v2_secondary":validation_results["v2_existing_secondary_map_at_5"],
        "v3_supported":require_new_v3,"cbf_configs":len(cbf_rows),"new_hybrids":validation_results["new_hybrid_configurations"],
        "fraction80_sensitivity":selected["fraction80_top5_sensitivity_map_at_5"],
        "runtime_seconds":validation_results["runtime_seconds"]["step9_total"],"peak_rss_mib":validation_results["peak_rss_mib"]},indent=2))


if __name__=="__main__":main()
