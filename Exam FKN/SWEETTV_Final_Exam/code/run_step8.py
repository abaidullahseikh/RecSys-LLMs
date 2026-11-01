"""Audit v1 and run a compact, historical-validation-only Step 8 comparison."""

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
import data_pipeline
import evaluation
import ppr
import personal_frequency
import popularity
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step8"
V1=ROOT/"submission"/"final_submission.csv"
FULL_HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"


def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def csv_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f:yield from csv.DictReader(f)


def load_v1():
    with V1.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=["user_id","tv_show_id"]:raise ValueError("v1 schema mismatch")
        rows=list(reader)
    result={r["user_id"]:tuple(r["tv_show_id"].split()) for r in rows}
    return rows,result


def schedule_pool(path,*,clip_to_test_period):
    lo,hi=validation.WINDOWS["test"]
    items=set()
    with Path(path).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError("schedule header mismatch")
        for row in reader:
            item=validation.schedule_program_id(row["tv_show_id"])
            if item=="0":continue
            start=data_pipeline.parse_timestamp(row["start_time"],schedule=True)
            duration=int(row["duration"])
            if duration<=0:continue
            if not clip_to_test_period or start<hi and start+duration>lo:items.add(item)
    return evaluation.candidate_pool(items)


def load_history():
    info=validation.json_read(validation.STEP2/"threshold_120s"/"interaction_manifest.json")
    expected=info["artifacts"][FULL_HISTORY.name]["sha256"]
    if digest(FULL_HISTORY)!=expected:raise ValueError("Weeks 11-30 history file hash mismatch")
    bounds=validation.WINDOWS["history"]
    profiles={};viewer=Counter();airs=Counter();seconds=Counter();rows=0
    with gzip.open(FULL_HISTORY,"rt",newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=validation.INTERACTION_COLUMNS:raise ValueError("history schema mismatch")
        for r in reader:
            u=validation.user_id(r["user_id"]);item=validation.program_id(r["tv_show_id"])
            count,secs,first,last=map(int,(r["qualifying_airings"],r["watched_seconds"],r["first_watch_utc"],r["last_watch_utc"]))
            if count<=0 or secs<count*120 or not bounds[0]<=first<last<=bounds[1]:raise ValueError("history threshold/time violation")
            profiles.setdefault(u,{})[item]=(count,secs,last)
            viewer[item]+=1;airs[item]+=count;seconds[item]+=secs;rows+=1
    nodes=tuple(sorted(viewer,key=int))
    model=ppr.TrainingProfiles(nodes,{i:n for n,i in enumerate(nodes)},profiles,dict(viewer),digest(FULL_HISTORY),bounds,rows,0)
    pop=popularity.PopularityModel((),dict(viewer),dict(airs),dict(seconds),digest(FULL_HISTORY),bounds,rows,0)
    return model,pop,info


def build_ppr_predictions(graph,data,cohort,candidates,fallback,personalization,alpha,*,collect_candidate_scores=True):
    pool=frozenset(candidates);fallback=tuple(fallback);tie={x:i for i,x in enumerate(fallback)}
    index={item:i for i,item in enumerate(graph.nodes)}
    predictions={};rankings={};infos={};candidate_scores={};selected_source={}
    for user in cohort:
        profile=data.profiles.get(user,{})
        if not profile:
            scores=None;ranked=[];info={"iterations":0,"converged":True,"l1_residual":0.0}
        else:
            restart=ppr.personalization_vector(profile,data,variant=personalization)
            scores,info=ppr.personalized_pagerank(graph,restart,restart_probability=alpha,tolerance=1e-8,max_iterations=250)
            ranked=[item for item in candidates if item in index and scores[index[item]]>0]
            ranked.sort(key=lambda item:(-scores[index[item]],tie.get(item,len(tie)),int(item)))
            if collect_candidate_scores:
                for item in candidates:
                    candidate_scores.setdefault(item,[]).append(float(scores[index[item]]) if item in index else 0.0)
        if collect_candidate_scores and scores is None:
            for item in candidates:candidate_scores.setdefault(item,[]).append(0.0)
        chosen=ranked[:5];from_ppr=len(chosen);seen=set(chosen)
        for item in fallback:
            if len(chosen)>=5:break
            if item in pool and item not in seen:chosen.append(item);seen.add(item)
        if len(chosen)!=5:raise AssertionError("PPR completion failure")
        ranked_seen=set(ranked)
        predictions[user]=tuple(chosen);rankings[user]=tuple(ranked+[x for x in fallback if x not in ranked_seen])
        infos[user]={**info,"ppr_ranked_slots":from_ppr,"fallback_slots":5-from_ppr,
                     "ppr_positive_candidates":len(ranked),"cold_start":not bool(profile)}
        selected_source[user]=tuple(["ppr"]*from_ppr+["popularity_fallback"]*(5-from_ppr))
    return predictions,rankings,infos,candidate_scores,selected_source


def top5_from_full_rankings(rankings):return {u:tuple(r[:5]) for u,r in rankings.items()}


def complete_scored(scores,candidates,fallback,tie_break=None):
    pool=set(candidates);fallback=tuple(fallback);tie_break=tie_break or {}
    ranked=sorted((i for i,s in scores.items() if i in pool and s>0),
                  key=lambda i:(-scores[i],tie_break.get(i,len(tie_break)),int(i)))
    out=list(ranked);seen=set(out)
    out.extend(i for i in fallback if i in pool and i not in seen)
    return tuple(out)


def score_map_rankings(score_maps,candidates,fallback,tie_break=None):
    return {u:complete_scored(scores,candidates,fallback,tie_break) for u,scores in score_maps.items()}


def full_personal_rankings(personal_model,cohort,candidates,fallback):
    result={};available=set(candidates);fallback_slots={}
    for user in cohort:
        values=personal_model.user_airings.get(user,{})
        eligible=[x for x in values if x in available]
        ranked=sorted(eligible,key=lambda x:(-values[x],int(x)))
        seen=set(ranked);result[user]=tuple(ranked+[x for x in fallback if x not in seen])
        fallback_slots[user]=max(0,5-len(eligible))
    return result,fallback_slots


def full_ar_rankings(model,profiles,cohort,candidates,fallback,*,score_type,weighting):
    result={};fallback_slots={}
    for user in cohort:
        scores,_=ar.score_user(model,profiles.get(user,{}),candidates,score_type=score_type,antecedent_weight=weighting)
        ranked=sorted(scores,key=lambda x:(-scores[x],-model.dataset.item_support.get(x,0),int(x)))
        seen=set(ranked);result[user]=tuple(ranked+[x for x in fallback if x not in seen])
        fallback_slots[user]=max(0,5-len(ranked))
    return result,fallback_slots


def rank_position_matrix(rankings,cohort,candidates):
    item_index={item:i for i,item in enumerate(candidates)}
    result=np.empty((len(cohort),len(candidates)),dtype=np.int32)
    positions=np.arange(len(candidates),dtype=np.int32)
    for r,user in enumerate(cohort):
        order=np.fromiter((item_index[x] for x in rankings[user]),dtype=np.int32,count=len(candidates))
        result[r,order]=positions
    return result


def rrf_from_positions(first_positions,second_positions,cohort,candidates,fallback,*,first_weight=.5,constant=60):
    out={};fallback_rank={item:i for i,item in enumerate(fallback)}
    tie=np.asarray([fallback_rank[x] for x in candidates],dtype=np.int32)
    numeric=np.asarray([int(x) for x in candidates],dtype=np.int64)
    for row,user in enumerate(cohort):
        scores=first_weight/(constant+first_positions[row])+(1-first_weight)/(constant+second_positions[row])
        order=np.lexsort((numeric,tie,-scores))
        out[user]=tuple(candidates[i] for i in order)
    return out


def popularity_positions(fallback,candidates):
    order={item:i for i,item in enumerate(fallback)}
    return np.asarray([order[item] for item in candidates],dtype=np.int32)


def popularity_rrf(first_positions,base_pop_positions,cohort,candidates,fallback,*,first_weight=.5,constant=60):
    return rrf_from_positions(first_positions,np.broadcast_to(base_pop_positions,first_positions.shape),cohort,candidates,
                              fallback,first_weight=first_weight,constant=constant)


def blend_ppr_pop_rankings(graph,data,cohort,candidates,fallback,pop_scores,*,pop_weight,personalization="frequency",alpha=.35):
    out={};pool=tuple(candidates);index={x:i for i,x in enumerate(graph.nodes)}
    pop_total=math.fsum(pop_scores.get(x,0) for x in pool);fallback_rank={x:i for i,x in enumerate(fallback)}
    tie=np.asarray([fallback_rank[x] for x in pool],dtype=np.int32)
    numeric=np.asarray([int(x) for x in pool],dtype=np.int64)
    candidate_idx=np.asarray([index.get(x,-1) for x in pool],dtype=np.int32)
    pop=np.asarray([pop_scores.get(x,0)/pop_total if pop_total else 0 for x in pool],dtype=np.float64)
    for user in cohort:
        profile=data.profiles.get(user,{})
        if not profile:out[user]=tuple(fallback);continue
        restart=ppr.personalization_vector(profile,data,variant=personalization)
        score,_=ppr.personalized_pagerank(graph,restart,restart_probability=alpha,tolerance=1e-8,max_iterations=250)
        pvalues=np.zeros(len(pool),dtype=np.float64);available=candidate_idx>=0
        pvalues[available]=score[candidate_idx[available]]
        ptotal=float(pvalues.sum()) or 1.0
        blended=(1-pop_weight)*(pvalues/ptotal)+pop_weight*pop
        order=np.lexsort((numeric,tie,-blended))
        out[user]=tuple(pool[i] for i in order)
    return out


def pairwise_diversity(predictions,cohort):
    users=list(cohort);n=len(users);total_pairs=n*(n-1)//2
    item_counts=Counter(x for u in users for x in predictions[u])
    overlap_sum=sum(c*(c-1)//2 for c in item_counts.values())
    exact=Counter(tuple(predictions[u]) for u in users)
    identical=sum(c*(c-1)//2 for c in exact.values())
    sets=Counter(tuple(sorted(predictions[u],key=int)) for u in users)
    same_set=sum(c*(c-1)//2 for c in sets.values())
    subset4=Counter();subset3=Counter();subset2=Counter()
    from itertools import combinations
    for u in users:
        ids=tuple(sorted(set(predictions[u]),key=int))
        subset4.update(combinations(ids,4));subset3.update(combinations(ids,3));subset2.update(combinations(ids,2))
    p4=sum(c*(c-1)//2 for c in subset4.values())
    p3=sum(c*(c-1)//2 for c in subset3.values())
    p2=sum(c*(c-1)//2 for c in subset2.values())
    exact5=same_set;exact4=p4-5*exact5;exact3=p3-4*exact4-10*exact5
    exact2=p2-3*exact3-6*exact4-10*exact5
    exact1=overlap_sum-2*exact2-3*exact3-4*exact4-5*exact5
    exact0=total_pairs-exact1-exact2-exact3-exact4-exact5
    atleast4=exact4+exact5;atleast3=exact3+exact4+exact5
    unique={x for u in users for x in predictions[u]}
    return {"unique_recommended_programs":len(unique),"candidate_coverage_fraction":0,
        "mean_pairwise_top5_item_overlap":overlap_sum/total_pairs if total_pairs else 0,
        "fraction_user_pairs_identical_ranked_list":identical/total_pairs if total_pairs else 0,
        "fraction_user_pairs_sharing_at_least_4_items":atleast4/total_pairs if total_pairs else 0,
        "fraction_user_pairs_sharing_at_least_3_items":atleast3/total_pairs if total_pairs else 0,
        "users_identical_to_first_user_list":sum(1 for u in users if tuple(predictions[u])==tuple(predictions[users[0]])),
        "users_identical_to_first_user_fraction":sum(1 for u in users if tuple(predictions[u])==tuple(predictions[users[0]]))/n}


def evaluate_variant(config_id,rankings,protocol,secondary,elapsed,diagnostics=None):
    preds=top5_from_full_rankings(rankings)
    primary=protocol.evaluate(preds)
    secondary_eval=evaluation.evaluate_predictions(preds,secondary,protocol.cohort,protocol.candidates,strict=True)
    freq=Counter(x for u in protocol.cohort for x in preds[u])
    diversity=pairwise_diversity(preds,protocol.cohort)
    diversity["candidate_coverage_fraction"]=len(freq)/len(protocol.candidates)
    fallback_slots=sum((diagnostics or {}).get(u,{}).get("fallback_slots",0) for u in protocol.cohort)
    primary_items=Counter(x for u in protocol.cohort for x in preds[u])
    return {"config_id":config_id,"primary_map_at_5":primary["primary_map_at_5"],
        "secondary_map_at_5":secondary_eval["primary_map_at_5"],
        "diagnostic_map_at_5_nonempty":primary["diagnostic_map_at_5_nonempty"],
        "hit_rate_at_5":sum(bool(set(preds[u])&protocol.truth[u]) for u in protocol.cohort)/len(protocol.cohort),
        "candidate_count":len(protocol.candidates),**diversity,"prediction_runtime_seconds":elapsed,
        "popularity_fallback_slots":fallback_slots,
        "users_with_any_popularity_fallback":sum((diagnostics or {}).get(u,{}).get("fallback_slots",0)>0 for u in protocol.cohort),
        "top10_recommended_program_counts":freq.most_common(10),"predictions":preds}


def norm_stats(values):
    if not values:return {"count":0}
    mean=math.fsum(values)/len(values);var=math.fsum((x-mean)**2 for x in values)/len(values)
    return {"count":len(values),"mean":mean,"std":math.sqrt(var),"min":min(values),"median":sorted(values)[len(values)//2],"max":max(values),
        "coefficient_of_variation":math.sqrt(var)/mean if mean else None}


def audit_v1(recomputed,infos,candidate_scores,selected_source,pool,all_pool):
    saved_rows,saved=load_v1()
    with (validation.ROOT.parent/"sweettv-tv-program-recommender"/"submission.csv").open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream);header=next(reader,None)
        if header!=["user_id","tv_show_id"]:raise ValueError("Original Kaggle template schema mismatch")
        template=tuple(validation.user_id(r[0]) for r in reader)
    require_order=tuple(saved)==tuple(template)
    if not require_order:raise ValueError("v1 order differs from original template")
    if any(saved[u]!=recomputed[u] for u in saved):raise ValueError("v1 predictions do not reproduce from the recorded PPR pipeline")
    freq=Counter(i for u in saved for i in saved[u])
    rank_counts=Counter(saved.values())
    source_counts=Counter(x for u in saved for x in selected_source[u])
    allpairs=len(saved)*(len(saved)-1)//2;users=list(saved)
    item_counts=Counter(x for u in users for x in saved[u])
    # Recover exact and near-identical unordered-set overlap efficiently from shared subsets.
    from itertools import combinations
    exact_rank=Counter(saved.values());identical=sum(c*(c-1)//2 for c in exact_rank.values())
    same_set=Counter(tuple(sorted(saved[u],key=int)) for u in users)
    same_set_pairs=sum(c*(c-1)//2 for c in same_set.values())
    sub4=Counter();sub3=Counter();sub2=Counter()
    for u in users:
        ids=tuple(sorted(set(saved[u]),key=int));sub4.update(combinations(ids,4));sub3.update(combinations(ids,3));sub2.update(combinations(ids,2))
    p4=sum(c*(c-1)//2 for c in sub4.values());p3=sum(c*(c-1)//2 for c in sub3.values());p2=sum(c*(c-1)//2 for c in sub2.values())
    exact5=same_set_pairs;exact4=p4-5*exact5;exact3=p3-4*exact4-10*exact5
    exact2=p2-3*exact3-6*exact4-10*exact5
    exact1=sum(c*(c-1)//2 for c in item_counts.values())-2*exact2-3*exact3-4*exact4-5*exact5
    exact0=allpairs-exact1-exact2-exact3-exact4-exact5
    overlap_counts=Counter({0:exact0,1:exact1,2:exact2,3:exact3,4:exact4,5:exact5})
    gt4=exact4+exact5;gt3=exact3+exact4+exact5
    score_variance={}
    for item in ("2400480","240081","2400508","6500479","2400467"):
        vals=candidate_scores.get(item,[])
        score_variance[item]=norm_stats(vals)
    per_user_ppr_counts=Counter(infos[u]["ppr_ranked_slots"] for u in users)
    per_user_fallback=Counter(infos[u]["fallback_slots"] for u in users)
    out={"v1_submission_sha256":digest(V1),"v1_reproduced_exactly_from_final_training_pipeline":True,
        "template_users":len(saved),"v1_unique_recommended_programs":len(freq),"candidate_pool_v1":len(pool),
        "valid_nonzero_ids_in_requested_test_schedule":len(all_pool),"candidate_pool_ids_omitted_by_v1":sorted(set(all_pool)-set(pool),key=int),
        "ppr_ranked_recommendation_slots":source_counts["ppr"],"popularity_fallback_recommendation_slots":source_counts["popularity_fallback"],
        "ppr_ranked_slots_per_user_distribution":dict(sorted(per_user_ppr_counts.items())),
        "fallback_slots_per_user_distribution":dict(sorted(per_user_fallback.items())),
        "item_frequency_top20":[{"program_id":i,"count":n,"user_fraction":n/len(saved)} for i,n in freq.most_common(20)],
        "unique_complete_rankings":len(rank_counts),"most_common_complete_rankings":[{"ranked_ids":list(x),"users":n} for x,n in rank_counts.most_common(10)],
        "identical_user_pairs":identical,"fraction_user_pairs_identical":identical/allpairs,
        "fraction_user_pairs_sharing_at_least_4_items":gt4/allpairs,
        "fraction_user_pairs_sharing_at_least_3_items":gt3/allpairs,
        "pairwise_top5_item_overlap_distribution":{str(k):v for k,v in sorted(overlap_counts.items())},
        "ppr_scores_across_users_for_common_programs":score_variance,
        "candidate_coverage_count":len(freq),"candidate_coverage_fraction":len(freq)/len(pool),
        "cold_start_users":sum(infos[u]["cold_start"] for u in users),
        "original_submission_program_ids_all_valid_within_v1_pool":all(i in set(pool) for u in saved for i in saved[u])}
    return out


def main():
    if OUT.exists() and not (OUT/"v1_submission_diagnostics.json").is_file():
        raise FileExistsError("Preserve existing Step 8 diagnostics; use a fresh output location")
    if not OUT.exists():OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();protocol=validation.EvaluationProtocol();cohort=protocol.cohort;candidates=protocol.candidates
    secondary={}
    for u,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
        ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):secondary[u]=frozenset(field.split()) if field else frozenset()
    if tuple(secondary)!=cohort:raise ValueError("Secondary validation cohort mismatch")

    # Preserve and reuse the completed v1 audit if this long comparison is resumed.
    if (OUT/"v1_submission_diagnostics.json").is_file():
        v1_audit=validation.json_read(OUT/"v1_submission_diagnostics.json")
    else:
        full_data,full_pop,history_info=load_history()
        print("Loaded verified full-history profiles",full_data.rows,flush=True)
        schedule=ROOT.parent/"sweettv-tv-program-recommender"/"export_arh_31-42-final.csv"
        v1_pool=schedule_pool(schedule,clip_to_test_period=True)
        all_v2_pool=schedule_pool(schedule,clip_to_test_period=False)
        v1_fallback=full_pop.__class__(v1_pool,full_pop.distinct_users,full_pop.qualifying_airings,full_pop.watched_seconds,
            full_pop.training_sha256,full_pop.training_window_utc,full_pop.training_rows,0).ranking("distinct_users")
        full_graph=ppr.build_graph(full_data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
        print("Fitted full-history PPR graph",full_graph.statistics()["nodes"],flush=True)
        v1_preds,v1_rankings,v1_infos,v1_scores,v1_sources=build_ppr_predictions(full_graph,full_data,tuple(load_v1()[1]),v1_pool,v1_fallback,"frequency",.35)
        print("Reproduced v1 PPR predictions",flush=True)
        v1_audit=audit_v1(v1_preds,v1_infos,v1_scores,v1_sources,v1_pool,all_v2_pool)
        v1_audit["full_history_training_rows"]=full_data.rows
        v1_audit["full_history_training_users"]=len(full_data.profiles)
        v1_audit["full_history_program_nodes"]=len(full_data.nodes)
        v1_audit["graph_statistics"]=full_graph.statistics()
        v1_audit["graph_fit_runtime_seconds"]=full_graph.runtime_seconds
        write_json(OUT/"v1_submission_diagnostics.json",v1_audit)

    # Reproduce the exact Step 3 training split and build model objects once.
    train=ppr.build_training_profiles(protocol.config)
    pop=popularity.fit_popularity(protocol.config,candidates);fallback=pop.ranking("distinct_users")
    personal=personal_frequency.fit_personal_frequency(protocol.config,candidates)
    graph=ppr.build_graph(train,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    base_start=time.monotonic()
    base_preds,base_rankings,base_infos,_,_=build_ppr_predictions(graph,train,cohort,candidates,fallback,"frequency",.35,collect_candidate_scores=False)
    print("Recreated selected validation PPR",flush=True)
    base_seconds=time.monotonic()-base_start
    saved_ppr={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv_rows(ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv")}
    if base_preds!=saved_ppr:raise ValueError("Reconstructed selected validation PPR differs from saved experiment")
    metrics=[]
    base_record=evaluate_variant("ppr_original_best_cosine_frequency_a035_k30",base_rankings,protocol,secondary,base_seconds,base_infos)
    metrics.append(base_record)
    sensitivity_path=ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k50"/"predictions.csv"
    sensitivity_preds={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv_rows(sensitivity_path)}
    sensitivity_rankings={u:tuple(sensitivity_preds[u])+tuple(x for x in fallback if x not in set(sensitivity_preds[u])) for u in cohort}
    sensitivity_meta=next(r for r in csv_rows(ROOT/"results"/"step5"/"ppr_experiments.csv") if r["config_id"]=="cosine_pfrequency_a0.35_k50")
    graph50=ppr.build_graph(train,weight_type="cosine",max_neighbors=50,history_cap=40,min_common_viewers=3)
    reconstructed50,_,sensitivity_infos,_,_=build_ppr_predictions(graph50,train,cohort,candidates,fallback,"frequency",.35,collect_candidate_scores=False)
    if reconstructed50!=sensitivity_preds:raise ValueError("Saved PPR sensitivity winner does not reproduce")
    metrics.append(evaluate_variant("ppr_cosine_frequency_a035_k50_saved_sensitivity_winner",sensitivity_rankings,
                                    protocol,secondary,float(sensitivity_meta["prediction_generation_seconds"]),
                                    sensitivity_infos))

    t=time.monotonic();personal_rankings,personal_fallback=full_personal_rankings(personal,cohort,candidates,fallback)
    saved_personal={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv_rows(ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv")}
    if top5_from_full_rankings(personal_rankings)!=saved_personal:raise ValueError("Reconstructed personal-frequency baseline differs from saved Step 4 output")
    metrics.append(evaluate_variant("personal_frequency_qualifying_airings",personal_rankings,protocol,secondary,time.monotonic()-t,
        {u:{"fallback_slots":personal_fallback[u]} for u in cohort}))

    # Build the selected Step 6 association rules from only the same Week 11-20 split.
    ar_profiles,source,profile_rows=__import__("run_step6").build_training_profiles(protocol.config)
    ds=ar.build_window_dataset(protocol.config,14)
    print("Built 14-day association dataset",flush=True)
    ar_model=ar.mine_rules(ds,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.0,top_neighbors=50)
    print("Mined selected association rules",ar_model.rule_count,flush=True)
    ar_start=time.monotonic();ar_rankings,ar_fallback=full_ar_rankings(ar_model,ar_profiles,cohort,candidates,fallback,
        score_type="confidence_log_lift",weighting="log_frequency")
    print("Scored association-rule rankings",flush=True)
    saved_ar={r["user_id"]:tuple(r["tv_show_id"].split()) for r in csv_rows(ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv")}
    if top5_from_full_rankings(ar_rankings)!=saved_ar:raise ValueError("Reconstructed selected Association Rules variant differs from saved Step 6 output")
    metrics.append(evaluate_variant("association_rules_time_window14_lift_log",ar_rankings,protocol,secondary,time.monotonic()-ar_start,
        {u:{"fallback_slots":ar_fallback[u]} for u in cohort}))

    # PPR adjusted personalization: small extension around saved frequency winner.
    for variant in ("uniform","duration","recency"):
        t=time.monotonic()
        _,rankings,infos,_,_=build_ppr_predictions(graph,train,cohort,candidates,fallback,variant,.35,collect_candidate_scores=False)
        cid=f"ppr_personalization_{variant}_a035_k30"
        metrics.append(evaluate_variant(cid,rankings,protocol,secondary,time.monotonic()-t,infos))

    # Apply a normalized PPR/probability blend one weight at a time to bound memory.
    pop_norm=dict(pop.distinct_users)
    for beta in (.05,.15,.30):
        t=time.monotonic();blended=blend_ppr_pop_rankings(graph,train,cohort,candidates,fallback,pop_norm,pop_weight=beta)
        metrics.append(evaluate_variant(f"ppr_probability_popularity_blend_{beta:.2f}",blended,protocol,secondary,time.monotonic()-t))
    print("Completed PPR/popularity score blends",flush=True)

    # Rank fusion variants all use complete, deterministic lists and same validation pool.
    base_positions=rank_position_matrix(base_rankings,cohort,candidates)
    personal_positions=rank_position_matrix(personal_rankings,cohort,candidates)
    ar_positions=rank_position_matrix(ar_rankings,cohort,candidates)
    for weight in (.50,.75):
        t=time.monotonic();rrf=rrf_from_positions(base_positions,personal_positions,cohort,candidates,fallback,first_weight=weight)
        metrics.append(evaluate_variant(f"rrf_ppr_personal_frequency_ppr_weight_{weight:.2f}",rrf,protocol,secondary,time.monotonic()-t))
    for weight in (.50,.75):
        t=time.monotonic();rrf=rrf_from_positions(base_positions,ar_positions,cohort,candidates,fallback,first_weight=weight)
        metrics.append(evaluate_variant(f"rrf_ppr_association_rules_ppr_weight_{weight:.2f}",rrf,protocol,secondary,time.monotonic()-t))

    # Simple popularity-aware rank fusion, separate from probability-score blending.
    pop_positions=popularity_positions(fallback,candidates)
    for weight in (.25,.50):
        t=time.monotonic();rrf=popularity_rrf(base_positions,pop_positions,cohort,candidates,fallback,first_weight=weight)
        metrics.append(evaluate_variant(f"rrf_ppr_global_popularity_ppr_weight_{weight:.2f}",rrf,protocol,secondary,time.monotonic()-t))
    print("Completed validation rank-fusion variants",flush=True)

    # Prespecified selection: maximize fixed secondary Top-5 sensitivity MAP;
    # break ties by primary MAP, then prediction time, then config ID.
    selected=sorted(metrics,key=lambda r:(-r["secondary_map_at_5"],-r["primary_map_at_5"],r["prediction_runtime_seconds"],r["config_id"]))[0]
    compact=[]
    for row in metrics:
        compact.append({k:row[k] for k in ("config_id","primary_map_at_5","secondary_map_at_5","diagnostic_map_at_5_nonempty",
            "hit_rate_at_5","unique_recommended_programs","candidate_coverage_fraction","mean_pairwise_top5_item_overlap",
            "fraction_user_pairs_identical_ranked_list","fraction_user_pairs_sharing_at_least_4_items",
            "fraction_user_pairs_sharing_at_least_3_items","users_identical_to_first_user_fraction",
            "prediction_runtime_seconds","popularity_fallback_slots","users_with_any_popularity_fallback","top10_recommended_program_counts")})
    with (OUT/"validation_model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(compact[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for r in compact:
            w.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    for row in metrics:
        folder=OUT/"validation_predictions"/row["config_id"];folder.mkdir(parents=True,exist_ok=True)
        with (folder/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f,lineterminator="\n");w.writerow(["user_id","tv_show_id"])
            for u in cohort:w.writerow([u," ".join(row["predictions"][u])])
    # Full baseline and PPR score distribution files support the v1 diagnosis.
    with (OUT/"v1_item_frequency.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(["tv_show_id","users","fraction_of_users"])
        _,v1_saved=load_v1()
        for item,count in Counter(x for u in v1_saved for x in v1_saved[u]).most_common():w.writerow([item,count,count/len(v1_saved)])
    result={"stage":"Step 8 Kaggle score diagnosis and validation-only optimization",
        "selection_rule":"maximum pre-existing secondary Top-5 sensitivity MAP@5; tie-break primary all-qualifying MAP@5, lower prediction runtime, then config ID",
        "selection_uses_kaggle_score":False,"kaggle_scores_used_for_selection":False,
        "user_reported_kaggle_private_map_at_5":0.37269,"user_reported_kaggle_public_map_at_5":0.39316,
        "validation_protocol_id":protocol.config["protocol_id"],"validation_candidates":len(candidates),
        "primary_truth":"all distinct qualifying Weeks 21-30 programs; empty-set AP=0; denominator min(5,|R|)",
        "secondary_truth":"per-user five most frequent qualifying Weeks 21-30 programs; existing frozen Step 5 sensitivity truth",
        "selected_config_id":selected["config_id"],"selected_primary_map_at_5":selected["primary_map_at_5"],
        "selected_secondary_map_at_5":selected["secondary_map_at_5"],"selected_model_predictions_sha256":digest(OUT/"validation_predictions"/selected["config_id"]/"predictions.csv"),
        "variants":compact,"v1_audit":v1_audit,
        "runtime_seconds":{"validation_ppr_graph_fit":graph.runtime_seconds,"association_dataset_build":ds.build_seconds,
            "association_pair_mining":ds.pair_mining_seconds,"total_step8_seconds":time.monotonic()-started},
        "environment":{"python":__import__("sys").version,"numpy":__import__("numpy").__version__,
            "scipy":__import__("scipy").__version__,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},
        "no_hidden_labels_or_other_participant_predictions_used":True,"no_test_viewing_behavior_used":True}
    write_json(OUT/"step8_experiment_results.json",result)
    print(json.dumps({"selected_config_id":result["selected_config_id"],"selected_primary_MAP@5":result["selected_primary_map_at_5"],
        "selected_secondary_MAP@5":result["selected_secondary_map_at_5"],"variants":len(compact),
        "v1_unique_items":v1_audit["v1_unique_recommended_programs"],"v1_ppr_slots":v1_audit["ppr_ranked_recommendation_slots"],
        "v1_fallback_slots":v1_audit["popularity_fallback_recommendation_slots"],
        "candidate_pool_v1":v1_audit["candidate_pool_v1"],"valid_file_pool":v1_audit["valid_nonzero_ids_in_requested_test_schedule"],
        "runtime_seconds":result["runtime_seconds"]["total_step8_seconds"]},indent=2))


if __name__=="__main__":main()
