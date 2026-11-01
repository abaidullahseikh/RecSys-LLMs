"""Stage/run bounded Association Rules experiments on the frozen Step 3 protocol."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import scipy

import association_rules as ar
import data_pipeline
import evaluation
import popularity
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step6"
BASE_FILTER={"min_support":0.0005,"min_confidence":0.01,"min_cooccurrence":3,"min_lift":1.0,"top_neighbors":50}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n")


def save_predictions(path,cohort,predictions):
    if tuple(predictions)!=tuple(cohort):
        raise ValueError("Association predictions must preserve evaluation cohort order")
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["user_id","tv_show_id"])
        for user in cohort:
            ids=tuple(evaluation.program_id(p) for p in predictions[user])
            if len(ids)!=5 or len(set(ids))!=5:
                raise ValueError(f"Invalid strict Top-5 for user {user}")
            writer.writerow((user," ".join(ids)))


def build_training_profiles(config):
    source=validation.training_input(config)
    profiles=defaultdict(dict)
    count=0
    for row in source.rows():
        profiles[row.user_id][row.program_id]=(row.qualifying_airings,row.watched_seconds,row.last_watch_utc)
        count+=1
    return {u:dict(items) for u,items in profiles.items()},source,count


def load_user_history_dataset(config, profiles):
    started=time.monotonic()
    transactions=tuple((u,ar.canonical_transaction(items.keys())) for u,items in sorted(profiles.items()))
    build_seconds=time.monotonic()-started
    source=validation.training_input(config)
    return ar.build_dataset(transactions,transaction_type="user_history",window_days=None,
        training_sha256=source.sha256,training_window_utc=source.window_utc,
        source_sha256=source.sha256,transaction_build_seconds=build_seconds)


def transaction_summary(dataset):
    sizes=[len(tx[-1]) for tx in dataset.transactions]
    return {"transaction_type":dataset.transaction_type,"window_days":dataset.window_days,
        "transactions":dataset.transaction_count,"distinct_programs":len(dataset.index_item),
        "item_support_total":sum(dataset.item_support.values()),"pair_occurrences":dataset.pair_occurrences,
        "distinct_pairs":len(dataset.pair_counts),
        "average_transaction_size":dataset.average_transaction_size,
        "median_transaction_size":float(np.median(sizes)) if sizes else 0,
        "p95_transaction_size":dataset.p95_transaction_size,"max_transaction_size":dataset.max_transaction_size,
        "transaction_build_seconds":dataset.build_seconds,"sparse_pair_mining_seconds":dataset.pair_mining_seconds,
        "source_sha256":dataset.source_sha256}


def evaluate_variant(protocol,dataset,profiles,pop_model,config_id,params,score_type,antecedent_weight,
                     training_seconds=0,training_interaction_rows=0):
    rule_filters={k:params[k] for k in ("min_support","min_confidence","min_cooccurrence","min_lift","top_neighbors")}
    model=ar.mine_rules(dataset,**rule_filters)
    fallback=pop_model.ranking("distinct_users")
    predictions,diagnostics={},{}
    started=time.monotonic()
    for user in protocol.cohort:
        predictions[user],diagnostics[user]=ar.recommend_user(
            model,profiles.get(user,{}),protocol.candidates,fallback,
            score_type=score_type,antecedent_weight=antecedent_weight)
    prediction_seconds=time.monotonic()-started
    evaluated=protocol.evaluate(predictions)
    rule_covered=sum(diagnostics[u]["matched_rule_user"] for u in protocol.cohort)
    distinct_recs=set(p for rec in predictions.values() for p in rec)
    hits=sum(bool(set(predictions[u])&protocol.truth[u]) for u in protocol.cohort)
    warm=[u for u in protocol.cohort if profiles.get(u)]
    cold=[u for u in protocol.cohort if not profiles.get(u)]
    stats={"config_id":config_id,"transaction_type":dataset.transaction_type,"window_days":dataset.window_days,
        **params,"score_type":score_type,"antecedent_weight":antecedent_weight,
        "transactions":dataset.transaction_count,"average_transaction_size":dataset.average_transaction_size,
        "p95_transaction_size":dataset.p95_transaction_size,"max_transaction_size":dataset.max_transaction_size,
        "pair_occurrences":dataset.pair_occurrences,"distinct_training_programs":len(dataset.index_item),
        "distinct_raw_pairs":len(dataset.pair_counts),"directed_rules_retained":model.rule_count,
        "antecedents_with_rules":len(model.rules_by_antecedent),
        "rule_coverage_users":rule_covered,"rule_coverage_fraction":rule_covered/len(protocol.cohort),
        "primary_map_at_5":evaluated["primary_map_at_5"],
        "diagnostic_map_at_5_nonempty":evaluated["diagnostic_map_at_5_nonempty"],
        "hit_rate_at_5":hits/len(protocol.cohort),"users_with_any_hit":hits,
        "warm_primary_map_at_5":math.fsum(evaluated["per_user_ap_at_5"][u] for u in warm)/len(warm) if warm else None,
        "cold_primary_map_at_5":math.fsum(evaluated["per_user_ap_at_5"][u] for u in cold)/len(cold) if cold else None,
        "cold_start_users":len(cold),"cold_start_users_using_fallback":sum(diagnostics[u]["cold_start"] for u in protocol.cohort),
        "popularity_backfill_users":sum(diagnostics[u]["fallback_slots"]>0 for u in protocol.cohort),
        "popularity_backfill_slots":sum(diagnostics[u]["fallback_slots"] for u in protocol.cohort),
        "distinct_recommended_programs":len(distinct_recs),"candidate_count":len(protocol.candidates),
        "candidate_coverage_fraction":len(distinct_recs)/len(protocol.candidates),
        "training_interaction_rows":training_interaction_rows,
        "transaction_item_occurrences":sum(dataset.item_support.values()),
        "transaction_build_seconds":dataset.build_seconds,"pair_mining_seconds":dataset.pair_mining_seconds,
        "rule_mining_seconds":model.build_seconds,"rule_training_runtime_seconds":training_seconds+model.build_seconds,
        "prediction_runtime_seconds":prediction_seconds,
        "evaluation_config_sha256":protocol.config_sha256,"protocol_id":protocol.config["protocol_id"],
        "training_sha256":dataset.training_sha256,"transaction_source_sha256":dataset.source_sha256,
        "candidate_ids_sha256":evaluation.identity_hash(protocol.candidates),
        "truth_sha256":protocol.config["truth"]["sha256"],
        "per_user_ap_at_5":evaluated["per_user_ap_at_5"],
        "user_predictions":predictions,"user_diagnostics":diagnostics,"model":model}
    return stats


def write_variant_artifacts(stats,protocol):
    folder=OUT/"experiments"/stats["config_id"]
    folder.mkdir(parents=True,exist_ok=True)
    prediction_path=folder/"predictions.csv"
    save_predictions(prediction_path,protocol.cohort,stats["user_predictions"])
    ap_path=folder/"user_ap.csv"
    with ap_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["user_id","ap_at_5","validation_relevant_count","used_fallback_slots","rule_coverage"])
        for user in protocol.cohort:
            writer.writerow((user,format(stats["per_user_ap_at_5"][user],".17g"),len(protocol.truth[user]),
                             stats["user_diagnostics"][user]["fallback_slots"],
                             int(stats["user_diagnostics"][user]["matched_rule_user"])))
    rules_path=folder/"rules.csv"
    with rules_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["antecedent_id","consequent_id","transaction_support_count","support","confidence","lift"])
        for antecedent in sorted(stats["model"].rules_by_antecedent,key=int):
            for rule in stats["model"].rules_by_antecedent[antecedent]:
                writer.writerow((antecedent,rule["consequent"],rule["pair_count"],format(rule["support"],".17g"),
                                 format(rule["confidence"],".17g"),format(rule["lift"],".17g")))
    serial={k:v for k,v in stats.items() if k not in {"per_user_ap_at_5","user_predictions","user_diagnostics","model"}}
    serial["prediction_file"]=str(prediction_path.relative_to(ROOT)); serial["prediction_sha256"]=validation.sha256(prediction_path)
    serial["user_ap_file"]=str(ap_path.relative_to(ROOT)); serial["user_ap_sha256"]=validation.sha256(ap_path)
    serial["rules_file"]=str(rules_path.relative_to(ROOT)); serial["rules_sha256"]=validation.sha256(rules_path)
    serial["popularity_fallback_policy"]=evaluation.FALLBACK_POLICY
    return serial


def secondary_truth(config,protocol):
    """Reuse Step 5's fixed secondary-label definition, never tune/redefine it here."""
    path=OUT.parent/"step5"/"secondary_top5_truth.csv"
    if not path.is_file():
        raise FileNotFoundError("Saved Step 5 secondary Top-5 sensitivity truth is required")
    if validation.sha256(path)!="440c61d96a78dcf346ac167729a6440f313e4980c1beee661c7d6ee9ced80d4f":
        raise ValueError("Step 5 fixed secondary sensitivity truth hash mismatch")
    truth={}
    for user,field,count,tie in validation.records(path,["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        items=tuple(field.split()) if field else ()
        if len(items)!=int(count) or tie!="qualifying_airings_desc,numeric_program_id_asc":
            raise ValueError("Saved Step 5 sensitivity target schema/policy mismatch")
        truth[user]=frozenset(items)
    if tuple(truth)!=protocol.cohort:
        raise ValueError("Secondary sensitivity target cohort/order differs")
    return truth,path


def run():
    if OUT.exists():
        raise FileExistsError("Preserve Step 6 outputs; use a fresh Step 6 output path for reruns")
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"experiments").mkdir()
    protocol=validation.EvaluationProtocol()
    config=protocol.config
    if config["protocol_id"]!="40f6d1c93367106b22d3b66a76605afd12a1ec929f4fe7929bd137749880f239":
        raise ValueError("Frozen Step 3 protocol changed")
    started=time.monotonic()
    rss_start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    profiles,source,profile_rows=build_training_profiles(config)
    profile_seconds=time.monotonic()-started
    train_pairs_path=source.path
    join=validation.json_read(validation.STEP2/"join_manifest.json")
    overlap_path=validation.STEP2/"train_airing_overlaps.csv.gz"
    overlap_hash=join["artifacts"][overlap_path.name]["sha256"]

    history_started=time.monotonic()
    history_tx=ar.user_history_transactions(source.rows())
    history_tx_seconds=time.monotonic()-history_started
    history_ds=ar.build_dataset(history_tx,transaction_type="user_history",window_days=None,
        training_sha256=source.sha256,training_window_utc=source.window_utc,
        source_sha256=source.sha256,transaction_build_seconds=history_tx_seconds)
    datasets={"user_history":history_ds}
    window_metadata={}
    for days in ar.WINDOW_SIZES_DAYS:
        t=time.monotonic()
        transactions=ar.time_window_transactions_from_airings(
            read_training_airing_rows(overlap_path,overlap_hash,source.window_utc),source.window_utc,days,120)
        build_seconds=time.monotonic()-t
        ds=ar.build_dataset(transactions,transaction_type="time_window",window_days=days,
            training_sha256=source.sha256,training_window_utc=source.window_utc,
            source_sha256=overlap_hash,transaction_build_seconds=build_seconds)
        datasets[f"time_window_{days}d"]=ds
        window_metadata[f"time_window_{days}d"]={"transaction_count":ds.transaction_count,
            "window_seconds":days*86400,"aligned_from_training_start_utc":source.window_utc[0],
            "transaction_anchor":"first_overlap_utc of each qualifying user-airing record",
            "boundary_policy":"only records clipped into Weeks 11-20 enter; buckets start at training boundary; exclusive train end rejected"}

    pop=popularity.fit_popularity(config,protocol.candidates)
    fallback=pop.ranking("distinct_users")
    variants=[]
    # Stage 1: compare the three transaction definitions under one fixed rule configuration.
    for dataset_name in ("user_history","time_window_7d","time_window_14d"):
        ds=datasets[dataset_name]
        spec={"min_support":0.0005,"min_confidence":0.01,"min_cooccurrence":3,
              "min_lift":1.0,"top_neighbors":50,"score_type":"confidence",
              "antecedent_weight":"log_frequency"}
        config_id=f"{dataset_name}_conf_lf_base"
        res=evaluate_variant(protocol,ds,profiles,pop,config_id,spec,spec["score_type"],spec["antecedent_weight"],
            profile_seconds+ds.build_seconds+ds.pair_mining_seconds,profile_rows)
        variants.append(res); print("Stage1",config_id,res["primary_map_at_5"],flush=True)
    stage1_best=max(variants,key=lambda x:(x["primary_map_at_5"],x["config_id"]))
    selected_name=("user_history" if stage1_best["transaction_type"]=="user_history" else f"time_window_{stage1_best['window_days']}d")
    best_dataset=datasets[selected_name]

    # Stage 2: small, one-factor-at-a-time rule/scoring grid on the best transaction definition.
    base={"min_support":0.0005,"min_confidence":0.01,"min_cooccurrence":3,"min_lift":1.0,"top_neighbors":50}
    tune_specs=[
        ("base_conf_lf",base,"confidence","log_frequency"),
        ("lift_log",base,"confidence_log_lift","log_frequency"),
        ("confidence_uniform_antecedent",base,"confidence","uniform"),
        ("min_pair5",{**base,"min_cooccurrence":5},"confidence","log_frequency"),
        ("min_conf005",{**base,"min_confidence":0.05},"confidence","log_frequency"),
        ("min_lift12",{**base,"min_lift":1.2},"confidence","log_frequency"),
        ("top100",{**base,"top_neighbors":100},"confidence","log_frequency"),
        ("support001",{**base,"min_support":0.001},"confidence","log_frequency")]
    for suffix,params,score_type,antecedent_weight in tune_specs:
        config_id=f"{selected_name}_{suffix}"
        if any(x["config_id"]==config_id for x in variants):
            continue
        res=evaluate_variant(protocol,best_dataset,profiles,pop,config_id,params,score_type,antecedent_weight,
            profile_seconds+best_dataset.build_seconds+best_dataset.pair_mining_seconds,profile_rows)
        variants.append(res); print("Stage2",config_id,res["primary_map_at_5"],flush=True)

    best=max(variants,key=lambda x:(x["primary_map_at_5"],x["config_id"]))
    secondary,secondary_path=secondary_truth(config,protocol)
    for item in variants:
        secondary_eval=evaluation.evaluate_predictions(item["user_predictions"],secondary,protocol.cohort,protocol.candidates,strict=True)
        item["secondary_top5_sensitivity_map_at_5"]=secondary_eval["primary_map_at_5"]
        item["secondary_top5_sensitivity_nonempty_map_at_5"]=secondary_eval["diagnostic_map_at_5_nonempty"]
    best=max(variants,key=lambda x:(x["primary_map_at_5"],x["config_id"]))
    results=[write_variant_artifacts(x,protocol) for x in variants]

    best_predictions=best["user_predictions"]
    prediction_path=OUT/"association_predictions.csv"
    save_predictions(prediction_path,protocol.cohort,best_predictions)
    best_ap_path=OUT/"association_user_ap.csv"
    best_eval=protocol.evaluate(best_predictions)
    with best_ap_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(["user_id","ap_at_5","validation_relevant_count","rule_coverage","fallback_slots"])
        for u in protocol.cohort:
            diag=best["user_diagnostics"][u]
            writer.writerow((u,format(best_eval["per_user_ap_at_5"][u],".17g"),len(protocol.truth[u]),
                             int(diag["matched_rule_user"]),diag["fallback_slots"]))

    # Secondary scores are checked/recomputed for every Step4 baseline and every PPR config.
    baseline_secondaries=secondary_baseline_evaluations(protocol,secondary)
    ppr_secondary=secondary_ppr_evaluations(protocol,secondary)

    reports={"transaction_datasets":{name:transaction_summary(ds) for name,ds in datasets.items()},
             "window_transaction_policy":window_metadata,
             "history_transaction_policy":{"transaction":"one basket per training user; all distinct qualifying training program IDs",
               "counting":"user contributes once to an unordered program pair"},
             "step4_baseline_secondary_map_at_5":baseline_secondaries,
             "step5_ppr_secondary_map_at_5":ppr_secondary}
    serial_results=[write_variant_artifacts(item,protocol) for item in variants]
    with (OUT/"association_experiments.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=sorted({k for row in serial_results for k in row})
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader()
        for row in serial_results: writer.writerow(row)
    best_config={k:v for k,v in best.items() if k not in ("per_user_ap_at_5","user_predictions","user_diagnostics","model")}
    best_config.update({"method":"Association Rules","evaluation_config_sha256":protocol.config_sha256,
        "protocol_id":config["protocol_id"],"cohort_order_sha256":evaluation.identity_hash(protocol.cohort),
        "candidate_ids_sha256":evaluation.identity_hash(protocol.candidates),"truth_sha256":config["truth"]["sha256"],
        "training_source_sha256":source.sha256,"training_window_utc":source.window_utc,
        "secondary_truth_sha256":validation.sha256(secondary_path),"transaction_summary":reports["transaction_datasets"][selected_name],
        "fallback_policy":evaluation.FALLBACK_POLICY,"random_seed":None,
        "python":__import__("sys").version,"numpy":np.__version__,"scipy":scipy.__version__})
    write_json(OUT/"association_best_config.json",best_config)
    examples=build_user_examples(protocol,profiles,best,best,config)
    write_json(OUT/"association_user_examples.json",examples)
    write_user_examples_md(OUT/"association_user_examples.md",examples,best["config_id"])
    experiment_json={"stage":"Step6 Association Rules only","protocol_id":config["protocol_id"],
        "evaluation_config_sha256":protocol.config_sha256,"cohort_users":len(protocol.cohort),
        "training_source_sha256":source.sha256,"training_rows":profile_rows,
        "training_profile_build_seconds":profile_seconds,"transaction_datasets":reports["transaction_datasets"],
        "window_transaction_policy":window_metadata,"variants":serial_results,
        "best_config_id":best["config_id"],"best_primary_map_at_5":best["primary_map_at_5"],
        "step4_baseline_secondary_map_at_5":baseline_secondaries,"step5_ppr_secondary_map_at_5":ppr_secondary,
        "best_prediction_sha256":validation.sha256(prediction_path),"best_user_ap_sha256":validation.sha256(best_ap_path),
        "best_config_sha256":validation.sha256(OUT/"association_best_config.json"),
        "secondary_truth_sha256":validation.sha256(secondary_path),
        "two_user_examples_sha256":validation.sha256(OUT/"association_user_examples.md"),
        "numpy":np.__version__,"scipy":scipy.__version__,"python":__import__("sys").version,
        "total_run_seconds":time.monotonic()-started,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "no_test_behavior_accessed":True,"no_kaggle_submission":True,"no_ppr_or_popularity_method_refit_during_validation":True}
    write_json(OUT/"association_run.json",experiment_json)
    # Common comparison CSV (all Stage6 variants plus best saved Step4/PPR references).
    comparison=[]
    for row in serial_results:
        comparison.append({"family":"association_rules","variant":row["config_id"],
            "primary_map_at_5":row["primary_map_at_5"],"diagnostic_map_at_5_nonempty":row["diagnostic_map_at_5_nonempty"],
            "secondary_top5_sensitivity_map_at_5":row["secondary_top5_sensitivity_map_at_5"],
            "hit_rate_at_5":row["hit_rate_at_5"],"rule_coverage_fraction":row["rule_coverage_fraction"],
            "candidate_coverage_fraction":row["candidate_coverage_fraction"],
            "cold_start_primary_map_at_5":row["cold_primary_map_at_5"],"cold_start_users_using_fallback":row["cold_start_users_using_fallback"],
            "training_runtime_seconds":row["rule_training_runtime_seconds"],"prediction_runtime_seconds":row["prediction_runtime_seconds"],
            "fallback_slots":row["popularity_backfill_slots"],"source":row["prediction_file"]})
    step4_run=validation.json_read(ROOT/"results"/"step4"/"step4_run.json")
    for key in ("popularity_distinct_users","personal_frequency_qualifying_airings"):
        r=step4_run["variants"][key]
        comparison.append({"family":r["family"],"variant":r["variant"],"primary_map_at_5":r["primary_map_at_5"],
            "diagnostic_map_at_5_nonempty":r["diagnostic_map_at_5_nonempty"],
            "secondary_top5_sensitivity_map_at_5":baseline_secondaries[key],
            "hit_rate_at_5":r["users_with_at_least_one_relevant_recommendation"]/1260,
            "rule_coverage_fraction":None,"candidate_coverage_fraction":r["candidate_coverage_fraction"],
            "cold_start_primary_map_at_5":r["cold_primary_map_at_5"],
            "cold_start_users_using_fallback":r["users_using_full_popularity_fallback"],
            "training_runtime_seconds":r["training_runtime_seconds"],"prediction_runtime_seconds":r["prediction_runtime_seconds"],
            "fallback_slots":r["popularity_backfill_slots_exact"],"source":r["prediction_file"]})
    best_ppr=validation.json_read(ROOT/"results"/"step5"/"ppr_model_config.json")["best_metrics"]
    comparison.append({"family":"ppr","variant":best_ppr["config_id"],"primary_map_at_5":best_ppr["primary_map_at_5"],
        "diagnostic_map_at_5_nonempty":best_ppr["diagnostic_map_at_5_nonempty"],
        "secondary_top5_sensitivity_map_at_5":ppr_secondary[best_ppr["config_id"]],
        "hit_rate_at_5":best_ppr["hit_rate_at_5"],"rule_coverage_fraction":None,
        "candidate_coverage_fraction":best_ppr["candidate_coverage_fraction"],
        "cold_start_primary_map_at_5":best_ppr["cold_primary_map_at_5"],
        "cold_start_users_using_fallback":best_ppr["cold_start_users_using_fallback"],
        "training_runtime_seconds":best_ppr["ppr_training_runtime_seconds"],
        "prediction_runtime_seconds":best_ppr["prediction_generation_seconds"],
        "fallback_slots":best_ppr["fallback_slots"],"source":best_ppr["prediction_file"]})
    with (ROOT/"results"/"step6_association_comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=sorted({k for row in comparison for k in row}); writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison)
    print(json.dumps({"best_config_id":best["config_id"],"best_MAP@5":best["primary_map_at_5"],
        "variants":len(serial_results),"transaction_summary":reports["transaction_datasets"],
        "training_profile_build_seconds":profile_seconds,"total_run_seconds":experiment_json["total_run_seconds"],
        "peak_rss_mib":experiment_json["peak_rss_mib"]},indent=2))


def read_training_airing_rows(path,expected_hash,window):
    """Streaming hashed train-only overlap records, reusable in deterministic fixtures."""
    if validation.sha256(path)!=expected_hash: raise ValueError("Training airing-overlap hash mismatch")
    with gzip.open(path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=ar.OVERLAP_COLUMNS: raise ValueError("Training overlap schema mismatch")
        for row in reader:
            first,last=int(row["first_overlap_utc"]),int(row["last_overlap_utc"])
            if not window[0]<=first<last<=window[1]: raise ValueError("Training time window leakage")
            yield row


def step4run_path(): return "results/step4"


def secondary_baseline_evaluations(protocol,secondary_truth):
    step4=validation.json_read(ROOT/step4run_path()/"step4_run.json")
    result={}
    for key,info in step4["variants"].items():
        path=ROOT/info["prediction_file"]
        if validation.sha256(path)!=info["prediction_sha256"]: raise ValueError("Step4 prediction hash mismatch")
        rows=validation.records(path,["user_id","tv_show_id"])
        preds=evaluation.parse_prediction_records({"user_id":u,"tv_show_id":items} for u,items in rows)
        result[key]=evaluation.evaluate_predictions(preds,secondary_truth,protocol.cohort,protocol.candidates,strict=True)["primary_map_at_5"]
    return result


def secondary_ppr_evaluations(protocol,secondary_truth):
    step5_out=ROOT/"results"/"step5"
    result={}
    for row in csv.DictReader((step5_out/"ppr_experiments.csv").open(newline="",encoding="utf-8")):
        path=step5_out/"experiments"/row["config_id"]/"predictions.csv"
        rows=validation.records(path,["user_id","tv_show_id"])
        preds=evaluation.parse_prediction_records({"user_id":u,"tv_show_id":items} for u,items in rows)
        result[row["config_id"]]=evaluation.evaluate_predictions(preds,secondary_truth,protocol.cohort,protocol.candidates,strict=True)["primary_map_at_5"]
    return result


def metadata_lookup(path):
    """Schedule metadata for display only; never used in transaction/rule features."""
    result={}
    for row in validation.records(path,data_pipeline.SCHEDULE_COLUMNS):
        item=validation.schedule_program_id(row[5])
        if item!="0" and item not in result:
            result[item]={"title":row[4],"category":row[6]}
    return result


def build_user_examples(protocol,profiles,best_predictions,best_variant,config):
    """Reuse exact Step 4 users and compare all requested methods on those users."""
    prior=validation.json_read(ROOT/"results"/"step4"/"provisional_two_users.json")
    chosen={"heavy":prior["heavy_user_id"],"light":prior["light_user_id"]}
    schedule_paths=config["schedule_source_hashes"]
    train_path=next(Path(p) for p in schedule_paths if Path(p).name=="export_arh_11-20-final.csv")
    validation_path=next(Path(p) for p in schedule_paths if Path(p).name=="export_arh_21-30-final.csv")
    train_meta=metadata_lookup(train_path)
    val_meta=metadata_lookup(validation_path)
    step4=validation.json_read(ROOT/"results"/"step4"/"step4_run.json")
    all_lists={}
    for key,info in step4["variants"].items():
        rows=validation.records(ROOT/info["prediction_file"],["user_id","tv_show_id"])
        all_lists[key]=evaluation.parse_prediction_records({"user_id":u,"tv_show_id":field} for u,field in rows)
    ppr_cfg=validation.json_read(ROOT/"results"/"step5"/"ppr_model_config.json")
    ppr_path=ROOT/ppr_cfg["best_metrics"]["prediction_file"]
    rows=validation.records(ppr_path,["user_id","tv_show_id"])
    all_lists["ppr_best"]=evaluation.parse_prediction_records({"user_id":u,"tv_show_id":field} for u,field in rows)
    all_lists["association_rules"]=best_variant["user_predictions"]
    users={}
    for role,user in chosen.items():
        profile=profiles.get(user,{})
        ordered=sorted(profile,key=lambda item:(-profile[item][0],-profile[item][1],int(item)))
        history=[{"program_id":item,"qualifying_airings":profile[item][0],"watched_seconds":profile[item][1],
                  "last_watch_utc":profile[item][2],"title":train_meta.get(item,{}).get("title"),
                  "category":train_meta.get(item,{}).get("category")} for item in ordered]
        relevant=sorted(protocol.truth[user],key=int)
        recs={}
        for name,ranking in all_lists.items():
            recs[name]=[{"program_id":item,"title":val_meta.get(item,{}).get("title"),
                         "category":val_meta.get(item,{}).get("category"),"validation_hit":item in protocol.truth[user]}
                        for item in ranking[user]]
        users[role]={"user_id":user,"training_distinct_programs":len(history),"training_history":history,
                     "validation_relevant_count":len(relevant),
                     "validation_relevant_programs":[{"program_id":item,"title":val_meta.get(item,{}).get("title"),
                         "category":val_meta.get(item,{}).get("category")} for item in relevant],
                     "recommendations":recs,
                     "association_rule_diagnostics":best_variant["user_diagnostics"][user]}
    return {"source":"exact Step 4 fixed provisional users; no re-selection",
            "selection_rule":prior["selection_rule"],"step4_ids_unchanged":chosen=={"heavy":prior["heavy_user_id"],"light":prior["light_user_id"]},
            "metadata_purpose":"display-only schedule metadata, not recommender input",
            "best_association_config_id":best_variant["config_id"],"users":users}


def write_user_examples_md(path,examples,best_config_id):
    with Path(path).open("w",encoding="utf-8") as stream:
        stream.write("# Step 6 Association Rules: fixed-user comparison\n\n")
        stream.write(f"Step 4's heavy/light IDs are reused without change. Best Association Rules setting: `{best_config_id}`. Titles and categories are schedule metadata for display only. Validation hits mean observed qualifying viewing, not enjoyment.\n")
        for role in ("heavy","light"):
            user=examples["users"][role]
            stream.write(f"\n## {role.title()} user `{user['user_id']}`\n\n")
            stream.write(f"Training: {user['training_distinct_programs']} distinct programs. Validation labels: {user['validation_relevant_count']} programs.\n\n")
            stream.write("### Training history\n\n| ID | Title | Category | Qualifying airings | Watched seconds | Last UTC |\n|---:|---|---|---:|---:|---:|\n")
            for item in user["training_history"]:
                stream.write(f"| {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} | {item['qualifying_airings']} | {item['watched_seconds']} | {item['last_watch_utc']} |\n")
            stream.write("\n### Validation relevant programs\n\n| ID | Title | Category |\n|---:|---|---|\n")
            for item in user["validation_relevant_programs"]:
                stream.write(f"| {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} |\n")
            stream.write("\n### Top-5 comparison\n\n| Method | Rank | ID | Title | Category | Validation hit |\n|---|---:|---:|---|---|---|\n")
            for method,rows in user["recommendations"].items():
                for rank,item in enumerate(rows,1):
                    stream.write(f"| {method} | {rank} | {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} | {'yes' if item['validation_hit'] else 'no'} |\n")
            stream.write(f"\nAssociation rule candidate-generation diagnostics: `{user['association_rule_diagnostics']}`.\n")


def save_predictions(path,cohort,predictions):
    if tuple(predictions)!=tuple(cohort): raise ValueError("Association predictions must preserve cohort order")
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n"); writer.writerow(["user_id","tv_show_id"])
        for u in cohort:
            row=tuple(evaluation.program_id(item) for item in predictions[u])
            if len(row)!=5 or len(set(row))!=5: raise ValueError("Invalid Association Rules Top-5")
            writer.writerow((u," ".join(row)))


if __name__=="__main__":
    run()
