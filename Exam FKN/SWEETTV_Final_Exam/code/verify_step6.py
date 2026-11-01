"""Independent transaction/rule/prediction verification for Step 6."""

from collections import Counter, defaultdict
from itertools import combinations
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import evaluation
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step6"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n")


def rows(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",newline="",encoding="utf-8") as f:yield from csv.DictReader(f)


def transactions_reference(config,cohort):
    """Independent set/dict transaction construction; no association_rules helpers."""
    base=validation.ROOT
    train_path=base/config["training_input"]["path"]
    users=defaultdict(set)
    profiles=defaultdict(dict)
    interactions=0
    with gzip.open(train_path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        expected=["user_id","tv_show_id","qualifying_airings","watched_seconds","first_watch_utc","last_watch_utc"]
        if reader.fieldnames!=expected:raise ValueError("Unexpected training transaction schema")
        for r in reader:
            u,item=r["user_id"],r["tv_show_id"]
            if int(item)<=0:raise ValueError("Unlabelled program present in training interactions")
            users[u].add(item)
            profiles[u][item]=(int(r["qualifying_airings"]),int(r["watched_seconds"]))
            interactions+=1
    hist=tuple((u,frozenset(items)) for u,items in sorted(users.items()))
    windows={7:defaultdict(set),14:defaultdict(set)}
    window_lo,window_hi=config["windows_utc"]["train"]
    join_manifest=read_json(base/"data"/"step2"/"join_manifest.json")
    overlap=base/"data"/"step2"/"train_airing_overlaps.csv.gz"
    if sha(overlap)!=join_manifest["artifacts"][overlap.name]["sha256"]:
        raise ValueError("Training overlap hash differs from Step 2 join manifest")
    with gzip.open(overlap,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        for expected_col in ("user_id","tv_show_id","overlap_seconds","first_overlap_utc","last_overlap_utc"):
            if expected_col not in reader.fieldnames:raise ValueError("Training airing overlap schema mismatch")
        for r in reader:
            u,item=r["user_id"],r["tv_show_id"]
            first,last,seconds=int(r["first_overlap_utc"]),int(r["last_overlap_utc"]),int(r["overlap_seconds"])
            if not window_lo<=first<last<=window_hi:raise ValueError("Time-window anchor outside training boundary")
            if int(item)<=0 or seconds<120:continue
            for days,baskets in windows.items():
                bucket=(first-window_lo)//(days*86400)
                baskets[(u,bucket)].add(item)
    tx={"user_history":hist}
    for days,baskets in windows.items():
        tx[f"time_window_{days}d"]=tuple((u,b,frozenset(items)) for (u,b),items in sorted(baskets.items()))
    stats={}
    for name,transactions in tx.items():
        sizes=[len(t[-1]) for t in transactions]
        pairs=sum(len(b)*(len(b)-1)//2 for b in (t[-1] for t in transactions))
        stats[name]={"transactions":len(transactions),"item_support_total":sum(sizes),
            "distinct_programs":len(set().union(*(t[-1] for t in transactions))) if transactions else 0,
            "pair_occurrences":pairs,"average_transaction_size":sum(sizes)/len(sizes) if sizes else 0,
            "max_transaction_size":max(sizes,default=0),"users":len(set(t[0] for t in transactions))}
    return tx,stats,profiles,interactions,sha(train_path),sha(overlap)


def secondary_truth_reference(config,cohort):
    path=OUT/"../step5"/"secondary_top5_truth.csv"
    path=path.resolve()
    expected="440c61d96a78dcf346ac167729a6440f313e4980c1beee661c7d6ee9ced80d4f"
    if sha(path)!=expected:raise ValueError("Step 5 secondary truth hash mismatch")
    result={}
    for r in rows(path):
        items=tuple(r["top5_validation_program_ids"].split()) if r["top5_validation_program_ids"] else ()
        result[r["user_id"]]=frozenset(items)
    if tuple(result)!=tuple(cohort):raise ValueError("Secondary sensitivity cohort order mismatch")
    return result,path


def independent_support(transactions):
    supports=Counter()
    for trans in transactions:
        supports.update(trans[-1])
    return supports


def sample_rule_reference(rule_path,transactions,transaction_count,limit=250):
    supports=independent_support(transactions)
    checks=[]; sampled=[]
    for row in rows(rule_path):
        if len(sampled)>=limit:break
        a,b=row["antecedent_id"],row["consequent_id"]
        pair=sum(a in t[-1] and b in t[-1] for t in transactions)
        if pair==0 or supports[a]==0 or supports[b]==0:raise ValueError("Saved rule has zero independent support")
        sup=pair/transaction_count
        conf=pair/supports[a]
        lift=conf/(supports[b]/transaction_count)
        stored=(int(row["transaction_support_count"]),float(row["support"]),float(row["confidence"]),float(row["lift"]))
        passed=(pair==stored[0] and math.isclose(sup,stored[1],rel_tol=1e-13,abs_tol=1e-15)
            and math.isclose(conf,stored[2],rel_tol=1e-13,abs_tol=1e-15)
            and math.isclose(lift,stored[3],rel_tol=1e-12,abs_tol=1e-14))
        checks.append(passed);sampled.append({"antecedent":a,"consequent":b,"pair_count":pair,
            "independent_support":sup,"independent_confidence":conf,"independent_lift":lift,"passed":passed})
    return all(checks) and bool(checks),sampled,supports


def parse_predictions(path,cohort):
    rows_list=list(rows(path))
    if [r["user_id"] for r in rows_list]!=list(cohort):raise ValueError(f"Saved prediction cohort/order mismatch: {path}")
    return evaluation.parse_prediction_records(rows_list)


def main():
    started=time.monotonic()
    protocol=validation.EvaluationProtocol(); config=protocol.config;cohort=protocol.cohort;candidates=protocol.candidates
    run=read_json(OUT/"association_run.json")
    best=read_json(OUT/"association_best_config.json")
    tests=read_json(sorted((ROOT/"results"/"step6_test_runs").glob("run_*.json"))[-1])
    checks=[]
    def check(name,passed,detail=None):checks.append({"name":name,"passed":bool(passed),"detail":detail})
    originals=read_json(ROOT/"results"/"original_sha256_before.json")
    for path,meta in originals.items():check("original_unchanged:"+Path(path).name,sha(path)==meta["sha256"])
    step3_snapshot=read_json(ROOT/"results"/"step3_inputs_before.json")["files"]
    changed=[p for p,m in step3_snapshot.items() if sha(p)!=m["sha256"] or Path(p).stat().st_size!=m["bytes"]]
    check("Step1_to_Step3_preservation",not changed,changed)
    step4run=read_json(ROOT/"results"/"step4"/"step4_run.json")
    step5verify=read_json(ROOT/"results"/"step5_verification.json")
    check("Step4_and_Step5_verification_preserved",step4run["protocol_id"]==config["protocol_id"] and step5verify["all_passed"])
    check("protocol_and_cohort_same",run["protocol_id"]==config["protocol_id"] and run["cohort_users"]==1260)
    tx,txstats,profiles,training_rows,train_hash,airing_hash=transactions_reference(config,cohort)
    check("training_hash_and_rows",train_hash==config["training_input"]["sha256"] and training_rows==716796)
    run_stats=run["transaction_datasets"]
    for name,stats in txstats.items():
        saved=run_stats[name]
        for field in ("transactions","item_support_total","distinct_programs","pair_occurrences","max_transaction_size"):
            check(f"transaction_stat:{name}:{field}",stats[field]==saved[field],{"independent":stats[field],"saved":saved[field]})
        check(f"transaction_average_size:{name}",math.isclose(stats["average_transaction_size"],saved["average_transaction_size"],rel_tol=0,abs_tol=1e-13))
    check("history_population_4837",txstats["user_history"]["transactions"]==4837)
    check("window_data_from_step2_train_airing_hash",airing_hash==read_json(validation.STEP2/"join_manifest.json")["artifacts"]["train_airing_overlaps.csv.gz"]["sha256"])
    # Rule table metrics: independently test sampled directed rules by basket membership.
    best_rules_path=ROOT/best["rules_file"]
    best_tx=tx["user_history"] if best["transaction_type"]=="user_history" else tx[f"time_window_{int(best['window_days'])}d"]
    sampled_ok,sampled_rules,supports=sample_rule_reference(best_rules_path,best_tx,len(best_tx),limit=250)
    check("sampled_best_rule_support_confidence_lift",sampled_ok,{"sampled_rules":len(sampled_rules),"examples":sampled_rules[:5]})
    saved_best_rules=list(rows(best_rules_path))
    by_antecedent=Counter(r["antecedent_id"] for r in saved_best_rules)
    check("top_neighbor_rule_cap",max(by_antecedent.values(),default=0)<=int(best["top_neighbors"]))
    check("rule_filters_and_positive_ids",all(int(r["transaction_support_count"])>=int(best["min_cooccurrence"])
        and float(r["support"])>=float(best["min_support"]) and float(r["confidence"])>=float(best["min_confidence"])
        and float(r["lift"])>=float(best["min_lift"]) and int(r["antecedent_id"])>0 and int(r["consequent_id"])>0
        and r["antecedent_id"]!=r["consequent_id"] for r in saved_best_rules))
    # All 11 association configurations, every saved prediction, AP/MAP, fallback and candidate checks.
    experiment_rows=list(rows(OUT/"association_experiments.csv"))
    check("association_experiment_variants_count",len(experiment_rows)==11)
    secondary_truth,_=secondary_truth_reference(config,cohort)
    exact_best=None
    for row in experiment_rows:
        cid=row["config_id"]
        pred_path=ROOT/row["prediction_file"]
        ap_path=ROOT/row["user_ap_file"]
        rules_path=ROOT/row["rules_file"]
        check("experiment_prediction_hash:"+cid,sha(pred_path)==row["prediction_sha256"])
        check("experiment_user_ap_hash:"+cid,sha(ap_path)==row["user_ap_sha256"])
        check("experiment_rules_hash:"+cid,sha(rules_path)==row["rules_sha256"])
        pred=parse_predictions(pred_path,cohort)
        check("experiment_top5_eligible:"+cid,all(len(pred[u])==5 and len(set(pred[u]))==5
            and set(pred[u]).issubset(set(candidates)) and all(int(p)>0 for p in pred[u]) for u in cohort))
        primary=protocol.evaluate(pred)
        sec=evaluation.evaluate_predictions(pred,secondary_truth,cohort,candidates,strict=True)
        check("experiment_primary_map_recalculated:"+cid,math.isclose(primary["primary_map_at_5"],float(row["primary_map_at_5"]),rel_tol=0,abs_tol=1e-15))
        check("experiment_diagnostic_map_recalculated:"+cid,math.isclose(primary["diagnostic_map_at_5_nonempty"],float(row["diagnostic_map_at_5_nonempty"]),rel_tol=0,abs_tol=1e-15))
        check("experiment_secondary_map_recalculated:"+cid,math.isclose(sec["primary_map_at_5"],float(row["secondary_top5_sensitivity_map_at_5"]),rel_tol=0,abs_tol=1e-15))
        ap_rows=list(rows(ap_path))
        check("experiment_per_user_AP_all_users:"+cid,[r["user_id"] for r in ap_rows]==list(cohort)
            and all(math.isclose(float(r["ap_at_5"]),primary["per_user_ap_at_5"][u],rel_tol=0,abs_tol=1e-15) for u,r in zip(cohort,ap_rows)))
        cold=[u for u in cohort if u not in profiles]
        popularity_top5=parse_predictions(ROOT/"results"/"step4"/"popularity_distinct_users_predictions.csv",cohort)
        check("experiment_cold_fallback:"+cid,all(pred[u]==popularity_top5[u] for u in cold))
        if cid==best["config_id"]:exact_best=pred
    if exact_best is not None:
        check("best_top_level_predictions_match_config",exact_best==parse_predictions(OUT/"association_predictions.csv",cohort))
    check("best_config_matches_max_MAP",best["config_id"]==max(experiment_rows,key=lambda r:(float(r["primary_map_at_5"]),r["config_id"]))["config_id"])
    check("best_config_sha_and_prediction",sha(OUT/"association_best_config.json")==run["best_config_sha256"]
        and sha(OUT/"association_predictions.csv")==run["best_prediction_sha256"])
    # Verify exact same Step4/5 baseline scores and selected fixed example identities.
    step4_compare=list(rows(ROOT/"results"/"step4_baseline_comparison.csv"))
    check("step4_reference_scores_match_saved_table",len(step4_compare)==5 and
        any(r["model"]=="popularity" and r["variant"]=="distinct_users" and math.isclose(float(r["primary_map_at_5"]),.7684179894179894,abs_tol=1e-15) for r in step4_compare)
        and any(r["model"]=="personal_frequency" and r["variant"]=="qualifying_airings" and math.isclose(float(r["primary_map_at_5"]),.7876269841269841,abs_tol=1e-15) for r in step4_compare))
    ppr_cfg=read_json(ROOT/"results"/"step5"/"ppr_model_config.json")
    check("step5_reference_score_match",math.isclose(ppr_cfg["best_metrics"]["primary_map_at_5"],.8151507936507936,abs_tol=1e-15))
    fixed_users=read_json(ROOT/"results"/"step4"/"provisional_two_users.json")
    examples=read_json(OUT/"association_user_examples.json")
    check("step4_fixed_users_reused",examples["users"]["heavy"]["user_id"]==fixed_users["heavy_user_id"]
        and examples["users"]["light"]["user_id"]==fixed_users["light_user_id"])
    check("all_baselines_and_AR_in_user_examples",all(set(examples["users"][role]["recommendations"])==
        {"Popularity distinct viewers","Personal frequency qualifying airings","Personalized PageRank","Association Rules"}
        for role in ("heavy","light")))
    artifacts=read_json(OUT/"association_artifacts.json")
    bad=[name for name,meta in artifacts.items() if (OUT/name).stat().st_size!=meta["bytes"] or sha(OUT/name)!=meta["sha256"]]
    check("all_step6_artifacts_match_manifest",not bad,bad)
    test_run=read_json(sorted((ROOT/"results"/"step6_test_runs").glob("run_*.json"))[-1])
    check("final_step6_unit_tests_pass",test_run["all_passed"] and test_run["tests_run"]==14)
    check("unit_test_source_hashes_current",all(sha(ROOT/"code"/name)==value for name,value in test_run["source_sha256"].items()))
    check("step3_config_unchanged",run["evaluation_config_sha256"]==protocol.config_sha256 and run["protocol_id"]==config["protocol_id"])
    check("no_kaggle_submission",not list((ROOT/"submission").glob("*.csv")))
    check("no_test_behavior_accessed",run["no_test_behavior_accessed"] is True and all("test_view" not in k for k in config["input_products"]))
    result={"stage":"Step6 Association Rules only","all_passed":all(x["passed"] for x in checks),
        "passed_checks":sum(x["passed"] for x in checks),"failed_checks":sum(not x["passed"] for x in checks),
        "checks":checks,"transaction_reference_statistics":txstats,
        "best_sampled_rules_independent_reference":sampled_rules,
        "rule_sample_count":len(sampled_rules),"experiment_configurations":len(experiment_rows),
        "best_config_id":best["config_id"],"best_primary_map_at_5":best["primary_map_at_5"],
        "training_interaction_rows":training_rows,"training_sha256":train_hash,"window_airing_hash":airing_hash,
        "test_run":test_run,"step4_scores_checked":True,"step5_score_checked":True,
        "verification_runtime_seconds":round(time.monotonic()-started,3),
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "no_ppr_or_rules_retrain_from_validation":True,"no_kaggle_submission":True}
    write_json(ROOT/"results"/"step6_verification.json",result)
    write_json(OUT/"step6_verification.json",result)
    print(json.dumps({k:result[k] for k in ("all_passed","passed_checks","failed_checks","experiment_configurations",
        "best_config_id","best_primary_map_at_5","rule_sample_count","verification_runtime_seconds","peak_rss_mib")},indent=2))
    if not result["all_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
