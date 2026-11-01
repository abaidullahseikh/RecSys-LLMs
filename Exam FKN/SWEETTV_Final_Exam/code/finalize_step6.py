"""Finalize saved Step 6 metrics/runtime and the fixed two-user report."""

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import data_pipeline
import evaluation
import validation
from run_step6 import build_training_profiles

ROOT=validation.ROOT
OUT=ROOT/"results"/"step6"


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))


def metadata(path):
    result={}
    for row in validation.records(path,data_pipeline.SCHEDULE_COLUMNS):
        item=validation.schedule_program_id(row[5])
        if item!="0" and item not in result:result[item]={"title":row[4],"category":row[6]}
    return result


def read_predictions(path,cohort):
    rows=validation.records(path,["user_id","tv_show_id"])
    return evaluation.parse_prediction_records({"user_id":u,"tv_show_id":items} for u,items in rows)


def main():
    protocol=validation.EvaluationProtocol(); config=protocol.config
    run=json.loads((OUT/"association_run.json").read_text())
    best_config=json.loads((OUT/"association_best_config.json").read_text())
    profiles,source,profile_rows=build_training_profiles(config)
    saved_predictions=read_predictions(OUT/"association_predictions.csv",protocol.cohort)
    metrics_by_id={r["config_id"]:r for r in read_rows(OUT/"association_experiments.csv")}
    pop_profile=read_predictions(ROOT/"results"/"step4"/"popularity_distinct_users_predictions.csv",protocol.cohort)
    personal=read_predictions(ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv",protocol.cohort)
    ppr_config=json.loads((ROOT/"results"/"step5"/"ppr_model_config.json").read_text())
    ppr_predictions=read_predictions(ROOT/ppr_config["best_metrics"]["prediction_file"],protocol.cohort)
    schedule_paths=[Path(p) for p in config["schedule_source_hashes"]]
    train_meta=next(metadata(p) for p in schedule_paths if p.name=="export_arh_11-20-final.csv")
    val_meta=next(metadata(p) for p in schedule_paths if p.name=="export_arh_21-30-final.csv")
    step4_users=json.loads((ROOT/"results"/"step4"/"provisional_two_users.json").read_text())
    selected={"heavy":step4_users["heavy_user_id"],"light":step4_users["light_user_id"]}
    examples={"selection_source":"exact Step 4 provisional user IDs; not reselected in Step 6",
              "selection_rule":step4_users["selection_rule"],"users":{}}
    for role,user in selected.items():
        profile=profiles.get(user,{})
        train_history=[]
        for item,(airings,seconds,last) in sorted(profile.items(),key=lambda p:(-p[1][0],-p[1][1],int(p[0]))):
            m=train_meta.get(item,{})
            train_history.append({"program_id":item,"qualifying_airings":airings,"watched_seconds":seconds,
                                  "last_watch_utc":last,"title":m.get("title"),"category":m.get("category")})
        labels=sorted(protocol.truth[user],key=int)
        relevant=[{"program_id":item,"title":val_meta.get(item,{}).get("title"),
                    "category":val_meta.get(item,{}).get("category")} for item in labels]
        matched=[]
        rule_file=ROOT/metrics_by_id[best_config["config_id"]]["rules_file"]
        for row in read_rows(rule_file):
            if row["antecedent_id"] in profile and row["consequent_id"] in protocol.candidates:
                matched.append({"antecedent_id":row["antecedent_id"],"consequent_id":row["consequent_id"],
                                "support":float(row["support"]),"confidence":float(row["confidence"]),"lift":float(row["lift"])})
        matched.sort(key=lambda r:(-r["confidence"],-r["lift"],int(r["antecedent_id"]),int(r["consequent_id"])))
        recs={}
        for name,preds in (("Popularity distinct viewers",pop_profile),("Personal frequency qualifying airings",personal),
                           ("Personalized PageRank",ppr_predictions),("Association Rules",saved_predictions)):
            recs[name]=[{"program_id":item,"title":val_meta.get(item,{}).get("title"),
                         "category":val_meta.get(item,{}).get("category"),"validation_hit":item in protocol.truth[user]}
                        for item in preds[user]]
        examples["users"][role]={"user_id":user,"training_distinct_programs":len(profile),
            "training_qualifying_airings":sum(v[0] for v in profile.values()),
            "training_watched_seconds":sum(v[1] for v in profile.values()),"training_history":train_history,
            "validation_relevant_count":len(labels),"validation_relevant_programs":relevant,
            "association_rule_candidate_coverage":bool(matched),"association_top_matched_rules":matched[:25],
            "association_rule_diagnostics":metrics_by_id[best_config["config_id"]].get("rule_coverage_fraction"),
            "recommendations":recs}
    examples_path=OUT/"association_user_examples.json"
    examples_path.write_text(json.dumps(examples,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    md=OUT/"association_user_examples.md"
    with md.open("w",encoding="utf-8") as stream:
        stream.write("# Step 6 Association Rules: Step 4 fixed-user comparison\n\n")
        stream.write(f"Selected Association Rules configuration: `{best_config['config_id']}`. The exact Step 4 heavy/light users are reused. Titles/categories are metadata only; validation matches indicate recorded qualifying viewing, not enjoyment.\n")
        for role,user in selected.items():
            e=examples["users"][role]
            stream.write(f"\n## {role.title()} user `{user}`\n\nTraining: {e['training_distinct_programs']} distinct programs, {e['training_qualifying_airings']} qualifying airings, {e['training_watched_seconds']} seconds. Validation relevance: {e['validation_relevant_count']} programs.\n\n")
            stream.write("### Actual training viewing history\n\n| ID | Title | Category | Qualifying airings | Seconds | Last UTC |\n|---:|---|---|---:|---:|---:|\n")
            for item in e["training_history"]:
                stream.write(f"| {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} | {item['qualifying_airings']} | {item['watched_seconds']} | {item['last_watch_utc']} |\n")
            stream.write("\n### Actual validation-relevant programs\n\n| ID | Title | Category |\n|---:|---|---|\n")
            for item in e["validation_relevant_programs"]:
                stream.write(f"| {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} |\n")
            stream.write("\n### Top-5 comparison\n\n| Method | Rank | ID | Title | Category | Validation hit |\n|---|---:|---:|---|---|---|\n")
            for method,rows in e["recommendations"].items():
                for rank,item in enumerate(rows,1):
                    stream.write(f"| {method} | {rank} | {item['program_id']} | {item['title'] or '(metadata unavailable)'} | {item['category'] or '(metadata unavailable)'} | {'yes' if item['validation_hit'] else 'no'} |\n")
            stream.write("\n### Association evidence\n\nMatched training-derived rule examples are listed as antecedent → consequent with support, confidence and lift:\n\n")
            for r in e["association_top_matched_rules"]:
                stream.write(f"- `{r['antecedent_id']} → {r['consequent_id']}`: support `{r['support']:.6g}`, confidence `{r['confidence']:.6g}`, lift `{r['lift']:.6g}`.\n")
    # Correct total model preparation time is profile build + transaction creation
    # + sparse pair mining + each configuration's rule filtering.
    exp_path=OUT/"association_experiments.csv"
    exp_rows=read_rows(exp_path)
    for row in exp_rows:
        row["rule_training_runtime_seconds"]=format(
            run["training_profile_build_seconds"]+float(row["transaction_build_seconds"])
            +float(row["pair_mining_seconds"])+float(row["rule_mining_seconds"]),".17g")
        by_id=metrics_by_id[row["config_id"]]
        by_id["rule_training_runtime_seconds"]=float(row["rule_training_runtime_seconds"])
    with exp_path.open("w",newline="",encoding="utf-8") as stream:
        fields=list(exp_rows[0])
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(exp_rows)
    run["variants"]=exp_rows
    run["best_rule_training_runtime_seconds"]=metrics_by_id[best_config["config_id"]]["rule_training_runtime_seconds"]
    # Keep the root comparison and machine-readable best config in sync.
    compare_path=ROOT/"results"/"step6_association_comparison.csv"
    comparison=read_rows(compare_path)
    for row in comparison:
        if row["family"]=="association_rules":
            match=metrics_by_id[row["variant"]]
            row["training_runtime_seconds"]=match["rule_training_runtime_seconds"]
    with compare_path.open("w",newline="",encoding="utf-8") as stream:
        fields=list(comparison[0]);writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(comparison)
    best_config.update(metrics_by_id[best_config["config_id"]])
    best_config.update({"two_user_examples_sha256":digest(md),"two_user_examples_json_sha256":digest(examples_path)})
    run["two_user_examples_sha256"]=digest(md)
    run["two_user_examples_json_sha256"]=digest(examples_path)
    run["best_prediction_sha256"]=digest(OUT/"association_predictions.csv")
    run["best_user_ap_sha256"]=digest(OUT/"association_user_ap.csv")
    run["variants"]=exp_rows
    (OUT/"association_best_config.json").write_text(json.dumps(best_config,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    run["best_config_sha256"]=digest(OUT/"association_best_config.json")
    (OUT/"association_run.json").write_text(json.dumps(run,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    artifacts={str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":digest(p)}
               for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in {"association_artifacts.json","step6_verification.json"}}
    (OUT/"association_artifacts.json").write_text(json.dumps(artifacts,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"fixed_users":selected,"best_config":best_config["config_id"],
                      "examples_sha256":digest(md),"artifact_count":len(artifacts)},indent=2))


if __name__=="__main__":main()
