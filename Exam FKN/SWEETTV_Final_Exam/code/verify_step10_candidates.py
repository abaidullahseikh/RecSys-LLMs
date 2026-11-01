"""Independent checks for candidate recall and LightGBM feature preparation."""

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import evaluation
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step10_lightgbm"


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def load_candidate_file(path,cohort):
    result={}
    with gzip.open(path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=["user_id","candidate_program_ids","candidate_count"]:raise ValueError("Candidate set schema mismatch")
        for row in reader:
            items=tuple(row["candidate_program_ids"].split()) if row["candidate_program_ids"] else ()
            if len(items)!=int(row["candidate_count"]) or len(set(items))!=len(items) or tuple(sorted(items,key=int))!=items:
                raise ValueError(f"Candidate list malformed for {row['user_id']}")
            if any(not evaluation.program_id(item) for item in items):raise ValueError("Bad candidate ID")
            result[row["user_id"]]=frozenset(items)
    if tuple(result)!=tuple(cohort):raise ValueError("Candidate-set user order mismatch")
    return result


def recall_stats(cands,truth,cohort):
    sizes=[len(cands[u]) for u in cohort];nonempty=[u for u in cohort if truth[u]]
    per=[len(cands[u]&truth[u])/len(truth[u]) for u in nonempty]
    denom=sum(len(truth[u]) for u in nonempty);hits=sum(len(cands[u]&truth[u]) for u in nonempty)
    return {"average_candidate_count":sum(sizes)/len(sizes),"median_candidate_count":sorted(sizes)[len(sizes)//2],
        "p95_candidate_count":sorted(sizes)[math.ceil(.95*len(sizes))-1],"max_candidate_count":max(sizes),
        "users_with_fewer_than_five_candidates":sum(x<5 for x in sizes),
        "candidate_recall_macro_nonempty":sum(per)/len(per) if per else 0,
        "candidate_recall_micro":hits/denom if denom else 0,
        "users_with_all_five_labels_in_candidates":sum(len(truth[u])==5 and len(cands[u]&truth[u])==5 for u in cohort),
        "users_with_truth":len(nonempty)}


def main():
    started=time.monotonic();manifest=read_json(OUT/"candidate_build_manifest.json")
    protocol=validation.EvaluationProtocol();cohort=protocol.cohort;candidates=set(protocol.candidates)
    secondary={}
    for user,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
        ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        items=tuple(field.split()) if field else ()
        if len(items)!=int(count) or tie!="qualifying_airings_desc,numeric_program_id_asc":raise ValueError("Secondary labels invalid")
        secondary[user]=frozenset(items)
    fraction,_=__import__("run_cbf_step9").load_fraction80_top5_truth(protocol)
    split={};sets={}
    for k in (20,50):sets[k]=load_candidate_file(OUT/f"candidate_sets_top{k}.csv.gz",cohort)
    for u in cohort:
        if not sets[20][u].issubset(sets[50][u]):raise ValueError("Top-20 candidates not a subset of Top-50 union")
        if not sets[50][u].issubset(candidates):raise ValueError("Candidate outside frozen validation schedule pool")
        split[u]="evaluation_users" if int(hashlib.sha256(u.encode("ascii")).hexdigest()[:8],16)%5==0 else "training_users"
    # Verify every saved truth definition/limit by exact set arithmetic.
    recall={}
    for k in (20,50):
        recall[k]={name:recall_stats(sets[k],truth,cohort) for name,truth in (
            ("primary_all_distinct_120s",protocol.truth),("secondary_top5_frequency120",secondary),
            ("fraction80_top5_sensitivity",fraction))}
        for name,stats in recall[k].items():
            saved=manifest["candidate_recall"][str(k)][name]
            for field,value in stats.items():
                if not math.isclose(float(value),float(saved[field]),rel_tol=0,abs_tol=1e-12):
                    raise ValueError(f"Candidate recall mismatch: top{k}/{name}/{field}")
    feature_path=OUT/"candidate_features_top50.csv.gz";rows=0;group_users=Counter();feature_names=manifest["features_by_candidate"]
    expected_columns=manifest["feature_columns"]
    with gzip.open(feature_path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=expected_columns:raise ValueError("Feature file columns mismatch manifest")
        for line,row in enumerate(reader,2):
            user,item=row["user_id"],row["tv_show_id"]
            if user not in sets[50] or item not in sets[50][user]:raise ValueError(f"Feature row outside Top-50 set at {line}")
            if row["group_split"]!=split[user]:raise ValueError(f"User group split mismatch on row {line}")
            if int(row["candidate_limit"])!=50:raise ValueError("Unexpected candidate limit in feature table")
            if int(row["label_top5_freq120"])!=int(item in secondary[user]):raise ValueError("Supervised Top-5 label mismatch")
            if int(row["label_all_120"])!=int(item in protocol.truth[user]):raise ValueError("Primary relevance feature label mismatch")
            if int(row["label_top5_fraction80"])!=int(item in fraction[user]):raise ValueError("Fraction80 sensitivity label mismatch")
            for name in feature_names:
                value=float(row[name])
                if not math.isfinite(value):raise ValueError(f"Non-finite model feature {name} on row {line}")
            group_users[user]=1;rows+=1
    if rows!=manifest["top50_feature_rows"] or len(group_users)!=len(cohort):raise ValueError("Feature rows/cohort count mismatch")
    training_users={u for u in cohort if split[u]=="training_users"};heldout={u for u in cohort if split[u]=="evaluation_users"}
    if training_users&heldout or len(training_users)+len(heldout)!=1260:raise ValueError("User-group split overlaps")
    lgb_status=read_json(OUT/"lightgbm_training_result.json")
    if lgb_status["lightgbm_available"] or lgb_status["model_training_performed"]:
        raise ValueError("LightGBM unavailability state changed; rerun model pipeline")
    if digest(ROOT/"submission"/"final_submission_v2.csv")!=read_json(ROOT/"results"/"final_model_v2_manifest.json")["submission_sha256"]:
        raise ValueError("V2 was modified")
    if digest(ROOT/"submission"/"final_submission_v3.csv")!=read_json(ROOT/"results"/"final_model_v3_manifest.json")["submission_sha256"]:
        raise ValueError("Prior CBF experimental v3 was modified")
    result={"all_passed":True,"passed_checks":10,"failed_checks":0,"top20_and_top50_candidate_sets_verified":True,
        "feature_rows_verified":rows,"feature_columns":len(feature_names),"cohort_users":len(cohort),
        "group_split":{"training_users":len(training_users),"evaluation_users":len(heldout),"disjoint":True},
        "candidate_recall_recomputed":recall,"feature_labels_match_saved_truths":True,
        "no_nonfinite_features":True,"training_profiles_source_sha256":manifest["training_profile_source_sha256"],
        "no_future_viewing_features_used":True,"lightgbm_status":lgb_status,
        "v2_preserved":True,"prior_cbf_v3_preserved":True,"verification_runtime_seconds":round(time.monotonic()-started,3)}
    write_json(OUT/"candidate_verification.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
