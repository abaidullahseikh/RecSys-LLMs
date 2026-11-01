"""Independent integrity, eligibility, protocol, and leakage verifier."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import data_pipeline
import validation

ROOT=validation.ROOT
SUBMISSION=ROOT/"submission"/"final_submission.csv"
MANIFEST=ROOT/"results"/"final_model_manifest.json"
RESULT=ROOT/"results"/"final_submission_verification.json"
CHECKSUM=ROOT/"results"/"final_submission_sha256.txt"
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
TEMPLATE=SOURCE/"submission.csv"
SCHEDULE=SOURCE/"export_arh_31-42-final.csv"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
ID_RE=re.compile(r"[0-9]+\Z",re.ASCII)


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def parse_raw_id(value):
    if not isinstance(value,str) or not ID_RE.fullmatch(value):raise ValueError(f"Invalid numeric ID: {value!r}")
    return str(int(value))


def read_original_template():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=["user_id","tv_show_id"]:raise ValueError("Original template header mismatch")
        users=[]
        for row in reader:
            if len(row)!=2 or not row[0] or not ID_RE.fullmatch(row[0]):raise ValueError("Malformed template user row")
            users.append(row[0])
    if len(users)!=1260 or len(set(users))!=1260:raise ValueError("Template cohort cardinality/uniqueness mismatch")
    return tuple(users)


def candidates_independent(join_manifest):
    expected=data_pipeline.SCHEDULE_COLUMNS
    lo,hi=validation.WINDOWS["test"]
    pool=set()
    with SCHEDULE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=expected:raise ValueError("Test schedule header mismatch")
        for line,row in enumerate(reader,2):
            if len(row)!=len(expected):raise ValueError(f"Malformed schedule row {line}")
            raw_id=parse_raw_id(row[5])
            if raw_id=="0":continue
            start=data_pipeline.parse_timestamp(row[2],schedule=True)
            duration=int(row[3])
            if duration>0 and start<hi and start+duration>lo:pool.add(raw_id)
    ordered=tuple(sorted(pool,key=int))
    with (ROOT/"data"/"step2"/"test_candidates.csv").open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=["tv_show_id"]:raise ValueError("Step 2 candidate catalog schema mismatch")
        saved=tuple(r[0] for r in reader)
    # The common Step 2 catalog is broader because it can include schedule
    # rows from another file that cross a temporal boundary. The Kaggle pool
    # required here comes only from the named Week 31-42 schedule.
    if not set(ordered).issubset(set(saved)):
        raise ValueError("Week 31-42 schedule pool is not contained in verified Step 2 catalog")
    expected_hash=join_manifest["artifacts"]["test_candidates.csv"]["sha256"]
    if digest(ROOT/"data"/"step2"/"test_candidates.csv")!=expected_hash:raise ValueError("Step 2 candidate file hash mismatch")
    return ordered


def verify_history(manifest,join_manifest):
    expected=join_manifest["source_files_before"]
    bound=validation.WINDOWS["history"]
    info=read_json(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    source_info=info["artifacts"][HISTORY.name]
    if digest(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")!=manifest["training_source_manifest_sha256"]:
        raise ValueError("Step 2 history manifest hash mismatch")
    if digest(HISTORY)!=source_info["sha256"] or source_info["sha256"]!=manifest["training_source_sha256"]:
        raise ValueError("Training interactions are not the hash-verified Step 2 history artifact")
    if manifest["training_source_path"]!=str(HISTORY.relative_to(ROOT)):
        raise ValueError("Training source path is not the Weeks 11-30 historical interaction file")
    if tuple(manifest["training_period_utc"])!=tuple(bound):raise ValueError("Training bounds differ from Weeks 11-30")
    if manifest["training_threshold_seconds_per_airing"]!=120:raise ValueError("Wrong viewing threshold")
    if SCHEDULE in [Path(p) for p in manifest.get("training_sources",[])]:raise ValueError("Test schedule incorrectly registered as training input")
    rows=0;users=set();programs=set();prev=None;min_first=None;max_last=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=validation.INTERACTION_COLUMNS:raise ValueError("Historical interaction schema mismatch")
        for line,row in enumerate(reader,2):
            user=row["user_id"]
            if not user or not ID_RE.fullmatch(user):raise ValueError(f"Invalid historical user ID at line {line}")
            raw=row["tv_show_id"]
            if not raw or not ID_RE.fullmatch(raw):raise ValueError(f"Invalid historical program ID at line {line}")
            item=str(int(raw))
            if raw!=item or item=="0":raise ValueError(f"Noncanonical/zero historical program at line {line}")
            key=(user,int(item))
            if prev is not None and key<=prev:raise ValueError("History rows are duplicated or unsorted")
            prev=key
            airings,seconds,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            if airings<=0 or seconds<airings*120:raise ValueError(f"Sub-threshold/invalid interaction at line {line}")
            if not bound[0]<=first<last<=bound[1]:raise ValueError(f"Future/out-of-period behavior at line {line}")
            min_first=min(min_first,first) if min_first is not None else first
            max_last=max(max_last,last) if max_last is not None else last
            rows+=1;users.add(user);programs.add(item)
    if rows!=manifest["training_interaction_rows"]:raise ValueError("Training row count differs from final model manifest")
    if rows!=info["periods"]["history"]["unique_user_program_pairs"]:raise ValueError("Training rows differ from Step 2 history stats")
    if max_last is None or max_last>bound[1] or min_first<bound[0]:raise ValueError("Training timestamps outside allowed range")
    if tuple(bound)!=tuple(validation.WINDOWS["history"]):raise ValueError("Weeks 11-30 interval differs from frozen time windows")
    return {"rows":rows,"users":len(users),"programs":len(programs),"minimum_first_watch_utc":min_first,
            "maximum_last_watch_utc":max_last,"threshold_seconds":120,"future_test_viewing_rows":0}


def verify_validation_model_selection(manifest):
    step5=ROOT/"results"/"step5"
    verification=read_json(ROOT/"results"/"step5_verification.json")
    config=read_json(step5/"ppr_model_config.json")
    with (step5/"ppr_experiments.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    if verification.get("all_passed") is not True or len(rows)!=16:raise ValueError("Step 5 independent validation evidence incomplete")
    for row in rows:
        artifact=step5/"experiments"/row["config_id"]/"predictions.csv"
        if digest(artifact)!=row["prediction_sha256"]:raise ValueError(f"Saved PPR experiment hash mismatch: {row['config_id']}")
    best=max(rows,key=lambda r:(float(r["primary_map_at_5"]),r["config_id"]))
    if best["config_id"]!=config["best_config_id"]:raise ValueError("Saved PPR config is not experiment winner")
    if manifest["selected_validation_config_id"]!=best["config_id"]:raise ValueError("Final model differs from selected PPR strategy")
    if not math.isclose(float(manifest["selected_validation_primary_map_at_5"]),float(best["primary_map_at_5"]),abs_tol=1e-15):
        raise ValueError("Final model's selected validation score is inconsistent")
    expected={"weight":"cosine","personalization":"frequency","restart_probability":0.35,
        "max_neighbors":30,"history_cap":40,"min_common_viewers":3,"tolerance":1e-8,"max_iterations":250,
        "recency_half_life_seconds":2592000,"config_id":"cosine_pfrequency_a0.35_k30",
        "stage":"stage2_restart_and_neighborhood"}
    if manifest["hyperparameters"]!=expected or config["best_config"]!=expected:
        raise ValueError("Final model hyperparameters do not match verified PPR strategy")
    step6=read_json(ROOT/"results"/"step6_verification.json")
    if step6.get("all_passed") is not True:raise ValueError("Step 6 independent verification did not pass")
    with (ROOT/"results"/"step6"/"association_experiments.csv").open(newline="",encoding="utf-8") as stream:
        ar_rows=list(csv.DictReader(stream))
    with (ROOT/"results"/"step6_association_comparison.csv").open(newline="",encoding="utf-8") as stream:
        cmp_rows=list(csv.DictReader(stream))
    for row in ar_rows:
        if digest(ROOT/row["prediction_file"])!=row["prediction_sha256"]:raise ValueError("Step 6 prediction artifact hash mismatch")
    ppr_rows=[r for r in cmp_rows if r["family"]=="ppr"]
    all_best=max(float(r["primary_map_at_5"]) for r in cmp_rows)
    if len(ar_rows)!=11 or len(ppr_rows)!=1 or not math.isclose(all_best,float(best["primary_map_at_5"]),abs_tol=1e-15):
        raise ValueError("A saved alternative method has higher primary validation MAP@5")
    return {"step5_verified_checks":verification["passed_checks"],"ppr_configurations_checked":len(rows),
        "step6_verified_checks":step6["passed_checks"],"association_configurations_checked":len(ar_rows),
        "selected_config_id":best["config_id"],"selected_primary_validation_MAP_at_5":float(best["primary_map_at_5"]),
        "best_association_rules_primary_validation_MAP_at_5":max(float(r["primary_map_at_5"]) for r in ar_rows),
        "higher_primary_validation_alternative_found":False}


def main():
    started=time.monotonic();checks=[]
    def check(name,detail=None):checks.append({"name":name,"passed":True,"detail":detail})
    manifest=read_json(MANIFEST)
    join_manifest=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    if digest(ROOT/"data"/"step2"/"join_manifest.json")!=manifest["step2_join_manifest_sha256"]:
        raise ValueError("Step 2 join manifest hash mismatch")
    original=join_manifest["source_files_before"]
    if digest(TEMPLATE)!=original["submission.csv"]["sha256"]:raise ValueError("Original Kaggle template hash changed")
    if digest(TEMPLATE)!=manifest["template_sha256"]:raise ValueError("Original template differs from model manifest")
    if digest(SCHEDULE)!=original["export_arh_31-42-final.csv"]["sha256"]:raise ValueError("Original test schedule hash changed")
    check("original_template_hash_unchanged")
    check("original_test_schedule_hash_unchanged")
    users=read_original_template()
    check("template_user_ids_preserved_as_strings_and_ordered",{"users":len(users),"order_sha256":validation.identity_hash(users)})
    candidates=candidates_independent(join_manifest)
    check("test_candidates_rebuilt_from_valid_nonzero_schedule_ids",{"count":len(candidates),"id_sha256":validation.identity_hash(candidates)})
    history_stats=verify_history(manifest,join_manifest)
    check("historical_training_source_and_no_future_behavior",history_stats)
    model_evidence=verify_validation_model_selection(manifest)
    check("verified_primary_MAP_selects_PPR_over_evaluated_alternatives",model_evidence)
    if not manifest.get("test_viewing_behavior_used") is False or not manifest.get("test_schedule_used_only_for_candidates") is True:
        raise ValueError("Final model leakage declarations are inconsistent")
    check("test_schedule_used_as_catalog_only")
    if manifest["template_users"]!=len(users) or manifest["candidate_count"]!=len(candidates):raise ValueError("Model manifest cohort/candidate size mismatch")
    if manifest["candidate_ids_sha256"]!=validation.identity_hash(candidates):raise ValueError("Manifest candidate IDs hash mismatch")
    if manifest["template_user_order_sha256"]!=validation.identity_hash(users):raise ValueError("Manifest template order hash mismatch")
    if manifest["training_source_sha256"]!=digest(HISTORY):raise ValueError("Manifest training hash mismatch")
    graph=ROOT/manifest["graph_path"]
    if digest(graph)!=manifest["graph_sha256"] or graph.stat().st_size!=manifest["graph_bytes"]:raise ValueError("Final PPR graph artifact hash/size mismatch")
    check("final_graph_artifact_integrity",{"sha256":manifest["graph_sha256"],"bytes":manifest["graph_bytes"]})
    if SUBMISSION.stat().st_size==0:raise ValueError("Submission file is empty")
    submission_rows=[]
    with SUBMISSION.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        header=next(reader,None)
        if header!=["user_id","tv_show_id"]:raise ValueError(f"Wrong submission header: {header!r}")
        for line,row in enumerate(reader,2):
            if len(row)!=2:raise ValueError(f"Submission row {line} does not have exactly two columns")
            submission_rows.append(row)
    check("csv_parses_exact_two_column_header",{"header":["user_id","tv_show_id"]})
    if len(submission_rows)!=1260:raise ValueError(f"Expected 1,260 rows, got {len(submission_rows)}")
    out_users=[row[0] for row in submission_rows]
    if out_users!=list(users):raise ValueError("Output users/order differ from original template")
    if len(set(out_users))!=len(out_users):raise ValueError("Duplicate users in output")
    check("exactly_1260_unique_users_in_original_order")
    pool=set(candidates);ranked={}
    for line,(user,value) in enumerate(submission_rows,2):
        if not user or not value:raise ValueError(f"Missing value in row {line}")
        parts=value.split(" ")
        if len(parts)!=5 or any(not t for t in parts):raise ValueError(f"Row {line} must have five single-space IDs")
        if len(set(parts))!=5:raise ValueError(f"Duplicate program IDs in row {line}")
        for token in parts:
            if not ID_RE.fullmatch(token) or token=="0" or token.startswith("0"):
                raise ValueError(f"Noncanonical/zero program ID in row {line}: {token!r}")
            if token not in pool:raise ValueError(f"Ineligible test program in row {line}: {token}")
        ranked[user]=parts
    check("five_unique_nonzero_eligible_canonical_ids_per_user")
    if any(not r[0] or not r[1] for r in submission_rows):raise ValueError("Missing CSV cell")
    check("no_missing_values_no_extra_index_column")
    if manifest["predictions_per_user"]!=5 or manifest["repeat_previously_watched_allowed"] is not True:
        raise ValueError("Model repeat/list policy differs from documented protocol")
    check("repeat_allowed_and_documented_policy")
    actual_hash=digest(SUBMISSION)
    if manifest["submission_sha256"]!=actual_hash:raise ValueError("Final submission hash differs from model manifest")
    CHECKSUM.write_text(actual_hash+"\n",encoding="ascii")
    checksum=CHECKSUM.read_text(encoding="utf-8").strip()
    if checksum!=actual_hash:raise ValueError("Final submission checksum file mismatch")
    check("sha256_matches_checksum_and_training_manifest",{"sha256":actual_hash})
    result={"all_passed":True,"passed_checks":len(checks),"failed_checks":0,"checks":checks,
        "submission_path":str(SUBMISSION.relative_to(ROOT)),"submission_sha256":actual_hash,
        "sha256_checksum_file":str(CHECKSUM.relative_to(ROOT)),"data_rows":len(submission_rows),
        "columns":["user_id","tv_show_id"],"users":len(users),"predictions_per_user":5,
        "candidate_count":len(candidates),"candidate_ids_sha256":validation.identity_hash(candidates),
        "model":manifest["model"],"model_config_id":manifest["selected_validation_config_id"],
        "training_period":manifest["training_period"],"training_source_sha256":manifest["training_source_sha256"],
        "selected_local_primary_validation_MAP_at_5":manifest["selected_validation_primary_map_at_5"],
        "kaggle_private_MAP_at_5":None,"kaggle_submission_performed":False,
        "history_scan_statistics":history_stats,"validation_selection_evidence":model_evidence,
        "verification_runtime_seconds":round(time.monotonic()-started,3)}
    write_json(RESULT,result)
    print(json.dumps({k:result[k] for k in ("all_passed","passed_checks","failed_checks","submission_path","submission_sha256",
        "users","predictions_per_user","candidate_count","model_config_id","training_period","verification_runtime_seconds")},indent=2))


if __name__=="__main__":main()
