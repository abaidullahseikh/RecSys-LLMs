"""Leakage-safe, group-split LightGBM reranking experiment (bounded grid)."""

from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

import evaluation
import popularity
import validation

ROOT=validation.ROOT
OUT=ROOT/"results"/"step10_lightgbm"
FEATURES=OUT/"candidate_features_top50.csv.gz"
BUILD=OUT/"candidate_build_manifest.json"


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def candidate_sets(path,cohort):
    result={}
    with gzip.open(path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=["user_id","candidate_program_ids","candidate_count"]:raise ValueError("Candidate set schema mismatch")
        for row in reader:
            items=tuple(row["candidate_program_ids"].split()) if row["candidate_program_ids"] else ()
            if len(items)!=int(row["candidate_count"]) or len(items)!=len(set(items)):raise ValueError("Malformed candidate set")
            result[row["user_id"]]=items
    if tuple(result)!=tuple(cohort):raise ValueError("Candidate set user order mismatch")
    return result


def read_features(feature_columns):
    excluded={"user_id","tv_show_id","candidate_limit","group_split","label_top5_freq120","label_all_120","label_top5_fraction80"}
    columns=[x for x in feature_columns if x not in excluded]
    X=[];y=[];labels={};users=[];items=[];groups=[]
    with __import__("gzip").open(FEATURES,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=feature_columns:raise ValueError("Feature table header differs from candidate manifest")
        for row in reader:
            X.append([float(row[k]) for k in columns]);y.append(int(row["label_top5_freq120"]))
            users.append(row["user_id"]);items.append(row["tv_show_id"]);groups.append(row["group_split"])
            labels[(row["user_id"],row["tv_show_id"])]=(int(row["label_all_120"]),int(row["label_top5_fraction80"]))
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int8),columns,users,items,groups,labels


def group_map_at5(predictions,truth,users):
    scores=[]
    for user in users:
        relevant=truth[user];hits=0;seen=set();terms=[]
        for rank,item in enumerate(predictions[user][:5],1):
            if item in relevant and item not in seen:
                seen.add(item);hits+=1;terms.append(hits/rank)
        scores.append(math.fsum(terms)/min(5,len(relevant)) if relevant else 0.)
    return math.fsum(scores)/len(scores) if scores else 0.


def predictions_for_group(model,X,users,items,groups,target_group,candidate_limit,candidate_map,fallback,cohort_order,cold_start_column):
    rows_by_user=defaultdict(list)
    mask=[i for i,u in enumerate(users) if groups[i]==target_group]
    if not mask:return {}
    score=model.predict(X[mask])
    for i,s in zip(mask,score):rows_by_user[users[i]].append((items[i],float(s),float(X[i,cold_start_column])))
    result={};eligible_fallback=tuple(fallback)
    for user in cohort_order:
        if user not in rows_by_user:continue
        if all(cold>0.5 for _,_,cold in rows_by_user[user]):
            result[user]=eligible_fallback[:5]
            continue
        allowed=set(candidate_map[user]);local=[(x,s) for x,s,cold in rows_by_user[user] if x in allowed]
        local.sort(key=lambda p:(-p[1],int(p[0])))
        chosen=[];seen=set()
        for item,_ in local:
            if item not in seen:chosen.append(item);seen.add(item)
            if len(chosen)==5:break
        for item in eligible_fallback:
            if len(chosen)>=5:break
            if item not in seen:chosen.append(item);seen.add(item)
        result[user]=tuple(chosen)
    return result


def feature_ablation(columns,ablation):
    if ablation=="full":return list(range(len(columns)))
    if ablation=="no_cbf":
        remove={"cbf_similarity","cbf_rank","user_category_affinity","user_genre_affinity"}
    elif ablation=="no_ppr":remove={"ppr_score","ppr_rank"}
    else:raise ValueError(ablation)
    return [i for i,name in enumerate(columns) if name not in remove]


def main():
    started=time.monotonic();result_path=OUT/"lightgbm_training_result.json"
    if importlib.util.find_spec("lightgbm") is None:
        result={"status":"unavailable_dependency","lightgbm_available":False,
            "reason":"LightGBM is not installed in the project environment; PyPI package installation timed out and no wheel/source is cached.",
            "install_attempt":".venv/bin/python3 -m pip install lightgbm",
            "candidate_data_ready":FEATURES.is_file(),"candidate_manifest_ready":BUILD.is_file(),
            "model_training_performed":False,"v3_generated":False,"runtime_seconds":round(time.monotonic()-started,3)}
        write_json(result_path,result);print(json.dumps(result,indent=2));return

    import lightgbm as lgb
    manifest=json.loads(BUILD.read_text());protocol=validation.EvaluationProtocol()
    users=protocol.cohort;features=manifest["feature_columns"]
    X,y,columns,row_users,row_items,row_groups,label_details=read_features(features)
    cold_start_column=columns.index("user_cold_start")
    top20=candidate_sets(OUT/"candidate_sets_top20.csv.gz",users)
    top50=candidate_sets(OUT/"candidate_sets_top50.csv.gz",users)
    eval_users={u for u,g in zip(row_users,row_groups) if g=="evaluation_users"}
    train_users=set(users)-eval_users
    if not eval_users or not train_users or eval_users&train_users:raise ValueError("Bad user-group split")
    secondary={}
    for u,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
            ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):secondary[u]=frozenset(field.split()) if field else frozenset()
    pop_model=popularity.fit_popularity(protocol.config,protocol.candidates)
    fallback=pop_model.ranking("distinct_users")
    # Refit globally on historical training users only after choosing configs with user-group CV.
    row_train=np.asarray([g=="training_users" for g in row_groups]);groups_arr=np.asarray(row_users)
    base_params={"objective":"binary","learning_rate":.05,"max_depth":-1,
        "bagging_fraction":.8,"bagging_freq":1,"feature_fraction":.8,"lambda_l2":1.0,
        "seed":2026,"feature_fraction_seed":2026,"bagging_seed":2026,"data_random_seed":2026,
        "num_threads":4,"verbosity":-1}
    configs=[{"num_leaves":15,"min_child_samples":100},{"num_leaves":31,"min_child_samples":100}]
    cv_results=[]
    training_row_indices=np.flatnonzero(row_train);training_users_rows=groups_arr[row_train]
    cv_user_fold={u:int(hashlib.sha256(("cv:"+u).encode("ascii")).hexdigest()[:8],16)%3 for u in train_users}
    row_fold=np.asarray([cv_user_fold[u] for u in training_users_rows],dtype=np.int8)
    for spec in configs:
        fold_scores=[];iterations=[]
        for fold in range(3):
            tr=np.flatnonzero(row_fold!=fold);va=np.flatnonzero(row_fold==fold)
            xtrain=X[training_row_indices[tr]];ytrain=y[training_row_indices[tr]]
            xvalid=X[training_row_indices[va]];yvalid=y[training_row_indices[va]]
            train_data=lgb.Dataset(xtrain,label=ytrain,free_raw_data=True)
            valid_data=lgb.Dataset(xvalid,label=yvalid,reference=train_data,free_raw_data=True)
            params={**base_params,**spec,"scale_pos_weight":float(np.count_nonzero(ytrain==0))/max(1,np.count_nonzero(ytrain==1))}
            model=lgb.train(params,train_data,num_boost_round=250,valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(0)])
            fold_users=tuple(dict.fromkeys(groups_arr[training_row_indices[va]].tolist()))
            pred=predictions_for_group(model,X,row_users,row_items,row_groups,"training_users",50,top50,fallback,users,cold_start_column)
            fold_subset={u:pred[u] for u in fold_users if u in pred}
            score=group_map_at5(fold_subset,secondary,tuple(fold_subset))
            fold_scores.append(score);iterations.append(model.best_iteration or 250)
        cv_results.append({"params":spec,"fold_secondary_map_at_5":fold_scores,
            "mean_secondary_map_at_5":float(np.mean(fold_scores)),"best_iterations":iterations})
    best=max(cv_results,key=lambda r:(r["mean_secondary_map_at_5"],-r["params"]["num_leaves"]))
    final_params={**base_params,**best["params"],"scale_pos_weight":float(np.count_nonzero(y[row_train]==0))/max(1,np.count_nonzero(y[row_train]==1))}
    final_rounds=max(1,int(np.median(best["best_iterations"])))
    final_model=lgb.train(final_params,lgb.Dataset(X[row_train],label=y[row_train]),
        num_boost_round=final_rounds,callbacks=[lgb.log_evaluation(0)])
    hold50=predictions_for_group(final_model,X,row_users,row_items,row_groups,"evaluation_users",50,top50,fallback,users,cold_start_column)
    hold20=predictions_for_group(final_model,X,row_users,row_items,row_groups,"evaluation_users",20,top20,fallback,users,cold_start_column)
    heldout_order=tuple(u for u in users if u in eval_users)
    eval_all=protocol.truth
    metrics={"top20_secondary_map_at_5":group_map_at5(hold20,secondary,heldout_order),
        "top50_secondary_map_at_5":group_map_at5(hold50,secondary,heldout_order),
        "top20_primary_map_at_5":group_map_at5(hold20,eval_all,heldout_order),
        "top50_primary_map_at_5":group_map_at5(hold50,eval_all,heldout_order)}
    # Baselines are measured on exactly the same held-out users.
    step4_run=validation.json_read(ROOT/"results"/"step4"/"step4_run.json")
    baseline_files={"popularity":ROOT/step4_run["variants"]["popularity_distinct_users"]["prediction_file"],
        "personal_frequency":ROOT/"results"/"step4"/"personal_frequency_qualifying_airings_predictions.csv",
        "ppr":ROOT/"results"/"step5"/"experiments"/"cosine_pfrequency_a0.35_k30"/"predictions.csv",
        "association_rules":ROOT/"results"/"step6"/"experiments"/"time_window_14d_lift_log"/"predictions.csv",
        "v2_ppr_personal_frequency":ROOT/"results"/"step8"/"validation_predictions"/"rrf_ppr_personal_frequency_ppr_weight_0.50"/"predictions.csv"}
    baseline_metrics={}
    for name,path in baseline_files.items():
        preds=evaluation.parse_prediction_records(list(csv.DictReader(path.open(newline="",encoding="utf-8"))))
        subset={u:preds[u] for u in heldout_order}
        baseline_metrics[name]={"primary_map_at_5":group_map_at5(subset,eval_all,heldout_order),
            "secondary_map_at_5":group_map_at5(subset,secondary,heldout_order)}
    ablations={}
    for name in ("full","no_cbf","no_ppr"):
        cols=feature_ablation(columns,name)
        model=lgb.train(final_params,lgb.Dataset(X[row_train][:,cols],label=y[row_train]),
            num_boost_round=final_rounds,callbacks=[lgb.log_evaluation(0)])
        preds=predictions_for_group(model,X[:,cols],row_users,row_items,row_groups,"evaluation_users",50,top50,fallback,users,cols.index(cold_start_column))
        ablations[name]={"secondary_map_at_5":group_map_at5(preds,secondary,heldout_order),
            "primary_map_at_5":group_map_at5(preds,eval_all,heldout_order),"feature_count":len(cols)}
    model_dir=OUT/"models";model_dir.mkdir(exist_ok=True)
    model_path=model_dir/"lightgbm_top50_binary.txt";final_model.save_model(str(model_path))
    importance=sorted(zip(columns,final_model.feature_importance(importance_type="gain").tolist()),key=lambda p:-p[1])
    # After leakage-safe user-group evaluation is frozen, refit on all permitted
    # Week 21-30 historical labels for future Weeks 31-42 inference.
    final_refit_params={**base_params,**best["params"],
        "scale_pos_weight":float(np.count_nonzero(y==0))/max(1,np.count_nonzero(y==1))}
    final_refit=lgb.train(final_refit_params,lgb.Dataset(X,label=y),num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(0)])
    final_refit_path=model_dir/"lightgbm_v4_final_refit_all_week21_30.txt"
    final_refit.save_model(str(final_refit_path))
    output={"status":"trained_and_evaluated","lightgbm_available":True,"objective":"binary",
        "label":"validation top-five by qualifying-airing frequency (secondary target); training users only",
        "user_group_split":{"train_users":len(train_users),"evaluation_users":len(eval_users),"evaluation_user_ids_sha256":hashlib.sha256("\n".join(heldout_order).encode()).hexdigest()},
        "candidate_limit_train":50,"candidate_limit_evaluation":[20,50],"cv_results":cv_results,"selected_params":final_params,
        "final_boosting_rounds_from_group_cv_median":final_rounds,
        "heldout_metrics":metrics,"same_heldout_baselines":baseline_metrics,"ablations":ablations,
        "feature_importance_gain_top30":importance[:30],"feature_columns":columns,
        "model_path":str(model_path.relative_to(ROOT)),"model_sha256":hashlib.file_digest(model_path.open("rb"),"sha256").hexdigest(),
        "final_refit_model_path":str(final_refit_path.relative_to(ROOT)),
        "final_refit_model_sha256":hashlib.file_digest(final_refit_path.open("rb"),"sha256").hexdigest(),
        "final_refit_training_users":len(set(users)),"final_refit_training_rows":len(y),
        "final_refit_target_period_used_as_historical_label":"Weeks 21-30; future inference target is Weeks 31-42",
        "fit_rows":int(row_train.sum()),"fit_positive_rows":int(y[row_train].sum()),
        "runtime_seconds":round(time.monotonic()-started,3),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "training_features_only_from_weeks11_20":True,"heldout_target_user_ids_not_used_in_training":True,
        "no_test_viewing_behavior_used":True,"v4_generated":False}
    write_json(OUT/"lightgbm_training_result.json",output)
    print(json.dumps({k:output[k] for k in ("status","selected_params","heldout_metrics","same_heldout_baselines","ablations",
        "fit_rows","fit_positive_rows","runtime_seconds","peak_rss_mib")},indent=2))


if __name__=="__main__":main()
