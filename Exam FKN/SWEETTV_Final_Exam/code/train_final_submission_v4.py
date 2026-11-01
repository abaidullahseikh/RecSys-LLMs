"""Build experimental LightGBM V4 from permitted history through Week 30."""

from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path
import platform
import resource
import sys
import time

import lightgbm as lgb
import numpy as np
import scipy

import association_rules as ar
import build_step10_candidates as step10
import content_based as cbf
import data_pipeline
import evaluation
import ppr
import popularity
import validation
from run_cbf_step9 import source_schedule_metadata

ROOT=validation.ROOT;SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
OUT=ROOT/"results"/"step10_lightgbm"
HISTORY=ROOT/"data"/"step2"/"threshold_120s"/"history_interactions.csv.gz"
HISTORY_OVERLAPS=ROOT/"data"/"step2"/"history_airing_overlaps.csv.gz"
TEMPLATE=SOURCE/"submission.csv"
SCHEDULE_TRAIN=SOURCE/"export_arh_11-20-final.csv"
SCHEDULE_VALID=SOURCE/"export_arh_21-30-final.csv"
SCHEDULE_TEST=SOURCE/"export_arh_31-42-final.csv"
SUBMISSION=ROOT/"submission"/"final_submission_v4.csv"
MANIFEST=ROOT/"results"/"final_model_v4_manifest.json"
FEATURES=ROOT/"results"/"step10_lightgbm"/"candidate_features_top50.csv.gz"
MODEL=ROOT/"results"/"step10_lightgbm"/"models"/"lightgbm_v4_final_refit_all_week21_30.txt"
STEP10=ROOT/"results"/"step10_lightgbm"/"lightgbm_training_result.json"
BUILD=ROOT/"results"/"step10_lightgbm"/"candidate_build_manifest.json"
PPR_PARAMS={"weight_type":"cosine","max_neighbors":30,"history_cap":40,"min_common_viewers":3,
    "personalization":"frequency","restart_probability":.35,"tolerance":1e-8,"max_iterations":250}
GENERATORS_V4=("ppr","personal_frequency","association_rules","cbf","popularity")
PROFILE_WEIGHT="airings"


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def require(flag,message):
    if not flag:raise RuntimeError(message)


def load_users():
    with TEMPLATE.open(newline="",encoding="utf-8") as stream:
        reader=csv.reader(stream)
        require(next(reader,None)==["user_id","tv_show_id"],"Template schema mismatch")
        users=[]
        for row in reader:
            require(len(row)==2,"Malformed original template record")
            users.append(evaluation.user_id(row[0]))
    require(len(users)==1260 and len(set(users))==1260,"Template user IDs/order invalid")
    return tuple(users)


def test_pool_and_metadata():
    docs_rows=[];items=set();stats=defaultdict(lambda:{"airings":0,"duration_total":0,"duration_max":0,
        "channels":Counter(),"hours":Counter(),"weeks":set()})
    lo,hi=validation.WINDOWS["test"]
    with SCHEDULE_TEST.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==data_pipeline.SCHEDULE_COLUMNS,"Test schedule schema mismatch")
        for row in reader:
            item=validation.schedule_program_id(row["tv_show_id"])
            start=data_pipeline.parse_timestamp(row["start_time"],schedule=True);duration=int(row["duration"])
            if item=="0" or duration<=0:continue
            items.add(item);docs_rows.append((item,row))
            info=stats[item];info["airings"]+=1;info["duration_total"]+=duration
            info["duration_max"]=max(info["duration_max"],duration);info["channels"][row["channel_id"]]+=1
            local=datetime.fromtimestamp(start,tz=timezone.utc).astimezone(data_pipeline.ZONE)
            info["hours"][local.hour]+=1
            info["weeks"].add(max(0,min(11,(start-lo)//(7*86400))))
    pool=evaluation.candidate_pool(items)
    return pool,cbf.aggregate_metadata_rows(docs_rows),stats


def load_full_profiles():
    im=read_json(ROOT/"data"/"step2"/"threshold_120s"/"interaction_manifest.json")
    require(digest(HISTORY)==im["artifacts"][HISTORY.name]["sha256"],"Weeks11-30 history hash mismatch")
    bounds=validation.WINDOWS["history"];profiles={};viewers=Counter();airings=Counter();seconds=Counter();rows=0;prev=None
    with gzip.open(HISTORY,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream);require(reader.fieldnames==validation.INTERACTION_COLUMNS,"History schema mismatch")
        for row in reader:
            u=evaluation.user_id(row["user_id"]);item=validation.program_id(row["tv_show_id"])
            count,secs,first,last=map(int,(row["qualifying_airings"],row["watched_seconds"],row["first_watch_utc"],row["last_watch_utc"]))
            key=(u,int(item));require(prev is None or key>prev,"Duplicate/unsorted history pair");prev=key
            require(count>0 and secs>=count*120 and bounds[0]<=first<last<=bounds[1],"History threshold/window violation")
            profiles.setdefault(u,{})[item]=(count,secs,last);viewers[item]+=1;airings[item]+=count;seconds[item]+=secs;rows+=1
    require(rows==im["periods"]["history"]["unique_user_program_pairs"],"History row count differs from Step2 manifest")
    nodes=tuple(sorted(viewers,key=int));data=ppr.TrainingProfiles(nodes,{i:n for n,i in enumerate(nodes)},profiles,dict(viewers),digest(HISTORY),bounds,rows,0)
    pop=popularity.PopularityModel((),dict(viewers),dict(airings),dict(seconds),digest(HISTORY),bounds,rows,0)
    return data,pop,im


def history_airing_features():
    join=read_json(ROOT/"data"/"step2"/"join_manifest.json")
    require(digest(HISTORY_OVERLAPS)==join["artifacts"][HISTORY_OVERLAPS.name]["sha256"],"History overlap hash mismatch")
    lo,hi=validation.WINDOWS["history"];pair=defaultdict(Counter);weeks=defaultdict(set);first_last={}
    user_channels=defaultdict(Counter);user_hours=defaultdict(Counter);user_weeks=defaultdict(Counter)
    with gzip.open(HISTORY_OVERLAPS,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        for row in reader:
            user=row["user_id"];item=validation.program_id(row["tv_show_id"])
            if item=="0":continue
            overlap=int(row["overlap_seconds"]);start=int(row["broadcast_start_utc"]);end=int(row["broadcast_end_utc"])
            first,last=int(row["first_overlap_utc"]),int(row["last_overlap_utc"])
            if not lo<=first<last<=hi or end<=start:raise ValueError("Full-history airing outside period")
            frac=overlap/(end-start);key=(user,item);cnt=pair[key]
            cnt["positive_airings"]+=1;cnt["overlap_seconds"]+=overlap;cnt["fraction_sum"]+=frac
            if overlap>=120:cnt["airings_ge_120"]+=1
            if overlap>=300:cnt["airings_ge_300"]+=1
            if overlap>=600:cnt["airings_ge_600"]+=1
            if 5*overlap>=4*(end-start):cnt["airings_fraction80"]+=1
            bucket=max(0,min(19,(first-lo)//(7*86400)));weeks[key].add(bucket)
            if overlap>=120:user_weeks[user][bucket]+=1
            old=first_last.get(key,(first,last));first_last[key]=(min(old[0],first),max(old[1],last))
            if overlap>=120:
                user_channels[user][row["channel_id"]]+=1
                local=datetime.fromtimestamp(start,tz=timezone.utc).astimezone(data_pipeline.ZONE)
                user_hours[user][local.hour]+=1
    return pair,weeks,first_last,user_channels,user_hours,user_weeks


def target_schedule_affinity(schedule_stats,user_channels,user_hours):
    channel_pref={};hour_pref={}
    for user,c in user_channels.items():
        total=sum(c.values()) or 1;channel_pref[user]={k:v/total for k,v in c.items()}
    for user,c in user_hours.items():
        total=sum(c.values()) or 1;hour_pref[user]={k:v/total for k,v in c.items()}
    return channel_pref,hour_pref


def ranks_for_models(graph,data,pop,profiles,ar_model,cbf_matrix,user_vectors,candidates,users):
    fallback=pop.__class__(candidates,pop.distinct_users,pop.qualifying_airings,pop.watched_seconds,
        pop.training_sha256,pop.training_window_utc,pop.training_rows,0).ranking("distinct_users")
    fallback_pos={x:i for i,x in enumerate(fallback)};candidate_pos={x:i for i,x in enumerate(candidates)}
    node_pos={x:i for i,x in enumerate(graph.nodes)}
    rankings={g:{} for g in ("ppr","personal_frequency","association_rules","cbf","popularity")}
    scores={g:{} for g in ("ppr","association_rules","cbf")};cbf_stats={}
    for uidx,user in enumerate(users):
        profile=profiles.get(user,{})
        if profile:
            restart=ppr.personalization_vector(profile,data,variant="frequency")
            vector,_=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
            pscore={x:float(vector[node_pos[x]]) for x in candidates if x in node_pos and vector[node_pos[x]]>0}
            ppos=sorted(pscore,key=lambda x:(-pscore[x],fallback_pos[x],int(x)))
            ascore,_=ar.score_user(ar_model,profile,candidates,score_type="confidence_log_lift",antecedent_weight="log_frequency")
            apos=sorted(ascore,key=lambda x:(-ascore[x],-ar_model.dataset.item_support.get(x,0),int(x)))
        else:pscore={};ppos=[];ascore={};apos=[]
        pseen=set(ppos);rankings["ppr"][user]=tuple(ppos+[x for x in fallback if x not in pseen]);scores["ppr"][user]=pscore
        values=profile
        fpos=sorted((x for x in values if x in candidate_pos),key=lambda x:(-values[x][0],int(x)))
        fseen=set(fpos);rankings["personal_frequency"][user]=tuple(fpos+[x for x in fallback if x not in fseen])
        aseen=set(apos);rankings["association_rules"][user]=tuple(apos+[x for x in fallback if x not in aseen]);scores["association_rules"][user]=ascore
        sim=(user_vectors.getrow(uidx)@cbf_matrix.T).tocsr();left,right=sim.indptr[0:2]
        cscore={candidates[int(i)]:float(s) for i,s in zip(sim.indices[left:right],sim.data[left:right]) if s>0}
        cpos=sorted(cscore,key=lambda x:(-cscore[x],fallback_pos[x],int(x)))
        cseen=set(cpos);rankings["cbf"][user]=tuple(cpos+[x for x in fallback if x not in cseen]);scores["cbf"][user]=cscore
        rankings["popularity"][user]=fallback
    return rankings,scores,fallback


def feature_matrix_for_test(data,pop,users,candidates,rankings,scores,profile_stats,pair,week_sets,first_last,
        user_channels,user_hours,user_weeks,candidate_schedule,profile_metadata_docs,candidate_metadata_docs,feature_columns):
    channel_pref,hour_pref=target_schedule_affinity(candidate_schedule,user_channels,user_hours)
    user_meta={"category":defaultdict(Counter),"genre":defaultdict(Counter)}
    for user in users:
        for item,(count,_,_) in data.profiles.get(user,{}).items():
            doc=profile_metadata_docs.get(item,{})
            for f in ("category","genre"):
                for term in doc.get(f,set()):user_meta[f][user][term]+=count
    feature_rows=[];row_users=[];row_items=[];cand_union={}
    for user in users:
        rank_maps={g:{item:rank for rank,item in enumerate(rankings[g][user])}
            for g in ("ppr","personal_frequency","association_rules","cbf","popularity")}
        per_source={g:rankings[g][user][:50] for g in rankings}
        candidate_set=set().union(*(set(x) for x in per_source.values()))
        cand_union[user]=candidate_set
        all_prof=data.profiles.get(user,{})
        total_air=sum(v[0] for v in all_prof.values());total_sec=sum(v[1] for v in all_prof.values())
        cat_count=user_meta["category"][user];genre_count=user_meta["genre"][user]
        cat_total=sum(cat_count.values()) or 1;genre_total=sum(genre_count.values()) or 1
        ch_pref=channel_pref.get(user,{});hr_pref=hour_pref.get(user,{})
        active_user_weeks=len(user_weeks.get(user,{}))
        for item in sorted(candidate_set,key=int):
            old=all_prof.get(item,(0,0,None));h=pair.get((user,item),Counter());last=first_last.get((user,item),(None,None))[1]
            recency=(data.training_window_utc[1]-last)/86400 if last is not None else 0.
            active_weeks=len(week_sets.get((user,item),set()))
            sch=candidate_schedule.get(item,{"airings":0,"duration_total":0,"duration_max":0,"channels":Counter(),"hours":Counter(),"weeks":set()})
            channel_total=sum(sch["channels"].values()) or 1;hour_total=sum(sch["hours"].values()) or 1
            channel_aff=sum(ch_pref.get(k,0)*(v/channel_total) for k,v in sch["channels"].items())
            hour_aff=sum(hr_pref.get(k,0)*(v/hour_total) for k,v in sch["hours"].items())
            entropy=-math.fsum((n/hour_total)*math.log(n/hour_total) for n in sch["hours"].values() if n)
            doc=candidate_metadata_docs.get(item,{})
            cat_aff=sum(cat_count.get(x,0) for x in doc.get("category",set()))/cat_total
            genre_aff=sum(genre_count.get(x,0) for x in doc.get("genre",set()))/genre_total
            src={g:rank_maps[g].get(item,10**6) for g in rankings}
            member={g:src[g]<50 for g in rankings}
            row={"candidate_source_count":sum(member.values()),"source_ppr":int(member["ppr"]),
                "source_personal_frequency":int(member["personal_frequency"]),"source_association_rules":int(member["association_rules"]),
                "source_cbf":int(member["cbf"]),"source_popularity":int(member["popularity"]),
                "ppr_score":scores["ppr"][user].get(item,0),"ppr_rank":src["ppr"],
                "personal_frequency_score":all_prof.get(item,(0,0,None))[0],"personal_frequency_rank":src["personal_frequency"],
                "association_rules_score":scores["association_rules"][user].get(item,0),"association_rules_rank":src["association_rules"],
                "cbf_similarity":scores["cbf"][user].get(item,0),"cbf_rank":src["cbf"],
                "pop_distinct_users":pop.distinct_users.get(item,0),"pop_qualifying_airings":pop.qualifying_airings.get(item,0),
                "pop_watched_seconds":pop.watched_seconds.get(item,0),"popularity_rank":src["popularity"],
                "user_history_program_count":len(all_prof),"user_history_airings_total":total_air,
                "user_history_seconds_total":total_sec,"user_cold_start":int(not bool(all_prof)),
                "user_history_active_weeks":active_user_weeks,"user_program_seen":int(old[0]>0),
                "user_program_airings":old[0],"user_program_watched_seconds":old[1],"user_program_log_airings":math.log1p(old[0]),
                "user_program_recency_days":recency,"airings_ge_120":h["airings_ge_120"],
                "airings_ge_300":h["airings_ge_300"],"airings_ge_600":h["airings_ge_600"],
                "airings_fraction80":h["airings_fraction80"],
                "mean_airing_fraction":h["fraction_sum"]/h["positive_airings"] if h["positive_airings"] else 0,
                "program_active_weeks":active_weeks,"program_week_consistency":active_weeks/20,
                "program_airings_recent_2w":h["recent_2w"],"program_airings_recent_4w":h["recent_4w"],
                "program_airings_recent_8w":h["recent_8w"],"target_airing_count":sch["airings"],
                "target_duration_total":sch["duration_total"],"target_duration_mean":sch["duration_total"]/sch["airings"] if sch["airings"] else 0,
                "target_duration_max":sch["duration_max"],"target_channel_count":len(sch["channels"]),
                "target_schedule_week_count":len(sch["weeks"]),"target_unique_hours":len(sch["hours"]),
                "target_hour_entropy":entropy,"candidate_channel_affinity":channel_aff,"candidate_hour_affinity":hour_aff,
                "user_category_affinity":cat_aff,"user_genre_affinity":genre_aff}
            feature_rows.append(np.asarray([float(row[name]) for name in feature_columns],dtype=np.float32))
            row_users.append(user);row_items.append(item)
    return np.vstack(feature_rows),row_users,row_items,cand_union


def main():
    if (ROOT/"submission"/"final_submission_v4.csv").exists():raise FileExistsError("V4 already exists; preserve previous output")
    start_all=time.monotonic()
    manifest=read_json(BUILD);training_result=read_json(OUT/"lightgbm_training_result.json")
    if training_result["status"]!="trained_and_evaluated":raise RuntimeError("LightGBM validation training did not complete")
    if not training_result.get("final_refit_model_path"):raise RuntimeError("Final LightGBM refit artifact is missing")
    users=load_users()
    # Never modify V1/V2/V3; V4 is an additional, experimental submission.
    v1_manifest=read_json(ROOT/"results"/"final_model_manifest.json")
    v2_manifest=read_json(ROOT/"results"/"final_model_v2_manifest.json")
    v3_manifest=read_json(ROOT/"results"/"final_model_v3_manifest.json")
    candidates,candidate_docs,test_schedule_stats=test_pool_and_metadata()
    require(len(candidates)==6636,"V4 candidate pool differs from named test schedule")
    require(users==validation.EvaluationProtocol().cohort,"V4 template differs from frozen cohort")
    data,pop,history_manifest=load_full_profiles();history_loaded=time.monotonic()
    graph=ppr.build_graph(data,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    graph_fit_done=time.monotonic()
    train_docs=source_schedule_metadata([SCHEDULE_TRAIN,SCHEDULE_VALID],validation.WINDOWS["history"])
    profile_stats=cbf.FIELD_WEIGHT_CONFIGS["channel_context_emphasis"]
    space=cbf.FeatureSpace.fit(train_docs,profile_stats)
    cbf_matrix=space.transform(candidate_docs,candidates,profile_stats)
    train_index={item:i for i,item in enumerate(space.training_items)}
    user_vectors,cbf_profile_stats=cbf.build_user_profiles(data.profiles,space.training_items,train_index,
        space.training_matrix,users,weighting=PROFILE_WEIGHT)

    # Refit the validated 14-day Association Rules component on all Weeks 11-30.
    ar_start=time.monotonic()
    with gzip.open(HISTORY_OVERLAPS,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        transactions=ar.time_window_transactions_from_airings(reader,data.training_window_utc,14,120)
    ar_dataset=ar.build_dataset(transactions,transaction_type="time_window",window_days=14,
        training_sha256=data.training_sha256,training_window_utc=data.training_window_utc,
        source_sha256=digest(HISTORY_OVERLAPS),transaction_build_seconds=time.monotonic()-ar_start)
    ar_model=ar.mine_rules(ar_dataset,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.0,top_neighbors=50)

    pair,weeks,first_last,user_channels,user_hours,user_weeks=step10.load_airing_features(
        validation.EvaluationProtocol(),data.profiles,overlap_path=HISTORY_OVERLAPS,window=data.training_window_utc)
    rankings,scores,fallback=ranks_for_models(graph,data,pop,data.profiles,ar_model,cbf_matrix,user_vectors,candidates,users)
    feature_columns=training_result["feature_columns"]
    feature_started=time.monotonic()
    test_matrix,row_users,row_items,candidate_sets=feature_matrix_for_test(data,pop,users,candidates,rankings,scores,
        cbf_profile_stats,pair,weeks,first_last,user_channels,user_hours,user_weeks,test_schedule_stats,
        train_docs,candidate_docs,feature_columns)
    if test_matrix.shape[1]!=len(feature_columns):raise RuntimeError("V4 feature schema differs from LightGBM training schema")
    feature_path=OUT/"v4_test_candidate_features.csv.gz"
    with gzip.open(feature_path,"wt",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","tv_show_id",*feature_columns])
        writer.writerows((u,i,*map(float,row)) for u,i,row in zip(row_users,row_items,test_matrix))

    model_path=ROOT/training_result["final_refit_model_path"]
    require(digest(model_path)==training_result["final_refit_model_sha256"],"Final LightGBM model hash mismatch")
    booster=lgb.Booster(model_file=str(model_path));prediction_scores=booster.predict(test_matrix,num_threads=4)
    fallback_rank={item:i for i,item in enumerate(fallback)};indices=defaultdict(list)
    for r,user in enumerate(row_users):indices[user].append(r)
    predictions={};candidate_counts={}
    for user in users:
        rows=indices.get(user,[]);candidate_counts[user]=len(rows)
        if not data.profiles.get(user):
            predictions[user]=tuple(fallback[:5])
            continue
        ordered=sorted(rows,key=lambda r:(-float(prediction_scores[r]),fallback_rank[row_items[r]],int(row_items[r])))
        selected=[];seen=set()
        for r in ordered:
            item=row_items[r]
            if item not in seen:selected.append(item);seen.add(item)
            if len(selected)==5:break
        for item in fallback:
            if len(selected)>=5:break
            if item not in seen:selected.append(item);seen.add(item)
        require(len(selected)==5 and len(set(selected))==5,"LightGBM reranker failed to complete Top-5")
        predictions[user]=tuple(selected)
    if any(not set(predictions[u]).issubset(set(candidates)) for u in users):raise RuntimeError("V4 candidate eligibility failed")
    if tuple(predictions)!=users:raise RuntimeError("V4 template order mismatch")
    output=ROOT/"submission"/"final_submission_v4.csv"
    with output.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","tv_show_id"])
        for user in users:writer.writerow([user," ".join(predictions[user])])
    result={"version":4,"status":"experimental_not_recommended_final","recommended_final":"submission/final_submission_v2.csv",
        "model":"LightGBM binary candidate reranker over PPR+Personal Frequency+Association Rules+CBF candidates/features",
        "lightgbm_version":lgb.__version__,"lightgbm_model_path":str(model_path.relative_to(ROOT)),
        "lightgbm_model_sha256":digest(model_path),"selected_params":training_result["selected_params"],
        "final_boosting_rounds":training_result["final_boosting_rounds_from_group_cv_median"],
        "binary_training_label":"Weeks21-30 per-user Top-5 by qualifying-airing frequency; label training rows are historical by Weeks31-42 inference",
        "rank_feature_candidate_limit":50,"candidate_generators":["PPR","Personal Frequency","Association Rules","CBF","Popularity"],
        "training_label_examples":training_result["final_refit_training_rows"],"candidate_feature_rows":test_matrix.shape[0],
        "test_candidate_count":len(candidates),"average_union_candidates_per_user":float(np.mean(list(candidate_counts.values()))),
        "max_union_candidates_per_user":max(candidate_counts.values()),"users_with_under_five_candidates":sum(x<5 for x in candidate_counts.values()),
        "candidate_source":"All valid positive-duration nonzero program IDs in export_arh_31-42-final.csv",
        "training_period":"Weeks11-30; user-item interactions matched at >=120 seconds per airing",
        "training_history_sha256":data.training_sha256,"training_history_rows":data.rows,
        "training_schedule_metadata_period":"Weeks11-30; IDF corpus is training-period schedule metadata only",
        "candidate_schedule_metadata_path":str(SCHEDULE_TEST),"future_viewing_behavior_used":False,
        "user_id_preserved_as_string":True,"original_template_order_sha256":evaluation.identity_hash(users),
        "columns":["user_id","tv_show_id"],"predictions_per_user":5,"repeat_previously_watched_allowed":True,
        "fallback":"Full-history distinct-viewer popularity, then qualifying airings, then numeric ID",
        "v2_validation_holdout_secondary_map_at_5":training_result["same_heldout_baselines"]["v2_ppr_personal_frequency"]["secondary_map_at_5"],
        "lightgbm_validation_holdout_secondary_map_at_5":training_result["heldout_metrics"]["top50_secondary_map_at_5"],
        "LightGBM_validation_holdout_primary_map_at_5":training_result["heldout_metrics"]["top50_primary_map_at_5"],
        "kaggle_score":None,"kaggle_submission_performed":False,
        "v1_sha256_preserved":v1_manifest["submission_sha256"],"v2_sha256_preserved":v2_manifest["submission_sha256"],
        "v3_sha256_preserved":v3_manifest["submission_sha256"],
        "candidate_features_path":str(feature_path.relative_to(ROOT)),"candidate_features_sha256":digest(feature_path),
        "submission_path":str(output.relative_to(ROOT)),"submission_sha256":digest(output),
        "runtime_seconds":{"history_profiles_and_stats":history_loaded-start_all,"ppr_graph_fit":graph_fit_done-history_loaded,
            "metadata_and_ar_fit":time.monotonic()-graph_fit_done,"candidate_feature_and_prediction":time.monotonic()-feature_started},
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"lightgbm":lgb.__version__}}
    write_json(ROOT/"results"/"final_model_v4_manifest.json",result)
    print(json.dumps({k:result[k] for k in ("submission_path","submission_sha256","status","test_candidate_count",
        "average_union_candidates_per_user","lightgbm_validation_holdout_secondary_map_at_5",
        "v2_validation_holdout_secondary_map_at_5","kaggle_submission_performed")},indent=2))


if __name__=="__main__":main()
