"""Build leak-safe multi-generator candidates and LightGBM feature rows."""

from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time
from datetime import datetime,timezone

import numpy as np

import association_rules as ar
import content_based as cbf
import data_pipeline
import evaluation
import ppr
import personal_frequency
import popularity
import run_step8 as step8
import run_cbf_step9 as step9
import validation

ROOT=validation.ROOT
SOURCE=ROOT.parent/"sweettv-tv-program-recommender"
OUT=ROOT/"results"/"step10_lightgbm"
SCHEDULES=[SOURCE/name for name in ("export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv")]
OVERLAPS=validation.STEP2/"train_airing_overlaps.csv.gz"

GENERATORS=("ppr","personal_frequency","association_rules","cbf","popularity")
LIMITS=(20,50)
TOP_K=50


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def group_split(user):
    value=int(hashlib.sha256(user.encode("ascii")).hexdigest()[:8],16)
    return "evaluation_users" if value%5==0 else "training_users"


def load_targets(protocol):
    secondary={}
    for user,field,count,tie in validation.records(ROOT/"results"/"step5"/"secondary_top5_truth.csv",
        ["user_id","top5_validation_program_ids","relevant_count","tie_break"]):
        secondary[user]=frozenset(field.split()) if field else frozenset()
    fraction,_=step9.load_fraction80_top5_truth(protocol)
    return secondary,fraction


def load_airing_features(protocol,train_profiles,*,overlap_path=None,window=None):
    join=validation.json_read(validation.STEP2/"join_manifest.json")
    overlap_path=Path(overlap_path) if overlap_path is not None else OVERLAPS
    if digest(overlap_path)!=join["artifacts"][overlap_path.name]["sha256"]:raise ValueError("Training overlap hash mismatch")
    lower,upper=tuple(window) if window is not None else validation.WINDOWS["train"]
    weeks_in_window=max(1,math.ceil((upper-lower)/86400/7))
    pair=defaultdict(Counter);pair_weeks=defaultdict(set);first_last={}
    user_channels=defaultdict(Counter);user_hours=defaultdict(Counter);user_weeks=defaultdict(Counter)
    with gzip.open(overlap_path,"rt",newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        expected=["user_id","channel_id","broadcast_start_utc","broadcast_end_utc","tv_show_id","overlap_seconds","first_overlap_utc","last_overlap_utc"]
        if reader.fieldnames!=expected:raise ValueError("Training overlap schema mismatch")
        for row in reader:
            user=row["user_id"];item=validation.program_id(row["tv_show_id"])
            if item=="0":continue
            seconds=int(row["overlap_seconds"]);start=int(row["broadcast_start_utc"]);end=int(row["broadcast_end_utc"])
            first,last=int(row["first_overlap_utc"]),int(row["last_overlap_utc"])
            if not lower<=first<last<=upper or end<=start:raise ValueError("Training overlap outside Week 11-20")
            key=(user,item);frac=seconds/(end-start)
            stat=pair[key];stat["positive_airings"]+=1;stat["overlap_seconds"]+=seconds
            stat["fraction_sum"]+=frac;stat["fraction_max"]=max(stat["fraction_max"],frac)
            if seconds>=120:stat["airings_ge_120"]+=1
            if seconds>=300:stat["airings_ge_300"]+=1
            if seconds>=600:stat["airings_ge_600"]+=1
            if 5*seconds>=4*(end-start):stat["airings_fraction80"]+=1
            bucket=min(weeks_in_window-1,max(0,(first-lower)//(7*86400)))
            pair_weeks[key].add(bucket);user_weeks[user][bucket]+=int(seconds>=120)
            old=first_last.get(key,(first,last));first_last[key]=(min(old[0],first),max(old[1],last))
            if seconds>=120:
                user_channels[user][row["channel_id"]]+=1
                local=datetime.fromtimestamp(start,tz=timezone.utc).astimezone(data_pipeline.ZONE)
                user_hours[user][local.hour]+=1
    return pair,pair_weeks,first_last,user_channels,user_hours,user_weeks


def schedule_features(protocol):
    lower,upper=validation.WINDOWS["validation"]
    eligible=set(protocol.candidates)
    stats=defaultdict(lambda:{"airings":0,"duration_total":0,"duration_max":0,"channels":Counter(),"hours":Counter(),"weeks":set()})
    metadata_rows=[]
    source_paths=[]
    for path in SCHEDULES:
        source_paths.append(str(path));file=path.name
        with path.open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=data_pipeline.SCHEDULE_COLUMNS:raise ValueError("Schedule schema mismatch")
            for row in reader:
                item=validation.schedule_program_id(row["tv_show_id"])
                if item=="0":continue
                start=data_pipeline.parse_timestamp(row["start_time"],schedule=True);duration=int(row["duration"])
                if duration<=0 or not start<upper or start+duration<=lower:continue
                if item in eligible:metadata_rows.append((item,row))
                current=stats[item];current["airings"]+=1;current["duration_total"]+=duration
                current["duration_max"]=max(current["duration_max"],duration);current["channels"][row["channel_id"]]+=1
                local=datetime.fromtimestamp(start,tz=timezone.utc).astimezone(data_pipeline.ZONE)
                current["hours"][local.hour]+=1
                current["weeks"].add(max(0,min(9,(start-lower)//(7*86400))))
    docs=cbf.aggregate_metadata_rows(metadata_rows)
    missing=set(protocol.candidates)-set(docs)
    if missing:raise ValueError(f"Validation metadata unavailable for {len(missing)} candidates")
    return stats,docs,source_paths


def schedule_affinity(stats,user_channels,user_hours):
    by_channel={};by_hour={}
    for item,info in stats.items():
        channel_total=sum(info["channels"].values()) or 1;hour_total=sum(info["hours"].values()) or 1
        by_channel[item]={k:v/channel_total for k,v in info["channels"].items()}
        by_hour[item]={k:v/hour_total for k,v in info["hours"].items()}
    user_channel_prob={};user_hour_prob={}
    for user,counts in user_channels.items():
        total=sum(counts.values()) or 1;user_channel_prob[user]={k:v/total for k,v in counts.items()}
    for user,counts in user_hours.items():
        total=sum(counts.values()) or 1;user_hour_prob[user]={k:v/total for k,v in counts.items()}
    return by_channel,by_hour,user_channel_prob,user_hour_prob


def topk_union(full_ranks,cohort,k):
    output={}
    for user in cohort:output[user]=set().union(*(full_ranks[g][user][:k] for g in GENERATORS))
    return output


def recall_summary(candidate_sets,truth,cohort):
    sizes=[len(candidate_sets[u]) for u in cohort]
    relevant=[(u,truth[u]) for u in cohort if truth[u]]
    per_user=[len(candidate_sets[u]&labels)/len(labels) for u,labels in relevant]
    denominator=sum(len(labels) for _,labels in relevant)
    hits=sum(len(candidate_sets[u]&labels) for u,labels in relevant)
    return {"average_candidate_count":float(np.mean(sizes)),"median_candidate_count":float(np.median(sizes)),
        "p95_candidate_count":float(np.percentile(sizes,95)),"max_candidate_count":max(sizes),
        "users_with_fewer_than_five_candidates":sum(n<5 for n in sizes),
        "candidate_recall_macro_nonempty":float(np.mean(per_user)) if per_user else 0,
        "candidate_recall_micro":hits/denominator if denominator else 0,
        "users_with_all_five_labels_in_candidates":sum(len(candidate_sets[u]&truth[u])==5 for u in cohort if len(truth[u])==5),
        "users_with_truth":len(relevant)}


def main():
    if OUT.exists() and (OUT/"candidate_build_manifest.json").exists():
        raise FileExistsError("Step 10 candidate artifacts already exist; preserve them")
    if not OUT.exists():OUT.mkdir(parents=True,exist_ok=False)
    start_all=time.monotonic();protocol=validation.EvaluationProtocol();cohort=protocol.cohort;candidates=protocol.candidates
    secondary,fraction=load_targets(protocol)
    train=ppr.build_training_profiles(protocol.config)
    pop=popularity.fit_popularity(protocol.config,candidates);fallback=pop.ranking("distinct_users")
    personal=personal_frequency.fit_personal_frequency(protocol.config,candidates)
    graph=ppr.build_graph(train,weight_type="cosine",max_neighbors=30,history_cap=40,min_common_viewers=3)
    ar_profiles,_,_=__import__("run_step6").build_training_profiles(protocol.config)
    ar_ds=ar.build_window_dataset(protocol.config,14)
    ar_model=ar.mine_rules(ar_ds,min_support=.0005,min_confidence=.01,min_cooccurrence=3,min_lift=1.0,top_neighbors=50)
    train_docs=step9.source_schedule_metadata([SCHEDULES[0]],validation.WINDOWS["train"])
    candidate_stats,candidate_docs,schedule_sources=schedule_features(protocol)
    feature_space=cbf.FeatureSpace.fit(train_docs,cbf.FIELD_WEIGHT_CONFIGS["channel_context_emphasis"])
    candidate_matrix=feature_space.transform(candidate_docs,candidates,cbf.FIELD_WEIGHT_CONFIGS["channel_context_emphasis"])
    train_index={item:i for i,item in enumerate(feature_space.training_items)}
    user_vectors,cbf_profile_stats=cbf.build_user_profiles(train.profiles,feature_space.training_items,train_index,
        feature_space.training_matrix,cohort,weighting="airings")
    # Keep generator rankings as compact candidate-index order arrays.
    n_users=len(cohort);n_items=len(candidates);item_index={x:i for i,x in enumerate(candidates)}
    ranks={g:np.empty((n_users,n_items),dtype=np.int32) for g in GENERATORS}
    ppr_scores=np.zeros((n_users,n_items),dtype=np.float32)
    ar_scores=np.zeros_like(ppr_scores);cbf_scores=np.zeros_like(ppr_scores)
    pf_scores=np.zeros_like(ppr_scores)
    pop_rank={item:i for i,item in enumerate(fallback)}
    popular_order=np.fromiter((item_index[x] for x in fallback),dtype=np.int32,count=n_items)
    feature_rows_path=OUT/"candidate_features_top50.csv.gz"
    fields=["user_id","tv_show_id","candidate_limit","group_split","label_top5_freq120","label_all_120","label_top5_fraction80"]
    feature_names=["candidate_source_count","source_ppr","source_personal_frequency","source_association_rules","source_cbf","source_popularity",
        "ppr_score","ppr_rank","personal_frequency_score","personal_frequency_rank","association_rules_score","association_rules_rank",
        "cbf_similarity","cbf_rank","pop_distinct_users","pop_qualifying_airings","pop_watched_seconds","popularity_rank",
        "user_history_program_count","user_history_airings_total","user_history_seconds_total","user_cold_start",
        "user_history_active_weeks",
        "user_program_seen","user_program_airings","user_program_watched_seconds","user_program_log_airings","user_program_recency_days",
        "airings_ge_120","airings_ge_300","airings_ge_600","airings_fraction80","mean_airing_fraction","program_active_weeks",
        "program_week_consistency","program_airings_recent_2w","program_airings_recent_4w","program_airings_recent_8w",
        "target_airing_count","target_duration_total","target_duration_mean","target_duration_max","target_channel_count",
        "target_schedule_week_count","target_unique_hours","target_hour_entropy","candidate_channel_affinity","candidate_hour_affinity",
        "user_category_affinity","user_genre_affinity"]
    fields+=feature_names
    train_bounds=validation.WINDOWS["train"];train_profiles=train.profiles
    weeks_in_training_window=max(1,math.ceil((train_bounds[1]-train_bounds[0])/86400/7))
    airing_stats,user_pair_weeks,first_last,user_channels,user_hours,user_week_counts=load_airing_features(protocol,train_profiles)
    by_channel,by_hour,user_channel_prob,user_hour_prob=schedule_affinity(candidate_stats,user_channels,user_hours)
    user_category={u:Counter() for u in cohort};user_genre={u:Counter() for u in cohort}
    for user in cohort:
        for item,(count,_,_) in train_profiles.get(user,{}).items():
            doc=train_docs.get(item,{})
            for term in doc.get("category",set()):user_category[user][term]+=count
            for term in doc.get("genre",set()):user_genre[user][term]+=count
    group_labels={u:group_split(u) for u in cohort}
    recall={};candidate_sets_by_k={k:{} for k in LIMITS}
    row_total=0;feature_started=time.monotonic()
    with gzip.open(feature_rows_path,"wt",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for u_index,user in enumerate(cohort):
            profile=train_profiles.get(user,{})
            # PPR scores and full candidate ranking.
            if profile:
                restart=ppr.personalization_vector(profile,train,variant="frequency")
                vector,_=ppr.personalized_pagerank(graph,restart,restart_probability=.35,tolerance=1e-8,max_iterations=250)
                for item,index in item_index.items():
                    if item in train.index:ppr_scores[u_index,index]=vector[train.index[item]]
                ppr_order=sorted(range(n_items),key=lambda j:(-ppr_scores[u_index,j],pop_rank[candidates[j]],int(candidates[j])))
            else:ppr_order=list(map(int,popular_order))
            ranks["ppr"][u_index,ppr_order]=np.arange(n_items,dtype=np.int32)
            # Personal-frequency scores/ranking.
            for item,values in profile.items():
                if item in item_index:pf_scores[u_index,item_index[item]]=values[0]
            pf_order=sorted(range(n_items),key=lambda j:(0,-pf_scores[u_index,j],int(candidates[j])) if pf_scores[u_index,j]>0
                else (1,pop_rank[candidates[j]],int(candidates[j])))
            ranks["personal_frequency"][u_index,pf_order]=np.arange(n_items,dtype=np.int32)
            # Association-rule score/ranking.
            ar_map,_=ar.score_user(ar_model,profile,candidates,score_type="confidence_log_lift",antecedent_weight="log_frequency")
            for item,score in ar_map.items():ar_scores[u_index,item_index[item]]=score
            ar_positive=sorted((j for j in range(n_items) if ar_scores[u_index,j]>0),
                key=lambda j:(-ar_scores[u_index,j],-ar_model.dataset.item_support.get(candidates[j],0),int(candidates[j])))
            ar_seen=set(ar_positive);ar_order=ar_positive+[j for j in popular_order if int(j) not in ar_seen]
            ranks["association_rules"][u_index,ar_order]=np.arange(n_items,dtype=np.int32)
            # CBF similarity scores/ranking.
            sim=(candidate_matrix@user_vectors.getrow(u_index).T).tocsr()
            cbf_row=sim.toarray().ravel()
            cbf_scores[u_index,:]=cbf_row.astype(np.float32)
            cbf_order=sorted(range(n_items),key=lambda j:(-cbf_row[j],pop_rank[candidates[j]],int(candidates[j])))
            ranks["cbf"][u_index,cbf_order]=np.arange(n_items,dtype=np.int32)
            ranks["popularity"][u_index,popular_order]=np.arange(n_items,dtype=np.int32)
            full_orders={g:np.argsort(ranks[g][u_index]) for g in GENERATORS}
            for k in LIMITS:
                candidate_sets_by_k[k][user]=set(candidates[int(i)] for g in GENERATORS for i in full_orders[g][:k])
            union=candidate_sets_by_k[TOP_K][user]
            user_items=profile
            total_airings=sum(x[0] for x in user_items.values());total_seconds=sum(x[1] for x in user_items.values())
            cat_counts=user_category[user];genre_counts=user_genre[user]
            cat_total=sum(cat_counts.values()) or 1;genre_total=sum(genre_counts.values()) or 1
            ch_counts=user_channel_prob.get(user,{});hour_counts=user_hour_prob.get(user,{})
            active_user_weeks=len(user_week_counts.get(user,{}))
            for item in sorted(union,key=int):
                ix=item_index[item];hist=user_items.get(item,(0,0,None));h=airing_stats.get((user,item),Counter())
                first,last=first_last.get((user,item),(None,None))
                recency=(train_bounds[1]-last)/86400 if last is not None else 0.0
                week_set=user_pair_weeks.get((user,item),set());program_weeks=len(week_set)
                sch=candidate_stats.get(item,{"airings":0,"duration_total":0,"duration_max":0,"channels":Counter(),"hours":Counter(),"weeks":set()})
                channels=sch["channels"];hours=sch["hours"];schedule_total=sum(channels.values()) or 1
                hours_total=sum(hours.values()) or 1
                ch_aff=sum(ch_counts.get(channel,0)*(count/schedule_total) for channel,count in channels.items())
                hour_aff=sum(hour_counts.get(hour,0)*(count/hours_total) for hour,count in hours.items())
                doc=candidate_docs.get(item,{})
                cat_aff=sum(cat_counts.get(term,0) for term in doc.get("category",set()))/cat_total
                genre_aff=sum(genre_counts.get(term,0) for term in doc.get("genre",set()))/genre_total
                entropy=-math.fsum((count/hours_total)*math.log(count/hours_total) for count in hours.values() if count)
                source_ranks={g:int(ranks[g][u_index,ix]) for g in GENERATORS}
                src_members={g:source_ranks[g]<TOP_K for g in GENERATORS}
                row={"user_id":user,"tv_show_id":item,"candidate_limit":TOP_K,"group_split":group_labels[user],
                    "label_top5_freq120":int(item in secondary[user]),"label_all_120":int(item in protocol.truth[user]),
                    "label_top5_fraction80":int(item in fraction[user]),
                    "candidate_source_count":sum(src_members.values()),
                    "source_ppr":int(src_members["ppr"]),"source_personal_frequency":int(src_members["personal_frequency"]),
                    "source_association_rules":int(src_members["association_rules"]),"source_cbf":int(src_members["cbf"]),
                    "source_popularity":int(src_members["popularity"]),
                    "ppr_score":float(ppr_scores[u_index,ix]),"ppr_rank":source_ranks["ppr"],
                    "personal_frequency_score":float(pf_scores[u_index,ix]),"personal_frequency_rank":source_ranks["personal_frequency"],
                    "association_rules_score":float(ar_scores[u_index,ix]),"association_rules_rank":source_ranks["association_rules"],
                    "cbf_similarity":float(cbf_scores[u_index,ix]),"cbf_rank":source_ranks["cbf"],
                    "pop_distinct_users":pop.distinct_users.get(item,0),"pop_qualifying_airings":pop.qualifying_airings.get(item,0),
                    "pop_watched_seconds":pop.watched_seconds.get(item,0),"popularity_rank":source_ranks["popularity"],
                    "user_history_program_count":len(user_items),"user_history_airings_total":total_airings,
                    "user_history_seconds_total":total_seconds,"user_cold_start":int(not bool(user_items)),
                    "user_program_seen":int(bool(hist[0])),"user_program_airings":hist[0],"user_program_watched_seconds":hist[1],
                    "user_program_log_airings":math.log1p(hist[0]),"user_program_recency_days":recency,
                    "airings_ge_120":h["airings_ge_120"],"airings_ge_300":h["airings_ge_300"],
                    "airings_ge_600":h["airings_ge_600"],"airings_fraction80":h["airings_fraction80"],
                    "mean_airing_fraction":h["fraction_sum"]/h["positive_airings"] if h["positive_airings"] else 0,
                    "program_active_weeks":program_weeks,"program_week_consistency":program_weeks/weeks_in_training_window,
                    "program_airings_recent_2w":h["recent_2w"],"program_airings_recent_4w":h["recent_4w"],
                    "program_airings_recent_8w":h["recent_8w"],"target_airing_count":sch["airings"],
                    "target_duration_total":sch["duration_total"],"target_duration_mean":sch["duration_total"]/sch["airings"] if sch["airings"] else 0,
                    "target_duration_max":sch["duration_max"],"target_channel_count":len(channels),
                    "target_schedule_week_count":len(sch["weeks"]),"target_unique_hours":len(hours),"target_hour_entropy":entropy,
                    "candidate_channel_affinity":ch_aff,"candidate_hour_affinity":hour_aff,
                    "user_category_affinity":cat_aff,"user_genre_affinity":genre_aff,
                    "user_history_active_weeks":active_user_weeks}
                for key,value in row.items():
                    if isinstance(value,float) and not math.isfinite(value):row[key]=0
                writer.writerow(row);row_total+=1
    recall={}
    for k,sets in candidate_sets_by_k.items():
        recall[k]={"primary_all_distinct_120s":recall_summary(sets,protocol.truth,cohort),
                   "secondary_top5_frequency120":recall_summary(sets,secondary,cohort),
                   "fraction80_top5_sensitivity":recall_summary(sets,fraction,cohort)}
        with gzip.open(OUT/f"candidate_sets_top{k}.csv.gz","wt",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream,lineterminator="\n");writer.writerow(["user_id","candidate_program_ids","candidate_count"])
            for user in cohort:
                items=tuple(sorted(sets[user],key=int));writer.writerow([user," ".join(items),len(items)])
    result={"stage":"Step10 candidate generation and historical feature preparation","protocol_id":protocol.config["protocol_id"],
        "candidate_limits":list(LIMITS),"generators":list(GENERATORS),"top50_feature_rows":row_total,
        "feature_columns":fields,"group_split":"sha256(user_id) first 32-bit value mod 5; 20% held out by user",
        "supervised_primary_label":"label_top5_freq120: per-user five most frequent qualifying Weeks 21-30 programs; sensitivity/top-five target",
        "features_by_candidate":feature_names,"candidate_recall":recall,
        "training_profile_source_sha256":train.training_sha256,"training_overlap_source_sha256":digest(OVERLAPS),
        "training_window_utc":list(validation.WINDOWS["train"]),"validation_window_utc":list(validation.WINDOWS["validation"]),
        "test_window_utc":list(validation.WINDOWS["test"]),"train_metadata_vocabulary_fit":True,
        "test_period_schedule_user_viewing_used":False,"other_participant_data_or_models_used":False,
        "schedule_sources_metadata_only":schedule_sources,
        "cbf_feature_space":{"vocabulary_size":len(feature_space.vocabulary),"training_documents":len(feature_space.training_items),
            "training_nnz":int(feature_space.training_matrix.nnz),"candidate_nnz":int(candidate_matrix.nnz),
            "field_weights":cbf.FIELD_WEIGHT_CONFIGS["channel_context_emphasis"],"profile_weighting":"airings"},
        "runtime_seconds":{"feature_and_candidate_build":time.monotonic()-start_all,"ppr_graph":graph.runtime_seconds,
            "ar_dataset":ar_ds.build_seconds,"ar_pair_mining":ar_ds.pair_mining_seconds},
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "environment":{"python":__import__("sys").version,"numpy":np.__version__,"scipy":__import__("scipy").__version__}}
    write_json(OUT/"candidate_build_manifest.json",result)
    print(json.dumps({"candidate_counts":{k:recall[k]["secondary_top5_frequency120"]["average_candidate_count"] for k in LIMITS},
        "candidate_recall_top20":recall[20]["secondary_top5_frequency120"],
        "candidate_recall_top50":recall[50]["secondary_top5_frequency120"],
        "feature_rows":row_total,"runtime_seconds":result["runtime_seconds"]["feature_and_candidate_build"],
        "peak_rss_mib":result["peak_rss_mib"]},indent=2))


if __name__=="__main__":main()
