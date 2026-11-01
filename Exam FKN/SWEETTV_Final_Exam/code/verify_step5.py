"""Independent full verification for Step 5 PPR and baseline comparisons."""

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

import numpy as np
import scipy
from scipy.sparse import csr_matrix

import evaluation
import validation

ROOT = validation.ROOT
OUT = ROOT / "results" / "step5"
STEP4_OUT = ROOT / "results" / "step4"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def load_prediction(path, cohort):
    rows = list(csv_rows(path))
    if len(rows) != len(cohort) or list(rows[0]) != ["user_id", "tv_show_id"]:
        raise ValueError(f"Prediction schema/row-count mismatch: {path}")
    if [r["user_id"] for r in rows] != list(cohort):
        raise ValueError(f"Prediction user order mismatch: {path}")
    return evaluation.parse_prediction_records(rows)


def secondary_truth_from_source(config, cohort, candidates):
    """Independent reference uses CSV rows/counters, not validation.iter_interactions."""
    source = ROOT / config["validation_labels_source"]["path"]
    if digest(source) != config["validation_labels_source"]["sha256"]:
        raise ValueError("Validation source hash mismatch")
    counts = {u: Counter() for u in cohort}
    cohort_set = set(cohort)
    header = ["user_id", "tv_show_id", "qualifying_airings", "watched_seconds", "first_watch_utc", "last_watch_utc"]
    with gzip.open(source, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != header:
            raise ValueError("Unexpected validation interaction schema")
        for r in reader:
            if r["user_id"] in cohort_set:
                counts[r["user_id"]][r["tv_show_id"]] += int(r["qualifying_airings"])
    truth = {}
    for u in cohort:
        ordered = sorted(counts[u], key=lambda item: (-counts[u][item], int(item)))[:5]
        truth[u] = frozenset(ordered)
    return truth, counts


def read_saved_secondary(path, cohort, truth, counts):
    rows = list(csv_rows(path))
    if len(rows) != len(cohort) or [r["user_id"] for r in rows] != list(cohort):
        return False
    return all((r["top5_validation_program_ids"].split() if r["top5_validation_program_ids"] else [])
               == sorted(counts[u],key=lambda item:(-counts[u][item],int(item)))[:5]
               and int(r["relevant_count"]) == len(truth[u])
               and r["tie_break"] == "qualifying_airings_desc,numeric_program_id_asc"
               for u, r in zip(cohort, rows))


def main():
    started = time.monotonic()
    protocol = validation.EvaluationProtocol()
    config = protocol.config
    cohort, candidates = protocol.cohort, protocol.candidates
    ppr_config = read_json(OUT / "ppr_model_config.json")
    experiments = list(csv_rows(OUT / "ppr_experiments.csv"))
    step4run = read_json(STEP4_OUT / "step4_run.json")
    checks = []

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    # Step 1 originals and Step 2–4 products/evidence remain exact.
    original = read_json(ROOT / "results" / "original_sha256_before.json")
    for path, info in original.items():
        check("original_unchanged:" + Path(path).name, digest(path) == info["sha256"])
    step3_snapshot = read_json(ROOT / "results" / "step3_inputs_before.json")["files"]
    changed = [path for path, info in step3_snapshot.items()
               if digest(path) != info["sha256"] or Path(path).stat().st_size != info["bytes"]]
    check("Step1_to_Step3_snapshot_unchanged", not changed, changed)
    step4_verification = read_json(ROOT / "results" / "step4_verification.json")
    check("Step4_verification_passed", step4_verification["all_passed"])
    step4_artifacts=read_json(STEP4_OUT/"step4_outputs.json")
    step4_changed=[name for name,meta in step4_artifacts.items()
                   if (STEP4_OUT/name).stat().st_size!=meta["bytes"] or digest(STEP4_OUT/name)!=meta["sha256"]]
    check("all_Step4_prediction_and_result_artifacts_unchanged",not step4_changed,step4_changed)
    step4_doc_hashes={}
    for line in (ROOT/"results"/"step4_documentation_sha256.txt").read_text().splitlines():
        parts=line.split()
        if len(parts)==2 and not parts[0].startswith("Report hash"):
            step4_doc_hashes[parts[1]]=parts[0]
    step4_doc_changed=[]
    for relative,expected in step4_doc_hashes.items():
        path=(ROOT/"results"/relative).resolve()
        if not path.is_file() or digest(path)!=expected:
            step4_doc_changed.append(relative)
    check("Step4_report_comparison_and_sessions_unchanged",not step4_doc_changed,step4_doc_changed)
    check("root_Step4_comparison_copy_unchanged",
          digest(ROOT/"results"/"step4_baseline_comparison.csv")=="18a3caea4d8b1ec777b8e99a83ead081449832a555654d3808a15560139c3f9e"
          and (ROOT/"results"/"step4_baseline_comparison.csv").read_bytes()==(STEP4_OUT/"step4_baseline_comparison.csv").read_bytes())
    check("Step4_baseline_run_uses_same_protocol", step4run["protocol_id"] == config["protocol_id"]
          and step4run["cohort_users"] == 1260)
    check("Step4_original_best_scores_unchanged", math.isclose(step4run["variants"]["popularity_distinct_users"]["primary_map_at_5"], 0.7684179894179894, abs_tol=1e-15)
          and math.isclose(step4run["variants"]["personal_frequency_qualifying_airings"]["primary_map_at_5"], 0.7876269841269841, abs_tol=1e-15))
    check("PPR_config_protocol_identity", ppr_config["evaluation_config_sha256"] == protocol.config_sha256
          and ppr_config["protocol_id"] == config["protocol_id"]
          and tuple(ppr_config["training_window_utc"]) == validation.WINDOWS["train"]
          and ppr_config["training_input_sha256"] == config["training_input"]["sha256"])
    check("step5_dependency_versions", ppr_config["numpy_version"] == np.__version__ and ppr_config["scipy_version"] == scipy.__version__)

    # Independent training aggregation and per-user long-history truncation.
    users = defaultdict(dict)
    item_support = Counter()
    item_airings = Counter()
    training_file = ROOT / config["training_input"]["path"]
    train_header = ["user_id", "tv_show_id", "qualifying_airings", "watched_seconds", "first_watch_utc", "last_watch_utc"]
    rows_read = 0
    with gzip.open(training_file, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        check("training_interaction_schema", reader.fieldnames == train_header)
        for r in reader:
            rows_read += 1
            users[r["user_id"]][r["tv_show_id"]] = (int(r["qualifying_airings"]), int(r["watched_seconds"]), int(r["last_watch_utc"]))
            item_support[r["tv_show_id"]] += 1
            item_airings[r["tv_show_id"]] += int(r["qualifying_airings"])
    nodes = tuple(sorted(item_support, key=int))
    check("training_only_rows_nodes_and_hash", rows_read == 716796 and len(item_support) == 2948
          and digest(training_file) == ppr_config["training_input_sha256"])
    cap = ppr_config["history_cap"]
    capped_users = sum(len(profile)>cap for profile in users.values())
    pair_reference = Counter()
    pair_occurrences, truncated = 0, 0
    for profile in users.values():
        ordered = sorted(profile.items(), key=lambda x: (-x[1][0], -x[1][1], int(x[0])))
        chosen = [x[0] for x in ordered[:cap]]
        truncated += max(0, len(profile)-len(chosen))
        for a, b in combinations(sorted(chosen, key=int), 2):
            pair_reference[(a,b)] += 1
            pair_occurrences += 1
    check("deterministic_training_pair_generation", pair_occurrences == 3524267 and truncated == 532492
          and len(pair_reference) == 592769,
          {"pair_occurrences":pair_occurrences,"truncated_program_entries":truncated,"distinct_pairs":len(pair_reference)})
    expected_nodes = np.asarray(nodes, dtype="U32")
    reference_neighbors={node:[] for node in range(len(nodes))}
    index_to_id=nodes
    node_index={item:i for i,item in enumerate(nodes)}
    for (a,b),common in pair_reference.items():
        i,j=node_index[a],node_index[b]
        if common>=ppr_config["graph_artifacts"]["common_viewers_top30"]["statistics"]["minimum_common_viewers"]:
            reference_neighbors[i].append((i,j,common))
            reference_neighbors[j].append((j,i,common))
    graph_info = ppr_config["graph_artifacts"]
    graph_arrays = {}
    for name, spec in graph_info.items():
        path = ROOT / spec["path"]
        check("graph_artifact_sha256:"+name, digest(path) == spec["sha256"])
        with np.load(path, allow_pickle=False) as z:
            nodes_saved = z["nodes"]
            indptr, indices, data = z["indptr"], z["indices"], z["transition_data"]
            edge_i, edge_j = z["undirected_edge_i"], z["undirected_edge_j"]
            edge_w, edge_common = z["undirected_edge_weight"], z["undirected_common_viewers"]
            check("graph_nodes_match_training_catalog:"+name, np.array_equal(nodes_saved, expected_nodes))
            check("graph_nodes_positive_distinct:"+name,len(set(nodes_saved.tolist()))==len(nodes_saved)
                  and all(int(item)>0 for item in nodes_saved.tolist()))
            matrix = csr_matrix((data, indices, indptr), shape=(len(nodes_saved),len(nodes_saved)))
            degrees = np.diff(indptr)
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()
            graph_stats=spec["statistics"]
            check("graph_stored_statistics_match_csr:"+name,
                  graph_stats["nodes"]==len(nodes_saved)
                  and graph_stats["directed_nonzero_edges"]==matrix.nnz
                  and graph_stats["undirected_nonzero_edges"]==matrix.nnz//2
                  and graph_stats["maximum_degree"]==int(degrees.max())
                  and graph_stats["dangling_nodes"]==int(np.count_nonzero(degrees==0))
                  and math.isclose(graph_stats["average_degree"],float(degrees.mean()),rel_tol=0,abs_tol=1e-14)
                  and math.isclose(graph_stats["directed_density"],matrix.nnz/(len(nodes_saved)**2),rel_tol=0,abs_tol=1e-15))
            check("sparse_csr_transition_normalization:"+name,
                  np.allclose(row_sums[degrees>0], 1.0, rtol=0, atol=2e-14) and np.all(row_sums[degrees==0] == 0))
            check("undirected_edge_serialization:"+name, len(edge_i)==len(edge_j)==len(edge_w)==len(edge_common)
                  and len(edge_i)==spec["statistics"]["retained_undirected_edges"])
            graph_arrays[name] = {"nodes":nodes_saved,"matrix":matrix,"edge_i":edge_i,"edge_j":edge_j,
                                  "edge_weight":edge_w,"edge_common":edge_common}
            weight_type=spec["statistics"]["weight_type"]
            topk=spec["statistics"]["max_neighbors_per_node_before_symmetrization"]
            selected_edges=set()
            for node,edges in reference_neighbors.items():
                def independent_weight(edge):
                    _,other,common=edge
                    if weight_type=="common_viewers":
                        return float(common)
                    return common/math.sqrt(item_support[nodes[node]]*item_support[nodes[other]])
                edges.sort(key=lambda e:(-independent_weight(e),int(nodes[e[1]])))
                for _,other,_ in edges[:topk]:
                    selected_edges.add((min(node,other),max(node,other)))
            saved_edges={(int(i),int(j)) for i,j in zip(edge_i,edge_j)}
            check("independent_topk_edge_set:"+name,saved_edges==selected_edges,
                  {"independent_edges":len(selected_edges),"saved_edges":len(saved_edges)})
            expected_weights=[]
            for i,j in zip(edge_i,edge_j):
                common=int(pair_reference[(nodes[int(i)],nodes[int(j)])])
                if weight_type=="common_viewers":
                    expected_weights.append(float(common))
                else:
                    expected_weights.append(common/math.sqrt(item_support[nodes[int(i)]]*item_support[nodes[int(j)]]))
            check("all_retained_edge_weights_independent:"+name,
                  np.allclose(edge_w,np.asarray(expected_weights),rtol=1e-14,atol=1e-14))
            directed_row_weights=np.zeros(len(nodes),dtype=np.float64)
            np.add.at(directed_row_weights,edge_i,edge_w)
            np.add.at(directed_row_weights,edge_j,edge_w)
            transition_checks=[]
            for i,j,w in zip(edge_i[:100],edge_j[:100],edge_w[:100]):
                transition_checks.append(math.isclose(float(matrix[int(i),int(j)]),float(w/directed_row_weights[int(i)]),rel_tol=1e-13,abs_tol=1e-14)
                    and math.isclose(float(matrix[int(j),int(i)]),float(w/directed_row_weights[int(j)]),rel_tol=1e-13,abs_tol=1e-14))
            check("sampled_sparse_transition_values:"+name,all(transition_checks))
    # Recompute common-viewer counts on a deterministic sample of edges using all training users.
    sample_edge_ids = set()
    for name, values in graph_arrays.items():
        for i,j in zip(values["edge_i"][:50], values["edge_j"][:50]):
            a,b = nodes[int(i)],nodes[int(j)]
            sample_edge_ids.add((a,b))
    sampled_counts=Counter()
    for profile in users.values():
        chosen=set(x for x,_ in sorted(profile.items(),key=lambda x:(-x[1][0],-x[1][1],int(x[0])))[:cap])
        for a,b in sample_edge_ids:
            sampled_counts[(a,b)] += a in chosen and b in chosen
    sampled_edge_checks=[]
    for name, values in graph_arrays.items():
        index = {node:i for i,node in enumerate(values["nodes"].tolist())}
        for pos,(i,j) in enumerate(zip(values["edge_i"][:50],values["edge_j"][:50])):
            a,b=nodes[int(i)],nodes[int(j)]
            common=sampled_counts[(a,b)]
            stored=int(values["edge_common"][pos])
            vi,vj=item_support[a],item_support[b]
            raw=float(common) if "common_viewers" in name else common/math.sqrt(vi*vj)
            stored_weight=float(values["edge_weight"][pos])
            ok=common==stored and math.isclose(raw,stored_weight,rel_tol=1e-14,abs_tol=1e-14)
            sampled_edge_checks.append(ok)
    check("sampled_common_viewer_and_edge_weight_reference", bool(sampled_edge_checks) and all(sampled_edge_checks),
          {"sampled_directed_graph_variants":len(sampled_edge_checks)})
    check("history_cap_user_count_matches_profile_data", capped_users == sum(len(profile)>cap for profile in users.values()))

    # Secondary Top-5 labels independently recomputed from validation airing frequency.
    secondary_truth, secondary_counts = secondary_truth_from_source(config, cohort, candidates)
    check("secondary_top5_truth_saved_consistently", read_saved_secondary(OUT/"secondary_top5_truth.csv",cohort,secondary_truth,secondary_counts))
    # Rebuild the policy fallback independently from training user-support counts and Step 4's saved ranked list.
    step4_run_again = read_json(STEP4_OUT/"step4_run.json")
    fallback_ids=tuple(map(str, step4_run_again["fallback_ranking"]))
    fallback_order=tuple(p for p in fallback_ids if p in set(candidates))
    independent_fallback=tuple(sorted(candidates,key=lambda p:(-item_support.get(p,0),-item_airings.get(p,0),int(p))))
    saved_popularity_top5=load_prediction(STEP4_OUT/"popularity_distinct_users_predictions.csv",cohort)[cohort[0]]
    check("fallback_is_step3_distinct_user_order", fallback_order == independent_fallback and fallback_order[:5] == tuple(
        saved_popularity_top5))

    # Verify every saved PPR config with the exact production evaluator.
    exp_rows=list(csv_rows(OUT/"ppr_experiments.csv"))
    with (OUT/"ppr_experiments.csv").open(newline="",encoding="utf-8") as experiment_stream:
        experiment_header=next(csv.reader(experiment_stream))
    check("ppr_experiment_table_has_unique_columns",len(experiment_header)==len(set(experiment_header)) and len(exp_rows)==16)
    exp_by_id={r["config_id"]:r for r in exp_rows}
    all_predictions={}
    for row in exp_rows:
        config_id=row["config_id"]
        pred_path=ROOT/"results"/"step5"/"experiments"/config_id/"predictions.csv"
        ap_path=ROOT/"results"/"step5"/"experiments"/config_id/"user_ap.csv"
        check("ppr_prediction_hash:"+config_id,digest(pred_path)==row["prediction_sha256"])
        pred=load_prediction(pred_path,cohort)
        check("ppr_candidate_top5_contract:"+config_id,all(len(pred[u])==5 and len(set(pred[u]))==5 and set(pred[u]).issubset(set(candidates)) and all(int(x)>0 for x in pred[u]) for u in cohort))
        scored=protocol.evaluate(pred)
        check("primary_map_recalculated:"+config_id,math.isclose(scored["primary_map_at_5"],float(row["primary_map_at_5"]),rel_tol=0,abs_tol=1e-15))
        check("diagnostic_map_recalculated:"+config_id,math.isclose(scored["diagnostic_map_at_5_nonempty"],float(row["diagnostic_map_at_5_nonempty"]),rel_tol=0,abs_tol=1e-15))
        sec=evaluation.evaluate_predictions(pred,secondary_truth,cohort,candidates,strict=True)
        check("secondary_map_recalculated:"+config_id,math.isclose(sec["primary_map_at_5"],float(row["secondary_top5_sensitivity_map_at_5"]),rel_tol=0,abs_tol=1e-15))
        ap_rows=list(csv_rows(ap_path))
        check("per_user_ap_recalculated:"+config_id,[r["user_id"] for r in ap_rows]==list(cohort) and all(math.isclose(float(r["ap_at_5"]),scored["per_user_ap_at_5"][u],rel_tol=0,abs_tol=1e-15) for u,r in zip(cohort,ap_rows)))
        all_predictions[config_id]=pred
    selected= max(exp_rows,key=lambda r:(float(r["primary_map_at_5"]),r["config_id"]))
    best_id=ppr_config["best_config_id"]
    check("best_configuration_matches_grid",selected["config_id"]==best_id)
    best_csv=load_prediction(OUT/"ppr_predictions.csv",cohort)
    check("canonical_prediction_matches_best_experiment",best_csv==all_predictions[best_id]
          and digest(OUT/"ppr_predictions.csv")==ppr_config["best_prediction_sha256"])
    check("best_user_AP_output",digest(OUT/"ppr_user_ap.csv")==ppr_config["best_user_ap_sha256"])

    # Cold-start: exactly the registered Step 4 training-only popularity fallback.
    cold=[u for u in cohort if u not in users]
    check("cold_user_count_is_step3_ten",len(cold)==10)
    if cold:
        check("cold_users_use_identical_fallback",all(best_csv[u]==fallback_order[:5] for u in cold),
              {"fallback":fallback_order[:5]})
    check("cold_user_MAP_matches_shared_fallback",math.isclose(float(selected["cold_primary_map_at_5"]),
          step4_run_again["variants"]["popularity_distinct_users"]["cold_primary_map_at_5"],rel_tol=0,abs_tol=1e-15))
    cold_hit_users=sum(bool(set(best_csv[u])&protocol.truth[u]) for u in cold)
    step4_pop_predictions=load_prediction(STEP4_OUT/"popularity_distinct_users_predictions.csv",cohort)
    fallback_cold_hits=sum(bool(set(step4_pop_predictions[u])&protocol.truth[u]) for u in cold)
    check("cold_user_hit_rate_matches_fallback",cold_hit_users==fallback_cold_hits)

    # Re-score all five baseline prediction files with primary and secondary labels.
    baseline_rows=[]
    step4run=read_json(STEP4_OUT/"step4_run.json")
    for name,metrics in step4run["variants"].items():
        pred_path=ROOT/metrics["prediction_file"]
        check("step4_baseline_hash:"+name,digest(pred_path)==metrics["prediction_sha256"])
        pred=load_prediction(pred_path,cohort)
        primary=protocol.evaluate(pred)
        secondary=evaluation.evaluate_predictions(pred,secondary_truth,cohort,candidates,strict=True)
        check("step4_baseline_primary_matches_step4:"+name,math.isclose(primary["primary_map_at_5"],metrics["primary_map_at_5"],rel_tol=0,abs_tol=1e-15))
        baseline_rows.append({"family":metrics["family"],"variant":metrics["variant"],
            "primary_map_at_5":primary["primary_map_at_5"],
            "diagnostic_map_at_5_nonempty":primary["diagnostic_map_at_5_nonempty"],
            "secondary_top5_sensitivity_map_at_5":secondary["primary_map_at_5"],
            "secondary_top5_sensitivity_nonempty_map_at_5":secondary["diagnostic_map_at_5_nonempty"],
            "hit_rate_at_5":metrics["users_with_at_least_one_relevant_recommendation"]/len(cohort),
            "top1_hit_rate":"", "training_runtime_seconds":metrics["training_runtime_seconds"],
            "prediction_runtime_seconds":metrics["prediction_runtime_seconds"],
            "candidate_coverage_fraction":metrics["candidate_coverage_fraction"],
            "cold_start_users_using_fallback":metrics["users_using_full_popularity_fallback"],
            "fallback_slots":metrics["popularity_backfill_slots_exact"],"source":metrics["prediction_file"]})

    # Update comparison to include sensitivity for every saved baseline variant.
    ppr_rows=[]
    for row in exp_rows:
        ppr_rows.append({"family":"PPR","variant":row["config_id"],
            "primary_map_at_5":float(row["primary_map_at_5"]),"diagnostic_map_at_5_nonempty":float(row["diagnostic_map_at_5_nonempty"]),
            "secondary_top5_sensitivity_map_at_5":float(row["secondary_top5_sensitivity_map_at_5"]),
            "secondary_top5_sensitivity_nonempty_map_at_5":float(row["secondary_top5_sensitivity_nonempty_map_at_5"]),
            "hit_rate_at_5":float(row["hit_rate_at_5"]),"top1_hit_rate":float(row["top1_hit_rate"]),
            "training_runtime_seconds":float(row["ppr_training_runtime_seconds"]),
            "prediction_runtime_seconds":float(row["prediction_generation_seconds"]),
            "candidate_coverage_fraction":float(row["candidate_coverage_fraction"]),
            "cold_start_users_using_fallback":int(row["cold_start_users_using_fallback"]),
            "fallback_slots":int(row["fallback_slots"]),"source":f"results/step5/experiments/{row['config_id']}/predictions.csv"})
    comparison=ppr_rows+baseline_rows
    comp_path=ROOT/"results"/"step5_ppr_comparison.csv"
    fields=sorted({k for r in comparison for k in r})
    with comp_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison)
    check("comparison_rows_all_configs_and_five_baselines",len(comparison)==len(exp_rows)+5)

    # Keep identical Step 4 user examples.
    prior=read_json(STEP4_OUT/"provisional_two_users.json")
    ex_path=OUT/"ppr_user_examples.md"
    check("fixed_step4_example_user_ids",prior["heavy_user_id"]!=prior["light_user_id"]
          and ppr_config["best_config_id"]==best_id)
    check("best_prediction_user_examples_present",ex_path.exists() and digest(ex_path)==ppr_config["ppr_user_examples_sha256"])
    example_text=ex_path.read_text()
    check("fixed_step4_user_histories_and_baselines_in_examples",
          all(f"## {role.title()} user `{prior[f'{role}_user_id']}`" in example_text for role in ("heavy","light"))
          and all(label in example_text for label in ("Training-period program history (all distinct qualifying programs)",
              "Validation-period relevant programs (all distinct qualifying programs)",
              "Popularity (distinct training users)","Personal frequency (qualifying airings)","Personalized PageRank")))
    check("no_kaggle_submission",not list((ROOT/"submission").glob("*.csv")))
    check("no_validation_or_test_view_source_in_model_config",ppr_config["training_input_sha256"]==config["training_input"]["sha256"]
          and tuple(ppr_config["training_window_utc"])==validation.WINDOWS["train"])

    # Step 4/Step 5 source/product hashes and dependencies.
    test_run=read_json(sorted((ROOT/"results"/"step5_test_runs").glob("run_*.json"))[-1])
    check("Step5_unit_tests_pass",test_run["all_passed"] and test_run["tests_run"]==22)
    check("test_hashes_match_current_sources",all(digest(ROOT/"code"/name)==sha for name,sha in test_run["source_sha256"].items()))
    check("Step4_verification_integrity",step4_verification["all_passed"])
    # Graph and PPR independent fixtures are also in this executed test run.
    source_names=("ppr.py","run_step5.py","verify_step5.py","test_step5.py")
    source_hashes={name:digest(ROOT/"code"/name) for name in source_names}
    requirements=(ROOT/"requirements-step5.txt").read_text().splitlines()
    check("Step5_graph_dependencies_pinned",f"numpy=={np.__version__}" in requirements and f"scipy=={scipy.__version__}" in requirements)
    artifact_manifest=read_json(OUT/"ppr_artifacts.json")
    bad_artifacts=[name for name,meta in artifact_manifest.items()
        if not (OUT/name).is_file() or (OUT/name).stat().st_size!=meta["bytes"] or digest(OUT/name)!=meta["sha256"]]
    check("Step5_ppr_artifact_manifest",not bad_artifacts,bad_artifacts)
    # Step 1–4 original/session evidence was checked again after PPR.
    original_manifest=read_json(ROOT/"results"/"original_sha256_before.json")
    for path,meta in original_manifest.items():
        check("original_input_unchanged:"+Path(path).name,digest(path)==meta["sha256"])
    step3_snapshot=read_json(ROOT/"results"/"step3_inputs_before.json")["files"]
    changed=[path for path,meta in step3_snapshot.items() if digest(path)!=meta["sha256"] or Path(path).stat().st_size!=meta["bytes"]]
    check("Step1_to_Step3_completed_artifacts_unchanged",not changed,changed)
    session_hashes={"step1_opencode_session.json":"797d19d6f36a03402f1bfeb69a40b7329c7c69b63bc540105fae2607943a7c6f",
                    "step2_opencode_session.json":"eb324cdbb5809ff709985d15a6386738873eee15a3a2da1cd9f8512c82dc0fb2"}
    session_hashes["step3_opencode_session.json"]=(ROOT/"evidence"/"step3_checkpoint_sha256.txt").read_text().splitlines()[0].split()[0]
    session_hashes["step4_opencode_session.json"]=(ROOT/"evidence"/"step4_checkpoint_sha256.txt").read_text().splitlines()[0].split()[0]
    for name, expected in session_hashes.items():
        check("session_export_unchanged:"+name,digest(ROOT/"evidence"/name)==expected)

    result={"stage":"Step 5 Personalized PageRank only","all_passed":all(c["passed"] for c in checks),
        "passed_checks":sum(c["passed"] for c in checks),"failed_checks":sum(not c["passed"] for c in checks),
        "checks":checks,"graph_reference":{"nodes":len(nodes),"sampled_pair_count_edges":len(sampled_edge_checks),
          "pair_occurrences":pair_occurrences,"unique_co_view_pairs":len(pair_reference)},
        "experiment_count":len(exp_rows),"best_config_id":best_id,
        "best_primary_MAP_at_5":float(selected["primary_map_at_5"]),
        "best_secondary_MAP_at_5":float(selected["secondary_top5_sensitivity_map_at_5"]),
        "cold_start_users":len(cold),"cold_users_with_primary_hit":cold_hit_users,
        "cold_hit_rate_at_5":cold_hit_users/len(cold) if cold else None,
        "capped_training_users":capped_users,"training_history_entries_truncated":truncated,
        "baseline_secondary_recalculated":{r["family"]+"_"+r["variant"]:r["secondary_top5_sensitivity_map_at_5"] for r in baseline_rows},
        "comparison_sha256":digest(comp_path),"ppr_prediction_sha256":digest(OUT/"ppr_predictions.csv"),
        "ppr_config_sha256":digest(OUT/"ppr_model_config.json"),"training_sha256":digest(training_file),
        "numpy":np.__version__,"scipy":scipy.__version__,"implementation_sha256":source_hashes,
        "verification_runtime_seconds":round(time.monotonic()-started,3),
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "no_test_user_behavior_accessed":True,"no_association_rules":True,"no_submission":True}
    prior_verification=ROOT/"results"/"step5_verification.json"
    preserved_attempt=OUT/"step5_verification_attempt_001.json"
    if prior_verification.exists() and not preserved_attempt.exists():
        preserved_attempt.write_bytes(prior_verification.read_bytes())
    write_json(prior_verification,result)
    write_json(OUT/"step5_verification.json",result)
    print(json.dumps({k:result[k] for k in ("all_passed","passed_checks","failed_checks","experiment_count",
        "best_config_id","best_primary_MAP_at_5","best_secondary_MAP_at_5","verification_runtime_seconds","peak_rss_mib")},indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
