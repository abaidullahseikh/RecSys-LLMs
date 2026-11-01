"""Build sparse program co-view PPR, staged tune on frozen validation, and save evidence."""

from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import scipy

import evaluation
import ppr
import popularity
import validation

ROOT = validation.ROOT
OUT = ROOT / "results" / "step5"
STAGE1_WEIGHTS = ("common_viewers", "cosine")
PERSONALIZATIONS = ppr.PERSONALIZATIONS
FIXED_ALPHA = 0.20
INITIAL_NEIGHBORS = 30
TUNED_NEIGHBORS = (20, 30, 50)
TUNED_ALPHAS = (0.10, 0.20, 0.35)
HISTORY_CAP = 40
MIN_COMMON = 3
TOLERANCE = 1e-8
MAX_ITERATIONS = 250
HALF_LIFE = ppr.RECENCY_HALF_LIFE_SECONDS


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_predictions(path, cohort, predictions):
    if tuple(predictions) != tuple(cohort):
        raise ValueError("PPR prediction order differs from frozen cohort")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        for user in cohort:
            ids = evaluation.strict_top5(list(predictions[user]), None) if False else tuple(predictions[user])
            if len(ids) != 5 or len(set(ids)) != 5 or any(int(i) <= 0 for i in ids):
                raise ValueError(f"Invalid PPR Top-5 for {user}")
            writer.writerow((user, " ".join(ids)))


def write_ap(path, cohort, evaluated, truth):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "ap_at_5", "validation_relevant_count"])
        for user in cohort:
            writer.writerow((user, format(evaluated["per_user_ap_at_5"][user], ".17g"), len(truth[user])))


def current_rss_mib():
    try:
        with open("/proc/self/statm") as stream:
            return int(stream.read().split()[1]) * __import__("os").sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None


def graph_filename(weight, neighbors):
    return OUT / f"graph_{weight}_top{neighbors}.npz"


def save_graph(graph, path):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **ppr.graph_npz_arrays(graph))
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": digest(path),
            "bytes": Path(path).stat().st_size, "statistics": graph.statistics(),
            "graph_build_seconds": graph.runtime_seconds}


def load_secondary_truth(config, cohort, candidates):
    """Secondary sensitivity only: user's five most frequent qualifying validation programs."""
    path = ROOT / config["validation_labels_source"]["path"]
    user_counts = {user: Counter() for user in cohort}
    cohort_set = set(cohort)
    for row in validation.iter_interactions(path, validation.WINDOWS["validation"], candidates,
                                           expected_hash=config["validation_labels_source"]["sha256"]):
        if row.user_id in cohort_set:
            user_counts[row.user_id][row.program_id] = row.qualifying_airings
    result = {}
    for user in cohort:
        ordered = sorted(user_counts[user], key=lambda item: (-user_counts[user][item], int(item)))
        result[user] = frozenset(ordered[:5])
    return result, user_counts


def score_predictions(protocol, predictions, truth):
    return evaluation.evaluate_predictions(predictions, truth, protocol.cohort,
                                           protocol.candidates, strict=True)


def summarize_scores(graph, infos, predictions, protocol, model):
    primary = protocol.evaluate(predictions)
    warm = [u for u in protocol.cohort if u in model.profiles]
    cold = [u for u in protocol.cohort if u not in model.profiles]
    warm_map = math.fsum(primary["per_user_ap_at_5"][u] for u in warm) / len(warm) if warm else None
    cold_map = math.fsum(primary["per_user_ap_at_5"][u] for u in cold) / len(cold) if cold else None
    top1 = sum(bool(predictions[u][0] in protocol.truth[u]) for u in protocol.cohort) / len(protocol.cohort)
    hits = sum(bool(set(predictions[u]) & protocol.truth[u]) for u in protocol.cohort)
    distinct = set(item for user in protocol.cohort for item in predictions[user])
    fallback_slots = sum(infos[u]["fallback_slots"] for u in protocol.cohort)
    cold_users = sum(infos[u]["cold_start"] for u in protocol.cohort)
    all_iterations = [infos[u]["iterations"] for u in protocol.cohort]
    nonconverged = [u for u in protocol.cohort if not infos[u]["converged"]]
    return {"primary_map_at_5": primary["primary_map_at_5"],
            "diagnostic_map_at_5_nonempty": primary["diagnostic_map_at_5_nonempty"],
            "hit_rate_at_5": hits / len(protocol.cohort), "users_with_at_least_one_primary_hit": hits,
            "top1_hit_rate": top1, "warm_primary_map_at_5": warm_map, "cold_primary_map_at_5": cold_map,
            "users_with_training_interactions": len(warm), "cold_start_users": cold_users,
            "cold_start_users_using_fallback": cold_users,
            "fallback_slots": fallback_slots,
            "distinct_recommended_programs": len(distinct),
            "candidate_coverage_fraction": len(distinct) / len(protocol.candidates),
            "candidate_count": len(protocol.candidates), "users": len(protocol.cohort),
            "nonempty_primary_label_users": primary["nonempty_users"], "empty_primary_label_users": primary["empty_users"],
            "mean_pagerank_iterations": math.fsum(all_iterations) / len(all_iterations),
            "max_pagerank_iterations": max(all_iterations), "nonconverged_users": len(nonconverged),
            "max_final_l1_residual": max(infos[u]["l1_residual"] for u in protocol.cohort),
            "graph": graph.statistics(), "graph_build_seconds": graph.runtime_seconds}


def run():
    if OUT.exists():
        raise FileExistsError("Preserve Step 5 outputs; use a fresh results/step5_replay directory for another run")
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = validation.EvaluationProtocol()
    config = protocol.config
    cohort, candidates = protocol.cohort, protocol.candidates
    output_started = time.monotonic()
    rss_start = current_rss_mib()

    # Step 4 baseline results and predictions are integrity checked before use.
    baseline_run_path = ROOT / "results" / "step4" / "step4_run.json"
    baseline_run = validation.json_read(baseline_run_path)
    if baseline_run["evaluation_config_sha256"] != protocol.config_sha256 or baseline_run["cohort_users"] != 1260:
        raise ValueError("Step 4 baseline was evaluated under a different protocol")
    step4_comparison = ROOT / "results" / "step4_baseline_comparison.csv"
    if digest(step4_comparison) != "18a3caea4d8b1ec777b8e99a83ead081449832a555654d3808a15560139c3f9e":
        raise ValueError("Verified Step 4 comparison CSV hash differs")
    best_popularity = "popularity_distinct_users"
    best_personal = "personal_frequency_qualifying_airings"
    baseline_best = {"popularity": baseline_run["variants"][best_popularity],
                     "personal_frequency": baseline_run["variants"][best_personal]}
    for family, key in (("popularity", best_popularity), ("personal_frequency", best_personal)):
        baseline_path = ROOT / baseline_run["variants"][key]["prediction_file"]
        if digest(baseline_path) != baseline_run["variants"][key]["prediction_sha256"]:
            raise ValueError(f"Step 4 {family} prediction hash mismatch")

    data = ppr.build_training_profiles(config)
    profiles_done = time.monotonic()
    pair_counts, pair_occurrences, truncated = ppr.build_copair_counts(data, history_cap=HISTORY_CAP)
    copairs_done = time.monotonic()
    graph_cache = {}
    graph_artifacts = {}
    graph_build_stage_seconds = {}
    def get_graph(weight, neighbors):
        key = (weight, neighbors)
        if key not in graph_cache:
            graph_started = time.monotonic()
            graph = ppr.build_graph_from_pair_counts(
                data.nodes, data.distinct_viewers, pair_counts,
                weight_type=weight, max_neighbors=neighbors,
                min_common_viewers=MIN_COMMON, history_cap=HISTORY_CAP,
                pair_occurrences_generated=pair_occurrences,
                profile_programs_truncated=truncated)
            graph = ppr.ProgramGraph(graph.nodes, graph.matrix, graph.raw_pair_counts,
                graph.retained_edge_weights, graph.weight_type, graph.max_neighbors,
                graph.history_cap, graph.min_common_viewers, graph.pair_occurrences_generated,
                graph.profile_programs_truncated, time.monotonic() - graph_started)
            graph_cache[key] = graph
            name = f"{weight}_top{neighbors}"
            graph_artifacts[name] = save_graph(graph, graph_filename(weight, neighbors))
            graph_build_stage_seconds[name] = graph.runtime_seconds
        return graph_cache[key]

    pop = popularity.fit_popularity(config, candidates)
    fallback = pop.ranking("distinct_users")
    experiment_specs = []
    for weight in STAGE1_WEIGHTS:
        for pers in PERSONALIZATIONS:
            experiment_specs.append({"stage": "stage1_graph_personalization", "weight": weight,
                "personalization": pers, "restart_probability": FIXED_ALPHA,
                "max_neighbors": INITIAL_NEIGHBORS, "history_cap": HISTORY_CAP,
                "min_common_viewers": MIN_COMMON, "tolerance": TOLERANCE,
                "max_iterations": MAX_ITERATIONS, "recency_half_life_seconds": HALF_LIFE})

    configs, predictions_by_id, infos_by_id, experiment_rows = {}, {}, {}, []
    for spec in experiment_specs:
        graph = get_graph(spec["weight"], spec["max_neighbors"])
        config_id = (f"{spec['weight']}_p{spec['personalization']}_a{spec['restart_probability']:.2f}"
                    f"_k{spec['max_neighbors']}")
        pstart = time.monotonic()
        preds, infos = ppr.predict_cohort(graph, data, cohort, candidates, fallback,
            personalization=spec["personalization"], restart_probability=spec["restart_probability"],
            tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, half_life_seconds=HALF_LIFE)
        pred_seconds = time.monotonic() - pstart
        primary = protocol.evaluate(preds)
        metrics = summarize_scores(graph, infos, preds, protocol, data)
        metrics.update(primary_map_at_5=primary["primary_map_at_5"],
                       diagnostic_map_at_5_nonempty=primary["diagnostic_map_at_5_nonempty"])
        metrics.update({k:v for k,v in spec.items() if k != "stage"})
        metrics["stage"] = spec["stage"]
        metrics["prediction_generation_seconds"] = pred_seconds
        metrics["config_id"] = config_id
        configs[config_id] = {"config_id": config_id, **spec}
        predictions_by_id[config_id], infos_by_id[config_id] = preds, infos
        pred_path = OUT / "experiments" / config_id / "predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_predictions(pred_path, cohort, preds)
        user_ap_path = OUT / "experiments" / config_id / "user_ap.csv"
        with user_ap_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["user_id", "ap_at_5", "validation_relevant_count", "cold_start"])
            for user in cohort:
                writer.writerow((user, format(primary["per_user_ap_at_5"][user], ".17g"),
                                 len(protocol.truth[user]),
                                 int(user not in data.profiles)))
        metrics["prediction_file"] = str(pred_path.relative_to(ROOT))
        metrics["prediction_sha256"] = digest(pred_path)
        metrics["user_ap_file"] = str(user_ap_path.relative_to(ROOT))
        metrics["user_ap_sha256"] = digest(user_ap_path)
        metrics["graph_file"] = graph_artifacts[f"{spec['weight']}_top{spec['max_neighbors']}"]["path"]
        metrics["ppr_training_runtime_seconds"] = data.runtime_seconds + (copairs_done - profiles_done) + graph.runtime_seconds
        metrics["popularity_fallback_training_runtime_seconds"] = pop.training_runtime_seconds
        experiment_rows.append(metrics)
        print("Stage1", config_id, "MAP@5", metrics["primary_map_at_5"], "sec", pred_seconds, flush=True)

    stage1_best = max(experiment_rows, key=lambda r: (r["primary_map_at_5"], r["config_id"]))
    selected_weight, selected_personalization = stage1_best["weight"], stage1_best["personalization"]
    # Stage 2: tune restart probability and neighborhood size only for the best stage-1 pair.
    for neighbors in TUNED_NEIGHBORS:
        for alpha in TUNED_ALPHAS:
            config_id = f"{selected_weight}_p{selected_personalization}_a{alpha:.2f}_k{neighbors}"
            if config_id in configs:
                continue
            spec = {"stage": "stage2_restart_and_neighborhood", "weight": selected_weight,
                "personalization": selected_personalization, "restart_probability": alpha,
                "max_neighbors": neighbors, "history_cap": HISTORY_CAP,
                "min_common_viewers": MIN_COMMON, "tolerance": TOLERANCE,
                "max_iterations": MAX_ITERATIONS, "recency_half_life_seconds": HALF_LIFE}
            graph = get_graph(selected_weight, neighbors)
            pstart = time.monotonic()
            preds, infos = ppr.predict_cohort(graph, data, cohort, candidates, fallback,
                personalization=selected_personalization, restart_probability=alpha,
                tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, half_life_seconds=HALF_LIFE)
            pred_seconds = time.monotonic() - pstart
            primary = protocol.evaluate(preds)
            metrics = summarize_scores(graph, infos, preds, protocol, data)
            metrics.update(primary_map_at_5=primary["primary_map_at_5"],
                           diagnostic_map_at_5_nonempty=primary["diagnostic_map_at_5_nonempty"])
            metrics.update({k:v for k,v in spec.items() if k != "stage"})
            metrics["stage"] = spec["stage"]
            metrics["prediction_generation_seconds"] = pred_seconds
            metrics["config_id"] = config_id
            configs[config_id] = {"config_id": config_id, **spec}
            predictions_by_id[config_id], infos_by_id[config_id] = preds, infos
            pred_path = OUT / "experiments" / config_id / "predictions.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            write_predictions(pred_path, cohort, preds)
            user_ap_path = OUT / "experiments" / config_id / "user_ap.csv"
            with user_ap_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["user_id", "ap_at_5", "validation_relevant_count", "cold_start"])
                for user in cohort:
                    writer.writerow((user, format(primary["per_user_ap_at_5"][user], ".17g"),
                                     len(protocol.truth[user]),
                                     int(user not in data.profiles)))
            metrics["prediction_file"] = str(pred_path.relative_to(ROOT))
            metrics["prediction_sha256"] = digest(pred_path)
            metrics["user_ap_file"] = str(user_ap_path.relative_to(ROOT))
            metrics["user_ap_sha256"] = digest(user_ap_path)
            metrics["graph_file"] = graph_artifacts[f"{selected_weight}_top{neighbors}"]["path"]
            metrics["ppr_training_runtime_seconds"] = data.runtime_seconds + (copairs_done - profiles_done) + graph.runtime_seconds
            metrics["popularity_fallback_training_runtime_seconds"] = pop.training_runtime_seconds
            experiment_rows.append(metrics)
            print("Stage2", config_id, "MAP@5", metrics["primary_map_at_5"], "sec", pred_seconds, flush=True)

    # Secondary Top-5-oriented targets are constructed ONLY after every ranking
    # is frozen. They are used exclusively for a separately named sensitivity score.
    secondary_truth, secondary_counts = load_secondary_truth(config, cohort, candidates)
    for row in experiment_rows:
        preds = predictions_by_id[row["config_id"]]
        secondary = evaluation.evaluate_predictions(preds, secondary_truth, cohort, candidates, strict=True)
        row["secondary_top5_sensitivity_map_at_5"] = secondary["primary_map_at_5"]
        row["secondary_top5_sensitivity_nonempty_map_at_5"] = secondary["diagnostic_map_at_5_nonempty"]

    best = max(experiment_rows, key=lambda r: (r["primary_map_at_5"], r["config_id"]))
    best_id = best["config_id"]
    best_predictions = predictions_by_id[best_id]
    best_graph = graph_cache[(best["weight"], best["max_neighbors"])]
    # Copy best artifact to the required short canonical filename.
    write_predictions(OUT / "ppr_predictions.csv", cohort, best_predictions)
    with (OUT / "ppr_user_ap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "ap_at_5", "validation_relevant_count", "cold_start"])
        primary = protocol.evaluate(best_predictions)
        for user in cohort:
            writer.writerow((user, format(primary["per_user_ap_at_5"][user], ".17g"), len(protocol.truth[user]),
                             int(user not in data.profiles)))
    # Persist candidate Top-5 target for every validation cohort user as a separate, clearly secondary label set.
    with (OUT / "secondary_top5_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "top5_validation_program_ids", "relevant_count", "tie_break"])
        for user in cohort:
            ordered = sorted(secondary_counts[user], key=lambda item: (-secondary_counts[user][item], int(item)))[:5]
            writer.writerow((user, " ".join(ordered), len(ordered), "qualifying_airings_desc,numeric_program_id_asc"))

    # Re-score each saved Step 4 baseline against the separately defined Top-5 target.
    baseline_secondary = {}
    step4run = validation.json_read(ROOT / "results" / "step4" / "step4_run.json")
    for key, info in step4run["variants"].items():
        prediction_path = ROOT / info["prediction_file"]
        if digest(prediction_path) != info["prediction_sha256"]:
            raise ValueError("A saved Step 4 baseline prediction hash does not match")
        raw = validation.records(prediction_path, ["user_id", "tv_show_id"])
        predictions = evaluation.parse_prediction_records({"user_id":u,"tv_show_id":items} for u,items in raw)
        baseline_secondary[key] = evaluation.evaluate_predictions(predictions, secondary_truth, cohort, candidates, strict=True)

    selected_graph_names = sorted(graph_artifacts)
    graph_summaries = {name: value for name, value in graph_artifacts.items()}
    final_config = {"step": 5, "method": "Personalized PageRank on sparse program co-view graph",
        "best_config_id": best_id, "best_config": configs[best_id], "selected_stage1_pair": {
            "edge_weighting": selected_weight, "personalization": selected_personalization},
        "selection_metric": "frozen Step 3 primary MAP@5 over all 1,260 users",
        "evaluation_config_sha256": protocol.config_sha256, "protocol_id": config["protocol_id"],
        "training_input_sha256": data.training_sha256, "training_window_utc": data.training_window_utc,
        "validation_truth_sha256": config["truth"]["sha256"], "candidate_pool_sha256": evaluation.identity_hash(candidates),
        "cohort_order_sha256": evaluation.identity_hash(cohort), "history_cap": HISTORY_CAP,
        "history_cap_rule": "per user top 40 programs by qualifying_airings desc, watched_seconds desc, numeric program ID asc",
        "graph_min_common_viewers": MIN_COMMON, "graph_symmetrization": "union of each node's top-k neighbors; shared raw weight retained symmetrically",
        "restart_probability_semantics": "alpha is restart probability in r=alpha*p+(1-alpha)*(P.T@r+dangling_mass*p)",
        "personalization_weights": {"uniform":"equal per distinct training program",
                                    "frequency":"qualifying_airings proportional",
                                    "duration":"watched_seconds proportional",
                                    "recency":f"2^(-age_seconds/{HALF_LIFE}) at training cutoff"},
        "restart_probabilities_stage1":[FIXED_ALPHA], "restart_probabilities_stage2":list(TUNED_ALPHAS),
        "neighbor_limits_stage1":[INITIAL_NEIGHBORS], "neighbor_limits_stage2":list(TUNED_NEIGHBORS),
        "convergence":{"absolute_l1_tolerance":TOLERANCE,"max_iterations":MAX_ITERATIONS,
                       "dangling_mass":"redistributed to the user's restart distribution"},
        "graph_artifacts":graph_summaries,"numpy_version":np.__version__,"scipy_version":scipy.__version__,
        "python_version":__import__("sys").version,"random_seed":None,
        "secondary_protocol":"each user's five validation programs with highest qualifying_airings; ties numeric ID ascending; not claimed to match Kaggle hidden labels",
        "prediction_policy":"score only validation schedule candidates with positive PPR; fill to five by Step 3 distinct-user popularity fallback; repeats across time allowed"}
    write_json(OUT / "ppr_model_config.json", final_config)

    # Compact experiment table and common comparison table.
    experiment_fields = ["config_id","stage","weight","personalization","restart_probability","max_neighbors",
        "history_cap","min_common_viewers","primary_map_at_5","diagnostic_map_at_5_nonempty",
        "secondary_top5_sensitivity_map_at_5","secondary_top5_sensitivity_nonempty_map_at_5",
        "hit_rate_at_5","top1_hit_rate","warm_primary_map_at_5","cold_primary_map_at_5",
        "distinct_recommended_programs","candidate_coverage_fraction","fallback_slots","cold_start_users_using_fallback",
        "graph_build_seconds","ppr_training_runtime_seconds","prediction_generation_seconds","mean_pagerank_iterations","max_pagerank_iterations",
        "nonconverged_users","max_final_l1_residual",
        "popularity_fallback_training_runtime_seconds","prediction_sha256","user_ap_sha256"]
    with (OUT / "ppr_experiments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=experiment_fields, lineterminator="\n")
        writer.writeheader()
        for row in experiment_rows:
            writer.writerow({k:row.get(k) for k in experiment_fields})

    step4_best = {"popularity": step4run["variants"]["popularity_distinct_users"],
                  "personal_frequency": step4run["variants"]["personal_frequency_qualifying_airings"]}
    comparison = [{"family":"PPR", "variant":r["config_id"], "primary_map_at_5":r["primary_map_at_5"],
                   "diagnostic_map_at_5_nonempty":r["diagnostic_map_at_5_nonempty"],
                   "secondary_top5_sensitivity_map_at_5":None,
                   "hit_rate_at_5":r["hit_rate_at_5"], "top1_hit_rate":r["top1_hit_rate"],
                   "training_runtime_seconds":r["ppr_training_runtime_seconds"],
                   "prediction_runtime_seconds":r["prediction_generation_seconds"],
                   "candidate_coverage_fraction":r["candidate_coverage_fraction"],
                   "cold_start_users_using_fallback":r["cold_start_users_using_fallback"],
                   "fallback_slots":r["fallback_slots"], "source":"results/step5/ppr_experiments.csv"} for r in experiment_rows]
    for family, info in step4_best.items():
        baseline_key = f"{info['family']}_{info['variant']}"
        comparison.append({"family":family,"variant":info["variant"],
            "primary_map_at_5":info["primary_map_at_5"],"diagnostic_map_at_5_nonempty":info["diagnostic_map_at_5_nonempty"],
            "secondary_top5_sensitivity_map_at_5":baseline_secondary[baseline_key]["primary_map_at_5"],
            "hit_rate_at_5":info["users_with_at_least_one_relevant_recommendation"]/len(cohort),
            "top1_hit_rate":None,"training_runtime_seconds":info["training_runtime_seconds"],
            "prediction_runtime_seconds":info["prediction_runtime_seconds"],
            "candidate_coverage_fraction":info["candidate_coverage_fraction"],
            "cold_start_users_using_fallback":info["users_using_full_popularity_fallback"],
            "fallback_slots":info["popularity_backfill_slots_exact"],
            "source":info["prediction_file"]})
    with (ROOT / "results" / "step5_ppr_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({k for row in comparison for k in row})
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)

    # Reuse the fixed Step 4 user IDs, full histories, and the baseline methods.
    from step5_examples import write_user_examples
    examples_summary = write_user_examples(OUT / "ppr_user_examples.md", protocol, best_predictions, data, best_id)
    graph_manifest = {name: {"path": info["path"], "sha256": info["sha256"],
                             "bytes": info["bytes"], "statistics": info["statistics"],
                             "graph_build_seconds": info["graph_build_seconds"]}
                      for name, info in graph_artifacts.items()}
    best_graph_info = graph_artifacts[f"{best['weight']}_top{best['max_neighbors']}"]
    ppr_config = {**final_config, "best_metrics": best, "best_prediction_sha256": digest(OUT / "ppr_predictions.csv"),
                  "best_user_ap_sha256": digest(OUT / "ppr_user_ap.csv"),
                  "secondary_top5_truth_sha256": digest(OUT / "secondary_top5_truth.csv"),
                  "ppr_user_examples_sha256": digest(OUT / "ppr_user_examples.md"),
                  "fixed_example_users": examples_summary["selected_users"],
                  "step4_baseline_run_sha256": digest(ROOT / "results" / "step4" / "step4_run.json"),
                  "step4_comparison_sha256": digest(ROOT / "results" / "step4_baseline_comparison.csv"),
                  "graph_artifacts": graph_manifest,
                  "profile_build_seconds": data.runtime_seconds,
                  "pair_count_build_seconds": copairs_done - profiles_done,
                  "popularity_fallback_training_seconds":pop.training_runtime_seconds,
                  "graph_cache_keys": sorted(graph_manifest),
                  "total_graph_and_ppr_run_seconds": time.monotonic()-output_started,
                  "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  "cold_start_users": sum(user not in data.profiles for user in cohort),
                  "nonconverged_users_best": best["nonconverged_users"]}
    write_json(OUT / "ppr_model_config.json", ppr_config)

    experiment_metadata = {str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":digest(p)}
                           for p in sorted(OUT.rglob("*")) if p.is_file()}
    write_json(OUT / "ppr_artifacts.json", experiment_metadata)
    print(json.dumps({"best_config_id":best_id,"best_primary_MAP@5":best["primary_map_at_5"],
        "best_secondary_MAP@5":best["secondary_top5_sensitivity_map_at_5"],
        "stage1_best":stage1_best["config_id"],"experiment_configurations":len(experiment_rows),
        "graph_artifacts":{k:v["statistics"] for k,v in graph_artifacts.items()},
        "profile_runtime_seconds":data.runtime_seconds,"co_pair_generation_seconds":copairs_done-profiles_done,
        "ppr_and_predictions_seconds":time.monotonic()-copairs_done,
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "baseline_primary_maps":{k:v["primary_map_at_5"] for k,v in step4_best.items()}},indent=2))


if __name__ == "__main__":
    run()
