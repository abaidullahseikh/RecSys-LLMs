"""Fit/evaluate baseline references on the frozen Step 3 protocol only."""

import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import evaluation
import personal_frequency
import popularity
import validation

ROOT = validation.ROOT
OUT = ROOT / "results" / "step4"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def current_rss_mib():
    try:
        with open("/proc/self/statm") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None


def save_predictions(path, cohort, predictions):
    if tuple(predictions) != tuple(cohort):
        raise ValueError("Predictions not returned in frozen cohort order")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        for u in cohort:
            ids = tuple(evaluation.program_id(i) for i in predictions[u])
            if len(ids) != 5 or len(set(ids)) != 5 or any(int(i) <= 0 for i in ids):
                raise ValueError(f"Invalid Top-5 for user {u}")
            writer.writerow([u, " ".join(ids)])


def save_user_ap(path, cohort, result, truth):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["user_id", "ap_at_5", "validation_relevant_count", "has_training_history"])
        for u in cohort:
            writer.writerow([u, format(result["per_user_ap_at_5"][u], ".17g"), len(truth[u]), int(u in result["training_users"])])


def load_validation_titles(schedule_paths, window):
    """Display-only metadata lookup from validation schedule, never a ranking feature."""
    from validation import records, SCHEDULE_COLUMNS, schedule_program_id
    from data_pipeline import parse_timestamp
    titles = {}
    conflicts = Counter()
    for path in schedule_paths:
        for row in records(path, SCHEDULE_COLUMNS):
            item = schedule_program_id(row[5])
            if item == "0":
                continue
            start = parse_timestamp(row[2], schedule=True)
            end = start + int(row[3])
            if start < window[1] and end > window[0]:
                value = {"title": row[4], "category": row[6]}
                if item not in titles:
                    titles[item] = value
                elif titles[item] != value:
                    conflicts[item] += 1
    return titles, conflicts


def choose_users(protocol, personal_model):
    """Predeclare one training-history maximum and one minimum positive history.

    Original cohort order breaks ties, and validation labels/outputs are not read.
    """
    sizes = {u: len(personal_model.user_airings.get(u, {})) for u in protocol.cohort}
    warm = [u for u in protocol.cohort if sizes[u] > 0]
    if len(warm) < 2:
        raise ValueError("Cannot select heavy and light examples from fewer than two warm users")
    # Python max/min preserve the first original cohort entry on tied history size.
    heavy = max(warm, key=lambda u: sizes[u])
    light = min((u for u in warm if u != heavy), key=lambda u: sizes[u])
    return {"selection_rule": "heaviest = maximum distinct training programs; lightest = minimum positive distinct training programs among remaining warm cohort users; ties use original cohort order",
            "uses_validation_labels": False,
            "heavy_user_id": heavy, "heavy_training_distinct_programs": sizes[heavy],
            "light_user_id": light, "light_training_distinct_programs": sizes[light]}


def run():
    if OUT.exists():
        raise FileExistsError("Preserve Step 4 outputs; use a new run directory instead of overwriting")
    OUT.mkdir(parents=True)
    protocol = validation.EvaluationProtocol()
    cohort, candidates, truth = protocol.cohort, protocol.candidates, protocol.truth
    config = protocol.config
    started = time.monotonic()
    rss_start = current_rss_mib()
    pop_model = popularity.fit_popularity(config, candidates)
    rss_after_pop = current_rss_mib()
    personal_model = personal_frequency.fit_personal_frequency(config, candidates)
    rss_after_personal = current_rss_mib()
    training_done = time.monotonic()

    training_users = set(personal_model.user_airings)

    variants = {}
    for family, names in (("popularity", popularity.VARIANTS), ("personal_frequency", personal_frequency.VARIANTS)):
        for variant in names:
            predict_started = time.monotonic()
            if family == "popularity":
                predictions = popularity.predictions(pop_model, variant, cohort)
                fallback_only = len(cohort)  # The global baseline is the common fallback ranking itself.
                fallback_slots = len(cohort) * 5
            else:
                predictions = personal_frequency.predict(personal_model, variant, cohort, pop_model)
                fallback_only = sum(u not in personal_model.user_airings for u in cohort)
                personal_eligible = {u: set(personal_model.user_airings.get(u, {})) & set(candidates)
                                     for u in cohort}
                fallback_slots = sum(max(0, 5 - len(personal_eligible[u])) for u in cohort)
            prediction_seconds = time.monotonic() - predict_started
            # Strict evaluator independently validates every user/list and computes all metrics.
            result = protocol.evaluate(predictions)
            result["training_users"] = training_users
            filename = f"{family}_{variant}"
            pred_path = OUT / f"{filename}_predictions.csv"
            save_predictions(pred_path, cohort, predictions)
            ap_path = OUT / f"{filename}_user_ap.csv"
            save_user_ap(ap_path, cohort, result, truth)
            all_recommended = [item for row in predictions.values() for item in row]
            hit_users = sum(bool(set(predictions[u]) & truth[u]) for u in cohort)
            warm_aps = [result["per_user_ap_at_5"][u] for u in cohort if u in training_users]
            cold_aps = [result["per_user_ap_at_5"][u] for u in cohort if u not in training_users]
            users_with_labels = [u for u in cohort if truth[u]]
            without_labels = [u for u in cohort if not truth[u]]
            metrics = {k: v for k, v in result.items() if k not in {"per_user_ap_at_5", "training_users"}}
            metrics.update({
                "family": family, "variant": variant,
                "definition": ("sort all 6,384 validation candidates by this training-only global score; ties use distinct users desc, qualifying airings desc, numeric ID asc"
                               if family == "popularity" else
                               "sort the user's training programs that are in the validation pool by this score; ties use numeric ID asc; append distinct-user popularity fallback for missing slots"),
                "training_period_utc": list(pop_model.training_window_utc),
                "training_interaction_rows_read": pop_model.training_rows if family == "popularity" else personal_model.training_rows,
                "training_runtime_seconds": pop_model.training_runtime_seconds if family == "popularity" else personal_model.training_runtime_seconds,
                "prediction_runtime_seconds": prediction_seconds,
                "fallback_model_training_runtime_seconds": pop_model.training_runtime_seconds if family == "personal_frequency" else None,
                "users_using_full_popularity_fallback": fallback_only,
                "users_using_popularity_backfill_any_slot": 0 if family == "popularity" else sum(
                    len(personal_eligible[u]) < 5 for u in cohort),
                "popularity_backfill_slots_exact": fallback_slots,
                "candidate_pool_size": len(candidates),
                "distinct_recommended_programs": len(set(all_recommended)),
                "candidate_coverage_fraction": len(set(all_recommended)) / len(candidates),
                "users_with_at_least_one_relevant_recommendation": hit_users,
                "warm_users": len(training_users & set(cohort)),
                "cold_users": len(set(cohort) - training_users),
                "warm_primary_map_at_5": sum(warm_aps) / len(warm_aps) if warm_aps else None,
                "cold_primary_map_at_5": sum(cold_aps) / len(cold_aps) if cold_aps else None,
                "labelled_users": len(users_with_labels),
                "empty_label_users": len(without_labels),
                "labelled_users_with_at_least_one_hit": sum(bool(set(predictions[u]) & truth[u]) for u in users_with_labels),
                "cold_labelled_users": sum(u not in training_users and bool(truth[u]) for u in cohort),
                "prediction_file": str(pred_path.relative_to(ROOT)), "prediction_sha256": digest(pred_path),
                "user_ap_file": str(ap_path.relative_to(ROOT)), "user_ap_sha256": digest(ap_path),
                "fallback_policy": evaluation.FALLBACK_POLICY if family == "personal_frequency" else "the popularity ranking itself is the global baseline",
                "repeat_previously_watched": True,
                "evaluation_config_sha256": protocol.config_sha256,
                "candidate_ids_sha256": evaluation.identity_hash(candidates),
                "truth_sha256": config["truth"]["sha256"],
                "peak_process_rss_mib_after_fit": rss_after_pop if family == "popularity" else rss_after_personal,
            })
            variants[filename] = {"predictions": predictions, "metrics": metrics, "per_user_ap": result["per_user_ap_at_5"]}

    # Remove non-JSON objects accidentally carried for in-process metrics before writing.
    for value in variants.values():
        value["metrics"].pop("per_user_ap_at_5", None)
    schedule_paths = [Path(p) for p in config["schedule_source_hashes"]]
    titles, title_conflicts = load_validation_titles(schedule_paths, validation.WINDOWS["validation"])
    examples = choose_users(protocol, personal_model)
    examples["users"] = {}
    for role, user in (("heavy", examples["heavy_user_id"]), ("light", examples["light_user_id"])):
        train_history = personal_model.user_airings.get(user, {})
        train_sec = personal_model.user_seconds.get(user, {})
        history = sorted(train_history, key=lambda p: (-train_history[p], -train_sec[p], int(p)))
        relevant = sorted(truth[user], key=int)
        examples["users"][role] = {
            "user_id": user,
            "training_program_count": len(train_history),
            "training_qualifying_airings": sum(train_history.values()),
            "training_watched_seconds": sum(train_sec.values()),
            "training_top_programs_by_airings": [
                {"tv_show_id": p, "qualifying_airings": train_history[p], "watched_seconds": train_sec[p],
                 "title": titles.get(p, {}).get("title"), "category": titles.get(p, {}).get("category")}
                for p in history[:10]],
            "all_validation_relevant_ids": relevant,
            "validation_relevant_count": len(relevant),
            "validation_relevant_metadata": [{"tv_show_id": p, "title": titles.get(p, {}).get("title"),
                                               "category": titles.get(p, {}).get("category")} for p in relevant[:20]],
            "recommendations": {name: [{"tv_show_id": p, "title": titles.get(p, {}).get("title"),
                                        "category": titles.get(p, {}).get("category"),
                                        "relevant_in_validation": p in truth[user]}
                                       for p in result["predictions"][user]] for name, result in variants.items()}}
    examples["metadata_source"] = "validation-period schedule rows; display-only lookup, not model feature"
    examples["metadata_conflicting_title_category_counts"] = len(title_conflicts)
    write_json(OUT / "provisional_two_users.json", examples)

    # Family variants are compared separately; no all-variants winner chosen across user-specific/global objectives.
    summary = {"stage": "Step 4 baselines only", "evaluation_config_sha256": protocol.config_sha256,
               "protocol_id": protocol.config["protocol_id"], "cohort_users": len(cohort),
               "training_runtime_seconds": {"popularity": pop_model.training_runtime_seconds,
                                            "personal_frequency": personal_model.training_runtime_seconds},
               "prediction_runtime_seconds": {k: v["metrics"]["prediction_runtime_seconds"] for k, v in variants.items()},
               "runtime_seconds_through_prediction": time.monotonic() - started,
               "rss_current_start_mib": rss_start, "rss_after_popularity_fit_mib": rss_after_pop,
               "rss_after_personal_fit_mib": rss_after_personal,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "training_rows": {"popularity": pop_model.training_rows, "personal_frequency": personal_model.training_rows},
               "training_hash": pop_model.training_sha256, "fallback_ranking": pop_model.ranking("distinct_users"),
               "popularity_statistics": {"distinct_users": pop_model.distinct_users,
                                         "qualifying_airings": pop_model.qualifying_airings,
                                         "watched_seconds": pop_model.watched_seconds},
               "variants": {k: v["metrics"] for k, v in variants.items()},
               "best_popularity_variant_by_primary_map": max(popularity.VARIANTS,
                   key=lambda name: variants[f"popularity_{name}"]["metrics"]["primary_map_at_5"]),
               "best_personal_frequency_variant_by_primary_map": max(personal_frequency.VARIANTS,
                   key=lambda name: variants[f"personal_frequency_{name}"]["metrics"]["primary_map_at_5"]),
               "synthetic_scores_only": False, "no_test_user_behavior_accessed": True,
               "no_kaggle_submission": True}
    write_json(OUT / "step4_run.json", summary)
    comparison_fields = ["model", "variant", "primary_map_at_5", "diagnostic_map_at_5_nonempty",
                         "warm_primary_map_at_5", "cold_primary_map_at_5", "training_runtime_seconds",
                         "prediction_runtime_seconds", "peak_process_rss_mib_after_fit", "candidate_pool_size",
                         "distinct_recommended_programs", "candidate_coverage_fraction",
                         "users_with_at_least_one_relevant_recommendation", "users_using_full_popularity_fallback",
                         "users_using_popularity_backfill_any_slot", "popularity_backfill_slots_exact",
                         "labelled_users_with_at_least_one_hit", "prediction_sha256", "user_ap_sha256"]
    with (OUT / "step4_baseline_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields, lineterminator="\n")
        writer.writeheader()
        for name, item in variants.items():
            row = {"model": item["metrics"]["family"], "variant": item["metrics"]["variant"]}
            row.update({field: item["metrics"].get(field) for field in comparison_fields if field not in {"model", "variant"}})
            writer.writerow(row)
    # Hash manifest written after all experiment artifacts exist.
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "step4_outputs.json"}
    write_json(OUT / "step4_outputs.json", outputs)
    print(json.dumps({"variants": {k: {x: v["metrics"].get(x) for x in (
        "primary_map_at_5", "diagnostic_map_at_5_nonempty", "warm_primary_map_at_5", "cold_primary_map_at_5",
        "training_runtime_seconds", "prediction_runtime_seconds", "users_with_at_least_one_relevant_recommendation",
        "distinct_recommended_programs", "users_using_full_popularity_fallback", "popularity_backfill_slots_exact")}
        for k, v in variants.items()}, "two_users": {k: examples[k] for k in ("heavy_user_id", "light_user_id")},
        "training_runtime": summary["training_runtime_seconds"], "peak_rss_mib": summary["peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    run()
