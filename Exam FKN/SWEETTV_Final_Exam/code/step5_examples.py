"""Build fixed Step 4 user histories/comparisons; metadata is display-only."""

import csv
from pathlib import Path

import evaluation
import validation


def _metadata(schedule_path):
    from validation import SCHEDULE_COLUMNS, records, schedule_program_id
    result = {}
    for row in records(schedule_path, SCHEDULE_COLUMNS):
        item = schedule_program_id(row[5])
        if item != "0" and item not in result:
            result[item] = {"title": row[4], "category": row[6]}
    return result


def _read_baseline(path, cohort):
    rows = list(validation.records(path, ["user_id", "tv_show_id"]))
    return evaluation.parse_prediction_records({"user_id": user, "tv_show_id": field} for user, field in rows)


def write_user_examples(output_path, protocol, best_predictions, train_profiles, best_config_id):
    """The fixed Step 4 heavy/light IDs; no reselection based on validation output."""
    import json
    step4 = json.loads((validation.ROOT / "results" / "step4" / "provisional_two_users.json").read_text())
    ids = {"heavy": step4["heavy_user_id"], "light": step4["light_user_id"]}
    config = protocol.config
    schedule = config["schedule_source_hashes"]
    train_path = next(path for path in schedule if Path(path).name == "export_arh_11-20-final.csv")
    validation_path = next(path for path in schedule if Path(path).name == "export_arh_21-30-final.csv")
    train_meta = _metadata(train_path)
    validation_meta = _metadata(validation_path)
    pop = _read_baseline(validation.ROOT / "results" / "step4" / "popularity_distinct_users_predictions.csv", protocol.cohort)
    personal = _read_baseline(validation.ROOT / "results" / "step4" / "personal_frequency_qualifying_airings_predictions.csv", protocol.cohort)
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Step 5 PPR examples for the fixed Step 4 users\n\n")
        stream.write("These are the exact heavy/light users designated in Step 4, chosen from training-history size before comparing recommendation results. Program titles/categories are display-only metadata from the corresponding historical schedule. Validation relevance indicates logged qualifying viewing, not enjoyment.\n\n")
        stream.write(f"Best PPR configuration: `{best_config_id}`.\n")
        for role in ("heavy", "light"):
            user = ids[role]
            profile = train_profiles.profiles.get(user, {})
            labels = protocol.truth[user]
            stream.write(f"\n## {role.title()} user `{user}`\n\n")
            stream.write(f"Training: {len(profile)} distinct programs; {sum(v[0] for v in profile.values())} qualifying airings; {sum(v[1] for v in profile.values())} watched seconds. Validation labels: {len(labels)} distinct programs.\n\n")
            train_items = sorted(profile, key=lambda item: (-profile[item][0], -profile[item][1], int(item)))
            stream.write("### Training-period program history (all distinct qualifying programs)\n\n")
            stream.write("| Program ID | Title | Category | Qualifying airings | Watched seconds | Last training watch (UTC) |\n|---:|---|---|---:|---:|---:|\n")
            for item in train_items:
                meta = train_meta.get(item, {})
                airs, seconds, last = profile[item]
                stream.write(f"| {item} | {meta.get('title','(metadata unavailable)')} | {meta.get('category','(metadata unavailable)')} | {airs} | {seconds} | {last} |\n")
            stream.write("\n### Validation-period relevant programs (all distinct qualifying programs)\n\n")
            stream.write("| Program ID | Title | Category |\n|---:|---|---|\n")
            for item in sorted(labels, key=int):
                meta = validation_meta.get(item, {})
                stream.write(f"| {item} | {meta.get('title','(metadata unavailable)')} | {meta.get('category','(metadata unavailable)')} |\n")
            stream.write("\n### Top-5 comparison\n\n")
            stream.write("| Method | Rank | Program ID | Title | Category | In validation truth |\n|---|---:|---:|---|---|---|\n")
            lists = [("Popularity (distinct training users)", pop[user]),
                     ("Personal frequency (qualifying airings)", personal[user]),
                     ("Personalized PageRank", best_predictions[user])]
            for method, ranking in lists:
                for rank, item in enumerate(ranking, 1):
                    meta = validation_meta.get(item, {})
                    stream.write(f"| {method} | {rank} | {item} | {meta.get('title','(metadata unavailable)')} | {meta.get('category','(metadata unavailable)')} | {'yes' if item in labels else 'no'} |\n")
    return {"selected_users": ids, "training_program_counts": {role: len(train_profiles.profiles.get(u, {})) for role,u in ids.items()},
            "validation_label_counts": {role: len(protocol.truth[u]) for role,u in ids.items()},
            "output_sha256": validation.sha256(path)}
