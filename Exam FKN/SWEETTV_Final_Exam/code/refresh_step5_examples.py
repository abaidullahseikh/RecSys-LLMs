"""Add required Step 4 baselines, complete histories and all labels to fixed examples."""

from pathlib import Path
import json

import evaluation
import ppr
import validation
from step5_examples import write_user_examples

ROOT=validation.ROOT
OUT=ROOT/"results"/"step5"


def main():
    protocol=validation.EvaluationProtocol()
    data=ppr.build_training_profiles(protocol.config)
    predictions=evaluation.parse_prediction_records(
        {"user_id":u,"tv_show_id":field} for u,field in validation.records(OUT/"ppr_predictions.csv",["user_id","tv_show_id"]))
    model=json.loads((OUT/"ppr_model_config.json").read_text())
    summary=write_user_examples(OUT/"ppr_user_examples.md",protocol,predictions,data,model["best_config_id"])
    model["ppr_user_examples_sha256"]=summary["output_sha256"]
    model["fixed_example_users"]=summary["selected_users"]
    (OUT/"ppr_model_config.json").write_text(json.dumps(model,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    files={str(p.relative_to(OUT)): {"bytes":p.stat().st_size,"sha256":validation.sha256(p)}
           for p in sorted(OUT.rglob("*")) if p.is_file()
           and p.name not in {"ppr_artifacts.json","step5_verification.json"}}
    (OUT/"ppr_artifacts.json").write_text(json.dumps(files,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"selected_users":summary["selected_users"],"training_program_counts":summary["training_program_counts"],
                      "validation_label_counts":summary["validation_label_counts"],"report_sha256":summary["output_sha256"]},indent=2))


if __name__=="__main__":
    main()
