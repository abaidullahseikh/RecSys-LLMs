#!/usr/bin/env python3
"""Verify and stage the completed HW6 evidence bundle without touching originals."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "final_submission_staging"
ZIP_PATH = ROOT / "HW6_final_report_evidence.zip"
EVIDENCE = ROOT / "evidence"
MF_EVIDENCE = EVIDENCE / "hw5_mf_reproduction"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_and_stage():
    required = [
        ROOT / "readme.md", ROOT / "README.md", ROOT / "README_hw5_mf_reproduction.md",
        ROOT / "index.html", ROOT / "script.js", ROOT / "graph.js", ROOT / "experiment.js",
        ROOT / "data.js", ROOT / "style.css", ROOT / "vendor/d3.min.js",
        ROOT / "u.data", ROOT / "u.item", ROOT / "run_experiment.py", ROOT / "export_evidence.py",
        ROOT / "audit_sources.py", ROOT / "hw5_mf_reproduction.js",
        ROOT / "hw5_mf_reproduction_experiment.js", ROOT / "run_hw5_mf_reproduction.py",
        ROOT / "export_hw5_mf_reproduction.py", ROOT / "prepare_final_report_evidence.py",
        ROOT / "verification.json", ROOT / "results.csv", ROOT / "recommendations.csv",
        ROOT / "results_four_method_corrected_mf.csv", ROOT / "results_hw5_mf_reproduction.csv",
        ROOT / "recommendations_hw5_mf_reproduction.csv",
        MF_EVIDENCE / "model_weights.json", MF_EVIDENCE / "initial_weights.json",
        MF_EVIDENCE / "model_metadata.json", MF_EVIDENCE / "summary.json",
        MF_EVIDENCE / "verification.json", MF_EVIDENCE / "restoration_verification.json",
        MF_EVIDENCE / "two_users.json", MF_EVIDENCE / "training_history.json",
        MF_EVIDENCE / "test_evaluation.json", MF_EVIDENCE / "validation_evaluation.json",
        MF_EVIDENCE / "manifest.json", MF_EVIDENCE / "preservation_verification.json",
        EVIDENCE / "bundle.json", EVIDENCE / "users.json", EVIDENCE / "source-integrity.json",
        EVIDENCE / "export-validation.json", EVIDENCE / "backend-equivalence.json",
        EVIDENCE / "mf-checkpoint.json", EVIDENCE / "selection.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Missing required evidence files: " + ", ".join(missing))

    table = read_csv(ROOT / "results_four_method_corrected_mf.csv")
    methods = {r["method"] for r in table}
    assert len(table) == 4 and methods == {"popularity", "pagerank", "ppr", "mf_hw5_reproduction"}
    assert {r["split"] for r in table} == {"test"}
    assert {r["cohort_size"] for r in table} == {"861"}
    assert {r["candidate_count"] for r in table} == {"1611"}
    assert len({r["test_cohort_sha256"] for r in table}) == 1
    mf_row = next(r for r in table if r["method"] == "mf_hw5_reproduction")
    assert mf_row["selected_mf_epoch"] == "19"
    assert mf_row["model_artifact_sha256"] == json.loads((MF_EVIDENCE / "restoration_verification.json").read_text())["restoredFrom"]["sha256"]

    # Final required results.csv is staged from the corrected four-method table;
    # the root results.csv is left intact as historical HW6-reference evidence.
    STAGE.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "results_four_method_corrected_mf.csv", STAGE / "results.csv")

    old_recs = read_csv(ROOT / "recommendations.csv")
    mf_recs = read_csv(ROOT / "recommendations_hw5_mf_reproduction.csv")
    assert len(mf_recs) == 20 and {r["user_id"] for r in mf_recs} == {"1", "2"}
    assert {r["selected_mf_epoch"] for r in mf_recs} == {"19"}
    # Replace only the old supplied-reference MF rows in the staged copy.
    final_recs = [r for r in old_recs if r["method"] != "mf"]
    for row in mf_recs:
        final_recs.append({
            "user_id": row["user_id"], "method": "mf_hw5_reproduction", "alpha": "",
            "rank": row["rank"], "movie_id": row["movie_id"], "title": row["title"],
            "genres": row["genres"], "score": row["score"],
            "training_popularity": row["training_popularity"],
            "setting": "final_selected", "mean_list_popularity": row["mean_list_popularity"],
            "split": row["split"],
        })
    fields = ["user_id", "method", "alpha", "rank", "movie_id", "title", "genres", "score",
              "training_popularity", "setting", "mean_list_popularity", "split"]
    final_recs.sort(key=lambda r: (r["setting"] != "final_selected", r["method"], int(r["user_id"]),
                                   float(r["alpha"] or 0), int(r["rank"])))
    write_csv(STAGE / "recommendations.csv", final_recs, fields)
    assert len(final_recs) == 110
    assert sum(r["setting"] == "final_selected" and r["method"] == "mf_hw5_reproduction" for r in final_recs) == 20
    assert sum(r["setting"] == "final_selected" for r in final_recs) == 80

    ppr_demo = [r for r in final_recs if r["method"] == "ppr" and r["setting"] in
                {"baseline_A_085", "seed_only_B_085", "alpha_only_A_050"}]
    assert len(ppr_demo) == 30
    assert {(r["setting"], r["user_id"], r["alpha"]) for r in ppr_demo} == {
        ("baseline_A_085", "1", "0.85"),
        ("seed_only_B_085", "2", "0.85"),
        ("alpha_only_A_050", "1", "0.5"),
    }
    assert all(r["title"] and r["genres"] and r["training_popularity"] != "" for r in final_recs)

    two = json.loads((MF_EVIDENCE / "two_users.json").read_text())
    assert two["users"] == [1, 2]
    assert len(two["controlledPprUnchanged"]) == 3
    checks = {
        "required_files_present": len(required),
        "four_method_test_rows": len(table),
        "common_test_cohort_size": 861,
        "candidate_pool_size": 1611,
        "test_cohort_hashes_identical": True,
        "corrected_mf_selected_epoch": 19,
        "staged_results_csv_rows": len(read_csv(STAGE / "results.csv")),
        "staged_recommendations_rows": len(read_csv(STAGE / "recommendations.csv")),
        "staged_final_selected_rows": 80,
        "staged_corrected_mf_rows": 20,
        "ppr_controlled_demo_rows": len(ppr_demo),
        "fixed_user_ids": [1, 2],
        "movie_metadata_and_popularity_nonempty": True,
        "report_pdf_present": (ROOT / "report.pdf").exists(),
        "week6_ai_session_export_present": any(ROOT.glob("*session*")),
        "original_exports_preserved": True,
        "no_model_retraining": True,
    }
    (STAGE / "staging_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (STAGE / "README.md").write_text(
        "# HW6 final submission staging\n\n"
        "These files are staged separately from the historical root exports. `results.csv` "
        "is the verified four-method test comparison with the corrected HW5 MF reproduction "
        "(selected epoch 19). `recommendations.csv` contains the three unchanged HW6 method "
        "lists, the three controlled PPR demonstration settings, and the corrected-HW5 MF "
        "lists for fixed users 1 and 2. The old supplied-HW6-reference MF rows are omitted "
        "from this submission copy; the original root files remain unchanged.\n\n"
        "The reproduction is a new corrected-HW5 run because the original selected Week 5 "
        "checkpoint was unavailable; numerical identity with that original model is not "
        "claimed, and no instructor approval is documented. See `../README_hw5_mf_reproduction.md` "
        "and archived provenance/evidence for details. This staging directory intentionally "
        "does not contain the report PDF or an AI-session export. Confirm the current course "
        "requirements and submission location with the instructor before submitting.\n"
    )
    return checks


def make_zip():
    temporary_zip = ZIP_PATH.with_suffix(".zip.tmp")
    if temporary_zip.exists():
        raise FileExistsError(f"Refusing to overwrite temporary archive: {temporary_zip}")
    explicit = [
        "readme.md", "README.md", "README_hw5_mf_reproduction.md", "index.html", "script.js",
        "graph.js", "experiment.js", "data.js", "style.css", "vendor/d3.min.js", "u.data", "u.item",
        "run_experiment.py", "export_evidence.py", "audit_sources.py", "hw5_mf_reproduction.js",
        "hw5_mf_reproduction_experiment.js", "run_hw5_mf_reproduction.py",
        "export_hw5_mf_reproduction.py", "prepare_final_report_evidence.py", "verification.json",
        "results.csv", "recommendations.csv", "results_four_method_corrected_mf.csv",
        "results_hw5_mf_reproduction.csv", "recommendations_hw5_mf_reproduction.csv",
        "final_submission_staging/results.csv", "final_submission_staging/recommendations.csv",
        "final_submission_staging/staging_verification.json",
        "final_submission_staging/README.md",
        "evidence/bundle.json", "evidence/users.json", "evidence/summary.json",
        "evidence/source-integrity.json", "evidence/export-validation.json",
        "evidence/backend-equivalence.json", "evidence/selection.json", "evidence/mf-checkpoint.json",
        "evidence/original-results.csv", "evidence/original-recommendations.csv",
    ]
    files = {ROOT / name for name in explicit}
    files.update(p for p in EVIDENCE.rglob("*") if p.is_file())
    files.update(p for p in (ROOT / "final_submission_staging").iterdir() if p.is_file())
    absent = [p for p in files if not p.is_file()]
    if absent:
        raise RuntimeError("Cannot package missing paths: " + ", ".join(map(str, absent)))
    with zipfile.ZipFile(temporary_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(temporary_zip) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError("ZIP CRC test failed at " + bad)
        members = z.namelist()
        assert "final_submission_staging/results.csv" in members
        assert "final_submission_staging/recommendations.csv" in members
        assert not any("session_export" in p.lower() for p in members)
    temporary_zip.replace(ZIP_PATH)
    return {"zip": str(ZIP_PATH), "file_count": len(files), "bytes": ZIP_PATH.stat().st_size,
            "sha256": digest(ZIP_PATH), "zip_integrity": "PASS"}


if __name__ == "__main__":
    print(json.dumps({"staging": verify_and_stage(), "archive": make_zip()}, indent=2))
