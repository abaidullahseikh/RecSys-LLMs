"""Remove duplicate export columns from the saved Step 5 experiment table."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step5"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    path = OUT / "ppr_experiments.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    unique = list(dict.fromkeys(header))
    if len(unique) == len(header):
        print("Experiment CSV schema already unique; unchanged.")
        return
    first_index = {name: header.index(name) for name in unique}
    normalized = [[row[first_index[name]] for name in unique] for row in rows]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(unique)
        writer.writerows(normalized)
    temporary.replace(path)
    manifest_path = OUT / "ppr_artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print({"path": str(path.relative_to(ROOT)), "old_columns": len(header),
           "unique_columns": len(unique), "rows": len(rows), "sha256": digest(path)})


if __name__ == "__main__":
    main()
