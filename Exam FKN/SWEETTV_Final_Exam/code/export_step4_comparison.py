"""Publish the verified Step 4 table at the course-requested path."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "step4" / "step4_baseline_comparison.csv"
TARGET = ROOT / "results" / "step4_baseline_comparison.csv"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish():
    rows = []
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows.extend(reader)
    if len(rows) != 5:
        raise ValueError("Expected the five previously verified baseline variants")
    with TARGET.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    if TARGET.read_bytes() != SOURCE.read_bytes():
        raise ValueError("Published comparison file differs from verified experiment table")
    print({"path": str(TARGET.relative_to(ROOT)), "sha256": sha256(TARGET), "rows": len(rows)})


if __name__ == "__main__":
    publish()
