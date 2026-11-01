"""Move an owned, incomplete Step 5 attempt aside without deleting evidence."""

from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("attempt", type=int, choices=(1, 2, 3, 4, 5), default=5)
args = parser.parse_args()
source = ROOT / "results" / "step5"
target = ROOT / "results" / f"step5_failed_{args.attempt:03d}"
if not source.is_dir():
    raise FileNotFoundError(source)
if target.exists():
    raise FileExistsError(target)
os.replace(source, target)
print(f"Preserved incomplete Step 5 output at {target.relative_to(ROOT)}")
