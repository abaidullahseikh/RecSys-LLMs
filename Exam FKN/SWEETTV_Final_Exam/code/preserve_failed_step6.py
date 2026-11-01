"""Preserve an incomplete owned Step 6 output before a corrected rerun."""

import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("attempt",type=int,default=4)
args=parser.parse_args()
src=ROOT/"results"/"step6"
dst=ROOT/"results"/f"step6_failed_{args.attempt:03d}"
if not src.is_dir(): raise FileNotFoundError(src)
if dst.exists(): raise FileExistsError(dst)
os.replace(src,dst)
print(dst.relative_to(ROOT))
