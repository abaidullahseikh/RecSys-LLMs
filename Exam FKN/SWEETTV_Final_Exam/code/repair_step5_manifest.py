"""Archive and repair Step 5 artifact manifest using unique relative paths."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step5"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest_path = OUT / "ppr_artifacts.json"
    archived = OUT / "ppr_artifacts_attempt_001.json"
    if archived.exists():
        raise FileExistsError(archived)
    archived.write_bytes(manifest_path.read_bytes())
    files = {str(path.relative_to(OUT)): {"bytes":path.stat().st_size,"sha256":sha256(path)}
             for path in sorted(OUT.rglob("*")) if path.is_file()
             and path not in (manifest_path, archived)}
    manifest_path.write_text(json.dumps(files,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"old_manifest_archived":str(archived.relative_to(ROOT)),
                      "new_manifest_sha256":sha256(manifest_path),"tracked_files":len(files)},indent=2))


if __name__=="__main__":
    main()
