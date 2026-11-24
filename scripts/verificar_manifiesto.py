import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: No existe el manifiesto {args.manifest}", file=sys.stderr)
        return 1

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in data.get("files", []):
        target = args.bronze / item["relative_path"]
        if not target.exists():
            print(f"ERROR: Falta archivo en Bronze: {target}", file=sys.stderr)
            return 1
        actual_hash = sha256_file(target)
        if actual_hash != item["sha256"]:
            print(f"ERROR: Hash alterado en {target}", file=sys.stderr)
            return 1

    print(f"OK: {len(data.get('files', []))} archivos verificados con SHA-256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
