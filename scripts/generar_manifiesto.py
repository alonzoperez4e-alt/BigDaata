import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(bronze_dir: Path, out_file: Path) -> None:
    entries = []
    for p in sorted(bronze_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            entries.append({
                "relative_path": p.relative_to(bronze_dir).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/manifests/bronze_sha256.json"), type=Path)
    args = parser.parse_args()
    build_manifest(args.bronze, args.out)
    print(f"Manifiesto generado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
