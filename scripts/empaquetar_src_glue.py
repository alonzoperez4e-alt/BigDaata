#!/usr/bin/env python3
"""
Empaquetador de src/ para despliegue agnóstico (p.ej. AWS Glue, Dataproc, HDInsight).

Genera:
  - dist/src.zip  : paquete Python con src/ (sin dashboard, __pycache__, *.pyc)
  - dist/build_manifest.json : SHA-256 y tamaño de artefactos clave
"""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", "dashboard"}
EXCLUDE_EXTS = {".pyc"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip(out_zip: str) -> None:
    """Empaqueta src/ manteniendo prefijo src/... dentro del zip."""
    os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
    src_root = Path("src")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(src_root):
            # Excluir directorios prohibidos
            dirnames[:] = [
                d for d in dirnames if d not in EXCLUDE_DIRS
            ]
            for fname in filenames:
                if Path(fname).suffix in EXCLUDE_EXTS:
                    continue
                full = os.path.join(dirpath, fname)
                arcname = full.replace("\\", "/")  # Windows compat
                zf.write(full, arcname)


def build_manifest(out_zip: str, manifest_out: str) -> dict:
    """Genera manifiesto con SHA-256 y tamaño de artefactos clave."""
    tracked_files = [
        out_zip,
        "src/etl/pipeline.py",
        "src/ml/feature_engineering.py",
        "src/serving/build_star_schema.py",
    ]
    manifest = {}
    for fpath in tracked_files:
        if os.path.exists(fpath):
            manifest[fpath] = {
                "sha256": sha256_file(fpath),
                "size_bytes": os.path.getsize(fpath),
            }
        else:
            manifest[fpath] = {"error": "file not found"}

    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Empaquetar src/ para despliegue")
    parser.add_argument(
        "--out-zip", default="dist/src.zip", help="Ruta del zip de salida"
    )
    parser.add_argument(
        "--manifest-out",
        default="dist/build_manifest.json",
        help="Ruta del manifiesto de build",
    )
    args = parser.parse_args()

    print(f"Empaquetando src/ → {args.out_zip} …")
    build_zip(args.out_zip)
    zip_size = os.path.getsize(args.out_zip)
    print(f"  Zip generado: {zip_size:,} bytes")

    print(f"Generando manifiesto → {args.manifest_out} …")
    manifest = build_manifest(args.out_zip, args.manifest_out)
    for k, v in manifest.items():
        if "sha256" in v:
            print(f"  {k}: {v['size_bytes']:,} B  sha256={v['sha256'][:16]}…")
        else:
            print(f"  {k}: {v.get('error', 'unknown')}")

    print("Empaquetado completado.")


if __name__ == "__main__":
    main()
