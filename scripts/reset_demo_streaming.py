#!/usr/bin/env python3
"""Utilidad de reinicio limpio para la demo de Structured Streaming (Paso L.5)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_INBOX_DIR = Path("lake_v2/streaming/inbox")
DEFAULT_CHECKPOINT_DIR = Path("lake_v2/streaming/checkpoint")
DEFAULT_OUTPUT_DIR = Path("lake_v2/serving/streaming_alerts")
ALLOWED_S3_PREFIX = "serving/streaming_alerts"


def _limpiar_directorio_local(path: Path, recrear: bool = True) -> int:
    if not path.exists():
        if recrear:
            path.mkdir(parents=True, exist_ok=True)
        return 0
    eliminados = sum(1 for p in path.rglob("*") if p.is_file())
    shutil.rmtree(path)
    if recrear:
        path.mkdir(parents=True, exist_ok=True)
    return eliminados


def _limpiar_prefijo_s3(bucket: str, s3_prefix: str) -> int:
    prefix_norm = s3_prefix.strip("/")
    if prefix_norm != ALLOWED_S3_PREFIX:
        raise ValueError(
            f"Prefijo S3 no permitido '{s3_prefix}'. Por seguridad solo se permite '{ALLOWED_S3_PREFIX}'."
        )

    import boto3

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    claves = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix_norm}/"):
        for obj in page.get("Contents", []):
            claves.append({"Key": obj["Key"]})

    total = len(claves)
    for idx in range(0, total, 1000):
        lote = claves[idx : idx + 1000]
        client.delete_objects(Bucket=bucket, Delete={"Objects": lote})
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Reinicio limpio del pipeline de streaming")
    parser.add_argument("--inbox-dir", type=Path, default=DEFAULT_INBOX_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-s3", action="store_true")
    parser.add_argument("--bucket", default="coes-sein-bigdata-datalake")
    parser.add_argument("--s3-prefix", default=ALLOWED_S3_PREFIX)
    args = parser.parse_args()

    n_inbox = _limpiar_directorio_local(args.inbox_dir, recrear=True)
    n_ckpt = _limpiar_directorio_local(args.checkpoint_dir, recrear=False)
    n_out = _limpiar_directorio_local(args.output_dir, recrear=True)
    print(
        f"[RESET LOCAL] inbox={n_inbox} archivos | checkpoint={n_ckpt} archivos | "
        f"streaming_alerts={n_out} archivos"
    )

    if args.include_s3:
        n_s3 = _limpiar_prefijo_s3(bucket=args.bucket, s3_prefix=args.s3_prefix)
        print(f"[RESET S3] eliminados={n_s3} objetos en s3://{args.bucket}/{args.s3_prefix.strip('/')}/")

    print("[OK] Reinicio limpio completado.")


if __name__ == "__main__":
    main()
