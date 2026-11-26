#!/usr/bin/env python3
"""Orquestador unificado del pipeline COES-SEIN en AWS Glue 4.0 (Fase B)."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

GOLD_TABLES_PARA_SERVING = (
    "despacho_central_30min",
    "precios_barra_30min",
    "gold_feature_store_30min",
    "gold_observabilidad_etiqueta_30min",
    "gold_precios_troncales_30min",
    "gold_eventos_criticos_30min",
    "gold_contexto_auditoria_30min",
    "ml_features_30min",
)


def construir_plan_etapas(
    bucket: str,
    profile: str = "distributed_managed",
    work_dir: str = "/tmp/coes_run",
    n_boot: int = 1000,
) -> list[dict]:
    """Construye el plan ordenado de las 6 etapas del pipeline completo sobre S3 + /tmp."""
    b = bucket.strip().rstrip("/")
    wd = work_dir.rstrip("/")
    s3_root = f"s3://{b}"

    return [
        {
            "orden": 1,
            "nombre": "etl_silver_gold",
            "modulo": "src.etl.pipeline",
            "tipo": "spark_s3",
            "argv": [
                "--input", f"{s3_root}/bronze",
                "--output", s3_root,
                "--profile", profile,
                "--stage", "all",
            ],
            "s3_descargas": [],
            "s3_subidas": [],
        },
        {
            "orden": 2,
            "nombre": "feature_engineering",
            "modulo": "src.ml.feature_engineering",
            "tipo": "spark_s3",
            "argv": [
                "--input", f"{s3_root}/gold/gold_feature_store_30min",
                "--output", f"{s3_root}/gold/ml_features_30min",
                "--summary-out", f"{wd}/evidencias_modelo/feature_engineering_summary.json",
            ],
            "s3_descargas": [],
            "s3_subidas": [],
        },
        {
            "orden": 3,
            "nombre": "entrenamiento_modelo",
            "modulo": "src.ml.entrenamiento_modelo",
            "tipo": "spark_s3",
            "argv": [
                "--input", f"{s3_root}/gold/ml_features_30min",
                "--output", f"{wd}/gold/ml_outputs",
                "--summary-out", f"{wd}/evidencias_modelo/comparativa_modelos.json",
                "--n-boot", str(n_boot),
            ],
            "s3_descargas": [],
            "s3_subidas": [],
        },
        {
            "orden": 4,
            "nombre": "explicabilidad_y_costos",
            "modulo": "src.ml.explicabilidad_y_costos",
            "tipo": "local_pandas",
            "argv": [
                "--features-input", f"{wd}/gold/ml_features_30min",
                "--feature-store-input", f"{wd}/gold/gold_feature_store_30min",
                "--preds-input", f"{wd}/gold/ml_outputs/predicciones_test_2024.parquet",
                "--output-parquet", f"{wd}/gold/ml_outputs/predicciones_alerta_30min_shap.parquet",
                "--summary-out", f"{wd}/evidencias_modelo/explicabilidad_y_costos.json",
                "--fig-dir", f"{wd}/evidencias_modelo/shap",
            ],
            "s3_descargas": [
                (f"{s3_root}/gold/ml_features_30min", f"{wd}/gold/ml_features_30min"),
                (f"{s3_root}/gold/gold_feature_store_30min", f"{wd}/gold/gold_feature_store_30min"),
            ],
            "s3_subidas": [],
        },
        {
            "orden": 5,
            "nombre": "backtest_economico",
            "modulo": "src.ml.backtest_economico",
            "tipo": "local_pandas",
            "argv": [
                "--features-input", f"{wd}/gold/ml_features_30min",
                "--feature-store-input", f"{wd}/gold/gold_feature_store_30min",
                "--preds-shap-input", f"{wd}/gold/ml_outputs/predicciones_alerta_30min_shap.parquet",
                "--comparativa-json", f"{wd}/evidencias_modelo/comparativa_modelos.json",
                "--explicabilidad-json", f"{wd}/evidencias_modelo/explicabilidad_y_costos.json",
                "--output-dir", f"{wd}/gold/ml_outputs",
                "--summary-out", f"{wd}/evidencias_modelo/backtest_economico_2024.json",
            ],
            "s3_descargas": [],
            "s3_subidas": [
                (f"{wd}/gold/ml_outputs", f"{s3_root}/gold/ml_outputs"),
            ],
        },
        {
            "orden": 6,
            "nombre": "build_star_schema",
            "modulo": "src.serving.build_star_schema",
            "tipo": "local_pyarrow",
            "argv": [
                "--gold-input", f"{wd}/gold",
                "--serving-output", f"{wd}/serving/star_schema",
                "--summary-out", f"{wd}/evidencias_serving/star_schema_summary.json",
            ],
            "s3_descargas": [
                (f"{s3_root}/gold/{t}", f"{wd}/gold/{t}")
                for t in GOLD_TABLES_PARA_SERVING
                if t not in ("ml_features_30min", "gold_feature_store_30min")
            ],
            "s3_subidas": [
                (f"{wd}/serving/star_schema", f"{s3_root}/serving/star_schema"),
                (f"{wd}/evidencias_modelo", f"{s3_root}/evidencias_run"),
                (f"{wd}/evidencias_serving", f"{s3_root}/evidencias_run"),
            ],
        },
    ]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    sin_prefijo = uri[5:]
    bucket, _, prefix = sin_prefijo.partition("/")
    return bucket, prefix.strip("/")


def _descargar_prefijo_s3(s3_uri: str, local_dir: Path) -> None:
    import boto3

    if local_dir.exists() and any(local_dir.rglob("*.parquet")):
        return
    bucket, prefix = _parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) + 1 :]
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))


def _subir_directorio_s3(local_dir: Path, s3_uri: str) -> None:
    import boto3

    if not local_dir.exists():
        return
    bucket, prefix = _parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            client.upload_file(str(p), bucket, key)


def _copiar_log_gold_spark_a_evidencias(bucket: str, work_dir: Path) -> None:
    """Normaliza logs/run_1x_gold.json.d generado por Spark hacia evidencias_run/run_1x_gold.json."""
    import boto3

    client = boto3.client("s3")
    prefix = "logs/run_1x_gold.json.d/"
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    partes = [
        obj["Key"]
        for obj in resp.get("Contents", [])
        if Path(obj["Key"]).name.startswith("part-")
    ]
    if not partes:
        return
    destino = work_dir / "evidencias_serving" / "run_1x_gold.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, sorted(partes)[0], str(destino))


def _asegurar_pythonpath() -> None:
    """Garantiza que src/ sea importable tanto localmente como desde --extra-py-files en Glue."""
    if any((Path(p) / "src").exists() for p in sys.path if p):
        return
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "src").exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _ejecutar_modulo(module_name: str, argv: list[str]) -> None:
    _asegurar_pythonpath()
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name, *argv]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    # Compatibilidad con argumentos --CLAVE valor inyectados por AWS Glue
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--") and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            os.environ.setdefault(arg[2:], sys.argv[i + 1])

    parser = argparse.ArgumentParser(description="AWS Glue 4.0 Runner COES-SEIN")
    parser.add_argument("--BUCKET", "--bucket", dest="bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--PROFILE", "--profile", dest="profile", default=os.environ.get("PROFILE", "distributed_managed"))
    parser.add_argument("--work-dir", default="/tmp/coes_run")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args()

    if not args.bucket:
        raise ValueError("Debe proporcionar --BUCKET <nombre-bucket>.")

    plan = construir_plan_etapas(
        bucket=args.bucket,
        profile=args.profile,
        work_dir=args.work_dir,
        n_boot=args.n_boot,
    )

    if args.dry_run:
        print(json.dumps({"bucket": args.bucket, "profile": args.profile, "etapas": plan}, indent=2))
        return

    # Evitar que spark.stop() en etapas Spark 1-3 cierre el SparkContext unico de Glue;
    # al terminar la ultima etapa Spark (3) se detiene limpiamente para liberar memoria a etapas 4-6.
    from pyspark.sql import SparkSession

    orig_stop = SparkSession.stop
    SparkSession.stop = lambda self: None  # type: ignore[method-assign]
    spark_detenido = False
    try:
        for etapa in plan:
            print(f"=== [GLUE RUNNER] Etapa {etapa['orden']}/6: {etapa['nombre']} ===")
            for s3_uri, local_dir in etapa["s3_descargas"]:
                _descargar_prefijo_s3(s3_uri, Path(local_dir))
            _ejecutar_modulo(etapa["modulo"], etapa["argv"])
            if etapa["nombre"] == "etl_silver_gold":
                _copiar_log_gold_spark_a_evidencias(args.bucket, Path(args.work_dir))
            if etapa["nombre"] == "entrenamiento_modelo" and not spark_detenido:
                SparkSession.stop = orig_stop  # type: ignore[method-assign]
                active_spark = SparkSession.getActiveSession()
                if active_spark is not None:
                    active_spark.stop()
                spark_detenido = True
            for local_dir, s3_uri in etapa["s3_subidas"]:
                _subir_directorio_s3(Path(local_dir), s3_uri)
    finally:
        SparkSession.stop = orig_stop  # type: ignore[method-assign]
        if not spark_detenido:
            active_spark = SparkSession.getActiveSession()
            if active_spark is not None:
                active_spark.stop()


if __name__ == "__main__":
    main()
