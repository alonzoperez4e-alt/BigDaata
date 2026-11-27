#!/usr/bin/env python3
"""Motor de inferencia en tiempo real con PySpark Structured Streaming (Paso L.2)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F, types as T

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INBOX_DIR = Path("lake_v2/streaming/inbox")
DEFAULT_MODEL_PATH = Path("lake_v2/gold/ml_outputs/modelo_ganador_y_ventana_2_4h")
DEFAULT_SCHEMA_SOURCE = Path("lake_v2/gold/ml_features_30min/anio=2024")
DEFAULT_OUTPUT_DIR = Path("lake_v2/serving/streaming_alerts")
DEFAULT_CHECKPOINT_DIR = Path("lake_v2/streaming/checkpoint")
DEFAULT_COMPARATIVA_JSON = Path("evidencias_modelo/comparativa_modelos.json")
MODELO_VERSION = "rf_v1_2024_walkforward"
DEFAULT_UMBRAL_FALLBACK = 0.45


def crear_spark_streaming(app_name: str = "coes-streaming-inference") -> SparkSession:
    """Inicializa SparkSession para Structured Streaming con 4 shuffle partitions."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def obtener_umbral_operativo(comparativa_json: Path = DEFAULT_COMPARATIVA_JSON) -> float:
    """Obtiene el ultimo umbral calibrado de recalibrado_60d_max_fp_3_sem."""
    if comparativa_json.is_file():
        try:
            data = json.loads(comparativa_json.read_text(encoding="utf-8"))
            traz = (
                data.get("targets", {})
                .get("y_ventana_2_4h", {})
                .get("test_final_2024", {})
                .get("trazabilidad_recalibracion_60d_maxfp3", [])
            )
            if traz:
                return float(traz[-1]["tau_mes"])
        except Exception:
            pass
    return DEFAULT_UMBRAL_FALLBACK


def esquema_explicito_fallback() -> T.StructType:
    """Esquema fijo explicito compatible con ml_features_30min/anio=2024."""
    D, I, S, B = T.DoubleType(), T.IntegerType(), T.StringType(), T.BooleanType()
    return T.StructType(
        [
            T.StructField("fecha_hora", T.TimestampType(), True),
            T.StructField("fecha_despacho", T.DateType(), True),
            T.StructField("tiene_a1", B, True),
            T.StructField("tiene_a2", B, True),
            T.StructField("tiene_a3", B, True),
            T.StructField("tiene_pr06", B, True),
            T.StructField("etiqueta_observable", B, True),
            T.StructField("motivo_censura", S, True),
            T.StructField("evento_termico_nop", I, True),
            T.StructField("inicio_termico_nop", I, True),
            T.StructField("flag_a1_disponible_t", I, True),
            T.StructField("brecha_term_cara_mw", D, True),
            T.StructField("ratio_term_ejec_vs_prog", D, True),
            T.StructField("term_cara_share", D, True),
            T.StructField("gen_total_mw", D, True),
            T.StructField("rampa_brecha_2h", D, True),
            T.StructField("rampa_gen_30m", D, True),
            T.StructField("rampa_gen_2h", D, True),
            T.StructField("cmg_troncal_mediana_t", D, True),
            T.StructField("cmg_troncal_p90_t", D, True),
            T.StructField("cmg_spread_p90_p50_t", D, True),
            T.StructField("rampa_cmg_1h", D, True),
            T.StructField("rampa_cmg_4h", D, True),
            T.StructField("cmg_zscore_24h", D, True),
            T.StructField("n_forzados_termicos_activos", I, True),
            T.StructField("n_forzados_transmision_activos", I, True),
            T.StructField("n_forzados_termicos_inicio_3h", I, True),
            T.StructField("n_mantenimientos_prog_termicos_activos", I, True),
            T.StructField("persistencia_term_nop_2h", I, True),
            T.StructField("hora_sin", D, True),
            T.StructField("hora_cos", D, True),
            T.StructField("mes_sin", D, True),
            T.StructField("mes_cos", D, True),
            T.StructField("es_fin_de_semana", I, True),
            T.StructField("n_obs_ventana_2_4h", I, True),
            T.StructField("y_ventana_2_4h", I, True),
            T.StructField("y_inicio_ventana_2_4h", I, True),
            T.StructField("origen_archivo", S, True),
        ]
    )


def resolver_esquema_fijo(spark: SparkSession, schema_source: Path) -> T.StructType:
    """Obtiene el esquema fijo desde Parquet historico o usa el esquema explicito."""
    if schema_source.exists():
        raw_schema = spark.read.parquet(str(schema_source)).schema
        fields = [T.StructField(f.name, f.dataType, True) for f in raw_schema.fields]
        if not any(f.name == "origen_archivo" for f in fields):
            fields.append(T.StructField("origen_archivo", T.StringType(), True))
        return T.StructType(fields)
    return esquema_explicito_fallback()


def _subir_archivo_s3(local_path: Path, bucket: str, s3_prefix: str) -> str:
    import boto3

    key = f"{s3_prefix.strip('/')}/{local_path.name}"
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def construir_procesador_lote(
    modelo: PipelineModel,
    umbral: float,
    output_dir: Path,
    sync_s3: bool = False,
    bucket: str = "coes-sein-bigdata-datalake",
    s3_prefix: str = "serving/streaming_alerts",
):
    """Construye la funcion foreachBatch para scoring, escritura Parquet y subida S3."""
    umbral_vig = float(umbral) * 0.80

    def _procesar_micro_lote(batch_df, batch_id: int) -> None:
        if batch_df.isEmpty():
            return

        scored = modelo.transform(batch_df)
        prob_col = vector_to_array(F.col("probability"))[1].cast("double")
        archivo_col = F.regexp_extract(F.input_file_name(), r"([^/\\]+)$", 1)
        origen_col = (
            F.coalesce(
                F.nullif(F.trim(F.col("origen_archivo").cast("string")), F.lit("")),
                archivo_col,
            )
            if "origen_archivo" in scored.columns
            else archivo_col
        )

        enriquecido = (
            scored.withColumn("prob_evento", prob_col)
            .withColumn(
                "estado_alerta",
                F.when(F.col("prob_evento") >= F.lit(float(umbral)), F.lit("ALERTA_CRITICA"))
                .when(F.col("prob_evento") >= F.lit(umbral_vig), F.lit("VIGILANCIA"))
                .otherwise(F.lit("NORMAL")),
            )
            .withColumn("batch_id", F.lit(int(batch_id)).cast("int"))
            .withColumn("ingested_at_utc", F.current_timestamp())
            .withColumn("umbral_operativo", F.lit(float(umbral)).cast("double"))
            .withColumn("modelo_version", F.lit(MODELO_VERSION))
            .withColumn("origen_archivo", origen_col)
        )

        cols_salida = [
            "fecha_hora",
            "fecha_despacho",
            "prob_evento",
            "estado_alerta",
            "umbral_operativo",
            "modelo_version",
            "batch_id",
            "ingested_at_utc",
            "origen_archivo",
            "etiqueta_observable",
            "motivo_censura",
            "flag_a1_disponible_t",
            "brecha_term_cara_mw",
            "ratio_term_ejec_vs_prog",
            "term_cara_share",
            "gen_total_mw",
            "cmg_troncal_mediana_t",
            "cmg_troncal_p90_t",
            "cmg_spread_p90_p50_t",
            "n_forzados_termicos_activos",
            "persistencia_term_nop_2h",
        ]
        cols_presentes = [c for c in cols_salida if c in enriquecido.columns]
        pdf = enriquecido.select(*cols_presentes).orderBy("fecha_hora").toPandas()

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"batch_{int(batch_id):05d}.parquet"
        pdf.to_parquet(out_file, compression="snappy", index=False)

        s3_info = ""
        if sync_s3:
            s3_uri = _subir_archivo_s3(out_file, bucket=bucket, s3_prefix=s3_prefix)
            s3_info = f" | s3={s3_uri}"

        n_bloques = len(pdf)
        max_prob = float(pdf["prob_evento"].max()) if n_bloques > 0 else 0.0
        n_crit = int((pdf["estado_alerta"] == "ALERTA_CRITICA").sum())
        n_vig = int((pdf["estado_alerta"] == "VIGILANCIA").sum())
        print(
            f"[BATCH {int(batch_id):05d}] bloques={n_bloques} | max_prob={max_prob:.4f} | "
            f"alertas_criticas={n_crit} | vigilancia={n_vig} | archivo={out_file.name}{s3_info}"
        )

    return _procesar_micro_lote


def ejecutar_streaming(
    inbox_dir: Path = DEFAULT_INBOX_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_source: Path = DEFAULT_SCHEMA_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    umbral: float | None = None,
    processing_time: str = "5 seconds",
    once: bool = False,
    sync_s3: bool = False,
    bucket: str = "coes-sein-bigdata-datalake",
    s3_prefix: str = "serving/streaming_alerts",
) -> None:
    """Ejecuta el pipeline de Structured Streaming (5 piezas explicitas)."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    umbral_activo = float(umbral) if umbral is not None else obtener_umbral_operativo()
    spark = crear_spark_streaming()
    try:
        # 1. Fuente: readStream con esquema fijo sobre inbox JSON
        esquema_fijo = resolver_esquema_fijo(spark, schema_source)
        stream_df = spark.readStream.schema(esquema_fijo).json(str(inbox_dir))

        # 2. Consulta: carga PipelineModel e inferencia por micro-lote
        modelo = PipelineModel.load(str(model_path))
        procesar_fn = construir_procesador_lote(
            modelo=modelo,
            umbral=umbral_activo,
            output_dir=output_dir,
            sync_s3=sync_s3,
            bucket=bucket,
            s3_prefix=s3_prefix,
        )

        # 3, 4 y 5. Disparador + outputMode append + checkpoint dedicado
        writer = (
            stream_df.writeStream.outputMode("append")
            .option("checkpointLocation", str(checkpoint_dir))
            .foreachBatch(procesar_fn)
        )
        if once:
            writer = writer.trigger(availableNow=True)
        else:
            writer = writer.trigger(processingTime=processing_time)

        print(
            f"[STREAMING] modelo={MODELO_VERSION} | umbral={umbral_activo:.4f} | "
            f"trigger={'availableNow' if once else processing_time} | inbox={inbox_dir}"
        )
        query = writer.start()
        query.awaitTermination()
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real PySpark Structured Streaming")
    parser.add_argument("--inbox-dir", type=Path, default=DEFAULT_INBOX_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--schema-source", type=Path, default=DEFAULT_SCHEMA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--umbral", type=float, default=None)
    parser.add_argument("--processing-time", default="5 seconds")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sync-s3", action="store_true")
    parser.add_argument("--bucket", default="coes-sein-bigdata-datalake")
    parser.add_argument("--s3-prefix", default="serving/streaming_alerts")
    args = parser.parse_args()

    ejecutar_streaming(
        inbox_dir=args.inbox_dir,
        model_path=args.model_path,
        schema_source=args.schema_source,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        umbral=args.umbral,
        processing_time=args.processing_time,
        once=args.once,
        sync_s3=args.sync_s3,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
    )


if __name__ == "__main__":
    main()
