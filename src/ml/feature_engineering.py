"""Tarea 3.1: Feature Engineering invariante al regimen y rezagos sin fuga en PySpark."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

FORBIDDEN_LEAKAGE_COLS = {
    "label_rdo",
    "alfa",
    "beta",
    "gamma",
    "chm",
    "alfa_primer_archivo",
    "beta_primer_archivo",
    "gamma_primer_archivo",
}

NON_PREDICTOR_RETROACTIVE_COLS = {
    "energia_nop_mwh",
    "exposicion_bruta_soles",
    "sobrecosto_neto_estimado_soles",
    "n_centrales_term_nop",
}

PREDICTOR_COLS = [
    # Familia A: Desviacion y regimen termico
    "flag_a1_disponible_t",
    "brecha_term_cara_mw",
    "ratio_term_ejec_vs_prog",
    "term_cara_share",
    "gen_total_mw",
    "rampa_brecha_2h",
    "rampa_gen_30m",
    "rampa_gen_2h",
    # Familia B: Estres nodal A4 220/500 kV
    "cmg_troncal_mediana_t",
    "cmg_troncal_p90_t",
    "cmg_spread_p90_p50_t",
    "rampa_cmg_1h",
    "rampa_cmg_4h",
    "cmg_zscore_24h",
    # Familia C: Confiabilidad B1
    "n_forzados_termicos_activos",
    "n_forzados_transmision_activos",
    "n_forzados_termicos_inicio_3h",
    "n_mantenimientos_prog_termicos_activos",
    # Familia D: Memoria y estacionalidad ciclica
    "persistencia_term_nop_2h",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "es_fin_de_semana",
]

assert not (set(PREDICTOR_COLS) & FORBIDDEN_LEAKAGE_COLS)
assert not (set(PREDICTOR_COLS) & NON_PREDICTOR_RETROACTIVE_COLS)

OUTPUT_METADATA_COLS = [
    "fecha_hora",
    "fecha_despacho",
    "anio",
    "tiene_a1",
    "tiene_a2",
    "tiene_a3",
    "tiene_pr06",
    "etiqueta_observable",
    "motivo_censura",
    "evento_termico_nop",
    "inicio_termico_nop",
]

TARGET_WINDOW_COLS = [
    "n_obs_ventana_2_4h",
    "y_ventana_2_4h",
    "y_inicio_ventana_2_4h",
]

_LAG_SPECS = (
    (1800, [("gen_total_mw", "_lag_gen_30m")]),
    (3600, [("cmg_troncal_mediana", "_lag_cmg_1h")]),
    (7200, [("brecha_despacho_mw", "_lag_brecha_2h"), ("gen_total_mw", "_lag_gen_2h")]),
    (14400, [("cmg_troncal_mediana", "_lag_cmg_4h")]),
)


def crear_spark(app_name: str = "coes-feature-engineering") -> SparkSession:
    """Inicializa una SparkSession en UTC con configuracion local segura."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def validar_entrada_sin_fuga(df: DataFrame) -> None:
    """Valida que no haya columnas prohibidas del PR-06 ni duplicados en fecha_hora."""
    fuga = sorted(FORBIDDEN_LEAKAGE_COLS.intersection(df.columns))
    if fuga:
        raise ValueError(f"Columnas prohibidas detectadas en la entrada: {fuga}")

    if "fecha_hora" not in df.columns:
        raise ValueError("La entrada debe contener la columna 'fecha_hora'.")

    if df.filter(F.col("fecha_hora").isNull()).limit(1).count() > 0:
        raise ValueError("La entrada contiene valores nulos en 'fecha_hora'.")

    duplicados = df.groupBy("fecha_hora").count().filter(F.col("count") > 1).limit(1).count()
    if duplicados > 0:
        raise ValueError("Se detectaron marcas temporales duplicadas en 'fecha_hora'.")


def _agregar_rezagos_puntuales(df: DataFrame) -> DataFrame:
    """Calcula rezagos exactos mediante left self-join sobre fecha_hora (sin F.first)."""
    out = df.withColumn("_ts_epoch", F.col("fecha_hora").cast("long"))
    base_ts = out.select(
        "_ts_epoch",
        "gen_total_mw",
        "brecha_despacho_mw",
        "cmg_troncal_mediana",
    )

    for offset_s, col_pairs in _LAG_SPECS:
        right = base_ts.select(
            (F.col("_ts_epoch") + F.lit(offset_s)).alias("_ts_epoch"),
            *[F.col(src).alias(dst) for src, dst in col_pairs],
        )
        out = out.join(right, on="_ts_epoch", how="left")

    return (
        out.withColumn("rampa_gen_30m", F.col("gen_total_mw") - F.col("_lag_gen_30m"))
        .withColumn("rampa_gen_2h", F.col("gen_total_mw") - F.col("_lag_gen_2h"))
        .withColumn("rampa_brecha_2h", F.col("brecha_despacho_mw") - F.col("_lag_brecha_2h"))
        .withColumn("rampa_cmg_1h", F.col("cmg_troncal_mediana") - F.col("_lag_cmg_1h"))
        .withColumn("rampa_cmg_4h", F.col("cmg_troncal_mediana") - F.col("_lag_cmg_4h"))
        .drop("_lag_gen_30m", "_lag_gen_2h", "_lag_brecha_2h", "_lag_cmg_1h", "_lag_cmg_4h")
    )


def _etiqueta_ventana_censurada(max_col, n_obs_col):
    """Aplica la regla de censura estricta sobre una ventana futura [t+2h, t+4h]."""
    return (
        F.when(max_col == F.lit(1), F.lit(1))
        .when(n_obs_col.isNull() | (n_obs_col < F.lit(3)), F.lit(None).cast("int"))
        .otherwise(F.lit(0))
    )


def construir_features_ml(df: DataFrame) -> DataFrame:
    """Construye Familias A-D sin fuga futura y objetivos censurados [t+2h, t+4h]."""
    validar_entrada_sin_fuga(df)

    base = _agregar_rezagos_puntuales(df)

    # Ventanas continuas ordenadas por tiempo UNIX (SIN partitionBy("anio"))
    w_base = Window.orderBy(F.col("_ts_epoch"))
    w_24h = w_base.rangeBetween(-86400, 0)
    w_2h_pasado = w_base.rangeBetween(-7200, 0)
    w_fut_2_4h = w_base.rangeBetween(7200, 14400)

    cmg_mean_24h = F.avg("cmg_troncal_mediana").over(w_24h)
    cmg_std_24h = F.stddev("cmg_troncal_mediana").over(w_24h)

    ev_obs = F.when(F.col("etiqueta_observable") == F.lit(True), F.col("evento_termico_nop").cast("int"))
    ini_obs = F.when(F.col("etiqueta_observable") == F.lit(True), F.col("inicio_termico_nop").cast("int"))

    persistencia_raw = F.sum(ev_obs).over(w_2h_pasado).cast("int")
    n_obs_fut = F.count(ev_obs).over(w_fut_2_4h).cast("int")
    max_ev_fut = F.max(ev_obs).over(w_fut_2_4h)
    max_ini_fut = F.max(ini_obs).over(w_fut_2_4h)

    hora_decimal = (
        F.hour("fecha_hora")
        + (F.minute("fecha_hora") / F.lit(60.0))
        + (F.second("fecha_hora") / F.lit(3600.0))
    )
    mes_num = F.month("fecha_despacho")

    tiene_a1_true = F.col("tiene_a1") == F.lit(True)

    feat_df = (
        base.withColumn("_cmg_mean_24h", cmg_mean_24h)
        .withColumn("_cmg_std_24h", cmg_std_24h)
        # Familia A: Desviacion y regimen termico
        .withColumn("flag_a1_disponible_t", F.coalesce(F.col("tiene_a1").cast("int"), F.lit(0)))
        .withColumn(
            "brecha_term_cara_mw",
            F.when(tiene_a1_true, F.col("term_cara_ejec_mw") - F.col("term_cara_prog_mw")).otherwise(
                F.lit(None).cast("double")
            ),
        )
        .withColumn(
            "ratio_term_ejec_vs_prog",
            F.when(
                tiene_a1_true
                & F.col("term_cara_prog_mw").isNotNull()
                & ((F.col("term_cara_prog_mw") + F.lit(1.0)) != F.lit(0.0)),
                F.col("term_cara_ejec_mw") / (F.col("term_cara_prog_mw") + F.lit(1.0)),
            ).otherwise(F.lit(None).cast("double")),
        )
        # Familia B: Estres nodal A4 220/500 kV
        .withColumn("cmg_troncal_mediana_t", F.col("cmg_troncal_mediana").cast("double"))
        .withColumn("cmg_troncal_p90_t", F.col("cmg_troncal_p90").cast("double"))
        .withColumn("cmg_spread_p90_p50_t", F.col("cmg_spread_p90_p50").cast("double"))
        .withColumn(
            "cmg_zscore_24h",
            F.when(
                F.col("cmg_troncal_mediana").isNotNull()
                & F.col("_cmg_std_24h").isNotNull()
                & (F.col("_cmg_std_24h") > F.lit(1e-6)),
                (F.col("cmg_troncal_mediana") - F.col("_cmg_mean_24h")) / F.col("_cmg_std_24h"),
            ).otherwise(F.lit(None).cast("double")),
        )
        # Familia D: Memoria y estacionalidad ciclica
        .withColumn(
            "persistencia_term_nop_2h",
            F.when(F.col("etiqueta_observable") == F.lit(True), F.coalesce(persistencia_raw, F.lit(0))).otherwise(
                F.lit(None).cast("int")
            ),
        )
        .withColumn("hora_sin", F.sin(F.lit(2.0 * math.pi) * hora_decimal / F.lit(24.0)))
        .withColumn("hora_cos", F.cos(F.lit(2.0 * math.pi) * hora_decimal / F.lit(24.0)))
        .withColumn("mes_sin", F.sin(F.lit(2.0 * math.pi) * mes_num / F.lit(12.0)))
        .withColumn("mes_cos", F.cos(F.lit(2.0 * math.pi) * mes_num / F.lit(12.0)))
        .withColumn("es_fin_de_semana", F.dayofweek("fecha_despacho").isin(1, 7).cast("int"))
        # Horizonte futuro [t+4, t+8] (2 a 4 horas)
        .withColumn("n_obs_ventana_2_4h", n_obs_fut)
        .withColumn("_max_ev_fut", max_ev_fut)
        .withColumn("_max_ini_fut", max_ini_fut)
        .withColumn("y_ventana_2_4h", _etiqueta_ventana_censurada(F.col("_max_ev_fut"), F.col("n_obs_ventana_2_4h")))
        .withColumn(
            "y_inicio_ventana_2_4h",
            _etiqueta_ventana_censurada(F.col("_max_ini_fut"), F.col("n_obs_ventana_2_4h")),
        )
        .drop("_ts_epoch", "_cmg_mean_24h", "_cmg_std_24h", "_max_ev_fut", "_max_ini_fut")
    )

    cols_salida = [*OUTPUT_METADATA_COLS, *PREDICTOR_COLS, *TARGET_WINDOW_COLS]
    # Garantizar ausencia absoluta de columnas prohibidas o retroactivas en la salida
    cols_prohibidas = set(cols_salida).intersection(FORBIDDEN_LEAKAGE_COLS | NON_PREDICTOR_RETROACTIVE_COLS)
    assert not cols_prohibidas, f"Columnas prohibidas/retroactivas en salida: {sorted(cols_prohibidas)}"
    return feat_df.select(*cols_salida)


def generar_resumen(df_out: DataFrame) -> dict:
    """Calcula metricas de auditoria y conteos de nulos sobre la tabla generada."""
    agg = df_out.agg(
        F.count("*").alias("total_bloques"),
        F.sum(F.when(F.col("y_ventana_2_4h").isNotNull(), 1).otherwise(0)).alias("n_etiquetables_ventana"),
        F.sum(F.when(F.col("y_inicio_ventana_2_4h").isNotNull(), 1).otherwise(0)).alias("n_etiquetables_inicio"),
        F.sum(F.when(F.col("y_ventana_2_4h") == 1, 1).otherwise(0)).alias("positivos_evento_2_4h"),
        F.sum(F.when(F.col("y_inicio_ventana_2_4h") == 1, 1).otherwise(0)).alias("positivos_inicio_2_4h"),
        F.sum(F.when(F.col("brecha_term_cara_mw").isNull(), 1).otherwise(0)).alias("nulos_brecha_term_cara_mw"),
        F.sum(F.when(F.col("rampa_cmg_1h").isNull(), 1).otherwise(0)).alias("nulos_rampa_cmg_1h"),
        F.sum(F.when(F.col("cmg_zscore_24h").isNull(), 1).otherwise(0)).alias("nulos_cmg_zscore_24h"),
    ).first()

    n_et = int(agg["n_etiquetables_ventana"] or 0)
    n_et_ini = int(agg["n_etiquetables_inicio"] or 0)
    pos_ev = int(agg["positivos_evento_2_4h"] or 0)
    pos_ini = int(agg["positivos_inicio_2_4h"] or 0)

    cols_prohibidas = sorted(
        set(df_out.columns).intersection(FORBIDDEN_LEAKAGE_COLS | NON_PREDICTOR_RETROACTIVE_COLS)
    )

    return {
        "total_bloques": int(agg["total_bloques"] or 0),
        "columnas_salida": list(df_out.columns),
        "columnas_prohibidas_presentes": cols_prohibidas,
        "n_etiquetables_ventana": n_et,
        "positivos_evento_2_4h": pos_ev,
        "positivos_inicio_2_4h": pos_ini,
        "prevalencia_y_ventana_2_4h": round(pos_ev / n_et, 6) if n_et > 0 else None,
        "prevalencia_y_inicio_ventana_2_4h": round(pos_ini / n_et_ini, 6) if n_et_ini > 0 else None,
        "nulos": {
            "brecha_term_cara_mw": int(agg["nulos_brecha_term_cara_mw"] or 0),
            "rampa_cmg_1h": int(agg["nulos_rampa_cmg_1h"] or 0),
            "cmg_zscore_24h": int(agg["nulos_cmg_zscore_24h"] or 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Engineering PySpark (Tarea 3.1)")
    parser.add_argument("--input", required=True, help="Ruta a gold/gold_feature_store_30min")
    parser.add_argument("--output", required=True, help="Ruta de salida para gold/ml_features_30min")
    parser.add_argument("--summary-out", default=None, help="Ruta opcional para JSON de resumen")
    args = parser.parse_args()

    spark = crear_spark()
    try:
        df_in = spark.read.parquet(args.input)
        df_feat = construir_features_ml(df_in)

        (
            df_feat.repartition(1, "anio")
            .sortWithinPartitions("fecha_hora")
            .write.mode("overwrite")
            .option("compression", "snappy")
            .partitionBy("anio")
            .parquet(args.output)
        )

        df_check = spark.read.parquet(args.output)
        resumen = generar_resumen(df_check)
        if args.summary_out:
            summary_path = Path(args.summary_out)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(resumen, indent=2), encoding="utf-8")

        print(json.dumps(resumen, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
