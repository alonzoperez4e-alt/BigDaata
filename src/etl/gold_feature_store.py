"""Unificacion oficial A + B en gold/gold_feature_store_30min con linaje explicito."""
from pyspark.sql import SparkSession, functions as F

from src.etl.gold_observabilidad import (
    COMBUSTIBLES_CAROS,
    UMBRAL_POTENCIA_NOP_MW,
    construir_grilla_maestra,
)

SCRIPT_VERSION = "2.4.0"

COLUMNAS_PROHIBIDAS_ANTIFUGA = {
    "label_rdo",
    "alfa",
    "beta",
    "gamma",
    "chm",
    "alfa_primer_archivo",
    "beta_primer_archivo",
    "gamma_primer_archivo",
}

LINAJE_COLUMNAS = {
    "fecha_hora": "grilla_maestra (sequence 30min)",
    "fecha_despacho": "grilla_maestra (to_date(fecha_hora - 1 min))",
    "anio": "grilla_maestra (year(fecha_despacho))",
    "gen_total_mw": "gold/despacho_central_30min (A2 sum potencia_ejecutada_mw)",
    "prog_total_mw": "gold/despacho_central_30min (A1/A2 sum potencia_programada_mw; NULL si sin programa)",
    "brecha_despacho_mw": "derivada (gen_total_mw - prog_total_mw)",
    "term_cara_ejec_mw": "gold/despacho_central_30min (A2+A3 sum potencia_ejecutada_mw en combustibles caros)",
    "term_cara_prog_mw": "gold/despacho_central_30min (A1+A3 sum potencia_programada_mw en combustibles caros)",
    "term_cara_share": "derivada (term_cara_ejec_mw / gen_total_mw)",
    "energia_nop_mwh": "gold/despacho_central_30min (A1+A2+A3 energia termica cara no programada >5MW, solo observables)",
    "exposicion_bruta_soles": "gold/despacho_central_30min (energia_nop_mwh * 1000 * costo_variable_s_kwh, solo observables)",
    "sobrecosto_neto_estimado_soles": "derivada (exposicion_bruta_soles - energia_nop_mwh * 1000 * cmg_troncal_mediana)",
    "cmg_troncal_mediana": "gold/gold_precios_troncales_30min (A4 220/500kV P50 sobre costo_marginal_limpio)",
    "cmg_troncal_p90": "gold/gold_precios_troncales_30min (A4 220/500kV P90 sobre costo_marginal_limpio)",
    "cmg_spread_p90_p50": "gold/gold_precios_troncales_30min (P90 - P50 congestion nodal)",
    "n_barras_troncales_validas": "gold/gold_precios_troncales_30min (conteo barras 220/500kV validas)",
    "n_forzados_termicos_activos": "gold/gold_eventos_criticos_30min (B1 FORZADO termico activo en t)",
    "n_forzados_transmision_activos": "gold/gold_eventos_criticos_30min (B1 FORZADO transmision activo en t)",
    "n_forzados_termicos_inicio_3h": "gold/gold_eventos_criticos_30min (B1 FORZADO termico iniciado en [t-3h, t])",
    "n_mantenimientos_prog_termicos_activos": "gold/gold_eventos_criticos_30min (B1 PROGRAMADO termico activo en t)",
    "tiene_a1": "gold/gold_observabilidad_etiqueta_30min",
    "tiene_a2": "gold/gold_observabilidad_etiqueta_30min",
    "tiene_a3": "gold/gold_observabilidad_etiqueta_30min",
    "tiene_pr06": "gold/gold_observabilidad_etiqueta_30min",
    "etiqueta_observable": "gold/gold_observabilidad_etiqueta_30min (tiene_a1 & tiene_a2 & tiene_a3)",
    "motivo_censura": "gold/gold_observabilidad_etiqueta_30min",
    "evento_termico_nop": "gold/gold_observabilidad_etiqueta_30min (target proxy; NULL si censurado)",
    "inicio_termico_nop": "gold/gold_observabilidad_etiqueta_30min (target inicio; NULL si censurado)",
    "n_centrales_term_nop": "gold/gold_observabilidad_etiqueta_30min",
}


def construir_gold_feature_store(
    spark: SparkSession,
    silver_dir: str,
    gold_dir: str,
    write_fn,
) -> dict:
    """Construye gold/gold_feature_store_30min integrando A1-A4 + B1 sobre la grilla maestra."""
    grid = construir_grilla_maestra(spark)

    despacho = spark.read.parquet(f"{gold_dir}/despacho_central_30min")
    a1 = spark.read.parquet(f"{silver_dir}/a1_despacho_programado")
    precios_tronc = spark.read.parquet(f"{gold_dir}/gold_precios_troncales_30min").drop("fecha_despacho", "anio")
    eventos_crit = spark.read.parquet(f"{gold_dir}/gold_eventos_criticos_30min").drop("fecha_despacho", "anio")
    observabilidad = spark.read.parquet(f"{gold_dir}/gold_observabilidad_etiqueta_30min").drop("fecha_despacho", "anio")

    # 1) Presencia mensual en A1 (consistente con gold_observabilidad)
    a1_mensual = (
        a1.withColumn("fecha_despacho_a1", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("anio_mes", F.date_format("fecha_despacho_a1", "yyyy-MM"))
        .select("central", "anio_mes")
        .distinct()
        .withColumn("central_en_a1_mes", F.lit(True))
    )

    desp_prep = (
        despacho.withColumn("fecha_despacho_d", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("anio_mes", F.date_format("fecha_despacho_d", "yyyy-MM"))
        .join(F.broadcast(a1_mensual), ["central", "anio_mes"], "left")
        .withColumn("central_en_a1_mes", F.coalesce(F.col("central_en_a1_mes"), F.lit(False)))
        .withColumn("comb_norm", F.upper(F.trim(F.col("combustible_elegido"))))
        .withColumn("es_term_cara", F.col("comb_norm").isin(*COMBUSTIBLES_CAROS))
    )

    sin_programa_valido = (F.col("potencia_programada_mw") == 0.0) | (
        F.col("potencia_programada_mw").isNull() & F.col("central_en_a1_mes")
    )
    cond_nop = (
        F.col("es_term_cara")
        & (F.col("potencia_ejecutada_mw") > F.lit(UMBRAL_POTENCIA_NOP_MW))
        & sin_programa_valido
    )

    # 2) Agregacion de despacho y costos a grano de bloque de 30 minutos
    desp_30min = (
        desp_prep.groupBy("fecha_hora")
        .agg(
            F.sum("potencia_ejecutada_mw").alias("gen_total_mw"),
            F.max(F.col("programado_disponible").cast("int")).alias("_prog_disp"),
            F.sum("potencia_programada_mw").alias("_prog_sum"),
            F.sum(F.when(F.col("es_term_cara"), F.coalesce(F.col("potencia_ejecutada_mw"), F.lit(0.0))).otherwise(0.0)).alias(
                "term_cara_ejec_mw"
            ),
            F.sum(F.when(F.col("es_term_cara"), F.coalesce(F.col("potencia_programada_mw"), F.lit(0.0))).otherwise(0.0)).alias(
                "_term_cara_prog_sum"
            ),
            F.sum(F.when(cond_nop, F.col("potencia_ejecutada_mw") * F.lit(0.5)).otherwise(0.0)).alias("_energia_nop_mwh_raw"),
            F.sum(
                F.when(
                    cond_nop & F.col("costo_variable_s_kwh").isNotNull(),
                    F.col("potencia_ejecutada_mw") * F.lit(0.5) * F.lit(1000.0) * F.col("costo_variable_s_kwh"),
                ).otherwise(0.0)
            ).alias("_exposicion_bruta_raw"),
        )
        .withColumn(
            "prog_total_mw",
            F.when(F.col("_prog_disp") == 1, F.col("_prog_sum")).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "term_cara_prog_mw",
            F.when(F.col("_prog_disp") == 1, F.col("_term_cara_prog_sum")).otherwise(F.lit(None).cast("double")),
        )
        .withColumn("brecha_despacho_mw", F.col("gen_total_mw") - F.col("prog_total_mw"))
        .withColumn(
            "term_cara_share",
            F.when(F.col("gen_total_mw") > 0.0, F.col("term_cara_ejec_mw") / F.col("gen_total_mw")).otherwise(
                F.lit(None).cast("double")
            ),
        )
        .drop("_prog_disp", "_prog_sum", "_term_cara_prog_sum")
    )

    # 3) Union sobre la grilla maestra (52 608 filas) — SIN gold_contexto_auditoria_30min
    fs = (
        grid.join(desp_30min, "fecha_hora", "left")
        .join(precios_tronc, "fecha_hora", "left")
        .join(eventos_crit, "fecha_hora", "left")
        .join(observabilidad, "fecha_hora", "left")
        .withColumn(
            "energia_nop_mwh",
            F.when(F.col("etiqueta_observable"), F.coalesce(F.col("_energia_nop_mwh_raw"), F.lit(0.0))).otherwise(
                F.lit(None).cast("double")
            ),
        )
        .withColumn(
            "exposicion_bruta_soles",
            F.when(F.col("etiqueta_observable"), F.coalesce(F.col("_exposicion_bruta_raw"), F.lit(0.0))).otherwise(
                F.lit(None).cast("double")
            ),
        )
        .withColumn(
            "sobrecosto_neto_estimado_soles",
            F.when(
                F.col("etiqueta_observable")
                & F.col("cmg_troncal_mediana").isNotNull()
                & F.col("exposicion_bruta_soles").isNotNull(),
                F.col("exposicion_bruta_soles")
                - (F.col("energia_nop_mwh") * F.lit(1000.0) * F.col("cmg_troncal_mediana")),
            ).otherwise(F.lit(None).cast("double")),
        )
        .drop("_energia_nop_mwh_raw", "_exposicion_bruta_raw")
        .select(*list(LINAJE_COLUMNAS.keys()))
    )

    # Barrera defensiva antifuga antes de escribir
    col_fuga = set(fs.columns).intersection(COLUMNAS_PROHIBIDAS_ANTIFUGA)
    assert not col_fuga, f"Fuga detectada en gold_feature_store_30min: {col_fuga}"

    out_path = f"{gold_dir}/gold_feature_store_30min"
    n_out = write_fn(spark, fs, out_path, ["fecha_hora"], files_per_year=1)

    # 4) Auditoria economica y de cobertura desde disco
    check = spark.read.parquet(out_path)
    por_anio = {}
    for r in (
        check.filter("etiqueta_observable = true")
        .groupBy("anio")
        .agg(
            F.round(F.sum("energia_nop_mwh"), 2).alias("mwh"),
            F.round(F.sum("exposicion_bruta_soles") / 1e6, 2).alias("exp_bruta_m_soles"),
            F.round(F.sum("sobrecosto_neto_estimado_soles") / 1e6, 2).alias("sobrecosto_neto_m_soles"),
            F.round(F.avg("term_cara_prog_mw"), 2).alias("term_cara_prog_prom_mw"),
        )
        .orderBy("anio")
        .collect()
    ):
        por_anio[str(r["anio"])] = {
            "energia_nop_mwh": float(r["mwh"] or 0.0),
            "exposicion_bruta_millones_soles": float(r["exp_bruta_m_soles"] or 0.0),
            "sobrecosto_neto_millones_soles": float(r["sobrecosto_neto_m_soles"] or 0.0),
            "term_cara_prog_prom_mw": float(r["term_cara_prog_prom_mw"]) if r["term_cara_prog_prom_mw"] is not None else None,
        }

    return {
        "filas": n_out,
        "duplicados_fecha_hora": check.groupBy("fecha_hora").count().filter("count > 1").count(),
        "columnas_totales": len(check.columns),
        "columnas_prohibidas_presentes": sorted(set(check.columns).intersection(COLUMNAS_PROHIBIDAS_ANTIFUGA)),
        "resumen_economico_por_anio": por_anio,
        "linaje": {
            "script": "src/etl/gold_feature_store.py",
            "version": SCRIPT_VERSION,
            "columnas": LINAJE_COLUMNAS,
        },
    }