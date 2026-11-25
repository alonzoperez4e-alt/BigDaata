"""Tablas analiticas especializadas (A4 troncal, B1 segmentado y B3/B4 auditoria)."""
from pyspark.sql import SparkSession, functions as F

from src.etl.gold_observabilidad import construir_grilla_maestra

GRID_START_TS = "2021-12-01 00:30:00"
GRID_END_TS = "2024-12-01 00:00:00"
REGEX_TRONCAL = r"(^|[^0-9])(220|500)([^0-9]|$)"


def _construir_precios_troncales(spark: SparkSession, grid, silver_dir: str, gold_dir: str, write_fn) -> dict:
    """Agrega A4 en barras troncales (220 y 500 kV) a nivel sistema por fecha_hora."""
    a4 = spark.read.parquet(f"{silver_dir}/a4_costos_marginales")

    troncales = (
        a4.withColumn("barra_norm", F.upper(F.trim(F.col("barra"))))
        .filter(F.col("barra_norm").rlike(REGEX_TRONCAL))
        .withColumn(
            "costo_marginal_limpio",
            F.when(F.col("costo_marginal_s_kwh") == 0.0, F.lit(None).cast("double")).otherwise(
                F.col("costo_marginal_s_kwh")
            ),
        )
    )

    n_barras_troncales_unicas = troncales.select("barra_id").distinct().count()

    agg_troncal = (
        troncales.groupBy("fecha_hora")
        .agg(
            F.expr("percentile(costo_marginal_limpio, 0.5)").alias("cmg_troncal_mediana"),
            F.expr("percentile(costo_marginal_limpio, 0.9)").alias("cmg_troncal_p90"),
            F.count("costo_marginal_limpio").cast("int").alias("n_barras_troncales_validas"),
        )
        .withColumn("cmg_spread_p90_p50", F.col("cmg_troncal_p90") - F.col("cmg_troncal_mediana"))
    )

    out_df = (
        grid.join(agg_troncal, "fecha_hora", "left")
        .withColumn("n_barras_troncales_validas", F.coalesce(F.col("n_barras_troncales_validas"), F.lit(0)))
        .select(
            "fecha_hora",
            "fecha_despacho",
            "anio",
            "cmg_troncal_mediana",
            "cmg_troncal_p90",
            "cmg_spread_p90_p50",
            "n_barras_troncales_validas",
        )
    )

    out_path = f"{gold_dir}/gold_precios_troncales_30min"
    n_out = write_fn(spark, out_df, out_path, ["fecha_hora"], files_per_year=1)

    check = spark.read.parquet(out_path)
    stats = check.agg(
        F.sum(F.when(F.col("cmg_troncal_mediana").isNotNull(), 1).otherwise(0)).alias("bloques_con_precio"),
        F.round(F.avg("cmg_troncal_mediana"), 4).alias("cmg_mediana_prom"),
        F.round(F.avg("cmg_spread_p90_p50"), 4).alias("spread_prom"),
        F.round(F.avg("n_barras_troncales_validas"), 2).alias("barras_validas_prom"),
    ).first()

    return {
        "filas": n_out,
        "barras_troncales_identificadas": n_barras_troncales_unicas,
        "bloques_con_precio": int(stats["bloques_con_precio"] or 0),
        "cmg_mediana_prom_s_kwh": float(stats["cmg_mediana_prom"]) if stats["cmg_mediana_prom"] is not None else None,
        "spread_p90_p50_prom_s_kwh": float(stats["spread_prom"]) if stats["spread_prom"] is not None else None,
        "barras_validas_prom_por_bloque": float(stats["barras_validas_prom"] or 0.0),
    }


def _construir_eventos_criticos(spark: SparkSession, grid, silver_dir: str, gold_dir: str, write_fn) -> dict:
    """Segmenta B1 en 4 indicadores operativos estrictamente conocidos en t."""
    b1 = (
        spark.read.parquet(f"{silver_dir}/b1_mantenimientos")
        .filter(F.col("fecha_inicio").isNotNull() & F.col("fecha_fin").isNotNull())
        .filter(F.col("fecha_fin") >= F.col("fecha_inicio"))
        .withColumn("prog_norm", F.upper(F.trim(F.col("prog"))))
        .withColumn("tipo_eq_norm", F.upper(F.trim(F.col("tipo_equipo"))))
        .withColumn("es_termico", F.col("tipo_eq_norm").rlike("TERMOEL"))
        .withColumn("es_transmision", F.col("tipo_eq_norm").rlike("LINEA DE TRANSMISION|TRANSFORMADOR"))
        .withColumn("es_forzado_termico", (F.col("prog_norm") == "FORZADO") & F.col("es_termico"))
        .withColumn("es_forzado_transmision", (F.col("prog_norm") == "FORZADO") & F.col("es_transmision"))
        .withColumn("es_prog_termico", (F.col("prog_norm") == "PROGRAMADO") & F.col("es_termico"))
        .filter(
            F.col("es_forzado_termico") | F.col("es_forzado_transmision") | F.col("es_prog_termico")
        )
    )

    # 1) Eventos activos en t: fecha_inicio <= fecha_hora <= fecha_fin
    # Expansion por dia de despacho (equi-join rapido y exacto)
    b1_act = (
        b1.withColumn("ini_c", F.greatest(F.col("fecha_inicio"), F.lit(GRID_START_TS).cast("timestamp")))
        .withColumn("fin_c", F.least(F.col("fecha_fin"), F.lit(GRID_END_TS).cast("timestamp")))
        .filter(F.col("ini_c") <= F.col("fin_c"))
        .withColumn("dia_ini", F.to_date(F.expr("ini_c - interval 1 minute")))
        .withColumn("dia_fin", F.to_date(F.expr("fin_c - interval 1 minute")))
        .filter(F.col("dia_ini") <= F.col("dia_fin"))
        .withColumn("fecha_despacho", F.explode(F.sequence("dia_ini", "dia_fin", F.expr("interval 1 day"))))
        .select(
            "fecha_despacho",
            "fecha_inicio",
            "fecha_fin",
            "es_forzado_termico",
            "es_forzado_transmision",
            "es_prog_termico",
        )
    )

    agg_activos = (
        grid.select("fecha_hora", "fecha_despacho")
        .join(b1_act, "fecha_despacho", "inner")
        .filter((F.col("fecha_inicio") <= F.col("fecha_hora")) & (F.col("fecha_fin") >= F.col("fecha_hora")))
        .groupBy("fecha_hora")
        .agg(
            F.sum(F.col("es_forzado_termico").cast("int")).cast("int").alias("n_forzados_termicos_activos"),
            F.sum(F.col("es_forzado_transmision").cast("int")).cast("int").alias("n_forzados_transmision_activos"),
            F.sum(F.col("es_prog_termico").cast("int")).cast("int").alias("n_mantenimientos_prog_termicos_activos"),
        )
    )

    # 2) Inicios recientes en ventana [fecha_hora - 3h, fecha_hora] (solo forzados termicos)
    b1_ini3h = (
        b1.filter(F.col("es_forzado_termico"))
        .filter(
            (F.col("fecha_inicio") >= F.expr(f"timestamp('{GRID_START_TS}') - interval 3 hours"))
            & (F.col("fecha_inicio") <= F.lit(GRID_END_TS).cast("timestamp"))
        )
        .withColumn("dia_ini", F.to_date(F.expr("fecha_inicio - interval 1 minute")))
        .withColumn("dia_fin", F.to_date(F.expr("fecha_inicio + interval 3 hours - interval 1 minute")))
        .filter(F.col("dia_ini") <= F.col("dia_fin"))
        .withColumn("fecha_despacho", F.explode(F.sequence("dia_ini", "dia_fin", F.expr("interval 1 day"))))
        .select("fecha_despacho", "fecha_inicio")
    )

    agg_ini3h = (
        grid.select("fecha_hora", "fecha_despacho")
        .join(b1_ini3h, "fecha_despacho", "inner")
        .filter(
            (F.col("fecha_inicio") >= F.expr("fecha_hora - interval 3 hours"))
            & (F.col("fecha_inicio") <= F.col("fecha_hora"))
        )
        .groupBy("fecha_hora")
        .agg(F.count("*").cast("int").alias("n_forzados_termicos_inicio_3h"))
    )

    out_df = (
        grid.join(agg_activos, "fecha_hora", "left")
        .join(agg_ini3h, "fecha_hora", "left")
        .fillna(
            0,
            subset=[
                "n_forzados_termicos_activos",
                "n_forzados_transmision_activos",
                "n_forzados_termicos_inicio_3h",
                "n_mantenimientos_prog_termicos_activos",
            ],
        )
        .select(
            "fecha_hora",
            "fecha_despacho",
            "anio",
            "n_forzados_termicos_activos",
            "n_forzados_transmision_activos",
            "n_forzados_termicos_inicio_3h",
            "n_mantenimientos_prog_termicos_activos",
        )
    )

    out_path = f"{gold_dir}/gold_eventos_criticos_30min"
    n_out = write_fn(spark, out_df, out_path, ["fecha_hora"], files_per_year=1)

    check = spark.read.parquet(out_path)
    cols_ind = [
        "n_forzados_termicos_activos",
        "n_forzados_transmision_activos",
        "n_forzados_termicos_inicio_3h",
        "n_mantenimientos_prog_termicos_activos",
    ]
    cobertura_activa = {}
    for c in cols_ind:
        r = check.agg(
            F.sum(F.when(F.col(c) > 0, 1).otherwise(0)).alias("n_pos"),
            F.max(c).alias("max_val"),
        ).first()
        n_pos = int(r["n_pos"] or 0)
        cobertura_activa[c] = {
            "bloques_mayor_a_0": n_pos,
            "pct_bloques_activos": round(100 * n_pos / n_out, 2) if n_out else 0.0,
            "max_simultaneos": int(r["max_val"] or 0),
        }

    return {"filas": n_out, "indicadores": cobertura_activa}


def _construir_contexto_auditoria(spark: SparkSession, grid, silver_dir: str, gold_dir: str, write_fn) -> dict:
    """Segrega B3 y B4 (label_rdo, alfa/beta/gamma/chm y motivos) solo para auditoria/dashboard."""
    b3 = spark.read.parquet(f"{silver_dir}/b3_reprograma")
    b4 = spark.read.parquet(f"{silver_dir}/b4_pr06")

    # Tipado seguro de columnas numericas de PR-06 si existen en la fuente
    cols_num_b4 = [
        "alfa", "beta", "gamma", "chm",
        "alfa_primer_archivo", "beta_primer_archivo", "gamma_primer_archivo",
    ]
    for c in cols_num_b4:
        if c in b4.columns:
            b4 = b4.withColumn(c, F.col(c).cast("double"))
        else:
            b4 = b4.withColumn(c, F.lit(None).cast("double"))

    for c in ("forzado_activo_b1", "forzado_inicio_3h_previas_b1"):
        if c in b4.columns:
            b4 = b4.withColumn(c, F.col(c).cast("int"))
        else:
            b4 = b4.withColumn(c, F.lit(None).cast("int"))

    b4_sel = b4.select(
        "fecha_hora",
        "id_reprograma",
        "n_archivos_con_rdo1",
        "label_rdo",
        "alfa",
        "beta",
        "gamma",
        "chm",
        "alfa_primer_archivo",
        "beta_primer_archivo",
        "gamma_primer_archivo",
        "forzado_activo_b1",
        "forzado_inicio_3h_previas_b1",
    )

    # Resumen diario de B3 (garantiza 1 fila por fecha_despacho)
    b3_dia = (
        b3.groupBy(F.col("fecha").alias("fecha_despacho"))
        .agg(
            F.count("*").cast("int").alias("n_reprogramas_dia"),
            F.concat_ws(" | ", F.sort_array(F.collect_set("motivo_categoria"))).alias("motivos_categoria_dia"),
            F.max(F.when(F.col("motivo_categoria").contains("falla"), F.lit(1)).otherwise(F.lit(0)))
            .cast("int")
            .alias("tiene_motivo_falla_dia"),
            F.sum("costo_total_soles").alias("costo_total_reprogramas_dia_soles"),
        )
    )

    # Motivo especifico de la letra asociada al bloque (garantiza unicidad por fecha + letra)
    b3_letra = (
        b3.groupBy(
            F.col("fecha").alias("fecha_despacho"),
            F.upper(F.trim(F.col("letra_reprograma"))).alias("id_reprograma_norm"),
        )
        .agg(
            F.first("motivo", ignorenulls=True).alias("motivo_reprograma_bloque"),
            F.first("motivo_categoria", ignorenulls=True).alias("motivo_categoria_bloque"),
            F.first("costo_total_soles", ignorenulls=True).alias("costo_reprograma_bloque_soles"),
        )
    )

    out_df = (
        grid.join(b4_sel, "fecha_hora", "left")
        .withColumn("id_reprograma_norm", F.upper(F.trim(F.col("id_reprograma"))))
        .join(b3_dia, "fecha_despacho", "left")
        .join(b3_letra, ["fecha_despacho", "id_reprograma_norm"], "left")
        .withColumn("n_reprogramas_dia", F.coalesce(F.col("n_reprogramas_dia"), F.lit(0)))
        .withColumn("tiene_motivo_falla_dia", F.coalesce(F.col("tiene_motivo_falla_dia"), F.lit(0)))
        .drop("id_reprograma_norm")
        .select(
            "fecha_hora",
            "fecha_despacho",
            "anio",
            "label_rdo",
            "id_reprograma",
            "n_archivos_con_rdo1",
            "alfa",
            "beta",
            "gamma",
            "chm",
            "alfa_primer_archivo",
            "beta_primer_archivo",
            "gamma_primer_archivo",
            "forzado_activo_b1",
            "forzado_inicio_3h_previas_b1",
            "n_reprogramas_dia",
            "motivos_categoria_dia",
            "tiene_motivo_falla_dia",
            "costo_total_reprogramas_dia_soles",
            "motivo_reprograma_bloque",
            "motivo_categoria_bloque",
            "costo_reprograma_bloque_soles",
        )
    )

    out_path = f"{gold_dir}/gold_contexto_auditoria_30min"
    n_out = write_fn(spark, out_df, out_path, ["fecha_hora"], files_per_year=1)

    check = spark.read.parquet(out_path)
    return {
        "filas": n_out,
        "bloques_con_label_rdo_1": check.filter("label_rdo = 1").count(),
        "bloques_sin_pr06": check.filter("label_rdo IS NULL").count(),
        "bloques_con_reprograma_dia": check.filter("n_reprogramas_dia > 0").count(),
    }


def construir_senales_externas(
    spark: SparkSession,
    silver_dir: str,
    gold_dir: str,
    write_fn,
) -> dict:
    """Construye las tres tablas especializadas de la Tarea 2.3."""
    grid = construir_grilla_maestra(spark)
    return {
        "gold_precios_troncales_30min": _construir_precios_troncales(spark, grid, silver_dir, gold_dir, write_fn),
        "gold_eventos_criticos_30min": _construir_eventos_criticos(spark, grid, silver_dir, gold_dir, write_fn),
        "gold_contexto_auditoria_30min": _construir_contexto_auditoria(spark, grid, silver_dir, gold_dir, write_fn),
    }