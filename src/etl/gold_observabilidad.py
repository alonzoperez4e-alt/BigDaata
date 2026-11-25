"""Grilla maestra temporal (52 608 bloques) y gold_observabilidad_etiqueta_30min."""
from pyspark.sql import DataFrame, SparkSession, Window, functions as F

COMBUSTIBLES_CAROS = ("D2", "R500", "NLH_RFG", "FX_RFG", "CARB")
UMBRAL_POTENCIA_NOP_MW = 5.0


def construir_grilla_maestra(spark: SparkSession) -> DataFrame:
    """Genera los 52 608 bloques exactos de 30 min entre 2021-12-01 00:30 y 2024-12-01 00:00."""
    return (
        spark.sql(
            """
            SELECT explode(sequence(
                timestamp('2021-12-01 00:30:00'),
                timestamp('2024-12-01 00:00:00'),
                interval 30 minutes
            )) AS fecha_hora
            """
        )
        .withColumn("fecha_despacho", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("anio", F.year("fecha_despacho"))
    )


def construir_gold_observabilidad(
    spark: SparkSession,
    silver_dir: str,
    gold_dir: str,
    write_fn,
) -> dict:
    """Materializa gold/gold_observabilidad_etiqueta_30min y retorna metricas de auditoria."""
    grid = construir_grilla_maestra(spark)

    a1 = spark.read.parquet(f"{silver_dir}/a1_despacho_programado")
    a2 = spark.read.parquet(f"{silver_dir}/a2_despacho_ejecutado")
    a3 = spark.read.parquet(f"{silver_dir}/a3_costos_variables")
    b4 = spark.read.parquet(f"{silver_dir}/b4_pr06")
    despacho_gold = spark.read.parquet(f"{gold_dir}/despacho_central_30min")

    # 1) Banderas de disponibilidad real por fuente
    disp_a1 = (
        a1.filter(F.col("potencia_programada_mw").isNotNull())
        .select("fecha_hora")
        .distinct()
        .withColumn("tiene_a1", F.lit(True))
    )
    disp_a2 = a2.select("fecha_hora").distinct().withColumn("tiene_a2", F.lit(True))
    disp_a3 = (
        a3.filter(F.col("costo_variable_s_kwh").isNotNull())
        .select(F.col("fecha").alias("fecha_despacho"))
        .distinct()
        .withColumn("tiene_a3", F.lit(True))
    )
    disp_pr06 = b4.select("fecha_hora").distinct().withColumn("tiene_pr06", F.lit(True))

    obs = (
        grid.join(disp_a1, "fecha_hora", "left")
        .join(disp_a2, "fecha_hora", "left")
        .join(disp_a3, "fecha_despacho", "left")
        .join(disp_pr06, "fecha_hora", "left")
        .withColumn("tiene_a1", F.coalesce(F.col("tiene_a1"), F.lit(False)))
        .withColumn("tiene_a2", F.coalesce(F.col("tiene_a2"), F.lit(False)))
        .withColumn("tiene_a3", F.coalesce(F.col("tiene_a3"), F.lit(False)))
        .withColumn("tiene_pr06", F.coalesce(F.col("tiene_pr06"), F.lit(False)))
        .withColumn("etiqueta_observable", F.col("tiene_a1") & F.col("tiene_a3") & F.col("tiene_a2"))
        .withColumn(
            "motivo_censura",
            F.when(~F.col("tiene_a1") & ~F.col("tiene_a3"), F.lit("SIN_A1_Y_A3"))
            .when(~F.col("tiene_a1"), F.lit("SIN_A1"))
            .when(~F.col("tiene_a3"), F.lit("SIN_A3"))
            .when(~F.col("tiene_a2"), F.lit("SIN_A2"))
            .when(~F.col("tiene_pr06"), F.lit("SIN_PR06"))
            .otherwise(F.lit("OK")),
        )
    )

    # 2) Presencia mensual en A1 para distinguir "no programada" de "omision de fuente"
    a1_mensual = (
        a1.withColumn("fecha_despacho_a1", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("anio_mes", F.date_format("fecha_despacho_a1", "yyyy-MM"))
        .select("central", "anio_mes")
        .distinct()
        .withColumn("central_en_a1_mes", F.lit(True))
    )

    # 3) Regla validada del evento termico no programado (proxy de redespacho forzado)
    desp_eval = (
        despacho_gold.withColumn("fecha_despacho_d", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("anio_mes", F.date_format("fecha_despacho_d", "yyyy-MM"))
        .join(F.broadcast(a1_mensual), ["central", "anio_mes"], "left")
        .withColumn("central_en_a1_mes", F.coalesce(F.col("central_en_a1_mes"), F.lit(False)))
        .withColumn("comb_norm", F.upper(F.trim(F.col("combustible_elegido"))))
    )

    sin_programa_valido = (F.col("potencia_programada_mw") == 0.0) | (
        F.col("potencia_programada_mw").isNull() & F.col("central_en_a1_mes")
    )
    cond_nop = (
        F.col("comb_norm").isin(*COMBUSTIBLES_CAROS)
        & (F.col("potencia_ejecutada_mw") > F.lit(UMBRAL_POTENCIA_NOP_MW))
        & sin_programa_valido
    )

    eventos_por_bloque = (
        desp_eval.filter(cond_nop)
        .groupBy("fecha_hora")
        .agg(
            F.lit(1).alias("evento_raw"),
            F.countDistinct("central").alias("n_centrales_term_nop"),
        )
    )

    # 4) Cruce sobre la grilla y calculo estricto de evento_termico_nop e inicio_termico_nop
    obs = obs.join(eventos_por_bloque, "fecha_hora", "left").withColumn(
        "evento_termico_nop",
        F.when(F.col("etiqueta_observable"), F.coalesce(F.col("evento_raw"), F.lit(0))).otherwise(
            F.lit(None).cast("int")
        ),
    )

    w_tiempo = Window.orderBy("fecha_hora")
    prev_evento = F.lag("evento_termico_nop", 1).over(w_tiempo)

    obs = (
        obs.withColumn(
            "inicio_termico_nop",
            F.when(~F.col("etiqueta_observable"), F.lit(None).cast("int"))
            .when((F.col("evento_termico_nop") == 1) & (prev_evento.isNull() | (prev_evento == 0)), F.lit(1))
            .otherwise(F.lit(0)),
        )
        .withColumn(
            "n_centrales_term_nop",
            F.when(F.col("etiqueta_observable"), F.coalesce(F.col("n_centrales_term_nop"), F.lit(0))).otherwise(
                F.lit(None).cast("int")
            ),
        )
        .drop("evento_raw")
        .select(
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
            "n_centrales_term_nop",
        )
    )

    out_path = f"{gold_dir}/gold_observabilidad_etiqueta_30min"
    n_out = write_fn(spark, obs, out_path, ["fecha_hora"], files_per_year=1)

    # 5) Auditoria desde disco para confirmar contrato
    check = spark.read.parquet(out_path)
    dist_motivos = {r["motivo_censura"]: int(r["count"]) for r in check.groupBy("motivo_censura").count().collect()}
    n_obs = check.filter("etiqueta_observable = true").count()
    n_obs_pr06 = check.filter("etiqueta_observable = true AND tiene_pr06 = true").count()
    n_ev = check.filter("evento_termico_nop = 1").count()
    n_ini = check.filter("inicio_termico_nop = 1").count()
    n_fuga_censura = check.filter(
        "etiqueta_observable = false AND (evento_termico_nop IS NOT NULL OR inicio_termico_nop IS NOT NULL)"
    ).count()

    por_anio = {}
    for r in (
        check.filter("etiqueta_observable = true")
        .groupBy("anio")
        .agg(
            F.count("*").alias("bloques_etiquetables"),
            F.sum("evento_termico_nop").alias("bloques_evento"),
            F.sum("inicio_termico_nop").alias("inicios"),
        )
        .orderBy("anio")
        .collect()
    ):
        b_et = int(r["bloques_etiquetables"])
        b_ev = int(r["bloques_evento"] or 0)
        por_anio[str(r["anio"])] = {
            "bloques_etiquetables": b_et,
            "bloques_evento": b_ev,
            "prevalencia_pct": round(100 * b_ev / b_et, 2) if b_et else 0.0,
            "inicios": int(r["inicios"] or 0),
        }

    return {
        "filas": n_out,
        "etiqueta_observable_true": n_obs,
        "etiqueta_observable_en_pr06": n_obs_pr06,
        "bloques_censurados": n_out - n_obs,
        "censurados_con_etiqueta_no_nula": n_fuga_censura,
        "bloques_evento_termico_nop": n_ev,
        "inicios_termico_nop": n_ini,
        "distribucion_motivo_censura": dist_motivos,
        "resumen_por_anio": por_anio,
    }