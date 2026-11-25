"""Pipeline COES-SEIN: Bronze -> Silver -> Gold (Fase 2: particionamiento anual y compactacion).
Uso:
  python src/etl/pipeline.py --input coes_pipeline/lake_real/bronze --output coes_pipeline/lake_real --profile local_dev --stage all
  LAKE_ROOT=s3://bigdata-coes python src/etl/pipeline.py
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

# Compatibilidad con parametros estilo AWS Glue ("--LAKE_ROOT s3://...")
for _i, _a in enumerate(sys.argv):
    if _a.startswith("--") and _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("--"):
        os.environ.setdefault(_a[2:], sys.argv[_i + 1])

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.etl.gold_observabilidad import construir_gold_observabilidad
from src.etl.gold_senales_externas import construir_senales_externas
from src.etl.gold_feature_store import construir_gold_feature_store
from src.etl.quality_checks import exigir_contratos_gold, validar_contratos_gold
PROFILES_PATH = REPO_ROOT / "config" / "profiles.json"

# ---- Esquemas estrictos (sin inferencia) ----
D, TS, S, B = T.DoubleType(), T.TimestampType(), T.StringType(), T.BooleanType()
ESQ = {
    "a1": {"fecha_hora": TS, "central": S, "empresa": S, "potencia_programada_mw": D},
    "a2": {
        "fecha_hora": TS, "central": S, "empresa": S, "potencia_programada_mw": D,
        "potencia_ejecutada_mw": D, "desviacion_mw": D, "origen": S, "programado_disponible": B,
    },
    "a3": {
        "fecha": T.DateType(), "grupo_elegido": S, "central": S, "empresa": S,
        "combustible_elegido": S, "tipo_elegido": S, "confianza": S,
        "potencia_efectiva_mw": D, "cvnc_s_kwh": D, "cvc_s_kwh": D, "costo_variable_s_kwh": D,
    },
    "a4": {"fecha_hora": TS, "barra_id": S, "barra": S, "costo_marginal_s_kwh": D},
}

CSV_B = {
    "b1_mantenimientos": (
        "fecha_inicio",
        ["fecha_inicio", "fecha_fin"],
        {
            "fecha_inicio": TS, "fecha_fin": TS, "ubicacion": S, "equipo": S, "tipo_equipo": S,
            "tipo_mantenimiento": S, "prog": S, "n_filas_origen": T.IntegerType(), "duracion_horas": D,
        },
    ),
    "b2_potencia_indisponible": (
        "fecha_hora",
        ["fecha_hora", "instalacion"],
        {
            "fecha_hora": TS, "instalacion": S, "tipo_instalacion": S,
            "tipo_mantenimiento": S, "empresa": S, "potencia_indisponible_mw": D,
        },
    ),
    "b3_reprograma": (
        "fecha",
        ["fecha", "letra_reprograma"],
        {
            "fecha": T.DateType(), "letra_reprograma": S, "motivo": S, "motivo_categoria": S,
            "costo_total_soles": D, "fecha_hora_inicio": TS, "fecha_hora_fin": TS,
            "n_bloques": T.IntegerType(), "horas_cubiertas": D,
        },
    ),
    "b4_pr06": (
        "fecha_hora",
        ["fecha_hora"],
        {
            "fecha_hora": TS, "id_dia": S, "id_reprograma": S, "id_fuente": S,
            "n_archivos_con_rdo1": T.IntegerType(), "label_rdo": T.IntegerType(),
        },
    ),
}


def cargar_perfil(profile_name: str | None) -> dict:
    if not profile_name or not PROFILES_PATH.exists():
        return {}
    perfiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    if profile_name not in perfiles:
        raise ValueError(f"Perfil '{profile_name}' no existe en {PROFILES_PATH}. Disponibles: {list(perfiles)}")
    return perfiles[profile_name]


def crear_spark(escala: str, perfil: dict, en_glue: bool) -> SparkSession:
    shuffle = str(os.environ.get("SHUFFLE_PARTITIONS", perfil.get("shuffle_partitions", 8)))
    broadcast_bytes = int(perfil.get("broadcast_threshold_bytes", 10485760))
    if os.environ.get("BROADCAST") == "0":
        broadcast_bytes = -1

    builder = SparkSession.builder.appName(f"coes-pipeline-{escala}")
    if not en_glue:
        master = perfil.get("master", f"local[{os.environ.get('CORES', '*')}]")
        driver_mem = os.environ.get("DRIVER_MEM", perfil.get("driver_memory", "4g"))
        builder = builder.master(master).config("spark.driver.memory", driver_mem)
        local_dir = perfil.get("local_dir")
        if local_dir and os.name != "nt":
            builder = builder.config("spark.local.dir", local_dir)

    spark = (
        builder.config("spark.sql.shuffle.partitions", shuffle)
        .config("spark.sql.autoBroadcastJoinThreshold", str(broadcast_bytes))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def tipar(df: DataFrame, esq: dict) -> DataFrame:
    faltan = set(esq) - set(df.columns)
    assert not faltan, f"Columnas faltantes en la fuente: {faltan}"
    return df.select([F.col(c).cast(t).alias(c) for c, t in esq.items()])


def con_particion(df: DataFrame, col_fecha: str) -> DataFrame:
    """Particion anual por DIA DE DESPACHO: el bloque 00:00 pertenece al dia anterior."""
    dia = F.to_date(F.expr(f"{col_fecha} - interval 1 minute")) if col_fecha == "fecha_hora" else F.to_date(col_fecha)
    out = df.withColumn("anio", F.year(dia))
    return out.drop("mes") if "mes" in out.columns else out


def write_parquet_optimizado(
    spark: SparkSession,
    df: DataFrame,
    path: str,
    sort_cols: list[str],
    files_per_year: int = 1,
) -> int:
    """Escribe Parquet Snappy particionado por anio con compactacion y ordenamiento interno."""
    if "mes" in df.columns:
        df = df.drop("mes")

    if files_per_year <= 1:
        out_df = df.repartition("anio").sortWithinPartitions("anio", *sort_cols)
    else:
        n_anios = max(1, df.select("anio").distinct().count())
        out_df = df.repartitionByRange(n_anios * files_per_year, "anio", *sort_cols).sortWithinPartitions(
            "anio", *sort_cols
        )

    (
        out_df.write.mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("anio")
        .parquet(path)
    )
    return spark.read.parquet(path).count()


def escribir_atomico_json(spark: SparkSession, ruta_str: str, payload: dict) -> None:
    if ruta_str.startswith("s3://"):
        spark.createDataFrame([(json.dumps(payload, indent=2),)], ["j"]).coalesce(1).write.mode("overwrite").text(
            ruta_str + ".d"
        )
        return
    destino = Path(ruta_str)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, destino)


def silver_a(spark: SparkSession, bronze: str, silver: str, report: dict) -> dict:
    out = {}
    fuentes = {
        "a1": ("a1_despacho_programado", "fecha_hora", ["fecha_hora", "central"], "potencia_programada_mw", 1),
        "a2": ("a2_despacho_ejecutado", "fecha_hora", ["fecha_hora", "central"], "potencia_ejecutada_mw", 1),
        "a3": ("a3_costos_variables", "fecha", ["fecha", "central"], "costo_variable_s_kwh", 1),
        "a4": ("a4_costos_marginales", "fecha_hora", ["fecha_hora", "barra_id"], "costo_marginal_s_kwh", 2),
    }
    for k, (carpeta, colf, clave, critica, n_files) in fuentes.items():
        df = tipar(spark.read.parquet(f"{bronze}/{carpeta}"), ESQ[k])
        n_in = df.count()
        n_fecha_nula = df.filter(F.col(colf).isNull()).count()
        assert n_fecha_nula == 0, f"{k}: {n_fecha_nula} filas con {colf} ilegible tras el cast; revisar el formato"
        dup = df.groupBy(*clave).count().filter("count > 1").count()
        nulos = df.filter(F.col(critica).isNull()).count()
        neg = df.filter(F.col(critica) < 0).count()
        n_out = write_parquet_optimizado(
            spark, con_particion(df, colf), f"{silver}/{carpeta}", clave, files_per_year=n_files
        )
        report["calidad"][k] = {
            "filas_in": n_in,
            "filas_silver": n_out,
            "claves_duplicadas": dup,
            "nulos_columna_critica": nulos,
            "pct_completitud": round(100 * (1 - nulos / n_in), 3) if n_in else 0.0,
            "negativos": neg,
        }
        if k == "a2":
            report["calidad"]["a2"]["por_origen"] = {
                r["origen"]: r["count"] for r in df.groupBy("origen").count().collect()
            }
        out[k] = n_out

    a2 = spark.read.parquet(f"{silver}/a2_despacho_ejecutado")
    r = a2.select(
        F.countDistinct(F.to_date(F.expr("fecha_hora - interval 1 minute"))).alias("dias"),
        F.countDistinct(
            F.when(F.col("programado_disponible"), F.to_date(F.expr("fecha_hora - interval 1 minute")))
        ).alias("dias_con_programa"),
    ).first()
    report["calidad"]["cobertura_a2"] = {"dias": r.dias, "dias_con_programa": r.dias_con_programa}
    return out


def silver_b(spark: SparkSession, bronze: str, silver: str, report: dict) -> dict:
    out = {}
    for carpeta, (colf, sort_cols, tipos) in CSV_B.items():
        ruta = f"{bronze}/{carpeta}"
        df = (
            spark.read.option("header", True)
            .option("encoding", "UTF-8")
            .option("multiLine", True)
            .option("escape", '"')
            .csv(ruta)
        )
        df = df.toDF(*[c.lstrip("\ufeff").strip() for c in df.columns])
        faltan = set(tipos) - set(df.columns)
        assert not faltan, f"{carpeta}: columnas faltantes {faltan}"
        n_in = df.count()
        casted = [F.col(c).cast(t).alias(c) for c, t in tipos.items()]
        resto = [F.col(c) for c in df.columns if c not in tipos]
        df = df.select(*casted, *resto)
        n_mal = df.filter(F.col(colf).isNull()).count()
        out[carpeta] = write_parquet_optimizado(
            spark,
            con_particion(df.filter(F.col(colf).isNotNull()), colf),
            f"{silver}/{carpeta}",
            sort_cols,
            files_per_year=1,
        )
        report["calidad"][carpeta] = {
            "filas_in": n_in,
            "filas_silver": out[carpeta],
            "fecha_ilegible_rechazadas": n_mal,
        }

    b4 = spark.read.parquet(f"{silver}/b4_pr06")
    report["calidad"]["b4"] = {
        "bloques": b4.count(),
        "claves_duplicadas": b4.groupBy("fecha_hora").count().filter("count > 1").count(),
        "label_rdo_1": b4.filter("label_rdo = 1").count(),
    }
    return out


BYTES_ESTIMADOS_POR_FILA_A3 = 164

def gold(
    spark: SparkSession,
    silver: str,
    gold_dir: str,
    report: dict,
    broadcast_threshold_bytes: int,
) -> dict:
    a2 = spark.read.parquet(f"{silver}/a2_despacho_ejecutado")
    a3 = spark.read.parquet(f"{silver}/a3_costos_variables").select(
        F.col("fecha").alias("fecha_despacho"),
        "central",
        "combustible_elegido",
        "confianza",
        "costo_variable_s_kwh",
    )
    a2 = a2.withColumn("fecha_despacho", F.to_date(F.expr("fecha_hora - interval 1 minute")))

    n_a3 = a3.count()
    a3_bytes_estimados = n_a3 * BYTES_ESTIMADOS_POR_FILA_A3
    usar_broadcast = (
        (os.environ.get("BROADCAST", "1") != "0")
        and (broadcast_threshold_bytes >= 0)
        and (a3_bytes_estimados <= broadcast_threshold_bytes)
    )
    if not usar_broadcast:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    g = a2.join(F.broadcast(a3) if usar_broadcast else a3, ["central", "fecha_despacho"], "left")
    plan = g._jdf.queryExecution().executedPlan().toString()
    estrategia = (
        "BroadcastHashJoin"
        if "BroadcastHashJoin" in plan
        else ("SortMergeJoin" if "SortMergeJoin" in plan else "otro")
    )
    cen = a3.select("central").distinct().withColumn("central_en_a3", F.lit(True))
    g = g.join(F.broadcast(cen), "central", "left").withColumn(
        "central_en_a3", F.coalesce("central_en_a3", F.lit(False))
    )
    g = g.withColumn("sin_costo_variable", F.col("costo_variable_s_kwh").isNull())
    g = g.withColumn(
        "costo_estado",
        F.when(F.col("costo_variable_s_kwh").isNotNull(), "con_costo")
        .when(F.col("central_en_a3"), "fecha_sin_costo")
        .otherwise("central_sin_costo_en_a3"),
    )
    n_a2 = a2.count()
    n_g = write_parquet_optimizado(
        spark, g.drop("fecha_despacho"), f"{gold_dir}/despacho_central_30min", ["fecha_hora", "central"], files_per_year=1
    )
    report["calidad"]["gold_despacho"] = {
        "filas_a2": n_a2,
        "filas_gold": n_g,
        "join_multiplico_filas": n_g != n_a2,
        "estrategia_join": estrategia,
        "filas_sin_costo": g.filter("sin_costo_variable").count(),
        "por_costo_estado": {r["costo_estado"]: r["count"] for r in g.groupBy("costo_estado").count().collect()},
        "a3_bytes_estimados": a3_bytes_estimados,
        "broadcast_threshold_bytes": broadcast_threshold_bytes,
        "broadcast_activado_adaptativo": usar_broadcast,
    }

    a4 = spark.read.parquet(f"{silver}/a4_costos_marginales")
    p = a4.withColumn("cero_como_faltante", F.col("costo_marginal_s_kwh") == 0).withColumn(
        "costo_marginal_limpio",
        F.when(F.col("costo_marginal_s_kwh") == 0, None).otherwise(F.col("costo_marginal_s_kwh")),
    )
    n_p = write_parquet_optimizado(
        spark, p, f"{gold_dir}/precios_barra_30min", ["fecha_hora", "barra_id"], files_per_year=2
    )
    report["calidad"]["gold_precios"] = {
        "filas": n_p,
        "ceros_tratados_como_faltantes": p.filter("cero_como_faltante").count(),
    }
    obs_metrics = construir_gold_observabilidad(spark, silver, gold_dir, write_parquet_optimizado)
    report["calidad"]["gold_observabilidad"] = obs_metrics
    senales_metrics = construir_senales_externas(spark, silver, gold_dir, write_parquet_optimizado)
    report["calidad"]["gold_senales_externas"] = senales_metrics
    fs_metrics = construir_gold_feature_store(spark, silver, gold_dir, write_parquet_optimizado)
    report["linaje_gold"] = fs_metrics.pop("linaje")
    report["calidad"]["gold_feature_store"] = fs_metrics
    return {
        "despacho": n_g,
        "precios": n_p,
        "observabilidad": obs_metrics["filas"],
        "precios_troncales": senales_metrics["gold_precios_troncales_30min"]["filas"],
        "eventos_criticos": senales_metrics["gold_eventos_criticos_30min"]["filas"],
        "contexto_auditoria": senales_metrics["gold_contexto_auditoria_30min"]["filas"],
        "feature_store": fs_metrics["filas"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--profile", default="local_dev")
    parser.add_argument("--stage", default=None)
    args, _ = parser.parse_known_args()

    en_glue = "JOB_NAME" in os.environ
    root = (args.output or os.environ.get("LAKE_ROOT", "lake_test")).rstrip("/")
    bronze = (args.input or f"{root}/bronze").rstrip("/")
    silver, gold_dir, logs = (f"{root}/{z}" for z in ("silver", "gold", "logs"))
    escala = os.environ.get("ESCALA", "1x")

    perfil = {} if en_glue else cargar_perfil(args.profile)
    broadcast_bytes = int(perfil.get("broadcast_threshold_bytes", 10485760))
    use_broadcast = (os.environ.get("BROADCAST", "1") != "0") and (broadcast_bytes >= 0)

    if args.stage:
        st = args.stage.strip().lower()
        if st == "silver":
            etapas = ["silver_a", "silver_b"]
        elif st == "gold":
            etapas = ["gold"]
        elif st == "all":
            etapas = ["silver_a", "silver_b", "gold"]
        else:
            etapas = [x.strip() for x in st.split(",") if x.strip()]
    else:
        etapas = os.environ.get("ETAPAS", "silver_a,silver_b,gold").split(",")

    spark = crear_spark(escala, perfil, en_glue)
    report = {
        "entorno": "aws_glue" if en_glue else "local",
        "perfil": "aws_glue" if en_glue else args.profile,
        "particionamiento": "anio",
        "escala": escala,
        "inicio": dt.datetime.now().isoformat(timespec="seconds"),
        "broadcast": use_broadcast,
        "etapas_solicitadas": list(etapas),
        "etapas": {},
        "calidad": {},
    }

    def ejecutar_etapa(nombre, fn, *fn_args):
        t_et = time.perf_counter()
        res = fn(*fn_args)
        report["etapas"][nombre] = {"segundos": round(time.perf_counter() - t_et, 2), **(res or {})}
        print(f"[{nombre}] {report['etapas'][nombre]}")

    t0 = time.perf_counter()
    try:
        silver_etapas = [e for e in ("silver_a", "silver_b") if e in etapas]
        if "silver_a" in silver_etapas:
            ejecutar_etapa("silver_a", silver_a, spark, bronze, silver, report)
        if "silver_b" in silver_etapas:
            ejecutar_etapa("silver_b", silver_b, spark, bronze, silver, report)
        if silver_etapas:
            report["total_segundos"] = round(time.perf_counter() - t0, 2)
            report["fin"] = dt.datetime.now().isoformat(timespec="seconds")
            escribir_atomico_json(spark, f"{logs}/run_{escala}_silver.json", report)

        if "gold" in etapas:
         ejecutar_etapa("gold", gold, spark, silver, gold_dir, report, broadcast_bytes)
         contratos = validar_contratos_gold(spark, silver, gold_dir, escala=escala)
         report["contratos_gold"] = contratos
         report["total_segundos"] = round(time.perf_counter() - t0, 2)
         report["fin"] = dt.datetime.now().isoformat(timespec="seconds")
         escribir_atomico_json(spark, f"{logs}/run_{escala}_gold.json", report)
         exigir_contratos_gold(contratos)
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        report["total_segundos"] = round(time.perf_counter() - t0, 2)
        report["fin"] = dt.datetime.now().isoformat(timespec="seconds")
        escribir_atomico_json(spark, f"{logs}/run_{escala}_error.json", report)
        print("FALLO:", report["error"])
        raise
    finally:
        print(json.dumps(report, indent=2))
        spark.stop()


if __name__ == "__main__":
    main()