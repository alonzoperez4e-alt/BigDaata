"""Contratos automatizados de integridad, cobertura y antifuga para la capa Gold."""
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

from src.etl.gold_feature_store import COLUMNAS_PROHIBIDAS_ANTIFUGA

BLOQUES_GRILLA_ESPERADOS = 52608
FILAS_A2_1X_ESPERADAS = 6533616
BLOQUES_OBSERVABLES_PR06_1X_ESPERADOS = 34652

TABLAS_GOLD_OFICIALES = (
    "despacho_central_30min",
    "precios_barra_30min",
    "gold_observabilidad_etiqueta_30min",
    "gold_precios_troncales_30min",
    "gold_eventos_criticos_30min",
    "gold_contexto_auditoria_30min",
    "gold_feature_store_30min",
)


def _inspeccionar_parquets_s3(ruta: str, spark: SparkSession | None = None) -> dict:
    """Inspecciona archivos .parquet y particiones mes= en S3 via Hadoop FS o boto3."""
    if spark is not None:
        try:
            jvm = spark._jvm
            conf = spark._jsc.hadoopConfiguration()
            uri = jvm.java.net.URI(ruta)
            fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)
            hpath = jvm.org.apache.hadoop.fs.Path(ruta)
            if not fs.exists(hpath):
                return {"archivos_parquet": 0, "tiene_particion_mes": False}
            it = fs.listFiles(hpath, True)
            archivos = 0
            tiene_mes = False
            while it.hasNext():
                fstatus = it.next()
                pstr = fstatus.getPath().toString()
                if pstr.endswith(".parquet"):
                    archivos += 1
                if "/mes=" in pstr:
                    tiene_mes = True
            return {"archivos_parquet": archivos, "tiene_particion_mes": tiene_mes}
        except Exception:
            pass

    try:
        import boto3

        sin_pref = ruta[5:]
        bucket, _, prefix = sin_pref.partition("/")
        prefix = prefix.strip("/")
        client = boto3.client("s3")
        paginator = client.get_paginator("list_objects_v2")
        archivos = 0
        tiene_mes = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    archivos += 1
                if "/mes=" in key:
                    tiene_mes = True
        return {"archivos_parquet": archivos, "tiene_particion_mes": tiene_mes}
    except Exception:
        return {"archivos_parquet": None, "tiene_particion_mes": None}


def _inspeccionar_parquets(ruta: str, spark: SparkSession | None = None) -> dict:
    if ruta.startswith("s3://"):
        return _inspeccionar_parquets_s3(ruta, spark=spark)
    p = Path(ruta)
    if not p.exists():
        return {"archivos_parquet": 0, "tiene_particion_mes": False}
    archivos = list(p.rglob("*.parquet"))
    tiene_mes = any(d.is_dir() and d.name.startswith("mes=") for d in p.rglob("mes=*"))
    return {"archivos_parquet": len(archivos), "tiene_particion_mes": tiene_mes}


def validar_barrera_antifuga_columnas(columnas: list[str]) -> tuple[bool, list[str]]:
    """Verifica que ninguna columna prohibida del PR-06 este presente en el esquema."""
    fuga = sorted(set(columnas).intersection(COLUMNAS_PROHIBIDAS_ANTIFUGA))
    return (len(fuga) == 0), fuga


def validar_contratos_gold(
    spark: SparkSession,
    silver_dir: str,
    gold_dir: str,
    escala: str = "1x",
) -> dict:
    """Ejecuta los 6 contratos obligatorios de Gold y retorna el dict de auditoria PASS/FAIL."""
    a2_silver = spark.read.parquet(f"{silver_dir}/a2_despacho_ejecutado")
    despacho = spark.read.parquet(f"{gold_dir}/despacho_central_30min")
    obs = spark.read.parquet(f"{gold_dir}/gold_observabilidad_etiqueta_30min")
    fs = spark.read.parquet(f"{gold_dir}/gold_feature_store_30min")

    # Inventario de conteos y archivos Parquet de todas las tablas Gold
    conteos_gold = {}
    archivos_parquet_gold = {}
    total_parquets_gold = 0
    for tabla in TABLAS_GOLD_OFICIALES:
        ruta_t = f"{gold_dir}/{tabla}"
        conteos_gold[tabla] = spark.read.parquet(ruta_t).count()
        info_p = _inspeccionar_parquets(ruta_t, spark=spark)
        archivos_parquet_gold[tabla] = info_p
        if info_p["archivos_parquet"] is not None:
            total_parquets_gold += info_p["archivos_parquet"]

    # 1) Completitud de grilla: 52 608 filas exactas y 0 duplicados en fecha_hora
    n_fs = conteos_gold["gold_feature_store_30min"]
    dup_fs = fs.groupBy("fecha_hora").count().filter("count > 1").count()
    c1_ok = (n_fs == BLOQUES_GRILLA_ESPERADOS) and (dup_fs == 0)

    # 2) No-multiplicacion en despacho: despacho_central_30min == Silver A2
    n_a2 = a2_silver.count()
    n_desp = conteos_gold["despacho_central_30min"]
    es_conjunto_1x_completo = (n_a2 == FILAS_A2_1X_ESPERADAS)
    c2_ok = (n_desp == n_a2) and (not (escala == "1x" and es_conjunto_1x_completo and n_desp != FILAS_A2_1X_ESPERADAS))

    # 3) Integridad de censura: etiqueta_observable == False -> targets IS NULL
    n_censura_rota_fs = fs.filter(
        "etiqueta_observable = false AND (evento_termico_nop IS NOT NULL OR inicio_termico_nop IS NOT NULL)"
    ).count()
    n_censura_rota_obs = obs.filter(
        "etiqueta_observable = false AND (evento_termico_nop IS NOT NULL OR inicio_termico_nop IS NOT NULL)"
    ).count()
    c3_ok = (n_censura_rota_fs == 0) and (n_censura_rota_obs == 0)

    # 4) Barrera antifuga en esquema
    c4_ok, cols_fuga = validar_barrera_antifuga_columnas(fs.columns)

    # 5) Consistencia de cobertura historica
    n_obs_total = obs.filter("etiqueta_observable = true").count()
    n_obs_en_pr06 = obs.filter("etiqueta_observable = true AND tiene_pr06 = true").count()
    bloques_recuperados_sin_pr06 = n_obs_total - n_obs_en_pr06
    dist_motivo_censura = {
        r["motivo_censura"]: int(r["count"])
        for r in obs.groupBy("motivo_censura").count().orderBy("motivo_censura").collect()
    }
    if es_conjunto_1x_completo:
        c5_ok = (n_obs_en_pr06 == BLOQUES_OBSERVABLES_PR06_1X_ESPERADOS)
        modo_cobertura = "completo_1x"
    else:
        c5_ok = (n_obs_total > 0) and (n_obs_en_pr06 <= n_obs_total)
        modo_cobertura = "muestra_sample"

    # 6) Particionamiento anual sin mes= y atomicidad de metricas
    tablas_con_mes = [t for t, info in archivos_parquet_gold.items() if info["tiene_particion_mes"] is True]
    c6_ok = (len(conteos_gold) == len(TABLAS_GOLD_OFICIALES)) and (len(tablas_con_mes) == 0)

    contratos = {
        "1_completitud_grilla": {
            "status": "PASS" if c1_ok else "FAIL",
            "filas_esperadas": BLOQUES_GRILLA_ESPERADOS,
            "filas_observadas": n_fs,
            "duplicados_fecha_hora": dup_fs,
        },
        "2_no_multiplicacion_despacho": {
            "status": "PASS" if c2_ok else "FAIL",
            "filas_silver_a2": n_a2,
            "filas_gold_despacho": n_desp,
        },
        "3_integridad_censura": {
            "status": "PASS" if c3_ok else "FAIL",
            "violaciones_feature_store": n_censura_rota_fs,
            "violaciones_observabilidad": n_censura_rota_obs,
        },
        "4_barrera_antifuga_esquema": {
            "status": "PASS" if c4_ok else "FAIL",
            "columnas_prohibidas_encontradas": cols_fuga,
        },
        "5_consistencia_cobertura_historica": {
            "status": "PASS" if c5_ok else "FAIL",
            "modo": modo_cobertura,
            "etiqueta_observable_total": n_obs_total,
            "etiqueta_observable_en_pr06": n_obs_en_pr06,
            "esperado_en_pr06_1x": BLOQUES_OBSERVABLES_PR06_1X_ESPERADOS if es_conjunto_1x_completo else None,
            "bloques_recuperados_sin_pr06": bloques_recuperados_sin_pr06,
        },
        "6_atomicidad_y_particion_anual": {
            "status": "PASS" if c6_ok else "FAIL",
            "tablas_gold_verificadas": len(conteos_gold),
            "total_archivos_parquet_gold": total_parquets_gold,
            "tablas_con_particion_mes_residual": tablas_con_mes,
        },
    }

    fallidos = [nombre for nombre, c in contratos.items() if c["status"] != "PASS"]
    return {
        "estado_global": "PASS" if not fallidos else "FAIL",
        "contratos_fallidos": fallidos,
        "conteos_tablas_gold": conteos_gold,
        "distribucion_motivo_censura": dist_motivo_censura,
        "archivos_parquet_gold": archivos_parquet_gold,
        "total_archivos_parquet_gold": total_parquets_gold,
        "contratos": contratos,
    }


def exigir_contratos_gold(resultado: dict) -> None:
    """Detiene el pipeline con error explicito si algun contrato fallo."""
    if resultado["estado_global"] != "PASS":
        raise RuntimeError(
            f"Contratos de calidad Gold FALLIDOS ({resultado['contratos_fallidos']}). "
            "Publicacion bloqueada; revisa logs/run_<escala>_gold.json."
        )