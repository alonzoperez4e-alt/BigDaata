"""Benchmark antes/despues de la compactacion y particionamiento anual."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


def inventariar_capa(capa_dir: Path) -> dict:
    por_tabla = {}
    total_archivos = 0
    total_bytes = 0
    if not capa_dir.exists():
        return {"total_archivos": 0, "total_mib": 0.0, "tamanio_medio_mib": 0.0, "tablas": {}}

    for tabla_dir in sorted(p for p in capa_dir.iterdir() if p.is_dir()):
        parquets = list(tabla_dir.rglob("*.parquet"))
        n = len(parquets)
        b = sum(f.stat().st_size for f in parquets)
        total_archivos += n
        total_bytes += b
        por_tabla[tabla_dir.name] = {
            "archivos": n,
            "total_mib": round(b / (1024 * 1024), 3),
            "tamanio_medio_mib": round((b / n) / (1024 * 1024), 3) if n else 0.0,
        }

    return {
        "total_archivos": total_archivos,
        "total_mib": round(total_bytes / (1024 * 1024), 3),
        "tamanio_medio_mib": round((total_bytes / total_archivos) / (1024 * 1024), 3) if total_archivos else 0.0,
        "tablas": por_tabla,
    }


def medir_lectura_mes(spark: SparkSession, ruta_parquet: Path, col_valor: str, reps: int = 5) -> dict:
    """Mide lectura de un mes especifico (2023-08) con 1 calentamiento + N repeticiones."""
    def leer_mes():
        df = spark.read.parquet(str(ruta_parquet))
        return (
            df.filter(
                (F.col("anio") == 2023)
                & (F.col("fecha_hora") >= F.lit("2023-08-01 00:30:00").cast("timestamp"))
                & (F.col("fecha_hora") <= F.lit("2023-09-01 00:00:00").cast("timestamp"))
            )
            .agg(F.count("*").alias("n"), F.avg(col_valor).alias("prom"))
            .collect()[0]
        )

    res = leer_mes()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        leer_mes()
        tiempos.append(round(time.perf_counter() - t0, 4))

    return {
        "mes_evaluado": "2023-08",
        "filas_leidas": int(res["n"]),
        "repeticiones": reps,
        "tiempos_s": tiempos,
        "mediana_s": round(statistics.median(tiempos), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", default="coes_pipeline/lake_real")
    parser.add_argument("--fase", required=True, choices=["antes", "despues"])
    parser.add_argument("--out", default="evidencias/benchmarks/compactacion_anual.json")
    args = parser.parse_args()

    lake = Path(args.lake)
    out_path = Path(args.out)

    spark = (
        SparkSession.builder.appName(f"benchmark-compactacion-{args.fase}")
        .master("local[8]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    silver_inv = inventariar_capa(lake / "silver")
    gold_inv = inventariar_capa(lake / "gold")
    tot_files = silver_inv["total_archivos"] + gold_inv["total_archivos"]
    tot_mib = round(silver_inv["total_mib"] + gold_inv["total_mib"], 3)

    medicion = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "particionamiento": "anio/mes" if args.fase == "antes" else "anio",
        "resumen_global": {
            "total_archivos_parquet": tot_files,
            "total_mib": tot_mib,
            "tamanio_medio_mib": round(tot_mib / tot_files, 3) if tot_files else 0.0,
        },
        "silver": silver_inv,
        "gold": gold_inv,
        "lectura_mes_especifico": {
            "silver_a2_despacho_ejecutado": medir_lectura_mes(
                spark, lake / "silver" / "a2_despacho_ejecutado", "potencia_ejecutada_mw"
            ),
            "gold_precios_barra_30min": medir_lectura_mes(
                spark, lake / "gold" / "precios_barra_30min", "costo_marginal_limpio"
            ),
        },
    }
    spark.stop()

    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    payload[args.fase] = medicion

    if "antes" in payload and "despues" in payload:
        a = payload["antes"]
        d = payload["despues"]
        n_a = a["resumen_global"]["total_archivos_parquet"]
        n_d = d["resumen_global"]["total_archivos_parquet"]
        payload["comparativa"] = {
            "reduccion_archivos_pct": round(100 * (n_a - n_d) / n_a, 2) if n_a else 0.0,
            "archivos_silver_antes_vs_despues": [a["silver"]["total_archivos"], d["silver"]["total_archivos"]],
            "archivos_gold_antes_vs_despues": [a["gold"]["total_archivos"], d["gold"]["total_archivos"]],
            "tamanio_medio_mib_antes_vs_despues": [
                a["resumen_global"]["tamanio_medio_mib"],
                d["resumen_global"]["tamanio_medio_mib"],
            ],
            "mediana_lectura_a2_s_antes_vs_despues": [
                a["lectura_mes_especifico"]["silver_a2_despacho_ejecutado"]["mediana_s"],
                d["lectura_mes_especifico"]["silver_a2_despacho_ejecutado"]["mediana_s"],
            ],
            "mediana_lectura_precios_s_antes_vs_despues": [
                a["lectura_mes_especifico"]["gold_precios_barra_30min"]["mediana_s"],
                d["lectura_mes_especifico"]["gold_precios_barra_30min"]["mediana_s"],
            ],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload[args.fase]["resumen_global"], indent=2))
    print(json.dumps(payload[args.fase]["lectura_mes_especifico"], indent=2))


if __name__ == "__main__":
    main()