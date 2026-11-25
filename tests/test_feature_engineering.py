"""Pruebas unitarias de Feature Engineering sin fuga (Tarea 3.1)."""
import datetime as dt
import os
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession, functions as F, types as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.feature_engineering import (
    FORBIDDEN_LEAKAGE_COLS,
    PREDICTOR_COLS,
    construir_features_ml,
)

SCHEMA_BASE = T.StructType(
    [
        T.StructField("fecha_hora", T.TimestampType(), False),
        T.StructField("fecha_despacho", T.DateType(), False),
        T.StructField("anio", T.IntegerType(), False),
        T.StructField("gen_total_mw", T.DoubleType(), True),
        T.StructField("prog_total_mw", T.DoubleType(), True),
        T.StructField("brecha_despacho_mw", T.DoubleType(), True),
        T.StructField("term_cara_ejec_mw", T.DoubleType(), True),
        T.StructField("term_cara_prog_mw", T.DoubleType(), True),
        T.StructField("term_cara_share", T.DoubleType(), True),
        T.StructField("energia_nop_mwh", T.DoubleType(), True),
        T.StructField("exposicion_bruta_soles", T.DoubleType(), True),
        T.StructField("sobrecosto_neto_estimado_soles", T.DoubleType(), True),
        T.StructField("cmg_troncal_mediana", T.DoubleType(), True),
        T.StructField("cmg_troncal_p90", T.DoubleType(), True),
        T.StructField("cmg_spread_p90_p50", T.DoubleType(), True),
        T.StructField("n_barras_troncales_validas", T.IntegerType(), True),
        T.StructField("n_forzados_termicos_activos", T.IntegerType(), True),
        T.StructField("n_forzados_transmision_activos", T.IntegerType(), True),
        T.StructField("n_forzados_termicos_inicio_3h", T.IntegerType(), True),
        T.StructField("n_mantenimientos_prog_termicos_activos", T.IntegerType(), True),
        T.StructField("tiene_a1", T.BooleanType(), False),
        T.StructField("tiene_a2", T.BooleanType(), False),
        T.StructField("tiene_a3", T.BooleanType(), False),
        T.StructField("tiene_pr06", T.BooleanType(), False),
        T.StructField("etiqueta_observable", T.BooleanType(), False),
        T.StructField("motivo_censura", T.StringType(), False),
        T.StructField("evento_termico_nop", T.IntegerType(), True),
        T.StructField("inicio_termico_nop", T.IntegerType(), True),
        T.StructField("n_centrales_term_nop", T.LongType(), True),
    ]
)


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.appName("pytest-feature-engineering")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _crear_serie(
    spark: SparkSession,
    n_bloques: int,
    inicio: dt.datetime = dt.datetime(2022, 12, 31, 18, 0, 0),
    overrides: dict[int, dict] | None = None,
    saltar_indices: set[int] | None = None,
):
    overrides = overrides or {}
    saltar_indices = saltar_indices or set()
    filas = []
    for i in range(n_bloques):
        if i in saltar_indices:
            continue
        ts = inicio + dt.timedelta(minutes=30 * i)
        fecha_desp = (ts - dt.timedelta(minutes=1)).date()
        base = {
            "fecha_hora": ts,
            "fecha_despacho": fecha_desp,
            "anio": fecha_desp.year,
            "gen_total_mw": 5000.0 + 10.0 * i,
            "prog_total_mw": 4950.0 + 9.0 * i,
            "brecha_despacho_mw": 50.0 + 1.0 * i,
            "term_cara_ejec_mw": 100.0 + 2.0 * i,
            "term_cara_prog_mw": 80.0 + 1.5 * i,
            "term_cara_share": 0.02 + 0.0001 * i,
            "energia_nop_mwh": 10.0 if i % 5 == 0 else 0.0,
            "exposicion_bruta_soles": 5000.0 if i % 5 == 0 else 0.0,
            "sobrecosto_neto_estimado_soles": 2000.0 if i % 5 == 0 else 0.0,
            "cmg_troncal_mediana": 0.15 + 0.005 * i,
            "cmg_troncal_p90": 0.18 + 0.006 * i,
            "cmg_spread_p90_p50": 0.03 + 0.001 * i,
            "n_barras_troncales_validas": 70,
            "n_forzados_termicos_activos": i % 3,
            "n_forzados_transmision_activos": i % 2,
            "n_forzados_termicos_inicio_3h": 1 if i % 4 == 0 else 0,
            "n_mantenimientos_prog_termicos_activos": 5,
            "tiene_a1": True,
            "tiene_a2": True,
            "tiene_a3": True,
            "tiene_pr06": True,
            "etiqueta_observable": True,
            "motivo_censura": "OK",
            "evento_termico_nop": 1 if i % 5 == 0 else 0,
            "inicio_termico_nop": 1 if i % 5 == 0 else 0,
            "n_centrales_term_nop": 1 if i % 5 == 0 else 0,
        }
        base.update(overrides.get(i, {}))
        filas.append(tuple(base[f.name] for f in SCHEMA_BASE.fields))
    return spark.createDataFrame(filas, schema=SCHEMA_BASE)


@pytest.mark.parametrize("col_prohibida", ["label_rdo", "alfa"])
def test_rechaza_columnas_prohibidas(spark, col_prohibida):
    assert col_prohibida in FORBIDDEN_LEAKAGE_COLS
    df = _crear_serie(spark, 10).withColumn(col_prohibida, F.lit(1.0))
    with pytest.raises(ValueError, match="prohibidas"):
        construir_features_ml(df)


def test_cero_fuga_futura_perturbacion(spark):
    n_bloques = 30
    idx_corte = 19  # T0 inclusive: bloques 0..19; futuro: bloques 20..29
    df_orig = _crear_serie(spark, n_bloques)

    perturbaciones = {}
    for i in range(idx_corte + 1, n_bloques):
        perturbaciones[i] = {
            "gen_total_mw": 99999.0 + i * 500.0,
            "prog_total_mw": 10.0,
            "brecha_despacho_mw": 88888.0,
            "term_cara_ejec_mw": 5000.0,
            "term_cara_prog_mw": 0.0,
            "term_cara_share": 0.99,
            "cmg_troncal_mediana": 9.99,
            "cmg_troncal_p90": 15.99,
            "cmg_spread_p90_p50": 6.00,
            "n_forzados_termicos_activos": 99,
            "n_forzados_transmision_activos": 99,
            "n_forzados_termicos_inicio_3h": 99,
            "n_mantenimientos_prog_termicos_activos": 99,
            "tiene_a1": False,
            "etiqueta_observable": False,
            "motivo_censura": "SIN_A1",
            "evento_termico_nop": None,
            "inicio_termico_nop": None,
            "n_centrales_term_nop": None,
        }
    df_pert = _crear_serie(spark, n_bloques, overrides=perturbaciones)

    fecha_corte = dt.datetime(2022, 12, 31, 18, 0, 0) + dt.timedelta(minutes=30 * idx_corte)
    res_orig = (
        construir_features_ml(df_orig)
        .filter(F.col("fecha_hora") <= F.lit(fecha_corte))
        .orderBy("fecha_hora")
        .select("fecha_hora", *PREDICTOR_COLS)
        .collect()
    )
    res_pert = (
        construir_features_ml(df_pert)
        .filter(F.col("fecha_hora") <= F.lit(fecha_corte))
        .orderBy("fecha_hora")
        .select("fecha_hora", *PREDICTOR_COLS)
        .collect()
    )

    assert len(res_orig) == idx_corte + 1
    assert len(res_pert) == idx_corte + 1
    for r1, r2 in zip(res_orig, res_pert):
        assert r1["fecha_hora"] == r2["fecha_hora"]
        for col in PREDICTOR_COLS:
            v1, v2 = r1[col], r2[col]
            if v1 is None or v2 is None:
                assert v1 is None and v2 is None, f"Diferencia de nulidad en {col} para {r1['fecha_hora']}"
            elif isinstance(v1, float):
                assert v1 == pytest.approx(v2, rel=1e-12, abs=1e-12), f"Fuga en {col} para {r1['fecha_hora']}"
            else:
                assert v1 == v2, f"Fuga en {col} para {r1['fecha_hora']}"


def test_regla_censura_ventana_2_4h(spark):
    # Evaluamos en t=0 (ventana futura: bloques 4, 5, 6, 7, 8) cuatro casos aislados
    def _evaluar_caso(ventana_specs: list[tuple[bool, int | None, int | None]]):
        overrides = {}
        for offset, (obs, ev, ini) in enumerate(ventana_specs, start=4):
            overrides[offset] = {
                "etiqueta_observable": obs,
                "tiene_a1": obs,
                "motivo_censura": "OK" if obs else "SIN_A1",
                "evento_termico_nop": ev,
                "inicio_termico_nop": ini,
                "n_centrales_term_nop": ev,
            }
        df = _crear_serie(spark, 9, overrides=overrides)
        return construir_features_ml(df).orderBy("fecha_hora").first()

    # Caso 1: 0 observables -> None
    r1 = _evaluar_caso([(False, None, None)] * 5)
    assert r1["n_obs_ventana_2_4h"] == 0
    assert r1["y_ventana_2_4h"] is None
    assert r1["y_inicio_ventana_2_4h"] is None

    # Caso 2: 2 observables en 0 -> None
    r2 = _evaluar_caso(
        [(True, 0, 0), (True, 0, 0), (False, None, None), (False, None, None), (False, None, None)]
    )
    assert r2["n_obs_ventana_2_4h"] == 2
    assert r2["y_ventana_2_4h"] is None
    assert r2["y_inicio_ventana_2_4h"] is None

    # Caso 3: 3 observables en 0 -> 0
    r3 = _evaluar_caso(
        [(True, 0, 0), (True, 0, 0), (True, 0, 0), (False, None, None), (False, None, None)]
    )
    assert r3["n_obs_ventana_2_4h"] == 3
    assert r3["y_ventana_2_4h"] == 0
    assert r3["y_inicio_ventana_2_4h"] == 0

    # Caso 4: 1 observable en 1 con 4 censurados -> 1
    r4 = _evaluar_caso(
        [(False, None, None), (False, None, None), (True, 1, 1), (False, None, None), (False, None, None)]
    )
    assert r4["n_obs_ventana_2_4h"] == 1
    assert r4["y_ventana_2_4h"] == 1
    assert r4["y_inicio_ventana_2_4h"] == 1


def test_inmunidad_rezagos_ante_huecos(spark):
    # Eliminamos bloques 4, 5, 6, 7 (hueco de 2 horas: 20:00..21:30)
    inicio = dt.datetime(2023, 5, 1, 18, 0, 0)
    overrides = {
        i: {"evento_termico_nop": 0, "inicio_termico_nop": 0}
        for i in range(16)
    }
    # Ponemos evento=1 en los bloques 8 y 9 (justo despues del hueco, fuera de [t+2h, t+4h] para t=0?):
    # Para t=0 (18:00), [t+2h, t+4h] = [20:00, 22:00] = indices 4, 5, 6, 7, 8.
    # Como 4..7 estan borrados, en [20:00, 22:00] solo queda el indice 8 (22:00)!
    # Si se usara filas (t+4..t+8), t=0 veria bloques 8,9,10,11,12 (5 obs -> 0).
    # Usando tiempo real (rangeBetween), t=0 solo ve el indice 8 (1 obs -> None si vale 0).
    df = _crear_serie(spark, 16, inicio=inicio, overrides=overrides, saltar_indices={4, 5, 6, 7})
    out = construir_features_ml(df).orderBy("fecha_hora").collect()
    por_ts = {row["fecha_hora"]: row for row in out}

    # Inmediatamente despues del hueco: indice 8 (22:00) no tiene t-1h (21:00 = indice 6, borrado)
    ts_post_hueco = inicio + dt.timedelta(minutes=30 * 8)
    assert por_ts[ts_post_hueco]["rampa_cmg_1h"] is None

    # Bloque siguiente (indice 9, 22:30) tampoco tiene t-1h (21:30 = indice 7, borrado)
    ts_post_hueco_2 = inicio + dt.timedelta(minutes=30 * 9)
    assert por_ts[ts_post_hueco_2]["rampa_cmg_1h"] is None

    # Bloque indice 10 (23:00) si tiene t-1h (22:00 = indice 8)
    ts_con_lag = inicio + dt.timedelta(minutes=30 * 10)
    assert por_ts[ts_con_lag]["rampa_cmg_1h"] is not None

    # Ventana futura por tiempo real en t=0 (18:00): solo indice 8 cae en [20:00, 22:00] -> 1 obs en 0 -> None
    assert por_ts[inicio]["n_obs_ventana_2_4h"] == 1
    assert por_ts[inicio]["y_ventana_2_4h"] is None
