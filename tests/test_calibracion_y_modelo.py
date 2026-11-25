"""Pruebas unitarias de calibracion ante Prior Shift y reglas antifuga de entrenamiento (Tarea 3.3)."""
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession, functions as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.calibracion_umbral import (
    ajustar_probabilidad_saerens,
    aplicar_calibracion_mensual_2024,
)
from src.ml.entrenamiento_modelo import SELECTION_FOLDS, agregar_pesos_clase


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.appName("pytest-calibracion-modelo")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_saerens_ajusta_probabilidades_direccion_correcta():
    probs = np.array([0.05, 0.20, 0.50, 0.80, 0.95])
    adj = ajustar_probabilidad_saerens(probs, pi_train=0.33, pi_target=0.12)

    assert np.all(adj > 0.0) and np.all(adj < 1.0)
    # Al bajar la prevalencia esperada (pi_target < pi_train), las probabilidades disminuyen
    assert np.all(adj < probs)
    # Preservacion estricta del orden monotonico
    assert np.all(np.diff(adj) > 0.0)


def test_calibracion_mensual_sin_fuga_futura():
    rng = np.random.default_rng(7)
    # Historia 2023-11 y 2023-12 (60 dias antes de 2024-01)
    fechas_2023 = pd.date_range("2023-11-01", "2023-12-31", freq="D")
    rows_2023 = []
    for d in fechas_2023:
        for _ in range(4):
            p = float(rng.uniform(0.05, 0.90))
            rows_2023.append({"fecha_despacho": d.date(), "y_true": int(rng.random() < p), "y_prob": p})
    df_2023 = pd.DataFrame(rows_2023)

    # Test 2024: enero, febrero y marzo
    fechas_2024 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    rows_2024 = []
    for d in fechas_2024:
        for _ in range(4):
            p = float(rng.uniform(0.05, 0.90))
            rows_2024.append({"fecha_despacho": d.date(), "y_true": int(rng.random() < p), "y_prob": p})
    df_2024_orig = pd.DataFrame(rows_2024)

    # Perturbar radicalmente etiquetas de febrero y marzo 2024
    df_2024_pert = df_2024_orig.copy()
    mask_feb_mar = pd.to_datetime(df_2024_pert["fecha_despacho"]).dt.month.isin([2, 3])
    df_2024_pert.loc[mask_feb_mar, "y_true"] = 1 - df_2024_pert.loc[mask_feb_mar, "y_true"]

    _, traz_orig = aplicar_calibracion_mensual_2024(df_2023, df_2024_orig, pi_train=0.25, ventana_dias=60)
    _, traz_pert = aplicar_calibracion_mensual_2024(df_2023, df_2024_pert, pi_train=0.25, ventana_dias=60)

    por_mes_orig = {t["anio_mes"]: t for t in traz_orig}
    por_mes_pert = {t["anio_mes"]: t for t in traz_pert}

    # Ni enero ni febrero deben cambiar cuando se modifican febrero/marzo
    assert por_mes_orig["2024-01"]["tau_mes"] == por_mes_pert["2024-01"]["tau_mes"]
    assert por_mes_orig["2024-01"]["pi_reciente"] == por_mes_pert["2024-01"]["pi_reciente"]
    assert por_mes_orig["2024-02"]["tau_mes"] == por_mes_pert["2024-02"]["tau_mes"]
    assert por_mes_orig["2024-02"]["pi_reciente"] == por_mes_pert["2024-02"]["pi_reciente"]


def test_pesos_de_clase_balanceados(spark):
    rows = [(1,) for _ in range(15)] + [(0,) for _ in range(85)]
    df = spark.createDataFrame(rows, ["y_ventana_2_4h"])
    weighted, info = agregar_pesos_clase(df, target_col="y_ventana_2_4h")

    sumas = (
        weighted.groupBy("y_ventana_2_4h")
        .agg(F.sum("class_weight").alias("suma_pesos"))
        .collect()
    )
    por_clase = {r["y_ventana_2_4h"]: r["suma_pesos"] for r in sumas}
    assert por_clase[1] == pytest.approx(por_clase[0], rel=1e-6)
    assert por_clase[1] == pytest.approx(50.0, rel=1e-6)
    assert info["n_pos"] == 15 and info["n_neg"] == 85


def test_seleccion_no_usa_folds_2024():
    assert set(SELECTION_FOLDS) == {"fold_1_2023_s1", "fold_2_2023_s2"}
    assert "fold_3_2024_h1" not in SELECTION_FOLDS
    assert "test_final_2024" not in SELECTION_FOLDS
