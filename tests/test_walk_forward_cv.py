"""Pruebas unitarias de validacion Walk-Forward, preprocesamiento sin fuga y bootstrap (Tarea 3.2)."""
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, functions as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.walk_forward_cv import (
    WALK_FORWARD_FOLDS,
    build_preprocessing_stages,
    dividir_fold_con_embargo,
    evaluar_predicciones_con_ic,
    filtrar_sin_meses_baja_cobertura_2024,
)


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    session = (
        SparkSession.builder.appName("pytest-walk-forward-cv")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_embargo_estricto_4h(spark):
    # Grilla continua de 30 min alrededor de todos los limites de corte 2022-12 a 2024-12
    grid = (
        spark.sql(
            """
            SELECT explode(sequence(
                timestamp('2022-12-01 00:30:00'),
                timestamp('2024-12-01 00:00:00'),
                interval 30 minutes
            )) AS fecha_hora
            """
        )
        .withColumn("fecha_despacho", F.to_date(F.expr("fecha_hora - interval 1 minute")))
        .withColumn("y_ventana_2_4h", F.lit(0))
        .cache()
    )

    for fold in WALK_FORWARD_FOLDS:
        train_df, val_df, _ = dividir_fold_con_embargo(
            grid,
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            target_col="y_ventana_2_4h",
        )
        max_train = train_df.agg(F.max("fecha_hora")).first()[0]
        min_val = val_df.agg(F.min("fecha_hora")).first()[0]

        assert max_train is not None and min_val is not None
        assert max_train <= min_val - dt.timedelta(hours=4)
        # El horizonte futuro del ultimo bloque de train [t+2h, t+4h] cierra antes de min(val)
        assert max_train + dt.timedelta(hours=4) < min_val

    grid.unpersist()


def test_preprocesamiento_fit_solo_en_train(spark):
    train_rows = [
        (1, 10.0, 100.0),
        (1, 20.0, 200.0),
        (0, 30.0, 300.0),
        (1, 40.0, 400.0),
        (0, 50.0, 500.0),
        (1, None, None),
    ]
    val_normal_rows = [
        (1, 15.0, 150.0),
        (0, None, 250.0),
    ]
    val_extremo_rows = [
        (1, 1e9, 1e9),
        (0, None, 1e9),
    ]
    cols = ["flag_a1_disponible_t", "brecha_term_cara_mw", "gen_total_mw"]
    train_df = spark.createDataFrame(train_rows, cols)
    val_normal = spark.createDataFrame(val_normal_rows, cols)
    val_extremo = spark.createDataFrame(val_extremo_rows, cols)

    stages_1 = build_preprocessing_stages(feature_cols=cols, scale=True)
    stages_2 = build_preprocessing_stages(feature_cols=cols, scale=True)

    # Fit siempre exclusivamente sobre train_df
    model_1 = Pipeline(stages=stages_1).fit(train_df)
    model_2 = Pipeline(stages=stages_2).fit(train_df)

    # Verificar que Imputer y StandardScaler producen parametros identicos sin tocar val
    imp_1 = model_1.stages[0].surrogateDF.collect()[0].asDict()
    imp_2 = model_2.stages[0].surrogateDF.collect()[0].asDict()
    assert imp_1 == imp_2
    assert imp_1["brecha_term_cara_mw"] == pytest.approx(30.0)
    assert imp_1["gen_total_mw"] == pytest.approx(300.0)

    # Transformar val_extremo no altera las estadisticas aprendidas ni las predicciones sobre val_normal
    scaler_mean = np.array(model_1.stages[2].mean.toArray())
    scaler_std = np.array(model_1.stages[2].std.toArray())
    _ = model_2.transform(val_extremo).collect()

    assert np.allclose(scaler_mean, model_2.stages[2].mean.toArray())
    assert np.allclose(scaler_std, model_2.stages[2].std.toArray())

    # flag_a1_disponible_t permanece intacta tras la transformacion
    out_val = model_1.transform(val_normal).select("flag_a1_disponible_t", "brecha_term_cara_mw__imp").collect()
    assert [r["flag_a1_disponible_t"] for r in out_val] == [1, 0]
    assert out_val[1]["brecha_term_cara_mw__imp"] == pytest.approx(30.0)


def test_block_bootstrap_reproducible_y_valido():
    rng = np.random.default_rng(123)
    fechas = []
    y_true = []
    y_prob = []
    base = dt.date(2024, 2, 1)
    for d in range(40):
        dia = base + dt.timedelta(days=d)
        for _ in range(12):
            fechas.append(dia)
            prob = float(rng.uniform(0.05, 0.95))
            yt = int(rng.random() < prob)
            y_true.append(yt)
            y_prob.append(prob)

    pdf = pd.DataFrame({"fecha_despacho": fechas, "y_true": y_true, "y_prob": y_prob})
    res_a = evaluar_predicciones_con_ic(pdf, threshold=0.5, n_boot=300, seed=42)
    res_b = evaluar_predicciones_con_ic(pdf, threshold=0.5, n_boot=300, seed=42)

    assert res_a == res_b
    for nombre, item in res_a["metricas"].items():
        val = item["valor"]
        ic = item["ic_95"]
        assert val is not None and ic is not None, f"Metrica faltante: {nombre}"
        assert ic[0] <= val <= ic[1], f"Fuera de IC en {nombre}: {ic[0]} <= {val} <= {ic[1]}"


def test_filtro_meses_baja_cobertura_2024(spark):
    fechas = [dt.date(2024, m, 15) for m in range(1, 12)]
    rows = [(f, f.strftime("%Y-%m")) for f in fechas]

    # Version Pandas
    pdf = pd.DataFrame(rows, columns=["fecha_despacho", "ym"])
    pdf_filt = filtrar_sin_meses_baja_cobertura_2024(pdf)
    assert set(pdf_filt["ym"]) == {f"2024-{m:02d}" for m in range(1, 12) if m not in (1, 4)}
    assert len(pdf_filt) == 9

    # Version Spark
    sdf = spark.createDataFrame(rows, ["fecha_despacho", "ym"])
    sdf_filt = filtrar_sin_meses_baja_cobertura_2024(sdf)
    meses_spark = {r["ym"] for r in sdf_filt.collect()}
    assert meses_spark == {f"2024-{m:02d}" for m in range(1, 12) if m not in (1, 4)}
    assert sdf_filt.count() == 9
