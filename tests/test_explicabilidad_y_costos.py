"""Pruebas unitarias de explicabilidad SHAP y analisis economico FP vs FN (Tarea 3.4)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.explicabilidad_y_costos import (
    analizar_regimen_economico,
    entrenar_surrogate_sklearn,
    extraer_shap_clase_positiva,
    formatear_top3_shap_local,
)


def test_aditividad_shap_en_muestra():
    rng = np.random.default_rng(42)
    cols = ["f1", "f2", "f3", "f4", "flag_a1_disponible_t"]
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=cols[:4])
    X["flag_a1_disponible_t"] = rng.integers(0, 2, size=120).astype(float)
    y = ((X["f1"] + 0.8 * X["f2"] - 0.5 * X["f3"]) > 0.2).astype(int).to_numpy()

    clf, _ = entrenar_surrogate_sklearn(X, y)
    explainer = shap.TreeExplainer(clf)
    X_sample = X.iloc[:15].copy()
    base_val, sv = extraer_shap_clase_positiva(explainer, X_sample)

    probs = clf.predict_proba(X_sample)[:, 1]
    reconstruido = base_val + np.sum(sv, axis=1)
    assert np.allclose(reconstruido, probs, atol=1e-4)


def test_formato_top3_shap_local():
    cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    sv = np.array(
        [
            [0.012, -0.345, 0.120, 0.005],
            [-0.800, 0.050, -0.210, 0.400],
        ]
    )
    t1, t2, t3 = formatear_top3_shap_local(sv, cols)

    assert len(t1) == 2 and len(t2) == 2 and len(t3) == 2
    assert all(isinstance(x, str) and len(x) > 0 for x in [*t1, *t2, *t3])

    # Fila 0: |-0.345| > |0.120| > |0.012|
    assert t1[0] == "feat_b (-0.345)"
    assert t2[0] == "feat_c (+0.120)"
    assert t3[0] == "feat_a (+0.012)"

    # Fila 1: |-0.800| > |0.400| > |-0.210|
    assert t1[1] == "feat_a (-0.800)"
    assert t2[1] == "feat_d (+0.400)"
    assert t3[1] == "feat_c (-0.210)"


def test_metricas_fp_fn_y_episodios():
    # Serie continua de 28 dias (4 semanas exactas) -> 1344 bloques de 30 min
    ts = pd.date_range("2024-02-01 00:30:00", periods=1344, freq="30min")
    df = pd.DataFrame(
        {
            "fecha_hora": ts,
            "fecha_despacho": (ts - pd.Timedelta(minutes=1)).date,
            "y_true_evento": np.zeros(1344, dtype=float),
            "evento_termico_nop": np.zeros(1344, dtype=float),
            "exposicion_bruta_soles": np.zeros(1344, dtype=float),
            "sobrecosto_neto_estimado_soles": np.zeros(1344, dtype=float),
            "pred_test": np.zeros(1344, dtype=int),
        }
    )

    # Episodio 1: bloques 20..22 (anticipado por alerta en bloque 14 = T_ini - 3h)
    df.loc[20:22, "evento_termico_nop"] = 1.0
    df.loc[20:22, "exposicion_bruta_soles"] = 1000.0
    df.loc[20:22, "sobrecosto_neto_estimado_soles"] = 400.0
    df.loc[14, "y_true_evento"] = 1.0
    df.loc[14, "pred_test"] = 1

    # Episodio 2: bloques 100..101 (sin alertas en [92..96] ni 100 -> omitido FN)
    df.loc[100:101, "evento_termico_nop"] = 1.0
    df.loc[100:101, "exposicion_bruta_soles"] = 2500.0
    df.loc[100:101, "sobrecosto_neto_estimado_soles"] = 1100.0

    # Falsas alarmas: 4 bloques FP formando 2 episodios FP (200..201 y 300..301) en 4 semanas
    df.loc[[200, 201, 300, 301], "pred_test"] = 1

    res = analizar_regimen_economico(df, pred_col="pred_test", nombre_regimen="sintetico")
    assert res["episodios_reales_2024"]["episodios_totales"] == 2
    assert res["episodios_reales_2024"]["episodios_anticipados"] == 1
    assert res["episodios_reales_2024"]["episodios_omitidos_fn"] == 1
    assert res["fatiga_operador"]["dias_observables_2024"] == 28
    assert res["fatiga_operador"]["fp_por_semana"] == pytest.approx(1.0)
    assert res["fatiga_operador"]["episodios_fp_totales"] == 2
    assert res["fatiga_operador"]["episodios_fp_por_semana"] == pytest.approx(0.5)
    assert res["exposicion_economica_soles"]["exposicion_bruta_anticipados"] == pytest.approx(3000.0)
    assert res["exposicion_economica_soles"]["exposicion_bruta_omitidos_fn"] == pytest.approx(5000.0)


def test_contrato_salida_parquet_shap(tmp_path):
    # Verifica contrato sobre predicciones_alerta_30min_shap.parquet si ya fue materializado, o sobre muestra sintetica
    ruta_real = REPO_ROOT / "lake_v2" / "gold" / "ml_outputs" / "predicciones_alerta_30min_shap.parquet"
    if ruta_real.exists():
        df = pd.read_parquet(ruta_real)
        assert len(df) == 16080
        assert df["fecha_hora"].nunique() == 16080
        for col in ("shap_top1", "shap_top2", "shap_top3"):
            assert col in df.columns
            assert df[col].notna().all()
    else:
        ts = pd.date_range("2024-01-01 00:30:00", periods=16080, freq="30min")
        sv = np.ones((16080, 4)) * np.array([0.4, -0.3, 0.2, 0.1])
        t1, t2, t3 = formatear_top3_shap_local(sv, ["a", "b", "c", "d"])
        df = pd.DataFrame({"fecha_hora": ts, "shap_top1": t1, "shap_top2": t2, "shap_top3": t3})
        out = tmp_path / "shap.parquet"
        df.to_parquet(out, index=False)
        leido = pd.read_parquet(out)
        assert len(leido) == 16080
        assert leido["fecha_hora"].nunique() == 16080
        assert all(leido[c].notna().all() for c in ("shap_top1", "shap_top2", "shap_top3"))
