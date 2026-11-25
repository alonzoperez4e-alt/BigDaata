"""Pruebas unitarias de backtest economico, PSI y contratos ML hacia Fase 4 (Tarea 3.5)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.backtest_economico import (
    calcular_escenarios_backtest,
    calcular_psi,
)


def test_psi_identico_y_desplazado():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=5000)

    # Distribucion identica -> PSI < 0.01
    psi_identico = calcular_psi(ref, ref.copy(), n_bins=10)
    assert psi_identico < 0.01

    # Distribucion desplazada en +2 desviaciones estandar -> PSI > 0.20
    curr_desplazado = rng.normal(loc=2.0, scale=1.0, size=5000)
    psi_desplazado = calcular_psi(ref, curr_desplazado, n_bins=10)
    assert psi_desplazado > 0.20


def test_backtest_monotonia_alpha():
    exp_tot = 19_079_554.64
    exp_ant = 15_966_327.02
    sob_tot = 15_223_546.20
    sob_ant = 12_321_451.23

    esc = calcular_escenarios_backtest(
        exposicion_bruta_anticipados=exp_ant,
        exposicion_bruta_total=exp_tot,
        sobrecosto_neto_anticipados=sob_ant,
        sobrecosto_neto_total=sob_tot,
        n_dias_observables=281,
    )
    por_alpha = {e["alpha"]: e for e in esc}
    assert por_alpha[0.05]["ahorro_bruto_soles"] < por_alpha[0.10]["ahorro_bruto_soles"] < por_alpha[0.15]["ahorro_bruto_soles"]
    assert por_alpha[0.05]["ahorro_neto_soles"] < por_alpha[0.10]["ahorro_neto_soles"] < por_alpha[0.15]["ahorro_neto_soles"]
    for e in esc:
        assert 0.0 < e["ahorro_bruto_soles"] <= exp_tot
        assert 0.0 < e["ahorro_neto_soles"] <= sob_tot


def test_contrato_predicciones_alerta_30min():
    ruta = REPO_ROOT / "lake_v2" / "gold" / "ml_outputs" / "predicciones_alerta_30min"
    if not ruta.exists():
        pytest.skip("lake_v2/gold/ml_outputs/predicciones_alerta_30min no materializado aun.")

    df = pd.read_parquet(ruta)
    assert len(df) == 16080
    assert df["fecha_hora"].nunique() == 16080

    cols_req = {
        "fecha_hora",
        "fecha_despacho",
        "anio",
        "model_version_id",
        "horizonte",
        "prob_cruda",
        "prob_recalibrada",
        "umbral_f1_60d",
        "alerta_f1_60d",
        "resultado_f1_60d",
        "umbral_max_fp3",
        "alerta_max_fp3",
        "resultado_max_fp3",
        "shap_top1",
        "shap_top2",
        "shap_top3",
    }
    assert cols_req.issubset(df.columns)
    assert set(df["resultado_f1_60d"]).issubset({"TP", "FP", "TN", "FN", "NO_VERIFICABLE"})
    assert set(df["resultado_max_fp3"]).issubset({"TP", "FP", "TN", "FN", "NO_VERIFICABLE"})
    for col in ("shap_top1", "shap_top2", "shap_top3"):
        assert df[col].notna().all()


def test_contrato_monitoreo_deriva_mensual():
    ruta_psi = REPO_ROOT / "lake_v2" / "gold" / "ml_outputs" / "monitoreo_deriva_mensual"
    ruta_reg = REPO_ROOT / "lake_v2" / "gold" / "ml_outputs" / "registro_modelos.json"
    if not ruta_psi.exists() or not ruta_reg.exists():
        pytest.skip("Contratos de monitoreo o registro no materializados aun.")

    df = pd.read_parquet(ruta_psi)
    assert len(df) == 11
    assert list(df["anio_mes"]) == [f"2024-{m:02d}" for m in range(1, 12)]

    por_mes = df.set_index("anio_mes")
    assert bool(por_mes.loc["2024-01", "alerta_baja_cobertura"]) is True
    assert bool(por_mes.loc["2024-04", "alerta_baja_cobertura"]) is True
    assert set(df["semaforo_deriva"]).issubset({"VERDE", "AMBAR", "ROJO"})
    assert set(df["estado_operativo_modelo"]).issubset({"ACTIVO_ALERTA", "MODO_MONITOREO_DESCRIPTIVO"})

    reg = json.loads(ruta_reg.read_text(encoding="utf-8"))
    for k in (
        "model_version_id",
        "algoritmo",
        "hiperparametros",
        "fecha_corte_entrenamiento",
        "n_features",
        "features",
        "top5_shap",
        "regimen_operativo_recomendado",
        "metricas_2024_global",
        "hash_esquema_features",
    ):
        assert k in reg
    assert reg["n_features"] == len(reg["features"]) == 24
