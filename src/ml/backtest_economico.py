"""Tarea 3.5: Backtest economico 2024, monitoreo PSI y contratos oficiales hacia Fase 4."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.explicabilidad_y_costos import analizar_regimen_economico
from src.ml.feature_engineering import PREDICTOR_COLS
from src.ml.walk_forward_cv import seleccionar_umbral_f1

MODEL_VERSION_ID = "rf_v1_2024_walkforward"
TOP5_SHAP_COLS = (
    "ratio_term_ejec_vs_prog",
    "term_cara_share",
    "cmg_troncal_p90_t",
    "cmg_troncal_mediana_t",
    "persistencia_term_nop_2h",
)
ALPHAS_MITIGACION = (
    ("conservador", 0.05),
    ("base", 0.10),
    ("optimista", 0.15),
)


def calcular_escenarios_backtest(
    exposicion_bruta_anticipados: float,
    exposicion_bruta_total: float,
    sobrecosto_neto_anticipados: float,
    sobrecosto_neto_total: float,
    n_dias_observables: int,
    alphas: tuple[tuple[str, float], ...] = ALPHAS_MITIGACION,
) -> list[dict]:
    """Calcula ahorros bruto, neto y diario bajo tasas explicitas de mitigacion operativa."""
    escenarios = []
    for nombre_alpha, alpha in alphas:
        ahorro_bruto = alpha * exposicion_bruta_anticipados
        ahorro_neto = alpha * sobrecosto_neto_anticipados
        pct_bruto = (100.0 * ahorro_bruto / exposicion_bruta_total) if exposicion_bruta_total > 0 else 0.0
        pct_neto = (100.0 * ahorro_neto / sobrecosto_neto_total) if sobrecosto_neto_total > 0 else 0.0
        ahorro_dia = ahorro_bruto / max(1, n_dias_observables)
        escenarios.append(
            {
                "escenario": nombre_alpha,
                "alpha": float(alpha),
                "ahorro_bruto_soles": round(ahorro_bruto, 2),
                "pct_ahorro_sobre_exposicion_bruta": round(pct_bruto, 2),
                "ahorro_neto_soles": round(ahorro_neto, 2),
                "pct_ahorro_sobre_sobrecosto_neto": round(pct_neto, 2),
                "ahorro_bruto_soles_por_dia_observable": round(ahorro_dia, 2),
                "cumple_meta_2a4_pct": bool(pct_bruto >= 2.0 and pct_neto >= 2.0),
            }
        )
    return escenarios


def calcular_psi(ref_values, curr_values, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Calcula el Population Stability Index (PSI) por cuantiles de referencia con suavizado eps."""
    ref = np.asarray(ref_values, dtype=float)
    curr = np.asarray(curr_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    if ref.size == 0 or curr.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        val = float(edges[0]) if edges.size == 1 else float(ref[0])
        edges = np.array([-np.inf, val - 1e-9, val + 1e-9, np.inf], dtype=float)
    else:
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    curr_counts, _ = np.histogram(curr, bins=edges)

    ref_pct = (ref_counts / ref.size) + eps
    curr_pct = (curr_counts / curr.size) + eps
    ref_pct = ref_pct / ref_pct.sum()
    curr_pct = curr_pct / curr_pct.sum()

    psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    return max(0.0, psi)


def _clasificar_resultado(y_true_series: pd.Series, pred_series: pd.Series) -> list[str]:
    out = []
    for yt, pr in zip(y_true_series, pred_series):
        if pd.isna(yt):
            out.append("NO_VERIFICABLE")
        elif int(yt) == 1 and int(pr) == 1:
            out.append("TP")
        elif int(yt) == 0 and int(pr) == 1:
            out.append("FP")
        elif int(yt) == 0 and int(pr) == 0:
            out.append("TN")
        else:
            out.append("FN")
    return out


def _calcular_prob_baseline_horaria_2024(df_feat: pd.DataFrame, df_2024: pd.DataFrame) -> np.ndarray:
    corte_train = pd.Timestamp("2024-01-01 00:30:00") - pd.Timedelta(hours=4)
    tr = df_feat[
        (pd.to_datetime(df_feat["fecha_hora"]) < corte_train) & df_feat["y_ventana_2_4h"].notna()
    ].copy()
    tr_ts = pd.to_datetime(tr["fecha_hora"])
    tr["hora_bloque"] = tr_ts.dt.hour * 2 + (tr_ts.dt.minute // 30)
    prev_tr = float(tr["y_ventana_2_4h"].mean())
    tasas = tr.groupby("hora_bloque")["y_ventana_2_4h"].mean().to_dict()

    ts24 = pd.to_datetime(df_2024["fecha_hora"])
    hb24 = ts24.dt.hour * 2 + (ts24.dt.minute // 30)
    return hb24.map(tasas).fillna(prev_tr).to_numpy(dtype=float)


def construir_predicciones_alerta_30min(df_shap: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla contractual predicciones_alerta_30min (16,080 filas, 0 duplicados)."""
    df = df_shap.sort_values("fecha_hora").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "fecha_hora": pd.to_datetime(df["fecha_hora"]),
            "fecha_despacho": pd.to_datetime(df["fecha_despacho"]).dt.date,
            "anio": df["anio"].astype("int32"),
            "model_version_id": MODEL_VERSION_ID,
            "horizonte": "2h_4h",
            "prob_cruda": df["prob_raw_evento"].astype(float),
            "prob_recalibrada": df["prob_recal_evento"].astype(float),
            "umbral_f1_60d": df["umbral_recal_evento"].astype(float),
            "alerta_f1_60d": df["pred_recal_evento"].astype("int32"),
            "resultado_f1_60d": _clasificar_resultado(df["y_true_evento"], df["pred_recal_evento"]),
            "umbral_max_fp3": df["umbral_recal_maxfp3_evento"].astype(float),
            "alerta_max_fp3": df["pred_recal_maxfp3_evento"].astype("int32"),
            "resultado_max_fp3": _clasificar_resultado(df["y_true_evento"], df["pred_recal_maxfp3_evento"]),
            "shap_top1": df["shap_top1"].astype(str),
            "shap_top2": df["shap_top2"].astype(str),
            "shap_top3": df["shap_top3"].astype(str),
        }
    )
    return out


def construir_monitoreo_deriva_mensual(
    df_feat: pd.DataFrame,
    df_shap: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla contractual monitoreo_deriva_mensual para los 11 meses de 2024."""
    df_ref = df_feat[pd.to_datetime(df_feat["fecha_hora"]) < pd.Timestamp("2024-01-01 00:30:00")].copy()

    df24 = df_shap.sort_values("fecha_hora").reset_index(drop=True).copy()
    df24["anio_mes"] = pd.to_datetime(df24["fecha_despacho"]).dt.strftime("%Y-%m")
    df24["prob_baseline_horaria"] = _calcular_prob_baseline_horaria_2024(df_feat, df24)

    filas = []
    for ym in sorted(df24["anio_mes"].unique()):
        mes_df = df24[df24["anio_mes"] == ym]
        tot = int(len(mes_df))
        obs_mask = mes_df["y_true_evento"].notna()
        mes_obs = mes_df[obs_mask]
        et = int(len(mes_obs))
        pct_cob = round(100.0 * et / tot, 2) if tot > 0 else 0.0
        alerta_baja_cob = bool(pct_cob < 50.0)

        if et > 0:
            yt = mes_obs["y_true_evento"].astype(int).to_numpy()
            prev = float(np.mean(yt))
            dos_clases = yt.min() != yt.max()
            pr_mod = (
                float(average_precision_score(yt, mes_obs["prob_recal_evento"].to_numpy(dtype=float)))
                if dos_clases
                else None
            )
            pr_base = (
                float(average_precision_score(yt, mes_obs["prob_baseline_horaria"].to_numpy(dtype=float)))
                if dos_clases
                else None
            )
            f1_mod = float(f1_score(yt, mes_obs["pred_recal_evento"].astype(int), pos_label=1, zero_division=0))
            f1_base = float(f1_score(yt, mes_obs["pred_baseline_horaria"].astype(int), pos_label=1, zero_division=0))

            fp_mes = int(np.sum((yt == 0) & (mes_obs["pred_recal_evento"].astype(int).to_numpy() == 1)))
            dias_obs = int(pd.to_datetime(mes_obs["fecha_despacho"]).dt.strftime("%Y-%m-%d").nunique())
            fp_sem = fp_mes / max(1e-9, dias_obs / 7.0)
        else:
            prev = None
            pr_mod = None
            pr_base = None
            f1_mod = None
            f1_base = None
            fp_sem = 0.0

        psis = {}
        for col in TOP5_SHAP_COLS:
            psis[f"psi_{col}"] = round(calcular_psi(df_ref[col].to_numpy(), mes_df[col].to_numpy(), n_bins=10), 6)
        psi_max = round(max(psis.values()), 6)

        if psi_max < 0.10:
            semaforo = "VERDE"
        elif psi_max < 0.20:
            semaforo = "AMBAR"
        else:
            semaforo = "ROJO"

        modelo_supera_base = (pr_mod is not None) and (pr_base is not None) and (pr_mod > pr_base)
        estado_op = "ACTIVO_ALERTA" if (modelo_supera_base and not alerta_baja_cob) else "MODO_MONITOREO_DESCRIPTIVO"

        filas.append(
            {
                "anio_mes": ym,
                "model_version_id": MODEL_VERSION_ID,
                "bloques_totales": tot,
                "bloques_etiquetables": et,
                "pct_cobertura_etiqueta": pct_cob,
                "alerta_baja_cobertura": alerta_baja_cob,
                "prevalencia_observada": round(prev, 6) if prev is not None else None,
                "pr_auc_modelo": round(pr_mod, 6) if pr_mod is not None else None,
                "pr_auc_baseline": round(pr_base, 6) if pr_base is not None else None,
                "f1_modelo_recal_f1": round(f1_mod, 6) if f1_mod is not None else None,
                "f1_baseline": round(f1_base, 6) if f1_base is not None else None,
                "fp_por_semana_recal_f1": round(fp_sem, 2),
                **psis,
                "psi_max_top5": psi_max,
                "semaforo_deriva": semaforo,
                "estado_operativo_modelo": estado_op,
            }
        )

    return pd.DataFrame(filas)


def construir_registro_modelos(
    comparativa: dict,
    explicabilidad: dict,
) -> dict:
    """Construye registro_modelos.json con gobernanza, metricas y hash de esquema."""
    target_ev = comparativa["targets"]["y_ventana_2_4h"]
    ganador = target_ev["modelo_ganador"]
    eval_global = target_ev["test_final_2024"]["2024_global"]
    features_ordenadas = sorted(PREDICTOR_COLS)
    hash_features = hashlib.sha256(json.dumps(features_ordenadas).encode("utf-8")).hexdigest()

    def _resumir_metricas(reg_dict: dict) -> dict:
        m = reg_dict["metricas"]
        return {
            k: m[k] for k in ("pr_auc", "auc_roc", "brier_score", "precision", "recall", "f1") if k in m
        }

    return {
        "model_version_id": MODEL_VERSION_ID,
        "target_principal": "y_ventana_2_4h",
        "algoritmo": f"PySpark MLlib {ganador['familia']}",
        "hiperparametros": ganador["params"],
        "fecha_corte_entrenamiento": "2023-12-31",
        "embargo_horas": 4,
        "n_features": len(PREDICTOR_COLS),
        "features": list(PREDICTOR_COLS),
        "hash_esquema_features": hash_features,
        "top5_shap": explicabilidad["shap_global_2024"]["top5"],
        "regimen_operativo_recomendado": "recalibrado_60d_f1",
        "regimen_operativo_alternativo": "recalibrado_60d_max_fp_3_sem",
        "metricas_2024_global": {
            "recalibrado_60d_f1": _resumir_metricas(eval_global["recalibrado_mensual_60d"]),
            "recalibrado_60d_max_fp_3_sem": _resumir_metricas(eval_global["recalibrado_mensual_60d_maxfp3"]),
            "umbral_fijo_2023_s2": _resumir_metricas(eval_global["umbral_fijo_2023_s2"]),
            "baseline_tasa_horaria": _resumir_metricas(eval_global["baseline_tasa_horaria"]),
        },
    }


def _escribir_parquet_directorio(df: pd.DataFrame, dir_path: Path, filename: str = "part-00000.snappy.parquet") -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    # Limpiar parquets previos en el directorio para mantener snapshot atomico limpio
    for old in dir_path.glob("*.parquet"):
        old.unlink()
    out_file = dir_path / filename
    df.to_parquet(out_file, compression="snappy", index=False)
    return str(dir_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest economico, PSI y contratos Fase 4 (Tarea 3.5)")
    parser.add_argument("--features-input", default="lake_v2/gold/ml_features_30min")
    parser.add_argument("--feature-store-input", default="lake_v2/gold/gold_feature_store_30min")
    parser.add_argument("--preds-shap-input", default="lake_v2/gold/ml_outputs/predicciones_alerta_30min_shap.parquet")
    parser.add_argument("--comparativa-json", default="evidencias_modelo/comparativa_modelos.json")
    parser.add_argument("--explicabilidad-json", default="evidencias_modelo/explicabilidad_y_costos.json")
    parser.add_argument("--output-dir", default="lake_v2/gold/ml_outputs")
    parser.add_argument("--summary-out", default="evidencias_modelo/backtest_economico_2024.json")
    args = parser.parse_args()

    df_feat = pd.read_parquet(args.features_input)
    df_feat["fecha_hora"] = pd.to_datetime(df_feat["fecha_hora"])

    df_shap = pd.read_parquet(args.preds_shap_input)
    df_shap["fecha_hora"] = pd.to_datetime(df_shap["fecha_hora"])
    df_shap = df_shap.sort_values("fecha_hora").reset_index(drop=True)

    comparativa = json.loads(Path(args.comparativa_json).read_text(encoding="utf-8"))
    explicabilidad = json.loads(Path(args.explicabilidad_json).read_text(encoding="utf-8"))

    # ---- BLOQUE A: Backtest economico auditable sobre 2024 ----
    regimenes_spec = (
        ("recalibrado_60d_f1", "pred_recal_evento", True, "Regimen recomendado balanceado (F1)"),
        ("recalibrado_60d_max_fp_3_sem", "pred_recal_maxfp3_evento", True, "Regimen conservador bajo fatiga <=3 FP/sem"),
        (
            "umbral_fijo_2023_s2",
            "pred_fijo_evento",
            False,
            "Referencia de contraste: inviable por fatiga operativa (101.81 FP/semana)",
        ),
    )

    backtest_regimenes = {}
    for nombre, pred_col, viable, nota in regimenes_spec:
        aud = analizar_regimen_economico(df_shap, pred_col=pred_col, nombre_regimen=nombre)
        exp = aud["exposicion_economica_soles"]
        n_dias = aud["fatiga_operador"]["dias_observables_2024"]
        escenarios = calcular_escenarios_backtest(
            exposicion_bruta_anticipados=exp["exposicion_bruta_anticipados"],
            exposicion_bruta_total=exp["exposicion_bruta_total_2024"],
            sobrecosto_neto_anticipados=exp["sobrecosto_neto_anticipados"],
            sobrecosto_neto_total=exp["sobrecosto_neto_total_2024"],
            n_dias_observables=n_dias,
        )
        backtest_regimenes[nombre] = {
            "viable_operativamente": viable,
            "nota_viabilidad": nota,
            "fp_por_semana": aud["fatiga_operador"]["fp_por_semana"],
            "episodios_fp_por_semana": aud["fatiga_operador"]["episodios_fp_por_semana"],
            "episodios_reales": aud["episodios_reales_2024"],
            "exposicion_base_soles": exp,
            "escenarios_mitigacion": escenarios,
        }

    backtest_report = {
        "anio_evaluado": 2024,
        "meta_objetivo_especifico_3_pct": [2.0, 4.0],
        "dias_observables_2024": 281,
        "regimenes": backtest_regimenes,
    }
    sum_path = Path(args.summary_out)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(backtest_report, indent=2), encoding="utf-8")

    # ---- BLOQUE B: Tres artefactos contractuales hacia Fase 4 ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pred_contrato = construir_predicciones_alerta_30min(df_shap)
    pred_dir = _escribir_parquet_directorio(df_pred_contrato, out_dir / "predicciones_alerta_30min")

    df_psi = construir_monitoreo_deriva_mensual(df_feat, df_shap)
    psi_dir = _escribir_parquet_directorio(df_psi, out_dir / "monitoreo_deriva_mensual")

    registro = construir_registro_modelos(comparativa, explicabilidad)
    reg_path = out_dir / "registro_modelos.json"
    reg_path.write_text(json.dumps(registro, indent=2), encoding="utf-8")

    print(f"[OK] Backtest economico guardado en: {sum_path}")
    print(f"[OK] Contrato predicciones_alerta_30min: {pred_dir} ({len(df_pred_contrato)} filas)")
    print(f"[OK] Contrato monitoreo_deriva_mensual: {psi_dir} ({len(df_psi)} meses)")
    print(f"[OK] Contrato registro_modelos.json: {reg_path}")


if __name__ == "__main__":
    main()
