"""Tarea 3.3: Ajuste bayesiano a priori (Saerens) y recalibracion mensual ante Prior Shift."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def ajustar_probabilidad_saerens(probs, pi_train: float, pi_target: float) -> np.ndarray:
    """Aplica ajuste a priori de Saerens et al. manteniendo probabilidades en (0, 1)."""
    eps = 1e-6
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    pi_tr = float(np.clip(pi_train, eps, 1.0 - eps))
    pi_tg = float(np.clip(pi_target, eps, 1.0 - eps))

    factor_pos = pi_tg / pi_tr
    factor_neg = (1.0 - pi_tg) / (1.0 - pi_tr)
    num = factor_pos * p
    den = num + factor_neg * (1.0 - p)
    p_adj = num / den
    return np.clip(p_adj, eps, 1.0 - eps)


def _seleccionar_umbral_ventana(
    y_true: np.ndarray,
    probs_eval: np.ndarray,
    n_dias: int,
    max_fp_semana: float | None = None,
) -> tuple[float, dict]:
    semanas = max(1, n_dias) / 7.0
    grilla = np.round(np.arange(0.05, 0.951, 0.01), 2)

    mejor = None
    fallback = None
    for tau in grilla:
        pred = (probs_eval >= tau).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fp_sem = fp / semanas

        candidato = {
            "tau": float(tau),
            "f1": float(f1),
            "recall": float(rec),
            "precision": float(prec),
            "fp_semana": float(fp_sem),
        }

        if max_fp_semana is None:
            clave = (f1, rec, prec, -tau)
            if mejor is None or clave > mejor[0]:
                mejor = (clave, candidato)
        else:
            if fp_sem <= max_fp_semana + 1e-12:
                clave = (rec, f1, prec, tau)
                if mejor is None or clave > mejor[0]:
                    mejor = (clave, candidato)
            clave_fb = (-fp_sem, rec, f1, tau)
            if fallback is None or clave_fb > fallback[0]:
                fallback = (clave_fb, candidato)

    elegido = mejor[1] if mejor is not None else fallback[1]
    return elegido["tau"], elegido


def calibrar_umbral_ventana_movil(
    df_hist_preds: pd.DataFrame,
    fecha_inicio_mes,
    ventana_dias: int = 60,
    max_fp_semana: float | None = None,
    pi_train: float | None = None,
) -> dict:
    """Calibra pi_reciente y umbral tau usando solo los ventana_dias previos a fecha_inicio_mes."""
    fecha_corte = pd.Timestamp(fecha_inicio_mes)
    fecha_ini = fecha_corte - pd.Timedelta(days=int(ventana_dias))

    df_h = df_hist_preds.copy()
    df_h["_fecha_desp"] = pd.to_datetime(df_h["fecha_despacho"])
    solo_pasado = df_h[(df_h["_fecha_desp"] < fecha_corte) & df_h["y_true"].notna()].sort_values("_fecha_desp")
    if len(solo_pasado) == 0:
        raise ValueError(f"Sin observaciones historicas previas a {fecha_corte.date()} para calibracion.")

    ventana = solo_pasado[solo_pasado["_fecha_desp"] >= fecha_ini]
    # Si la ventana calendario tiene muy pocos bloques observables (<48), respaldar con los ultimos disponibles
    if len(ventana) < 48:
        ventana = solo_pasado.tail(int(ventana_dias) * 48)

    y_true = ventana["y_true"].to_numpy(dtype=int)
    probs_raw = ventana["y_prob"].to_numpy(dtype=float)
    pi_reciente = float(np.mean(y_true))

    if pi_train is not None:
        probs_eval = ajustar_probabilidad_saerens(probs_raw, pi_train=pi_train, pi_target=pi_reciente)
    else:
        probs_eval = probs_raw

    n_dias = int(ventana["_fecha_desp"].dt.strftime("%Y-%m-%d").nunique())
    if y_true.min() == y_true.max():
        tau_opt = 0.50
        pred = (probs_eval >= tau_opt).astype(int)
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        stats = {
            "tau": tau_opt,
            "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
            "recall": 0.0,
            "precision": 0.0,
            "fp_semana": fp / (max(1, n_dias) / 7.0),
        }
    else:
        tau_opt, stats = _seleccionar_umbral_ventana(
            y_true=y_true,
            probs_eval=probs_eval,
            n_dias=n_dias,
            max_fp_semana=max_fp_semana,
        )

    return {
        "pi_reciente": round(pi_reciente, 6),
        "tau": round(float(tau_opt), 4),
        "n_obs_hist": int(len(ventana)),
        "n_dias_hist": n_dias,
        "fp_semana_hist": round(float(stats["fp_semana"]), 4),
        "f1_hist": round(float(stats["f1"]), 6),
        "recall_hist": round(float(stats["recall"]), 6),
    }


def aplicar_calibracion_mensual_2024(
    df_val_2023: pd.DataFrame,
    df_test_2024: pd.DataFrame,
    pi_train: float,
    ventana_dias: int = 60,
    max_fp_semana: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Recalcula pi_reciente y tau al inicio de cada mes de 2024 sin usar datos del mes ni futuros."""
    hist_2023 = df_val_2023[["fecha_despacho", "y_true", "y_prob"]].copy()
    test_base = df_test_2024.copy()
    test_base["_ym"] = pd.to_datetime(test_base["fecha_despacho"]).dt.strftime("%Y-%m")
    hist_2024 = test_base[["fecha_despacho", "y_true", "y_prob"]].copy()
    hist_total = pd.concat([hist_2023, hist_2024], ignore_index=True)

    meses = sorted(test_base["_ym"].unique())
    partes = []
    trazabilidad = []

    for ym in meses:
        fecha_inicio_mes = pd.Timestamp(f"{ym}-01 00:00:00")
        cal = calibrar_umbral_ventana_movil(
            df_hist_preds=hist_total,
            fecha_inicio_mes=fecha_inicio_mes,
            ventana_dias=ventana_dias,
            max_fp_semana=max_fp_semana,
            pi_train=pi_train,
        )

        bloque_mes = test_base[test_base["_ym"] == ym].copy()
        prob_rec = ajustar_probabilidad_saerens(
            bloque_mes["y_prob"].to_numpy(dtype=float),
            pi_train=pi_train,
            pi_target=cal["pi_reciente"],
        )
        tau_mes = float(cal["tau"])
        bloque_mes["prob_recalibrada"] = prob_rec
        bloque_mes["umbral_aplicado"] = tau_mes
        bloque_mes["y_pred_recalibrado"] = (prob_rec >= tau_mes).astype(int)
        partes.append(bloque_mes)

        trazabilidad.append(
            {
                "anio_mes": ym,
                "pi_reciente": cal["pi_reciente"],
                "tau_mes": tau_mes,
                "n_obs_hist": cal["n_obs_hist"],
                "n_dias_hist": cal["n_dias_hist"],
                "fp_semana_hist": cal["fp_semana_hist"],
                "n_bloques_mes": int(len(bloque_mes)),
                "n_alertas_mes": int(bloque_mes["y_pred_recalibrado"].sum()),
            }
        )

    out_df = pd.concat(partes, ignore_index=False).sort_index() if partes else test_base.copy()
    out_df = out_df.drop(columns=["_ym"], errors="ignore")
    return out_df, trazabilidad
