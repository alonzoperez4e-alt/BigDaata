"""Tarea 3.2: Validacion temporal rodante Walk-Forward, preprocesamiento sin fuga y lineas base."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml.feature import Imputer, StandardScaler, SQLTransformer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.feature_engineering import PREDICTOR_COLS

EMBARGO_HORAS = 4
MESES_BAJA_COBERTURA_2024 = ("2024-01", "2024-04")
TARGETS_OFICIALES = ("y_ventana_2_4h", "y_inicio_ventana_2_4h")

WALK_FORWARD_FOLDS = (
    {
        "nombre": "fold_1_2023_s1",
        "descripcion": "2023 S1",
        "val_start": "2023-01-01 00:30:00",
        "val_end": "2023-06-30 23:30:00",
    },
    {
        "nombre": "fold_2_2023_s2",
        "descripcion": "2023 S2 (regimen extremo 24.7%)",
        "val_start": "2023-07-01 00:30:00",
        "val_end": "2023-12-31 23:30:00",
    },
    {
        "nombre": "fold_3_2024_h1",
        "descripcion": "2024 H1 (ene-may con ene/abr < 50% cobertura)",
        "val_start": "2024-01-01 00:30:00",
        "val_end": "2024-05-31 23:30:00",
    },
    {
        "nombre": "test_final_2024",
        "descripcion": "Evaluacion oficial ciega 2024",
        "val_start": "2024-01-01 00:30:00",
        "val_end": "2024-12-01 00:00:00",
    },
)


def crear_spark(app_name: str = "coes-walk-forward-cv") -> SparkSession:
    """Inicializa SparkSession en UTC con configuracion segura para Windows."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def dividir_fold_con_embargo(
    df: DataFrame,
    val_start: str,
    val_end: str,
    target_col: str | None = None,
    embargo_horas: int = EMBARGO_HORAS,
) -> tuple[DataFrame, DataFrame, dict]:
    """Divide df en train/val aplicando embargo estricto de 4h y retorna estadisticas de cobertura."""
    train_raw = df.filter(F.col("fecha_hora") < F.expr(f"timestamp('{val_start}') - interval {embargo_horas} hours"))
    val_raw = df.filter(
        (F.col("fecha_hora") >= F.lit(val_start).cast("timestamp"))
        & (F.col("fecha_hora") <= F.lit(val_end).cast("timestamp"))
    )

    if target_col is not None:
        train_df = train_raw.filter(F.col(target_col).isNotNull())
        val_df = val_raw.filter(F.col(target_col).isNotNull())
    else:
        train_df = train_raw
        val_df = val_raw

    stats = {}
    for split_name, raw_df, filt_df in (("train", train_raw, train_df), ("val", val_raw, val_df)):
        tot = raw_df.count()
        if target_col is not None:
            agg = filt_df.agg(
                F.count("*").alias("etiquetables"),
                F.sum(F.when(F.col(target_col) == 1, 1).otherwise(0)).alias("positivos"),
                F.min("fecha_hora").alias("min_ts"),
                F.max("fecha_hora").alias("max_ts"),
            ).first()
        else:
            agg = filt_df.agg(
                F.count("*").alias("etiquetables"),
                F.lit(0).alias("positivos"),
                F.min("fecha_hora").alias("min_ts"),
                F.max("fecha_hora").alias("max_ts"),
            ).first()
        et = int(agg["etiquetables"] or 0)
        pos = int(agg["positivos"] or 0)
        stats[split_name] = {
            "bloques_totales": int(tot),
            "bloques_etiquetables": et,
            "pct_cobertura": round(100.0 * et / tot, 2) if tot > 0 else 0.0,
            "positivos": pos,
            "prevalencia": round(pos / et, 6) if et > 0 else None,
            "min_fecha_hora": agg["min_ts"].isoformat() if agg["min_ts"] is not None else None,
            "max_fecha_hora": agg["max_ts"].isoformat() if agg["max_ts"] is not None else None,
        }

    return train_df, val_df, stats


def generar_folds_walk_forward(
    df: DataFrame,
    target_col: str,
    folds: tuple[dict, ...] = WALK_FORWARD_FOLDS,
):
    """Itera los 4 cortes cronologicos expansivos con embargo de 4 horas."""
    for fold in folds:
        train_df, val_df, stats = dividir_fold_con_embargo(
            df,
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            target_col=target_col,
        )
        yield fold, train_df, val_df, stats


def build_preprocessing_stages(
    feature_cols: list[str] | None = None,
    scale: bool = False,
) -> list:
    """Construye etapas de pyspark.ml.Pipeline para ajustar (fit) solo sobre train_df.

    - Imputer(strategy='median') sobre columnas numericas excepto 'flag_a1_disponible_t'.
    - VectorAssembler -> 'features_raw'.
    - StandardScaler(withMean=True, withStd=True) -> 'features' si scale=True;
      si scale=False, expone alias 'features' = 'features_raw' para uniformidad en modelos de arboles.
    """
    cols = list(feature_cols) if feature_cols is not None else list(PREDICTOR_COLS)
    impute_cols = [c for c in cols if c != "flag_a1_disponible_t"]
    imputed_map = {c: f"{c}__imp" for c in impute_cols}
    assembled_cols = [imputed_map.get(c, c) for c in cols]

    stages = []
    if impute_cols:
        stages.append(
            Imputer(
                strategy="median",
                inputCols=impute_cols,
                outputCols=[imputed_map[c] for c in impute_cols],
            )
        )

    stages.append(VectorAssembler(inputCols=assembled_cols, outputCol="features_raw", handleInvalid="error"))

    if scale:
        stages.append(
            StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withMean=True,
                withStd=True,
            )
        )
    else:
        stages.append(SQLTransformer(statement="SELECT *, features_raw AS features FROM __THIS__"))

    return stages


def filtrar_sin_meses_baja_cobertura_2024(df, date_col: str = "fecha_despacho"):
    """Excluye enero 2024 y abril 2024 (<50% cobertura) sin alterar el resto de meses."""
    if isinstance(df, pd.DataFrame):
        ym = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")
        return df[~ym.isin(MESES_BAJA_COBERTURA_2024)].copy()
    return df.filter(~F.date_format(F.col(date_col), "yyyy-MM").isin(*MESES_BAJA_COBERTURA_2024))


def _calcular_metricas_puntuales(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    dos_clases = (y_true.size > 0) and (y_true.min() != y_true.max())
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)) if dos_clases else None,
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if dos_clases else None,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "prevalencia": float(np.mean(y_true)),
    }


def evaluar_predicciones_con_ic(
    df,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
    date_col: str = "fecha_despacho",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Calcula metricas de clasificacion e IC 95% mediante block bootstrap por dia operativo."""
    pdf = df.toPandas() if isinstance(df, DataFrame) else df.copy()
    if len(pdf) == 0:
        raise ValueError("No hay filas para evaluar en evaluar_predicciones_con_ic.")

    y_true = pdf[y_true_col].to_numpy(dtype=int)
    y_prob = pdf[y_prob_col].to_numpy(dtype=float)
    thr_arr = pdf["threshold"].to_numpy(dtype=float) if "threshold" in pdf.columns else float(threshold)
    y_pred = (y_prob >= thr_arr).astype(int)

    puntuales = _calcular_metricas_puntuales(y_true, y_prob, y_pred)

    # Agrupar indices por dia operativo ordenado para reproducibilidad deterministica
    fechas = pd.to_datetime(pdf[date_col]).dt.strftime("%Y-%m-%d").to_numpy()
    orden_dias = sorted(np.unique(fechas))
    grupos_idx = [np.flatnonzero(fechas == d) for d in orden_dias]
    n_dias = len(grupos_idx)

    rng = np.random.default_rng(seed)
    boot_vals: dict[str, list[float]] = {k: [] for k in puntuales}
    n_boot_validos_auc = 0

    for _ in range(n_boot):
        muestra_dias = rng.integers(0, n_dias, size=n_dias)
        idx = np.concatenate([grupos_idx[d] for d in muestra_dias])
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        yh_b = y_pred[idx]

        # Calculo rapido de metricas siempre definidas
        tp = int(np.sum((yt_b == 1) & (yh_b == 1)))
        fp = int(np.sum((yt_b == 0) & (yh_b == 1)))
        fn = int(np.sum((yt_b == 1) & (yh_b == 0)))
        prec_b = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_b = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_b = (2.0 * prec_b * rec_b / (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0.0
        brier_b = float(np.mean((yp_b - yt_b) ** 2))
        prev_b = float(np.mean(yt_b))

        boot_vals["brier_score"].append(brier_b)
        boot_vals["precision"].append(prec_b)
        boot_vals["recall"].append(rec_b)
        boot_vals["f1"].append(f1_b)
        boot_vals["prevalencia"].append(prev_b)

        if yt_b.min() != yt_b.max():
            n_boot_validos_auc += 1
            boot_vals["pr_auc"].append(float(average_precision_score(yt_b, yp_b)))
            boot_vals["auc_roc"].append(float(roc_auc_score(yt_b, yp_b)))

    resumen = {
        "n_obs": int(len(y_true)),
        "n_positivos": int(np.sum(y_true == 1)),
        "n_dias": int(n_dias),
        "n_boot": int(n_boot),
        "n_boot_validos_auc": int(n_boot_validos_auc),
        "threshold": round(float(threshold), 6) if np.isscalar(thr_arr) else "por_fila",
        "metricas": {},
    }
    for k, val in puntuales.items():
        arr = boot_vals[k]
        ic = [round(float(np.percentile(arr, 2.5)), 6), round(float(np.percentile(arr, 97.5)), 6)] if arr else None
        resumen["metricas"][k] = {
            "valor": round(val, 6) if val is not None else None,
            "ic_95": ic,
        }
    return resumen


def seleccionar_umbral_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Selecciona el umbral que maximiza F1 de forma deterministica."""
    if y_true.size == 0 or y_true.min() == y_true.max():
        return 0.5
    candidatos = np.unique(np.concatenate([np.round(y_prob, 6), np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])]))
    mejor_f1 = -1.0
    mejor_thr = 0.5
    for thr in candidatos:
        pred = (y_prob >= thr).astype(int)
        score = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        if score > mejor_f1:
            mejor_f1 = score
            mejor_thr = float(thr)
    return mejor_thr


def evaluar_baselines_fold(
    train_pdf: pd.DataFrame,
    val_pdf: pd.DataFrame,
    target_col: str,
    optimize_threshold_on: str = "train",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Evalua baseline_tasa_horaria y baseline_siempre_positivo sobre un split."""
    tr = train_pdf.copy()
    va = val_pdf.copy()
    tr["hora_bloque"] = tr["fecha_hora"].dt.hour * 2 + (tr["fecha_hora"].dt.minute // 30)
    va["hora_bloque"] = va["fecha_hora"].dt.hour * 2 + (va["fecha_hora"].dt.minute // 30)

    prev_train = float(tr[target_col].mean())
    tasas_bloque = tr.groupby("hora_bloque")[target_col].mean().to_dict()

    tr_prob_hor = tr["hora_bloque"].map(tasas_bloque).fillna(prev_train).to_numpy(dtype=float)
    va_prob_hor = va["hora_bloque"].map(tasas_bloque).fillna(prev_train).to_numpy(dtype=float)

    y_tr = tr[target_col].to_numpy(dtype=int)
    y_va = va[target_col].to_numpy(dtype=int)

    if optimize_threshold_on == "val":
        thr_hor = seleccionar_umbral_f1(y_va, va_prob_hor)
    else:
        thr_hor = seleccionar_umbral_f1(y_tr, tr_prob_hor)

    df_eval_hor = pd.DataFrame(
        {
            "fecha_despacho": va["fecha_despacho"],
            "y_true": y_va,
            "y_prob": va_prob_hor,
        }
    )
    res_hor = evaluar_predicciones_con_ic(
        df_eval_hor,
        threshold=thr_hor,
        n_boot=n_boot,
        seed=seed,
    )

    df_eval_pos = pd.DataFrame(
        {
            "fecha_despacho": va["fecha_despacho"],
            "y_true": y_va,
            "y_prob": np.full(len(va), prev_train, dtype=float),
        }
    )
    res_pos = evaluar_predicciones_con_ic(
        df_eval_pos,
        threshold=0.0,
        n_boot=n_boot,
        seed=seed,
    )
    res_pos["probabilidad_constante_train"] = round(prev_train, 6)

    return {
        "baseline_tasa_horaria": res_hor,
        "baseline_siempre_positivo": res_pos,
    }


def construir_tabla_mensual_2024(df: DataFrame) -> list[dict]:
    """Construye tabla mensual 2024 de cobertura y prevalencia para ambos targets."""
    m24 = (
        df.filter(F.year("fecha_despacho") == 2024)
        .withColumn("anio_mes", F.date_format("fecha_despacho", "yyyy-MM"))
        .groupBy("anio_mes")
        .agg(
            F.count("*").alias("bloques_totales"),
            F.sum(F.when(F.col("y_ventana_2_4h").isNotNull(), 1).otherwise(0)).alias("obs_ev"),
            F.sum(F.when(F.col("y_ventana_2_4h") == 1, 1).otherwise(0)).alias("pos_ev"),
            F.sum(F.when(F.col("y_inicio_ventana_2_4h").isNotNull(), 1).otherwise(0)).alias("obs_ini"),
            F.sum(F.when(F.col("y_inicio_ventana_2_4h") == 1, 1).otherwise(0)).alias("pos_ini"),
        )
        .orderBy("anio_mes")
        .collect()
    )
    tabla = []
    for r in m24:
        tot = int(r["bloques_totales"])
        obs_ev = int(r["obs_ev"] or 0)
        pos_ev = int(r["pos_ev"] or 0)
        obs_ini = int(r["obs_ini"] or 0)
        pos_ini = int(r["pos_ini"] or 0)
        tabla.append(
            {
                "anio_mes": r["anio_mes"],
                "baja_cobertura": r["anio_mes"] in MESES_BAJA_COBERTURA_2024,
                "bloques_totales": tot,
                "y_ventana_2_4h": {
                    "bloques_etiquetables": obs_ev,
                    "pct_cobertura": round(100.0 * obs_ev / tot, 2) if tot > 0 else 0.0,
                    "positivos": pos_ev,
                    "prevalencia": round(pos_ev / obs_ev, 6) if obs_ev > 0 else None,
                },
                "y_inicio_ventana_2_4h": {
                    "bloques_etiquetables": obs_ini,
                    "pct_cobertura": round(100.0 * obs_ini / tot, 2) if tot > 0 else 0.0,
                    "positivos": pos_ini,
                    "prevalencia": round(pos_ini / obs_ini, 6) if obs_ini > 0 else None,
                },
            }
        )
    return tabla


def ejecutar_auditoria_walk_forward(
    spark: SparkSession,
    input_path: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Ejecuta la auditoria completa de folds y lineas base sobre ml_features_30min."""
    df = spark.read.parquet(input_path).cache()

    resultado = {
        "input": input_path,
        "embargo_horas": EMBARGO_HORAS,
        "meses_baja_cobertura_excluidos": list(MESES_BAJA_COBERTURA_2024),
        "n_boot": n_boot,
        "seed": seed,
        "tabla_mensual_2024": construir_tabla_mensual_2024(df),
        "resultados_por_target": {},
    }

    for target_col in TARGETS_OFICIALES:
        folds_out = {}
        for fold, train_df, val_df, stats in generar_folds_walk_forward(df, target_col):
            train_pdf = train_df.select("fecha_hora", "fecha_despacho", target_col).toPandas()
            val_pdf = val_df.select("fecha_hora", "fecha_despacho", target_col).toPandas()
            train_pdf["fecha_hora"] = pd.to_datetime(train_pdf["fecha_hora"])
            val_pdf["fecha_hora"] = pd.to_datetime(val_pdf["fecha_hora"])

            baselines = evaluar_baselines_fold(
                train_pdf,
                val_pdf,
                target_col=target_col,
                optimize_threshold_on="train",
                n_boot=n_boot,
                seed=seed,
            )

            entry = {
                "descripcion": fold["descripcion"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "splits": stats,
                "baselines": baselines,
            }

            if fold["nombre"] == "test_final_2024":
                val_sin_baja = filtrar_sin_meses_baja_cobertura_2024(val_pdf)
                entry["sensibilidad_cobertura_2024"] = {
                    "2024_global": baselines,
                    "2024_sin_meses_baja_cobertura": evaluar_baselines_fold(
                        train_pdf,
                        val_sin_baja,
                        target_col=target_col,
                        optimize_threshold_on="train",
                        n_boot=n_boot,
                        seed=seed,
                    ),
                }

            folds_out[fold["nombre"]] = entry

        resultado["resultados_por_target"][target_col] = folds_out

    df.unpersist()
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria Walk-Forward CV y lineas base (Tarea 3.2)")
    parser.add_argument("--input", required=True, help="Ruta a gold/ml_features_30min")
    parser.add_argument("--out-json", required=True, help="Ruta de salida para JSON de auditoria")
    parser.add_argument("--n-boot", type=int, default=1000, help="Iteraciones block bootstrap diario")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    spark = crear_spark()
    try:
        report = ejecutar_auditoria_walk_forward(
            spark,
            input_path=args.input,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Reporte Walk-Forward guardado en: {out_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
