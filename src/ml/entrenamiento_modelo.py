"""Tarea 3.3: Entrenamiento comparativo PySpark MLlib, seleccion antifuga y recalibracion Prior Shift."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, functions as F
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.calibracion_umbral import (
    aplicar_calibracion_mensual_2024,
    calibrar_umbral_ventana_movil,
)
from src.ml.feature_engineering import (
    FORBIDDEN_LEAKAGE_COLS,
    NON_PREDICTOR_RETROACTIVE_COLS,
    PREDICTOR_COLS,
)
from src.ml.walk_forward_cv import (
    TARGETS_OFICIALES,
    WALK_FORWARD_FOLDS,
    build_preprocessing_stages,
    dividir_fold_con_embargo,
    evaluar_baselines_fold,
    evaluar_predicciones_con_ic,
    filtrar_sin_meses_baja_cobertura_2024,
)

# Regla estricta antifuga: SOLO los folds pre-2024 pueden usarse para elegir hiperparametros y familia
SELECTION_FOLDS = ("fold_1_2023_s1", "fold_2_2023_s2")
assert set(SELECTION_FOLDS) == {"fold_1_2023_s1", "fold_2_2023_s2"}

MODEL_GRIDS = {
    "GBTClassifier": {
        "scale": False,
        "configs": [
            {"maxDepth": 3, "maxIter": 40, "stepSize": 0.05},
            {"maxDepth": 3, "maxIter": 60, "stepSize": 0.1},
            {"maxDepth": 5, "maxIter": 40, "stepSize": 0.05},
            {"maxDepth": 5, "maxIter": 60, "stepSize": 0.1},
        ],
    },
    "RandomForestClassifier": {
        "scale": False,
        "configs": [
            {"maxDepth": 5, "numTrees": 50},
            {"maxDepth": 5, "numTrees": 100},
            {"maxDepth": 8, "numTrees": 50},
            {"maxDepth": 8, "numTrees": 100},
        ],
    },
    "LogisticRegression": {
        "scale": True,
        "configs": [
            {"regParam": 0.01, "elasticNetParam": 0.0},
            {"regParam": 0.01, "elasticNetParam": 0.5},
            {"regParam": 0.1, "elasticNetParam": 0.0},
            {"regParam": 0.1, "elasticNetParam": 0.5},
        ],
    },
}

SEMANA_7_BASELINE = {
    "y_ventana_2_4h": {"auc_roc": 0.463, "pr_auc": 0.119, "f1": 0.192},
    "y_inicio_ventana_2_4h": {"auc_roc": 0.638, "pr_auc": 0.061, "f1": 0.112},
}


def crear_spark(app_name: str = "coes-ml-training") -> SparkSession:
    """Inicializa SparkSession en UTC con configuracion segura para Windows."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def agregar_pesos_clase(
    train_df: DataFrame,
    target_col: str,
    weight_col: str = "class_weight",
) -> tuple[DataFrame, dict]:
    """Calcula pesos dinamicos de clase sobre train_df sin sobremuestreo sintetico."""
    agg = train_df.agg(
        F.count("*").alias("n_tot"),
        F.sum(F.when(F.col(target_col) == 1, 1).otherwise(0)).alias("n_pos"),
    ).first()
    n_tot = int(agg["n_tot"] or 0)
    n_pos = int(agg["n_pos"] or 0)
    n_neg = n_tot - n_pos

    if n_pos == 0 or n_neg == 0:
        pos_rate = (n_pos / n_tot) if n_tot > 0 else 0.0
        w_pos, w_neg = 1.0, 1.0
    else:
        pos_rate = n_pos / n_tot
        w_pos = 0.5 / pos_rate
        w_neg = 0.5 / (1.0 - pos_rate)

    weighted = train_df.withColumn(
        weight_col,
        F.when(F.col(target_col) == 1, F.lit(float(w_pos))).otherwise(F.lit(float(w_neg))),
    )
    return weighted, {
        "n_tot": n_tot,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": round(pos_rate, 6),
        "w_pos": round(w_pos, 6),
        "w_neg": round(w_neg, 6),
    }


def _instanciar_clasificador(familia: str, params: dict, target_col: str, weight_col: str = "class_weight"):
    if familia == "GBTClassifier":
        return GBTClassifier(
            labelCol=target_col,
            featuresCol="features",
            weightCol=weight_col,
            seed=42,
            **params,
        )
    if familia == "RandomForestClassifier":
        return RandomForestClassifier(
            labelCol=target_col,
            featuresCol="features",
            weightCol=weight_col,
            seed=42,
            **params,
        )
    if familia == "LogisticRegression":
        return LogisticRegression(
            labelCol=target_col,
            featuresCol="features",
            weightCol=weight_col,
            maxIter=100,
            **params,
        )
    raise ValueError(f"Familia no soportada: {familia}")


def _predecir_probabilidades(model, df: DataFrame, target_col: str) -> pd.DataFrame:
    scored = model.transform(df)
    if "probability" in scored.columns:
        prob_expr = vector_to_array(F.col("probability"))[1]
    else:
        # Fallback defensivo si rawPrediction no produjera probability
        prob_expr = F.lit(0.5)
    cols = ["fecha_hora", "fecha_despacho", target_col, prob_expr.alias("y_prob")]
    pdf = scored.select(*cols).toPandas()
    pdf = pdf.rename(columns={target_col: "y_true"})
    pdf["fecha_hora"] = pd.to_datetime(pdf["fecha_hora"])
    return pdf


def _pr_auc_seguro(pdf: pd.DataFrame) -> float | None:
    y = pdf["y_true"].dropna().to_numpy(dtype=int)
    if y.size == 0 or y.min() == y.max():
        return None
    p = pdf.loc[pdf["y_true"].notna(), "y_prob"].to_numpy(dtype=float)
    return float(average_precision_score(y, p))


def _folds_por_nombre() -> dict[str, dict]:
    return {f["nombre"]: f for f in WALK_FORWARD_FOLDS}


def seleccionar_hiperparametros_pre2024(
    df_feat: DataFrame,
    target_col: str,
    feature_cols: list[str],
    familias: tuple[str, ...] = tuple(MODEL_GRIDS.keys()),
) -> dict:
    """Selecciona la mejor config por familia y el ganador global SOLO sobre SELECTION_FOLDS."""
    assert set(SELECTION_FOLDS) == {"fold_1_2023_s1", "fold_2_2023_s2"}
    fold_map = _folds_por_nombre()

    # Preparar splits pre-2024 con pesos dinamicos y cachear transformaciones de preprocesamiento por fold
    fold_cache = {}
    for fname in SELECTION_FOLDS:
        fmeta = fold_map[fname]
        tr_df, va_df, split_stats = dividir_fold_con_embargo(
            df_feat,
            val_start=fmeta["val_start"],
            val_end=fmeta["val_end"],
            target_col=target_col,
        )
        tr_w, w_info = agregar_pesos_clase(tr_df, target_col=target_col)

        prep_by_scale = {}
        for scale in (False, True):
            prep_pipe = Pipeline(stages=build_preprocessing_stages(feature_cols=feature_cols, scale=scale))
            prep_model = prep_pipe.fit(tr_w)
            tr_p = prep_model.transform(tr_w).select("fecha_hora", "fecha_despacho", target_col, "class_weight", "features").cache()
            va_p = prep_model.transform(va_df).select("fecha_hora", "fecha_despacho", target_col, "features").cache()
            tr_p.count()
            va_p.count()
            prep_by_scale[scale] = (tr_p, va_p)

        fold_cache[fname] = {
            "splits": split_stats,
            "pesos": w_info,
            "prep": prep_by_scale,
        }

    candidatos = []
    mejores_por_familia = {}

    for familia in familias:
        spec = MODEL_GRIDS[familia]
        scale = spec["scale"]
        mejor_fam = None

        for cfg in spec["configs"]:
            por_fold = {}
            val_scores = []
            gaps = []

            for fname in SELECTION_FOLDS:
                tr_p, va_p = fold_cache[fname]["prep"][scale]
                clf = _instanciar_clasificador(familia, cfg, target_col=target_col)
                clf_model = clf.fit(tr_p)

                tr_pred = _predecir_probabilidades(clf_model, tr_p, target_col=target_col)
                va_pred = _predecir_probabilidades(clf_model, va_p, target_col=target_col)

                pr_tr = _pr_auc_seguro(tr_pred)
                pr_va = _pr_auc_seguro(va_pred)
                gap = (pr_tr - pr_va) if (pr_tr is not None and pr_va is not None) else None

                if pr_va is not None:
                    val_scores.append(pr_va)
                if gap is not None:
                    gaps.append(gap)

                por_fold[fname] = {
                    "pr_auc_train": round(pr_tr, 6) if pr_tr is not None else None,
                    "pr_auc_val": round(pr_va, 6) if pr_va is not None else None,
                    "brecha_overfit": round(gap, 6) if gap is not None else None,
                }

            pr_mean = float(np.mean(val_scores)) if val_scores else -1.0
            gap_mean = float(np.mean(gaps)) if gaps else None

            registro = {
                "familia": familia,
                "scale": scale,
                "params": cfg,
                "pr_auc_val_promedio_pre2024": round(pr_mean, 6),
                "brecha_overfit_promedio": round(gap_mean, 6) if gap_mean is not None else None,
                "folds_seleccion": por_fold,
            }
            candidatos.append(registro)

            if mejor_fam is None or registro["pr_auc_val_promedio_pre2024"] > mejor_fam["pr_auc_val_promedio_pre2024"]:
                mejor_fam = registro

        mejores_por_familia[familia] = mejor_fam

    for fname in SELECTION_FOLDS:
        for tr_p, va_p in fold_cache[fname]["prep"].values():
            tr_p.unpersist()
            va_p.unpersist()

    ganador = max(mejores_por_familia.values(), key=lambda r: r["pr_auc_val_promedio_pre2024"])
    return {
        "folds_usados_seleccion": list(SELECTION_FOLDS),
        "n_candidatos_evaluados": len(candidatos),
        "mejores_por_familia": mejores_por_familia,
        "ganador": ganador,
        "candidatos": candidatos,
    }


def _entrenar_y_predecir_fold(
    train_df: DataFrame,
    eval_df: DataFrame,
    familia: str,
    params: dict,
    scale: bool,
    target_col: str,
    feature_cols: list[str],
):
    tr_w, w_info = agregar_pesos_clase(train_df, target_col=target_col)
    prep_stages = build_preprocessing_stages(feature_cols=feature_cols, scale=scale)
    clf = _instanciar_clasificador(familia, params, target_col=target_col)
    pipe = Pipeline(stages=[*prep_stages, clf])
    model = pipe.fit(tr_w)

    tr_pred = _predecir_probabilidades(model, tr_w, target_col=target_col)
    ev_pred = _predecir_probabilidades(model, eval_df, target_col=target_col)
    return model, w_info, tr_pred, ev_pred


def _evaluar_comparativa_2024(
    val_2023_s2_pred: pd.DataFrame,
    test_2024_pred: pd.DataFrame,
    train_2024_pdf: pd.DataFrame,
    pi_train_2024: float,
    target_col: str,
    n_boot: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Evalua umbral fijo 2023-S2, Saerens+60d y Saerens+60d (max_fp=3) en 2024 global y sin ene/abr."""
    # Umbral fijo heredado de 2023-S2 (calibrado sin fuga sobre validacion OOS de 2023-S2)
    cal_fijo = calibrar_umbral_ventana_movil(
        df_hist_preds=val_2023_s2_pred,
        fecha_inicio_mes="2024-01-01",
        ventana_dias=185,
        max_fp_semana=None,
        pi_train=None,
    )
    tau_fijo = float(cal_fijo["tau"])

    # Recalibracion mensual 60d libre y con techo de 3 FP/semana
    pred_rec_libre, traz_libre = aplicar_calibracion_mensual_2024(
        df_val_2023=val_2023_s2_pred,
        df_test_2024=test_2024_pred,
        pi_train=pi_train_2024,
        ventana_dias=60,
        max_fp_semana=None,
    )
    pred_rec_fp3, traz_fp3 = aplicar_calibracion_mensual_2024(
        df_val_2023=val_2023_s2_pred,
        df_test_2024=test_2024_pred,
        pi_train=pi_train_2024,
        ventana_dias=60,
        max_fp_semana=3.0,
    )

    test_combinado = test_2024_pred.copy()
    test_combinado["umbral_fijo_2023_s2"] = tau_fijo
    test_combinado["y_pred_fijo"] = (test_combinado["y_prob"] >= tau_fijo).astype(int)
    test_combinado["prob_recalibrada"] = pred_rec_libre["prob_recalibrada"].to_numpy()
    test_combinado["umbral_recalibrado"] = pred_rec_libre["umbral_aplicado"].to_numpy()
    test_combinado["y_pred_recalibrado"] = pred_rec_libre["y_pred_recalibrado"].to_numpy()
    test_combinado["prob_recalibrada_maxfp3"] = pred_rec_fp3["prob_recalibrada"].to_numpy()
    test_combinado["umbral_recalibrado_maxfp3"] = pred_rec_fp3["umbral_aplicado"].to_numpy()
    test_combinado["y_pred_recalibrado_maxfp3"] = pred_rec_fp3["y_pred_recalibrado"].to_numpy()

    def _evaluar_vista(vista_df: pd.DataFrame, train_ref: pd.DataFrame) -> dict:
        verificables = vista_df[vista_df["y_true"].notna()].copy()
        verificables["y_true"] = verificables["y_true"].astype(int)

        df_fijo = verificables[["fecha_despacho", "y_true", "y_prob"]].copy()
        res_fijo = evaluar_predicciones_con_ic(
            df_fijo,
            threshold=tau_fijo,
            n_boot=n_boot,
            seed=seed,
        )

        df_rec = verificables[["fecha_despacho", "y_true", "prob_recalibrada", "umbral_recalibrado"]].rename(
            columns={"prob_recalibrada": "y_prob", "umbral_recalibrado": "threshold"}
        )
        res_rec = evaluar_predicciones_con_ic(
            df_rec,
            threshold=0.5,
            n_boot=n_boot,
            seed=seed,
        )

        df_fp3 = verificables[
            ["fecha_despacho", "y_true", "prob_recalibrada_maxfp3", "umbral_recalibrado_maxfp3"]
        ].rename(columns={"prob_recalibrada_maxfp3": "y_prob", "umbral_recalibrado_maxfp3": "threshold"})
        res_fp3 = evaluar_predicciones_con_ic(
            df_fp3,
            threshold=0.5,
            n_boot=n_boot,
            seed=seed,
        )

        val_bl = verificables[["fecha_hora", "fecha_despacho", "y_true"]].rename(columns={"y_true": target_col})
        baselines = evaluar_baselines_fold(
            train_pdf=train_ref,
            val_pdf=val_bl,
            target_col=target_col,
            optimize_threshold_on="train",
            n_boot=n_boot,
            seed=seed,
        )

        return {
            "umbral_fijo_2023_s2": res_fijo,
            "recalibrado_mensual_60d": res_rec,
            "recalibrado_mensual_60d_maxfp3": res_fp3,
            **baselines,
            "semana_7_baseline": SEMANA_7_BASELINE[target_col],
        }

    vista_global = _evaluar_vista(test_combinado, train_2024_pdf)
    vista_sin_baja = _evaluar_vista(filtrar_sin_meses_baja_cobertura_2024(test_combinado), train_2024_pdf)

    report = {
        "umbral_fijo_heredado_2023_s2": tau_fijo,
        "trazabilidad_recalibracion_60d": traz_libre,
        "trazabilidad_recalibracion_60d_maxfp3": traz_fp3,
        "2024_global": vista_global,
        "2024_sin_meses_baja_cobertura": vista_sin_baja,
    }
    return report, test_combinado


def ejecutar_experimento_target(
    df_feat: DataFrame,
    target_col: str,
    feature_cols: list[str],
    out_dir: Path,
    n_boot: int,
    seed: int,
    guardar_modelo: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Ejecuta seleccion antifuga pre-2024, evalua fold_3 y test_final_2024, y persiste modelo ganador."""
    fold_map = _folds_por_nombre()

    seleccion = seleccionar_hiperparametros_pre2024(
        df_feat=df_feat,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    ganador = seleccion["ganador"]
    familia = ganador["familia"]
    params = ganador["params"]
    scale = ganador["scale"]

    # Evaluacion diagnostica del ganador en fold_3_2024_h1 (sin haberlo usado para seleccionar)
    f3 = fold_map["fold_3_2024_h1"]
    tr3, va3, st3 = dividir_fold_con_embargo(df_feat, f3["val_start"], f3["val_end"], target_col=target_col)
    _, _, tr3_pred, va3_pred = _entrenar_y_predecir_fold(tr3, va3, familia, params, scale, target_col, feature_cols)
    pr_tr3 = _pr_auc_seguro(tr3_pred)
    pr_va3 = _pr_auc_seguro(va3_pred)

    # Predicciones OOS de 2023-S2 del ganador (para calibrar umbral fijo y arranque enero 2024)
    f2 = fold_map["fold_2_2023_s2"]
    tr2, va2, _ = dividir_fold_con_embargo(df_feat, f2["val_start"], f2["val_end"], target_col=target_col)
    _, _, _, val_2023_s2_pred = _entrenar_y_predecir_fold(tr2, va2, familia, params, scale, target_col, feature_cols)

    # Entrenamiento final oficial (< 2024-01-01 00:30 - 4h) y scoring sobre TODO 2024 (incluidos censurados)
    f_test = fold_map["test_final_2024"]
    tr_final, va_final_obs, st_final = dividir_fold_con_embargo(
        df_feat, f_test["val_start"], f_test["val_end"], target_col=target_col
    )
    va_final_todo = df_feat.filter(
        (F.col("fecha_hora") >= F.lit(f_test["val_start"]).cast("timestamp"))
        & (F.col("fecha_hora") <= F.lit(f_test["val_end"]).cast("timestamp"))
    )

    model_final, w_info_final, tr_final_pred, va_final_todo_pred = _entrenar_y_predecir_fold(
        tr_final, va_final_todo, familia, params, scale, target_col, feature_cols
    )
    pr_tr_final = _pr_auc_seguro(tr_final_pred)
    pr_va_final = _pr_auc_seguro(va_final_todo_pred)

    if guardar_modelo:
        model_path = out_dir / f"modelo_ganador_{target_col}"
        model_final.write().overwrite().save(str(model_path))

    # Evaluar las otras dos familias ganadoras en test_final_2024 para tabla comparativa completa
    comparativa_familias_2024 = {}
    for fam_nombre, fam_best in seleccion["mejores_por_familia"].items():
        if fam_nombre == familia:
            fam_pred_2024 = va_final_todo_pred
        else:
            _, _, _, fam_pred_2024 = _entrenar_y_predecir_fold(
                tr_final,
                va_final_todo,
                fam_nombre,
                fam_best["params"],
                fam_best["scale"],
                target_col,
                feature_cols,
            )
        verif = fam_pred_2024[fam_pred_2024["y_true"].notna()].copy()
        verif["y_true"] = verif["y_true"].astype(int)
        comparativa_familias_2024[fam_nombre] = evaluar_predicciones_con_ic(
            verif[["fecha_despacho", "y_true", "y_prob"]],
            threshold=0.5,
            n_boot=n_boot,
            seed=seed,
        )

    tr_final_pdf = tr_final.select("fecha_hora", "fecha_despacho", target_col).toPandas()
    tr_final_pdf["fecha_hora"] = pd.to_datetime(tr_final_pdf["fecha_hora"])
    pi_train_2024 = float(tr_final_pdf[target_col].mean())

    eval_2024, test_combinado_pdf = _evaluar_comparativa_2024(
        val_2023_s2_pred=val_2023_s2_pred,
        test_2024_pred=va_final_todo_pred,
        train_2024_pdf=tr_final_pdf,
        pi_train_2024=pi_train_2024,
        target_col=target_col,
        n_boot=n_boot,
        seed=seed,
    )

    resumen = {
        "target": target_col,
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "seleccion_pre2024": seleccion,
        "modelo_ganador": {
            "familia": familia,
            "params": params,
            "scale": scale,
            "pesos_train_final": w_info_final,
            "pr_auc_train_final": round(pr_tr_final, 6) if pr_tr_final is not None else None,
            "pr_auc_test_final_raw": round(pr_va_final, 6) if pr_va_final is not None else None,
            "brecha_overfit_final": round(pr_tr_final - pr_va_final, 6)
            if (pr_tr_final is not None and pr_va_final is not None)
            else None,
            "fold_3_2024_h1_diagnostico": {
                "splits": st3,
                "pr_auc_train": round(pr_tr3, 6) if pr_tr3 is not None else None,
                "pr_auc_val": round(pr_va3, 6) if pr_va3 is not None else None,
                "brecha_overfit": round(pr_tr3 - pr_va3, 6) if (pr_tr3 is not None and pr_va3 is not None) else None,
            },
        },
        "comparativa_familias_test_final_2024_raw": comparativa_familias_2024,
        "test_final_2024": {
            "splits": st_final,
            **eval_2024,
        },
    }
    return resumen, test_combinado_pdf


def _ejecutar_modo_muestra_ligero(df_feat: DataFrame, out_dir: Path, summary_out: Path) -> dict:
    """Ruta ligera para smoke test en data/sample cuando no hay cobertura multi-anio completa."""
    target_col = "y_ventana_2_4h"
    df_obs = df_feat.filter(F.col(target_col).isNotNull()).orderBy("fecha_hora")
    n_obs = df_obs.count()
    resumen = {
        "modo": "muestra_ligera_sample",
        "total_bloques": int(df_feat.count()),
        "bloques_etiquetables": int(n_obs),
        "mensaje": "Dataset de muestra insuficiente para Walk-Forward 2023-2024; validacion de esquema y prediccion smoke ejecutada.",
    }
    if n_obs >= 20:
        n_pos = df_obs.filter(F.col(target_col) == 1).count()
        if 0 < n_pos < n_obs:
            tr_w, w_info = agregar_pesos_clase(df_obs, target_col=target_col)
            pipe = Pipeline(
                stages=[
                    *build_preprocessing_stages(feature_cols=PREDICTOR_COLS, scale=True),
                    LogisticRegression(
                        labelCol=target_col,
                        featuresCol="features",
                        weightCol="class_weight",
                        maxIter=20,
                        regParam=0.1,
                    ),
                ]
            )
            model = pipe.fit(tr_w)
            preds = _predecir_probabilidades(model, df_feat, target_col=target_col)
            resumen["pesos"] = w_info
            resumen["n_predicciones"] = int(len(preds))
            preds.to_parquet(out_dir / "predicciones_test_2024.parquet", index=False)
            model.write().overwrite().save(str(out_dir / f"modelo_ganador_{target_col}"))
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento comparativo PySpark MLlib (Tarea 3.3)")
    parser.add_argument("--input", default="data/sample/gold/ml_features_30min", help="Ruta a ml_features_30min")
    parser.add_argument("--output", default="data/sample/ml_out", help="Directorio de salida ML")
    parser.add_argument(
        "--summary-out",
        default="evidencias_modelo/comparativa_modelos.json",
        help="Ruta JSON de comparativa",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Iteraciones block bootstrap diario")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out)

    spark = crear_spark()
    try:
        df_feat = spark.read.parquet(args.input).cache()

        # Barrera antifuga en entrada
        fuga = sorted(FORBIDDEN_LEAKAGE_COLS.intersection(df_feat.columns))
        if fuga:
            raise ValueError(f"Barrera antifuga violada en entrada: {fuga}")
        assert not (set(PREDICTOR_COLS) & (FORBIDDEN_LEAKAGE_COLS | NON_PREDICTOR_RETROACTIVE_COLS))

        # Detectar si hay suficientes datos en folds pre-2024 y 2024
        n_obs_2023 = df_feat.filter(
            (F.year("fecha_despacho") == 2023) & F.col("y_ventana_2_4h").isNotNull()
        ).count()
        n_obs_2024 = df_feat.filter(
            (F.year("fecha_despacho") == 2024) & F.col("y_ventana_2_4h").isNotNull()
        ).count()

        if n_obs_2023 < 100 or n_obs_2024 < 100:
            resumen = _ejecutar_modo_muestra_ligero(df_feat, out_dir, summary_path)
            print(json.dumps(resumen, indent=2))
            return

        # 1) Target 1: y_ventana_2_4h (PREDICTOR_COLS completo)
        res_ev, pred_ev_2024 = ejecutar_experimento_target(
            df_feat=df_feat,
            target_col="y_ventana_2_4h",
            feature_cols=PREDICTOR_COLS,
            out_dir=out_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            guardar_modelo=True,
        )

        # 2) Target 2: y_inicio_ventana_2_4h (PREDICTOR_COLS completo + ablacion sin persistencia_term_nop_2h)
        res_ini_full, pred_ini_2024 = ejecutar_experimento_target(
            df_feat=df_feat,
            target_col="y_inicio_ventana_2_4h",
            feature_cols=PREDICTOR_COLS,
            out_dir=out_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            guardar_modelo=True,
        )

        cols_sin_persistencia = [c for c in PREDICTOR_COLS if c != "persistencia_term_nop_2h"]
        res_ini_abl, _ = ejecutar_experimento_target(
            df_feat=df_feat,
            target_col="y_inicio_ventana_2_4h",
            feature_cols=cols_sin_persistencia,
            out_dir=out_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            guardar_modelo=False,
        )

        # Persistir predicciones detalladas de 2024 (incluyendo NO_VERIFICABLE donde y_true IS NULL)
        f_test = _folds_por_nombre()["test_final_2024"]
        base_2024 = (
            df_feat.filter(
                (F.col("fecha_hora") >= F.lit(f_test["val_start"]).cast("timestamp"))
                & (F.col("fecha_hora") <= F.lit(f_test["val_end"]).cast("timestamp"))
            )
            .select(
                "fecha_hora",
                "fecha_despacho",
                "anio",
                "etiqueta_observable",
                "motivo_censura",
                *PREDICTOR_COLS,
            )
            .toPandas()
        )
        base_2024["fecha_hora"] = pd.to_datetime(base_2024["fecha_hora"])

        def _renombrar_pred(pdf: pd.DataFrame, sufijo: str) -> pd.DataFrame:
            out = pdf[
                [
                    "fecha_hora",
                    "y_true",
                    "y_prob",
                    "umbral_fijo_2023_s2",
                    "y_pred_fijo",
                    "prob_recalibrada",
                    "umbral_recalibrado",
                    "y_pred_recalibrado",
                    "prob_recalibrada_maxfp3",
                    "umbral_recalibrado_maxfp3",
                    "y_pred_recalibrado_maxfp3",
                ]
            ].copy()
            out[f"estado_{sufijo}"] = np.where(out["y_true"].notna(), "VERIFICABLE", "NO_VERIFICABLE")
            return out.rename(
                columns={
                    "y_true": f"y_true_{sufijo}",
                    "y_prob": f"prob_raw_{sufijo}",
                    "umbral_fijo_2023_s2": f"umbral_fijo_{sufijo}",
                    "y_pred_fijo": f"pred_fijo_{sufijo}",
                    "prob_recalibrada": f"prob_recal_{sufijo}",
                    "umbral_recalibrado": f"umbral_recal_{sufijo}",
                    "y_pred_recalibrado": f"pred_recal_{sufijo}",
                    "prob_recalibrada_maxfp3": f"prob_recal_maxfp3_{sufijo}",
                    "umbral_recalibrado_maxfp3": f"umbral_recal_maxfp3_{sufijo}",
                    "y_pred_recalibrado_maxfp3": f"pred_recal_maxfp3_{sufijo}",
                }
            )

        pred_2024_final = (
            base_2024.merge(_renombrar_pred(pred_ev_2024, "evento"), on="fecha_hora", how="left")
            .merge(_renombrar_pred(pred_ini_2024, "inicio"), on="fecha_hora", how="left")
            .sort_values("fecha_hora")
            .reset_index(drop=True)
        )
        pred_path = out_dir / "predicciones_test_2024.parquet"
        pred_2024_final.to_parquet(pred_path, index=False)

        reporte = {
            "input": args.input,
            "output": str(out_dir),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "regla_seleccion_antifuga": list(SELECTION_FOLDS),
            "predicciones_2024_parquet": str(pred_path),
            "total_bloques_2024": int(len(pred_2024_final)),
            "targets": {
                "y_ventana_2_4h": res_ev,
                "y_inicio_ventana_2_4h": res_ini_full,
                "y_inicio_ventana_2_4h_ablacion_sin_persistencia": res_ini_abl,
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
        print(f"[OK] Comparativa de modelos guardada en: {summary_path}")
        print(f"[OK] Predicciones 2024 guardadas en: {pred_path}")
    finally:
        df_feat.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
