"""Entrenamiento y evaluacion base consumiendo exclusivamente gold/gold_feature_store_30min."""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

COLUMNAS_PROHIBIDAS_ANTIFUGA = {
    "label_rdo",
    "alfa",
    "beta",
    "gamma",
    "chm",
    "alfa_primer_archivo",
    "beta_primer_archivo",
    "gamma_primer_archivo",
}


def cargar_feature_store(gold_dir: str) -> pd.DataFrame:
    """Lee unicamente gold/gold_feature_store_30min y valida barrera antifuga."""
    fs_dir = Path(gold_dir) / "gold_feature_store_30min"
    if not fs_dir.exists():
        raise FileNotFoundError(
            f"No existe {fs_dir}. Ejecuta primero src/etl/pipeline.py --stage gold "
            "(se prohibe leer dataset_analitico_termico.parquet o B_dataset_maestro.csv)."
        )
    files = sorted(glob.glob(str(fs_dir / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet dentro de {fs_dir}.")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    fuga = COLUMNAS_PROHIBIDAS_ANTIFUGA.intersection(df.columns)
    if fuga:
        raise ValueError(f"Barrera antifuga violada en gold_feature_store_30min: {sorted(fuga)}")

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    return df.sort_values("fecha_hora").reset_index(drop=True)


def construir_objetivo_ventana_2a4h(df: pd.DataFrame) -> pd.Series:
    """Target en ventana futura [t+4, t+8] (2 a 4 horas) sobre la grilla continua de 30 min."""
    ev = df["evento_termico_nop"].astype("float64")
    futuros = pd.concat([ev.shift(-k) for k in range(4, 9)], axis=1)
    algun_evento = (futuros == 1.0).any(axis=1)
    todos_observables = futuros.notna().all(axis=1)
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y[algun_evento] = 1.0
    y[~algun_evento & todos_observables] = 0.0
    return y


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construye features estrictamente conocidas en el instante t."""
    df = df.copy()
    df["bloque_30min"] = df["fecha_hora"].dt.hour * 2 + (df["fecha_hora"].dt.minute // 30)
    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["mes"] = df["fecha_hora"].dt.month
    df["es_finde"] = (df["dia_semana"] >= 5).astype(int)

    df["rampa_1"] = df["gen_total_mw"].diff(1)
    df["rampa_4"] = df["gen_total_mw"].diff(4)

    features = [
        "bloque_30min",
        "dia_semana",
        "mes",
        "es_finde",
        "gen_total_mw",
        "rampa_1",
        "rampa_4",
        "term_cara_ejec_mw",
        "term_cara_prog_mw",
        "term_cara_share",
        "cmg_troncal_mediana",
        "cmg_spread_p90_p50",
        "n_forzados_termicos_activos",
        "n_forzados_transmision_activos",
        "n_forzados_termicos_inicio_3h",
        "n_mantenimientos_prog_termicos_activos",
    ]
    return df, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/gold", help="Directorio Gold oficial")
    parser.add_argument("--output", default="data/sample/ml_out", help="Directorio de salida ML")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ENTRENAMIENTO BASE DESDE GOLD_FEATURE_STORE_30MIN (FASE 2)")
    print("=" * 65)

    df = cargar_feature_store(args.input)
    df, features = preparar_features(df)
    df["target_ventana_2a4h"] = construir_objetivo_ventana_2a4h(df)

    # Filtrar solo bloques con etiqueta observable (nunca imputar 0 en bloques censurados)
    df_model = df[df["etiqueta_observable"] & df["target_ventana_2a4h"].notna()].copy()
    df_model["target_ventana_2a4h"] = df_model["target_ventana_2a4h"].astype(int)

    if len(df_model) < 20:
        raise RuntimeError(f"Muy pocos bloques etiquetables ({len(df_model)}) para entrenar/evaluar.")

    # Particion temporal estricta con margen de separacion de 4 horas (Train <= 2023-12-31 | Test >= 2024-01-01)
    corte_test = pd.Timestamp("2024-01-01 00:00:00")
    train_mask = df_model["fecha_hora"] < (corte_test - pd.Timedelta(hours=4))
    test_mask = df_model["fecha_hora"] >= corte_test

    modo_split = "estricto_2024"
    if train_mask.sum() < 10 or test_mask.sum() < 10:
        modo_split = "cronologico_80_20_muestra"
        split_idx = int(len(df_model) * 0.8)
        train_mask = pd.Series(False, index=df_model.index)
        train_mask.iloc[:split_idx] = True
        test_mask = ~train_mask

    # Imputacion simple con medianas de entrenamiento (sin fuga del conjunto de prueba)
    medianas_train = df_model.loc[train_mask, features].median().fillna(0.0)
    X_train = df_model.loc[train_mask, features].fillna(medianas_train)
    y_train = df_model.loc[train_mask, "target_ventana_2a4h"]
    X_test = df_model.loc[test_mask, features].fillna(medianas_train)
    y_test = df_model.loc[test_mask, "target_ventana_2a4h"]

    print(f"Fuente oficial : {Path(args.input) / 'gold_feature_store_30min'}")
    print(f"Split temporal : {modo_split}")
    print(f"Train          : {len(X_train):,} bloques | Positivos={int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"Test           : {len(X_test):,} bloques | Positivos={int(y_test.sum()):,} ({y_test.mean():.2%})")

    if y_train.nunique() < 2:
        print("[AVISO] La muestra tiene una sola clase en entrenamiento (smoke test). Se omiten AUC/PR-AUC.")
        clase_unica = int(y_train.iloc[0])
        y_pred = np.full(len(X_test), clase_unica, dtype=int)
        y_proba = np.full(len(X_test), float(clase_unica), dtype=float)
        imp = pd.Series(0.0, index=features)
    else:
        modelo = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        imp = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tiene_dos_clases_test = y_test.nunique() == 2
    auc_roc = float(roc_auc_score(y_test, y_proba)) if tiene_dos_clases_test else None
    pr_auc = float(average_precision_score(y_test, y_proba)) if tiene_dos_clases_test else None
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))


    metricas = {
        "fuente_oficial": str(Path(args.input) / "gold_feature_store_30min"),
        "target": "evento_termico_nop (ventana t+4..t+8)",
        "modo_split": modo_split,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positivos_train": int(y_train.sum()),
        "positivos_test": int(y_test.sum()),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "matriz_confusion": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
        "importancia_variables": {k: round(float(v), 4) for k, v in imp.items()},
    }

    (out_dir / "metricas_modelo_base.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")

    print("\n--- RESULTADOS ---")
    print(f"AUC-ROC   : {metricas['auc_roc']}")
    print(f"PR-AUC    : {metricas['pr_auc']}")
    print(f"F1        : {metricas['f1']}")
    print(f"Precision : {metricas['precision']}")
    print(f"Recall    : {metricas['recall']}")
    print(f"Matriz    : TN={cm[0,0]} | FP={cm[0,1]} | FN={cm[1,0]} | TP={cm[1,1]}")
    print("\nTop 8 variables:")
    for feat, val in imp.head(8).items():
        print(f"  - {feat:<38}: {val:.4f}")

    # Guardar matriz de confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sin evento (0)", "Con evento (1)"], fontweight="bold")
    ax.set_yticklabels(["Sin evento (0)", "Con evento (1)"], fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real (evento_termico_nop 2-4h)")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
                fontweight="bold",
            )
    plt.title("Matriz de Confusion - Base sobre Feature Store", fontweight="bold", pad=10)
    plt.tight_layout()
    fig_path = out_dir / "fig_4_2_matriz_confusion_baseline.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\n[OK] Metricas y grafica guardadas en: {out_dir}")


if __name__ == "__main__":
    main()