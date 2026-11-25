#!/usr/bin/env python3
"""
Tarea 4.1 – Capa Semántica: Modelo Dimensional Estrella
========================================================
Genera 4 dimensiones + 3 tablas de hechos en Parquet (Snappy)
a partir de las tablas gold existentes. 100 % agnóstico de proveedor:
solo PyArrow, Pandas y SQL estándar. Cero datos sintéticos.

Uso:
    python src/serving/build_star_schema.py \
        --gold-input  lake_v2/gold \
        --serving-output lake_v2/serving/star_schema \
        --summary-out evidencias_serving/star_schema_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
COMBUSTIBLES_CAROS = {"D2", "R500", "NLH_RFG", "FX_RFG", "CARB"}

# Tolerancia de reconciliación contable (soles)
RECON_TOL = 0.01

EXPECTED_EXPOSICION = {
    2021: 404_599.93,
    2022: 75_030_153.71,
    2023: 175_600_703.10,
    2024: 19_079_554.64,
}
EXPECTED_SOBRECOSTO = {
    2021: 239_930.13,
    2022: 15_114_059.99,
    2023: 48_189_150.53,
    2024: 15_223_546.20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_gold(gold: Path, name: str) -> pd.DataFrame:
    """Lee tabla gold (particionada Hive o archivo único)."""
    path = gold / name
    if path.is_dir():
        return pq.read_table(str(path)).to_pandas()
    elif path.with_suffix(".parquet").is_file():
        return pq.read_table(str(path.with_suffix(".parquet"))).to_pandas()
    raise FileNotFoundError(f"No se encontró {path}")


def write_parquet(df: pd.DataFrame, out: Path, name: str) -> Path:
    """Escribe Parquet + Snappy, devuelve ruta del archivo."""
    dest = out / f"{name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(dest), compression="snappy")
    return dest


def extract_kv(barra: str) -> int:
    """Extrae nivel de tensión (kV) del nombre de barra."""
    m = re.search(r"(\d{3})$", str(barra).strip())
    return int(m.group(1)) if m else 0


def periodo_hidrologico(mes: int) -> str:
    """Dic-Abr → avenida, May-Nov → estiaje."""
    return "avenida" if mes in (12, 1, 2, 3, 4) else "estiaje"


# =====================================================================
#  DIMENSIONES
# =====================================================================
def build_dim_tiempo_operativo(gold: Path) -> pd.DataFrame:
    """
    dim_tiempo_operativo – 52 608 filas (grano 30 min).
    Genera la dimensión de tiempo a partir del rango real de fecha_hora
    existente en gold_feature_store_30min.
    """
    fs = read_gold(gold, "gold_feature_store_30min")
    fechas = fs[["fecha_hora", "fecha_despacho"]].drop_duplicates().sort_values("fecha_hora").reset_index(drop=True)
    assert len(fechas) == 52_608, f"Se esperaban 52608 bloques, encontrados {len(fechas)}"

    df = pd.DataFrame()
    df["fecha_hora"] = pd.to_datetime(fechas["fecha_hora"])
    df["fecha_despacho"] = pd.to_datetime(fechas["fecha_despacho"]).dt.date
    df["tiempo_key"] = df["fecha_hora"].dt.strftime("%Y%m%d_%H%M")
    df["bloque_30min"] = df["fecha_hora"].dt.hour * 2 + df["fecha_hora"].dt.minute // 30 + 1  # 1-48
    df["hora"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.day_name()
    df["es_fin_semana"] = df["fecha_hora"].dt.dayofweek.isin([5, 6])
    df["mes"] = df["fecha_hora"].dt.month
    df["anio"] = df["fecha_hora"].dt.year
    df["anio_mes"] = df["fecha_hora"].dt.strftime("%Y-%m")
    df["periodo_hidrologico"] = df["mes"].apply(periodo_hidrologico)

    # Orden de columnas
    df = df[
        [
            "tiempo_key",
            "fecha_hora",
            "fecha_despacho",
            "bloque_30min",
            "hora",
            "dia_semana",
            "es_fin_semana",
            "mes",
            "anio",
            "anio_mes",
            "periodo_hidrologico",
        ]
    ]
    return df


def build_dim_central_combustible(gold: Path) -> pd.DataFrame:
    """
    dim_central_combustible – 171 filas.
    Se deriva de despacho_central_30min; una fila por central.
    """
    desp = read_gold(gold, "despacho_central_30min")
    # Tomar la primera ocurrencia por central para empresa y combustible
    dim = (
        desp.groupby("central", sort=False)
        .agg(
            empresa=("empresa", "first"),
            combustible=("combustible_elegido", "first"),
            central_en_a3=("central_en_a3", "first"),
        )
        .reset_index()
    )
    dim["central_id"] = range(1, len(dim) + 1)
    dim["es_combustible_caro"] = dim["combustible"].isin(COMBUSTIBLES_CAROS)
    dim["estado_cobertura_a3"] = dim["central_en_a3"].apply(
        lambda x: "con_costo_A3" if x else "sin_costo_A3"
    )
    dim = dim[
        [
            "central_id",
            "central",
            "empresa",
            "combustible",
            "es_combustible_caro",
            "central_en_a3",
            "estado_cobertura_a3",
        ]
    ]
    assert len(dim) == 171, f"Se esperaban 171 centrales, encontradas {len(dim)}"
    return dim


def build_dim_barra_troncal(gold: Path) -> pd.DataFrame:
    """
    dim_barra_troncal – 84 filas.
    Barras troncales (220 kV y 500 kV) de precios_barra_30min.
    """
    bp = read_gold(gold, "precios_barra_30min")
    bp["nivel_tension_kv"] = bp["barra"].apply(extract_kv)
    troncales = bp[bp["nivel_tension_kv"].isin([220, 500])].copy()

    dim = (
        troncales.groupby(["barra_id", "barra", "nivel_tension_kv"], sort=False)
        .agg(
            bloques_total=("fecha_hora", "count"),
            bloques_validos=("cero_como_faltante", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    dim["pct_bloques_validos"] = (dim["bloques_validos"] / dim["bloques_total"] * 100).round(4)
    dim = dim[
        [
            "barra_id",
            "barra",
            "nivel_tension_kv",
            "bloques_total",
            "bloques_validos",
            "pct_bloques_validos",
        ]
    ]
    assert len(dim) == 84, f"Se esperaban 84 barras troncales, encontradas {len(dim)}"
    return dim


def build_dim_modelo_version(gold: Path) -> pd.DataFrame:
    """
    dim_modelo_version – poblada desde registro_modelos.json.
    """
    json_path = gold / "ml_outputs" / "registro_modelos.json"
    with open(json_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    # Si es un dict (un solo modelo), convertir a lista
    if isinstance(reg, dict):
        reg = [reg]

    rows = []
    for m in reg:
        rows.append(
            {
                "model_version_id": m["model_version_id"],
                "algoritmo": m["algoritmo"],
                "hiperparametros_json": json.dumps(m["hiperparametros"], ensure_ascii=False),
                "fecha_corte_train": m["fecha_corte_entrenamiento"],
                "hash_esquema_features": m["hash_esquema_features"],
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
#  TABLAS DE HECHOS
# =====================================================================
def build_fact_operacion_bloque_30min(gold: Path, dim_tiempo: pd.DataFrame) -> pd.DataFrame:
    """
    fact_operacion_bloque_30min – 52 608 filas (grano tiempo_key).
    Métricas agregadas de generación, precios troncales, observabilidad,
    eventos, exposición y sobrecosto.
    """
    # Cargar tablas fuente
    fs = read_gold(gold, "gold_feature_store_30min")
    obs = read_gold(gold, "gold_observabilidad_etiqueta_30min")
    pt = read_gold(gold, "gold_precios_troncales_30min")
    ev = read_gold(gold, "gold_eventos_criticos_30min")
    ca = read_gold(gold, "gold_contexto_auditoria_30min")

    # Normalizar fecha_hora
    for df in [fs, obs, pt, ev, ca]:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    # Empezar con dim_tiempo como eje (garantiza 52608 filas)
    fact = dim_tiempo[["tiempo_key", "fecha_hora"]].copy()

    # --- Feature store: métricas de generación y costos ---
    fs_cols = [
        "fecha_hora",
        "gen_total_mw",
        "prog_total_mw",
        "term_cara_ejec_mw",
        "term_cara_prog_mw",
        "term_cara_share",
        "energia_nop_mwh",
        "exposicion_bruta_soles",
        "sobrecosto_neto_estimado_soles",
    ]
    fact = fact.merge(fs[fs_cols], on="fecha_hora", how="left")

    # --- Precios troncales ---
    pt_cols = [
        "fecha_hora",
        "cmg_troncal_mediana",
        "cmg_troncal_p90",
        "cmg_spread_p90_p50",
        "n_barras_troncales_validas",
    ]
    fact = fact.merge(pt[pt_cols], on="fecha_hora", how="left")

    # --- Observabilidad: etiqueta, motivo, targets ---
    obs_cols = [
        "fecha_hora",
        "etiqueta_observable",
        "motivo_censura",
        "evento_termico_nop",
        "inicio_termico_nop",
        "n_centrales_term_nop",
    ]
    fact = fact.merge(obs[obs_cols], on="fecha_hora", how="left")

    # --- Eventos críticos: conteos B1 segmentados ---
    ev_cols = [
        "fecha_hora",
        "n_forzados_termicos_activos",
        "n_forzados_transmision_activos",
        "n_forzados_termicos_inicio_3h",
        "n_mantenimientos_prog_termicos_activos",
    ]
    fact = fact.merge(ev[ev_cols], on="fecha_hora", how="left")

    # --- Auditoría: forzados B1 ---
    ca_b1 = ca[["fecha_hora", "forzado_activo_b1", "forzado_inicio_3h_previas_b1"]].copy()
    fact = fact.merge(ca_b1, on="fecha_hora", how="left")

    # Garantizar NULL estricto en targets cuando etiqueta_observable == False
    mask_no_obs = fact["etiqueta_observable"] == False
    fact.loc[mask_no_obs, "evento_termico_nop"] = None
    fact.loc[mask_no_obs, "inicio_termico_nop"] = None

    # Convertir tipos para Parquet
    for col in ["evento_termico_nop", "inicio_termico_nop"]:
        fact[col] = fact[col].astype("Int32")  # nullable integer
    for col in ["n_centrales_term_nop", "forzado_activo_b1", "forzado_inicio_3h_previas_b1"]:
        if col in fact.columns:
            fact[col] = fact[col].astype("Int64") if col == "n_centrales_term_nop" else fact[col].astype("Int32")

    # Seleccionar columnas finales
    fact = fact[
        [
            "tiempo_key",
            "gen_total_mw",
            "prog_total_mw",
            "term_cara_ejec_mw",
            "term_cara_prog_mw",
            "term_cara_share",
            "energia_nop_mwh",
            "cmg_troncal_mediana",
            "cmg_troncal_p90",
            "cmg_spread_p90_p50",
            "n_barras_troncales_validas",
            "n_forzados_termicos_activos",
            "n_forzados_transmision_activos",
            "n_forzados_termicos_inicio_3h",
            "n_mantenimientos_prog_termicos_activos",
            "forzado_activo_b1",
            "forzado_inicio_3h_previas_b1",
            "etiqueta_observable",
            "motivo_censura",
            "evento_termico_nop",
            "inicio_termico_nop",
            "n_centrales_term_nop",
            "exposicion_bruta_soles",
            "sobrecosto_neto_estimado_soles",
        ]
    ]

    assert len(fact) == 52_608, f"Se esperaban 52608 filas, encontradas {len(fact)}"
    return fact


def build_fact_predicciones_alerta(gold: Path, dim_tiempo: pd.DataFrame) -> pd.DataFrame:
    """
    fact_predicciones_alerta – 16 080 filas (grano tiempo_key + model_version_id).
    """
    pred = read_gold(gold, "ml_outputs/predicciones_alerta_30min")
    pred["fecha_hora"] = pd.to_datetime(pred["fecha_hora"])

    # Unir con dim_tiempo para obtener tiempo_key
    tk = dim_tiempo[["tiempo_key", "fecha_hora"]].copy()
    fact = pred.merge(tk, on="fecha_hora", how="inner")

    # Columnas finales
    fact = fact[
        [
            "tiempo_key",
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
        ]
    ]

    assert len(fact) == 16_080, f"Se esperaban 16080 filas, encontradas {len(fact)}"
    return fact


def build_fact_calidad_deriva_mensual(
    gold: Path,
    dim_tiempo: pd.DataFrame,
    fact_op: pd.DataFrame,
) -> pd.DataFrame:
    """
    fact_calidad_deriva_mensual – 11 filas (2024-01..2024-11).
    Grano anio_mes + model_version_id.
    """
    deriva = read_gold(gold, "ml_outputs/monitoreo_deriva_mensual")

    # Calcular exposición normalizada por día observable desde fact_operacion
    # Necesitamos anio_mes del dim_tiempo
    tk_mes = dim_tiempo[["tiempo_key", "anio_mes", "fecha_hora"]].copy()
    op_with_mes = fact_op[["tiempo_key", "etiqueta_observable", "exposicion_bruta_soles", "sobrecosto_neto_estimado_soles"]].merge(
        tk_mes[["tiempo_key", "anio_mes", "fecha_hora"]], on="tiempo_key", how="inner"
    )

    # Filtrar solo 2024
    op_2024 = op_with_mes[op_with_mes["anio_mes"].str.startswith("2024")].copy()

    # Para cada mes: sum de exposición y sobrecosto, y conteo de días observables
    agg_mes = (
        op_2024.groupby("anio_mes")
        .agg(
            exposicion_bruta_soles_mes=("exposicion_bruta_soles", "sum"),
            sobrecosto_neto_soles_mes=("sobrecosto_neto_estimado_soles", "sum"),
            bloques_etiquetables_fact=(
                "etiqueta_observable",
                lambda s: int(s.sum()),
            ),
        )
        .reset_index()
    )
    # Días observables = bloques etiquetables / 48 (bloques por día)
    agg_mes["dias_observables"] = agg_mes["bloques_etiquetables_fact"] / 48.0
    agg_mes["exposicion_bruta_soles_dia_obs"] = (
        agg_mes["exposicion_bruta_soles_mes"] / agg_mes["dias_observables"]
    ).where(agg_mes["dias_observables"] > 0, None)
    agg_mes["sobrecosto_neto_soles_dia_obs"] = (
        agg_mes["sobrecosto_neto_soles_mes"] / agg_mes["dias_observables"]
    ).where(agg_mes["dias_observables"] > 0, None)

    # Merge con monitoreo_deriva
    fact = deriva.merge(
        agg_mes[
            [
                "anio_mes",
                "exposicion_bruta_soles_dia_obs",
                "sobrecosto_neto_soles_dia_obs",
            ]
        ],
        on="anio_mes",
        how="left",
    )

    # Renombrar para claridad
    fact = fact.rename(columns={"pct_cobertura_etiqueta": "pct_bloques_etiquetables"})

    # Columnas finales
    fact = fact[
        [
            "anio_mes",
            "model_version_id",
            "bloques_totales",
            "bloques_etiquetables",
            "pct_bloques_etiquetables",
            "alerta_baja_cobertura",
            "prevalencia_observada",
            "psi_max_top5",
            "pr_auc_modelo",
            "pr_auc_baseline",
            "f1_modelo_recal_f1",
            "f1_baseline",
            "fp_por_semana_recal_f1",
            "semaforo_deriva",
            "estado_operativo_modelo",
            "exposicion_bruta_soles_dia_obs",
            "sobrecosto_neto_soles_dia_obs",
        ]
    ]

    assert len(fact) == 11, f"Se esperaban 11 filas, encontradas {len(fact)}"
    return fact


# =====================================================================
#  VALIDACIÓN INTERNA
# =====================================================================
def validate_no_orphan_keys(
    dim_tiempo: pd.DataFrame,
    dim_modelo: pd.DataFrame,
    fact_op: pd.DataFrame,
    fact_pred: pd.DataFrame,
    fact_der: pd.DataFrame,
) -> list[str]:
    """Verifica 0 claves huérfanas entre hechos y dimensiones."""
    errors = []

    # fact_operacion -> dim_tiempo (tiempo_key)
    orphans = set(fact_op["tiempo_key"]) - set(dim_tiempo["tiempo_key"])
    if orphans:
        errors.append(f"fact_operacion->dim_tiempo: {len(orphans)} claves huérfanas")

    # fact_predicciones -> dim_tiempo (tiempo_key)
    orphans = set(fact_pred["tiempo_key"]) - set(dim_tiempo["tiempo_key"])
    if orphans:
        errors.append(f"fact_predicciones->dim_tiempo: {len(orphans)} claves huérfanas")

    # fact_predicciones -> dim_modelo (model_version_id)
    orphans = set(fact_pred["model_version_id"]) - set(dim_modelo["model_version_id"])
    if orphans:
        errors.append(f"fact_predicciones->dim_modelo: {len(orphans)} claves huérfanas")

    # fact_calidad -> dim_modelo (model_version_id)
    orphans = set(fact_der["model_version_id"]) - set(dim_modelo["model_version_id"])
    if orphans:
        errors.append(f"fact_calidad->dim_modelo: {len(orphans)} claves huérfanas")

    # fact_calidad -> dim_tiempo (anio_mes)
    dim_meses = set(dim_tiempo["anio_mes"])
    orphans = set(fact_der["anio_mes"]) - dim_meses
    if orphans:
        errors.append(f"fact_calidad->dim_tiempo(anio_mes): {len(orphans)} claves huérfanas")

    return errors


def validate_reconciliation(fact_op: pd.DataFrame, dim_tiempo: pd.DataFrame) -> dict:
    """
    Reconcilia totales anuales de exposición y sobrecosto contra
    los valores esperados de gold_feature_store_30min.
    """
    merged = fact_op[["tiempo_key", "exposicion_bruta_soles", "sobrecosto_neto_estimado_soles"]].merge(
        dim_tiempo[["tiempo_key", "anio"]], on="tiempo_key"
    )
    agg = merged.groupby("anio").agg(
        exp_total=("exposicion_bruta_soles", "sum"),
        sob_total=("sobrecosto_neto_estimado_soles", "sum"),
    )

    result = {}
    errors = []
    for yr in sorted(EXPECTED_EXPOSICION.keys()):
        actual_exp = agg.loc[yr, "exp_total"] if yr in agg.index else 0.0
        actual_sob = agg.loc[yr, "sob_total"] if yr in agg.index else 0.0
        expected_exp = EXPECTED_EXPOSICION[yr]
        expected_sob = EXPECTED_SOBRECOSTO[yr]
        diff_exp = abs(actual_exp - expected_exp)
        diff_sob = abs(actual_sob - expected_sob)

        result[str(yr)] = {
            "exposicion_bruta_esperada": round(expected_exp, 2),
            "exposicion_bruta_actual": round(actual_exp, 2),
            "diff_exposicion": round(diff_exp, 4),
            "sobrecosto_esperado": round(expected_sob, 2),
            "sobrecosto_actual": round(actual_sob, 2),
            "diff_sobrecosto": round(diff_sob, 4),
            "reconcilia_ok": diff_exp < RECON_TOL and diff_sob < RECON_TOL,
        }
        if diff_exp >= RECON_TOL:
            errors.append(f"{yr}: diff exposición = {diff_exp:.4f} S/ (tol={RECON_TOL})")
        if diff_sob >= RECON_TOL:
            errors.append(f"{yr}: diff sobrecosto = {diff_sob:.4f} S/ (tol={RECON_TOL})")

    return {"detalle": result, "errores": errors}


def validate_null_targets(fact_op: pd.DataFrame) -> dict:
    """
    Verifica que cuando etiqueta_observable == False,
    los targets evento_termico_nop e inicio_termico_nop sean NULL.
    """
    no_obs = fact_op[fact_op["etiqueta_observable"] == False]
    violaciones_evento = int(no_obs["evento_termico_nop"].notna().sum())
    violaciones_inicio = int(no_obs["inicio_termico_nop"].notna().sum())
    return {
        "bloques_no_observables": len(no_obs),
        "violaciones_null_evento": violaciones_evento,
        "violaciones_null_inicio": violaciones_inicio,
        "ok": violaciones_evento == 0 and violaciones_inicio == 0,
    }


# =====================================================================
#  MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Tarea 4.1 – Build Star Schema")
    parser.add_argument("--gold-input", required=True, help="Ruta a lake_v2/gold")
    parser.add_argument("--serving-output", required=True, help="Ruta de salida para star_schema")
    parser.add_argument("--summary-out", required=True, help="Ruta del JSON de resumen")
    args = parser.parse_args()

    gold = Path(args.gold_input)
    out = Path(args.serving_output)
    summary_path = Path(args.summary_out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  TAREA 4.1 – Modelo Dimensional Estrella")
    print("=" * 72)

    # ----- DIMENSIONES -----
    print("\n[1/7] Construyendo dim_tiempo_operativo ...")
    dim_tiempo = build_dim_tiempo_operativo(gold)
    write_parquet(dim_tiempo, out, "dim_tiempo_operativo")
    print(f"       ✓ {len(dim_tiempo)} filas")

    print("[2/7] Construyendo dim_central_combustible ...")
    dim_central = build_dim_central_combustible(gold)
    write_parquet(dim_central, out, "dim_central_combustible")
    print(f"       ✓ {len(dim_central)} filas")

    print("[3/7] Construyendo dim_barra_troncal ...")
    dim_barra = build_dim_barra_troncal(gold)
    write_parquet(dim_barra, out, "dim_barra_troncal")
    print(f"       ✓ {len(dim_barra)} filas")

    print("[4/7] Construyendo dim_modelo_version ...")
    dim_modelo = build_dim_modelo_version(gold)
    write_parquet(dim_modelo, out, "dim_modelo_version")
    print(f"       ✓ {len(dim_modelo)} filas")

    # ----- TABLAS DE HECHOS -----
    print("\n[5/7] Construyendo fact_operacion_bloque_30min ...")
    fact_op = build_fact_operacion_bloque_30min(gold, dim_tiempo)
    write_parquet(fact_op, out, "fact_operacion_bloque_30min")
    print(f"       ✓ {len(fact_op)} filas")

    print("[6/7] Construyendo fact_predicciones_alerta ...")
    fact_pred = build_fact_predicciones_alerta(gold, dim_tiempo)
    write_parquet(fact_pred, out, "fact_predicciones_alerta")
    print(f"       ✓ {len(fact_pred)} filas")

    print("[7/7] Construyendo fact_calidad_deriva_mensual ...")
    fact_der = build_fact_calidad_deriva_mensual(gold, dim_tiempo, fact_op)
    write_parquet(fact_der, out, "fact_calidad_deriva_mensual")
    print(f"       ✓ {len(fact_der)} filas")

    # ----- VALIDACIONES -----
    print("\n" + "=" * 72)
    print("  VALIDACIONES INTERNAS")
    print("=" * 72)

    # Claves huérfanas
    orphan_errors = validate_no_orphan_keys(dim_tiempo, dim_modelo, fact_op, fact_pred, fact_der)
    if orphan_errors:
        for e in orphan_errors:
            print(f"  ✗ {e}")
    else:
        print("  ✓ 0 claves huérfanas en todas las FK")

    # NULL en targets
    null_check = validate_null_targets(fact_op)
    if null_check["ok"]:
        print(f"  ✓ NULL preservado en {null_check['bloques_no_observables']} bloques no observables")
    else:
        print(f"  ✗ Violaciones NULL: evento={null_check['violaciones_null_evento']}, inicio={null_check['violaciones_null_inicio']}")

    # Reconciliación contable
    recon = validate_reconciliation(fact_op, dim_tiempo)
    print("\n  RECONCILIACIÓN CONTABLE (tolerancia < 0.01 S/):")
    for yr, detail in recon["detalle"].items():
        status = "✓" if detail["reconcilia_ok"] else "✗"
        print(
            f"    {status} {yr}: exp diff={detail['diff_exposicion']:.4f} S/ | "
            f"sob diff={detail['diff_sobrecosto']:.4f} S/"
        )
    if recon["errores"]:
        for e in recon["errores"]:
            print(f"    ✗ {e}")

    # ----- RESUMEN JSON -----
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gold_input": str(gold),
        "serving_output": str(out),
        "tablas": {
            "dim_tiempo_operativo": {"filas": len(dim_tiempo), "pk": "tiempo_key"},
            "dim_central_combustible": {"filas": len(dim_central), "pk": "central_id"},
            "dim_barra_troncal": {"filas": len(dim_barra), "pk": "barra_id"},
            "dim_modelo_version": {"filas": len(dim_modelo), "pk": "model_version_id"},
            "fact_operacion_bloque_30min": {"filas": len(fact_op), "fk": ["tiempo_key"]},
            "fact_predicciones_alerta": {
                "filas": len(fact_pred),
                "fk": ["tiempo_key", "model_version_id"],
            },
            "fact_calidad_deriva_mensual": {
                "filas": len(fact_der),
                "fk": ["anio_mes", "model_version_id"],
            },
        },
        "validaciones": {
            "claves_huerfanas": {"errores": orphan_errors, "ok": len(orphan_errors) == 0},
            "null_targets": null_check,
            "reconciliacion_contable": recon,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Resumen guardado en: {summary_path}")

    # Exit code
    all_ok = (
        len(orphan_errors) == 0
        and null_check["ok"]
        and len(recon["errores"]) == 0
    )
    if all_ok:
        print("\n✅ Star schema construido y validado exitosamente.")
    else:
        print("\n❌ Se encontraron errores de validación.")
        sys.exit(1)


if __name__ == "__main__":
    main()

