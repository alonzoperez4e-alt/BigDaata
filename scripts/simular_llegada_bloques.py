#!/usr/bin/env python3
"""Productor de flujo (Paso L.1): simula llegada de bloques operativos de 30 min en JSON."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE_DIR = Path("lake_v2/gold/ml_features_30min/anio=2024")
DEFAULT_INBOX_DIR = Path("lake_v2/streaming/inbox")
DEFAULT_SHAP_JSON = Path("evidencias_modelo/explicabilidad_y_costos.json")

FALLBACK_TOP_FEATURES = (
    "ratio_term_ejec_vs_prog",
    "term_cara_share",
    "cmg_troncal_p90_t",
    "cmg_troncal_mediana_t",
    "persistencia_term_nop_2h",
    "cmg_spread_p90_p50_t",
    "brecha_term_cara_mw",
)


def _cargar_top_features(shap_json: Path) -> list[str]:
    if shap_json.is_file():
        try:
            data = json.loads(shap_json.read_text(encoding="utf-8"))
            top5 = [item["feature"] for item in data.get("shap_global_2024", {}).get("top5", [])]
            extra = ["cmg_spread_p90_p50_t", "brecha_term_cara_mw", "n_forzados_termicos_activos", "cmg_zscore_24h"]
            seen = []
            for c in [*top5, *extra]:
                if c not in seen:
                    seen.append(c)
            if seen:
                return seen
        except Exception:
            pass
    return list(FALLBACK_TOP_FEATURES)


def _serializar_fila(row: dict) -> str:
    limpio = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            limpio[k] = None
        elif isinstance(v, pd.Timestamp):
            limpio[k] = v.strftime("%Y-%m-%dT%H:%M:%S")
        elif hasattr(v, "strftime"):
            limpio[k] = v.strftime("%Y-%m-%d")
        elif isinstance(v, (np.bool_, bool)):
            limpio[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            limpio[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            limpio[k] = float(v)
        else:
            limpio[k] = str(v)
    return json.dumps(limpio, ensure_ascii=True, allow_nan=False)


def _escribir_json_atomico(inbox_dir: Path, filename: str, payload: str) -> Path:
    inbox_dir.mkdir(parents=True, exist_ok=True)
    destino = inbox_dir / filename
    tmp = inbox_dir / f".{filename}.tmp"
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, destino)
    return destino


def _preparar_bloques_replay(df: pd.DataFrame, cantidad: int, start_ts: str | None) -> pd.DataFrame:
    df_ord = df.sort_values("fecha_hora").reset_index(drop=True)
    if start_ts:
        corte = pd.Timestamp(start_ts)
        cand = df_ord[df_ord["fecha_hora"] >= corte]
    else:
        # Zona 2024-11 con eventos observables reales y senales completas
        cand = df_ord[
            (df_ord["fecha_hora"] >= pd.Timestamp("2024-11-10 00:30:00"))
            & (df_ord["etiqueta_observable"] == True)
        ]
        if len(cand) < cantidad:
            cand = df_ord[df_ord["etiqueta_observable"] == True]
        if cand.empty:
            cand = df_ord

    if cand.empty:
        return df_ord.head(cantidad)
    idx_ini = int(cand.index[0])
    return df_ord.iloc[idx_ini : idx_ini + cantidad].copy()


def _preparar_bloques_estres(
    df: pd.DataFrame,
    cantidad: int,
    multiplicador: float,
    top_features: list[str],
) -> pd.DataFrame:
    df_ord = df.sort_values("fecha_hora").reset_index(drop=True)
    observables = df_ord[(df_ord["etiqueta_observable"] == True) & (df_ord["tiene_a1"] == True)]
    base_row = (observables.iloc[-1] if not observables.empty else df_ord.iloc[-1]).copy()

    # Calcular percentiles extremos p99 sobre el historico 2024
    p99 = {}
    for col in top_features:
        if col in df_ord.columns:
            serie = df_ord[col].dropna().astype(float)
            p99[col] = float(serie.quantile(0.99)) if not serie.empty else 1.0

    # Escenario critico de redespacho forzado termico + congestion nodal
    base_ts = pd.Timestamp(base_row["fecha_hora"])
    filas = []
    for seq in range(cantidad):
        r = base_row.copy()
        ts_seq = base_ts + pd.Timedelta(minutes=30 * seq)
        r["fecha_hora"] = ts_seq
        r["fecha_despacho"] = (ts_seq - pd.Timedelta(minutes=1)).date()
        r["tiene_a1"] = True
        r["flag_a1_disponible_t"] = 1

        if "ratio_term_ejec_vs_prog" in r.index:
            r["ratio_term_ejec_vs_prog"] = max(
                float(r["ratio_term_ejec_vs_prog"] or 1.0) * multiplicador,
                p99.get("ratio_term_ejec_vs_prog", 20.0) * multiplicador,
                25.0,
            )
        if "term_cara_share" in r.index:
            r["term_cara_share"] = min(
                0.95,
                max(
                    float(r["term_cara_share"] or 0.01) * multiplicador,
                    p99.get("term_cara_share", 0.035) * multiplicador,
                    0.08,
                ),
            )
        if "cmg_troncal_p90_t" in r.index:
            r["cmg_troncal_p90_t"] = max(
                float(r["cmg_troncal_p90_t"] or 0.1) * multiplicador,
                p99.get("cmg_troncal_p90_t", 0.90) * max(1.0, multiplicador / 2.0),
            )
        if "cmg_troncal_mediana_t" in r.index:
            r["cmg_troncal_mediana_t"] = max(
                float(r["cmg_troncal_mediana_t"] or 0.1) * multiplicador,
                p99.get("cmg_troncal_mediana_t", 0.18) * multiplicador,
            )
        if "cmg_spread_p90_p50_t" in r.index:
            r["cmg_spread_p90_p50_t"] = max(
                float(r["cmg_spread_p90_p50_t"] or 0.01) * multiplicador,
                p99.get("cmg_spread_p90_p50_t", 0.14) * multiplicador,
            )
        if "brecha_term_cara_mw" in r.index:
            r["brecha_term_cara_mw"] = max(
                float(r["brecha_term_cara_mw"] or 10.0) * multiplicador,
                p99.get("brecha_term_cara_mw", 150.0) * max(1.0, multiplicador / 2.0),
            )
        if "persistencia_term_nop_2h" in r.index:
            r["persistencia_term_nop_2h"] = 5
        if "n_forzados_termicos_activos" in r.index:
            r["n_forzados_termicos_activos"] = max(
                4,
                int(round(p99.get("n_forzados_termicos_activos", 3.0) * max(1.0, multiplicador / 2.0))),
            )
        if "cmg_zscore_24h" in r.index:
            r["cmg_zscore_24h"] = max(3.5, float(r["cmg_zscore_24h"] or 1.0) * multiplicador)
        filas.append(r)

    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de llegada de bloques JSON 30 min")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--inbox-dir", type=Path, default=DEFAULT_INBOX_DIR)
    parser.add_argument("--modo", choices=["replay", "estres"], default="replay")
    parser.add_argument("--cantidad", type=int, default=None)
    parser.add_argument("--intervalo", type=float, default=4.0)
    parser.add_argument("--multiplicador", type=float, default=3.0)
    parser.add_argument("--start-ts", default=None)
    parser.add_argument("--shap-json", type=Path, default=DEFAULT_SHAP_JSON)
    args = parser.parse_args()

    cantidad = args.cantidad if args.cantidad is not None else (5 if args.modo == "replay" else 1)
    if cantidad <= 0:
        raise ValueError("--cantidad debe ser >= 1")

    df = pd.read_parquet(args.source_dir)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    if args.modo == "replay":
        seleccion = _preparar_bloques_replay(df, cantidad=cantidad, start_ts=args.start_ts)
    else:
        top_feats = _cargar_top_features(args.shap_json)
        seleccion = _preparar_bloques_estres(
            df,
            cantidad=cantidad,
            multiplicador=args.multiplicador,
            top_features=top_feats,
        )

    print(
        f"[PRODUCER] modo={args.modo} | cantidad={len(seleccion)} | "
        f"intervalo={args.intervalo}s | inbox={args.inbox_dir}"
    )

    total = len(seleccion)
    for seq, (_, row) in enumerate(seleccion.iterrows(), start=1):
        ts_val = pd.Timestamp(row["fecha_hora"])
        ts_tag = ts_val.strftime("%Y%m%d_%H%M%S")
        fname = f"bloque_{ts_tag}_{seq:03d}.json"
        payload = _serializar_fila(row.to_dict())
        dest = _escribir_json_atomico(args.inbox_dir, fname, payload)
        print(
            f"[{args.modo.upper()}] ({seq}/{total}) archivo={dest.name} | "
            f"fecha_hora={ts_val.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if seq < total and args.intervalo > 0:
            time.sleep(args.intervalo)

    print("[OK] Emision de bloques completada.")


if __name__ == "__main__":
    main()
