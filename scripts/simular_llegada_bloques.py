#!/usr/bin/env python3
"""Productor de flujo (Paso L.1): simula llegada de bloques operativos de 30 min en JSON."""
from __future__ import annotations

import argparse
import json
import math
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


def _actualizar_variables_temporales(fila: pd.Series, ts_seq: pd.Timestamp) -> None:
    fila["fecha_hora"] = ts_seq
    fila["fecha_despacho"] = (ts_seq - pd.Timedelta(minutes=1)).date()
    fila["tiene_a1"] = True
    fila["flag_a1_disponible_t"] = 1
    hora_dec = ts_seq.hour + (ts_seq.minute / 60.0)
    ang_h = 2.0 * math.pi * (hora_dec / 24.0)
    ang_m = 2.0 * math.pi * (ts_seq.month / 12.0)
    if "hora_sin" in fila.index:
        fila["hora_sin"] = math.sin(ang_h)
    if "hora_cos" in fila.index:
        fila["hora_cos"] = math.cos(ang_h)
    if "mes_sin" in fila.index:
        fila["mes_sin"] = math.sin(ang_m)
    if "mes_cos" in fila.index:
        fila["mes_cos"] = math.cos(ang_m)
    if "es_fin_de_semana" in fila.index:
        fila["es_fin_de_semana"] = 1 if ts_seq.weekday() >= 5 else 0


def _cuantiles_referencia(df_ord: pd.DataFrame) -> dict[str, dict[str, float]]:
    cols = [
        "ratio_term_ejec_vs_prog",
        "term_cara_share",
        "cmg_troncal_p90_t",
        "cmg_troncal_mediana_t",
        "cmg_spread_p90_p50_t",
        "brecha_term_cara_mw",
        "n_forzados_termicos_activos",
        "cmg_zscore_24h",
    ]
    stats: dict[str, dict[str, float]] = {}
    for col in cols:
        if col in df_ord.columns:
            serie = df_ord[col].dropna().astype(float)
            if not serie.empty:
                stats[col] = {
                    "p25": float(serie.quantile(0.25)),
                    "p50": float(serie.quantile(0.50)),
                    "p75": float(serie.quantile(0.75)),
                    "p90": float(serie.quantile(0.90)),
                    "p99": float(serie.quantile(0.99)),
                }
    return stats


def _asignar_si_existe(fila: pd.Series, col: str, valor: float | int) -> None:
    if col in fila.index:
        fila[col] = valor


def _preparar_bloques_demo_completa(
    df: pd.DataFrame,
    cantidad: int = 16,
    start_ts: str | None = None,
    multiplicador: float = 3.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye una jornada continua de 8 horas (16 bloques de 30 min) en 4 fases:
      1) Operacion estable (NORMAL)
      2) Degradacion progresiva (VIGILANCIA)
      3) Contingencia severa p99+ (ALERTA_CRITICA)
      4) Mitigacion y retorno a normalidad (NORMAL)
    """
    df_ord = df.sort_values("fecha_hora").reset_index(drop=True)
    observables = df_ord[(df_ord["etiqueta_observable"] == True) & (df_ord["tiene_a1"] == True)]
    base_calma = observables[
        (observables["evento_termico_nop"] == 0)
        & (observables["brecha_term_cara_mw"].fillna(0.0) <= 0.0)
    ]
    base_row = (
        base_calma.iloc[-1]
        if not base_calma.empty
        else (observables.iloc[-1] if not observables.empty else df_ord.iloc[-1])
    ).copy()

    stats = _cuantiles_referencia(df_ord)
    base_ts = pd.Timestamp(start_ts) if start_ts else pd.Timestamp("2024-11-15 08:00:00")

    # Dividir en 4 fases (para 16 bloques -> 4 + 4 + 4 + 4)
    n1 = max(1, cantidad // 4)
    n2 = max(1, cantidad // 4)
    n3 = max(1, cantidad // 4)
    n4 = max(1, cantidad - (n1 + n2 + n3))
    fases_plan = (
        [("FASE 1 - Operacion estable (NORMAL)", 1, i, n1) for i in range(n1)]
        + [("FASE 2 - Degradacion progresiva (VIGILANCIA)", 2, i, n2) for i in range(n2)]
        + [("FASE 3 - Contingencia severa (ALERTA_CRITICA)", 3, i, n3) for i in range(n3)]
        + [("FASE 4 - Mitigacion y retorno (NORMAL)", 4, i, n4) for i in range(n4)]
    )[:cantidad]

    def _q(col: str, qname: str, default: float) -> float:
        return stats.get(col, {}).get(qname, default)

    filas: list[pd.Series] = []
    fases_etiquetas: list[str] = []

    for seq, (etiqueta_fase, fase_id, idx_fase, total_fase) in enumerate(fases_plan):
        r = base_row.copy()
        ts_seq = base_ts + pd.Timedelta(minutes=30 * seq)
        _actualizar_variables_temporales(r, ts_seq)
        progreso = (idx_fase + 1) / max(1, total_fase)

        if fase_id == 1:
            _asignar_si_existe(r, "ratio_term_ejec_vs_prog", 0.90 + 0.03 * progreso)
            _asignar_si_existe(r, "term_cara_share", min(0.005, _q("term_cara_share", "p25", 0.002)))
            _asignar_si_existe(r, "cmg_troncal_mediana_t", _q("cmg_troncal_mediana_t", "p25", 0.11))
            _asignar_si_existe(r, "cmg_troncal_p90_t", _q("cmg_troncal_p90_t", "p25", 0.12))
            _asignar_si_existe(r, "cmg_spread_p90_p50_t", max(0.005, _q("cmg_spread_p90_p50_t", "p25", 0.01)))
            _asignar_si_existe(r, "brecha_term_cara_mw", 0.0)
            _asignar_si_existe(r, "rampa_brecha_2h", 0.0)
            _asignar_si_existe(r, "persistencia_term_nop_2h", 0)
            _asignar_si_existe(r, "n_forzados_termicos_activos", 0)
            _asignar_si_existe(r, "cmg_zscore_24h", -0.2 + 0.15 * progreso)
        elif fase_id == 2:
            _asignar_si_existe(r, "ratio_term_ejec_vs_prog", 1.02 + 0.45 * progreso)
            _asignar_si_existe(
                r,
                "term_cara_share",
                _q("term_cara_share", "p50", 0.005) + 0.015 * progreso,
            )
            _asignar_si_existe(
                r,
                "cmg_troncal_mediana_t",
                _q("cmg_troncal_mediana_t", "p75", 0.16) * (1.0 + 0.25 * progreso),
            )
            _asignar_si_existe(
                r,
                "cmg_troncal_p90_t",
                _q("cmg_troncal_p90_t", "p75", 0.20) * (1.0 + 0.45 * progreso),
            )
            _asignar_si_existe(
                r,
                "cmg_spread_p90_p50_t",
                _q("cmg_spread_p90_p50_t", "p75", 0.03) + 0.05 * progreso,
            )
            _asignar_si_existe(r, "brecha_term_cara_mw", 8.0 + 28.0 * progreso)
            _asignar_si_existe(r, "rampa_brecha_2h", 6.0 + 18.0 * progreso)
            _asignar_si_existe(r, "persistencia_term_nop_2h", 1 if idx_fase >= 2 else 0)
            _asignar_si_existe(r, "n_forzados_termicos_activos", 1 if idx_fase < 2 else 2)
            _asignar_si_existe(r, "cmg_zscore_24h", 0.75 + 0.75 * progreso)
        elif fase_id == 3:
            amp = max(1.5, multiplicador) * (0.9 + 0.25 * progreso)
            _asignar_si_existe(
                r,
                "ratio_term_ejec_vs_prog",
                max(_q("ratio_term_ejec_vs_prog", "p99", 21.0) * amp, 40.0 + 25.0 * progreso),
            )
            _asignar_si_existe(
                r,
                "term_cara_share",
                min(0.90, max(_q("term_cara_share", "p99", 0.035) * amp, 0.12)),
            )
            _asignar_si_existe(
                r,
                "cmg_troncal_mediana_t",
                max(_q("cmg_troncal_mediana_t", "p99", 0.45) * 1.4, 0.55),
            )
            _asignar_si_existe(
                r,
                "cmg_troncal_p90_t",
                max(_q("cmg_troncal_p90_t", "p99", 0.90) * 1.6, 0.95),
            )
            _asignar_si_existe(
                r,
                "cmg_spread_p90_p50_t",
                max(_q("cmg_spread_p90_p50_t", "p99", 0.14) * amp, 0.25),
            )
            _asignar_si_existe(
                r,
                "brecha_term_cara_mw",
                max(_q("brecha_term_cara_mw", "p99", 130.0) * 1.25, 130.0 + 30.0 * progreso),
            )
            _asignar_si_existe(r, "rampa_brecha_2h", 60.0 + 30.0 * progreso)
            _asignar_si_existe(r, "persistencia_term_nop_2h", 4 if idx_fase == 0 else 5)
            _asignar_si_existe(r, "n_forzados_termicos_activos", 4 + (idx_fase % 2))
            _asignar_si_existe(r, "cmg_zscore_24h", 3.2 + 0.8 * progreso)
        else:
            desc = max(0.0, 1.0 - progreso)
            _asignar_si_existe(r, "ratio_term_ejec_vs_prog", 0.92 + 0.35 * desc)
            _asignar_si_existe(r, "term_cara_share", 0.002 + 0.012 * desc)
            _asignar_si_existe(
                r,
                "cmg_troncal_mediana_t",
                _q("cmg_troncal_mediana_t", "p25", 0.11) + 0.10 * desc,
            )
            _asignar_si_existe(
                r,
                "cmg_troncal_p90_t",
                _q("cmg_troncal_p90_t", "p25", 0.12) + 0.16 * desc,
            )
            _asignar_si_existe(r, "cmg_spread_p90_p50_t", 0.01 + 0.04 * desc)
            _asignar_si_existe(r, "brecha_term_cara_mw", 18.0 * desc)
            _asignar_si_existe(r, "rampa_brecha_2h", -20.0 * (1.0 - desc))
            _asignar_si_existe(r, "persistencia_term_nop_2h", max(0, 2 - idx_fase))
            _asignar_si_existe(r, "n_forzados_termicos_activos", max(0, 1 - idx_fase))
            _asignar_si_existe(r, "cmg_zscore_24h", -0.1 + 0.9 * desc)

        filas.append(r)
        fases_etiquetas.append(etiqueta_fase)

    return pd.DataFrame(filas), fases_etiquetas


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de llegada de bloques JSON 30 min")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--inbox-dir", "--out-dir", dest="inbox_dir", type=Path, default=DEFAULT_INBOX_DIR)
    parser.add_argument(
        "--modo",
        choices=["replay", "estres", "demo_completa", "rafaga"],
        default="replay",
    )
    parser.add_argument("--cantidad", type=int, default=None)
    parser.add_argument("--intervalo", type=float, default=4.0)
    parser.add_argument("--multiplicador", type=float, default=3.0)
    parser.add_argument("--start-ts", default=None)
    parser.add_argument("--shap-json", type=Path, default=DEFAULT_SHAP_JSON)
    args = parser.parse_args()

    defaults_cantidad = {"replay": 5, "estres": 1, "demo_completa": 16, "rafaga": 6}
    cantidad = args.cantidad if args.cantidad is not None else defaults_cantidad[args.modo]
    if cantidad <= 0:
        raise ValueError("--cantidad debe ser >= 1")

    intervalo_efectivo = 0.0 if args.modo == "rafaga" else float(args.intervalo)

    df = pd.read_parquet(args.source_dir)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    fases_etiquetas: list[str] | None = None
    if args.modo in ("replay", "rafaga"):
        seleccion = _preparar_bloques_replay(df, cantidad=cantidad, start_ts=args.start_ts)
    elif args.modo == "demo_completa":
        seleccion, fases_etiquetas = _preparar_bloques_demo_completa(
            df,
            cantidad=cantidad,
            start_ts=args.start_ts,
            multiplicador=args.multiplicador,
        )
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
        f"intervalo={intervalo_efectivo}s | inbox={args.inbox_dir}"
    )

    total = len(seleccion)
    for seq, (_, row) in enumerate(seleccion.iterrows(), start=1):
        ts_val = pd.Timestamp(row["fecha_hora"])
        ts_tag = ts_val.strftime("%Y%m%d_%H%M%S")
        fname = f"bloque_{ts_tag}_{seq:03d}.json"
        payload = _serializar_fila(row.to_dict())
        dest = _escribir_json_atomico(args.inbox_dir, fname, payload)
        fase_txt = f" | fase={fases_etiquetas[seq - 1]}" if fases_etiquetas else ""
        print(
            f"[{args.modo.upper()}] ({seq}/{total}){fase_txt} | "
            f"archivo={dest.name} | fecha_hora={ts_val.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if seq < total and intervalo_efectivo > 0:
            time.sleep(intervalo_efectivo)

    print("[OK] Emision de bloques completada.")


if __name__ == "__main__":
    main()
