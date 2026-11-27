#!/usr/bin/env python3
"""Generador de archivos muestra inspeccionables para el Portal Web de Ingesta."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simular_llegada_bloques import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    _actualizar_variables_temporales,
    _asignar_si_existe,
    _cuantiles_referencia,
)

DEFAULT_SAMPLES_DIR = Path("data/demo_samples")
TARGET_LEAKAGE_COLS = (
    "evento_termico_nop",
    "inicio_termico_nop",
    "y_ventana_2_4h",
    "y_inicio_ventana_2_4h",
    "n_obs_ventana_2_4h",
)


def _limpiar_fuga_futura(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TARGET_LEAKAGE_COLS:
        if col in out.columns:
            out[col] = pd.NA
    if "fecha_hora" in out.columns:
        out["fecha_hora"] = pd.to_datetime(out["fecha_hora"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "fecha_despacho" in out.columns:
        out["fecha_despacho"] = pd.to_datetime(out["fecha_despacho"]).dt.strftime("%Y-%m-%d")
    return out


def _obtener_base_calma(df_ord: pd.DataFrame) -> pd.Series:
    observables = df_ord[(df_ord["etiqueta_observable"] == True) & (df_ord["tiene_a1"] == True)]
    base_calma = observables[
        (observables["evento_termico_nop"] == 0)
        & (observables["brecha_term_cara_mw"].fillna(0.0) <= 0.0)
        & (observables["persistencia_term_nop_2h"].fillna(0.0) == 0.0)
    ]
    if not base_calma.empty:
        return base_calma.iloc[-1].copy()
    if not observables.empty:
        return observables.iloc[-1].copy()
    return df_ord.iloc[-1].copy()


def _construir_muestra_normal(df_ord: pd.DataFrame) -> pd.DataFrame:
    row = _obtener_base_calma(df_ord)
    stats = _cuantiles_referencia(df_ord)
    ts = pd.Timestamp("2024-11-18 09:00:00")
    _actualizar_variables_temporales(row, ts)
    _asignar_si_existe(row, "ratio_term_ejec_vs_prog", 0.91)
    _asignar_si_existe(row, "term_cara_share", min(0.004, stats.get("term_cara_share", {}).get("p25", 0.002)))
    _asignar_si_existe(row, "cmg_troncal_mediana_t", stats.get("cmg_troncal_mediana_t", {}).get("p25", 0.11))
    _asignar_si_existe(row, "cmg_troncal_p90_t", stats.get("cmg_troncal_p90_t", {}).get("p25", 0.12))
    _asignar_si_existe(row, "cmg_spread_p90_p50_t", 0.008)
    _asignar_si_existe(row, "brecha_term_cara_mw", 0.0)
    _asignar_si_existe(row, "rampa_brecha_2h", 0.0)
    _asignar_si_existe(row, "persistencia_term_nop_2h", 0)
    _asignar_si_existe(row, "n_forzados_termicos_activos", 0)
    _asignar_si_existe(row, "cmg_zscore_24h", -0.15)
    return _limpiar_fuga_futura(pd.DataFrame([row]))


def _construir_muestra_vigilancia(df_ord: pd.DataFrame) -> pd.DataFrame:
    # Bloque calibrado en banda intermedia VIGILANCIA (0.36 <= prob < 0.45)
    row = _obtener_base_calma(df_ord)
    stats = _cuantiles_referencia(df_ord)
    ts = pd.Timestamp("2024-11-18 10:30:00")
    _actualizar_variables_temporales(row, ts)
    _asignar_si_existe(row, "ratio_term_ejec_vs_prog", 1.02)
    _asignar_si_existe(row, "term_cara_share", stats.get("term_cara_share", {}).get("p50", 0.005))
    _asignar_si_existe(row, "cmg_troncal_mediana_t", stats.get("cmg_troncal_mediana_t", {}).get("p50", 0.14))
    _asignar_si_existe(row, "cmg_troncal_p90_t", stats.get("cmg_troncal_p90_t", {}).get("p50", 0.17))
    _asignar_si_existe(row, "cmg_spread_p90_p50_t", 0.025)
    _asignar_si_existe(row, "brecha_term_cara_mw", 4.0)
    _asignar_si_existe(row, "rampa_brecha_2h", 2.0)
    _asignar_si_existe(row, "persistencia_term_nop_2h", 0)
    _asignar_si_existe(row, "n_forzados_termicos_activos", 1)
    _asignar_si_existe(row, "n_forzados_transmision_activos", 0)
    _asignar_si_existe(row, "n_forzados_termicos_inicio_3h", 0)
    _asignar_si_existe(row, "cmg_zscore_24h", 0.55)
    return _limpiar_fuga_futura(pd.DataFrame([row]))


def _construir_muestra_critica(df_ord: pd.DataFrame) -> pd.DataFrame:
    row = _obtener_base_calma(df_ord)
    stats = _cuantiles_referencia(df_ord)
    ts = pd.Timestamp("2024-11-18 12:00:00")
    _actualizar_variables_temporales(row, ts)
    _asignar_si_existe(
        row,
        "ratio_term_ejec_vs_prog",
        max(stats.get("ratio_term_ejec_vs_prog", {}).get("p99", 21.0) * 3.0, 65.0),
    )
    _asignar_si_existe(
        row,
        "term_cara_share",
        min(0.85, max(stats.get("term_cara_share", {}).get("p99", 0.035) * 3.0, 0.14)),
    )
    _asignar_si_existe(
        row,
        "cmg_troncal_mediana_t",
        max(stats.get("cmg_troncal_mediana_t", {}).get("p99", 0.45) * 1.4, 0.60),
    )
    _asignar_si_existe(
        row,
        "cmg_troncal_p90_t",
        max(stats.get("cmg_troncal_p90_t", {}).get("p99", 0.90) * 1.5, 0.95),
    )
    _asignar_si_existe(
        row,
        "cmg_spread_p90_p50_t",
        max(stats.get("cmg_spread_p90_p50_t", {}).get("p99", 0.14) * 2.5, 0.30),
    )
    _asignar_si_existe(
        row,
        "brecha_term_cara_mw",
        max(stats.get("brecha_term_cara_mw", {}).get("p99", 130.0) * 1.3, 145.0),
    )
    _asignar_si_existe(row, "rampa_brecha_2h", 85.0)
    _asignar_si_existe(row, "persistencia_term_nop_2h", 5)
    _asignar_si_existe(row, "n_forzados_termicos_activos", 5)
    _asignar_si_existe(row, "cmg_zscore_24h", 3.8)
    return _limpiar_fuga_futura(pd.DataFrame([row]))


def _construir_lote_jornada_4h(df_ord: pd.DataFrame) -> pd.DataFrame:
    # 8 bloques consecutivos de 30 min (4 h): 2 normales -> 2 vigilancia -> 2 criticos -> 2 retorno
    stats = _cuantiles_referencia(df_ord)
    base_row = _obtener_base_calma(df_ord)
    base_ts = pd.Timestamp("2024-11-18 07:30:00")

    def _q(col: str, qname: str, default: float) -> float:
        return stats.get(col, {}).get(qname, default)

    perfiles = [
        (0.85, 0.000, _q("cmg_troncal_mediana_t", "p25", 0.10), _q("cmg_troncal_p90_t", "p25", 0.11), 0.005, -5.0, 0, 0, -0.3),
        (0.90, 0.002, _q("cmg_troncal_mediana_t", "p25", 0.11), _q("cmg_troncal_p90_t", "p25", 0.12), 0.008, 0.0, 0, 0, -0.15),
        (1.02, 0.005, _q("cmg_troncal_mediana_t", "p50", 0.14), _q("cmg_troncal_p90_t", "p50", 0.17), 0.024, 4.0, 0, 1, 0.55),
        (1.06, 0.008, _q("cmg_troncal_mediana_t", "p75", 0.15), _q("cmg_troncal_p90_t", "p75", 0.19), 0.028, 7.0, 0, 1, 0.70),
        (55.0, 0.140, 0.58, 0.92, 0.28, 138.0, 4, 4, 3.4),
        (72.0, 0.180, 0.65, 1.05, 0.34, 160.0, 5, 5, 3.9),
        (0.96, 0.004, _q("cmg_troncal_mediana_t", "p50", 0.13), _q("cmg_troncal_p90_t", "p50", 0.15), 0.014, 1.5, 0, 0, 0.10),
        (0.88, 0.001, _q("cmg_troncal_mediana_t", "p25", 0.10), _q("cmg_troncal_p90_t", "p25", 0.11), 0.006, -3.0, 0, 0, -0.25),
    ]

    filas = []
    for seq, (ratio, share, cmg_med, cmg_p90, spread, brecha, pers, forz, zscore) in enumerate(perfiles):
        r = base_row.copy()
        _actualizar_variables_temporales(r, base_ts + pd.Timedelta(minutes=30 * seq))
        _asignar_si_existe(r, "ratio_term_ejec_vs_prog", ratio)
        _asignar_si_existe(r, "term_cara_share", share)
        _asignar_si_existe(r, "cmg_troncal_mediana_t", cmg_med)
        _asignar_si_existe(r, "cmg_troncal_p90_t", cmg_p90)
        _asignar_si_existe(r, "cmg_spread_p90_p50_t", spread)
        _asignar_si_existe(r, "brecha_term_cara_mw", brecha)
        _asignar_si_existe(r, "persistencia_term_nop_2h", pers)
        _asignar_si_existe(r, "n_forzados_termicos_activos", forz)
        _asignar_si_existe(r, "n_forzados_transmision_activos", 0)
        _asignar_si_existe(r, "n_forzados_termicos_inicio_3h", 0)
        _asignar_si_existe(r, "cmg_zscore_24h", zscore)
        filas.append(r)

    return _limpiar_fuga_futura(pd.DataFrame(filas))


def generar_muestras(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    out_dir: Path = DEFAULT_SAMPLES_DIR,
) -> list[Path]:
    df = pd.read_parquet(source_dir)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df_ord = df.sort_values("fecha_hora").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("01_bloque_normal.csv", _construir_muestra_normal(df_ord)),
        ("02_bloque_vigilancia.csv", _construir_muestra_vigilancia(df_ord)),
        ("03_bloque_estres_critico.csv", _construir_muestra_critica(df_ord)),
        ("04_lote_jornada_4h.csv", _construir_lote_jornada_4h(df_ord)),
    ]

    generados: list[Path] = []
    for fname, sub_df in datasets:
        dest = out_dir / fname
        sub_df.to_csv(dest, index=False, encoding="utf-8")
        print(f"[SAMPLE] archivo={dest.as_posix()} | filas={len(sub_df)} | cols={len(sub_df.columns)}")
        generados.append(dest)
    return generados


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera CSVs muestra en data/demo_samples/")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    args = parser.parse_args()
    generar_muestras(source_dir=args.source_dir, out_dir=args.out_dir)
    print("[OK] Muestras generadas correctamente.")


if __name__ == "__main__":
    main()
