#!/usr/bin/env python3
"""Verificador de paridad funcional y contable entre corrida local y corrida AWS Glue."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_BLOQUES_GRILLA = 52_608
EXPECTED_OBSERVABLES = 36_336
TOL_SOLES = 0.01

EXPECTED_STAR_TABLES = (
    "dim_tiempo_operativo",
    "dim_central_combustible",
    "dim_barra_troncal",
    "dim_modelo_version",
    "fact_operacion_bloque_30min",
    "fact_predicciones_alerta",
    "fact_calidad_deriva_mensual",
)


def _cargar_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verificar_paridad(
    local_evidencias_dir: Path,
    local_modelo_dir: Path,
    local_serving_dir: Path,
    cloud_dir: Path,
    tol_soles: float = TOL_SOLES,
) -> dict:
    """Compara reportes JSON locales vs. nube (S3) y retorna dict PASS/FAIL."""
    errores: list[str] = []
    checks: dict = {}

    # 1) Gold (run_1x_gold.json)
    loc_gold = _cargar_json(local_evidencias_dir / "run_1x_gold.json")
    cld_gold = _cargar_json(cloud_dir / "run_1x_gold.json")

    estado_cld = cld_gold.get("contratos_gold", {}).get("estado_global")
    bloques_cld = cld_gold.get("contratos_gold", {}).get("conteos_tablas_gold", {}).get("gold_feature_store_30min")
    obs_cld = cld_gold.get("calidad", {}).get("gold_observabilidad", {}).get("etiqueta_observable_true")
    bloques_loc = loc_gold.get("contratos_gold", {}).get("conteos_tablas_gold", {}).get("gold_feature_store_30min")
    obs_loc = loc_gold.get("calidad", {}).get("gold_observabilidad", {}).get("etiqueta_observable_true")

    if estado_cld != "PASS":
        errores.append(f"Gold nube estado_global != PASS: {estado_cld}")
    if bloques_cld != EXPECTED_BLOQUES_GRILLA or bloques_cld != bloques_loc:
        errores.append(f"Gold total_bloques divergente: nube={bloques_cld}, local={bloques_loc}, esperado={EXPECTED_BLOQUES_GRILLA}")
    if obs_cld != EXPECTED_OBSERVABLES or obs_cld != obs_loc:
        errores.append(f"Gold etiqueta_observable divergente: nube={obs_cld}, local={obs_loc}, esperado={EXPECTED_OBSERVABLES}")

    checks["gold"] = {
        "estado_global_nube": estado_cld,
        "total_bloques_nube": bloques_cld,
        "etiqueta_observable_nube": obs_cld,
    }

    # 2) Feature Engineering (feature_engineering_summary.json)
    loc_fe_path = local_modelo_dir / "feature_engineering_summary.json"
    cld_fe_path = cloud_dir / "feature_engineering_summary.json"
    if loc_fe_path.exists() and cld_fe_path.exists():
        loc_fe = _cargar_json(loc_fe_path)
        cld_fe = _cargar_json(cld_fe_path)
        if cld_fe.get("total_bloques") != EXPECTED_BLOQUES_GRILLA:
            errores.append(f"FE nube total_bloques={cld_fe.get('total_bloques')} != {EXPECTED_BLOQUES_GRILLA}")
        if cld_fe.get("columnas_prohibidas_presentes"):
            errores.append(f"FE nube columnas prohibidas: {cld_fe.get('columnas_prohibidas_presentes')}")
        for k in ("n_etiquetables_ventana", "positivos_evento_2_4h", "positivos_inicio_2_4h"):
            if cld_fe.get(k) != loc_fe.get(k):
                errores.append(f"FE {k} divergente: nube={cld_fe.get(k)}, local={loc_fe.get(k)}")
        checks["feature_engineering"] = {
            "total_bloques_nube": cld_fe.get("total_bloques"),
            "n_etiquetables_ventana_nube": cld_fe.get("n_etiquetables_ventana"),
            "positivos_evento_2_4h_nube": cld_fe.get("positivos_evento_2_4h"),
        }

    # 3) Serving Star Schema (star_schema_summary.json) + totales economicos 2021-2024
    loc_star = _cargar_json(local_serving_dir / "star_schema_summary.json")
    cld_star = _cargar_json(cloud_dir / "star_schema_summary.json")

    tablas_loc = loc_star.get("tablas", {})
    tablas_cld = cld_star.get("tablas", {})
    conteos_check = {}
    for t in EXPECTED_STAR_TABLES:
        n_loc = tablas_loc.get(t, {}).get("filas")
        n_cld = tablas_cld.get(t, {}).get("filas")
        conteos_check[t] = {"local": n_loc, "nube": n_cld, "coincide": (n_loc is not None and n_loc == n_cld)}
        if n_loc is None or n_cld is None or n_loc != n_cld:
            errores.append(f"Star schema tabla {t} divergente: nube={n_cld}, local={n_loc}")
    checks["star_schema_tablas"] = conteos_check

    recon_loc = loc_star.get("validaciones", {}).get("reconciliacion_contable", {}).get("detalle", {})
    recon_cld = cld_star.get("validaciones", {}).get("reconciliacion_contable", {}).get("detalle", {})
    recon_check = {}
    for anio in ("2021", "2022", "2023", "2024"):
        exp_loc = recon_loc.get(anio, {}).get("exposicion_bruta_actual")
        exp_cld = recon_cld.get(anio, {}).get("exposicion_bruta_actual")
        sob_loc = recon_loc.get(anio, {}).get("sobrecosto_actual")
        sob_cld = recon_cld.get(anio, {}).get("sobrecosto_actual")
        if None in (exp_loc, exp_cld, sob_loc, sob_cld):
            errores.append(f"Faltan totales economicos para anio {anio}")
            continue
        diff_exp = abs(float(exp_cld) - float(exp_loc))
        diff_sob = abs(float(sob_cld) - float(sob_loc))
        ok_anio = (diff_exp < tol_soles) and (diff_sob < tol_soles)
        recon_check[anio] = {
            "diff_exposicion_soles": round(diff_exp, 4),
            "diff_sobrecosto_soles": round(diff_sob, 4),
            "ok": ok_anio,
        }
        if not ok_anio:
            errores.append(
                f"Reconciliacion economica {anio} excede tolerancia ({tol_soles} S/): "
                f"diff_exp={diff_exp:.4f}, diff_sob={diff_sob:.4f}"
            )
    checks["reconciliacion_economica"] = recon_check

    val_cld = cld_star.get("validaciones", {})
    if not val_cld.get("claves_huerfanas", {}).get("ok", False):
        errores.append("Star schema nube reporta claves huerfanas.")
    if not val_cld.get("null_targets", {}).get("ok", False):
        errores.append("Star schema nube reporta violaciones NULL en targets censurados.")

    # 4) Backtest economico 2024 (opcional si presente en ambos)
    loc_bt_path = local_modelo_dir / "backtest_economico_2024.json"
    cld_bt_path = cloud_dir / "backtest_economico_2024.json"
    if loc_bt_path.exists() and cld_bt_path.exists():
        loc_bt = _cargar_json(loc_bt_path)
        cld_bt = _cargar_json(cld_bt_path)
        for reg in ("recalibrado_60d_f1", "recalibrado_60d_max_fp_3_sem"):
            e_loc = loc_bt.get("regimenes", {}).get(reg, {}).get("exposicion_base_soles", {}).get("exposicion_bruta_anticipados")
            e_cld = cld_bt.get("regimenes", {}).get(reg, {}).get("exposicion_base_soles", {}).get("exposicion_bruta_anticipados")
            if e_loc is not None and e_cld is not None and abs(float(e_cld) - float(e_loc)) >= tol_soles:
                errores.append(f"Backtest {reg} divergente: nube={e_cld}, local={e_loc}")

    estado = "PASS" if not errores else "FAIL"
    return {
        "estado": estado,
        "tolerancia_soles": tol_soles,
        "checks": checks,
        "errores": errores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verificar paridad local vs. AWS Glue")
    parser.add_argument("--local-evidencias", type=Path, default=Path("evidencias"))
    parser.add_argument("--local-modelo", type=Path, default=Path("evidencias_modelo"))
    parser.add_argument("--local-serving", type=Path, default=Path("evidencias_serving"))
    parser.add_argument("--cloud-dir", type=Path, default=Path("evidencias/aws/run_outputs"))
    parser.add_argument("--out", type=Path, default=Path("evidencias/aws/paridad_nube.json"))
    parser.add_argument("--tol-soles", type=float, default=TOL_SOLES)
    args = parser.parse_args()

    resultado = verificar_paridad(
        local_evidencias_dir=args.local_evidencias,
        local_modelo_dir=args.local_modelo,
        local_serving_dir=args.local_serving,
        cloud_dir=args.cloud_dir,
        tol_soles=args.tol_soles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0 if resultado["estado"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
