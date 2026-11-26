"""Pruebas unitarias y estaticas de Terraform, Glue Runner y paridad nube (Fase B)."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.glue_runner import construir_plan_etapas
from scripts.verificar_paridad_nube import verificar_paridad

TF_DIR = REPO_ROOT / "infra" / "terraform"


def test_terraform_no_contiene_s3_full_access():
    tf_files = sorted(TF_DIR.glob("*.tf"))
    assert len(tf_files) >= 5, "Faltan archivos .tf en infra/terraform/"
    for tf_file in tf_files:
        contenido = tf_file.read_text(encoding="utf-8")
        assert "AmazonS3FullAccess" not in contenido, f"Se encontro AmazonS3FullAccess en {tf_file.name}"
        assert not re.search(r'Resource\s*[=:]\s*"\*"', contenido), (
            f"Se encontro Resource = '*' prohibido en {tf_file.name}"
        )


def test_terraform_exige_tls_y_block_public_access():
    main_tf = (TF_DIR / "main.tf").read_text(encoding="utf-8")
    assert "aws:SecureTransport" in main_tf
    assert '"false"' in main_tf
    for campo in (
        "block_public_acls",
        "block_public_policy",
        "ignore_public_acls",
        "restrict_public_buckets",
    ):
        assert re.search(rf"{campo}\s*=\s*true", main_tf), f"Falta {campo} = true en main.tf"


def test_terraform_glue_timeout_y_reintentos():
    glue_tf = (TF_DIR / "glue.tf").read_text(encoding="utf-8")
    assert re.search(r"timeout\s*=\s*30\b", glue_tf)
    assert re.search(r"max_retries\s*=\s*1\b", glue_tf)
    assert re.search(r'glue_version\s*=\s*"4\.0"', glue_tf)


def test_glue_runner_dry_run():
    plan = construir_plan_etapas(bucket="mi-bucket-coes-test", profile="distributed_managed")
    assert [e["nombre"] for e in plan] == [
        "etl_silver_gold",
        "feature_engineering",
        "entrenamiento_modelo",
        "explicabilidad_y_costos",
        "backtest_economico",
        "build_star_schema",
    ]
    assert "--BUCKET" not in plan[0]["argv"]
    assert "--profile" in plan[0]["argv"] and "distributed_managed" in plan[0]["argv"]
    assert "s3://mi-bucket-coes-test/bronze" in plan[0]["argv"]

    # Verificar ejecucion real de CLI --dry-run
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "glue_runner.py"), "--dry-run", "--BUCKET", "mi-bucket-coes-test"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["bucket"] == "mi-bucket-coes-test"
    assert len(payload["etapas"]) == 6


def test_verificar_paridad_nube_detecta_divergencia(tmp_path):
    loc_ev = tmp_path / "local_ev"
    loc_mod = tmp_path / "local_mod"
    loc_srv = tmp_path / "local_srv"
    cloud = tmp_path / "cloud"
    for d in (loc_ev, loc_mod, loc_srv, cloud):
        d.mkdir(parents=True, exist_ok=True)

    # Copiar JSONs reales como caso base identico -> debe retornar PASS
    shutil.copy(REPO_ROOT / "evidencias" / "run_1x_gold.json", loc_ev / "run_1x_gold.json")
    shutil.copy(REPO_ROOT / "evidencias" / "run_1x_gold.json", cloud / "run_1x_gold.json")
    shutil.copy(REPO_ROOT / "evidencias_modelo" / "feature_engineering_summary.json", loc_mod / "feature_engineering_summary.json")
    shutil.copy(REPO_ROOT / "evidencias_modelo" / "feature_engineering_summary.json", cloud / "feature_engineering_summary.json")
    shutil.copy(REPO_ROOT / "evidencias_serving" / "star_schema_summary.json", loc_srv / "star_schema_summary.json")
    shutil.copy(REPO_ROOT / "evidencias_serving" / "star_schema_summary.json", cloud / "star_schema_summary.json")

    res_ok = verificar_paridad(loc_ev, loc_mod, loc_srv, cloud)
    assert res_ok["estado"] == "PASS"
    assert res_ok["errores"] == []

    # Caso FAIL 1: diverge conteo en tabla estrella
    star_bad = json.loads((cloud / "star_schema_summary.json").read_text(encoding="utf-8"))
    star_bad["tablas"]["fact_predicciones_alerta"]["filas"] = 16000
    (cloud / "star_schema_summary.json").write_text(json.dumps(star_bad), encoding="utf-8")

    res_fail_cnt = verificar_paridad(loc_ev, loc_mod, loc_srv, cloud)
    assert res_fail_cnt["estado"] == "FAIL"
    assert any("fact_predicciones_alerta" in e for e in res_fail_cnt["errores"])

    # Caso FAIL 2: restaurar conteo y divergir monto economico 2024 > 0.01 S/
    shutil.copy(REPO_ROOT / "evidencias_serving" / "star_schema_summary.json", cloud / "star_schema_summary.json")
    star_bad_econ = json.loads((cloud / "star_schema_summary.json").read_text(encoding="utf-8"))
    star_bad_econ["validaciones"]["reconciliacion_contable"]["detalle"]["2024"]["exposicion_bruta_actual"] += 5.0
    (cloud / "star_schema_summary.json").write_text(json.dumps(star_bad_econ), encoding="utf-8")

    res_fail_econ = verificar_paridad(loc_ev, loc_mod, loc_srv, cloud)
    assert res_fail_econ["estado"] == "FAIL"
    assert any("2024" in e for e in res_fail_econ["errores"])
