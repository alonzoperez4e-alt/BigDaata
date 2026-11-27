"""
Tarea 4.2 – Pruebas de la capa de consultas del Dashboard (`src/dashboard/queries.py`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.queries import (
    get_active_data_source_label,
    get_auditoria_modelo,
    get_cobertura_y_deriva,
    get_exposicion_normalizada,
    get_kpis_ejecutivos,
    get_serie_alerta_30min,
    get_storage_config,
)

QUERIES_FILE = REPO_ROOT / "src" / "dashboard" / "queries.py"
SUMMARY_JSON = REPO_ROOT / "evidencias_serving" / "star_schema_summary.json"
RECON_TOL = 0.01


def test_queries_no_direct_medallion_references():
    """queries.py no debe referenciar directamente bronze/, silver/ ni gold/."""
    source = QUERIES_FILE.read_text(encoding="utf-8").lower()
    for forbidden in ("bronze/", "silver/", "gold/"):
        assert forbidden not in source, (
            f"Referencia prohibida '{forbidden}' encontrada en src/dashboard/queries.py"
        )


@pytest.mark.parametrize(
    "strategy", ["recalibrado_60d_max_fp_3_sem", "recalibrado_60d_f1"]
)
def test_kpis_ejecutivos_validos(strategy):
    kpis = get_kpis_ejecutivos(strategy=strategy, alpha=0.10)
    assert kpis["tp"] > 0
    assert kpis["fp"] >= 0
    assert kpis["dias_observables_2024"] > 0
    assert kpis["exposicion_bruta_dia_obs_2024"] > 0
    assert kpis["sobrecosto_neto_dia_obs_2024"] > 0
    assert kpis["ahorro_neto_soles"] > 0


def test_reconciliacion_economica_vs_summary():
    """Totales anuales y globales en queries.py deben cuadrar (< 0.01 S/) con star_schema_summary.json."""
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    detalle = summary["validaciones"]["reconciliacion_contable"]["detalle"]

    df_anual = get_exposicion_normalizada("anual")
    assert set(df_anual["anio"].tolist()) == {2021, 2022, 2023, 2024}

    for _, row in df_anual.iterrows():
        yr = str(int(row["anio"]))
        exp_expected = detalle[yr]["exposicion_bruta_esperada"]
        sob_expected = detalle[yr]["sobrecosto_esperado"]
        assert abs(row["exposicion_bruta_anual"] - exp_expected) < RECON_TOL
        assert abs(row["sobrecosto_neto_anual"] - sob_expected) < RECON_TOL

    kpis = get_kpis_ejecutivos()
    assert abs(kpis["exposicion_bruta_2024_soles"] - detalle["2024"]["exposicion_bruta_esperada"]) < RECON_TOL
    assert abs(kpis["sobrecosto_neto_2024_soles"] - detalle["2024"]["sobrecosto_esperado"]) < RECON_TOL


def test_cobertura_y_deriva_alertas_enero_abril_2024():
    """2024-01 y 2024-04 deben marcar alerta_baja_cobertura == True (<50%)."""
    df_cov = get_cobertura_y_deriva()
    assert len(df_cov) == 11

    alert_months = set(df_cov[df_cov["alerta_baja_cobertura"] == True]["anio_mes"].tolist())
    assert alert_months == {"2024-01", "2024-04"}

    jan = df_cov[df_cov["anio_mes"] == "2024-01"].iloc[0]
    apr = df_cov[df_cov["anio_mes"] == "2024-04"].iloc[0]
    assert jan["pct_bloques_etiquetables"] == pytest.approx(32.26, abs=0.01)
    assert apr["pct_bloques_etiquetables"] == pytest.approx(37.08, abs=0.01)
    assert jan["estado_operativo_modelo"] == "MODO_MONITOREO_DESCRIPTIVO"
    assert apr["estado_operativo_modelo"] == "MODO_MONITOREO_DESCRIPTIVO"


def test_serie_alerta_30min_normaliza_timestamps_y_censura():
    df = get_serie_alerta_30min("2024-01-01", "2024-01-07", strategy="recalibrado_60d_max_fp_3_sem")
    assert not df.empty
    assert str(df["fecha_hora"].dtype).startswith("datetime64")

    no_obs = df[df["etiqueta_observable"] == False]
    if not no_obs.empty:
        assert no_obs["evento_termico_nop"].isna().all()
        assert no_obs["inicio_termico_nop"].isna().all()


def test_auditoria_modelo_retorna_11_meses():
    df_aud = get_auditoria_modelo("recalibrado_60d_max_fp_3_sem")
    assert len(df_aud) == 11
    assert "pr_auc_modelo" in df_aud.columns
    assert "pr_auc_baseline" in df_aud.columns
    assert "fp_por_semana_estrategia" in df_aud.columns


def test_storage_config_defaults_to_local(monkeypatch):
    for var in (
        "COES_S3_BUCKET",
        "COES_S3_PREFIX",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = get_storage_config()
    assert cfg["mode"] == "local"
    assert cfg["bucket"] == ""
    assert get_active_data_source_label() == "Local (lake_v2)"


def test_storage_config_resolves_s3_from_env(monkeypatch):
    monkeypatch.setenv("COES_S3_BUCKET", "coes-sein-bigdata-datalake")
    monkeypatch.delenv("COES_S3_PREFIX", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    cfg = get_storage_config()
    assert cfg["mode"] == "s3"
    assert cfg["bucket"] == "coes-sein-bigdata-datalake"
    assert cfg["s3_uri_prefix"] == "s3://coes-sein-bigdata-datalake/serving/star_schema/"
    assert cfg["aws_region"] == "us-east-1"
    assert get_active_data_source_label() == "AWS S3 (Read-Only · TLS)"


def test_dashboard_requirements_excludes_heavy_ml():
    req_path = REPO_ROOT / "src" / "dashboard" / "requirements.txt"
    assert req_path.is_file()
    content = req_path.read_text(encoding="utf-8").lower()
    for pkg in ("duckdb", "streamlit", "plotly", "pandas", "pyarrow"):
        assert pkg in content, f"Falta {pkg} en src/dashboard/requirements.txt"
    for heavy in ("pyspark", "shap", "scikit-learn", "matplotlib"):
        assert heavy not in content, f"Dependencia pesada '{heavy}' prohibida en src/dashboard/requirements.txt"


def test_shap_assets_exist_for_cloud():
    from src.dashboard.app import SHAP_FIG_FILENAMES, _resolve_shap_figure

    for fname in SHAP_FIG_FILENAMES:
        resolved = _resolve_shap_figure(fname)
        assert resolved is not None and resolved.is_file(), f"Falta grafica SHAP: {fname}"
        assert (REPO_ROOT / "evidencias" / "aws" / "run_outputs" / "shap" / fname).is_file()


def test_configure_s3_duckdb_timeouts_y_load_primero():
    import duckdb
    from src.dashboard.queries import _configure_s3_duckdb

    executed = []

    class _FakeCon:
        def execute(self, sql):
            executed.append(sql.strip())

    cfg = {
        "aws_region": "us-east-1",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
    }
    _configure_s3_duckdb(_FakeCon(), cfg)
    assert executed[0] == "LOAD httpfs;"
    assert "SET http_timeout = 5000; SET http_retries = 1;" in executed
    assert all("INSTALL httpfs" not in s for s in executed)

    executed_fallback = []

    class _FakeConMissingExt:
        def execute(self, sql):
            executed_fallback.append(sql.strip())
            if sql.strip() == "LOAD httpfs;":
                raise duckdb.IOException("Extension httpfs not found")

    _configure_s3_duckdb(_FakeConMissingExt(), cfg)
    assert executed_fallback[0] == "LOAD httpfs;"
    assert executed_fallback[1] == "INSTALL httpfs; LOAD httpfs;"
    assert "SET http_timeout = 5000; SET http_retries = 1;" in executed_fallback


def test_get_alertas_streaming_distingue_403_vs_404(monkeypatch, caplog):
    import duckdb
    import src.dashboard.queries as q

    monkeypatch.setenv("COES_S3_BUCKET", "coes-sein-bigdata-datalake")
    monkeypatch.setattr(q, "_configure_s3_duckdb", lambda con, cfg: None)

    class _FakeConError:
        def __init__(self, msg):
            self._msg = msg

        def execute(self, sql, params=None):
            raise duckdb.IOException(self._msg)

        def close(self):
            pass

    monkeypatch.setattr(q.duckdb, "connect", lambda database=":memory:": _FakeConError("HTTP Error: 403 Forbidden"))
    with caplog.at_level("INFO"):
        df_403 = q.get_alertas_streaming()
    assert df_403.empty
    assert df_403.attrs.get("error_kind") == "403_FORBIDDEN"
    assert "403" in caplog.text

    caplog.clear()
    monkeypatch.setattr(q.duckdb, "connect", lambda database=":memory:": _FakeConError("No files found that match the pattern"))
    with caplog.at_level("INFO"):
        df_404 = q.get_alertas_streaming()
    assert df_404.empty
    assert df_404.attrs.get("error_kind") == "EMPTY_OR_404"


def test_app_usa_cache_selectivo_en_pestana_5():
    app_src = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "st.cache_data.clear()" not in app_src
    assert "@st.fragment(run_every=5)" in app_src
    assert "@st.cache_data(ttl=3600, show_spinner=False)" in app_src


