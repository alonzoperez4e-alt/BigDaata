"""Suite de regresion y contratos de la capa Gold.
Ejecuta los 6 contratos sobre data/sample/ y sobre el conjunto completo 1x (coes_pipeline/lake_real),
ademas de pruebas negativas de regresion para la barrera antifuga y censura.
"""
import json
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.etl.gold_feature_store import COLUMNAS_PROHIBIDAS_ANTIFUGA
from src.etl.quality_checks import (
    BLOQUES_GRILLA_ESPERADOS,
    BLOQUES_OBSERVABLES_PR06_1X_ESPERADOS,
    FILAS_A2_1X_ESPERADAS,
    exigir_contratos_gold,
    validar_barrera_antifuga_columnas,
    validar_contratos_gold,
)

LAKES_A_EVALUAR = [pytest.param("data/sample", id="sample")]
if (REPO_ROOT / "coes_pipeline" / "lake_real" / "gold" / "gold_feature_store_30min").exists():
    LAKES_A_EVALUAR.append(pytest.param("coes_pipeline/lake_real", id="completo_1x"))


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.appName("pytest-gold-contracts")
        .master("local[4]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.mark.parametrize("lake_rel", LAKES_A_EVALUAR)
def test_contratos_gold_end_to_end(spark, lake_rel):
    lake = REPO_ROOT / lake_rel
    res = validar_contratos_gold(spark, str(lake / "silver"), str(lake / "gold"), escala="1x")

    assert res["estado_global"] == "PASS", f"Contratos fallidos en {lake_rel}: {res['contratos_fallidos']}"
    assert res["contratos"]["1_completitud_grilla"]["filas_observadas"] == BLOQUES_GRILLA_ESPERADOS
    assert res["contratos"]["1_completitud_grilla"]["duplicados_fecha_hora"] == 0
    assert res["contratos"]["3_integridad_censura"]["violaciones_feature_store"] == 0
    assert res["contratos"]["3_integridad_censura"]["violaciones_observabilidad"] == 0
    assert res["contratos"]["4_barrera_antifuga_esquema"]["columnas_prohibidas_encontradas"] == []
    assert res["contratos"]["6_atomicidad_y_particion_anual"]["tablas_con_particion_mes_residual"] == []

    if lake_rel == "coes_pipeline/lake_real":
        assert res["contratos"]["2_no_multiplicacion_despacho"]["filas_gold_despacho"] == FILAS_A2_1X_ESPERADAS
        assert (
            res["contratos"]["5_consistencia_cobertura_historica"]["etiqueta_observable_en_pr06"]
            == BLOQUES_OBSERVABLES_PR06_1X_ESPERADOS
        )


@pytest.mark.parametrize("lake_rel", LAKES_A_EVALUAR)
def test_reporte_atomico_gold_json(lake_rel):
    ruta_log = REPO_ROOT / lake_rel / "logs" / "run_1x_gold.json"
    assert ruta_log.exists(), f"Falta el reporte atomico {ruta_log}"
    data = json.loads(ruta_log.read_text(encoding="utf-8"))

    assert data.get("particionamiento") == "anio"
    assert "linaje_gold" in data
    assert "contratos_gold" in data
    assert data["contratos_gold"]["estado_global"] == "PASS"
    assert "distribucion_motivo_censura" in data["contratos_gold"]
    assert data["contratos_gold"]["total_archivos_parquet_gold"] > 0


def test_regresion_barrera_antifuga_bloquea_columnas_prohibidas():
    for col_prohibida in sorted(COLUMNAS_PROHIBIDAS_ANTIFUGA):
        ok, encontradas = validar_barrera_antifuga_columnas(["fecha_hora", "gen_total_mw", col_prohibida])
        assert not ok
        assert col_prohibida in encontradas


def test_regresion_exigir_contratos_lanza_error_si_falla():
    with pytest.raises(RuntimeError, match="FALLIDOS"):
        exigir_contratos_gold({"estado_global": "FAIL", "contratos_fallidos": ["4_barrera_antifuga_esquema"]})