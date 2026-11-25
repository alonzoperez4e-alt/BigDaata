"""
Tarea 4.1 – Tests del Modelo Dimensional Estrella
==================================================
Suite pytest que valida:
  (a) Conteos exactos de filas en 4 dimensiones y 3 hechos
  (b) 0 claves huérfanas en todas las FK
  (c) Preservación de NULL en targets cuando etiqueta_observable == False
  (d) Reconciliación contable exacta (< 0.01 S/) contra gold_feature_store_30min
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
STAR = Path("lake_v2/serving/star_schema")
GOLD = Path("lake_v2/gold")

RECON_TOL = 0.01  # tolerancia en soles


# ---------------------------------------------------------------------------
# Fixtures: cargar tablas una sola vez por sesión
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dim_tiempo() -> pd.DataFrame:
    return pq.read_table(str(STAR / "dim_tiempo_operativo.parquet")).to_pandas()


@pytest.fixture(scope="session")
def dim_central() -> pd.DataFrame:
    return pq.read_table(str(STAR / "dim_central_combustible.parquet")).to_pandas()


@pytest.fixture(scope="session")
def dim_barra() -> pd.DataFrame:
    return pq.read_table(str(STAR / "dim_barra_troncal.parquet")).to_pandas()


@pytest.fixture(scope="session")
def dim_modelo() -> pd.DataFrame:
    return pq.read_table(str(STAR / "dim_modelo_version.parquet")).to_pandas()


@pytest.fixture(scope="session")
def fact_op() -> pd.DataFrame:
    return pq.read_table(str(STAR / "fact_operacion_bloque_30min.parquet")).to_pandas()


@pytest.fixture(scope="session")
def fact_pred() -> pd.DataFrame:
    return pq.read_table(str(STAR / "fact_predicciones_alerta.parquet")).to_pandas()


@pytest.fixture(scope="session")
def fact_der() -> pd.DataFrame:
    return pq.read_table(str(STAR / "fact_calidad_deriva_mensual.parquet")).to_pandas()


@pytest.fixture(scope="session")
def gold_fs() -> pd.DataFrame:
    return pq.read_table(str(GOLD / "gold_feature_store_30min")).to_pandas()


# =====================================================================
# (a) CONTEOS EXACTOS DE FILAS
# =====================================================================
class TestRowCounts:
    """Verifica conteos exactos de filas en dimensiones y hechos."""

    def test_dim_tiempo_operativo(self, dim_tiempo):
        assert len(dim_tiempo) == 52_608, (
            f"dim_tiempo_operativo: esperadas 52608, encontradas {len(dim_tiempo)}"
        )

    def test_dim_central_combustible(self, dim_central):
        assert len(dim_central) == 171, (
            f"dim_central_combustible: esperadas 171, encontradas {len(dim_central)}"
        )

    def test_dim_barra_troncal(self, dim_barra):
        assert len(dim_barra) == 84, (
            f"dim_barra_troncal: esperadas 84, encontradas {len(dim_barra)}"
        )

    def test_dim_modelo_version_not_empty(self, dim_modelo):
        assert len(dim_modelo) >= 1, "dim_modelo_version no debe estar vacía"

    def test_fact_operacion_bloque_30min(self, fact_op):
        assert len(fact_op) == 52_608, (
            f"fact_operacion: esperadas 52608, encontradas {len(fact_op)}"
        )

    def test_fact_predicciones_alerta(self, fact_pred):
        assert len(fact_pred) == 16_080, (
            f"fact_predicciones: esperadas 16080, encontradas {len(fact_pred)}"
        )

    def test_fact_calidad_deriva_mensual(self, fact_der):
        assert len(fact_der) == 11, (
            f"fact_calidad_deriva: esperadas 11, encontradas {len(fact_der)}"
        )


# =====================================================================
# (b) 0 CLAVES HUÉRFANAS EN TODAS LAS FK
# =====================================================================
class TestOrphanKeys:
    """Verifica que no existan claves huérfanas entre hechos y dimensiones."""

    def test_fact_op_to_dim_tiempo(self, fact_op, dim_tiempo):
        orphans = set(fact_op["tiempo_key"]) - set(dim_tiempo["tiempo_key"])
        assert len(orphans) == 0, f"fact_op→dim_tiempo: {len(orphans)} claves huérfanas"

    def test_fact_pred_to_dim_tiempo(self, fact_pred, dim_tiempo):
        orphans = set(fact_pred["tiempo_key"]) - set(dim_tiempo["tiempo_key"])
        assert len(orphans) == 0, f"fact_pred→dim_tiempo: {len(orphans)} claves huérfanas"

    def test_fact_pred_to_dim_modelo(self, fact_pred, dim_modelo):
        orphans = set(fact_pred["model_version_id"]) - set(dim_modelo["model_version_id"])
        assert len(orphans) == 0, f"fact_pred→dim_modelo: {len(orphans)} claves huérfanas"

    def test_fact_der_to_dim_modelo(self, fact_der, dim_modelo):
        orphans = set(fact_der["model_version_id"]) - set(dim_modelo["model_version_id"])
        assert len(orphans) == 0, f"fact_der→dim_modelo: {len(orphans)} claves huérfanas"

    def test_fact_der_to_dim_tiempo_anio_mes(self, fact_der, dim_tiempo):
        dim_meses = set(dim_tiempo["anio_mes"])
        orphans = set(fact_der["anio_mes"]) - dim_meses
        assert len(orphans) == 0, f"fact_der→dim_tiempo(anio_mes): {len(orphans)} claves huérfanas"


# =====================================================================
# (c) PRESERVACIÓN DE NULL EN TARGETS CUANDO etiqueta_observable == False
# =====================================================================
class TestNullPreservation:
    """
    Cuando etiqueta_observable == False, los campos evento_termico_nop
    e inicio_termico_nop deben ser NULL (NaN/None).
    """

    def test_evento_termico_nop_null_when_not_observable(self, fact_op):
        no_obs = fact_op[fact_op["etiqueta_observable"] == False]
        assert len(no_obs) > 0, "Debe haber bloques no observables"
        violaciones = no_obs["evento_termico_nop"].notna().sum()
        assert violaciones == 0, (
            f"{violaciones} bloques no-observables con evento_termico_nop no-NULL"
        )

    def test_inicio_termico_nop_null_when_not_observable(self, fact_op):
        no_obs = fact_op[fact_op["etiqueta_observable"] == False]
        violaciones = no_obs["inicio_termico_nop"].notna().sum()
        assert violaciones == 0, (
            f"{violaciones} bloques no-observables con inicio_termico_nop no-NULL"
        )

    def test_observable_blocks_may_have_values(self, fact_op):
        """Los bloques observables pueden (y deben) tener valores no-NULL."""
        obs = fact_op[fact_op["etiqueta_observable"] == True]
        assert obs["evento_termico_nop"].notna().any(), (
            "Ningún bloque observable tiene evento_termico_nop — verificar datos"
        )


# =====================================================================
# (d) RECONCILIACIÓN CONTABLE EXACTA (< 0.01 S/) VS. GOLD
# =====================================================================
class TestReconciliacionContable:
    """
    Reconcilia totales anuales de exposición bruta y sobrecosto neto
    del star schema contra gold_feature_store_30min.
    """

    EXPECTED_EXPO = {
        2021: 404_599.93,
        2022: 75_030_153.71,
        2023: 175_600_703.10,
        2024: 19_079_554.64,
    }
    EXPECTED_SOB = {
        2021: 239_930.13,
        2022: 15_114_059.99,
        2023: 48_189_150.53,
        2024: 15_223_546.20,
    }

    def _get_annual_totals(self, fact_op, dim_tiempo):
        """Calcula totales anuales uniendo fact con dim_tiempo."""
        merged = fact_op[["tiempo_key", "exposicion_bruta_soles", "sobrecosto_neto_estimado_soles"]].merge(
            dim_tiempo[["tiempo_key", "anio"]], on="tiempo_key"
        )
        return merged.groupby("anio").agg(
            exp=("exposicion_bruta_soles", "sum"),
            sob=("sobrecosto_neto_estimado_soles", "sum"),
        )

    @pytest.mark.parametrize("anio", [2021, 2022, 2023, 2024])
    def test_exposicion_bruta_anual(self, fact_op, dim_tiempo, anio):
        totals = self._get_annual_totals(fact_op, dim_tiempo)
        actual = totals.loc[anio, "exp"]
        expected = self.EXPECTED_EXPO[anio]
        diff = abs(actual - expected)
        assert diff < RECON_TOL, (
            f"{anio}: exposición diff={diff:.4f} S/ (esperado={expected:,.2f}, actual={actual:,.2f})"
        )

    @pytest.mark.parametrize("anio", [2021, 2022, 2023, 2024])
    def test_sobrecosto_neto_anual(self, fact_op, dim_tiempo, anio):
        totals = self._get_annual_totals(fact_op, dim_tiempo)
        actual = totals.loc[anio, "sob"]
        expected = self.EXPECTED_SOB[anio]
        diff = abs(actual - expected)
        assert diff < RECON_TOL, (
            f"{anio}: sobrecosto diff={diff:.4f} S/ (esperado={expected:,.2f}, actual={actual:,.2f})"
        )

    def test_reconcilia_vs_gold_feature_store(self, fact_op, dim_tiempo, gold_fs):
        """
        Validación cruzada directa: sumas totales del star schema
        deben igualar sumas totales del gold_feature_store_30min.
        """
        # Gold totals
        gold_exp = gold_fs["exposicion_bruta_soles"].sum()
        gold_sob = gold_fs["sobrecosto_neto_estimado_soles"].sum()

        # Star schema totals
        star_exp = fact_op["exposicion_bruta_soles"].sum()
        star_sob = fact_op["sobrecosto_neto_estimado_soles"].sum()

        assert abs(star_exp - gold_exp) < RECON_TOL, (
            f"Total exposición: star={star_exp:,.2f} vs gold={gold_exp:,.2f}"
        )
        assert abs(star_sob - gold_sob) < RECON_TOL, (
            f"Total sobrecosto: star={star_sob:,.2f} vs gold={gold_sob:,.2f}"
        )


# =====================================================================
# TESTS ADICIONALES DE INTEGRIDAD
# =====================================================================
class TestIntegridadAdicional:
    """Tests complementarios de integridad del star schema."""

    def test_dim_tiempo_pk_unique(self, dim_tiempo):
        """PK tiempo_key debe ser única."""
        assert dim_tiempo["tiempo_key"].is_unique, "tiempo_key no es única"

    def test_dim_central_pk_unique(self, dim_central):
        """PK central_id debe ser única."""
        assert dim_central["central_id"].is_unique, "central_id no es única"

    def test_dim_barra_pk_unique(self, dim_barra):
        """PK barra_id debe ser única."""
        assert dim_barra["barra_id"].is_unique, "barra_id no es única"

    def test_dim_modelo_pk_unique(self, dim_modelo):
        """PK model_version_id debe ser única."""
        assert dim_modelo["model_version_id"].is_unique, "model_version_id no es única"

    def test_dim_central_a3_counts(self, dim_central):
        """70 centrales con A3, 101 sin A3."""
        with_a3 = (dim_central["central_en_a3"] == True).sum()
        without_a3 = (dim_central["central_en_a3"] == False).sum()
        assert with_a3 == 70, f"Esperadas 70 centrales con A3, encontradas {with_a3}"
        assert without_a3 == 101, f"Esperadas 101 centrales sin A3, encontradas {without_a3}"

    def test_dim_barra_tension_values(self, dim_barra):
        """Solo 220 kV y 500 kV."""
        niveles = set(dim_barra["nivel_tension_kv"].unique())
        assert niveles.issubset({220, 500}), f"Niveles inesperados: {niveles - {220, 500}}"

    def test_fact_pred_strategies_present(self, fact_pred):
        """Ambas estrategias calibradas deben tener datos."""
        assert fact_pred["resultado_f1_60d"].notna().any(), "Sin datos para resultado_f1_60d"
        assert fact_pred["resultado_max_fp3"].notna().any(), "Sin datos para resultado_max_fp3"

    def test_fact_der_months_range(self, fact_der):
        """Meses de 2024-01 a 2024-11."""
        meses = sorted(fact_der["anio_mes"].tolist())
        expected = [f"2024-{m:02d}" for m in range(1, 12)]
        assert meses == expected, f"Meses inesperados: {meses}"

