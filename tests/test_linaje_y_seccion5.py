"""
Tarea 4.3 – Pruebas de linaje y Sección 5 (Ética, Privacidad, Sesgos).
Validación ligera sin levantar Spark.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
LINAJE_CSV = DOCS / "linaje_datos.csv"
SECCION5_MD = DOCS / "seccion_5_etica_linaje_sesgos.md"

REQUIRED_CSV_COLUMNS = [
    "kpi_o_metrica",
    "vista_dashboard",
    "tabla_serving",
    "columna_serving",
    "tabla_gold_o_ml",
    "columna_gold_o_ml",
    "fuente_bronze_coes",
    "transformacion_y_regla",
    "responsable_tecnico",
]

KPI_KEYWORDS = [
    "KPI 1",
    "KPI 2",
    "KPI 3",
    "KPI 4",
    "KPI 5",
]

REQUIRED_SECTION_HEADERS = [
    "5.1",
    "5.2",
    "5.3",
    "5.4",
    "5.5",
]

REQUIRED_FIGURES_AND_CONCEPTS = [
    "29733",
    "31,2%",
    "2024-01",
    "2024-04",
    "101",
    "108,50",
    "1,97",
    "13,28",
]


class TestLinajeCSV:
    """Validaciones sobre docs/linaje_datos.csv."""

    def test_linaje_csv_exists_and_not_empty(self):
        assert LINAJE_CSV.is_file(), f"{LINAJE_CSV} no existe"
        content = LINAJE_CSV.read_text(encoding="utf-8").strip()
        assert len(content) > 100, "linaje_datos.csv está vacío o demasiado corto"

    def test_linaje_csv_has_required_columns(self):
        with open(LINAJE_CSV, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        for col in REQUIRED_CSV_COLUMNS:
            assert col in fieldnames, f"Columna requerida '{col}' faltante en linaje_datos.csv"

    def test_linaje_csv_covers_5_kpis(self):
        with open(LINAJE_CSV, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            all_kpi_col = [row["kpi_o_metrica"] for row in reader]
        joined = " | ".join(all_kpi_col)
        for kpi in KPI_KEYWORDS:
            assert kpi in joined, (
                f"{kpi} no está cubierto en linaje_datos.csv (columna kpi_o_metrica)"
            )

    def test_linaje_csv_minimum_rows(self):
        with open(LINAJE_CSV, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 10, (
            f"linaje_datos.csv tiene solo {len(rows)} filas de datos; se esperan al menos 10"
        )


class TestSeccion5MD:
    """Validaciones sobre docs/seccion_5_etica_linaje_sesgos.md."""

    def test_seccion5_exists_and_not_empty(self):
        assert SECCION5_MD.is_file(), f"{SECCION5_MD} no existe"
        content = SECCION5_MD.read_text(encoding="utf-8").strip()
        assert len(content) > 500, "seccion_5 está vacía o demasiado corta"

    @pytest.mark.parametrize("header", REQUIRED_SECTION_HEADERS)
    def test_seccion5_contains_subsections(self, header):
        content = SECCION5_MD.read_text(encoding="utf-8")
        assert header in content, (
            f"Subsección '{header}' no encontrada en seccion_5_etica_linaje_sesgos.md"
        )

    @pytest.mark.parametrize("concept", REQUIRED_FIGURES_AND_CONCEPTS)
    def test_seccion5_contains_key_figures(self, concept):
        content = SECCION5_MD.read_text(encoding="utf-8")
        assert concept in content, (
            f"Cifra/concepto clave '{concept}' no encontrado en seccion_5_etica_linaje_sesgos.md"
        )

    def test_seccion5_references_linaje_csv(self):
        content = SECCION5_MD.read_text(encoding="utf-8")
        assert "linaje_datos.csv" in content, (
            "seccion_5 debe referenciar linaje_datos.csv"
        )

    def test_seccion5_mentions_forbidden_leakage_cols(self):
        content = SECCION5_MD.read_text(encoding="utf-8")
        for col in ("label_rdo", "alfa", "beta", "gamma", "chm"):
            assert col in content, (
                f"seccion_5 debe mencionar la columna bloqueada '{col}'"
            )

    def test_seccion5_mentions_null_policy(self):
        content = SECCION5_MD.read_text(encoding="utf-8")
        assert "NULL" in content, "seccion_5 debe mencionar la política de NULL estricto"

    def test_seccion5_mentions_human_in_the_loop(self):
        content = SECCION5_MD.read_text(encoding="utf-8").lower()
        assert "human" in content or "consultiva" in content, (
            "seccion_5 debe mencionar la salvaguarda human-in-the-loop o alerta consultiva"
        )
