"""Tests de validacion para la Seccion 6 (Tarea 4.4)."""

import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SECCION6_PATH = ROOT_DIR / "docs" / "seccion_6_trazabilidad_cierre.md"


class TestSeccion6MD:
    """Valida la completitud tecnica y consistencia de Seccion 6."""

    def test_seccion6_exists_and_not_empty(self):
        assert SECCION6_PATH.exists(), f"No existe {SECCION6_PATH}"
        content = SECCION6_PATH.read_text(encoding="utf-8")
        assert len(content) > 2000, "Seccion 6 parece demasiado corta o incompleta"

    @pytest.mark.parametrize(
        "subseccion",
        [
            "## 6.1",
            "## 6.2",
            "## 6.3",
            "## 6.4",
        ],
    )
    def test_seccion6_contains_subsections(self, subseccion: str):
        content = SECCION6_PATH.read_text(encoding="utf-8")
        assert subseccion in content, f"Falta subseccion {subseccion} en Seccion 6"

    def test_seccion6_table_has_5_required_columns_and_no_empty_cells(self):
        content = SECCION6_PATH.read_text(encoding="utf-8")
        lines = content.splitlines()

        expected_headers = [
            "Objetivo",
            "Hallazgo clave",
            "Decisión operativa",
            "Responsable",
            "Efecto sobre la línea base",
        ]

        header_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("|") and "Objetivo" in line and "Hallazgo clave" in line:
                header_idx = idx
                break

        assert header_idx is not None, "No se encontro la tabla de trazabilidad 6.1"

        header_cells = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
        assert header_cells == expected_headers, (
            f"Columnas de tabla 6.1 no coinciden. Esperadas={expected_headers}, Actuales={header_cells}"
        )

        # Validar fila separadora y filas de datos sin celdas vacias
        sep_line = lines[header_idx + 1].strip()
        assert re.match(r"^\|(\s*:?-{3,}:?\s*\|){5}$", sep_line), (
            f"Separador Markdown invalido en tabla 6.1: {sep_line}"
        )

        data_rows = []
        for line in lines[header_idx + 2 :]:
            stripped = line.strip()
            if not stripped.startswith("|"):
                break
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            assert len(cells) == 5, f"Fila con numero distinto de 5 columnas: {stripped}"
            for col_name, cell_val in zip(expected_headers, cells):
                assert cell_val != "", f"Celda vacia en columna '{col_name}': {stripped}"
            data_rows.append(cells)

        assert len(data_rows) >= 3, (
            f"Tabla 6.1 debe cubrir al menos 3 objetivos/hallazgos; encontradas={len(data_rows)}"
        )

    @pytest.mark.parametrize(
        "cifra",
        [
            "19,08",
            "15,22",
            "2,54%",
            "4,18%",
            "1,97",
            "108,50",
            "0,0542",
            "0,0562",
            "ratio_term_ejec_vs_prog",
            "term_cara_share",
            "cmg_troncal_p90_t",
            "cmg_troncal_mediana_t",
            "persistencia_term_nop_2h",
            "recalibrado_60d_max_fp_3_sem",
            "y_inicio_ventana_2_4h",
            "baseline_tasa_horaria",
            "31,2%",
            "2024-01",
            "2024-04",
        ],
    )
    def test_seccion6_contains_key_figures_and_concepts(self, cifra: str):
        content = SECCION6_PATH.read_text(encoding="utf-8")
        assert cifra in content, f"Falta cifra o concepto clave '{cifra}' en Seccion 6"
