"""Tests de validacion para el cierre de Fase 4 y Anexo B (Tarea 4.5)."""

from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
ARCHIVO_MMD = ROOT_DIR / "docs" / "figura_2_1_arquitectura_final.mmd"
ARCHIVO_ANEXO_B = ROOT_DIR / "docs" / "actualizacion_arquitectura_y_anexo_b.md"
ARCHIVO_GUIA = ROOT_DIR / "docs" / "guia_defensa_sustentacion.md"


class TestCierreFase4AnexoB:
    """Valida los artefactos de cierre de arquitectura, Anexo B y guia de defensa."""

    @pytest.mark.parametrize(
        "ruta",
        [
            ARCHIVO_MMD,
            ARCHIVO_ANEXO_B,
            ARCHIVO_GUIA,
        ],
    )
    def test_archivos_existen_y_no_estan_vacios(self, ruta: Path):
        assert ruta.exists(), f"Falta archivo requerido: {ruta}"
        contenido = ruta.read_text(encoding="utf-8")
        assert len(contenido.strip()) > 500, f"Archivo demasiado corto o vacio: {ruta}"

    def test_diagrama_mermaid_sintaxis_y_sin_nube_ni_sinteticos(self):
        contenido = ARCHIVO_MMD.read_text(encoding="utf-8")
        assert contenido.lstrip().startswith("flowchart "), "Debe iniciar con flowchart Mermaid"
        assert "subgraph " in contenido and "end" in contenido
        lowered = contenido.lower()
        for prohibido in ["aws", "azure", "gcp", "redshift", "bigquery", "snowflake", "sintetico", "synthetic"]:
            assert prohibido not in lowered, f"Termino no agnostico o sintético detectado en diagrama: {prohibido}"

    @pytest.mark.parametrize(
        "referencia",
        [
            "star_schema",
            "52 608",
            "16 080",
            "11",
            "rf_v1_2024_walkforward",
            "SHAP",
            "29733",
        ],
    )
    def test_referencias_clave_presentes_en_artefactos_cierre(self, referencia: str):
        texto_total = "\n".join(
            [
                ARCHIVO_MMD.read_text(encoding="utf-8"),
                ARCHIVO_ANEXO_B.read_text(encoding="utf-8"),
                ARCHIVO_GUIA.read_text(encoding="utf-8"),
            ]
        )
        assert referencia in texto_total, f"Falta referencia clave '{referencia}' en artefactos de cierre"

    def test_anexo_b_contiene_checklists_y_consistencia(self):
        contenido = ARCHIVO_ANEXO_B.read_text(encoding="utf-8")
        assert "Anexo A" in contenido
        assert "Anexo B" in contenido
        assert "Figura 2.1" in contenido
        assert "Consistencia Numérica" in contenido or "consistencia" in contenido.lower()
        assert "19,08" in contenido and "15,22" in contenido

    def test_guia_defensa_cubre_cuatro_ejes(self):
        contenido = ARCHIVO_GUIA.read_text(encoding="utf-8")
        assert "Ingeniería de Datos" in contenido
        assert "Modelado" in contenido
        assert "Gobierno del Dato" in contenido
        assert "Dashboard" in contenido
