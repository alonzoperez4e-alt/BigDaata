"""
Tests del empaquetador Glue (scripts/empaquetar_src_glue.py).

Valida:
  (a) El zip contiene __init__.py y módulos clave.
  (b) No contiene __pycache__, .pyc ni src/dashboard/.
  (c) El SHA-256 del manifiesto coincide con el archivo generado.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import pytest


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def packager_output():
    """Ejecuta el empaquetador en un directorio temporal y retorna (zip_path, manifest_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "src.zip")
        manifest_path = os.path.join(tmpdir, "build_manifest.json")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/empaquetar_src_glue.py",
                "--out-zip", zip_path,
                "--manifest-out", manifest_path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Empaquetador falló: {result.stderr}"
        yield zip_path, manifest_path


def test_zip_contains_init_and_key_modules(packager_output):
    """(a) El zip contiene __init__.py y módulos clave de src/."""
    zip_path, _ = packager_output
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # __init__.py en paquetes raíz
        assert "src/__init__.py" in names, "Falta src/__init__.py en el zip"
        assert "src/etl/__init__.py" in names, "Falta src/etl/__init__.py en el zip"
        assert "src/ml/__init__.py" in names, "Falta src/ml/__init__.py en el zip"
        assert "src/serving/__init__.py" in names, "Falta src/serving/__init__.py en el zip"
        # Módulos clave
        assert "src/etl/pipeline.py" in names, "Falta src/etl/pipeline.py"
        assert "src/ml/feature_engineering.py" in names, "Falta src/ml/feature_engineering.py"
        assert "src/ml/entrenamiento_modelo.py" in names, "Falta src/ml/entrenamiento_modelo.py"
        assert "src/serving/build_star_schema.py" in names, "Falta src/serving/build_star_schema.py"


def test_zip_excludes_pycache_and_dashboard(packager_output):
    """(b) El zip no contiene __pycache__, .pyc ni src/dashboard/."""
    zip_path, _ = packager_output
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        pycache = [n for n in names if "__pycache__" in n]
        assert pycache == [], f"Encontrados archivos __pycache__: {pycache}"
        pyc = [n for n in names if n.endswith(".pyc")]
        assert pyc == [], f"Encontrados archivos .pyc: {pyc}"
        dashboard = [n for n in names if n.startswith("src/dashboard/")]
        assert dashboard == [], f"Encontrados archivos dashboard: {dashboard}"


def test_manifest_sha256_matches_zip(packager_output):
    """(c) El SHA-256 del manifiesto coincide con el archivo generado."""
    zip_path, manifest_path = packager_output
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # El manifiesto debe contener una entrada para el zip
    # La clave puede ser la ruta absoluta del tmpdir; buscamos la que termine en src.zip
    zip_key = None
    for key in manifest:
        if key.endswith("src.zip"):
            zip_key = key
            break
    assert zip_key is not None, f"No se encontró entrada src.zip en manifiesto: {list(manifest.keys())}"

    expected_sha = manifest[zip_key]["sha256"]
    actual_sha = _sha256(zip_path)
    assert actual_sha == expected_sha, (
        f"SHA-256 no coincide: manifiesto={expected_sha[:16]}… real={actual_sha[:16]}…"
    )
