"""Pruebas automatizadas de Structured Streaming (Capa de Velocidad) y Dashboard (Paso L.4)."""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simular_llegada_bloques import (  # noqa: E402
    _escribir_json_atomico,
    _preparar_bloques_replay,
    _serializar_fila,
)
from src.dashboard.queries import STREAMING_EXPECTED_COLS, get_alertas_streaming  # noqa: E402
from src.streaming.inferencia_streaming import ejecutar_streaming  # noqa: E402

MODEL_PATH = REPO_ROOT / "lake_v2" / "gold" / "ml_outputs" / "modelo_ganador_y_ventana_2_4h"
SCHEMA_SOURCE = REPO_ROOT / "lake_v2" / "gold" / "ml_features_30min" / "anio=2024"


def _depositar_bloques_prueba(inbox_dir: Path, cantidad: int = 2) -> None:
    df = pd.read_parquet(SCHEMA_SOURCE)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    seleccion = _preparar_bloques_replay(df, cantidad=cantidad, start_ts=None)
    for seq, (_, row) in enumerate(seleccion.iterrows(), start=1):
        ts_val = pd.Timestamp(row["fecha_hora"])
        fname = f"bloque_{ts_val.strftime('%Y%m%d_%H%M%S')}_{seq:03d}.json"
        _escribir_json_atomico(inbox_dir, fname, _serializar_fila(row.to_dict()))


def _leer_parquets_salida(out_dir: Path) -> pd.DataFrame:
    files = sorted(out_dir.glob("*.parquet"))
    assert files, f"No se encontraron archivos Parquet en {out_dir}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def test_streaming_microbatch_con_modelo_real(tmp_path):
    os.environ.pop("SPARK_HOME", None)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    inbox = tmp_path / "inbox"
    out_dir = tmp_path / "streaming_alerts"
    ckpt = tmp_path / "checkpoint"
    _depositar_bloques_prueba(inbox, cantidad=2)

    ejecutar_streaming(
        inbox_dir=inbox,
        model_path=MODEL_PATH,
        schema_source=SCHEMA_SOURCE,
        output_dir=out_dir,
        checkpoint_dir=ckpt,
        once=True,
    )

    df_out = _leer_parquets_salida(out_dir)
    assert len(df_out) == 2
    assert (df_out["prob_evento"] >= 0.0).all() and (df_out["prob_evento"] <= 1.0).all()
    assert set(df_out["estado_alerta"]).issubset({"ALERTA_CRITICA", "VIGILANCIA", "NORMAL"})


def test_checkpoint_evita_reprocesar_archivos(tmp_path):
    os.environ.pop("SPARK_HOME", None)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    inbox = tmp_path / "inbox"
    out_dir = tmp_path / "streaming_alerts"
    ckpt = tmp_path / "checkpoint"
    _depositar_bloques_prueba(inbox, cantidad=2)

    # Primera pasada: procesa los 2 archivos iniciales
    ejecutar_streaming(
        inbox_dir=inbox,
        model_path=MODEL_PATH,
        schema_source=SCHEMA_SOURCE,
        output_dir=out_dir,
        checkpoint_dir=ckpt,
        once=True,
    )
    assert len(_leer_parquets_salida(out_dir)) == 2

    # Segunda pasada con mismo inbox y checkpoint: no debe reprocesar ni duplicar filas
    ejecutar_streaming(
        inbox_dir=inbox,
        model_path=MODEL_PATH,
        schema_source=SCHEMA_SOURCE,
        output_dir=out_dir,
        checkpoint_dir=ckpt,
        once=True,
    )
    df_final = _leer_parquets_salida(out_dir)
    assert len(df_final) == 2


def test_get_alertas_streaming_vacio_y_con_datos(tmp_path, monkeypatch):
    monkeypatch.delenv("COES_S3_BUCKET", raising=False)

    empty_dir = tmp_path / "empty_alerts"
    empty_dir.mkdir(parents=True, exist_ok=True)
    df_empty = get_alertas_streaming(serving_dir=empty_dir)
    assert df_empty.empty
    for col in STREAMING_EXPECTED_COLS:
        assert col in df_empty.columns

    pop_dir = tmp_path / "populated_alerts"
    pop_dir.mkdir(parents=True, exist_ok=True)
    df_sample = pd.DataFrame(
        [
            {
                "fecha_hora": dt.datetime(2024, 11, 10, 10, 0),
                "prob_evento": 0.22,
                "umbral_operativo": 0.45,
                "estado_alerta": "NORMAL",
                "batch_id": 0,
                "ingested_at_utc": dt.datetime(2026, 11, 26, 12, 0, 0),
                "modelo_version": "rf_v1_2024_walkforward",
            },
            {
                "fecha_hora": dt.datetime(2024, 11, 10, 10, 30),
                "prob_evento": 0.78,
                "umbral_operativo": 0.45,
                "estado_alerta": "ALERTA_CRITICA",
                "batch_id": 1,
                "ingested_at_utc": dt.datetime(2026, 11, 26, 12, 0, 5),
                "modelo_version": "rf_v1_2024_walkforward",
            },
        ]
    )
    df_sample.to_parquet(pop_dir / "batch_00000.parquet", compression="snappy", index=False)

    df_res = get_alertas_streaming(limit=10, serving_dir=pop_dir)
    assert len(df_res) == 2
    assert df_res.iloc[0]["batch_id"] == 1
    assert df_res.iloc[0]["estado_alerta"] == "ALERTA_CRITICA"
    assert df_res.iloc[1]["batch_id"] == 0


def test_simulador_demo_completa_y_rafaga(tmp_path):
    import json
    import subprocess

    out_demo = tmp_path / "demo_inbox"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "simular_llegada_bloques.py"),
            "--modo",
            "demo_completa",
            "--intervalo",
            "0",
            "--out-dir",
            str(out_demo),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "FASE 1" in proc.stdout and "FASE 4" in proc.stdout

    archivos = sorted(out_demo.glob("*.json"))
    assert len(archivos) == 16

    timestamps = []
    for f in archivos:
        data = json.loads(f.read_text(encoding="utf-8"))
        timestamps.append(pd.Timestamp(data["fecha_hora"]))

    for prev_ts, next_ts in zip(timestamps, timestamps[1:]):
        assert (next_ts - prev_ts) == pd.Timedelta(minutes=30)

    out_rafaga = tmp_path / "rafaga_inbox"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "simular_llegada_bloques.py"),
            "--modo",
            "rafaga",
            "--out-dir",
            str(out_rafaga),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(list(out_rafaga.glob("*.json"))) == 6


def test_muestras_y_portal_ingesta_e2e(tmp_path):
    from src.streaming.portal_ingesta import (
        REQUIRED_MODEL_COLS,
        cargar_dataframe_desde_bytes,
        ingestar_dataframe_a_inbox,
        validar_columnas_entrada,
    )

    samples_dir = REPO_ROOT / "data" / "demo_samples"
    expected_rows = {
        "01_bloque_normal.csv": 1,
        "02_bloque_vigilancia.csv": 1,
        "03_bloque_estres_critico.csv": 1,
        "04_lote_jornada_4h.csv": 8,
    }
    for fname, n_rows in expected_rows.items():
        path = samples_dir / fname
        assert path.is_file(), f"Falta muestra requerida: {path}"
        df_s = cargar_dataframe_desde_bytes(path.read_bytes(), fname)
        assert len(df_s) == n_rows
        assert validar_columnas_entrada(df_s) == []
        for col in REQUIRED_MODEL_COLS:
            assert col in df_s.columns
        assert df_s["evento_termico_nop"].isna().all()
        assert df_s["inicio_termico_nop"].isna().all()

    os.environ.pop("SPARK_HOME", None)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    inbox = tmp_path / "inbox"
    out_dir = tmp_path / "streaming_alerts"
    ckpt = tmp_path / "checkpoint"

    sample_crit = samples_dir / "03_bloque_estres_critico.csv"
    df_crit = cargar_dataframe_desde_bytes(sample_crit.read_bytes(), sample_crit.name)
    json_dest = ingestar_dataframe_a_inbox(
        df_crit,
        inbox_dir=inbox,
        origen_archivo=sample_crit.name,
        suffix="upload",
    )
    assert json_dest.is_file()
    assert json_dest.name.endswith("_upload.json")

    ejecutar_streaming(
        inbox_dir=inbox,
        model_path=MODEL_PATH,
        schema_source=SCHEMA_SOURCE,
        output_dir=out_dir,
        checkpoint_dir=ckpt,
        once=True,
    )

    df_alertas = get_alertas_streaming(limit=10, serving_dir=out_dir)
    assert len(df_alertas) == 1
    assert df_alertas.iloc[0]["origen_archivo"] == "03_bloque_estres_critico.csv"
    assert df_alertas.iloc[0]["estado_alerta"] == "ALERTA_CRITICA"
    assert float(df_alertas.iloc[0]["brecha_term_cara_mw"]) > 100.0
