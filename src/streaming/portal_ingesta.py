#!/usr/bin/env python3
"""
Portal Web Local de Ingesta (Capa de Velocidad - Puerto 8502)
=============================================================
Separado del dashboard público read-only (`src/dashboard/app.py`):
valida y deposita eventos JSON crudos en `lake_v2/streaming/inbox/`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, time as dt_time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reset_demo_streaming import (  # noqa: E402
    ALLOWED_S3_PREFIX,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_INBOX_DIR,
    DEFAULT_OUTPUT_DIR,
    _limpiar_directorio_local,
    _limpiar_prefijo_s3,
)
from scripts.simular_llegada_bloques import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    _actualizar_variables_temporales,
    _escribir_json_atomico,
)
from src.dashboard.queries import get_alertas_streaming  # noqa: E402
from src.ml.feature_engineering import PREDICTOR_COLS  # noqa: E402

DEMO_SAMPLES_DIR = REPO_ROOT / "data" / "demo_samples"
REQUIRED_MODEL_COLS = ["fecha_hora", *PREDICTOR_COLS]
LEAKAGE_COLS = (
    "evento_termico_nop",
    "inicio_termico_nop",
    "y_ventana_2_4h",
    "y_inicio_ventana_2_4h",
    "n_obs_ventana_2_4h",
)
FEATURE_EDITOR_SPECS = (
    ("brecha_term_cara_mw", "Brecha térmica cara (MW)", -200.0, 500.0, 1.0),
    ("cmg_spread_p90_p50_t", "Spread CMg nodal P90-P50 (S//MWh o USD/MWh)", 0.0, 2.0, 0.005),
    ("ratio_term_ejec_vs_prog", "Ratio térmico ejec vs. prog", 0.0, 200.0, 0.1),
    ("term_cara_share", "Participación térmica cara (share)", 0.0, 1.0, 0.005),
    ("cmg_troncal_p90_t", "CMg troncal P90", 0.0, 5.0, 0.01),
    ("persistencia_term_nop_2h", "Persistencia térmica no prog. 2h (bloques)", 0.0, 10.0, 1.0),
)


def cargar_dataframe_desde_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parsea un archivo CSV, JSON o JSONL subido al portal web."""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(BytesIO(raw_bytes))
    if ext in (".json", ".jsonl", ".ndjson"):
        texto = raw_bytes.decode("utf-8").strip()
        if not texto:
            return pd.DataFrame()
        if texto.startswith("["):
            return pd.DataFrame(json.loads(texto))
        if "\n" in texto:
            records = [json.loads(line) for line in texto.splitlines() if line.strip()]
            return pd.DataFrame(records)
        parsed = json.loads(texto)
        return pd.DataFrame([parsed] if isinstance(parsed, dict) else parsed)
    raise ValueError(f"Formato no soportado '{ext}'. Usa .csv, .json o .jsonl.")


def validar_columnas_entrada(df: pd.DataFrame) -> list[str]:
    """Retorna columnas requeridas faltantes para puntuar con PipelineModel."""
    if df is None or df.empty:
        return list(REQUIRED_MODEL_COLS)
    return [col for col in REQUIRED_MODEL_COLS if col not in df.columns]


def _valor_json_seguro(val):
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%dT%H:%M:%S")
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return str(val)


def normalizar_filas_para_ingesta(df: pd.DataFrame, origen_archivo: str) -> list[dict]:
    """Normaliza filas a diccionarios JSON-safe sin fuga futura y con trazabilidad."""
    faltantes = validar_columnas_entrada(df)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {faltantes}")

    trabajo = df.copy()
    trabajo["fecha_hora"] = pd.to_datetime(trabajo["fecha_hora"])
    if "fecha_despacho" in trabajo.columns:
        trabajo["fecha_despacho"] = pd.to_datetime(trabajo["fecha_despacho"], errors="coerce").dt.strftime("%Y-%m-%d")
    for leak in LEAKAGE_COLS:
        if leak in trabajo.columns:
            trabajo[leak] = None

    filas_json: list[dict] = []
    for _, row in trabajo.iterrows():
        registro = {k: _valor_json_seguro(v) for k, v in row.to_dict().items()}
        registro["fecha_hora"] = pd.Timestamp(row["fecha_hora"]).strftime("%Y-%m-%dT%H:%M:%S")
        if not registro.get("fecha_despacho"):
            registro["fecha_despacho"] = (pd.Timestamp(row["fecha_hora"]) - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")
        registro["origen_archivo"] = str(origen_archivo)
        filas_json.append(registro)
    return filas_json


def ingestar_dataframe_a_inbox(
    df: pd.DataFrame,
    inbox_dir: Path,
    origen_archivo: str,
    suffix: str = "upload",
) -> Path:
    """Valida, normaliza y escribe atomicamente evento_<timestamp>_<suffix>.json en inbox/."""
    registros = normalizar_filas_para_ingesta(df, origen_archivo=origen_archivo)
    primer_ts = pd.to_datetime(df["fecha_hora"]).iloc[0]
    ts_tag = pd.Timestamp(primer_ts).strftime("%Y%m%d_%H%M%S")
    base_name = f"evento_{ts_tag}_{suffix}.json"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    if (inbox_dir / base_name).exists():
        nonce = datetime.utcnow().strftime("%H%M%S%f")
        base_name = f"evento_{ts_tag}_{nonce}_{suffix}.json"

    ndjson_payload = "\n".join(json.dumps(r, ensure_ascii=True, allow_nan=False) for r in registros)
    return _escribir_json_atomico(inbox_dir, base_name, ndjson_payload)


@st.cache_data(ttl=3600, show_spinner=False)
def _cargar_bloque_base_2024() -> pd.Series:
    src = REPO_ROOT / DEFAULT_SOURCE_DIR
    df = pd.read_parquet(src)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df_ord = df.sort_values("fecha_hora").reset_index(drop=True)
    calma = df_ord[
        (df_ord["etiqueta_observable"] == True)
        & (df_ord["tiene_a1"] == True)
        & (df_ord["evento_termico_nop"] == 0)
        & (df_ord["brecha_term_cara_mw"].fillna(0.0) <= 0.0)
    ]
    return (calma.iloc[-1] if not calma.empty else df_ord.iloc[-1]).copy()


def _render_pestana_a_upload(inbox_dir: Path) -> None:
    st.subheader("Pestaña A — Subir archivo crudo (CSV / JSON)")
    muestras = sorted(DEMO_SAMPLES_DIR.glob("*.csv")) if DEMO_SAMPLES_DIR.exists() else []

    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Selecciona un archivo crudo (.csv, .json, .jsonl)",
            type=["csv", "json", "jsonl"],
            key="portal_file_uploader",
        )
    with col_sample:
        muestra_elegida = st.selectbox(
            "O carga rápida desde data/demo_samples/",
            options=["(Ninguna)"] + [m.name for m in muestras],
            index=0,
            key="portal_sample_selector",
        )

    df_in: pd.DataFrame | None = None
    nombre_origen = ""
    try:
        if uploaded is not None:
            df_in = cargar_dataframe_desde_bytes(uploaded.getvalue(), uploaded.name)
            nombre_origen = uploaded.name
        elif muestra_elegida != "(Ninguna)":
            sample_path = DEMO_SAMPLES_DIR / muestra_elegida
            df_in = cargar_dataframe_desde_bytes(sample_path.read_bytes(), sample_path.name)
            nombre_origen = sample_path.name
    except Exception as exc:
        st.error(f"Error al parsear el archivo: {exc}")
        return

    if df_in is None:
        st.info("Sube un archivo CSV/JSON o elige una muestra en `data/demo_samples/` para inspeccionarla.")
        return

    st.caption(f"Archivo cargado: `{nombre_origen}` · **{len(df_in)}** fila(s) x **{len(df_in.columns)}** columnas")
    st.dataframe(df_in, use_container_width=True, hide_index=True)

    faltantes = validar_columnas_entrada(df_in)
    if faltantes:
        st.error(f"Validación fallida. Faltan columnas requeridas por el modelo: `{faltantes}`")
        return

    st.success("Esquema validado correctamente contra las 24 variables predictoras + `fecha_hora`.")
    registros_preview = normalizar_filas_para_ingesta(df_in, origen_archivo=nombre_origen)
    with st.expander("Ver JSON crudo exacto que se enviará al flujo", expanded=False):
        st.json(registros_preview if len(registros_preview) > 1 else registros_preview[0])

    if st.button("Ingestar archivo al flujo (inbox/)", type="primary", key="btn_ingest_upload"):
        dest = ingestar_dataframe_a_inbox(df_in, inbox_dir=inbox_dir, origen_archivo=nombre_origen, suffix="upload")
        st.success(f"Archivo depositado en inbox: `{dest.name}` ({len(registros_preview)} registro(s)).")


def _render_pestana_b_editor(inbox_dir: Path) -> None:
    st.subheader("Pestaña B — Inspector y Editor de Bloque en Vivo")
    try:
        base_row = _cargar_bloque_base_2024().copy()
    except Exception as exc:
        st.error(f"No se pudo cargar bloque base desde `{DEFAULT_SOURCE_DIR}`: {exc}")
        return

    c_dt1, c_dt2 = st.columns(2)
    with c_dt1:
        fecha_sel = st.date_input("Fecha del bloque", value=datetime(2024, 11, 19).date(), key="ed_fecha")
    with c_dt2:
        hora_sel = st.time_input("Hora del bloque (UTC)", value=dt_time(12, 0), step=1800, key="ed_hora")

    ts_sel = pd.Timestamp.combine(fecha_sel, hora_sel)
    _actualizar_variables_temporales(base_row, ts_sel)

    st.markdown("**Ajuste interactivo de variables SHAP clave**")
    cols_ed = st.columns(3)
    for idx, (col_name, label, min_v, max_v, step_v) in enumerate(FEATURE_EDITOR_SPECS):
        default_raw = base_row.get(col_name, 0.0)
        default_val = 0.0 if pd.isna(default_raw) else float(default_raw)
        default_val = min(max(default_val, min_v), max_v)
        with cols_ed[idx % 3]:
            nuevo_val = st.number_input(
                label,
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default_val),
                step=float(step_v),
                key=f"ed_{col_name}",
            )
            base_row[col_name] = int(round(nuevo_val)) if col_name == "persistencia_term_nop_2h" else float(nuevo_val)

    df_edit = pd.DataFrame([base_row])
    registros_ed = normalizar_filas_para_ingesta(df_edit, origen_archivo="editor_en_vivo")
    with st.expander("JSON crudo resultante en vivo", expanded=True):
        st.json(registros_ed[0])

    if st.button("Enviar bloque editado al flujo", type="primary", key="btn_ingest_editor"):
        dest = ingestar_dataframe_a_inbox(df_edit, inbox_dir=inbox_dir, origen_archivo="editor_en_vivo", suffix="upload")
        st.success(f"Bloque editado enviado a inbox: `{dest.name}`")


def _ejecutar_simulador(modo: str, intervalo: float, inbox_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "simular_llegada_bloques.py"),
        "--modo",
        modo,
        "--intervalo",
        str(intervalo),
        "--inbox-dir",
        str(inbox_dir),
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)


def _render_pestana_c_simulador(inbox_dir: Path, output_dir: Path, checkpoint_dir: Path) -> None:
    st.subheader("Pestaña C — Simulador de Escenarios y Reinicio")
    intervalo_sim = st.slider(
        "Intervalo entre bloques para demo_completa (segundos)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.5,
        key="sim_intervalo",
    )
    incluir_s3 = st.checkbox("Limpiar también prefijo S3 (`serving/streaming_alerts/`)", value=True, key="sim_reset_s3")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Lanzar demo_completa (16 bloques)", use_container_width=True, type="primary"):
            with st.spinner("Depositando los 16 bloques de 4 fases..."):
                res = _ejecutar_simulador("demo_completa", intervalo_sim, inbox_dir)
            if res.returncode == 0:
                st.success("Simulación demo_completa finalizada.")
            else:
                st.error(f"Error en simulador (code={res.returncode}).")
            st.code(res.stdout + ("\n" + res.stderr if res.stderr else ""), language="text")

    with c2:
        if st.button("Lanzar rafaga (6 bloques)", use_container_width=True):
            with st.spinner("Depositando ráfaga de 6 bloques..."):
                res = _ejecutar_simulador("rafaga", 0.0, inbox_dir)
            if res.returncode == 0:
                st.success("Ráfaga de 6 bloques depositada.")
            else:
                st.error(f"Error en ráfaga (code={res.returncode}).")
            st.code(res.stdout + ("\n" + res.stderr if res.stderr else ""), language="text")

    with c3:
        if st.button("Limpiar demo (local + S3)", use_container_width=True):
            n_in = _limpiar_directorio_local(inbox_dir, recrear=True)
            n_ck = _limpiar_directorio_local(checkpoint_dir, recrear=False)
            n_out = _limpiar_directorio_local(output_dir, recrear=True)
            msg = f"Local limpio: inbox={n_in}, checkpoint={n_ck}, streaming_alerts={n_out}."
            if incluir_s3:
                bucket = os.environ.get("COES_S3_BUCKET", "coes-sein-bigdata-datalake")
                try:
                    n_s3 = _limpiar_prefijo_s3(bucket=bucket, s3_prefix=ALLOWED_S3_PREFIX)
                    msg += f" S3 limpio: {n_s3} objeto(s) eliminados en s3://{bucket}/{ALLOWED_S3_PREFIX}/."
                except Exception as exc:
                    st.warning(f"{msg} Aviso S3: {exc}")
                    return
            st.success(msg)


@st.fragment(run_every=3)
def _panel_trazabilidad_vivo(inbox_dir: Path, output_dir: Path) -> None:
    st.divider()
    st.subheader("Panel de Trazabilidad en Vivo (refresco automático cada 3 s)")
    st.caption(f"Última comprobación: **{datetime.now().strftime('%H:%M:%S')}**")

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("**1. Últimos archivos depositados en `inbox/`**")
        if inbox_dir.exists():
            archivos = sorted(
                [p for p in inbox_dir.glob("*.json") if not p.name.startswith(".")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:15]
        else:
            archivos = []
        if not archivos:
            st.info("Sin archivos JSON en `lake_v2/streaming/inbox/`.")
        else:
            df_in_list = pd.DataFrame(
                [
                    {
                        "archivo": p.name,
                        "bytes": p.stat().st_size,
                        "modificado": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for p in archivos
                ]
            )
            st.dataframe(df_in_list, use_container_width=True, hide_index=True)

    with col_out:
        st.markdown("**2. Últimos registros procesados por Spark (`streaming_alerts/`)**")
        df_proc = get_alertas_streaming(limit=25, serving_dir=output_dir)
        if df_proc.empty:
            st.info("Sin micro-lotes procesados todavía en `lake_v2/serving/streaming_alerts/`.")
        else:
            cols = [
                c
                for c in (
                    "fecha_hora",
                    "origen_archivo",
                    "batch_id",
                    "prob_evento",
                    "estado_alerta",
                    "brecha_term_cara_mw",
                    "cmg_spread_p90_p50_t",
                )
                if c in df_proc.columns
            ]
            st.dataframe(df_proc[cols], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="COES — Portal Web Local de Ingesta (Puerto 8502)",
        layout="wide",
    )
    st.title("COES — Portal Web Local de Ingesta (Productor Streaming)")
    st.caption(
        "Productor local desacoplado (`:8502`): valida CSV/JSON y deposita eventos en "
        "`lake_v2/streaming/inbox/`. El dashboard público (`app.py`) mantiene IAM Read-Only."
    )

    inbox_dir = (REPO_ROOT / DEFAULT_INBOX_DIR).resolve()
    output_dir = (REPO_ROOT / DEFAULT_OUTPUT_DIR).resolve()
    checkpoint_dir = (REPO_ROOT / DEFAULT_CHECKPOINT_DIR).resolve()

    tab_a, tab_b, tab_c = st.tabs(
        [
            "A. Subir archivo crudo (CSV / JSON)",
            "B. Inspector y Editor de Bloque",
            "C. Simulador de Escenarios",
        ]
    )
    with tab_a:
        _render_pestana_a_upload(inbox_dir)
    with tab_b:
        _render_pestana_b_editor(inbox_dir)
    with tab_c:
        _render_pestana_c_simulador(inbox_dir, output_dir, checkpoint_dir)

    _panel_trazabilidad_vivo(inbox_dir, output_dir)


if __name__ == "__main__":
    main()
