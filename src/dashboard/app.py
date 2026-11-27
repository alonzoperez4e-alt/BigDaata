"""
Tarea 4.2 – Dashboard Operativo-Analítico COES (Streamlit + DuckDB + Plotly)
===========================================================================
Ejecución:
    streamlit run src/dashboard/app.py -- --serving-dir lake_v2/serving/star_schema
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.queries import (  # noqa: E402
    DEFAULT_SERVING_DIR,
    get_active_data_source_label,
    get_alertas_streaming,
    get_auditoria_modelo,
    get_cobertura_y_deriva,
    get_exposicion_normalizada,
    get_kpis_ejecutivos,
    get_serie_alerta_30min,
)

SHAP_FIG_FILENAMES = (
    "fig_4_3_shap_top5_global.png",
    "fig_4_3_shap_beeswarm_2024.png",
)


def _resolve_shap_figure(filename: str) -> Path | None:
    """Resuelve ruta de graficas SHAP relativa a REPO_ROOT con fallback a evidencias AWS."""
    candidatos = (
        REPO_ROOT / "evidencias_modelo" / "shap" / filename,
        REPO_ROOT / "evidencias" / "aws" / "run_outputs" / "shap" / filename,
    )
    for path in candidatos:
        if path.is_file():
            return path
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_kpis(strategy: str, alpha: float, serving_dir: str) -> dict:
    return get_kpis_ejecutivos(strategy=strategy, alpha=alpha, serving_dir=serving_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_serie(fecha_inicio: date, fecha_fin: date, strategy: str, serving_dir: str) -> pd.DataFrame:
    return get_serie_alerta_30min(fecha_inicio, fecha_fin, strategy=strategy, serving_dir=serving_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_exposicion(granularidad: str, serving_dir: str) -> pd.DataFrame:
    return get_exposicion_normalizada(granularidad, serving_dir=serving_dir)  # type: ignore[arg-type]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_cobertura(serving_dir: str) -> pd.DataFrame:
    return get_cobertura_y_deriva(serving_dir=serving_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_auditoria(strategy: str, serving_dir: str) -> pd.DataFrame:
    return get_auditoria_modelo(strategy=strategy, serving_dir=serving_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serving-dir", default=str(DEFAULT_SERVING_DIR))
    args, _ = parser.parse_known_args()
    return args


def _render_kpis(kpis: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Detección Anticipada (Recall)",
        f"{kpis['recall_bloques_pct']:.2f}%",
        f"Precisión: {kpis['precision_bloques_pct']:.2f}% (TP={kpis['tp']})",
    )
    c2.metric(
        "Falsas Alarmas / Semana",
        f"{kpis['fp_por_semana']:.2f}",
        f"Total FP 2024: {kpis['fp']}",
        delta_color="inverse",
    )
    c3.metric(
        "Sobrecosto Neto / Día Obs. (2024)",
        f"S/ {kpis['sobrecosto_neto_dia_obs_2024']:,.2f}",
        f"Exp. bruta: S/ {kpis['exposicion_bruta_dia_obs_2024']:,.2f}/día",
    )
    c4.metric(
        f"Ahorro Potencial (α={int(kpis['alpha'] * 100)}%)",
        f"S/ {kpis['ahorro_neto_soles']:,.0f}",
        f"{kpis['pct_ahorro_neto_2024']:.2f}% sobre sobrecosto 2024",
    )
    c5.metric(
        "Cobertura Promedio 2024",
        f"{kpis['cobertura_promedio_pct']:.2f}%",
        f"{kpis['meses_baja_cobertura']} meses < 50% | PSI máx: {kpis['psi_max_2024']:.2f}",
    )


def _render_vista_1(df_serie: pd.DataFrame, strategy: str) -> None:
    st.subheader("Vista 1 — Sala de Alerta Temprana (Horizonte 2–4 h)")
    if df_serie.empty:
        st.warning("No hay datos para el rango de fechas seleccionado.")
        return

    # Aviso si falta disponibilidad de señal en t
    bloques_sin_senal = int((~df_serie["senial_disponible_t"]).sum())
    if bloques_sin_senal > 0:
        st.warning(
            f"⚠️ Disponibilidad parcial en t: {bloques_sin_senal} bloques de "
            f"{len(df_serie)} presentan ausencia de señal de despacho programado (A1) o CMg troncal (A4)."
        )

    bloques_censurados = int((~df_serie["etiqueta_observable"]).sum())
    if bloques_censurados > 0:
        st.info(
            f"ℹ️ Transparencia de censura: {bloques_censurados} bloques en la ventana "
            f"tienen `etiqueta_observable = False` (sombreados en gris; target preservado en NULL)."
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"Probabilidad Recalibrada vs. Umbral Dinámico ({strategy})",
            "Estrés Nodal Troncal: Spread CMg P90 - P50 (S/ / kWh)",
        ),
    )

    # Zonas sombreadas de censura (etiqueta_observable = False)
    cens = df_serie[~df_serie["etiqueta_observable"]]
    if not cens.empty:
        # Agrupar tramos contiguos para evitar exceso de rectángulos
        cens_idx = cens.index.to_series()
        groups = (cens_idx.diff() > 1).cumsum()
        for _, grp in cens.groupby(groups):
            x0 = grp["fecha_hora"].iloc[0]
            x1 = grp["fecha_hora"].iloc[-1] + pd.Timedelta(minutes=30)
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(150, 150, 150, 0.22)",
                line_width=0,
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=df_serie["fecha_hora"],
            y=df_serie["prob_recalibrada"],
            name="Prob. recalibrada",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_serie["fecha_hora"],
            y=df_serie["umbral_activo"],
            name="Umbral dinámico",
            line=dict(color="#d62728", width=1.8, dash="dash"),
        ),
        row=1,
        col=1,
    )

    ev_obs = df_serie[(df_serie["etiqueta_observable"]) & (df_serie["evento_termico_nop"] == 1)]
    if not ev_obs.empty:
        fig.add_trace(
            go.Scatter(
                x=ev_obs["fecha_hora"],
                y=ev_obs["prob_recalibrada"],
                mode="markers",
                name="Evento térmico NOP observable",
                marker=dict(color="#ff7f0e", size=8, symbol="diamond"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_serie["fecha_hora"],
            y=df_serie["cmg_spread_p90_p50"],
            name="CMg Spread P90-P50",
            line=dict(color="#2ca02c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.12)",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=540, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de últimas alertas con explicabilidad SHAP
    st.markdown("**Últimas alertas disparadas y top-3 factores SHAP**")
    alertas = df_serie[df_serie["alerta_activa"] == 1][
        [
            "fecha_hora",
            "prob_recalibrada",
            "umbral_activo",
            "resultado_activo",
            "cmg_spread_p90_p50",
            "shap_top1",
            "shap_top2",
            "shap_top3",
            "etiqueta_observable",
            "motivo_censura",
        ]
    ].tail(25)
    if alertas.empty:
        st.caption("No se dispararon alertas en la ventana seleccionada.")
    else:
        st.dataframe(alertas, use_container_width=True, hide_index=True)


def _render_vista_2(serving_dir: str, alpha: float, kpis: dict) -> None:
    st.subheader("Vista 2 — Exposición Económica y Sobrecosto Normalizado")
    df_anual = _cached_exposicion("anual", serving_dir)
    df_mensual = _cached_exposicion("mensual", serving_dir)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_anual = go.Figure()
        fig_anual.add_trace(
            go.Bar(
                x=df_anual["anio"].astype(str),
                y=df_anual["exposicion_bruta_por_dia_obs"],
                name="Exposición bruta (S/ / día obs.)",
                marker_color="#1f77b4",
            )
        )
        fig_anual.add_trace(
            go.Bar(
                x=df_anual["anio"].astype(str),
                y=df_anual["sobrecosto_neto_por_dia_obs"],
                name="Sobrecosto neto (S/ / día obs.)",
                marker_color="#d62728",
            )
        )
        fig_anual.update_layout(
            title="Intensidad Económica Anual (S/ por día observable)",
            barmode="group",
            height=380,
        )
        st.plotly_chart(fig_anual, use_container_width=True)

    with col_b:
        fig_mes = go.Figure()
        fig_mes.add_trace(
            go.Scatter(
                x=df_mensual["anio_mes"],
                y=df_mensual["exposicion_bruta_por_dia_obs"],
                name="Exposición bruta / día obs.",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig_mes.add_trace(
            go.Scatter(
                x=df_mensual["anio_mes"],
                y=df_mensual["sobrecosto_neto_por_dia_obs"],
                name="Sobrecosto neto / día obs.",
                line=dict(color="#d62728", width=2),
            )
        )
        fig_mes.update_layout(
            title="Evolución Mensual Normalizada (S/ por día observable)",
            height=380,
        )
        st.plotly_chart(fig_mes, use_container_width=True)

    st.markdown("**Impacto económico del Backtest 2024 por tasa de mitigación (α)**")
    escenarios = []
    for a_val, nombre in [(0.05, "Conservador (5%)"), (0.10, "Base (10%)"), (0.15, "Optimista (15%)")]:
        k_a = _cached_kpis(kpis["strategy"], a_val, serving_dir)
        escenarios.append(
            {
                "Escenario": nombre,
                "alpha": f"{int(a_val * 100)}%",
                "Ahorro Bruto 2024 (S/)": k_a["ahorro_bruto_soles"],
                "% s/ Exp. Bruta 2024": k_a["pct_ahorro_bruto_2024"],
                "Ahorro Neto 2024 (S/)": k_a["ahorro_neto_soles"],
                "% s/ Sobrecosto Neto 2024": k_a["pct_ahorro_neto_2024"],
                "Activo en Selector": "✓" if abs(a_val - alpha) < 1e-6 else "",
            }
        )
    st.dataframe(pd.DataFrame(escenarios), use_container_width=True, hide_index=True)


def _render_vista_3(serving_dir: str) -> None:
    st.subheader("Vista 3 — Transparencia de Cobertura y Deriva (Gobierno del Dato)")
    df_cov = _cached_cobertura(serving_dir)

    col_a, col_b = st.columns(2)
    with col_a:
        colors = ["#d62728" if flag else "#2ca02c" for flag in df_cov["alerta_baja_cobertura"]]
        fig_cov = go.Figure(
            go.Bar(
                x=df_cov["anio_mes"],
                y=df_cov["pct_bloques_etiquetables"],
                marker_color=colors,
                text=df_cov["pct_bloques_etiquetables"].round(2),
                textposition="outside",
                name="% Bloques etiquetables",
            )
        )
        fig_cov.add_hline(
            y=50.0,
            line_dash="dash",
            line_color="#111111",
            annotation_text="Corte mínimo operativo (50%)",
        )
        fig_cov.update_layout(
            title="Cobertura Mensual de Etiqueta Observable (2024)",
            yaxis_range=[0, 105],
            height=380,
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with col_b:
        fig_psi = make_subplots(specs=[[{"secondary_y": True}]])
        fig_psi.add_trace(
            go.Bar(
                x=df_cov["anio_mes"],
                y=df_cov["psi_max_top5"],
                name="PSI máximo Top-5",
                marker_color="#ff7f0e",
            ),
            secondary_y=False,
        )
        fig_psi.add_trace(
            go.Scatter(
                x=df_cov["anio_mes"],
                y=df_cov["prevalencia_observada"] * 100.0,
                name="Prevalencia observada (%)",
                line=dict(color="#1f77b4", width=2.5),
            ),
            secondary_y=True,
        )
        fig_psi.update_layout(
            title="Deriva de Covariables (PSI máx Top-5) y Prevalencia",
            height=380,
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    st.info(
        "📌 **Nota metodológica de deriva (2024):** El cambio de régimen hidrológico y de despacho "
        "en 2024 frente al histórico de entrenamiento 2021–2023 genera deciles de referencia vacíos en "
        "variables críticas; el cálculo de PSI aplica suavizado `eps = 1e-6` para evitar divisiones por cero. "
        "Los meses `2024-01` (32,26%) y `2024-04` (37,08%) quedan bajo el corte de 50% y pasan a `MODO_MONITOREO_DESCRIPTIVO`."
    )
    st.dataframe(
        df_cov[
            [
                "anio_mes",
                "pct_bloques_etiquetables",
                "alerta_baja_cobertura",
                "motivo_censura_principal",
                "prevalencia_observada",
                "psi_max_top5",
                "semaforo_deriva",
                "estado_operativo_modelo",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_vista_4(serving_dir: str, strategy: str) -> None:
    st.subheader("Vista 4 — Auditoría del Modelo vs. Línea Base")
    df_aud = _cached_auditoria(strategy, serving_dir)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=df_aud["anio_mes"],
                y=df_aud["pr_auc_modelo"],
                name="rf_v1_2024_walkforward (PR-AUC)",
                line=dict(color="#1f77b4", width=2.5),
            )
        )
        fig_pr.add_trace(
            go.Scatter(
                x=df_aud["anio_mes"],
                y=df_aud["pr_auc_baseline"],
                name="baseline_tasa_horaria (PR-AUC)",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
            )
        )
        fig_pr.update_layout(title="PR-AUC Mensual: Modelo vs. Baseline Horaria", height=380)
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_b:
        fig_fp = go.Figure()
        fig_fp.add_trace(
            go.Bar(
                x=df_aud["anio_mes"],
                y=df_aud["fp_por_semana_estrategia"],
                name=f"FP/sem ({strategy})",
                marker_color="#1f77b4",
            )
        )
        fig_fp.add_trace(
            go.Scatter(
                x=df_aud["anio_mes"],
                y=df_aud["fp_por_semana_recal_f1"],
                name="FP/sem (recalibrado_60d_f1)",
                line=dict(color="#d62728", width=2, dash="dot"),
            )
        )
        fig_fp.add_hline(y=3.0, line_dash="dash", line_color="#2ca02c", annotation_text="Meta fatiga ≤ 3 FP/sem")
        fig_fp.update_layout(title="Falsas Alarmas por Semana: Operativa vs. F1-60d", height=380)
        st.plotly_chart(fig_fp, use_container_width=True)

    st.warning(
        "🔍 **Explicación del pico de mayo 2024 (`108.50 FP/sem` en `recalibrado_60d_f1`):** "
        "La baja cobertura de abril 2024 (37,08%) dejó ausencia de positivos etiquetables en la ventana "
        "móvil de calibración de 60 días, colapsando el umbral F1 en mayo. La estrategia operativa "
        "`recalibrado_60d_max_fp_3_sem` acota explícitamente la tasa de falsos positivos y sostiene "
        "`1.97 FP/sem` promedio anual (bloques)."
    )
    st.dataframe(
        df_aud[
            [
                "anio_mes",
                "tp",
                "fp",
                "fn",
                "tn",
                "no_verificable",
                "fp_por_semana_estrategia",
                "fp_por_semana_recal_f1",
                "pr_auc_modelo",
                "pr_auc_baseline",
                "estado_operativo_modelo",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Explicabilidad Global SHAP (2024 — Top-5 Factores)**")
    shap_cols = st.columns(2)
    titulos_shap = (
        "Importancia Global SHAP (Top-10 / Top-5)",
        "Distribución Beeswarm SHAP (2024)",
    )
    for idx, (fname, titulo) in enumerate(zip(SHAP_FIG_FILENAMES, titulos_shap)):
        fig_path = _resolve_shap_figure(fname)
        with shap_cols[idx]:
            if fig_path is not None:
                st.image(str(fig_path), caption=titulo, use_container_width=True)
            else:
                st.caption(f"Gráfica no disponible: {fname}")


def _resolve_streaming_dir(serving_dir: str | Path | None) -> str | Path | None:
    if serving_dir is None:
        return None
    s = str(serving_dir).strip()
    if s.startswith("s3://"):
        return s.replace("serving/star_schema", "serving/streaming_alerts")
    p = Path(s)
    if p.resolve() == DEFAULT_SERVING_DIR.resolve():
        return None
    if p.name == "star_schema":
        return p.parent / "streaming_alerts"
    return p


def _render_vista_5(serving_dir: str | Path | None) -> None:
    st.subheader("Vista 5 — Monitor en Vivo (Capa de Velocidad · Arquitectura Lambda)")
    st.caption(
        "Arquitectura Lambda activa: **Capa por Lotes** (AWS Glue 4.0 · `serving/star_schema/`) + "
        "**Capa de Velocidad** (PySpark Structured Streaming · `serving/streaming_alerts/`)."
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Refrescar flujo en vivo", use_container_width=True):
            st.cache_data.clear()

    stream_dir = _resolve_streaming_dir(serving_dir)
    df_live = get_alertas_streaming(limit=200, serving_dir=stream_dir)

    with col_info:
        st.caption(f"Origen monitorizado: `{get_active_data_source_label(serving_dir)}` · prefijo `serving/streaming_alerts/`")

    if df_live.empty:
        st.info(
            "⏳ Monitor en espera de micro-lotes desde `serving/streaming_alerts/`. "
            "Ejecuta el productor (`simular_llegada_bloques.py`) y el motor de streaming "
            "(`inferencia_streaming.py`) para observar alertas en tiempo real."
        )
        return

    # 4 KPIs en vivo
    n_bloques = len(df_live)
    n_crit = int((df_live["estado_alerta"] == "ALERTA_CRITICA").sum())
    ultima_prob = float(df_live.iloc[0]["prob_evento"])
    ultimo_batch = int(df_live.iloc[0]["batch_id"])
    ultima_ingesta = pd.Timestamp(df_live.iloc[0]["ingested_at_utc"]).strftime("%Y-%m-%d %H:%M:%S")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bloques procesados", f"{n_bloques:,}")
    k2.metric("Alertas críticas", f"{n_crit:,}")
    k3.metric("Última probabilidad", f"{ultima_prob:.4f}")
    k4.metric("Último micro-lote (UTC)", f"batch_{ultimo_batch:05d}", ultima_ingesta)

    # Gráfico de evolución de probabilidad vs. umbral operativo
    df_plot = df_live.sort_values("fecha_hora").reset_index(drop=True)
    umbral_val = float(df_plot["umbral_operativo"].iloc[-1])

    fig_live = go.Figure()
    fig_live.add_trace(
        go.Scatter(
            x=df_plot["fecha_hora"],
            y=df_plot["prob_evento"],
            mode="lines+markers",
            name="Probabilidad evento (streaming)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6, color="#1f77b4"),
        )
    )

    crit_df = df_plot[df_plot["estado_alerta"] == "ALERTA_CRITICA"]
    if not crit_df.empty:
        fig_live.add_trace(
            go.Scatter(
                x=crit_df["fecha_hora"],
                y=crit_df["prob_evento"],
                mode="markers",
                name="ALERTA_CRITICA",
                marker=dict(color="#d62728", size=11, symbol="diamond"),
            )
        )

    fig_live.add_hline(
        y=umbral_val,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Umbral operativo ({umbral_val:.2f})",
    )
    fig_live.update_layout(
        title="Evolución de Probabilidad en Flujo vs. Umbral Operativo",
        yaxis_title="Probabilidad evento (2–4 h)",
        yaxis_range=[0, 1.05],
        height=400,
        legend=dict(orientation="h", y=1.10),
    )
    st.plotly_chart(fig_live, use_container_width=True)

    st.markdown("**Últimos eventos procesados (más reciente primero)**")
    cols_tabla = [
        c
        for c in (
            "fecha_hora",
            "prob_evento",
            "umbral_operativo",
            "estado_alerta",
            "batch_id",
            "ingested_at_utc",
            "modelo_version",
            "ratio_term_ejec_vs_prog",
            "cmg_spread_p90_p50_t",
        )
        if c in df_live.columns
    ]
    st.dataframe(df_live[cols_tabla], use_container_width=True, hide_index=True)


def main() -> None:
    args = _parse_args()
    serving_dir = args.serving_dir

    st.set_page_config(
        page_title="COES — Alerta Temprana de Generación Térmica No Programada",
        layout="wide",
    )
    st.title("⚡ COES — Dashboard Operativo-Analítico de Alerta Temprana (2–4 h)")
    st.caption("Arquitectura agnóstica de proveedor: Apache Parquet + DuckDB + Streamlit + Plotly")

    with st.sidebar:
        st.header("Controles Operativos")
        st.info(f"Origen de datos: **{get_active_data_source_label(serving_dir)}**")
        strategy = st.selectbox(
            "Estrategia de Umbral",
            options=["recalibrado_60d_max_fp_3_sem", "recalibrado_60d_f1"],
            index=0,
            format_func=lambda s: (
                "Operativa (≤3 FP/sem)" if s == "recalibrado_60d_max_fp_3_sem" else "Alta Sensibilidad (F1-60d)"
            ),
        )
        alpha = st.select_slider(
            "Factor de Mitigación Operativa (α)",
            options=[0.05, 0.10, 0.15],
            value=0.10,
            format_func=lambda x: f"{int(x * 100)}%",
        )
        rango_fechas = st.date_input(
            "Ventana de Análisis (Vista 1)",
            value=(date(2024, 10, 1), date(2024, 10, 15)),
            min_value=date(2021, 12, 1),
            max_value=date(2024, 12, 1),
        )
        if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
            fecha_inicio, fecha_fin = rango_fechas
        else:
            fecha_inicio, fecha_fin = date(2024, 10, 1), date(2024, 10, 15)

    kpis = _cached_kpis(strategy=strategy, alpha=alpha, serving_dir=serving_dir)
    _render_kpis(kpis)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Sala de Alerta Temprana (2–4 h)",
            "2. Exposición y Sobrecosto Normalizado",
            "3. Cobertura y Deriva (Gobierno del Dato)",
            "4. Auditoría Modelo vs. Baseline",
            "5. Monitor en Vivo (Capa de Velocidad)",
        ]
    )

    with tab1:
        df_serie = _cached_serie(fecha_inicio, fecha_fin, strategy=strategy, serving_dir=serving_dir)
        _render_vista_1(df_serie, strategy)
    with tab2:
        _render_vista_2(serving_dir, alpha, kpis)
    with tab3:
        _render_vista_3(serving_dir)
    with tab4:
        _render_vista_4(serving_dir, strategy)
    with tab5:
        _render_vista_5(serving_dir)


if __name__ == "__main__":
    main()

