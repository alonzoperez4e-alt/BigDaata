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
    get_auditoria_modelo,
    get_cobertura_y_deriva,
    get_exposicion_normalizada,
    get_kpis_ejecutivos,
    get_serie_alerta_30min,
)


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
    df_anual = get_exposicion_normalizada("anual", serving_dir)
    df_mensual = get_exposicion_normalizada("mensual", serving_dir)

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
        k_a = get_kpis_ejecutivos(kpis["strategy"], a_val, serving_dir)
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
    df_cov = get_cobertura_y_deriva(serving_dir)

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
    df_aud = get_auditoria_modelo(strategy, serving_dir)

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

    kpis = get_kpis_ejecutivos(strategy=strategy, alpha=alpha, serving_dir=serving_dir)
    _render_kpis(kpis)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Sala de Alerta Temprana (2–4 h)",
            "2. Exposición y Sobrecosto Normalizado",
            "3. Cobertura y Deriva (Gobierno del Dato)",
            "4. Auditoría Modelo vs. Baseline",
        ]
    )

    with tab1:
        df_serie = get_serie_alerta_30min(fecha_inicio, fecha_fin, strategy=strategy, serving_dir=serving_dir)
        _render_vista_1(df_serie, strategy)
    with tab2:
        _render_vista_2(serving_dir, alpha, kpis)
    with tab3:
        _render_vista_3(serving_dir)
    with tab4:
        _render_vista_4(serving_dir, strategy)


if __name__ == "__main__":
    main()

