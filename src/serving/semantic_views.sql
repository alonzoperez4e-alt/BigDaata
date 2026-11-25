-- ============================================================================
-- Tarea 4.1 – Capa Semántica: Vistas Analíticas
-- ============================================================================
-- Script SQL estándar compatible con DuckDB / Spark SQL.
-- Registra las 7 tablas Parquet como tablas semánticas y define
-- 4 vistas analíticas listas para el dashboard.
--
-- Uso (DuckDB):
--   .read src/serving/semantic_views.sql
-- Uso (Spark SQL):
--   spark.sql(open('src/serving/semantic_views.sql').read())
--
-- Nota: Las rutas Parquet deben ajustarse según el motor.
--       Los ejemplos usan rutas relativas para DuckDB.
-- ============================================================================

-- ============================================================================
-- 1. REGISTRO DE TABLAS SEMÁNTICAS
-- ============================================================================

-- ---- Dimensiones ----
CREATE OR REPLACE VIEW dim_tiempo_operativo AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/dim_tiempo_operativo.parquet');

CREATE OR REPLACE VIEW dim_central_combustible AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/dim_central_combustible.parquet');

CREATE OR REPLACE VIEW dim_barra_troncal AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/dim_barra_troncal.parquet');

CREATE OR REPLACE VIEW dim_modelo_version AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/dim_modelo_version.parquet');

-- ---- Tablas de hechos ----
CREATE OR REPLACE VIEW fact_operacion_bloque_30min AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/fact_operacion_bloque_30min.parquet');

CREATE OR REPLACE VIEW fact_predicciones_alerta AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/fact_predicciones_alerta.parquet');

CREATE OR REPLACE VIEW fact_calidad_deriva_mensual AS
SELECT * FROM read_parquet('lake_v2/serving/star_schema/fact_calidad_deriva_mensual.parquet');


-- ============================================================================
-- 2. VISTAS ANALÍTICAS PARA EL DASHBOARD
-- ============================================================================

-- ============================================================================
-- 2.1  vw_alerta_temprana_30min
-- Vista principal del dashboard: cada bloque de 30 min con toda la
-- información necesaria para la alerta temprana operativa.
-- Incluye: contexto temporal, probabilidades calibradas, alertas de ambas
-- estrategias, resultado verificado, top-3 SHAP features, métricas de
-- generación, precios troncales, y targets observados.
-- ============================================================================
CREATE OR REPLACE VIEW vw_alerta_temprana_30min AS
SELECT
    t.tiempo_key,
    t.fecha_hora,
    t.fecha_despacho,
    t.bloque_30min,
    t.hora,
    t.dia_semana,
    t.es_fin_semana,
    t.mes,
    t.anio,
    t.anio_mes,
    t.periodo_hidrologico,

    -- Predicciones y alertas
    p.model_version_id,
    p.horizonte,
    p.prob_cruda,
    p.prob_recalibrada,
    p.umbral_f1_60d,
    p.alerta_f1_60d,
    p.resultado_f1_60d,
    p.umbral_max_fp3,
    p.alerta_max_fp3,
    p.resultado_max_fp3,
    p.shap_top1,
    p.shap_top2,
    p.shap_top3,

    -- Contexto operativo
    f.gen_total_mw,
    f.prog_total_mw,
    f.term_cara_ejec_mw,
    f.term_cara_prog_mw,
    f.term_cara_share,
    f.cmg_troncal_mediana,
    f.cmg_troncal_p90,
    f.cmg_spread_p90_p50,

    -- Eventos
    f.n_forzados_termicos_activos,
    f.n_forzados_transmision_activos,
    f.n_forzados_termicos_inicio_3h,
    f.n_mantenimientos_prog_termicos_activos,

    -- Observabilidad y targets
    f.etiqueta_observable,
    f.motivo_censura,
    f.evento_termico_nop,
    f.inicio_termico_nop,

    -- Costos
    f.exposicion_bruta_soles,
    f.sobrecosto_neto_estimado_soles

FROM dim_tiempo_operativo       t
LEFT JOIN fact_predicciones_alerta     p ON t.tiempo_key = p.tiempo_key
LEFT JOIN fact_operacion_bloque_30min  f ON t.tiempo_key = f.tiempo_key;


-- ============================================================================
-- 2.2  vw_exposicion_normalizada_mensual
-- Exposición y sobrecosto acumulado mensual, normalizado por día observable.
-- Permite comparar meses con diferente cobertura de observabilidad.
-- ============================================================================
CREATE OR REPLACE VIEW vw_exposicion_normalizada_mensual AS
SELECT
    t.anio_mes,
    t.anio,

    -- Totales mensuales
    SUM(f.exposicion_bruta_soles)             AS exposicion_bruta_mes,
    SUM(f.sobrecosto_neto_estimado_soles)     AS sobrecosto_neto_mes,

    -- Conteo de bloques observables
    SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END)
                                              AS bloques_etiquetables,
    COUNT(*)                                  AS bloques_totales,
    ROUND(
        CAST(SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) AS DOUBLE)
        / COUNT(*) * 100, 2
    )                                         AS pct_cobertura_etiqueta,

    -- Días observables (48 bloques por día)
    ROUND(
        CAST(SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) AS DOUBLE)
        / 48.0, 2
    )                                         AS dias_observables,

    -- Métricas normalizadas por día observable
    CASE
        WHEN SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) > 0
        THEN ROUND(
            SUM(f.exposicion_bruta_soles)
            / (CAST(SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) AS DOUBLE) / 48.0),
            2
        )
    END                                       AS exposicion_bruta_por_dia_obs,

    CASE
        WHEN SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) > 0
        THEN ROUND(
            SUM(f.sobrecosto_neto_estimado_soles)
            / (CAST(SUM(CASE WHEN f.etiqueta_observable THEN 1 ELSE 0 END) AS DOUBLE) / 48.0),
            2
        )
    END                                       AS sobrecosto_neto_por_dia_obs,

    -- Conteo de eventos en bloques observables
    SUM(f.evento_termico_nop)                 AS eventos_termico_nop_mes,
    SUM(f.inicio_termico_nop)                 AS inicios_termico_nop_mes

FROM dim_tiempo_operativo       t
JOIN fact_operacion_bloque_30min f ON t.tiempo_key = f.tiempo_key
GROUP BY t.anio_mes, t.anio
ORDER BY t.anio_mes;


-- ============================================================================
-- 2.3  vw_cobertura_y_deriva_mensual
-- Vista consolidada de cobertura de observabilidad, deriva del modelo,
-- y estado operativo. Para el panel de monitoreo de modelo.
-- ============================================================================
CREATE OR REPLACE VIEW vw_cobertura_y_deriva_mensual AS
SELECT
    d.anio_mes,
    d.model_version_id,
    m.algoritmo,

    -- Cobertura
    d.bloques_totales,
    d.bloques_etiquetables,
    d.pct_bloques_etiquetables,
    d.alerta_baja_cobertura,

    -- Prevalencia y deriva
    d.prevalencia_observada,
    d.psi_max_top5,
    d.semaforo_deriva,
    d.estado_operativo_modelo,

    -- Métricas del modelo
    d.pr_auc_modelo,
    d.f1_modelo_recal_f1,
    d.fp_por_semana_recal_f1,

    -- Métricas del baseline
    d.pr_auc_baseline,
    d.f1_baseline,

    -- Ventaja del modelo sobre baseline
    ROUND(d.pr_auc_modelo - d.pr_auc_baseline, 6)
                                              AS delta_pr_auc_vs_baseline,
    ROUND(d.f1_modelo_recal_f1 - d.f1_baseline, 6)
                                              AS delta_f1_vs_baseline,

    -- Exposición normalizada
    d.exposicion_bruta_soles_dia_obs,
    d.sobrecosto_neto_soles_dia_obs

FROM fact_calidad_deriva_mensual d
JOIN dim_modelo_version          m ON d.model_version_id = m.model_version_id
ORDER BY d.anio_mes;


-- ============================================================================
-- 2.4  vw_auditoria_modelo_vs_baseline
-- Auditoría comparativa: modelo recalibrado vs. baseline por mes.
-- Muestra distribución de resultados (TP/FP/TN/FN/NO_VERIFICABLE)
-- y métricas complementarias para evaluar valor agregado del modelo.
-- ============================================================================
CREATE OR REPLACE VIEW vw_auditoria_modelo_vs_baseline AS
SELECT
    t.anio_mes,
    p.model_version_id,

    -- Distribución de resultados – estrategia F1
    SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END)
                                              AS tp_f1,
    SUM(CASE WHEN p.resultado_f1_60d = 'FP' THEN 1 ELSE 0 END)
                                              AS fp_f1,
    SUM(CASE WHEN p.resultado_f1_60d = 'TN' THEN 1 ELSE 0 END)
                                              AS tn_f1,
    SUM(CASE WHEN p.resultado_f1_60d = 'FN' THEN 1 ELSE 0 END)
                                              AS fn_f1,
    SUM(CASE WHEN p.resultado_f1_60d = 'NO_VERIFICABLE' THEN 1 ELSE 0 END)
                                              AS no_verificable_f1,

    -- Distribución de resultados – estrategia max_fp_3
    SUM(CASE WHEN p.resultado_max_fp3 = 'TP' THEN 1 ELSE 0 END)
                                              AS tp_max_fp3,
    SUM(CASE WHEN p.resultado_max_fp3 = 'FP' THEN 1 ELSE 0 END)
                                              AS fp_max_fp3,
    SUM(CASE WHEN p.resultado_max_fp3 = 'TN' THEN 1 ELSE 0 END)
                                              AS tn_max_fp3,
    SUM(CASE WHEN p.resultado_max_fp3 = 'FN' THEN 1 ELSE 0 END)
                                              AS fn_max_fp3,
    SUM(CASE WHEN p.resultado_max_fp3 = 'NO_VERIFICABLE' THEN 1 ELSE 0 END)
                                              AS no_verificable_max_fp3,

    -- Métricas agregadas
    COUNT(*)                                  AS total_bloques_predichos,

    -- Precisión y recall calculados por estrategia F1
    CASE
        WHEN (SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END)
            + SUM(CASE WHEN p.resultado_f1_60d = 'FP' THEN 1 ELSE 0 END)) > 0
        THEN ROUND(
            CAST(SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END) AS DOUBLE)
            / (SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END)
             + SUM(CASE WHEN p.resultado_f1_60d = 'FP' THEN 1 ELSE 0 END)),
            4
        )
    END                                       AS precision_f1_mes,

    CASE
        WHEN (SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END)
            + SUM(CASE WHEN p.resultado_f1_60d = 'FN' THEN 1 ELSE 0 END)) > 0
        THEN ROUND(
            CAST(SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END) AS DOUBLE)
            / (SUM(CASE WHEN p.resultado_f1_60d = 'TP' THEN 1 ELSE 0 END)
             + SUM(CASE WHEN p.resultado_f1_60d = 'FN' THEN 1 ELSE 0 END)),
            4
        )
    END                                       AS recall_f1_mes,

    -- Métricas complementarias del monitoreo de deriva
    d.pr_auc_modelo,
    d.pr_auc_baseline,
    d.f1_modelo_recal_f1,
    d.f1_baseline,
    d.estado_operativo_modelo

FROM fact_predicciones_alerta     p
JOIN dim_tiempo_operativo         t ON p.tiempo_key = t.tiempo_key
LEFT JOIN fact_calidad_deriva_mensual d
       ON t.anio_mes = d.anio_mes
      AND p.model_version_id = d.model_version_id
GROUP BY
    t.anio_mes,
    p.model_version_id,
    d.pr_auc_modelo,
    d.pr_auc_baseline,
    d.f1_modelo_recal_f1,
    d.f1_baseline,
    d.estado_operativo_modelo
ORDER BY t.anio_mes;

