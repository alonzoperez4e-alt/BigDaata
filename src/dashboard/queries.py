"""
Capa de consultas desacoplada para el Dashboard Operativo-Analítico (Tarea 4.2 / Fase C).
Usa exclusivamente DuckDB y las vistas semánticas definidas en
`src/serving/semantic_views.sql` sobre almacenamiento local (`lake_v2/serving/star_schema/`)
o remoto en AWS S3 Read-Only con TLS obligatorio.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVING_DIR = REPO_ROOT / "lake_v2" / "serving" / "star_schema"
DEFAULT_SQL_PATH = REPO_ROOT / "src" / "serving" / "semantic_views.sql"
DEFAULT_S3_PREFIX = "serving/star_schema"
DEFAULT_AWS_REGION = "us-east-1"

VALID_STRATEGIES = ("recalibrado_60d_max_fp_3_sem", "recalibrado_60d_f1")

_STRATEGY_COL_MAP = {
    "recalibrado_60d_max_fp_3_sem": {
        "umbral": "umbral_max_fp3",
        "alerta": "alerta_max_fp3",
        "resultado": "resultado_max_fp3",
    },
    "recalibrado_60d_f1": {
        "umbral": "umbral_f1_60d",
        "alerta": "alerta_f1_60d",
        "resultado": "resultado_f1_60d",
    },
}


def _read_setting(key: str, default: str = "") -> str:
    """Lee un parámetro primero desde os.environ y como respaldo desde st.secrets."""
    env_val = os.environ.get(key)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val).strip()
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and key in secrets:
            sec_val = secrets[key]
            if sec_val is not None and str(sec_val).strip() != "":
                return str(sec_val).strip()
    except Exception:
        pass
    return default


def get_storage_config() -> dict[str, str]:
    """Resuelve configuración de almacenamiento (Local vs. AWS S3 Read-Only)."""
    bucket = _read_setting("COES_S3_BUCKET", "").strip()
    prefix = _read_setting("COES_S3_PREFIX", DEFAULT_S3_PREFIX).strip("/") or DEFAULT_S3_PREFIX
    access_key = _read_setting("AWS_ACCESS_KEY_ID", "")
    secret_key = _read_setting("AWS_SECRET_ACCESS_KEY", "")
    region = _read_setting("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION) or DEFAULT_AWS_REGION

    if bucket:
        return {
            "mode": "s3",
            "bucket": bucket,
            "prefix": prefix,
            "s3_uri_prefix": f"s3://{bucket}/{prefix}/",
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "aws_region": region,
        }
    return {
        "mode": "local",
        "bucket": "",
        "prefix": prefix,
        "s3_uri_prefix": "",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        "aws_region": region,
    }


def _is_default_serving_dir(serving_dir: str | Path | None) -> bool:
    if serving_dir is None:
        return True
    s = str(serving_dir).strip()
    if not s:
        return True
    if s.startswith("s3://"):
        return False
    p = Path(s)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p == DEFAULT_SERVING_DIR.resolve()


def get_active_data_source_label(serving_dir: str | Path | None = None) -> str:
    """Retorna la etiqueta legible del origen de datos activo sin exponer credenciales."""
    if serving_dir is not None and str(serving_dir).strip().startswith("s3://"):
        return "AWS S3 (Read-Only · TLS)"
    if _is_default_serving_dir(serving_dir):
        cfg = get_storage_config()
        if cfg["mode"] == "s3":
            return "AWS S3 (Read-Only · TLS)"
    return "Local (lake_v2)"


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _validate_strategy(strategy: str) -> dict[str, str]:
    if strategy not in _STRATEGY_COL_MAP:
        raise ValueError(
            f"Estrategia inválida '{strategy}'. Opciones válidas: {VALID_STRATEGIES}"
        )
    return _STRATEGY_COL_MAP[strategy]


def _get_connection(
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
    sql_path: str | Path = DEFAULT_SQL_PATH,
) -> duckdb.DuckDBPyConnection:
    """Crea una conexión DuckDB en memoria y registra las vistas de semantic_views.sql."""
    sql_file = Path(sql_path)
    if not sql_file.is_absolute():
        sql_file = (REPO_ROOT / sql_file).resolve()
    sql_text = sql_file.read_text(encoding="utf-8")

    cfg = get_storage_config()
    explicit_s3 = serving_dir is not None and str(serving_dir).strip().startswith("s3://")
    use_s3 = explicit_s3 or (_is_default_serving_dir(serving_dir) and cfg["mode"] == "s3")

    con = duckdb.connect(database=":memory:")
    if use_s3:
        target_prefix = (
            f"{str(serving_dir).strip().rstrip('/')}/" if explicit_s3 else cfg["s3_uri_prefix"]
        )
        con.execute("INSTALL httpfs; LOAD httpfs;")
        region_esc = _escape_sql_literal(cfg["aws_region"])
        if cfg["aws_access_key_id"] and cfg["aws_secret_access_key"]:
            key_esc = _escape_sql_literal(cfg["aws_access_key_id"])
            sec_esc = _escape_sql_literal(cfg["aws_secret_access_key"])
            con.execute(
                f"""
                CREATE OR REPLACE SECRET coes_s3 (
                    TYPE S3,
                    KEY_ID '{key_esc}',
                    SECRET '{sec_esc}',
                    REGION '{region_esc}',
                    USE_SSL true
                );
                """
            )
        else:
            con.execute(
                f"""
                CREATE OR REPLACE SECRET coes_s3 (
                    TYPE S3,
                    PROVIDER CREDENTIAL_CHAIN,
                    REGION '{region_esc}',
                    USE_SSL true
                );
                """
            )
    else:
        serving_path = Path(serving_dir if serving_dir is not None else DEFAULT_SERVING_DIR)
        if not serving_path.is_absolute():
            serving_path = (REPO_ROOT / serving_path).resolve()
        target_prefix = f"{serving_path.as_posix().rstrip('/')}/"

    sql_text = sql_text.replace("lake_v2/serving/star_schema/", target_prefix)
    con.execute(sql_text)
    return con


def get_kpis_ejecutivos(
    strategy: str = "recalibrado_60d_max_fp_3_sem",
    alpha: float = 0.10,
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
) -> dict:
    """
    Retorna los 5 KPIs ejecutivos y métricas de respaldo calculados desde
    las vistas semánticas en DuckDB.
    """
    cols = _validate_strategy(strategy)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha debe estar en (0, 1], recibido: {alpha}")

    res_col = cols["resultado"]
    alerta_col = cols["alerta"]

    with _get_connection(serving_dir) as con:
        # 1. KPIs de desempeño de alertas en 2024 (sobre bloques con predicción)
        pred_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS bloques_evaluados,
                SUM(CASE WHEN {res_col} = 'TP' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN {res_col} = 'FP' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN {res_col} = 'TN' THEN 1 ELSE 0 END) AS tn,
                SUM(CASE WHEN {res_col} = 'FN' THEN 1 ELSE 0 END) AS fn,
                SUM(CASE WHEN {res_col} = 'NO_VERIFICABLE' THEN 1 ELSE 0 END) AS no_verificable,
                SUM(CASE WHEN {alerta_col} = 1 THEN 1 ELSE 0 END) AS total_alertas,
                SUM(CASE WHEN {res_col} = 'TP' THEN exposicion_bruta_soles ELSE 0.0 END) AS exp_tp_soles,
                SUM(CASE WHEN {res_col} = 'TP' THEN sobrecosto_neto_estimado_soles ELSE 0.0 END) AS sob_tp_soles,
                SUM(CASE WHEN etiqueta_observable THEN 1 ELSE 0 END) AS bloques_obs_2024
            FROM vw_alerta_temprana_30min
            WHERE model_version_id IS NOT NULL
            """
        ).fetchone()

        (
            bloques_evaluados,
            tp,
            fp,
            tn,
            fn,
            no_verificable,
            total_alertas,
            exp_tp_soles,
            sob_tp_soles,
            bloques_obs_2024,
        ) = pred_row

        tp = int(tp or 0)
        fp = int(fp or 0)
        tn = int(tn or 0)
        fn = int(fn or 0)
        no_verificable = int(no_verificable or 0)
        bloques_obs_2024 = int(bloques_obs_2024 or 0)

        recall_pct = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision_pct = (100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        semanas_obs = bloques_obs_2024 / 336.0 if bloques_obs_2024 > 0 else 1.0
        fp_por_semana = fp / semanas_obs

        # 2. Exposición y sobrecosto normalizado (2024 y total histórico)
        exp_2024_row = con.execute(
            """
            SELECT
                SUM(exposicion_bruta_mes) AS exp_2024,
                SUM(sobrecosto_neto_mes) AS sob_2024,
                SUM(bloques_etiquetables) / 48.0 AS dias_obs_2024
            FROM vw_exposicion_normalizada_mensual
            WHERE anio = 2024
            """
        ).fetchone()
        exp_2024 = float(exp_2024_row[0] or 0.0)
        sob_2024 = float(exp_2024_row[1] or 0.0)
        dias_obs_2024 = float(exp_2024_row[2] or 1.0)

        exp_hist_row = con.execute(
            """
            SELECT
                SUM(exposicion_bruta_mes) AS exp_total,
                SUM(sobrecosto_neto_mes) AS sob_total
            FROM vw_exposicion_normalizada_mensual
            """
        ).fetchone()
        exp_total_hist = float(exp_hist_row[0] or 0.0)
        sob_total_hist = float(exp_hist_row[1] or 0.0)

        # 3. Ahorro potencial bajo tasa de mitigación alpha
        ahorro_bruto_soles = alpha * float(exp_tp_soles or 0.0)
        ahorro_neto_soles = alpha * float(sob_tp_soles or 0.0)
        pct_ahorro_bruto = (100.0 * ahorro_bruto_soles / exp_2024) if exp_2024 > 0 else 0.0
        pct_ahorro_neto = (100.0 * ahorro_neto_soles / sob_2024) if sob_2024 > 0 else 0.0

        # 4. Cobertura y deriva mensual
        cov_row = con.execute(
            """
            SELECT
                AVG(pct_bloques_etiquetables) AS pct_cob_prom,
                SUM(CASE WHEN alerta_baja_cobertura THEN 1 ELSE 0 END) AS meses_baja_cob,
                MAX(psi_max_top5) AS psi_max_2024,
                (SELECT estado_operativo_modelo
                 FROM vw_cobertura_y_deriva_mensual
                 ORDER BY anio_mes DESC LIMIT 1) AS ultimo_estado
            FROM vw_cobertura_y_deriva_mensual
            """
        ).fetchone()

    return {
        "strategy": strategy,
        "alpha": float(alpha),
        # KPI 1: Tasa de detección anticipada (Recall) y precisión
        "recall_bloques_pct": round(recall_pct, 2),
        "precision_bloques_pct": round(precision_pct, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "no_verificable": no_verificable,
        "total_alertas": int(total_alertas or 0),
        # KPI 2: Fatiga operativa (FP/semana)
        "fp_por_semana": round(fp_por_semana, 2),
        # KPI 3: Exposición bruta y sobrecosto neto por día observable (2024)
        "exposicion_bruta_2024_soles": round(exp_2024, 2),
        "sobrecosto_neto_2024_soles": round(sob_2024, 2),
        "dias_observables_2024": round(dias_obs_2024, 2),
        "exposicion_bruta_dia_obs_2024": round(exp_2024 / dias_obs_2024, 2),
        "sobrecosto_neto_dia_obs_2024": round(sob_2024 / dias_obs_2024, 2),
        "exposicion_bruta_total_hist": round(exp_total_hist, 2),
        "sobrecosto_neto_total_hist": round(sob_total_hist, 2),
        # KPI 4: Ahorro potencial por mitigación
        "ahorro_bruto_soles": round(ahorro_bruto_soles, 2),
        "ahorro_neto_soles": round(ahorro_neto_soles, 2),
        "pct_ahorro_bruto_2024": round(pct_ahorro_bruto, 2),
        "pct_ahorro_neto_2024": round(pct_ahorro_neto, 2),
        # KPI 5: Cobertura y deriva
        "cobertura_promedio_pct": round(float(cov_row[0] or 0.0), 2),
        "meses_baja_cobertura": int(cov_row[1] or 0),
        "psi_max_2024": round(float(cov_row[2] or 0.0), 4),
        "ultimo_estado_operativo": str(cov_row[3] or "DESCONOCIDO"),
    }


def get_serie_alerta_30min(
    fecha_inicio: str | date,
    fecha_fin: str | date,
    strategy: str = "recalibrado_60d_max_fp_3_sem",
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
) -> pd.DataFrame:
    """
    Retorna la serie temporal de 30 minutos desde `vw_alerta_temprana_30min`,
    normalizando `fecha_hora` (TIMESTAMP_NS -> TIMESTAMP estándar).
    """
    cols = _validate_strategy(strategy)
    f_ini = str(fecha_inicio)
    f_fin = str(fecha_fin)

    query = f"""
        SELECT
            tiempo_key,
            CAST(fecha_hora AS TIMESTAMP) AS fecha_hora,
            fecha_despacho,
            bloque_30min,
            hora,
            anio,
            anio_mes,
            periodo_hidrologico,
            model_version_id,
            horizonte,
            prob_cruda,
            prob_recalibrada,
            {cols["umbral"]} AS umbral_activo,
            {cols["alerta"]} AS alerta_activa,
            {cols["resultado"]} AS resultado_activo,
            shap_top1,
            shap_top2,
            shap_top3,
            gen_total_mw,
            prog_total_mw,
            term_cara_ejec_mw,
            term_cara_prog_mw,
            term_cara_share,
            cmg_troncal_mediana,
            cmg_troncal_p90,
            cmg_spread_p90_p50,
            n_forzados_termicos_activos,
            n_forzados_transmision_activos,
            n_forzados_termicos_inicio_3h,
            n_mantenimientos_prog_termicos_activos,
            etiqueta_observable,
            motivo_censura,
            evento_termico_nop,
            inicio_termico_nop,
            exposicion_bruta_soles,
            sobrecosto_neto_estimado_soles,
            (prog_total_mw IS NOT NULL AND cmg_troncal_mediana IS NOT NULL) AS senial_disponible_t
        FROM vw_alerta_temprana_30min
        WHERE fecha_despacho BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        ORDER BY CAST(fecha_hora AS TIMESTAMP)
    """
    with _get_connection(serving_dir) as con:
        df = con.execute(query, [f_ini, f_fin]).df()
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    return df


def get_exposicion_normalizada(
    granularidad: Literal["mensual", "anual"] = "mensual",
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
) -> pd.DataFrame:
    """
    Retorna exposición bruta y sobrecosto neto normalizados por día observable,
    en granularidad 'mensual' o 'anual'.
    """
    if granularidad not in ("mensual", "anual"):
        raise ValueError("granularidad debe ser 'mensual' o 'anual'")

    with _get_connection(serving_dir) as con:
        if granularidad == "mensual":
            return con.execute(
                """
                SELECT *
                FROM vw_exposicion_normalizada_mensual
                ORDER BY anio_mes
                """
            ).df()

        return con.execute(
            """
            SELECT
                anio,
                SUM(exposicion_bruta_mes) AS exposicion_bruta_anual,
                SUM(sobrecosto_neto_mes) AS sobrecosto_neto_anual,
                SUM(bloques_etiquetables) AS bloques_etiquetables,
                SUM(bloques_totales) AS bloques_totales,
                ROUND(
                    CAST(SUM(bloques_etiquetables) AS DOUBLE) / SUM(bloques_totales) * 100.0,
                    2
                ) AS pct_cobertura_etiqueta,
                ROUND(CAST(SUM(bloques_etiquetables) AS DOUBLE) / 48.0, 2) AS dias_observables,
                ROUND(
                    SUM(exposicion_bruta_mes) / (CAST(SUM(bloques_etiquetables) AS DOUBLE) / 48.0),
                    2
                ) AS exposicion_bruta_por_dia_obs,
                ROUND(
                    SUM(sobrecosto_neto_mes) / (CAST(SUM(bloques_etiquetables) AS DOUBLE) / 48.0),
                    2
                ) AS sobrecosto_neto_por_dia_obs,
                SUM(eventos_termico_nop_mes) AS eventos_termico_nop_anual,
                SUM(inicios_termico_nop_mes) AS inicios_termico_nop_anual
            FROM vw_exposicion_normalizada_mensual
            GROUP BY anio
            ORDER BY anio
            """
        ).df()


def get_cobertura_y_deriva(
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
) -> pd.DataFrame:
    """
    Retorna cobertura de observabilidad, prevalencia, PSI máximo top-5,
    estado operativo y motivo principal de censura por mes (2024-01..2024-11).
    """
    query = """
        WITH censura_dom AS (
            SELECT
                anio_mes,
                FIRST(motivo_censura ORDER BY cnt DESC) AS motivo_censura_principal
            FROM (
                SELECT
                    anio_mes,
                    motivo_censura,
                    COUNT(*) AS cnt
                FROM vw_alerta_temprana_30min
                WHERE NOT etiqueta_observable
                  AND anio = 2024
                GROUP BY anio_mes, motivo_censura
            ) sub
            GROUP BY anio_mes
        )
        SELECT
            c.*,
            COALESCE(cd.motivo_censura_principal, 'SIN_CENSURA') AS motivo_censura_principal
        FROM vw_cobertura_y_deriva_mensual c
        LEFT JOIN censura_dom cd ON c.anio_mes = cd.anio_mes
        ORDER BY c.anio_mes
    """
    with _get_connection(serving_dir) as con:
        return con.execute(query).df()


def get_auditoria_modelo(
    strategy: str = "recalibrado_60d_max_fp_3_sem",
    serving_dir: str | Path | None = DEFAULT_SERVING_DIR,
) -> pd.DataFrame:
    """
    Retorna la auditoría mensual del modelo `rf_v1_2024_walkforward` frente
    a `baseline_tasa_horaria`, incluyendo falsas alarmas por semana y estado operativo.
    """
    _validate_strategy(strategy)
    suffix = "max_fp3" if strategy == "recalibrado_60d_max_fp_3_sem" else "f1"

    query = f"""
        SELECT
            a.anio_mes,
            a.model_version_id,
            a.tp_{suffix} AS tp,
            a.fp_{suffix} AS fp,
            a.tn_{suffix} AS tn,
            a.fn_{suffix} AS fn,
            a.no_verificable_{suffix} AS no_verificable,
            a.total_bloques_predichos,
            CASE
                WHEN (a.tp_{suffix} + a.fp_{suffix}) > 0
                THEN ROUND(CAST(a.tp_{suffix} AS DOUBLE) / (a.tp_{suffix} + a.fp_{suffix}), 4)
            END AS precision_mes,
            CASE
                WHEN (a.tp_{suffix} + a.fn_{suffix}) > 0
                THEN ROUND(CAST(a.tp_{suffix} AS DOUBLE) / (a.tp_{suffix} + a.fn_{suffix}), 4)
            END AS recall_mes,
            CASE
                WHEN c.bloques_etiquetables > 0
                THEN ROUND(CAST(a.fp_{suffix} AS DOUBLE) / (CAST(c.bloques_etiquetables AS DOUBLE) / 336.0), 2)
            END AS fp_por_semana_estrategia,
            c.fp_por_semana_recal_f1,
            a.pr_auc_modelo,
            a.pr_auc_baseline,
            a.f1_modelo_recal_f1,
            a.f1_baseline,
            c.alerta_baja_cobertura,
            c.pct_bloques_etiquetables,
            a.estado_operativo_modelo
        FROM vw_auditoria_modelo_vs_baseline a
        JOIN vw_cobertura_y_deriva_mensual c
          ON a.anio_mes = c.anio_mes
         AND a.model_version_id = c.model_version_id
        ORDER BY a.anio_mes
    """
    with _get_connection(serving_dir) as con:
        return con.execute(query).df()
