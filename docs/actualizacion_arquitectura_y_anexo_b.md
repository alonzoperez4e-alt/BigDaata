# Actualización de Arquitectura (Sección 2.1 / Figura 2.1) y Auditoría de Cierre Anexo A y Anexo B

## 1. Actualización de la Sección 2.1 y Figura 2.1 (Arquitectura Definitiva)

> **Estado:** Arquitectura implementada y verificada de extremo a extremo sobre la rama `feat/fase4-serving-dashboard` (cero componentes "previstos" o "pendientes", 100% agnóstica de proveedor y con cero datos sintéticos).

El diagrama definitivo se encuentra en [`docs/figura_2_1_arquitectura_final.mmd`](figura_2_1_arquitectura_final.mmd) y sustituye cualquier referencia preliminar de las Fases 1–3 sobre particionamiento mensual o capas de consumo no construidas:

1. **Ingesta e Integridad (Bronze):** 8 archivos reales del COES-SINAC (fuentes Parte A: `A1`–`A4` en Parquet; fuentes Parte B: `B1`–`B4` en CSV) verificados criptográficamente mediante manifiesto SHA-256 (`data/manifests/bronze_real_sha256.json`).
2. **Procesamiento y Curación (Silver y Gold en PySpark 3.5.1 + Parquet/Snappy):**
   - Particionamiento anual definitivo (`anio`, formato Hive) que consolidó la capa Gold en 35 archivos Parquet (eliminando la sobrecarga de archivos pequeños de esquemas mensuales anteriores).
   - Tablas de detalle: `despacho_central_30min` (6 533 616 filas, 171 centrales) y `precios_barra_30min` (9 587 472 filas, 84 barras troncales de 220/500 kV).
   - Tablas de grilla maestra de 30 minutos (**52 608** bloques entre diciembre 2021 y noviembre 2024): `gold_observabilidad_etiqueta_30min`, `gold_precios_troncales_30min`, `gold_eventos_criticos_30min`, `gold_contexto_auditoria_30min` y `gold_feature_store_30min` (29 columnas sin fuga, `NULL` estricto en los 16 272 bloques censurados).
3. **Capa Analítica y Machine Learning (`src/ml/`):**
   - `ml_features_30min` (52 608 bloques, 24 predictores válidos tras bloqueo explícito de `FORBIDDEN_LEAKAGE_COLS` = `{label_rdo, alfa, beta, gamma, chm, ...}`).
   - Validación walk-forward con embargo estricto de 4 horas (`walk_forward_cv.py`), selección pre-2024 de `RandomForestClassifier` (`maxDepth=5`, `numTrees=50`, `model_version_id = rf_v1_2024_walkforward`), explicabilidad TreeSHAP local/global (Spearman `0,9054`), recalibración móvil de 60 días (`recalibrado_60d_max_fp_3_sem` y `recalibrado_60d_f1`) y backtest económico 2024.
   - Tres contratos oficiales en `lake_v2/gold/ml_outputs/`: `predicciones_alerta_30min` (**16 080** filas), `monitoreo_deriva_mensual` (**11** filas, `2024-01` a `2024-11`) y `registro_modelos.json`.
4. **Capa Semántica Agnóstica (`lake_v2/serving/star_schema/`):**
   - Generada por `src/serving/build_star_schema.py` en formato Parquet + Snappy con **4 dimensiones** (`dim_tiempo_operativo` [52 608], `dim_central_combustible` [171], `dim_barra_troncal` [84], `dim_modelo_version` [1]) y **3 tablas de hechos** (`fact_operacion_bloque_30min` [52 608], `fact_predicciones_alerta` [16 080], `fact_calidad_deriva_mensual` [11]).
   - Vistas SQL estándar portables definidas en `src/serving/semantic_views.sql` (`vw_alerta_temprana_30min`, `vw_exposicion_normalizada_mensual`, `vw_cobertura_y_deriva_mensual`, `vw_auditoria_modelo_vs_baseline`), con reconciliación contable `< S/ 0,01` frente a Gold en los cuatro años.
5. **Capa de Consumo y Analítica Visual (`src/dashboard/`):**
   - Motor `DuckDB` in-memory leyendo exclusivamente `lake_v2/serving/star_schema/` (`queries.py`, normalizando `TIMESTAMP_NS` a `TIMESTAMP` estándar), sin lectura directa de Bronze, Silver o Gold.
   - Aplicación `Streamlit + Plotly` (`app.py`) con selector de estrategia (`recalibrado_60d_max_fp_3_sem` por defecto vs. `recalibrado_60d_f1`), factor $\alpha \in \{5\%, 10\%, 15\%\}$, 5 KPIs ejecutivos y 4 vistas analíticas.

### Etiquetas preliminares retiradas del informe

| Concepto preliminar (Fases 1–3) | Estado definitivo en Fase 4 |
|---|---|
| Particionamiento mensual (`anio_mes`) en Gold | Sustituido por particionamiento anual (`anio`) en 35 Parquets consolidados |
| Capa de consumo / BI "pendiente para Fase 4" | Implementada en `lake_v2/serving/star_schema/` + `src/dashboard/app.py` |
| Posibles conectores propietarios de nube | 100% agnóstico: Parquet + Snappy + DuckDB + PySpark + Streamlit/Plotly |
| Reglas de gobierno y ética "por definir" | Formalizadas en `docs/linaje_datos.csv`, `docs/seccion_5_etica_linaje_sesgos.md` (Ley N.º 29733) y `MODO_MONITOREO_DESCRIPTIVO` |

---

## 2. Lista de Chequeo de Consistencia Numérica (Informe ↔ JSONs ↔ Dashboard)

| Indicador / Entidad | Valor Oficial Auditado | Fuente JSON / Parquet | Verificación en Dashboard (`queries.py` / `app.py`) | Estado |
|---|---|---|---|---|
| Total de bloques (2021-12 a 2024-11) | `52 608` | `star_schema_summary.json`, `run_1x_gold.json` | `dim_tiempo_operativo`, `fact_operacion_bloque_30min` | Verificado (`0` huérfanas) |
| Bloques censurados (`etiqueta_observable = False`) | `16 272` (31,0% estricto / 31,2% en ventana ML) | `run_1x_gold.json`, `star_schema_summary.json` | Zonas grises Vista 1 + `0` violaciones `NULL` | Verificado (`TestNullPreservation`) |
| Centrales en catálogo / con costo A3 | `171` totales (`70` con A3, `101` sin A3) | `dim_central_combustible.parquet` | Documentado en Secciones 5.3.2 y 6.1 | Verificado (`test_dim_central_a3_counts`) |
| Barras troncales (220 y 500 kV) | `84` | `dim_barra_troncal.parquet`, `run_1x_gold.json` | Filtro `220/500 kV` en `build_star_schema.py` | Verificado (`test_dim_barra_tension_values`) |
| Bloques evaluados 2024 (test ciego) | `16 080` (`10 712` obs. en `y_ventana_2_4h`, `281` días obs.) | `comparativa_modelos.json`, `backtest_economico_2024.json` | `fact_predicciones_alerta` (`16 080` filas) | Verificado (`test_fact_predicciones_alerta`) |
| Registros mensuales de deriva 2024 | `11` (`2024-01` a `2024-11`) | `fact_calidad_deriva_mensual.parquet` | Vista 3 y Vista 4 | Verificado (`test_auditoria_modelo_retorna_11_meses`) |
| Modelo activo en producción | `rf_v1_2024_walkforward` (`RandomForestClassifier`, `maxDepth=5`, `numTrees=50`) | `registro_modelos.json`, `dim_modelo_version.parquet` | `vw_cobertura_y_deriva_mensual`, Vista 4 | Verificado (`test_dim_modelo_pk_unique`) |
| Exposición bruta 2021–2024 (S/) | `2021`: 404 599,93 · `2022`: 75 030 153,71 · `2023`: 175 600 703,10 · `2024`: 19 079 554,64 (`19,08 M`) | `star_schema_summary.json`, `backtest_economico_2024.json` | Vista 2 (`get_exposicion_normalizada`) | Verificado (`diff < S/ 0,01`) |
| Sobrecosto neto 2021–2024 (S/) | `2021`: 239 930,13 · `2022`: 15 114 059,99 · `2023`: 48 189 150,53 · `2024`: 15 223 546,20 (`15,22 M`) | `star_schema_summary.json`, `backtest_economico_2024.json` | KPI 3 y Vista 2 | Verificado (`diff < S/ 0,01`) |
| Ahorro conservador ($\alpha = 5\%$) 2024 | `max_fp<=3/sem`: `2,54%` bruto (`S/ 484 382`) / `2,23%` neto (`S/ 339 144`) · `F1-60d`: `4,18%` bruto (`S/ 798 316`) / `4,05%` neto (`S/ 616 073`) | `backtest_economico_2024.json` | KPI 4 y tabla de escenarios Vista 2 | Verificado |
| Ahorro base ($\alpha = 10\%$) 2024 | `max_fp<=3/sem`: `5,08%` bruto (`S/ 968 765`) / `4,46%` neto (`S/ 678 288`) · `F1-60d`: `8,37%` bruto (`S/ 1 596 633`) / `8,09%` neto (`S/ 1 232 145`) | `backtest_economico_2024.json` | KPI 4 y tabla de escenarios Vista 2 | Verificado |
| Fatiga operativa 2024 (`FP/sem`) | `max_fp<=3/sem`: `1,97 FP/sem` · `F1-60d`: `18,76 FP/sem` (pico `108,50` en `2024-05`) · `baseline_tasa_horaria`: `174,28 FP/sem` | `explicabilidad_y_costos.json`, `fact_calidad_deriva_mensual.parquet` | KPI 2 y Vista 4 | Verificado |
| Cobertura crítica 2024 y PSI máximo | `2024-01`: `32,26%` · `2024-04`: `37,08%` (`0` positivos) · `psi_max_top5` en `2024-11`: `13,28` (`eps = 1e-6`) | `walk_forward_baselines.json`, `fact_calidad_deriva_mensual.parquet` | KPI 5 y Vista 3 | Verificado |
| Techo `y_inicio_ventana_2_4h` | `0,0480` completo / `0,0542` ablación vs. `0,0562` `baseline_tasa_horaria` | `comparativa_modelos.json` | Excluido de alerta automática; explicado en Sección 6.2 | Verificado |

---

## 3. Checklist de Cierre para Anexo A y Anexo B

### Anexo A — Declaración de Uso de IA (Actualización a Fase 4)

| Ítem de verificación | Evidencia en repositorio | Estado |
|---|---|---|
| Registro de módulos asistidos por IA (Fases 1–4) | `docs/anexo_a_ia.md` (ampliar con Fases 2–4: ETL Gold, ML/SHAP, Star Schema, Dashboard DuckDB+Streamlit, Secciones 5 y 6) | Listo para firma |
| Validación humana mediante pruebas automatizadas | Suite `pytest` completa en verde (incluyendo tests de contratos Gold, antifuga ML, reconciliación Star Schema, queries DuckDB, linaje Sección 5, trazabilidad Sección 6 y cierre Anexo B) | Verificado |
| Firmas y códigos universitarios de los 4 integrantes | Campos en `docs/anexo_a_ia.md` (Fabian Cubas, Cristhian Alonzo, Joan Solar, Carlos Yon) | Acción manual final antes de entrega PDF |

### Anexo B — Auditoría Técnica y Entrega Final

| Requisito Anexo B | Evidencia técnica verificada | Estado |
|---|---|---|
| 1. Repositorio Git limpio sin datos ni secretos | `.gitignore` excluye `lake_v2/`, `lake_v2_gold.zip`, `.venv*`, `.env`, `*.pem`, `*.key`; historial trazable en `feat/fase4-serving-dashboard` | Verificado |
| 2. Integridad criptográfica de entrada | Manifiestos SHA-256 en `data/manifests/bronze_real_sha256.json` y `bronze_sha256.json` (8 fuentes COES) | Verificado |
| 3. Arquitectura 100% agnóstica y cero datos sintéticos | Apache Parquet + Snappy + PySpark 3.5.1 + DuckDB + Streamlit + Plotly; cero SDK propietarios de nube y cero generadores sintéticos | Verificado |
| 4. Escalabilidad y particionamiento definitivo | Pruebas 100x Silver/Gold en `evidencias/escalabilidad/` + compactación anual (`anio`) en `evidencias/run_1x_gold.json` (35 Parquets) | Verificado |
| 5. Rigor ML sin fuga temporal | `FORBIDDEN_LEAKAGE_COLS` bloqueadas, embargo de 4 h en `walk_forward_cv.py`, selección pre-2024 y calibración estrictamente causal (`60d`) | Verificado |
| 6. Explicabilidad y valorización económica | TreeSHAP (`explicabilidad_y_costos.json`, Spearman `0,9054`), análisis asimétrico FP vs. FN y backtest 2024 (`backtest_economico_2024.json`) | Verificado |
| 7. Capa Semántica y Dashboard desacoplado | `src/serving/build_star_schema.py` + `semantic_views.sql` + `src/dashboard/{queries,app}.py` (`queries.py` no referencia `bronze/`, `silver/` ni `gold/`) | Verificado |
| 8. Gobierno, Ética, Linaje y Normativa | `docs/linaje_datos.csv` (17 rutas extremo a extremo) + `docs/seccion_5_etica_linaje_sesgos.md` (Ley N.º 29733 y PR-06) | Verificado |
| 9. Trazabilidad de Valor y Cierre | `docs/seccion_6_trazabilidad_cierre.md` + `docs/figura_2_1_arquitectura_final.mmd` + `docs/guia_defensa_sustentacion.md` | Verificado |
| 10. Prueba de arranque en limpio y PR hacia `main` | Ejecutar `streamlit run src/dashboard/app.py -- --serving-dir lake_v2/serving/star_schema` y `pytest -v`; fusionar rama a `main` tras verificación de URL pública | Paso final de entrega |
