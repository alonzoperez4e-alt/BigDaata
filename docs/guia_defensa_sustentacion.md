# Guía de Defensa y Sustentación Individual (60% de la Evaluación)

Esta guía sintetiza respuestas técnicas directas, verificables en el repositorio y respaldadas por métricas reales del proyecto COES-SEIN para enfrentar preguntas cruzadas durante la sustentación individual.

---

## 1. Preguntas de Ingeniería de Datos y Arquitectura

### P1. ¿Por qué migraron de particionamiento mensual a particionamiento anual (`anio`) en la capa Gold y por qué usar Star Schema en Serving?
- **Respuesta:** El particionamiento mensual generaba fragmentación de archivos pequeños (*small files problem*), penalizando la planificación de Spark y lecturas I/O. La consolidación por `anio` redujo la capa Gold a **35 archivos Parquet** manteniendo lecturas rápidas (`43,83 s` en `run_1x_gold.json`). En Serving (`lake_v2/serving/star_schema/`), el modelo estrella separa **4 dimensiones** (`dim_tiempo_operativo` [52 608], `dim_central_combustible` [171], `dim_barra_troncal` [84], `dim_modelo_version` [1]) y **3 hechos** (`fact_operacion_bloque_30min` [52 608], `fact_predicciones_alerta` [16 080], `fact_calidad_deriva_mensual` [11]), desacoplando al dashboard (`DuckDB`) de la capa ETL Gold y garantizando reconciliación contable `< S/ 0,01`.

### P2. ¿Cómo garantizan que el cruce entre A2 (despacho) y A3 (costos variables) no multiplique filas ni invente costos?
- **Respuesta:** En `src/etl/gold_feature_store.py` y `evidencias/run_1x_gold.json`, `A3` (~10,46 MB) se une mediante `BroadcastHashJoin` deduplicado previamente por `(central, fecha)`. El contrato automatizado `2_no_multiplicacion_despacho` verifica que las filas de entrada en `A2` (`6 533 616`) coincidan exactamente con `gold/despacho_central_30min` (`6 533 616`). De las `171` centrales, `70` cuentan con costo declarado en A3 (`con_costo_A3`) y `101` quedan explícitamente marcadas como `sin_costo_A3` en `dim_central_combustible` sin imputación sintética.

### P3. ¿Cómo manejan la censura del 31,2% de bloques y la compatibilidad de timestamps entre PyArrow, Spark y DuckDB?
- **Respuesta:** Cuando falta alguna fuente requerida (`A1`, `A2`, `A3` o `PR-06`, totalizando `16 272` bloques de `52 608`), `etiqueta_observable = False` y los targets `evento_termico_nop` e `inicio_termico_nop` se preservan en **NULL estricto** (0 violaciones verificadas en `TestNullPreservation`). En Serving, dado que PyArrow exporta marcas temporales en `TIMESTAMP(NANOS,false)` (que Spark 3.5.1 rechaza leer directamente), la capa de consultas `src/dashboard/queries.py` normaliza explícitamente con `CAST(fecha_hora AS TIMESTAMP)` en DuckDB para operar de forma portátil.

### P4. ¿Por qué la arquitectura es 100% agnóstica de proveedor?
- **Respuesta:** Todo el almacenamiento usa formatos abiertos (**Apache Parquet + Snappy**, CSV y JSON), el procesamiento distribuido usa **PySpark 3.5.1** con SQL estándar (`semantic_views.sql`) y el serving analítico usa **DuckDB + Streamlit + Plotly**. No existe ninguna dependencia de SDKs de AWS, GCP o Azure (solo soporte opcional de parámetro `LAKE_ROOT` compatible con rutas locales o S3 estándar).

---

## 2. Preguntas de Modelado, Estadística y Explicabilidad

### P5. ¿Cómo evitaron fuga temporal (*data leakage*) al construir las 24 variables y al validar el modelo?
- **Respuesta:** Primero, `src/ml/feature_engineering.py` bloquea explícitamente `FORBIDDEN_LEAKAGE_COLS` = `{label_rdo, alfa, beta, gamma, chm, alfa_primer_archivo, beta_primer_archivo, gamma_primer_archivo}` provenientes del `B4_pr06_rdo.csv` (resultados retroactivos post-operación), verificado en `test_rechaza_columnas_prohibidas`. Segundo, `walk_forward_cv.py` aplica un **embargo estricto de 4 horas** entre train y validación para impedir solapamiento de la ventana objetivo $t+4\dots t+8$. Tercero, la selección de hiperparámetros (`RandomForestClassifier` con `maxDepth=5`, `numTrees=50`) se realizó únicamente con los folds pre-2024 (`fold_1_2023_s1` y `fold_2_2023_s2`), dejando 2024 como test ciego.

### P6. ¿Por qué eligieron `RandomForestClassifier` si `GBTClassifier` o `LogisticRegression` tuvieron buen PR-AUC en test 2024?
- **Respuesta:** La regla antifuga del proyecto obligó a seleccionar el ganador **exclusivamente antes de ver 2024**, usando el promedio de validación pre-2024 en 2023 S1 y 2023 S2: `RandomForestClassifier(maxDepth=5, numTrees=50)` obtuvo el mejor promedio pre-2024 (**PR-AUC = 0,6455**, brecha overfit `0,1179`) frente a `LogisticRegression` (`0,6255`) y `GBTClassifier` (`0,5079`). Cambiar de modelo después de inspeccionar las métricas de test 2024 violaría la integridad experimental del test ciego.

### P7. ¿Por qué falló la predicción del inicio exacto (`y_inicio_ventana_2_4h`) frente al baseline y cómo lo interpretan?
- **Respuesta:** En `y_inicio_ventana_2_4h` (tasa base 3,74% en 2024), el modelo completo logró **PR-AUC = 0,0480** y la ablación sin persistencia **0,0542**, ambos por debajo del `baseline_tasa_horaria` (**0,0562**). Técnicamente, cuando aún no existe desviación térmica activa ni estrés nodal acumulado, los disparos súbitos responden a eventos exógenos (errores de pronóstico de demanda, variación de caudales en tiempo real o disparos imprevistos de transmisión aún no registrados en B1/B3) que no están presentes en las fuentes actuales. Por honestidad científica, ese target se descartó para alertas automáticas y motivó la principal recomendación de trabajo futuro.

### P8. ¿Qué explica el modelo según SHAP y cómo validaron la fidelidad de esa explicación?
- **Respuesta:** Al no haber TreeExplainer nativo directo en Spark MLlib, se entrenó un surrogate equivalente en scikit-learn (`max_depth=5`, `n_estimators=50`) que alcanzó una correlación de **Spearman = 0,9054** con las probabilidades de Spark. Las 5 variables SHAP globales dominantes en 2024 son: (1) `ratio_term_ejec_vs_prog` (`0,1023`), (2) `term_cara_share` (`0,0873`), (3) `cmg_troncal_p90_t` (`0,0572`), (4) `cmg_troncal_mediana_t` (`0,0399`) y (5) `persistencia_term_nop_2h` (`0,0391`), confirmando que el modelo captura desviación térmica presente y congestión marginal en barras 220/500 kV.

### P9. ¿Por qué explotó a `108,50 FP/sem` en mayo de 2024 con `recalibrado_60d_f1` y por qué `recalibrado_60d_max_fp_3_sem` es la estrategia operativa?
- **Respuesta:** En `2024-04`, la cobertura de observabilidad cayó a `37,08%` y hubo **0 bloques positivos etiquetables**. Al recalibrar el umbral F1 con una ventana móvil de 60 días sin ejemplos positivos recientes suficientes (`pi_reciente = 0,00899`), el umbral cayó a `tau = 0,05` en mayo, disparando 489 alertas (`108,50 FP/sem`). La estrategia operativa `recalibrado_60d_max_fp_3_sem` impone una restricción de fatiga (`<= 3 FP/sem` históricos en la ventana), manteniendo **1,97 FP/sem** promedio anual con **85,66% de precisión**, logrando aun así un ahorro conservador ($\alpha = 5\%$) de **2,54% bruto** (S/ 484 382) y **2,23% neto** (S/ 339 144), dentro de la meta del 2%–4%.

---

## 3. Preguntas de Gobierno del Dato, Ética y Normativa

### P10. ¿Aplica la Ley N.º 29733 a este sistema y qué salvaguardas existen sobre texto libre?
- **Respuesta:** La **Ley N.º 29733** (Protección de Datos Personales del Perú) protege información que identifica o hace identificable a una **persona natural**. Las 8 fuentes del COES solo contienen registros de infraestructura operativa (`171` centrales, `84` barras troncales y empresas generadoras jurídicas de acceso público). Por tanto, no requiere inscripción de banco de datos personales ni EIPD. Aun así, por buen gobierno, los campos de texto libre (`motivo_reprograma_bloque`, derivados de PR-06) no ingresan al modelo (`FORBIDDEN_LEAKAGE_COLS`), los archivos Bronze se sellan con **SHA-256** (`bronze_real_sha256.json`) y Git excluye datos y secretos.

### P11. ¿Cómo explican un `psi_max_top5 = 13,28` en noviembre de 2024 y qué acción toma el sistema?
- **Respuesta:** En 2024 ocurrió un cambio estructural de régimen respecto a 2021–2023 (la generación térmica cara programada se multiplicó por `×2,3`), dejando deciles de referencia vacíos en las variables más influyentes; al suavizar con `eps = 1e-6`, la divergencia KL elevó el `psi_max_top5` hasta **13,28** en `2024-11` (semáforo `ROJO`). El sistema expone esta deriva transparentemente en Vista 3 y restringe la emisión de alertas automáticas (`MODO_MONITOREO_DESCRIPTIVO`) cuando la cobertura mensual cae debajo del 50% (`2024-01` con `32,26%` y `2024-04` con `37,08%`).

### P12. ¿Cuál es la asimetría entre un Falso Positivo (FP) y un Falso Negativo (FN) en el SEIN?
- **Respuesta:** Un **FN** omite la ventana preventiva de 2–4 horas ante un despacho térmico caro no programado, generando sobrecosto económico directo (promedio de **S/ 54 171 / día observable** en 2024). Un **FP** genera costo de verificación y fatiga de alarma en sala de control. Por ello se implementa un esquema **human-in-the-loop**: la alerta es estrictamente consultiva (nunca automática) y se opera por defecto bajo `recalibrado_60d_max_fp_3_sem` (`1,97 FP/sem`).

---

## 4. Preguntas del Dashboard Operativo-Analítico

### P13. ¿Cómo evitan sesgos al comparar costos mensuales de meses con distinta cobertura (ej. enero vs. julio 2024)?
- **Respuesta:** En Vista 2 y `vw_exposicion_normalizada_mensual`, nunca comparamos únicamente totales absolutos mensuales sin contexto. Normalizamos tanto la exposición bruta (`S/ 19,08 M` en 2024) como el sobrecosto neto (`S/ 15,22 M` en 2024) por **día observable** (`bloques_etiquetables / 48.0`, equivalente a `281` días observables en 2024), permitiendo comparaciones justas entre meses con `32,26%` de cobertura (`2024-01`) y `87,10%` (`2024-07`).

### P14. ¿Cómo realiza un auditor la trazabilidad inversa desde un punto del dashboard hasta el archivo Bronze original?
- **Respuesta:** Siguiendo `docs/linaje_datos.csv` y el protocolo de la Sección 5.2.4: (1) tomar `tiempo_key` (`YYYYMMDD_HHMM`) en Vista 1; (2) consultar `fact_predicciones_alerta` y `dim_tiempo_operativo` en DuckDB; (3) cruzar `fecha_hora` en `ml_outputs/predicciones_alerta_30min` o `gold_feature_store_30min`; (4) descender a `silver/` y localizar el archivo fuente en `lake_v2/bronze/`; y (5) verificar su hash contra `data/manifests/bronze_real_sha256.json`.
