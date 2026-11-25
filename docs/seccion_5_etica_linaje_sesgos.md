# Sección 5 — Ética, Privacidad, Linaje y Análisis de Sesgos

## 5.1 Datos sensibles y protección aplicada

### 5.1.1 Ausencia de datos personales en las fuentes operativas del COES

Las ocho fuentes Bronze ingestadas desde los portales públicos del COES-SINAC corresponden exclusivamente a registros operativos del Sistema Eléctrico Interconectado Nacional (SEIN):

| Fuente Bronze | Archivo | Contenido |
|---|---|---|
| A1 – Despacho Programado | `a1_despacho_programado.parquet` | Potencia programada por central y bloque de 30 min |
| A2 – Despacho Ejecutado | `a2_con_desviacion.parquet` | Potencia ejecutada con desviación calculada |
| A3 – Costos Variables | `a3_costo_central_fecha.parquet` | Costo variable declarado por central y fecha |
| A4 – Costos Marginales | `a4_costos_marginales_30min.parquet` | CMg por barra y bloque de 30 min |
| B1 – Mantenimientos | `B1_mantenimientos.csv` | Eventos de mantenimiento y forzados |
| B2 – Potencia Indisponible | `B2_potencia_indisponible.csv` | Registros de indisponibilidad |
| B3 – Reprogramas | `B3_reprograma_diario.csv` | Reprogramaciones diarias del despacho |
| B4 – PR-06 (RDO) | `B4_pr06_rdo.csv` | Resultados de operación del Procedimiento Técnico PR-06 |

Ninguna fuente contiene datos personales de usuarios finales del servicio eléctrico: no hay nombres, DNI, direcciones, consumo residencial, mediciones de acometida ni información de facturación individual. Las únicas entidades nominales son **centrales de generación** (171 unidades) y **empresas generadoras** (personas jurídicas de registro público), que constituyen información operativa publicada por el COES en cumplimiento de sus funciones regulatorias.

### 5.1.2 Control de campos de texto libre

Los campos de texto libre en el pipeline son:

- `motivo_reprograma_bloque` y `motivo_categoria_bloque` (en `gold_contexto_auditoria_30min`): descripciones estandarizadas de motivos de reprogramación del despacho (e.g., "FALLA DE EQUIPO", "MANTENIMIENTO PROGRAMADO"). Estos campos son categóricos y proceden de nomenclatura regulada por el COES; no obstante, se aplica la siguiente salvaguarda: **estos campos no se propagan al modelo predictivo ni a la capa de serving del dashboard**. Las columnas `label_rdo`, `alfa`, `beta`, `gamma`, `chm` y derivados de texto libre del PR-06 están explícitamente bloqueadas en `FORBIDDEN_LEAKAGE_COLS` del módulo `src/ml/feature_engineering.py` para evitar tanto fuga temporal como exposición incidental de menciones de personal operativo.

- `shap_top1`, `shap_top2`, `shap_top3` (en `fact_predicciones_alerta`): contienen exclusivamente nombres de variables predictoras del modelo (e.g., `ratio_term_ejec_vs_prog=0.85`), sin posibilidad de datos personales.

### 5.1.3 Integridad criptográfica y control de accesos

- **Hashes SHA-256**: cada archivo Bronze tiene su firma registrada en `data/manifests/bronze_real_sha256.json` (8 archivos, generado en UTC). Cualquier modificación posterior a los archivos fuente es detectable por comparación de hash.
- **Exclusión de datos y secretos en Git**: `.gitignore` excluye `lake_v2/`, `lake_v2_gold.zip`, `.venv_wsl/`, `.env`, `*.pem`, `*.key` y directorios de datos intermedios. El repositorio no contiene credenciales, tokens ni datos brutos.
- **Control de acceso al pipeline**: la ejecución se realiza en entorno local controlado (`~/.venvs/coes_fase4`) con dependencias fijadas en `requirements.txt`. El dashboard Streamlit opera en modo lectura exclusiva sobre `lake_v2/serving/star_schema/` a través de DuckDB in-memory.

---

## 5.2 Linaje y gobierno del dato

### 5.2.1 Arquitectura de linaje extremo a extremo

La matriz completa de linaje se encuentra en [`docs/linaje_datos.csv`](linaje_datos.csv), que traza cada KPI y métrica del dashboard desde la vista de Streamlit hasta el archivo Bronze del COES, pasando por tabla serving, tabla gold/ML, transformación aplicada y responsable técnico.

El linaje sigue la arquitectura medallón estricta:

```
Bronze (8 archivos COES)
  → Silver (limpieza, tipado, deduplicación)
    → Gold (7 tablas analíticas particionadas por anio, Hive-style)
      → ML Outputs (predicciones, deriva, registro de modelos)
        → Serving / Star Schema (4 dimensiones + 3 hechos, Parquet + Snappy)
          → DuckDB semantic_views.sql (4 vistas analíticas)
            → Dashboard Streamlit (5 KPIs + 4 vistas)
```

### 5.2.2 Política de no-multiplicación y NULL estricto

Cuando `etiqueta_observable = False` (el bloque carece de alguna fuente requerida para construir el target sin ambigüedad), los campos `evento_termico_nop` e `inicio_termico_nop` se preservan como **NULL estricto** en toda la cadena: `gold_observabilidad_etiqueta_30min` → `gold_feature_store_30min` → `fact_operacion_bloque_30min`. Esta política se valida automáticamente en `tests/test_star_schema.py::TestNullPreservation` (0 violaciones sobre 16 272 bloques censurados).

No se imputa, no se rellena con ceros y no se genera dato sintético alguno. Los bloques censurados se propagan con su `motivo_censura` para transparencia en el dashboard (zonas sombreadas en gris en la Vista 1).

### 5.2.3 Bloqueo de columnas con fuga temporal

El módulo `src/ml/feature_engineering.py` define `FORBIDDEN_LEAKAGE_COLS` = {`label_rdo`, `alfa`, `beta`, `gamma`, `chm`, `alfa_primer_archivo`, `beta_primer_archivo`, `gamma_primer_archivo`}. Estas columnas del PR-06 contienen resultados retroactivos de la operación que solo se conocen después del cierre del bloque y, por tanto, constituyen fuga temporal si se usan como predictores. El bloqueo se verifica en `tests/test_feature_engineering.py::test_rechaza_columnas_prohibidas`.

Adicionalmente, `NON_PREDICTOR_RETROACTIVE_COLS` = {`energia_nop_mwh`, `exposicion_bruta_soles`, `sobrecosto_neto_estimado_soles`, `n_centrales_term_nop`} se excluyen del vector de features pero se conservan en la capa gold para valorización económica ex-post.

### 5.2.4 Procedimiento de auditoría inversa (dashboard → Bronze)

Para rastrear cualquier dato mostrado en el dashboard hasta su archivo fuente original:

1. **Identificar el bloque en el dashboard**: anotar `tiempo_key` (formato `YYYYMMDD_HHMM`) y la métrica o alerta observada (e.g., `prob_recalibrada = 0.72` en Vista 1).

2. **Consultar la tabla de hechos en serving**: ejecutar en DuckDB:
   ```sql
   SELECT * FROM fact_predicciones_alerta WHERE tiempo_key = '20241015_1430';
   ```

3. **Cruzar con la dimensión temporal**: obtener `fecha_hora` y `fecha_despacho` desde `dim_tiempo_operativo` para el mismo `tiempo_key`.

4. **Verificar en la tabla gold fuente**: leer `gold_feature_store_30min` o `ml_outputs/predicciones_alerta_30min` filtrando por `fecha_hora`.

5. **Descender a Silver y Bronze**: con la `fecha_hora` y la `central` (si aplica), consultar los archivos Silver correspondientes (e.g., `silver/a1_despacho_programado`) y verificar contra el archivo Bronze original cuyo SHA-256 consta en `data/manifests/bronze_real_sha256.json`.

6. **Validar integridad**: recalcular el SHA-256 del archivo Bronze en disco y compararlo con el manifiesto. Si coincide, el dato es trazable y no ha sido alterado.

---

## 5.3 Sesgos y consecuencias del modelo

### 5.3.1 Sesgo de censura: 31,2% de bloques no observables

Del total de 52 608 bloques de 30 minutos en el período analizado (diciembre 2021 – noviembre 2024), 16 272 bloques (31,0%) tienen `etiqueta_observable = False`. La censura no es aleatoria: se concentra en períodos de indisponibilidad de las fuentes A1, A2, A3 o PR-06 del COES, lo que genera un sesgo de selección en el entrenamiento del modelo.

**Meses con cobertura crítica en 2024:**

| Mes | % Bloques Etiquetables | Alerta Baja Cobertura | Consecuencia |
|---|---|---|---|
| `2024-01` | 32,26% | Sí | `MODO_MONITOREO_DESCRIPTIVO`; 0 positivos etiquetables en enero |
| `2024-04` | 37,08% | Sí | `MODO_MONITOREO_DESCRIPTIVO`; ausencia de positivos observable → colapso de la calibración F1-60d en mayo |

La salvaguarda implementada es doble: (a) `alerta_baja_cobertura = True` cuando la cobertura cae por debajo del 50%, activando automáticamente `MODO_MONITOREO_DESCRIPTIVO`; y (b) el target se preserva como `NULL` estricto en los bloques censurados, evitando contaminar las métricas de desempeño con falsos negativos por ausencia de dato.

### 5.3.2 Cobertura parcial de activos: 70 vs. 101 centrales

De las 171 centrales únicas en `despacho_central_30min`, solo 70 tienen información de costo variable en la fuente A3 (`central_en_a3 = True`). Las 101 centrales restantes (`sin_costo_A3`) operan con combustibles no caros o no reportan costo variable, y quedan fuera del cálculo de exposición bruta y sobrecosto neto.

Esto implica que las métricas económicas del dashboard reflejan una **cota inferior** del impacto real, limitada a las centrales con visibilidad completa de costos. La dimensión `dim_central_combustible` expone explícitamente `estado_cobertura_a3` para que el analista identifique esta limitación.

### 5.3.3 Deriva de régimen en 2024: prior shift y PSI

El año 2024 presenta un cambio estructural de régimen frente al período de entrenamiento (2021-2023):

- **Prior shift**: la programación de generación térmica cara aumentó ×2,3 respecto al promedio histórico, alterando la prevalencia del target `evento_termico_nop`.
- **PSI máximo Top-5**: el `psi_max_top5` creció sostenidamente desde 0,65 en enero hasta **13,28 en noviembre 2024**, indicando deriva severa en las covariables más influyentes. El cálculo de PSI aplica suavizado `eps = 1e-6` en deciles vacíos para evitar divisiones por cero en la divergencia KL, lo cual produce valores de PSI conservadores cuando la distribución de 2024 se aleja radicalmente de la referencia.
- **Pico de 108,50 FP/sem en mayo 2024** (estrategia `recalibrado_60d_f1`): la ausencia de positivos etiquetables en abril 2024 (cobertura 37,08%) dejó la ventana de calibración de 60 días sin ejemplos positivos, colapsando el umbral F1 y disparando masivamente falsas alarmas. La estrategia operativa `recalibrado_60d_max_fp_3_sem` se diseñó precisamente para acotar este riesgo: su promedio anual es **1,97 FP/sem** en bloques, muy por debajo del límite de fatiga operativa.

### 5.3.4 Consecuencias asimétricas de FP vs. FN y salvaguarda human-in-the-loop

Las consecuencias de error del modelo son marcadamente asimétricas:

| Tipo de error | Consecuencia operativa | Gravedad |
|---|---|---|
| **Falso Positivo (FP)** | El operador de sala recibe una alerta de generación térmica cara inminente que no se materializa. Genera fatiga operativa y potencial desensibilización si es recurrente (> 3 FP/semana). | Moderada (costo de atención) |
| **Falso Negativo (FN)** | Un evento de generación térmica cara no programada ocurre sin alerta previa. Se pierde la ventana de 2–4 horas para redespacho preventivo, con exposición económica directa (sobrecosto neto promedio de S/ 54 171 / día observable en 2024). | Alta (costo económico directo) |

**Salvaguardas implementadas:**

1. **Alerta consultiva, no ejecutiva**: el sistema emite alertas informativas al operador de sala de control. No existe despacho automático ni decisión vinculante basada en el modelo. La acción de redespacho preventivo requiere validación humana del coordinador de despacho.

2. **Doble régimen de umbral**: el operador puede alternar entre la estrategia de alta sensibilidad (`recalibrado_60d_f1`, que prioriza recall a costa de más falsas alarmas) y la estrategia operativa (`recalibrado_60d_max_fp_3_sem`, que acota falsas alarmas ≤ 3/semana a costa de menor recall).

3. **`MODO_MONITOREO_DESCRIPTIVO`**: cuando la cobertura mensual cae por debajo del 50%, el sistema desactiva automáticamente la función de alerta y pasa a modo descriptivo, mostrando únicamente métricas históricas sin emitir predicciones operativas. Esto previene que decisiones se tomen sobre alertas generadas con datos insuficientes.

---

## 5.4 Marco normativo aplicable

### 5.4.1 Ley N.º 29733 (Ley de Protección de Datos Personales del Perú)

La Ley N.º 29733 y su reglamento (D.S. N.º 003-2013-JUS) regulan el tratamiento de datos personales en el Perú. En el contexto de este proyecto:

- **Datos personales**: la ley define dato personal como "toda información sobre una persona natural que la identifica o la hace identificable". El dataset operativo del COES no contiene datos de personas naturales: las entidades registradas son centrales de generación (identificadas por nombre técnico) y empresas generadoras (personas jurídicas de registro público en OSINERGMIN).

- **Alcance**: al no procesarse datos personales según la definición del artículo 2 de la Ley N.º 29733, **no se requiere inscripción de banco de datos personales, consentimiento informado de titulares ni evaluación de impacto en protección de datos personales (EIPD)**. No obstante, se mantienen las medidas de seguridad técnica descritas en la sección 5.1.3 como práctica de buen gobierno.

- **Salvaguarda preventiva**: si en iteraciones futuras se incorporaran datos que pudieran contener información identificable (e.g., registros de personal de planta o consumo por subestación de distribución), deberá realizarse una EIPD y registrar el banco de datos ante la Autoridad Nacional de Protección de Datos Personales (ANPDP) antes de cualquier tratamiento.

### 5.4.2 Términos de uso de información pública del COES

Las fuentes A1–A4 y B1–B4 provienen del portal público de información del COES-SINAC, que publica datos operativos del SEIN en cumplimiento de la Ley N.º 28832 (Ley para Asegurar el Desarrollo Eficiente de la Generación Eléctrica) y los Procedimientos Técnicos del COES. El uso de esta información con fines de análisis, investigación y mejora de la eficiencia operativa del sector eléctrico es consistente con los objetivos regulatorios del COES.

### 5.4.3 Rigor semántico sobre el Procedimiento Técnico PR-06

El PR-06 (Programación de la Operación de Corto Plazo) del COES establece las reglas para la programación diaria del despacho. En el pipeline:

- Los campos `label_rdo`, `alfa`, `beta`, `gamma` y `chm` del archivo `B4_pr06_rdo.csv` representan los **Resultados de la Operación (RDO)**, que son retroactivos por naturaleza: se calculan después del cierre del bloque operativo. Por esta razón, están bloqueados como predictores (`FORBIDDEN_LEAKAGE_COLS`) para evitar fuga temporal.

- El campo `motivo_reprograma_bloque` utiliza la taxonomía oficial de motivos del COES (falla de equipo, restricción de transmisión, mantenimiento programado, etc.) sin modificación ni reinterpretación.

---

## 5.5 Aporte al desarrollo sostenible de la región

### 5.5.1 Eficiencia económica y operativa del SEIN

El sistema de alerta temprana tiene como objetivo reducir el sobrecosto de la generación térmica cara no programada (D2 y R500 principalmente) al anticipar estos eventos con 2 a 4 horas de margen, habilitando al coordinador de despacho para explorar opciones de redespacho preventivo (reprogramar unidades de menor costo, gestionar reserva rotante, o coordinar con generadores hidráulicos).

**Magnitud del impacto potencial (backtest 2024):**

| Estrategia | Escenario α | Ahorro Bruto Estimado (S/) | % sobre Exposición Bruta |
|---|---|---|---|
| `recalibrado_60d_max_fp_3_sem` | Conservador (5%) | 484 382 | 2,54% |
| `recalibrado_60d_max_fp_3_sem` | Base (10%) | 968 765 | 5,08% |
| `recalibrado_60d_f1` | Conservador (5%) | 798 316 | 4,18% |
| `recalibrado_60d_f1` | Base (10%) | 1 596 633 | 8,37% |

Estos ahorros benefician al sistema eléctrico en su conjunto y, en última instancia, a los consumidores finales del SEIN, al reducir los costos marginales que se trasladan a las tarifas reguladas.

### 5.5.2 Contribución a la macrorregión norte (La Libertad)

La concentración de centrales térmicas caras en la zona norte del SEIN (barras troncales de 220 kV en la macrorregión norte: TALARA 220, PIURA OESTE 220, CAJAMARCA 220) implica que la reducción de generación térmica no programada en estas zonas tendría un impacto directo en la estabilidad de precios nodales de la región, beneficiando la competitividad industrial y el acceso energético en La Libertad y regiones aledañas.

### 5.5.3 Alcance y limitaciones sobre emisiones de CO2

El pipeline cuantifica la generación térmica no programada en **MWh** (`energia_nop_mwh`) y su valorización económica en soles (`exposicion_bruta_soles`, `sobrecosto_neto_estimado_soles`). **No se miden directamente las emisiones de CO2** asociadas, dado que:

1. Los factores de emisión por central y combustible no forman parte de las fuentes COES ingestadas (A1–A4, B1–B4).
2. La relación entre generación térmica no programada y emisiones evitables requiere un análisis de despacho contrafactual (qué unidad habría generado en su lugar) que excede el alcance del presente sistema de alerta.

No obstante, la reducción de la generación térmica cara con combustibles como D2 (diésel) y R500 (residual) —que tienen factores de emisión significativamente superiores a las alternativas hidroeléctricas y de gas natural— contribuye indirectamente a la reducción de la huella de carbono del SEIN. Una extensión futura del sistema podría incorporar factores de emisión por tecnología para cuantificar explícitamente las toneladas de CO2eq evitables.
