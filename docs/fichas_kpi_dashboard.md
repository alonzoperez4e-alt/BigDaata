# Fichas Técnicas de KPIs — Dashboard Operativo-Analítico COES (Tarea 4.2)

Este documento especifica los **5 KPIs centrales** expuestos en la capa de serving (`lake_v2/serving/star_schema/`) y visualizados en `src/dashboard/app.py`.

---

## KPI 1: Tasa de Detección Anticipada (Recall en Bloques Observables)

- **Nombre:** Tasa de Detección Anticipada de Eventos Térmicos No Programados (`recall_bloques_pct`).
- **Definición operativa:** Proporción de bloques de 30 minutos con ocurrencia real y observable de generación térmica cara no programada (`evento_termico_nop = 1` y `etiqueta_observable = TRUE`) que fueron alertados con 2 a 4 horas de anticipación por la estrategia activa.
- **Fórmula exacta:**
  $$\text{Recall (\%)} = \frac{TP}{TP + FN} \times 100$$
  donde $TP$ y $FN$ se contabilizan exclusivamente sobre bloques con `resultado IN ('TP', 'FN')` (excluyendo `NO_VERIFICABLE` cuando existe censura).
- **Frecuencia de actualización:** Cada bloque de 30 minutos (evaluación retrospectiva una vez cerrada la ventana de observabilidad $t + 4\text{ h}$).
- **Umbral de disparo:** $< 35\%$ sostenido durante una ventana móvil de 30 días en régimen operativo (`recalibrado_60d_max_fp_3_sem`).
- **Cargo responsable y acción concreta en la Dirección de Operaciones del COES:** **Subdirector de Programación y Coordinación de la Operación (CCO)** — Ordenar revisión del umbral de calibración y verificar si la pérdida de sensibilidad responde a nueva configuración de despacho o a censura de señales A1/A4.

---

## KPI 2: Carga de Falsas Alarmas Semanales (Fatiga Operativa)

- **Nombre:** Tasa de Falsas Alarmas por Semana Observable (`fp_por_semana`).
- **Definición operativa:** Número promedio de alertas falsas (`resultado = 'FP'`) emitidas por semana equivalente de operación observable (336 bloques de 30 min = 7 días). Controla la fatiga del operador en sala de control.
- **Fórmula exacta:**
  $$\text{FP/semana} = \frac{FP}{\text{bloques\_etiquetables} / 336.0}$$
- **Frecuencia de actualización:** Semanal y mensual (con seguimiento continuo en bloques de 30 min).
- **Umbral de disparo:** $> 3.0\text{ FP/semana}$ en la estrategia operativa (`recalibrado_60d_max_fp_3_sem`), o picos anómalos $> 15.0\text{ FP/semana}$ en la estrategia de alta sensibilidad (`recalibrado_60d_f1`, como `108.50 FP/sem` en `2024-05`).
- **Cargo responsable y acción concreta en la Dirección de Operaciones del COES:** **Jefe de Sala de Control (Centro de Control COES)** — Congelar la emisión automática de alertas preventivas de la estrategia afectada, operar temporalmente bajo `recalibrado_60d_max_fp_3_sem` y solicitar recalibración con ventana de 60 días validada.

---

## KPI 3: Exposición Bruta y Sobrecosto Neto por Día Observable

- **Nombre:** Intensidad Económica Diaria Normalizada (`exposicion_bruta_dia_obs`, `sobrecosto_neto_dia_obs`).
- **Definición operativa:** Valorización económica diaria (en S/ por día observable de 48 bloques) de la energía térmica cara no programada (exposición bruta) y del diferencial neto frente al costo marginal troncal (sobrecosto neto), evitando sesgos por meses con distinta cobertura de observabilidad.
- **Fórmula exacta:**
  $$\text{Sobrecosto Neto / Día Obs.} = \frac{\sum \text{sobrecosto\_neto\_estimado\_soles}}{\text{bloques\_etiquetables} / 48.0}$$
  $$\text{Exposición Bruta / Día Obs.} = \frac{\sum \text{exposicion\_bruta\_soles}}{\text{bloques\_etiquetables} / 48.0}$$
- **Frecuencia de actualización:** Diaria y mensual.
- **Umbral de disparo:** Incremento $> +30\%$ respecto al promedio móvil de los últimos 3 meses o superación de $\text{S/ } 65{,}000\text{ / día observable}$ en sobrecosto neto.
- **Cargo responsable y acción concreta en la Dirección de Operaciones del COES:** **Director de Operaciones / Coordinador de Despacho Económico** — Priorizar auditoría de reprogramas (PR-06) y restricciones de transmisión asociadas al despacho térmico forzado en las barras troncales con mayor `cmg_spread_p90_p50`.

---

## KPI 4: Ahorro Potencial Estimado por Mitigación Operativa ($\alpha$)

- **Nombre:** Ahorro Neto y Bruto Proyectado por Anticipación (`ahorro_neto_soles`, `ahorro_bruto_soles`).
- **Definición operativa:** Estimación conservadora ($\alpha = 5\%$), base ($\alpha = 10\%$) u optimista ($\alpha = 15\%$) del ahorro económico alcanzable al anticipar bloques térmicos no programados (`resultado = 'TP'`) con margen de 2–4 horas para redespacho preventivo o gestión de reserva rotante.
- **Fórmula exacta:**
  $$\text{Ahorro Neto (S/)} = \alpha \times \sum_{\text{TP}} \text{sobrecosto\_neto\_estimado\_soles}$$
  $$\text{\% Ahorro Neto} = \frac{\text{Ahorro Neto (S/)}}{\text{Sobrecosto Neto Total Anual (S/)}} \times 100$$
- **Frecuencia de actualización:** Mensual y trimestral (cierre de backtest económico).
- **Umbral de disparo:** $\% \text{ Ahorro Neto} < 2.0\%$ (incumplimiento de la meta mínima de impacto económico del 2%–4% bajo $\alpha$ operativo).
- **Cargo responsable y acción concreta en la Dirección de Operaciones del COES:** **Subdirector de Evaluación y Análisis Post-Operativo** — Evaluar la efectividad del protocolo de respuesta temprana ante alertas en sala de control y ajustar las acciones de mitigación de despacho preventivo.

---

## KPI 5: Cobertura de Observabilidad y Salud de Deriva (PSI Máximo Top-5)

- **Nombre:** Cobertura Mensual de Etiqueta y Deriva de Covariables (`pct_bloques_etiquetables`, `psi_max_top5`).
- **Definición operativa:** Porcentaje de bloques de 30 minutos donde están presentes todas las fuentes requeridas para construir la etiqueta sin censura (`etiqueta_observable = TRUE`), acompañado del Population Stability Index (PSI) máximo entre las 5 variables SHAP más influyentes (con suavizado `eps = 1e-6`).
- **Fórmula exacta:**
  $$\text{Cobertura (\%)} = \frac{\text{bloques\_etiquetables}}{\text{bloques\_totales}} \times 100$$
  $$\text{PSI} = \sum_{b=1}^{10} (p_{\text{actual}, b} - p_{\text{ref}, b}) \ln\left(\frac{p_{\text{actual}, b}}{p_{\text{ref}, b}}\right), \quad p_b \leftarrow \text{norm}(f_b + 10^{-6})$$
- **Frecuencia de actualización:** Mensual (con monitoreo diario de `motivo_censura`).
- **Umbral de disparo:** `pct_bloques_etiquetables < 50.0%` (`alerta_baja_cobertura = TRUE`, observado en `2024-01` con 32.26% y `2024-04` con 37.08%) o `psi_max_top5 >= 0.25` (deriva severa por cambio de régimen).
- **Cargo responsable y acción concreta en la Dirección de Operaciones del COES:** **Responsable de Gobierno del Dato y Analítica Operativa (TI / Operaciones COES)** — Activar automáticamente `MODO_MONITOREO_DESCRIPTIVO` cuando la cobertura caiga por debajo del 50%, auditar el `motivo_censura` dominante (fallas de ingesta A1/A2/A3/PR-06) y programar reentrenamiento de `rf_v1_2024_walkforward` cuando la deriva sea estructural.

