#!/usr/bin/env bash
set -euo pipefail

# ── Parámetros configurables ────────────────────────────────────────
DATA_ROOT="${1:-data/sample}"
PROFILE="${2:-local_dev}"

# Manifiesto: por defecto usa el de sample; para datos completos usa el real.
if [ "${DATA_ROOT}" = "data/sample" ]; then
    MANIFEST="${3:-data/manifests/bronze_sha256.json}"
else
    MANIFEST="${3:-data/manifests/bronze_real_sha256.json}"
fi

echo "=== Pipeline COES Big Data ==="
echo "DATA_ROOT : ${DATA_ROOT}"
echo "PROFILE   : ${PROFILE}"
echo "MANIFEST  : ${MANIFEST}"
echo ""

# ── PASO 1: Verificación de manifiesto Bronze ───────────────────────
echo "[1/7] Verificando integridad Bronze (SHA-256) …"
python scripts/verificar_manifiesto.py \
    --bronze "${DATA_ROOT}/bronze" \
    --manifest "${MANIFEST}"

# ── PASO 2: ETL (Bronze → Silver → Gold) ────────────────────────────
echo "[2/7] Ejecutando ETL pipeline (all stages) …"
python src/etl/pipeline.py \
    --input "${DATA_ROOT}/bronze" \
    --output "${DATA_ROOT}" \
    --profile "${PROFILE}" \
    --stage all

# ── PASO 3: Feature engineering ─────────────────────────────────────
echo "[3/7] Feature engineering …"
python src/ml/feature_engineering.py \
    --input "${DATA_ROOT}/gold/gold_feature_store_30min" \
    --output "${DATA_ROOT}/gold/ml_features_30min" \
    --summary-out "evidencias_modelo/feature_engineering_summary.json"

# ── PASO 4: Entrenamiento del modelo ────────────────────────────────
echo "[4/7] Entrenamiento del modelo (walk-forward CV) …"
python src/ml/entrenamiento_modelo.py \
    --input "${DATA_ROOT}/gold/ml_features_30min" \
    --output "${DATA_ROOT}/ml_out" \
    --summary-out "evidencias_modelo/comparativa_modelos.json" \
    --n-boot 1000 \
    --seed 42

# ── PASOS 5-7: Solo para datos completos (no sample) ────────────────
if [ "${DATA_ROOT}" != "data/sample" ]; then

    # ── PASO 5: Explicabilidad SHAP y costos ────────────────────────
    echo "[5/7] Explicabilidad SHAP y costos de arranque …"
    python src/ml/explicabilidad_y_costos.py \
        --features-input "${DATA_ROOT}/gold/ml_features_30min" \
        --feature-store-input "${DATA_ROOT}/gold/gold_feature_store_30min" \
        --preds-input "${DATA_ROOT}/ml_out/rf_v1_2024_walkforward/predicciones_oof.parquet" \
        --output-parquet "${DATA_ROOT}/ml_out/rf_v1_2024_walkforward/preds_shap.parquet" \
        --summary-out "evidencias_modelo/explicabilidad_summary.json" \
        --fig-dir "evidencias_modelo/figuras"

    # ── PASO 6: Backtest económico ──────────────────────────────────
    echo "[6/7] Backtest económico 2024 …"
    python src/ml/backtest_economico.py \
        --features-input "${DATA_ROOT}/gold/ml_features_30min" \
        --feature-store-input "${DATA_ROOT}/gold/gold_feature_store_30min" \
        --preds-shap-input "${DATA_ROOT}/ml_out/rf_v1_2024_walkforward/preds_shap.parquet" \
        --comparativa-json "evidencias_modelo/comparativa_modelos.json" \
        --explicabilidad-json "evidencias_modelo/explicabilidad_summary.json" \
        --output-dir "${DATA_ROOT}/ml_out" \
        --summary-out "evidencias_modelo/backtest_economico_summary.json"

    # ── PASO 7: Capa semántica Serving (Star Schema) ────────────────
    echo "[7/7] Construyendo Star Schema (serving) …"
    python src/serving/build_star_schema.py \
        --gold-input "${DATA_ROOT}/gold" \
        --serving-output "${DATA_ROOT}/serving/star_schema" \
        --summary-out "evidencias_serving/star_schema_summary.json"

else
    echo "[5-7/7] Saltados (DATA_ROOT = data/sample)"
fi

echo ""
echo "=== Pipeline completado exitosamente ==="
