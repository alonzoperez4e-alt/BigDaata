#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/sample}"
PROFILE="${2:-local_dev}"

python scripts/verificar_manifiesto.py --bronze coes_pipeline/lake_real/bronze --manifest data/manifests/bronze_sha256.json
LAKE_ROOT="$DATA_ROOT" ESCALA="1x" SHUFFLE_PARTITIONS="8" BROADCAST="1" CORES="8" DRIVER_MEM="4g" python src/etl/pipeline.py
python src/ml/entrenamiento_modelo.py
