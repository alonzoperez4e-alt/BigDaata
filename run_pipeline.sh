#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:-data/sample}"
PROFILE="${2:-local_dev}"
python scripts/verificar_manifiesto.py --bronze "$DATA_ROOT/bronze" --manifest data/manifests/bronze_sha256.json
python src/etl/pipeline.py --input "$DATA_ROOT/bronze" --output "$DATA_ROOT" --profile "$PROFILE" --stage all
python src/ml/entrenamiento_modelo.py --input "$DATA_ROOT/gold" --output "$DATA_ROOT/ml_out"
