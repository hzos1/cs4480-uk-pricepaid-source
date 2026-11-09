#!/bin/sh
# Choose k (2..8, n_init=10) on 200k, 1M, 5M, 10M, 20M, and every row.
# Run this on the other computer. Do not start it on the Mac.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
DATA="$ROOT/data/parallel/uk_features_scaled_all.txt"

if [ ! -f "$DATA" ]; then
  echo "Missing $DATA" >&2
  echo "Copy data/parallel/uk_features_scaled_all.txt with the project." >&2
  exit 1
fi

"$PY" -c "import numpy, pandas, sklearn" || {
  echo "Install numpy pandas scikit-learn first." >&2
  exit 1
}

"$PY" -u "$ROOT/local_python/choose_k_by_size.py"
echo "Results: $ROOT/data/processed/choose_k_by_size/best_k_by_size.csv"
