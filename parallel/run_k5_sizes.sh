#!/bin/sh
# k=5 on 200k, 1M, 5M, 10M, 20M, and every row.
# Run this on the other computer. Do not start it on the Mac.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
DATA="$ROOT/data/parallel/uk_features_scaled_all.txt"

if [ ! -f "$DATA" ]; then
  echo "Missing $DATA" >&2
  echo "Copy data/parallel/uk_features_scaled_all.txt with the project, or put pp-complete.csv in $ROOT and run:" >&2
  echo "  $PY -u $ROOT/local_python/build_uk_panel.py" >&2
  exit 1
fi

"$PY" -c "import numpy, pandas, sklearn" || {
  echo "Install numpy pandas scikit-learn first." >&2
  exit 1
}

"$PY" -u "$ROOT/local_python/kmeans_k5_sizes.py"
echo "Results: $ROOT/data/processed/kmeans_k5/"
