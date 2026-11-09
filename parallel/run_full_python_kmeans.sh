#!/bin/sh
# Run Lloyd k-means on every Price Paid row (not the 2018-only subset, and not a copied file).
# Use this on the machine with more memory. Build the feature file first if it is missing.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/parallel/fair_config.env"
PY=${PY:-python3}
DATA="$ROOT/data/parallel/uk_features_scaled_all.txt"
CENT="$ROOT/data/parallel/fair/init_centroids_k${K}.txt"

if [ ! -f "$ROOT/pp-complete.csv" ]; then
  echo "Put pp-complete.csv in $ROOT first (archive (2))." >&2
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "Building all-rows feature file ..."
  "$PY" -u "$ROOT/local_python/build_uk_panel.py"
fi

if [ ! -f "$CENT" ]; then
  echo "Missing $CENT" >&2
  echo "Run: $PY $ROOT/parallel/prepare_fair_inputs.py" >&2
  exit 1
fi

"$PY" -u "$ROOT/parallel/python_kmeans_uk.py" \
  --data "$DATA" \
  --centroids "$CENT" \
  --k "$K" \
  --max-iter "$MAX_ITER" \
  --tag all \
  --out "$ROOT/data/processed/python_parallel_timing_all.txt"
