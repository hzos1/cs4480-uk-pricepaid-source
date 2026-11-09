#!/bin/sh
# Time Python Lloyd k-means at every fair size on THIS machine.
# CPU model, core count, and memory capacity are written into each result file.
# Full-file runs copy the matrix to each worker. On a 16 GB machine set
# PYTHON_KMEANS_WORKERS=2 before starting.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/parallel/fair_config.env"
PY=${PY:-python3}
CENT="$ROOT/data/parallel/fair/init_centroids_k${K}.txt"
FULL="$ROOT/data/parallel/uk_features_scaled_all.txt"
HOST=$(hostname 2>/dev/null || echo unknown)
HOST=$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_')
OUTDIR="$ROOT/data/processed/fair_timing/$HOST"
mkdir -p "$OUTDIR"

if [ ! -f "$CENT" ]; then
  echo "Missing $CENT" >&2
  echo "Run: $PY $ROOT/parallel/prepare_fair_inputs.py" >&2
  exit 1
fi

run_one() {
  tag=$1
  data=$2
  echo "Python $tag  $data"
  "$PY" -u "$ROOT/parallel/python_kmeans_uk.py" \
    --data "$data" \
    --centroids "$CENT" \
    --k "$K" \
    --max-iter "$MAX_ITER" \
    --tag "$tag" \
    --out "$OUTDIR/python_${tag}.txt"
}

for spec in $SIZES; do
  run_one "n${spec}" "$ROOT/data/parallel/fair/n${spec}.txt"
done
run_one "all" "$FULL"
echo "Python results: $OUTDIR"
