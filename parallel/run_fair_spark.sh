#!/bin/sh
# Time Spark Lloyd k-means at every fair size on THIS machine.
# Uses every core Spark puts in local[*], unless SPARK_MASTER is already set.
# Requires spark-shell (the course VM's Spark 1.2.2 shell is enough).
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/parallel/fair_config.env"
if [ -f "$ROOT/parallel/env.sh" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/parallel/env.sh"
fi
export HOUSING="$ROOT"
export KMEANS_K="$K"
export KMEANS_MAX_ITER="$MAX_ITER"
export KMEANS_EPS="$EPS"
export KMEANS_BATCH=1
CENT="$ROOT/data/parallel/fair/init_centroids_k${K}.txt"
FULL="$ROOT/data/parallel/uk_features_scaled_all.txt"
HOST=$(hostname 2>/dev/null || echo unknown)
HOST=$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_')
OUTDIR="$ROOT/data/processed/fair_timing/$HOST"
mkdir -p "$OUTDIR"
SPARK_MASTER="${SPARK_MASTER:-local[*]}"
PY=${PY:-python3}

if ! command -v spark-shell >/dev/null 2>&1; then
  echo "spark-shell is not on PATH. Run this on the Hadoop/Spark machine." >&2
  exit 1
fi
if [ ! -f "$CENT" ]; then
  echo "Missing $CENT" >&2
  echo "Run prepare_fair_inputs.py and copy data/parallel/fair/ here." >&2
  exit 1
fi

run_one() {
  tag=$1
  data=$2
  out="$OUTDIR/spark_${tag}.txt"
  echo "Spark $tag  $data"
  export KMEANS_TAG="$tag"
  export KMEANS_DATA="$data"
  export KMEANS_CENTROIDS="$CENT"
  export KMEANS_OUT="$out"
  rm -f "$out"
  spark-shell --master "$SPARK_MASTER" -i "$ROOT/parallel/spark_kmeans_uk.scala"
  if ! grep -q '^wall_seconds=' "$out"; then
    echo "Spark did not write $out" >&2
    exit 1
  fi
  "$PY" "$ROOT/parallel/machine_info.py" --append "$out"
  printf 'spark_master=%s\n' "$SPARK_MASTER" >> "$out"
}

for spec in $SIZES; do
  run_one "n${spec}" "$ROOT/data/parallel/fair/n${spec}.txt"
done
run_one "all" "$FULL"
echo "Spark results: $OUTDIR"
