#!/bin/sh
# Time Hadoop Streaming k-means at every fair size on THIS machine.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/parallel/fair_config.env"
export HOUSING="$ROOT"
CENT="$ROOT/data/parallel/fair/init_centroids_k${K}.txt"
FULL="$ROOT/data/parallel/uk_features_scaled_all.txt"
HOST=$(hostname 2>/dev/null || echo unknown)
HOST=$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_')
OUTDIR="$ROOT/data/processed/fair_timing/$HOST"
mkdir -p "$OUTDIR"

if [ ! -f "$CENT" ]; then
  echo "Missing $CENT" >&2
  echo "Run prepare_fair_inputs.py and copy data/parallel/fair/ here." >&2
  exit 1
fi

run_one() {
  tag=$1
  data=$2
  echo "MapReduce $tag  $data"
  HOUSING="$ROOT" \
    DATA="$data" \
    INIT="$CENT" \
    K="$K" \
    MAX_ITER="$MAX_ITER" \
    EPS="$EPS" \
    KMEANS_TAG="$tag" \
    OUT_FILE="$OUTDIR/mapreduce_${tag}.txt" \
    HDFS_IN="/user/$(whoami)/uk_kmeans_in_${tag}" \
    HDFS_OUT_BASE="/user/$(whoami)/uk_kmeans_out_${tag}" \
    "$ROOT/parallel/mr_kmeans/run_mr_kmeans.sh"
}

for spec in $SIZES; do
  run_one "n${spec}" "$ROOT/data/parallel/fair/n${spec}.txt"
done
run_one "all" "$FULL"
echo "MapReduce results: $OUTDIR"
