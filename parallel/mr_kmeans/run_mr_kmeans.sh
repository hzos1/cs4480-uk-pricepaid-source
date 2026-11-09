#!/usr/bin/env bash
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
HOUSING="${HOUSING:-$ROOT}"
# shellcheck disable=SC1091
if [ -f "$HOUSING/parallel/fair_config.env" ]; then
  . "$HOUSING/parallel/fair_config.env"
fi
# shellcheck disable=SC1091
if [ -f "$HOUSING/parallel/env.sh" ]; then
  . "$HOUSING/parallel/env.sh"
fi

DATA="${DATA:-$HOUSING/data/parallel/fair/n200000.txt}"
K="${K:-3}"
MAX_ITER="${MAX_ITER:-300}"
EPS="${EPS:-0.0001}"
INIT="${INIT:-$HOUSING/data/parallel/fair/init_centroids_k${K}.txt}"
WORKDIR="${WORKDIR:-$HOUSING/parallel/mr_kmeans}"
HDFS_USER="${HDFS_USER:-$(whoami)}"
HDFS_IN="${HDFS_IN:-/user/$HDFS_USER/uk_kmeans_in}"
HDFS_OUT_BASE="${HDFS_OUT_BASE:-/user/$HDFS_USER/uk_kmeans_out}"
KMEANS_TAG="${KMEANS_TAG:-run}"
OUT_FILE="${OUT_FILE:-$HOUSING/data/processed/fair_timing/mapreduce_${KMEANS_TAG}.txt}"
STREAMING_JAR="${STREAMING_JAR:-}"

if [[ -z "$STREAMING_JAR" || ! -f "$STREAMING_JAR" ]]; then
  echo "ERROR: hadoop-streaming jar not found: $STREAMING_JAR" >&2
  exit 1
fi

mkdir -p "$WORKDIR/centroids" "$WORKDIR/logs"
cp "$INIT" "$WORKDIR/centroids/centroids.txt"
chmod +x "$WORKDIR/mapper.py" "$WORKDIR/reducer.py" "$WORKDIR/update_centroids.py"

hdfs dfs -rm -r -f "$HDFS_IN" >/dev/null 2>&1 || true
hdfs dfs -mkdir -p "$(dirname "$HDFS_IN")" 2>/dev/null || true
hdfs dfs -mkdir -p "$HDFS_IN"
PUT0=$(date +%s.%N)
hdfs dfs -put -f "$DATA" "$HDFS_IN/points.txt"
PUT1=$(date +%s.%N)

echo "Streaming jar: $STREAMING_JAR"
echo "Data: $DATA"
START=$(date +%s.%N)

iter=0
moved=1
while [[ $iter -lt $MAX_ITER && $moved -eq 1 ]]; do
  iter=$((iter + 1))
  OUT="$HDFS_OUT_BASE/iter_$iter"
  hdfs dfs -rm -r -f "$OUT" >/dev/null 2>&1 || true
  CENT_LOCAL="$WORKDIR/centroids/centroids.txt"

  hadoop jar "$STREAMING_JAR" \
    -D mapreduce.job.name="uk_kmeans_iter_$iter" \
    -files "$CENT_LOCAL#centroids.txt,$WORKDIR/mapper.py,$WORKDIR/reducer.py" \
    -cmdenv CENTROIDS_FILE=centroids.txt \
    -mapper "python mapper.py" \
    -reducer "python reducer.py" \
    -input "$HDFS_IN" \
    -output "$OUT" \
    -numReduceTasks "$K"

  hdfs dfs -cat "$OUT/part-"* > "$WORKDIR/centroids/new_centroids_raw.txt"
  python "$WORKDIR/update_centroids.py" \
    "$CENT_LOCAL" \
    "$WORKDIR/centroids/new_centroids_raw.txt" \
    "$WORKDIR/centroids/centroids_next.txt" \
    "$EPS" \
    "$K"
  SHIFT=$(cat "$CENT_LOCAL.shift")
  moved=$(cat "$CENT_LOCAL.moved")
  mv "$WORKDIR/centroids/centroids_next.txt" "$CENT_LOCAL"
  echo "Iter $iter max shift = $SHIFT moved=$moved"
done

END=$(date +%s.%N)
POINTS=$(wc -l < "$DATA" | tr -d ' ')
mkdir -p "$(dirname "$OUT_FILE")"
python - <<PY
start=float("$START"); end=float("$END")
put0=float("$PUT0"); put1=float("$PUT1")
wall=end-start
put=put1-put0
print("=== MapReduce k-means done ===")
print("iters=$iter wall_seconds=%.3f" % wall)
text = (
    "engine=Hadoop_Streaming\\n"
    "tag=$KMEANS_TAG\\n"
    "points=$POINTS\\n"
    "k=$K\\n"
    "epsilon=$EPS\\n"
    "max_iter=$MAX_ITER\\n"
    "iters=$iter\\n"
    "reducers=$K\\n"
    "wall_seconds=%.6f\\n"
    "hdfs_put_seconds=%.6f\\n"
    "data=$DATA\\n"
    "centroids=$INIT\\n"
) % (wall, put)
with open("$OUT_FILE", "w") as handle:
    handle.write(text)
print("Final centroids:")
print(open("$WORKDIR/centroids/centroids.txt").read())
PY
if command -v python >/dev/null 2>&1; then
  PY_BIN=python
else
  PY_BIN=python3
fi
"$PY_BIN" "$HOUSING/parallel/machine_info.py" --append "$OUT_FILE"
echo "wrote $OUT_FILE"
