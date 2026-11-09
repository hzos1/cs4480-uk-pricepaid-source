#!/usr/bin/env bash
set -euo pipefail

HOUSING="${HOUSING:-/home/bitnami/uk_housing}"
# shellcheck disable=SC1091
source "$HOUSING/parallel/env.sh"

RESULT_DIR="$HOUSING/data/processed"
mkdir -p "$RESULT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$RESULT_DIR/parallel_timing_$TS.txt"

{
  echo "=== UK Price Paid k-means parallel benchmark ==="
  echo "host=$(hostname) time=$(date -Is)"
  echo "features=6 (z-scored) k=4 data=uk_features_scaled_2018plus_sample10pct.txt"
  echo "HADOOP_HOME=$HADOOP_HOME SPARK_HOME=$SPARK_HOME"
  echo
} | tee "$LOG"

if ! jps 2>/dev/null | grep -q NameNode; then
  echo ">>> Starting HDFS" | tee -a "$LOG"
  start-dfs.sh 2>&1 | tee -a "$LOG" || true
fi
if ! jps 2>/dev/null | grep -q ResourceManager; then
  echo ">>> Starting YARN" | tee -a "$LOG"
  start-yarn.sh 2>&1 | tee -a "$LOG" || true
fi
sleep 3
jps 2>&1 | tee -a "$LOG" || true
echo | tee -a "$LOG"

SPARK_SHELL="$SPARK_HOME/bin/spark-shell"
echo "Spark shell: $SPARK_SHELL" | tee -a "$LOG"
if [[ -x "$SPARK_SHELL" ]]; then
  echo ">>> Running Spark k-means" | tee -a "$LOG"
  START=$(date +%s.%N)
  # -i leaves an interactive prompt; pipe :quit so the benchmark can continue to MR.
  printf ':quit\n' | "$SPARK_SHELL" -i "$HOUSING/parallel/spark_kmeans_uk.scala" \
    2>&1 | tee "$RESULT_DIR/spark_run_$TS.log" | tee -a "$LOG"
  END=$(date +%s.%N)
  python - <<PY | tee -a "$LOG"
print("spark_wrapper_wall_seconds=%.3f" % (float("$END")-float("$START")))
PY
else
  echo "SKIP Spark: spark-shell not found" | tee -a "$LOG"
fi

echo | tee -a "$LOG"
echo ">>> Running MapReduce Streaming k-means" | tee -a "$LOG"
bash "$HOUSING/parallel/mr_kmeans/run_mr_kmeans.sh" \
  2>&1 | tee "$RESULT_DIR/mr_run_$TS.log" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Results written to $LOG" | tee -a "$LOG"
