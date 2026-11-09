#!/usr/bin/env bash
# Scale sweep: Python ProcessPool+NumPy vs Spark on same VM / same sizes.
set -euo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:/usr/local/hadoop-2.6.0/bin:/usr/local/hadoop-2.6.0/sbin:/usr/local/spark-1.2.2-bin-hadoop2.4/bin:/usr/local/jdk1.8.0_40/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark-1.2.2-bin-hadoop2.4}"
export HOUSING="${HOUSING:-/home/bitnami/uk_housing}"
PAR="$HOUSING/data/parallel"
OUTCSV="$HOUSING/data/processed/scaling_python_vs_spark.csv"
MAX_ITER="${MAX_ITER:-5}"
WORKERS="${WORKERS:-4}"
MANIFEST="$PAR/scaling_manifest.csv"

mkdir -p "$HOUSING/data/processed"
echo "engine,tag,points,iters,workers,wall_seconds,status" > "$OUTCSV"

# Ensure numpy
python -c 'import numpy' 2>/dev/null || {
  echo bitnami | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python-numpy
}

run_python() {
  local tag="$1" points="$2" data="$3"
  local outlog="$HOUSING/data/processed/scale_python_${tag}.log"
  local timing="$HOUSING/data/processed/scale_python_${tag}_timing.txt"
  echo ">>> Python $tag points=$points"
  set +e
  # Whole array in RAM; OOM / crash => status=oom_or_error
  python "$HOUSING/parallel/python_kmeans_uk.py" \
      --data "$data" \
      --centroids "$PAR/init_centroids_k4.txt" \
      --k 4 \
      --max-iter "$MAX_ITER" \
      --workers "$WORKERS" \
      --out "$timing" \
      >"$outlog" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "python,$tag,$points,$MAX_ITER,$WORKERS,,oom_or_error" >> "$OUTCSV"
    echo "Python FAILED tag=$tag rc=$rc (see $outlog)"
    tail -n 20 "$outlog" || true
    return 0
  fi
  local wall
  wall=$(python - <<PY
import re
t=open("$timing").read()
m=re.search(r"wall_seconds=([0-9.]+)", t)
print(m.group(1) if m else "")
PY
)
  if [[ -z "$wall" ]]; then
    echo "python,$tag,$points,$MAX_ITER,$WORKERS,,oom_or_error" >> "$OUTCSV"
    echo "Python missing wall tag=$tag"
    return 0
  fi
  echo "python,$tag,$points,$MAX_ITER,$WORKERS,$wall,ok" >> "$OUTCSV"
  echo "Python OK $tag wall=$wall"
}

run_spark() {
  local tag="$1" points="$2" datafile="$3"
  local base
  base=$(basename "$datafile")
  local scala="$HOUSING/parallel/_gen_spark_${tag}.scala"
  local outlog="$HOUSING/data/processed/scale_spark_${tag}.log"
  sed -e "s|__DATA_FILE__|$base|g" \
      -e "s|__TAG__|$tag|g" \
      -e "s|__MAX_ITER__|$MAX_ITER|g" \
      "$HOUSING/parallel/spark_kmeans_scale.scala" > "$scala"
  echo ">>> Spark $tag points=$points file=$base"
  set +e
  # Give Spark most of the 6GB VM so it can cache partitions (default heap is tiny).
  printf ':quit\n' | "$SPARK_HOME/bin/spark-shell" \
      --driver-memory 3g \
      --executor-memory 3g \
      -i "$scala" >"$outlog" 2>&1
  local rc=$?
  set -e
  local wall
  wall=$(python - <<PY
import re
t=open("$outlog").read()
ms=re.findall(r"wall_seconds=([0-9.eE+-]+)", t)
print(ms[-1] if ms else "")
PY
)
  if [[ -z "$wall" ]]; then
    echo "spark,$tag,$points,$MAX_ITER,4,,error" >> "$OUTCSV"
    echo "Spark FAILED tag=$tag"
    return 0
  fi
  echo "spark,$tag,$points,$MAX_ITER,4,$wall,ok" >> "$OUTCSV"
  echo "Spark OK $tag wall=$wall"
}

# Prefer smaller → larger so we get early results even if later OOMs
# Manifest path column is a filename under $PAR (LF, no drive letter).
while IFS=, read -r tag points path; do
  tag=$(printf '%s' "$tag" | tr -d '\r')
  points=$(printf '%s' "$points" | tr -d '\r')
  path=$(printf '%s' "$path" | tr -d '\r')
  [[ "$tag" == "tag" ]] && continue
  [[ -z "$tag" ]] && continue
  base="${path##*/}"
  data="$PAR/$base"
  if [[ ! -f "$data" ]]; then
    echo "SKIP missing $data"
    continue
  fi
  pts=$(wc -l < "$data" | tr -d ' ')
  run_python "$tag" "$pts" "$data"
  run_spark "$tag" "$pts" "$data"
done < "$MANIFEST"

echo "=== scaling CSV ==="
cat "$OUTCSV"
