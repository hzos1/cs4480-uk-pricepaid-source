#!/usr/bin/env bash
# Resume incomplete tags in scaling_python_vs_spark.csv (Spark with 3g heap).
set -euo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:/usr/local/spark-1.2.2-bin-hadoop2.4/bin:/usr/local/jdk1.8.0_40/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark-1.2.2-bin-hadoop2.4}"
export HOUSING="${HOUSING:-/home/bitnami/uk_housing}"
PAR="$HOUSING/data/parallel"
OUTCSV="$HOUSING/data/processed/scaling_python_vs_spark.csv"
MANIFEST="$PAR/scaling_manifest.csv"
MAX_ITER="${MAX_ITER:-5}"
WORKERS="${WORKERS:-4}"

pkill -f "spark-shell" 2>/dev/null || true
pkill -f "python_kmeans_uk.py" 2>/dev/null || true
pkill -f "run_scaling_benchmark" 2>/dev/null || true
sleep 3

has_done() {
  # Treat ok / oom_or_error / error as finished (do not retry OOMs).
  awk -F, -v e="$1" -v t="$2" '($1==e && $2==t && ($NF=="ok" || $NF=="oom_or_error" || $NF=="error")){f=1} END{exit !f}' "$OUTCSV"
}

upsert() {
  # upsert(engine,tag,points,wall_or_empty,status)
  local eng="$1" tag="$2" pts="$3" wall="$4" st="$5"
  grep -v "^${eng},${tag}," "$OUTCSV" > "${OUTCSV}.tmp" || true
  mv "${OUTCSV}.tmp" "$OUTCSV"
  echo "${eng},${tag},${pts},${MAX_ITER},${WORKERS},${wall},${st}" >> "$OUTCSV"
}

while IFS=, read -r tag points path; do
  tag=$(printf '%s' "$tag" | tr -d '\r')
  path=$(printf '%s' "$path" | tr -d '\r')
  [[ "$tag" == "tag" || -z "$tag" ]] && continue
  base="${path##*/}"
  data="$PAR/$base"
  [[ -f "$data" ]] || { echo "missing $data"; continue; }
  pts=$(wc -l < "$data" | tr -d ' ')

  if ! has_done python "$tag"; then
    echo ">>> RESUME Python $tag pts=$pts"
    timing="$HOUSING/data/processed/scale_python_${tag}_timing.txt"
    outlog="$HOUSING/data/processed/scale_python_${tag}.log"
    set +e
    python "$HOUSING/parallel/python_kmeans_uk.py" \
      --data "$data" --centroids "$PAR/init_centroids_k4.txt" \
      --k 4 --max-iter "$MAX_ITER" --workers "$WORKERS" --out "$timing" \
      >"$outlog" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      upsert python "$tag" "$pts" "" oom_or_error
      tail -n 15 "$outlog" || true
    else
      wall=$(python -c "import re;t=open('$timing').read();m=re.search(r'wall_seconds=([0-9.]+)',t);print(m.group(1) if m else '')")
      upsert python "$tag" "$pts" "$wall" ok
      echo "Python OK $tag $wall"
    fi
  else
    echo "skip python $tag (already done)"
  fi

  if ! has_done spark "$tag"; then
    echo ">>> RESUME Spark $tag pts=$pts"
    scala="$HOUSING/parallel/_gen_spark_${tag}.scala"
    outlog="$HOUSING/data/processed/scale_spark_${tag}.log"
    sed -e "s|__DATA_FILE__|$base|g" -e "s|__TAG__|$tag|g" -e "s|__MAX_ITER__|$MAX_ITER|g" \
      "$HOUSING/parallel/spark_kmeans_scale.scala" > "$scala"
    set +e
    printf ':quit\n' | "$SPARK_HOME/bin/spark-shell" --driver-memory 3g --executor-memory 3g -i "$scala" \
      >"$outlog" 2>&1
    set -e
    wall=$(python -c "import re;t=open('$outlog').read();ms=re.findall(r'wall_seconds=([0-9.eE+-]+)',t);print(ms[-1] if ms else '')")
    if [[ -z "$wall" ]]; then
      upsert spark "$tag" "$pts" "" error
      echo "Spark FAIL $tag"; tail -n 20 "$outlog" || true
    else
      upsert spark "$tag" "$pts" "$wall" ok
      echo "Spark OK $tag $wall"
    fi
  else
    echo "skip spark $tag (already done)"
  fi
done < "$MANIFEST"

echo "=== CSV ==="
cat "$OUTCSV"
