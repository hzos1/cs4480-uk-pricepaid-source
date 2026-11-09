#!/bin/bash
# MapReduce scaling sweep: same sizes/iters as Python+Spark (MAX_ITER=5).
# Appends/updates mapreduce rows in scaling_python_vs_spark.csv; keeps other engines.
set -euo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:/usr/local/hadoop-2.6.0/bin:/usr/local/hadoop-2.6.0/sbin:/usr/local/jdk1.8.0_40/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export HADOOP_HOME="${HADOOP_HOME:-/usr/local/hadoop-2.6.0}"
export HOUSING="${HOUSING:-/home/bitnami/uk_housing}"

mountpoint -q "$HOUSING" || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing "$HOUSING"
# shellcheck disable=SC1091
source "$HOUSING/parallel/env.sh"

PAR="$HOUSING/data/parallel"
MANIFEST="$PAR/scaling_manifest.csv"
OUTCSV="$HOUSING/data/processed/scaling_python_vs_spark.csv"
LOG="$HOUSING/data/processed/mr_scaling_nohup.log"
MAX_ITER="${MAX_ITER:-5}"
K="${K:-4}"
INIT="$PAR/init_centroids_k4.txt"
WORKDIR="$HOUSING/parallel/mr_kmeans"
# Space-separated tags already timed (n5e5 recovered below if needed).
SKIP_TAGS="${SKIP_TAGS:-n2p6e7}"

mkdir -p "$HOUSING/data/processed"
: >"$LOG"

if ! jps 2>/dev/null | grep -q NameNode; then start-dfs.sh; fi
if ! jps 2>/dev/null | grep -q ResourceManager; then start-yarn.sh; fi

echo "Waiting for NameNode to leave safe mode..." | tee -a "$LOG"
for i in $(seq 1 60); do
  if hdfs dfsadmin -safemode get 2>/dev/null | grep -qi 'OFF'; then
    echo "Safe mode OFF" | tee -a "$LOG"
    break
  fi
  if [[ $i -ge 18 ]]; then
    echo "Forcing safemode leave" | tee -a "$LOG"
    hdfs dfsadmin -safemode leave || true
  fi
  sleep 5
done

upsert_mr_csv() {
  local tag="$1" points="$2" wall="$3" status="$4"
  # Bitnami guest python is 2.7 — avoid pathlib / nonlocal syntax.
  python - <<PY
import csv, os
path = "$OUTCSV"
fieldnames = ["engine", "tag", "points", "iters", "workers", "wall_seconds", "status"]
rows = []
if os.path.exists(path):
    f = open(path, "rb")
    rows = list(csv.DictReader(f))
    f.close()
kept = [r for r in rows if not (r.get("engine") == "mapreduce" and r.get("tag") == "$tag")]
kept.append({
    "engine": "mapreduce",
    "tag": "$tag",
    "points": str($points),
    "iters": str($MAX_ITER),
    "workers": "4",
    "wall_seconds": "$wall",
    "status": "$status",
})
def key(r):
    eng_order = {"python": 0, "spark": 1, "mapreduce": 2}.get(r.get("engine", ""), 9)
    try:
        pts = int(float(r.get("points") or 0))
    except ValueError:
        pts = 0
    return (pts, eng_order, r.get("tag", ""))
kept.sort(key=key)
f = open(path, "wb")
w = csv.DictWriter(f, fieldnames=fieldnames)
w.writeheader()
for r in kept:
    w.writerow(dict((k, r.get(k, "")) for k in fieldnames))
f.close()
print("CSV upsert mapreduce tag=$tag status=$status wall=$wall")
PY
}

echo "=== MapReduce scaling start $(date -Is) MAX_ITER=$MAX_ITER SKIP=$SKIP_TAGS ===" | tee -a "$LOG"

while IFS=, read -r tag points path; do
  tag=$(printf '%s' "$tag" | tr -d '\r')
  points=$(printf '%s' "$points" | tr -d '\r')
  path=$(printf '%s' "$path" | tr -d '\r')
  [[ "$tag" == "tag" ]] && continue
  [[ -z "$tag" ]] && continue

  skip=0
  for s in $SKIP_TAGS; do
    [[ "$tag" == "$s" ]] && skip=1 && break
  done
  if [[ $skip -eq 1 ]]; then
    echo "SKIP tag=$tag (SKIP_TAGS)" | tee -a "$LOG"
    continue
  fi

  base="${path##*/}"
  data="$PAR/$base"
  if [[ ! -f "$data" ]]; then
    echo "SKIP missing $data" | tee -a "$LOG"
    upsert_mr_csv "$tag" "$points" "" "missing_data"
    continue
  fi
  pts=$(wc -l < "$data" | tr -d ' ')

  echo ">>> MapReduce $tag points=$pts $(date -Is)" | tee -a "$LOG"
  export DATA="$data"
  export INIT
  export MAX_ITER K
  export HDFS_IN="/user/bitnami/uk_kmeans_in_${tag}"
  export HDFS_OUT_BASE="/user/bitnami/uk_kmeans_out_${tag}"
  export WORKDIR

  taglog="$HOUSING/data/processed/scale_mr_${tag}.log"
  set +e
  bash "$WORKDIR/run_mr_kmeans.sh" >"$taglog" 2>&1
  rc=$?
  set -e
  wall=$(python - <<PY
import re
t=open("$taglog").read()
ms=re.findall(r"wall_seconds=([0-9.]+)", t)
print(ms[-1] if ms else "")
PY
)
  if [[ $rc -ne 0 || -z "$wall" ]]; then
    echo "MapReduce FAILED tag=$tag rc=$rc" | tee -a "$LOG"
    tail -n 30 "$taglog" | tee -a "$LOG" || true
    upsert_mr_csv "$tag" "$pts" "" "error"
    continue
  fi
  echo "MapReduce OK $tag wall=$wall" | tee -a "$LOG"
  upsert_mr_csv "$tag" "$pts" "$wall" "ok"
  # per-tag timing file
  cat >"$HOUSING/data/processed/scale_mr_${tag}_timing.txt" <<EOF
engine=MapReduce
tag=$tag
points=$pts
k=$K
iters=$MAX_ITER
wall_seconds=$wall
data=$base
EOF
done < "$MANIFEST"

echo "=== MapReduce scaling end $(date -Is) ===" | tee -a "$LOG"
echo "=== CSV ===" | tee -a "$LOG"
cat "$OUTCSV" | tee -a "$LOG"
