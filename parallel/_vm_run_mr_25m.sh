#!/bin/bash
set -euo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:/usr/local/hadoop-2.6.0/bin:/usr/local/hadoop-2.6.0/sbin:/usr/local/jdk1.8.0_40/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export HADOOP_HOME="${HADOOP_HOME:-/usr/local/hadoop-2.6.0}"
export HOUSING=/home/bitnami/uk_housing

mountpoint -q "$HOUSING" || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing "$HOUSING"
# shellcheck disable=SC1091
source "$HOUSING/parallel/env.sh"

DATA="$HOUSING/data/parallel/uk_features_scaled_scale_n2p6e7.txt"
INIT="$HOUSING/data/parallel/init_centroids_k4.txt"
LOG="$HOUSING/data/processed/mr_25m_nohup.log"
OUT_TIMING="$HOUSING/data/processed/mr_25m_timing.txt"

ls -lh "$DATA"
if ! jps 2>/dev/null | grep -q NameNode; then start-dfs.sh; fi
if ! jps 2>/dev/null | grep -q ResourceManager; then start-yarn.sh; fi

echo "Waiting for NameNode to leave safe mode..."
for i in $(seq 1 60); do
  if hdfs dfsadmin -safemode get 2>/dev/null | grep -qi 'OFF'; then
    echo "Safe mode OFF"
    break
  fi
  # force leave after ~90s if cluster is healthy enough for course VM
  if [[ $i -ge 18 ]]; then
    echo "Forcing safemode leave"
    hdfs dfsadmin -safemode leave || true
  fi
  sleep 5
done
jps
hdfs dfsadmin -safemode get || true

export DATA INIT
export MAX_ITER=5
export K=4
export HDFS_IN=/user/bitnami/uk_kmeans_in_25m
export HDFS_OUT_BASE=/user/bitnami/uk_kmeans_out_25m
export WORKDIR="$HOUSING/parallel/mr_kmeans"

{
  echo "=== MapReduce 25M start $(date -Is) ==="
  echo "DATA=$DATA MAX_ITER=$MAX_ITER"
  bash "$HOUSING/parallel/mr_kmeans/run_mr_kmeans.sh"
  echo "=== MapReduce 25M end $(date -Is) ==="
} >"$LOG" 2>&1

python - <<'PY'
import re
log=open("/home/bitnami/uk_housing/data/processed/mr_25m_nohup.log").read()
m=re.findall(r"wall_seconds=([0-9.]+)", log)
iters=re.findall(r"^iters=([0-9]+)", log, re.M)
open("/home/bitnami/uk_housing/data/processed/mr_25m_timing.txt","w").write(
  "engine=MapReduce\npoints=25655817\nk=4\niters=%s\nwall_seconds=%s\ndata=uk_features_scaled_scale_n2p6e7.txt\n"
  % ((iters[-1] if iters else "5"), (m[-1] if m else ""))
)
print("wrote timing wall=", m[-1] if m else "MISSING")
PY
