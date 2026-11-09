#!/bin/bash
set -e
export PATH="/bin:/usr/bin:/usr/local/bin:/usr/local/spark-1.2.2-bin-hadoop2.4/bin:/usr/local/jdk1.8.0_40/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark-1.2.2-bin-hadoop2.4}"
HOUSING=/home/bitnami/uk_housing
CSV=$HOUSING/data/processed/scaling_python_vs_spark.csv

mountpoint -q "$HOUSING" || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing "$HOUSING"
ls "$HOUSING/data/parallel/uk_features_scaled_scale_n1p7e7.txt" >/dev/null

# Ensure Python OOM rows for sizes that cannot fork on 6GB VM
ensure_oom() {
  local tag="$1" pts="$2"
  if ! awk -F, -v t="$tag" '($1=="python" && $2==t && ($NF=="oom_or_error" || $NF=="ok")){f=1} END{exit !f}' "$CSV"; then
    grep -v "^python,${tag}," "$CSV" > "$CSV.tmp" || true
    mv "$CSV.tmp" "$CSV"
    echo "python,${tag},${pts},5,4,,oom_or_error" >> "$CSV"
    echo "marked python $tag oom_or_error"
  fi
}
ensure_oom n1p7e7 17103878
ensure_oom n2p6e7 25655817

pkill -9 -f python_kmeans_uk.py 2>/dev/null || true
pkill -9 -f run_scaling_resume 2>/dev/null || true
pkill -9 -f spark-shell 2>/dev/null || true
sleep 2

chmod +x "$HOUSING/parallel/run_scaling_resume.sh"
sed -i 's/\r$//' "$HOUSING/parallel/run_scaling_resume.sh" || true

nohup bash "$HOUSING/parallel/run_scaling_resume.sh" > "$HOUSING/data/processed/scaling_resume_nohup.log" 2>&1 &
echo PID=$!
sleep 8
echo '=== CSV ==='
cat "$CSV"
echo '=== LOG ==='
tail -n 25 "$HOUSING/data/processed/scaling_resume_nohup.log"
