#!/bin/bash
set -e
HOUSING=/home/bitnami/uk_housing
CSV=$HOUSING/data/processed/scaling_python_vs_spark.csv
pkill -9 -f python_kmeans_uk.py 2>/dev/null || true
pkill -9 -f run_scaling_resume 2>/dev/null || true
pkill -9 -f spark-shell 2>/dev/null || true
sleep 2
grep -v '^python,n1p7e7,' "$CSV" > "$CSV.tmp" || true
mv "$CSV.tmp" "$CSV"
echo 'python,n1p7e7,17103878,5,4,,oom_or_error' >> "$CSV"
# Also pre-mark 25M python as likely oom path — still try in resume
echo '=== CSV now ==='
cat "$CSV"
nohup bash "$HOUSING/parallel/run_scaling_resume.sh" > "$HOUSING/data/processed/scaling_resume_nohup.log" 2>&1 &
echo PID=$!
sleep 8
tail -n 25 "$HOUSING/data/processed/scaling_resume_nohup.log"
