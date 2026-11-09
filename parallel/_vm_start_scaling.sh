#!/bin/bash
export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
HOUSING=/home/bitnami/uk_housing
mountpoint -q "$HOUSING" || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing "$HOUSING"
chmod +x "$HOUSING/parallel/run_scaling_benchmark.sh" || true
# Fix LF just in case
sed -i 's/\r$//' "$HOUSING/parallel/run_scaling_benchmark.sh" "$HOUSING/parallel/python_kmeans_uk.py" || true
export MAX_ITER=5
export WORKERS=4
export HOUSING
nohup bash "$HOUSING/parallel/run_scaling_benchmark.sh" > "$HOUSING/data/processed/scaling_nohup.log" 2>&1 &
echo PID=$!
sleep 2
head -n 30 "$HOUSING/data/processed/scaling_nohup.log" || true
