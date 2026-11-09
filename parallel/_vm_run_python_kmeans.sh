#!/bin/bash
set -e
export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
export HOUSING=/home/bitnami/uk_housing
mountpoint -q "$HOUSING" || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing "$HOUSING"

# Ubuntu 14.04 Bitnami image: prefer apt numpy for Python 2.7
if ! python -c 'import numpy' 2>/dev/null; then
  echo "Installing python-numpy via apt..."
  echo bitnami | sudo -S apt-get update -qq
  echo bitnami | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python-numpy
fi
python -c 'import numpy; print("numpy", numpy.__version__)'

LOG="$HOUSING/data/processed/python_vm_nohup.log"
OUT="$HOUSING/data/processed/python_parallel_timing_vm.txt"
python "$HOUSING/parallel/python_kmeans_uk.py" \
  --data "$HOUSING/data/parallel/uk_features_scaled_2018plus_sample10pct.txt" \
  --centroids "$HOUSING/data/parallel/init_centroids_k4.txt" \
  --workers 4 \
  --out "$OUT" | tee "$LOG"
echo "DONE"
