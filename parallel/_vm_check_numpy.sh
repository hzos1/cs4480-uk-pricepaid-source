#!/bin/bash
export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
echo "=== mount ==="
mountpoint -q /home/bitnami/uk_housing || echo bitnami | sudo -S mount -t vboxsf -o uid=1000,gid=1000 housing /home/bitnami/uk_housing
ls /home/bitnami/uk_housing/parallel/python_kmeans_uk.py
echo "=== python ==="
which python || true
which python3 || true
python -V 2>&1 || true
python3 -V 2>&1 || true
nproc
echo "=== numpy ==="
python -c 'import numpy; print(numpy.__version__)' 2>&1 || true
python3 -c 'import numpy; print(numpy.__version__)' 2>&1 || true
