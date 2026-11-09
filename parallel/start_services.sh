#!/bin/bash
export PATH="/bin:/usr/bin:/usr/local/bin:/sbin:/usr/sbin"
source /home/bitnami/uk_housing/parallel/env.sh
echo "Mount check:"
ls -la /home/bitnami/uk_housing/data/parallel | head
echo "Starting DFS/YARN if needed..."
if ! jps | grep -q NameNode; then start-dfs.sh; fi
if ! jps | grep -q ResourceManager; then start-yarn.sh; fi
sleep 5
jps
hdfs dfs -ls / 2>&1 | head -n 20
