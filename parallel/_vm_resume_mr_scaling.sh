#!/bin/bash
set -euo pipefail
source /home/bitnami/uk_housing/parallel/env.sh
export SKIP_TAGS='n5e5 n2p6e7'
# recover n5e5 into CSV via the fixed upsert by sourcing functions... call python2 upsert inline
python - <<'PY'
import csv, os
path = "/home/bitnami/uk_housing/data/processed/scaling_python_vs_spark.csv"
fieldnames = ["engine", "tag", "points", "iters", "workers", "wall_seconds", "status"]
rows = list(csv.DictReader(open(path, "rb")))
kept = [r for r in rows if not (r.get("engine") == "mapreduce" and r.get("tag") == "n5e5")]
kept.append({"engine":"mapreduce","tag":"n5e5","points":"500000","iters":"5","workers":"4","wall_seconds":"312.653","status":"ok"})
def key(r):
    eng_order = {"python": 0, "spark": 1, "mapreduce": 2}.get(r.get("engine", ""), 9)
    try: pts = int(float(r.get("points") or 0))
    except ValueError: pts = 0
    return (pts, eng_order, r.get("tag", ""))
kept.sort(key=key)
f = open(path, "wb")
w = csv.DictWriter(f, fieldnames=fieldnames)
w.writeheader()
for r in kept:
    w.writerow(dict((k, r.get(k, "")) for k in fieldnames))
f.close()
open("/home/bitnami/uk_housing/data/processed/scale_mr_n5e5_timing.txt","w").write(
"engine=MapReduce\ntag=n5e5\npoints=500000\nk=4\niters=5\nwall_seconds=312.653\ndata=uk_features_scaled_scale_n5e5.txt\n")
print("recovered n5e5")
PY
nohup env SKIP_TAGS='n5e5 n2p6e7' bash /home/bitnami/uk_housing/parallel/_vm_run_mr_scaling.sh > /home/bitnami/uk_housing/data/processed/mr_scaling_wrapper.log 2>&1 &
echo PID=$!
sleep 6
grep -E '>>>|OK|SKIP|FAIL|start|ImportError|Traceback' /home/bitnami/uk_housing/data/processed/mr_scaling_nohup.log | tail -20
echo '---CSV mr---'
grep mapreduce /home/bitnami/uk_housing/data/processed/scaling_python_vs_spark.csv