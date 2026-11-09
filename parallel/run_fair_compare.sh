#!/bin/sh
# Usage: sh parallel/run_fair_compare.sh python
#        sh parallel/run_fair_compare.sh spark
#        sh parallel/run_fair_compare.sh mr
#
# Same row sets, same k, same initial centres, same epsilon on every device.
# Each timing file records that device's CPU and memory.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
ENGINE=${1:-}
case "$ENGINE" in
  python) exec sh "$ROOT/parallel/run_fair_python.sh" ;;
  spark) exec sh "$ROOT/parallel/run_fair_spark.sh" ;;
  mr|mapreduce) exec sh "$ROOT/parallel/run_fair_mr.sh" ;;
  *)
    echo "Usage: sh parallel/run_fair_compare.sh python|spark|mr" >&2
    exit 1
    ;;
esac
