#!/usr/bin/env bash
# Run a command while sampling GPU and CPU utilisation once per second.
#   usage: ./trace_run.sh OUT_DIR command [args...]
# Writes OUT_DIR/gpu_trace.csv (nvidia-smi) and OUT_DIR/cpu_trace.csv (whole-machine busy %, from /proc/stat).
set -u
out=$1; shift
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$out/gpu_trace.csv" &
gpu_pid=$!
(
  echo "timestamp,cpu_busy_pct"
  read -r _ u n s i w q sq st _ < /proc/stat; prev_busy=$((u+n+s+q+sq+st)); prev_total=$((prev_busy+i+w))
  while sleep 1; do
    read -r _ u n s i w q sq st _ < /proc/stat; busy=$((u+n+s+q+sq+st)); total=$((busy+i+w))
    echo "$(date '+%F %T'),$(( 100 * (busy - prev_busy) / (total - prev_total) ))"
    prev_busy=$busy; prev_total=$total
  done
) > "$out/cpu_trace.csv" &
cpu_pid=$!
"$@"
status=$?
kill "$gpu_pid" "$cpu_pid" 2>/dev/null
exit $status
