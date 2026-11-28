#!/usr/bin/env bash
# Log whole-GPU memory/utilisation and the 1-minute load average once per second.
# Usage: gpu_log.sh out.csv   (runs until killed)
# Note: under WSL2 nvidia-smi cannot attribute memory to processes, so memory.used is
# the whole GPU, including any other job's allocation.
out=$1
echo "ts,mem_used_mib,util_pct,load1" > "$out"
while true; do
  g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  l=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date +%s.%N),$g,$l" >> "$out"
  sleep 1
done
