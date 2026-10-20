#!/usr/bin/env bash
# Pause this project's run_grid processes while the shared disk is nearly full (other users' jobs fill it),
# so cache and result writes never fail mid-run; resume when space returns.
#   usage: nohup bash scripts/disk_watchdog.sh > logs/disk_watchdog.log 2>&1 &
LOW=${LOW_MB:-800}; HIGH=${HIGH_MB:-2000}
paused=0
while true; do
  free=$(df -Pm / | awk 'NR==2 {print $4}')
  if [ "$free" -lt "$LOW" ] || { [ $paused -eq 1 ] && [ "$free" -lt "$HIGH" ]; }; then
    pkill -STOP -f "^python3 experiments/run_grid.py"           # re-applied every pass: catches new runs
    [ $paused -eq 0 ] && echo "$(date '+%m-%d %H:%M:%S') paused at ${free} MB free"
    paused=1
  elif [ $paused -eq 1 ]; then
    pkill -CONT -f "^python3 experiments/run_grid.py"
    echo "$(date '+%m-%d %H:%M:%S') resumed at ${free} MB free"
    paused=0
  fi
  sleep 15
done
