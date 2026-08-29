#!/usr/bin/env bash
# 停掉所有 runner，切到 250 题 + 基线优先，全部并行重启。
cd /home/myid/hj67104/consultation_saturation
ps -eo pid,cmd | grep "[r]un_grid.py" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done
ps -eo pid,cmd | grep "[w]atch_grid.sh" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done
sleep 4
for b in medxpertqa medqa medagentsbench; do head -250 "data/${b}_500.jsonl" > "data/${b}_250.jsonl"; done
for spec in "T1 gpt-4.1-nano none" "T2 gpt-5-nano low" "T3 gpt-5-mini low"; do
  set -- $spec
  for b in medxpertqa medqa; do
    nohup bash scripts/run_one.sh "$1" "$2" "$3" "$b" 250 > "logs/r_$1_$b.log" 2>&1 &
  done
done
nohup bash scripts/run_one.sh T1 gpt-4.1-nano none medagentsbench 250 > logs/r_T1_medagentsbench.log 2>&1 &
sleep 12
echo "runner 进程数: $(ps -eo cmd | grep -c '[r]un_grid.py')"
