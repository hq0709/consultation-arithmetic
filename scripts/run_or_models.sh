#!/usr/bin/env bash
# 六个新模型经 OpenRouter 的完整网格。
#
# 并行而非排队：它们分属 deepseek / qwen / z-ai 三家不同 provider，限流互相独立，
# 串行跑等于白白浪费另外两家的带宽。每个模型一个进程、一个输出文件，
# run_grid.py 的 flock 是按文件加的，互不干扰。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
export LLM_MAX_INFLIGHT=6 LLM_WORKERS=6
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 6 --arches $ARCHES"

one_model () {
  local M=$1
  for B in medxpertqa medagentsbench medqa; do
    echo "=== $M / $B ==="
    python3 -u experiments/run_grid.py $COMMON --model "$M" \
      --items "data/${B}_250.jsonl" --limit 250 --out "OR_${M}_${B}.jsonl" || {
        echo "!! $M/$B 非零退出，跳过该模型剩余 benchmark"; return 1; }
  done
  echo "=== $M 全部完成 ==="
}

for M in deepseek-v4-flash glm-5.3-flash qwen3.8-flash \
         deepseek-v4-pro qwen3.8-max glm-5.3; do
  one_model "$M" > "logs/or_${M}.log" 2>&1 &
  echo "启动 $M (PID $!)"
  sleep 2
done
wait
echo "OR MODELS GRID DONE"
