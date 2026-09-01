#!/usr/bin/env bash
# DeepSeek 全网格。两个模型 -> fig1 里成为第四个厂商列，且 deepseek-v4-pro (I=59.7)
# 与 gpt-5-mini (59.2) 构成跨太平洋的能力匹配对照。
# 已跑完的 cot/zeroshot 基线会被跳过（run_grid 按 qid+cfg_hash 判重）。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
export LLM_MAX_INFLIGHT=24 LLM_WORKERS=24
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 24 --arches $ARCHES"
for M in deepseek-v4-flash deepseek-v4-pro; do
  (
    for B in medxpertqa medagentsbench medqa; do
      echo "=== $M / $B ==="
      python3 -u experiments/run_grid.py $COMMON --model "$M" \
        --items "data/${B}_250.jsonl" --limit 250 --out "OR_${M}_${B}.jsonl" || break
    done
    echo "=== $M 全网格完成 ==="
  ) > "logs/ds_${M}.log" 2>&1 &
  echo "启动 $M (PID $!)"
done
wait
echo "DEEPSEEK GRID DONE"
