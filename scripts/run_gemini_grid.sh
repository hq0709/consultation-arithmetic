#!/usr/bin/env bash
# Gemini 网格：与 OpenAI 臂完全同参数，便于直接并列比较。
# 注意 Gemini 端点不支持 seed 与 n>1（common/llm.py 已分别丢弃/改顺序采样）。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
export LLM_MAX_INFLIGHT=12 LLM_WORKERS=8
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 8 --arches $ARCHES"

run () {
  echo "=== $1 / $2 ==="
  python3 -u experiments/run_grid.py $COMMON --model "$1" \
    --items "data/$2_250.jsonl" --limit 250 --out "GEM_$1_$2.jsonl"
  python3 -c "from common.llm import global_spend_usd; print('  累计: \$%.2f'%global_spend_usd())"
}

# 便宜的先跑：出问题时损失小，且能先验证整条链路
for B in medxpertqa medagentsbench medqa; do run gemini-3.5-flash-lite "$B"; done
for B in medxpertqa medagentsbench medqa; do run gemini-3.7-flash      "$B"; done
echo "GEMINI GRID DONE"
python3 scripts/true_spend.py
