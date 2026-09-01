#!/usr/bin/env bash
# Claude 全架构网格：与 OpenAI / Gemini 臂完全同参数。
# Claude 的兼容端点不支持 seed 与 n>1，也只认 json_schema —— common/llm.py 已分别处理。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
export LLM_MAX_INFLIGHT=16 LLM_WORKERS=16
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 16 --arches $ARCHES"
run () {
  echo "=== $1 / $2 ==="
  python3 -u experiments/run_grid.py $COMMON --model "$1" \
    --items "data/$2_250.jsonl" --limit 250 --out "CLA_$1_$2.jsonl"
  python3 -c "from common.llm import global_spend_usd; print('  累计: \$%.2f'%global_spend_usd())"
}
# 便宜的先跑：出问题时损失小
# 由便宜到贵，出问题时损失最小
for B in medxpertqa medagentsbench medqa; do run claude-haiku-4-5-20251001 "$B"; done
for B in medxpertqa medagentsbench medqa; do run claude-sonnet-5 "$B"; done
# opus 按用户指示不跑（2026-08-31）：单价是 haiku 的 5 倍，全网格约 $626。
# 需要时手动解开这一行。
# for B in medxpertqa medagentsbench medqa; do run claude-opus-5 "$B"; done
echo "CLAUDE GRID DONE"
python3 scripts/true_spend.py
