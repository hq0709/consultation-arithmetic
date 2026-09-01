#!/usr/bin/env bash
# Sonnet 5 段。haiku 已 100% 完成，opus 因预算不启动。
# 按 benchmark 串行：触到 ANTHROPIC_CAP_USD 时留下的是完整 benchmark 而非半截 cell。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
export LLM_MAX_INFLIGHT=16 LLM_WORKERS=16
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 16 --arches $ARCHES"
for B in medxpertqa medqa medagentsbench; do
  echo "=== claude-sonnet-5 / $B ==="
  python3 -u experiments/run_grid.py $COMMON --model claude-sonnet-5 \
    --items "data/${B}_250.jsonl" --limit 250 --out "CLA_claude-sonnet-5_${B}.jsonl" || {
      echo "!! $B 非零退出（多半是预算上限），停止后续 benchmark"; break; }
  python3 -c "from common.llm import global_spend_usd; print('  累计: \$%.2f'%global_spend_usd())"
done
echo "SONNET GRID DONE"
