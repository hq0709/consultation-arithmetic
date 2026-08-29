#!/usr/bin/env bash
# 单个 tier 的全部 benchmark。不同模型的 rate limit 相互独立 -> 三个 tier 可并行。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
TIER="$1"; MODEL="$2"; EFFORT="$3"
export LLM_MAX_INFLIGHT=18 LLM_WORKERS=8
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
for B in medxpertqa medqa medagentsbench; do
  echo "=== $TIER / $B ==="
  python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1,3,5,7,9 \
    --sc-ks 1,3,5,9,15 --seeds 1 --workers 8 --arches "$ARCHES" \
    --model "$MODEL" --effort "$EFFORT" --items "data/${B}_500.jsonl" --out "G_${TIER}_${B}.jsonl"
done
echo "=== $TIER 完成 ==="
