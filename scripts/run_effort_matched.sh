#!/usr/bin/env bash
# 算力配平的单医生基线：给单个医生更高的推理预算，对标 Nature 那篇
# "SAS received proportionally more reasoning rounds to compensate for lack of
#  parallel deliberation"。只有推理模型有这个旋钮。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
export LLM_MAX_INFLIGHT=10 LLM_WORKERS=6
for M in gpt-5-nano gpt-5-mini; do
  for B in medxpertqa medagentsbench medqa; do
    echo "=== $M / $B (effort=high) ==="
    python3 -u experiments/run_grid.py --items "data/${B}_250.jsonl" --limit 250 \
      --model "$M" --effort high --arches cot --Ns 1 --seeds 1 --temp 0.3 \
      --workers 6 --out "EFF_${M}_${B}.jsonl"
  done
done
echo "EFFORT MATCHED DONE"
python3 scripts/true_spend.py
