#!/usr/bin/env bash
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
TIER="$1"; MODEL="$2"; EFFORT="$3"; B="$4"; ITEMS="${5:-250}"
export LLM_MAX_INFLIGHT=14 LLM_WORKERS=7
# 基线优先：fig3(45% 阈值) 与 fig4(协调指标) 都以 CoT 为分母，必须最先拿到
python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1,3,5,7,9 \
  --sc-ks 1,3,5,9,15 --seeds 1 --workers 7 \
  --arches cot,zeroshot,independent,centralized,discussion,tiered,sc \
  --model "$MODEL" --effort "$EFFORT" --items "data/${B}_${ITEMS}.jsonl" --out "G_${TIER}_${B}.jsonl"
