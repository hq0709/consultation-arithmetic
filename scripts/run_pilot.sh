#!/usr/bin/env bash
# Plan §5 mandatory pilot: 200 MedXpertQA x {N=1,5,9} x architectures {(1),(2)} x T2 x 1 seed,
# plus the §3.3 baselines needed to interpret it (zero-shot, CoT, budget-matched SC).
set -euo pipefail
cd /home/myid/hj67104/consultation_saturation
LLM_MAX_INFLIGHT=18 LLM_WORKERS=8 python3 -u experiments/run_grid.py \
  --items data/medxpertqa_pilot200.jsonl \
  --model gpt-5-nano --effort low --temp 0.7 --theta 80 \
  --arches independent,discussion,zeroshot,cot,sc \
  --Ns 1,5,9 --sc-ks 1,3,5,9,15,27 --seeds 1 \
  --workers 10 --out pilot_mxq_T2.jsonl
