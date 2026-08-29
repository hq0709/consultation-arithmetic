#!/usr/bin/env bash
# Scope C (user-selected): the six contrasts the power analysis says are resolvable,
# on 1000 MedXpertQA items, T2 + T3, plus the generic-internist confound control.
# Sequential by design: gpt-5-* tops out near 250 calls/min before 429s.
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
COMMON="--items data/medxpertqa_1000.jsonl --temp 0.7 --theta 80 --Ns 1,5,9 --workers 8"
export LLM_MAX_INFLIGHT=18 LLM_WORKERS=8

echo "=== [1/3] T2 gpt-5-nano, specialty roles, seeds 1-3 ==="
python3 -u experiments/run_grid.py $COMMON --model gpt-5-nano --effort low \
  --arches independent,discussion,zeroshot,cot,sc --sc-ks 1,3,5,9,15 --seeds 1,2,3 \
  --out C_T2_specialty.jsonl

echo "=== [2/3] T2 gpt-5-nano, GENERIC internists (relevance-decay control), seed 1 ==="
python3 -u experiments/run_grid.py $COMMON --model gpt-5-nano --effort low --generic-roles \
  --arches independent,discussion --seeds 1 \
  --out C_T2_generic.jsonl

echo "=== [3/3] T3 gpt-5-mini, specialty roles, seed 1 ==="
python3 -u experiments/run_grid.py $COMMON --model gpt-5-mini --effort low \
  --arches independent,discussion,zeroshot,cot,sc --sc-ks 1,3,5,9,15 --seeds 1 \
  --out C_T3_specialty.jsonl

echo "=== SCOPE C COMPLETE ==="
