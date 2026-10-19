#!/usr/bin/env bash
# Qwen C1 and C2 in parallel with its A1 phase, into separate files (docs/IMAGING_PROTOCOL.md section 12).
cd /home/myid/hj67104/consultation_saturation
export OPENROUTER_MODELS="gpt-5-mini,gemini-3.7-flash,claude-haiku-4-5-20251001,qwen3.8-flash,gpt-4.1-nano"
export LLM_MAX_INFLIGHT=48
G="python3 experiments/run_grid.py --workers 24 --model qwen3.8-flash"
RENDER=",,,gamma0.7,gamma0.7,gamma0.7,clahe,clahe,clahe"
for s in vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250; do
  ( echo "[$(date '+%m-%d %H:%M')] qwen C1/C2 $s start"
    $G --items data/$s.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --out IMG2_qwen3.8-flash_${s}_c.jsonl
    $G --items data/$s.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --render "$RENDER" --out IMG2_qwen3.8-flash_${s}_c.jsonl
    echo "[$(date '+%m-%d %H:%M')] qwen C1/C2 $s done" ) >> logs/IMG2/qwen3.8-flash_${s}_c.log 2>&1 &
done
wait
