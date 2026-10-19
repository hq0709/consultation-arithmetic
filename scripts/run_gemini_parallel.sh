#!/usr/bin/env bash
# Gemini's remaining C4 cells and its B phase in parallel with the running driver, each into its own
# file (docs/IMAGING_PROTOCOL.md section 12). Same arguments and routing as scripts/run_imaging_v2_full.sh,
# so the cfg_hash and cache keys are the same; the driver's later calls are cache hits.
cd "$(dirname "$0")/.."
export OPENROUTER_MODELS="gpt-5-mini,gemini-3.7-flash,claude-haiku-4-5-20251001,qwen3.8-flash,gpt-4.1-nano"
m=gemini-3.7-flash
G="python3 experiments/run_grid.py --workers 8 --model $m"
for s in vqarad_closed_250 slake_closed_250; do
  ( for mem in private-image shared-text private-text; do
      $G --items data/${s}_mem.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --memory $mem --out IMG2_${m}_${s}_mem_p.jsonl
    done ) >> logs/IMG2/${m}_${s}_mem_p.log 2>&1 &
  ( for mem in shared-image shared-text; do
      $G --items data/${s}_mem.jsonl --arches cot --image-mode real --memory $mem --out IMG2_${m}_${s}_mem_s.jsonl
    done ) >> logs/IMG2/${m}_${s}_mem_s.log 2>&1 &
done
for s in vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250; do
  ( for p in gpt-5-mini gemini-3.7-flash claude-haiku-4-5-20251001; do
      $G --items data/$s.jsonl --arches report --Ns 9 --perceiver $p --image-mode real --out IMG2_${m}_${s}_report_p.jsonl
      [ "$p" = "$m" ] && $G --items data/$s.jsonl --arches report --Ns 9 --perceiver $p --shared-report --image-mode real --out IMG2_${m}_${s}_report_p.jsonl
    done ) >> logs/IMG2/${m}_${s}_report_p.log 2>&1 &
done
wait
echo "[$(date '+%m-%d %H:%M')] gemini parallel cells done"
