#!/usr/bin/env bash
# Protocol v2 full runs (docs/IMAGING_PROTOCOL.md sections 6 and 11). All calls via OpenRouter.
# Final data: results/IMG2_<model>_<set>.jsonl. Every run_grid call resumes, so re-running the
# script after an interruption only does what is missing.
#   usage: scripts/run_imaging_v2_full.sh <model> [phase ...]
#   phases (default all): A0 A1 C1 C2 A2 A3 C4 B
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENROUTER_MODELS="gpt-5-mini,gemini-3.7-flash,claude-haiku-4-5-20251001,qwen3.8-flash,gpt-4.1-nano"
m=$1; shift
phases=${*:-A0 A1 C1 C2 A2 A3 C4 B}
E=""; [ "$m" = "gpt-5-mini" ] && E="--effort low"
W=${WORKERS:-8}
G="python3 experiments/run_grid.py --workers $W --model $m $E"
RENDER="${RENDER:-,,,gamma0.7,gamma0.7,gamma0.7,clahe,clahe,clahe}"   # V2: inversion failed -> gamma 0.7
S13="vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250"
mkdir -p logs/IMG2
log() { echo "logs/IMG2/${m}_$1.log"; }

phase_A0() { for s in $S13; do $G --items data/$s.jsonl --arches cot,independent,discussion --Ns 1,3,9 --image-mode real --out IMG2_${m}_$s.jsonl >> $(log $s) 2>&1 & done; wait; }
phase_A1() { for s in $S13; do ( for mode in blank res256 res128; do $G --items data/$s.jsonl --arches cot,independent --Ns 1,3,9 --image-mode $mode --out IMG2_${m}_$s.jsonl; done ) >> $(log $s) 2>&1 & done; wait; }
phase_C1() { for s in $S13; do $G --items data/$s.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --out IMG2_${m}_$s.jsonl >> $(log $s) 2>&1 & done; wait; }
phase_C2() { for s in $S13; do $G --items data/$s.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --render "$RENDER" --out IMG2_${m}_$s.jsonl >> $(log $s) 2>&1 & done; wait; }
phase_A2() { for mode in real occ_target occ_control; do $G --items data/slake_target_250.jsonl --arches cot,independent --Ns 1,3,9 --image-mode $mode --out IMG2_${m}_slake_target_250.jsonl >> $(log slake_target) 2>&1; done; }
phase_A3() { $G --items data/slake_pairs_200.jsonl --arches cot,independent --Ns 1,3,9 --image-mode real --out IMG2_${m}_slake_pairs_200.jsonl >> $(log slake_pairs) 2>&1; }
phase_C4() {
  for s in vqarad_closed_250 slake_closed_250; do
    ( for mem in shared-image private-image shared-text private-text; do
        $G --items data/${s}_mem.jsonl --arches independent --Ns 9 --generic-roles --image-mode real --memory $mem --out IMG2_${m}_${s}_mem.jsonl
      done
      for mem in shared-image shared-text; do
        $G --items data/${s}_mem.jsonl --arches cot --image-mode real --memory $mem --out IMG2_${m}_${s}_mem.jsonl
      done ) >> $(log ${s}_mem) 2>&1 &
  done; wait; }
phase_B() {   # reasoner = $m; perceivers = the three B-arm vendors; shared report only when perceiver = reasoner
  for s in $S13; do
    ( for p in gpt-5-mini gemini-3.7-flash claude-haiku-4-5-20251001; do
        $G --items data/$s.jsonl --arches report --Ns 9 --perceiver $p --image-mode real --out IMG2_${m}_${s}_report.jsonl
        [ "$p" = "$m" ] && $G --items data/$s.jsonl --arches report --Ns 9 --perceiver $p --shared-report --image-mode real --out IMG2_${m}_${s}_report.jsonl
      done ) >> $(log ${s}_report) 2>&1 &
  done; wait; }

for ph in $phases; do
  echo "[$(date '+%m-%d %H:%M')] $m phase $ph start"
  phase_$ph
  echo "[$(date '+%m-%d %H:%M')] $m phase $ph done"
done
