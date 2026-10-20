#!/usr/bin/env bash
# N = 9 variant of run_imaging_nmi.sh (NS_MAS, default 3,9). NMI-style imaging grid (2026-10-20): the five canonical architectures of Kim et al. (NMI 2026) --
# SAS (cot), Independent, Centralized, Decentralized (discussion), Hybrid (tiered) -- on VQA-RAD,
# SLAKE and MedXpertQA-MM with original images, across a within-family capability ladder:
#   OpenAI    gpt-5-nano < gpt-5-mini < gpt-5          (reasoning effort low throughout)
#   Google    gemini-3.5-flash-lite < gemini-3.7-flash  (Flash models only)
#   Anthropic claude-haiku-4-5 < claude-sonnet-5
#   Qwen      qwen3.8-flash < qwen3.8-max-0902
# All calls via OpenRouter. Agents' initial answers are cache-shared across architectures and N
# (panels/architectures.py), so Centralized and Hybrid add only orchestrator / generalist calls.
#   usage: scripts/run_imaging_nmi.sh <model> <phase>
#   phases: pilot (20 items of VQA-RAD and MedXpertQA-MM, every architecture, N = 9, PILOT_ file)
#           core  (SAS, Independent N = 1,3,9, Decentralized N = $NS_MAS)   -- new ladder models
#           coord (Centralized and Hybrid, N = $NS_MAS)                       -- every model
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENROUTER_MODELS="gpt-5-nano,gpt-5-mini,gpt-5,gemini-3.5-flash-lite,gemini-3.7-flash,claude-haiku-4-5-20251001,claude-sonnet-5,qwen3.8-flash,qwen3.8-max-0902,gpt-4.1-nano"
m=$1; ph=$2
E=""; case "$m" in gpt-5*) E="--effort low";; esac
W=${WORKERS:-8}
G="python3 experiments/run_grid.py --workers $W --model $m $E"
S3="vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250"
# panel sizes; NS_MAS=9 restricts the multi-round architectures to N = 9 (used for qwen3.8-max-0902, whose
# reasoning makes each call about 30x a qwen3.8-flash call); Independent keeps N = 1,3,9 (nested, free)
NS_MAS=${NS_MAS:-3,9}
mkdir -p logs/NMI
log() { echo "logs/NMI/${m}_$1.log"; }

case "$ph" in
  pilot)
    for s in vqarad_closed_250 mxqa_mm_rad_250; do
      $G --items data/$s.jsonl --limit 20 --arches cot,independent,discussion,centralized,tiered --Ns 9 \
         --image-mode real --out PILOT_IMG2_${m}_$s.jsonl >> $(log pilot) 2>&1 &
    done; wait ;;
  core)
    for s in $S3; do
      ( $G --items data/$s.jsonl --arches cot,independent --Ns 1,3,9 --image-mode real --out IMG2_${m}_$s.jsonl
        $G --items data/$s.jsonl --arches discussion --Ns $NS_MAS --image-mode real --out IMG2_${m}_$s.jsonl ) >> $(log $s) 2>&1 &
    done; wait ;;
  coord)
    for s in $S3; do
      $G --items data/$s.jsonl --arches centralized,tiered --Ns $NS_MAS --image-mode real \
         --out IMG2_${m}_$s.jsonl >> $(log ${s}_coord) 2>&1 &
    done; wait ;;
  *) echo "unknown phase $ph"; exit 2 ;;
esac
echo "[$(date '+%m-%d %H:%M')] $m $ph done"
