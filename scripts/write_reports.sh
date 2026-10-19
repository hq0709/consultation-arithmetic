#!/usr/bin/env bash
# Experiment B perceiver pass: every findings report written once, before any reasoner reads it
# (docs/IMAGING_PROTOCOL.md section 12). Same routing as scripts/run_imaging_v2_full.sh, so the
# cache keys match the B runs.
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENROUTER_MODELS="gpt-5-mini,gemini-3.7-flash,claude-haiku-4-5-20251001,qwen3.8-flash,gpt-4.1-nano"
mkdir -p logs/IMG2
for p in gpt-5-mini gemini-3.7-flash claude-haiku-4-5-20251001; do
  LLM_MAX_INFLIGHT=32 python3 experiments/write_reports.py --perceiver $p --workers 24 > logs/IMG2/reports_$p.log 2>&1 &
done
wait
