#!/usr/bin/env bash
# Protocol v2 validation runs V1, V2, V3, V5 (docs/IMAGING_PROTOCOL.md section 8). All calls via OpenRouter.
# Output: results/IMG2V_<model>_<set>.jsonl (validation only; never pooled with the final IMG2_ files).
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENROUTER_MODELS="gpt-5-mini,gemini-3.7-flash,claude-haiku-4-5-20251001,qwen3.8-flash,gpt-4.1-nano"
mkdir -p logs
G="python3 experiments/run_grid.py --workers 8"
eff() { [ "$1" = "gpt-5-mini" ] && echo "--effort low" || true; }
val_model() {
  local m=$1 e; e=$(eff $m)
  for s in vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250; do
    out="IMG2V_${m}_${s}.jsonl"
    $G --items data/$s.jsonl --limit 50 --model $m $e --arches cot --image-mode real --out $out          # V1/V2 reference
    $G --items data/$s.jsonl --limit 30 --model $m $e --arches independent --Ns 3 --image-mode real --out $out
    for mode in res256 res128 inv clahe; do                                                               # V2
      $G --items data/$s.jsonl --limit 50 --model $m $e --arches cot --image-mode $mode --out $out
    done
  done
  for s in vqarad_closed_250 slake_closed_250; do                                                         # V3
    for mem in shared-image shared-text; do
      $G --items data/${s}_mem.jsonl --limit 50 --model $m $e --arches cot --image-mode real --memory $mem --out IMG2V_${m}_${s}_mem.jsonl
    done
  done
}
for m in gpt-5-mini gemini-3.7-flash claude-haiku-4-5-20251001 qwen3.8-flash; do
  val_model $m > logs/IMG2V_${m}.log 2>&1 &
done
wait
for p in gpt-5-mini gemini-3.7-flash claude-haiku-4-5-20251001; do                                         # V5
  e=$(eff $p)
  $G --items data/slake_closed_250.jsonl --limit 20 --model $p $e --arches report --Ns 3 --perceiver $p \
     --image-mode real --out IMG2V_${p}_report.jsonl >> logs/IMG2V_report.log 2>&1
done
echo "v2 validation done"
