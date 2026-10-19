#!/usr/bin/env bash
# Resume the two gemini blank-image Independent cells cut off by the direct API's daily quota
# (2026-10-19). Same route, same budget as the rest of the cell; run_grid retries only the
# errored episodes.
cd "$(dirname "$0")/.."
for b in vqarad_closed_250 mxqa_mm_rad_250; do
  python3 experiments/run_grid.py --items data/$b.jsonl --model gemini-3.7-flash \
    --arches cot,independent --Ns 3,9 --image-mode blank --workers 6 \
    --out IMG_gemini-3.7-flash_$b.jsonl >> logs/IMG_gemini-3.7-flash_${b}_resume.log 2>&1
done
echo "gemini blank resume done"
