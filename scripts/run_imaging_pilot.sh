#!/usr/bin/env bash
# Imaging pilot: two VLMs x two radiology benchmarks, 250 items each.
#   real image : single reader (cot), Independent and Decentralized at N = 3, 9
#   blank image: single reader and Independent at N = 3, 9 (same-size grey image; prompt unchanged)
# Panels at N = 3 are prefixes of N = 9 (nested round-0 calls), so they cost nothing extra.
# Output: results/IMG_<model>_<bench>.jsonl ; logs/IMG_<model>_<bench>.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
run() {   # $1 model, $2 extra flags, $3 bench
  local m=$1 extra=$2 b=$3 out="IMG_${1}_${3}.jsonl" log="logs/IMG_${1}_${3}.log"
  {
    python3 experiments/run_grid.py --items data/$b.jsonl --model $m $extra \
      --arches cot,independent,discussion --Ns 3,9 --image-mode real --workers 8 --out $out
    python3 experiments/run_grid.py --items data/$b.jsonl --model $m $extra \
      --arches cot,independent --Ns 3,9 --image-mode blank --workers 8 --out $out
  } >"$log" 2>&1
}
for b in vqarad_closed_250 mxqa_mm_rad_250; do
  run gpt-5-mini "--effort low" $b &
  run gemini-3.7-flash "" $b &
done
wait
echo "imaging pilot done"
