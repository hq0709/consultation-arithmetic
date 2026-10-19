#!/usr/bin/env bash
# Experiment D full runs (docs/IMAGING_PROTOCOL.md sections 6, 11): five local VLMs, one GPU each.
# Output results/OW_<model>_<set>.jsonl (resumable). Renderings follow the API arm (gamma0.7 for inversion).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/ow
run() {   # $1 model, $2 gpu
  export CUDA_VISIBLE_DEVICES=$2
  for s in vqarad_closed_250 slake_closed_250 mxqa_mm_rad_250; do
    python experiments/openweight_probe.py --model $1 --items data/$s.jsonl --conditions real,blank,res256,res128,gamma0.7,clahe --out results/OW_$1_$s.jsonl
  done
  python experiments/openweight_probe.py --model $1 --items data/slake_target_250.jsonl --conditions real,blank,occ_target,occ_control --out results/OW_$1_slake_target_250.jsonl
  python experiments/openweight_probe.py --model $1 --items data/slake_pairs_200.jsonl --conditions real --out results/OW_$1_slake_pairs_200.jsonl
}
run medgemma-4b 2 > logs/ow/full_medgemma-4b.log 2>&1 &
run huatuogpt-vision-7b 3 > logs/ow/full_huatuogpt-vision-7b.log 2>&1 &
run qwen3-vl-8b 4 > logs/ow/full_qwen3-vl-8b.log 2>&1 &
run qwen2-vl-7b 5 > logs/ow/full_qwen2-vl-7b.log 2>&1 &
run llava-1.5-7b 6 > logs/ow/full_llava-1.5-7b.log 2>&1 &
wait
echo "open-weight full runs done"
