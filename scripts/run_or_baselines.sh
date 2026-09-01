#!/usr/bin/env bash
# 三大厂商之外的模型只做最基本的实验：单医生 CoT + 零样本基线。
# 这两条正是跨厂商误差相关性 phi 的原料（phi_decomposition 的 collect_solo
# 只取 arch=cot / N=1），每模型 1500 个 episode，而非全网格的 20000。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
# 并发按 provider 的承受力定，不是越高越好：qwen3.8-flash 在 24 并发下
# p95 延迟 349s、23/250 因 429 耗尽重试而失败。
workers_for () {
  case "$1" in
    qwen3.8-flash) echo 3 ;;
    qwen3.8-max)   echo 8 ;;
    *)             echo 24 ;;
  esac
}
for M in deepseek-v4-flash deepseek-v4-pro qwen3.8-flash qwen3.8-max glm-5.3 glm-5.3-flash; do
  W=$(workers_for "$M")
  (
    export LLM_MAX_INFLIGHT=$W LLM_WORKERS=$W
    for B in medxpertqa medagentsbench medqa; do
      python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1 --seeds 1 --workers "$W" \
        --arches cot,zeroshot --model "$M" --items "data/${B}_250.jsonl" --limit 250 \
        --out "OR_${M}_${B}.jsonl"
    done
    echo "=== $M 基线完成 ==="
  ) > "logs/or_cot_${M}.log" 2>&1 &
done
wait
echo "OR BASELINES DONE"
