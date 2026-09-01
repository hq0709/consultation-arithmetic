#!/usr/bin/env bash
# 两阶段：先把六个模型的单医生 CoT 基线跑出来，再补全网格。
#
# 跨厂商多样性分析（phi_decomposition 的 collect_solo）只用 arch=cot / N=1 的逐题正误，
# 每个模型 750 次调用就够；全网格是 20000 次。先出 CoT 意味着论文的核心结论
# 几小时内可得，而不是等几十小时。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
export LLM_MAX_INFLIGHT=24 LLM_WORKERS=24
MODELS="deepseek-v4-flash deepseek-v4-pro qwen3.8-flash qwen3.8-max glm-5.3 glm-5.3-flash"

echo "########## 阶段一：CoT 基线 ##########"
for M in $MODELS; do
  (
    for B in medxpertqa medagentsbench medqa; do
      python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1 --seeds 1 --workers 24 \
        --arches cot,zeroshot --model "$M" --items "data/${B}_250.jsonl" --limit 250 \
        --out "OR_${M}_${B}.jsonl"
    done
    echo "=== $M CoT 完成 ==="
  ) > "logs/or_cot_${M}.log" 2>&1 &
done
wait
echo "########## 阶段一完成 ##########"

# 阶段二（全网格）按用户指示不跑：三大厂商之外只做最基本的 CoT/零样本基线。
# echo "########## 阶段二：全网格 ##########"
# ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
# COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 24 --arches $ARCHES"
# for M in $MODELS; do
#   (
#     for B in medxpertqa medagentsbench medqa; do
#       python3 -u experiments/run_grid.py $COMMON --model "$M" \
#         --items "data/${B}_250.jsonl" --limit 250 --out "OR_${M}_${B}.jsonl" || break
#     done
#     echo "=== $M 全网格完成 ==="
#   ) > "logs/or_${M}.log" 2>&1 &
# done
# wait
# echo "OR MODELS GRID DONE"
echo "OR COT DONE"
