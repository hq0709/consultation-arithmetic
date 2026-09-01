#!/usr/bin/env bash
# NMI 2026 对标网格：3 tier x 3 benchmark x 4 MAS 架构 x N{1,3,5,7,9} = 180 配置
# (NMI: 180 configurations across 5 architectures x 3 LLM families x benchmarks)
# 架构映射：independent=Independent · centralized=Centralized · discussion=Decentralized
#           tiered=Hybrid · zeroshot/cot=SAS · sc=SAS-repeated(等预算对照)
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
export LLM_MAX_INFLIGHT=18 LLM_WORKERS=8
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 8 --arches $ARCHES"

run () {  # run <tier_label> <model> <effort> <bench> <items>
  echo "=== $1 / $4 ==="
  python3 -u experiments/run_grid.py $COMMON --model "$2" --effort "$3" \
    --items "data/$5" --out "G_$1_$4.jsonl"
  python3 -c "from common.llm import global_spend_usd; print('  累计花费: \$%.2f'%global_spend_usd())"
}

# T2 先跑（MedXpertQA 已有大量缓存，最快见到完整 180 网格的第一个切面）
run T2 gpt-5-nano low medxpertqa   medxpertqa_500.jsonl
run T2 gpt-5-nano low medqa        medqa_500.jsonl
run T2 gpt-5-nano low medagentsbench medagentsbench_500.jsonl

# T1 最便宜
run T1 gpt-4.1-nano none medxpertqa   medxpertqa_500.jsonl
run T1 gpt-4.1-nano none medqa        medqa_500.jsonl
run T1 gpt-4.1-nano none medagentsbench medagentsbench_500.jsonl

# 用户批准的混淆对照：通用内科医生（排除"相关性衰减"解释）
echo "=== 对照：generic roles, T2/MedXpertQA ==="
python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1,3,5,7,9 --seeds 1 --workers 8 \
  --arches independent,centralized,discussion --generic-roles \
  --model gpt-5-nano --effort low --items data/medxpertqa_500.jsonl --out CTRL_T2_generic.jsonl

# T3 最贵，放最后
run T3 gpt-5-mini low medxpertqa   medxpertqa_500.jsonl
run T3 gpt-5-mini low medqa        medqa_500.jsonl
run T3 gpt-5-mini low medagentsbench medagentsbench_500.jsonl

echo "=== NMI 网格完成 ==="
