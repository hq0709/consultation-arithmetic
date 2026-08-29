#!/usr/bin/env bash
# 开源权重模型臂：在三个 benchmark 上跑单医生基线，用于扩展 Table 5 的多样性阶梯。
#
# 前置：
#   1) bash scripts/serve_local.sh <name> <port> <gpus>   （每个模型一个端点）
#   2) 在 .env 里设 LOCAL_MODELS=<name>=http://localhost:<port>/v1,...
#   3) python3 scripts/preflight.py --local                （必须全绿）
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-medgemma-4b lingshu-7b llava-med qwen2.5-7b}"
BENCHES="${BENCHES:-medxpertqa medagentsbench medqa}"
LIMIT="${LIMIT:-250}"

mkdir -p logs results
for M in $MODELS; do
  for B in $BENCHES; do
    OUT="OPEN_${M}_${B}.jsonl"
    if [ -s "results/$OUT" ]; then echo "跳过 $OUT（已存在）"; continue; fi
    echo "=== $M / $B ==="
    python3 experiments/run_grid.py \
      --items "data/${B}_${LIMIT}.jsonl" --limit "$LIMIT" \
      --model "local/$M" --arches cot --Ns 1 --seeds 1 --temp 0.3 \
      --workers 16 --out "$OUT" 2>&1 | tee -a logs/open_models.log
  done
done
echo "全部完成，跑分析："
echo "  python3 experiments/phi_decomposition.py"
