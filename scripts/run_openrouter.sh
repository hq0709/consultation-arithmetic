#!/usr/bin/env bash
# 经 OpenRouter 的网格。opus-5 按用户指示不跑。
# 注意：OpenRouter 把思考 token 计入 max_tokens，所以这些模型走 _thinks() 的大预算。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env; set +a
export OPENROUTER_MODELS=gemini-3.7-flash
export LLM_MAX_INFLIGHT=8 LLM_WORKERS=8
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 8 --arches $ARCHES"
run () {
  echo "=== $1 / $2 ==="
  python3 -u experiments/run_grid.py $COMMON --model "$1" \
    --items "data/$2_250.jsonl" --limit 250 --out "OR_$1_$2.jsonl" || {
      echo "!! $1/$2 非零退出，停止"; return 1; }
}
# 1) 补完 gemini-3.7-flash 的 MedQA（整个 benchmark 走同一条路由）
run gemini-3.7-flash medqa
# 2) gemini-3.1-pro 三个 benchmark
# gemini-3.1-pro 按用户指示不跑（2026-09-01）。需要时解开下一行。
# for B in medxpertqa medagentsbench medqa; do run gemini-3.1-pro "$B" || break; done
echo "OPENROUTER GRID DONE"
