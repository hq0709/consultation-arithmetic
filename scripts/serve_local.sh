#!/usr/bin/env bash
# 起 vLLM 的 OpenAI 兼容端点，供开源权重模型臂使用。
#
#   bash scripts/serve_local.sh medgemma-4b 8010 0
#   bash scripts/serve_local.sh lingshu-7b  8011 1,2
#
# 参数：<注册名> <端口> <CUDA_VISIBLE_DEVICES>
# 注册名必须与 .env 里 LOCAL_MODELS 的键一致 —— 代码用 local/<注册名> 调用它。
set -euo pipefail

NAME="${1:?用法: serve_local.sh <name> <port> <gpus>}"
PORT="${2:?}"
GPUS="${3:-0}"

# 注册名 -> HuggingFace 仓库
case "$NAME" in
  medgemma-4b)  REPO="google/medgemma-4b-it" ;;
  medgemma-27b) REPO="google/medgemma-27b-text-it" ;;
  lingshu-7b)   REPO="lingshu-medical-mllm/Lingshu-7B" ;;
  lingshu-32b)  REPO="lingshu-medical-mllm/Lingshu-32B" ;;
  llava-med)    REPO="microsoft/llava-med-v1.5-mistral-7b" ;;
  huatuo-7b)    REPO="FreedomIntelligence/HuatuoGPT-Vision-7B" ;;
  qwen2.5-7b)   REPO="Qwen/Qwen2.5-7B-Instruct" ;;
  *) echo "未知模型 '$NAME'，请在 serve_local.sh 的 case 里登记" >&2; exit 1 ;;
esac

NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

echo "启动 $NAME  ($REPO)  端口 $PORT  GPU $GPUS  TP=$NGPU"
CUDA_VISIBLE_DEVICES="$GPUS" python3 -m vllm.entrypoints.openai.api_server \
  --model "$REPO" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$NGPU" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --guided-decoding-backend outlines \
  --disable-log-requests
