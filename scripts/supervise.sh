#!/usr/bin/env bash
# 无人值守整夜运行的看护进程。
#
# 原来的 watch_runs.sh 只读日志内容，有三类故障它看不见：
#   1. 进程被 OOM / 异常退出 —— 日志停在最后一行，看起来和"正在跑"一模一样
#   2. 进程活着但卡死 —— 长时间没有新的 API 调用
#   3. 磁盘写满 —— 今晚已经发生过一次，会静默写出截断的 JSON
# 这里逐条检查，并且对第 1 类直接自动拉起（run_grid.py 会跳过已完成的 episode，
# 重启是幂等的；输出文件有 flock，重复启动也不会写坏）。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env 2>/dev/null; set +a

MODELS="deepseek-v4-flash deepseek-v4-pro qwen3.8-flash qwen3.8-max glm-5.3 glm-5.3-flash"
declare -A RESTARTS SEEN
STALL_MIN=25          # 超过这么久没有新调用就算卡死
MAX_RESTART=5         # 单个模型最多自动拉起几次，避免无限重启风暴

alive () { pgrep -f -- "--model $1 " >/dev/null 2>&1; }

done_for () {   # 该模型三个 benchmark 的基线是否都已齐全（1500 个有效 episode）
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import sys, glob, json
m=sys.argv[1]; n=0
for f in glob.glob(f"results/OR_{m}_*.jsonl"):
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        if r.get("status")=="ok": n+=1
print(1 if n>=1490 else 0)
PY
}

# 每个模型的并发。qwen3.8-flash 在 OpenRouter 上只有阿里云一个 provider 且限流很紧：
# 24 并发时 p95 延迟 349s、23/250 因 429 耗尽重试而失败 —— 并发越高越触发限流，
# 退避越久，实际吞吐反而更低。低并发稳跑才是对的。
workers_for () {
  case "$1" in
    qwen3.8-flash) echo 3 ;;
    qwen3.8-max)   echo 8 ;;
    *)             echo 24 ;;
  esac
}

restart () {
  local M=$1
  local W; W=$(workers_for "$M")
  export LLM_MAX_INFLIGHT=$W LLM_WORKERS=$W
  (
    for B in medxpertqa medagentsbench medqa; do
      python3 -u experiments/run_grid.py --temp 0.7 --theta 80 --Ns 1 --seeds 1 --workers "$W" \
        --arches cot,zeroshot --model "$M" --items "data/${B}_250.jsonl" --limit 250 \
        --out "OR_${M}_${B}.jsonl"
    done
    echo "=== $M 基线完成 ==="
  ) >> "logs/or_cot_${M}.log" 2>&1 &
}

while true; do
  # ---- 磁盘 ----
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${FREE:-999}" -lt 5 ] && [ -z "${SEEN[disk]:-}" ]; then
    SEEN[disk]=1
    echo "[严重] 磁盘仅剩 ${FREE}G —— 实验可能写出截断数据，需要人工清理"
  fi

  # ---- 逐模型：存活 / 完成 / 卡死 ----
  for M in $MODELS; do
    if [ "$(done_for "$M")" = "1" ]; then
      if [ -z "${SEEN[done$M]:-}" ]; then SEEN[done$M]=1; echo "[完成] $M 基线已齐 (1500 episodes)"; fi
      continue
    fi
    if ! alive "$M"; then
      n=${RESTARTS[$M]:-0}
      if [ "$n" -lt "$MAX_RESTART" ]; then
        RESTARTS[$M]=$((n+1))
        echo "[自动拉起] $M 进程已不在且数据未跑完，第 $((n+1)) 次重启"
        restart "$M"
      elif [ -z "${SEEN[give$M]:-}" ]; then
        SEEN[give$M]=1
        echo "[放弃] $M 已重启 $MAX_RESTART 次仍失败，需要人工介入"
      fi
      continue
    fi
    # 卡死：进程在，但这个模型很久没有新的 API 调用
    LAST=$(python3 - "$M" <<'PY' 2>/dev/null || echo 0
import sys, glob, json, time
m=sys.argv[1]; last=0
fs=sorted(glob.glob("logs/calls_*.jsonl"))
for f in fs[-2:]:
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        if r.get("model")==m: last=max(last, r.get("ts",0))
print(int((time.time()-last)/60) if last else 0)
PY
)
    if [ "${LAST:-0}" -gt "$STALL_MIN" ]; then
      # 卡死必须重启，光告警没用：整夜无人值守时，一个挂住的进程会白白占掉几小时。
      # 已完成的 episode 不会重跑，未完成的调用大多能命中缓存，重启代价很低。
      n=${RESTARTS[$M]:-0}
      if [ "$n" -lt "$MAX_RESTART" ]; then
        RESTARTS[$M]=$((n+1))
        echo "[卡死重启] $M 已 ${LAST} 分钟无调用，杀掉并重启（第 $((n+1)) 次）"
        pkill -9 -f -- "--model $M " 2>/dev/null
        sleep 3
        restart "$M"
      elif [ -z "${SEEN[give$M]:-}" ]; then
        SEEN[give$M]=1
        echo "[放弃] $M 反复卡死，需要人工介入"
      fi
    fi
  done

  # ---- gemini 那条单独看 ----
  if ! pgrep -f -- "--model gemini-3.7-flash " >/dev/null 2>&1; then
    if [ -z "${SEEN[gem]:-}" ]; then
      SEEN[gem]=1
      echo "[提示] gemini-3.7-flash/MedQA 的进程已结束（正常完成或退出，醒来核对）"
    fi
  fi

  sleep 180
done
