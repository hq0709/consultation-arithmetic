#!/usr/bin/env bash
# 等 gemini-3.7-flash 的每日配额重置后补全网格。
#   - 每日额度 10,000 请求/模型，用完后约 7.4 小时重置
#   - run_grid.py 会跳过已完成的 episode、重试失败的，所以直接重跑同一条命令即可
#   - 并发压到 4，因为 3.7-flash 的限流比 lite 严重得多（293 分钟 vs 12 分钟）
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
export LLM_MAX_INFLIGHT=4 LLM_WORKERS=4
ARCHES="independent,centralized,discussion,tiered,zeroshot,cot,sc"
COMMON="--temp 0.7 --theta 80 --Ns 1,3,5,7,9 --sc-ks 1,3,5,9,15 --seeds 1 --workers 4 --arches $ARCHES"
M=gemini-3.7-flash

probe () {  # 配额恢复了吗
  python3 - <<'PY'
import sys, os, pathlib
sys.path.insert(0, ".")
for l in pathlib.Path(".env").read_text().splitlines():
    if "=" in l and not l.strip().startswith("#"):
        k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
from common.llm import chat
try:
    chat("gemini-3.7-flash", [{"role": "user", "content": "ok"}],
         max_tokens=8, use_cache=False, tag="quota-probe")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

DAY=0
while :; do
  DAY=$((DAY+1))
  echo "=== 第 $DAY 轮：等待配额 ($(date '+%m-%d %H:%M')) ==="
  # 连续 3 次成功才算配额真回来：单次成功可能只是限流的抖动，
  # 据此启动会让整轮 6750 个 episode 全部 429，白耗一天
  while :; do
    ok=0
    for _ in 1 2 3; do probe && ok=$((ok+1)) || break; sleep 20; done
    [ "$ok" -ge 3 ] && break
    sleep 600
  done
  echo "=== 配额已恢复 ($(date '+%m-%d %H:%M'))，开始补全 ==="
  for B in medxpertqa medagentsbench medqa; do   # 由缺口小到大：先让接近完成的 benchmark 落地
    echo "--- $M / $B ---"
    python3 -u experiments/run_grid.py $COMMON --model "$M" \
      --items "data/${B}_250.jsonl" --limit 250 --out "GEM_${M}_${B}.jsonl"
  done
  # 检查是否还有失败
  REMAIN=$(python3 - <<'PY'
import json, glob
n = 0
for f in ["results/GEM_gemini-3.7-flash_medqa.jsonl",
          "results/GEM_gemini-3.7-flash_medxpertqa.jsonl"]:
    try:
        ok = err = 0
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "error": err += 1
            else: ok += 1
        n += 6750 - ok
    except FileNotFoundError:
        n += 6750
print(max(0, n))
PY
)
  echo "=== 本轮结束，仍缺 $REMAIN 个 episode ==="
  [ "$REMAIN" -le 50 ] && break
done
echo "GEMINI RESUME DONE"
python3 scripts/true_spend.py
