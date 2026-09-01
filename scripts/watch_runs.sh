#!/usr/bin/env bash
# 监控全部在跑的实验。只在出问题或到里程碑时输出一行，避免刷屏。
cd /home/myid/hj67104/consultation_saturation
declare -A SEEN
LOGS="logs/ds_deepseek-v4-flash.log logs/ds_deepseek-v4-pro.log logs/or_cot_qwen3.8-flash.log"
LAST_SUMMARY=0
while true; do
  for L in $LOGS; do
    [ -f "$L" ] || continue
    tag=$(basename "$L" .log)
    # 1) 质量守卫触发
    if grep -q "QUALITY STOP" "$L" && [ -z "${SEEN[q$tag]:-}" ]; then
      SEEN[q$tag]=1
      echo "[ALERT $tag] 质量守卫触发，实验已中止: $(grep -m1 -A1 'QUALITY STOP' "$L" | tail -1 | cut -c1-120)"
    fi
    # 2) 空输出/解析失败率过高的 cell（单 cell 超过 5%，即 250 题里超过 12 条）。
    #    正常水平是 0-2/250，对此报警只会刷屏。
    n_bad=$(grep -cE "blank=([1-9][0-9]{1,2})|unparsed=([1-9][0-9]{1,2})" "$L" 2>/dev/null | head -1)
    n_bad=${n_bad:-0}
    if [ "$n_bad" -gt "${SEEN[b$tag]:-0}" ]; then
      SEEN[b$tag]=$n_bad
      echo "[WARN $tag] $n_bad 个 cell 的无效输出超过 5%: $(grep -hE 'blank=([1-9][0-9]{1,2})|unparsed=([1-9][0-9]{1,2})' "$L" | tail -1 | cut -c1-110)"
    fi
    # 3) 高错误率 cell
    if grep -qE "err=([1-9][0-9]|[1-9][0-9][0-9])" "$L" && [ -z "${SEEN[e$tag]:-}" ]; then
      SEEN[e$tag]=1
      echo "[WARN $tag] 出现高错误率 cell: $(grep -hE 'err=([1-9][0-9]|[1-9][0-9][0-9])' "$L" | tail -1 | cut -c1-110)"
    fi
    # 4) 完成
    if grep -qE "GRID DONE|RESUME DONE|MATCHED DONE" "$L" && [ -z "${SEEN[d$tag]:-}" ]; then
      SEEN[d$tag]=1
      echo "[DONE $tag] 实验完成 $(date '+%H:%M')"
    fi
  done
  # 5) 每 30 分钟一次进度摘要
  NOW=$(date +%s)
  if [ $((NOW-LAST_SUMMARY)) -ge 1800 ]; then
    LAST_SUMMARY=$NOW
    # 直接数结果里已完成的 cell（>=240 题），不依赖日志 —— 日志改名/重启都会让行数统计归零
    read C S <<<"$(python3 - <<'PYEOF' 2>/dev/null || echo "? ?"
import json, glob, collections, sys
sys.path.insert(0, ".")
cells = collections.defaultdict(set)
for f in glob.glob("results/CLA_*.jsonl") + glob.glob("results/OR_*.jsonl"):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        cells[(r.get("model"), r.get("bench"), r.get("arch"), r.get("N"))].add(r.get("qid"))
from common.llm import global_spend_usd, vendor_spend_usd
print(sum(1 for q in cells.values() if len(q) >= 240),
      f"{global_spend_usd():.2f}/{vendor_spend_usd('anthropic'):.2f}")
PYEOF
)"
    # 只在有变化时播报：Claude 跑完后状态会连续多小时不变，
    # 每 30 分钟重复同一行只会淹没真正的告警
    LINE="OpenRouter+Claude $C/351 cell · 今日/Anthropic累计 \$$S"
    if [ "$LINE" != "${SEEN[summary]:-}" ]; then
      SEEN[summary]="$LINE"
      echo "[进度 $(date '+%H:%M')] $LINE"
    fi
  fi
  sleep 60
done
