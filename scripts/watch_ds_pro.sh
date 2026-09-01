#!/usr/bin/env bash
# deepseek-v4-pro 补跑主网格的看护。三类故障：进程没了 / 卡死 / 截断复发。
set -uo pipefail
cd /home/myid/hj67104/consultation_saturation
set -a; source .env 2>/dev/null; set +a
STALL_MIN=30
declare -A SEEN
while true; do
  for B in medagentsbench medqa; do
    F="results/OR_deepseek-v4-pro_${B}.jsonl"
    n=$(python3 - "$F" <<'PY' 2>/dev/null || echo 0
import sys, json, collections
c=collections.defaultdict(set)
try:
    for l in open(sys.argv[1]):
        if not l.strip(): continue
        try: r=json.loads(l)
        except Exception: continue
        if r.get("status")=="ok": c[(r["arch"],r["N"])].add(r["qid"])
except FileNotFoundError: pass
print(sum(1 for v in c.values() if len(v)>=240))
PY
)
    echo "[$(date +%H:%M)] $B  满额 ${n}/27"
    if [ "$n" -ge 27 ] && [ -z "${SEEN[$B]:-}" ]; then SEEN[$B]=1; echo "  [完成] $B"; fi
  done
  # 截断复发
  T=$(grep -c "预算不足.*deepseek-v4-pro" logs/ds_pro_fullgrid.log 2>/dev/null || echo 0)
  [ "$T" -gt 0 ] && echo "  [警告] 仍有 $T 次截断重试 —— 262144 的预算不够"
  # 进程存活
  pgrep -f "model deepseek-v4-pro" >/dev/null || { echo "  [进程已退出]"; break; }
  sleep 600
done
