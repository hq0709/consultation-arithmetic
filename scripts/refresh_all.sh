#!/usr/bin/env bash
# 网格变动之后重算论文里的一切。顺序有依赖：先分析后图表，因为图读 JSON。
#
# 用法: scripts/refresh_all.sh [--figs-only]
# 跑完记得核对 grid_files.py 打印的规模是否与论文正文一致。
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== 主网格 ==="
python3 experiments/grid_files.py || exit 1

if [ "${1:-}" != "--figs-only" ]; then
  echo; echo "=== 分析 ==="
  for s in ceiling_numbers independence refresh_stale selection_signals \
           aggregation_ceiling learned_selector phi_decomposition \
           diversity_ladder domain_complexity robustness generic_control; do
    printf "  %-22s " "$s"
    if python3 "experiments/$s.py" >"/tmp/refresh_$s.log" 2>&1; then echo ok
    else echo "失败 —— 见 /tmp/refresh_$s.log"; fi
  done
fi

echo; echo "=== 图 ==="
for s in fig_main fig_rest fig_forest fig_ceiling fig_diversity \
         fig_benchmark_dynamics fig_heterogeneity fig_arch; do
  printf "  %-22s " "$s"
  if python3 "experiments/$s.py" >"/tmp/refresh_$s.log" 2>&1; then echo ok
  else echo "失败 —— 见 /tmp/refresh_$s.log"; fi
done

echo; echo "=== 表 ==="
for s in make_paper_tables make_appendix_tables make_table5; do
  printf "  %-22s " "$s"
  if python3 "experiments/$s.py" >"/tmp/refresh_$s.log" 2>&1; then echo ok
  else echo "失败 —— 见 /tmp/refresh_$s.log"; fi
done

echo; echo "=== 编译 ==="
( cd paper && tectonic -X compile main.tex 2>&1 | grep -iE "^error|Fatal" )
python3 - <<'PY'
import fitz
d = fitz.open('paper/main.pdf')
bad = [i+1 for i in range(len(d)) if '??' in d[i].get_text()]
print(f"  {len(d)} 页 · 未解析引用 {bad or '无'}")
PY

echo; echo "=== 提醒：正文里这些数字要手工核对是否跟上 ==="
python3 - <<'PY'
import json
cn = json.load(open('results/ceiling_numbers.json'))
d  = cn['decomp']
print(f"  单医生 {cn['single']:.1f} · oracle {cn['oracle_n9']:.1f} · headroom {cn['oracle_n9']-cn['single']:.1f}pp")
print(f"  最好架构 {cn['best_acc_n9']:.1f} · 名册 {d['panel_oracle']-d['sc_oracle']:+.2f}pp ({d['roster_share']:.1f}%)")
print("  kappa:", {k: round(v,1) for k,v in cn['kappa_by_arch'].items()})
sr = json.load(open('results/stale_refresh.json'))
for k in ('Independent|N=9','Decentralized|N=9'):
    r = sr['consensus'][k]
    print(f"  {k}: 一致 {r['unanimity']:.1f}% · P(错|一致) {r['p_wrong_given_unanimous']:.1f}%")
PY
