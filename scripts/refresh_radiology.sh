#!/usr/bin/env bash
# Rebuild every radiology-manuscript number, table and figure from results/IMG2_*.jsonl and results/OW_*.jsonl.
#   scripts/refresh_radiology.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for e in A B C M D; do
  echo "[analysis] $e"; python3 experiments/imaging_v2_analysis.py $e > /dev/null
done
for f in fig_panels fig_arithmetic fig_evidence fig_prior fig_region fig_report fig_budget fig_memory fig_conformity fig_openweight; do
  echo "[figure] $f"; (cd experiments/radiology_fig && python3 $f.py > /dev/null)
done
python3 experiments/radiology_numbers.py
cd paper/radiology
for doc in main supplement; do
  tectonic -X compile $doc.tex > /dev/null 2>&1 && echo "[pdf] $doc.pdf"
done
