#!/usr/bin/env bash
# nature-figure rendered QA for one candidate: static preflight, glyph floor, collisions.
# usage: qa_run.sh <script-stem>
set -u
S=~/.claude/skills/nature-figure/scripts
F=../../paper/cvpr/figure_candidates/$1
echo "== $1"
python3 $S/validate_figure.py $1.py | grep -E "^\[(FAIL|WARN)\]|verdict"
python3 $S/audit_pdf_text.py $F.pdf --min-pt 5 --json > qa/$1.pdf-text.json; python3 $S/audit_pdf_text.py $F.pdf --min-pt 5 | grep -E "below minimum|verdict"
python3 $S/audit_figure_collisions.py $F.pdf --json-out qa/$1.collision-audit.json --overlay-pdf qa/$1.collision-audit.pdf | grep -E "^\[|verdict|summary"
