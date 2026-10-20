#!/usr/bin/env bash
# build_s25.sh NAME: assemble NAME.py from NAME.doc + _header.txt + NAME.body, render, and run the
# skill's QA (own copy of build.sh for the S2/S5 candidates, so edits to the shared one cannot change it)
set -u
cd "$(dirname "$0")"
n=$1
cat $n.doc _header.txt $n.body > $n.py
python3 $n.py 2>&1 | grep -v "^findfont" | tail -3
S=~/.claude/skills/nature-figure/scripts
F=../../../paper/cvpr/figure_candidates/round3/$n
python3 $S/validate_figure.py $n.py | grep -E "^\[FAIL|^\[WARN|verdict"
python3 $S/audit_pdf_text.py $F.pdf --min-pt 5 | grep -E "below minimum|verdict"
python3 $S/audit_figure_collisions.py $F.pdf --json-out ../qa/r3/$n.collision.json --overlay-pdf ../qa/r3/$n.collision.pdf | grep -E "summary|FAIL\]|WARN\] page"
