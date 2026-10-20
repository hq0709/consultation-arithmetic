#!/usr/bin/env bash
# build.sh NAME: assemble NAME.py from NAME.doc + _header.txt + NAME.body, render, and run the skill's QA
set -u
cd "$(dirname "$0")"
n=$1
cat $n.doc _header.txt $n.body > $n.py
python3 $n.py 2>&1 | grep -v "^findfont" | tail -2
S=~/.claude/skills/nature-figure/scripts
F=../../../paper/cvpr/figure_candidates/alt/$n
python3 $S/validate_figure.py $n.py | grep -E "^\[FAIL|verdict"
python3 $S/audit_pdf_text.py $F.pdf --min-pt 5 | grep verdict
python3 $S/audit_figure_collisions.py $F.pdf --json-out ../qa/alt/$n.collision.json | grep -E "summary|FAIL\]|WARN\] page"
