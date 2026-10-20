#!/usr/bin/env bash
# build.sh NAME: assemble NAME.py from NAME.doc + ../alt/_header.txt + NAME.body, render, and run the
# nature-figure QA (static preflight, glyph floor, collisions). Outputs: paper/cvpr/figure_candidates/round3.
set -u
cd "$(dirname "$0")"
n=$1
cat $n.doc ../alt/_header.txt $n.body > $n.py
python3 $n.py 2>&1 | grep -v "^findfont" | tail -4
S=~/.claude/skills/nature-figure/scripts
F=../../../paper/cvpr/figure_candidates/round3/$n
python3 $S/validate_figure.py $n.py | grep -E "^\[FAIL|^\[WARN|verdict"
python3 $S/audit_pdf_text.py $F.pdf --min-pt 5 | grep -E "below minimum|verdict"
python3 $S/audit_figure_collisions.py $F.pdf --json-out ../qa/r3/$n.collision.json \
    --overlay-pdf ../qa/r3/$n.collision.pdf | grep -E "summary|FAIL\]|WARN\] page"
