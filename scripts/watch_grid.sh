#!/usr/bin/env bash
cd /home/myid/hj67104/consultation_saturation
while ps -eo cmd | grep -q '[r]un_grid.py'; do
  D=$(python3 -c "
import json,glob,time
tot=0
for f in glob.glob('logs/calls_*.jsonl'):
    for l in open(f):
        try: tot+=json.loads(l).get('usd',0)
        except: pass
now=time.time(); n=0
for l in open('logs/calls_20260829.jsonl'):
    try:
        if now-json.loads(l)['ts']<60: n+=1
    except: pass
print(f'{tot:.2f} {n}')")
  echo "[$(date +%H:%M)] 花费 \$$(echo $D|cut -d' ' -f1) | $(echo $D|cut -d' ' -f2) 调用/分 | 进程 $(ps -eo cmd|grep -c '[r]un_grid.py') | cells: $(grep -hcE '^\[[0-9]+/27\]' logs/tier_T*.log logs/one_*.log 2>/dev/null|paste -sd+|bc)"
  sleep 900
done
echo "ALL_GRID_DONE $(date)"
