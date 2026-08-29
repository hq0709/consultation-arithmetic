#!/usr/bin/env bash
cd /home/myid/hj67104/consultation_saturation
while [ "$(ps -eo args | grep -c '[r]un_grid.py')" -gt 0 ]; do
  python3 - <<'PY'
import json, glob, time, subprocess
tot = 0.0
for f in glob.glob('logs/calls_*.jsonl'):
    for l in open(f):
        try: tot += json.loads(l).get('usd', 0.0)
        except Exception: pass
now = time.time(); n = 0
for l in open('logs/calls_%s.jsonl' % time.strftime('%Y%m%d')):
    try:
        if now - json.loads(l)['ts'] < 60: n += 1
    except Exception: pass
procs = subprocess.run(['ps','-eo','args'],capture_output=True,text=True).stdout
np = sum(1 for x in procs.splitlines() if 'run_grid.py' in x)
done = 0
for f in glob.glob('logs/r_*.log'):
    done += sum(1 for l in open(f) if l.startswith('[') and '/27]' in l)
print(f"[{time.strftime('%H:%M')}] ${tot:.2f} | {n} calls/min | {np} procs | {done}/189 cells", flush=True)
PY
  sleep 600
done
echo "ALL_DONE $(date)"
