"""从 per-call 日志重算真实花费（唯一可信来源；共享 journal 会被并发写竞争）。"""
import json, glob, collections, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
tot = 0.0; bym = collections.Counter(); byday = collections.Counter()
import time
for f in glob.glob(str(ROOT / "logs/calls_*.jsonl")):
    day = pathlib.Path(f).stem.replace("calls_", "")
    for l in open(f):
        try:
            r = json.loads(l); u = r.get("usd", 0.0)
            tot += u; bym[r["model"]] += u; byday[day] += u
        except Exception:
            pass
print(f"真实累计花费  ${tot:.2f}")
for d, v in sorted(byday.items()):
    print(f"  {d}  ${v:.2f}")
for m, v in bym.most_common():
    print(f"  {m:16s} ${v:.2f}")
