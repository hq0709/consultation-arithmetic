"""Experiment B perceiver pass (docs/IMAGING_PROTOCOL.md section 12): write every findings report
once, before any reasoner reads one.

Reports are keyed on (perceiver, image, slot) only, so every reasoner and every question on the same
image reads the same cached report. Without this pass, two workers that reach the same image at the
same time both call the perceiver and the later cache write wins, so reasoners could read different
reports for the same slot.
  python experiments/write_reports.py --perceiver gpt-5-mini
"""
import argparse, collections, json, pathlib, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import LEDGER, pmap  # noqa: E402
from panels.architectures import write_report  # noqa: E402

SETS = "vqarad_closed_250,slake_closed_250,mxqa_mm_rad_250"


class Count:
    def __init__(self):
        self.lock, self.c = threading.Lock(), collections.Counter()

    def add(self, r):
        with self.lock:
            self.c["cached" if r.get("cached") else "new"] += 1
            self.c["empty"] += int(not r["text"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perceiver", required=True)
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--slots", type=int, default=9)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    seen, images = set(), []
    for s in a.sets.split(","):
        for l in list(open(ROOT / f"data/{s}.jsonl"))[:a.limit]:
            t = tuple(json.loads(l).get("images") or [])
            if t and t not in seen:
                seen.add(t); images.append(list(t))
    jobs = [(im, i) for im in images for i in range(a.slots)]
    cnt, err, t0 = Count(), collections.Counter(), time.time()
    print(f"[reports] {a.perceiver}: {len(images)} images x {a.slots} slots = {len(jobs)} reports", flush=True)

    def one(j):
        try:
            return len(write_report(a.perceiver, j[1], j[0], a.temp, cnt).split())
        except Exception as e:  # noqa: BLE001  a failed slot is written later by the B run itself
            err[type(e).__name__] += 1
            print(f"[reports] {a.perceiver} slot {j[1]} {j[0]}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            return None
    words = [w for w in pmap(one, jobs, workers=a.workers) if w is not None]
    print(f"[reports] {a.perceiver}: {dict(cnt.c)} errors={dict(err)} words median="
          f"{sorted(words)[len(words) // 2] if words else None} max={max(words, default=None)} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    print(LEDGER.report(), flush=True)


if __name__ == "__main__":
    main()
