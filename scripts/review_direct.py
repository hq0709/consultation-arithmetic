"""Cross-model code review via the OpenAI API directly (codex CLI proved unreliable here)."""
import sys, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import chat, LEDGER

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-5.4")
ap.add_argument("--effort", default="high")
a = ap.parse_args()

p = (ROOT / a.prompt).read_text()
r = chat(a.model, [{"role": "user", "content": p}],
         system=("You are a meticulous senior reviewer for a top medical-AI venue who also "
                 "writes production research code. Be concrete, terse, and adversarial."),
         effort=a.effort, max_tokens=16000, tag="codereview", use_cache=True)
(ROOT / a.out).write_text(r["text"])
print(f"wrote {a.out}  ({len(r['text'])} chars, in={r['input_tokens']} out={r['output_tokens']} "
      f"rsn={r['reasoning_tokens']})")
print(LEDGER.report())
