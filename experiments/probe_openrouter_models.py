"""新模型发车前的 API 兼容性检查。

run_grid 会用到三个可选参数，任何一个不被后端接受都会在网格跑到一半时抛 400：
  seed            —— 复现实验用；Gemini 直连不认，OpenRouter 上各家不一
  n>1             —— 自洽性基线按 n=8 成块采样
  response_format —— json_object 模式
外加最要命的一条：思考 token 是否会把 max_tokens 吃光导致空输出（Sonnet 5 的坑）。

逐项单独试，失败只记不抛，最后打一张表。
"""
import sys, os, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from openai import OpenAI
from common.llm import OPENROUTER_MAP, OPENROUTER_BASE

MODELS = [m.strip() for m in (os.environ.get("PROBE_MODELS") or "").split(",") if m.strip()]
PROMPT = ('A 55-year-old man has crushing substernal chest pain radiating to the left arm. '
          'Options: A) MI  B) GERD  C) Costochondritis  D) Anxiety.\n'
          'Reply with JSON only: {"answer":"<A-D>","confidence":<0-100>,"reason":"<20 words>"}')


def main():
    cli = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE, timeout=180.0)
    print(f"{'model':<20}{'基本':>6}{'seed':>7}{'n=4':>7}{'json':>7}{'出tok':>8}{'思考':>7}{'空输出风险':>12}")
    for m in MODELS:
        wire = OPENROUTER_MAP[m]
        base = dict(model=wire, messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=2000, temperature=0.7)

        def try_(extra, want_n=1):
            try:
                r = cli.chat.completions.create(**{**base, **extra})
                txts = [(c.message.content or "") for c in r.choices]
                if len(txts) < want_n:
                    return None, f"只返回{len(txts)}"
                return r, None
            except Exception as e:
                return None, str(e)[:40]

        r0, e0 = try_({})
        if r0 is None:
            print(f"{m:<20}{'✗':>6}  {e0}")
            continue
        out = r0.usage.completion_tokens
        ctd = getattr(r0.usage, "completion_tokens_details", None)
        rsn = (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
        txt = r0.choices[0].message.content or ""
        r1, _ = try_({"seed": 7})
        r2, _ = try_({"n": 4}, want_n=4)
        r3, _ = try_({"response_format": {"type": "json_object"}})
        risk = "高(思考>60%)" if rsn > 0.6 * max(out, 1) else ("中" if rsn else "低")
        print(f"{m:<20}{'✓' if txt.strip() else '空':>6}{'✓' if r1 else '✗':>7}"
              f"{'✓' if r2 else '✗':>7}{'✓' if r3 else '✗':>7}{out:>8}{rsn:>7}{risk:>12}")


if __name__ == "__main__":
    main()
