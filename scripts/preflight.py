"""换服务器后先跑这个。任何一项失败都不要开始正式实验。

  python3 scripts/preflight.py                 # 只检查闭源臂
  python3 scripts/preflight.py --local         # 连开源权重端点一起检查
"""
import sys, os, pathlib, json, argparse, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  [ok] ", "  [!!] ", "  [--] "
fails = []


def check(name, fn, fatal=True):
    try:
        msg = fn()
        print(OK + f"{name}: {msg}")
        return True
    except Exception as e:                                    # noqa: BLE001
        print((BAD if fatal else WARN) + f"{name}: {e}")
        if fatal:
            fails.append(name)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="同时检查 vLLM 端点")
    ap.add_argument("--items", type=int, default=3, help="冒烟测试的题目数")
    a = ap.parse_args()

    print("=" * 66); print("1. 环境"); print("=" * 66)

    def _py():
        assert sys.version_info >= (3, 10), f"需要 3.10+，当前 {sys.version_info[:2]}"
        return f"python {'.'.join(map(str, sys.version_info[:3]))}"
    check("python 版本", _py)

    def _deps():
        import numpy, scipy, sklearn, matplotlib, openai      # noqa: F401
        return "openai/numpy/scipy/sklearn/matplotlib 齐全"
    check("依赖", _deps)

    def _env():
        from common.llm import DAILY_CAP_USD
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 未设置（.env 是否存在？）")
        return f"API key 已加载，日上限 ${DAILY_CAP_USD}"
    check(".env", _env)

    def _dirs():
        missing = [d for d in ("data", "results", "cache", "logs") if not (ROOT / d).exists()]
        for d in missing:
            (ROOT / d).mkdir(parents=True, exist_ok=True)
        return "data/results/cache/logs 就位" + (f"（新建了 {missing}）" if missing else "")
    check("目录", _dirs)

    print("\n" + "=" * 66); print("2. 数据"); print("=" * 66)

    def _data():
        need = ["medxpertqa_250.jsonl", "medagentsbench_250.jsonl", "medqa_250.jsonl"]
        miss = [n for n in need if not (ROOT / "data" / n).exists()]
        if miss:
            raise RuntimeError(f"缺 {miss} —— 先跑 python3 data/build_datasets.py")
        counts = {n: sum(1 for _ in (ROOT / "data" / n).open()) for n in need}
        bad = {k: v for k, v in counts.items() if v != 250}
        if bad:
            raise RuntimeError(f"条数不对: {bad}（应各为 250）")
        return "三个 benchmark 各 250 题"
    check("题目文件", _data)

    print("\n" + "=" * 66); print("3. 闭源模型冒烟测试"); print("=" * 66)

    def _smoke(model, effort=None):
        def f():
            from common.llm import chat
            t0 = time.time()
            r = chat(model, [{"role": "user",
                              "content": 'Reply with JSON only: {"answer": "B"}'}],
                     max_tokens=64, temperature=0.0, effort=effort,
                     json_mode=True, use_cache=False, tag="preflight")
            if not r["text"].strip():
                raise RuntimeError("返回空串（推理模型的 max_tokens 是否太小？）")
            return f'{r["text"].strip()[:40]!r}  {time.time()-t0:.1f}s'
        return f
    for m, e in [("gpt-4.1-nano", None), ("gpt-5-nano", "low"), ("gpt-5-mini", "low")]:
        check(m, _smoke(m, e))

    if a.local:
        print("\n" + "=" * 66); print("4. 开源权重端点"); print("=" * 66)
        from common.llm import _local_registry
        reg = _local_registry()
        if not reg and os.environ.get("VLLM_BASE_URL"):
            reg = {"(VLLM_BASE_URL)": os.environ["VLLM_BASE_URL"]}
        if not reg:
            print(WARN + "未注册任何本地模型：设置 LOCAL_MODELS 或 VLLM_BASE_URL")
        for name, url in reg.items():
            def f(name=name, url=url):
                import urllib.request
                d = json.load(urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10))
                served = [m["id"] for m in d.get("data", [])]
                if name.startswith("(") is False and name not in served:
                    raise RuntimeError(f"端点在服，但没有 '{name}'（在服的是 {served}）。"
                                       f"--served-model-name 必须与注册名一致")
                return f"{url} 在服: {served}"
            check(f"端点 {name}", f)

        for name in [n for n in reg if not n.startswith("(")]:
            def g(name=name):
                from common.llm import chat
                from panels.base import parse_opinion
                t0 = time.time()
                r = chat(f"local/{name}", [{"role": "user", "content":
                         "A 62-year-old with crushing chest pain radiating to the left arm. "
                         "Options: A) GERD B) Myocardial infarction C) Costochondritis "
                         "D) Anxiety. Reply as JSON: "
                         '{"answer":"<letter>","confidence":<0-100>,"reason":"<short>"}'}],
                         max_tokens=256, temperature=0.3, json_mode=True,
                         use_cache=False, tag="preflight")
                op = parse_opinion(r["text"], list("ABCD"), agent="probe")
                if op.answer is None:
                    raise RuntimeError(f"答案解析失败，原始输出: {r['text'][:120]!r}")
                return f"解析出 {op.answer}（置信 {op.confidence:.0f}） {time.time()-t0:.1f}s"
            check(f"生成+解析 local/{name}", g)

    print("\n" + "=" * 66)
    if fails:
        print(f"预检未通过：{fails}"); sys.exit(1)
    print("预检全部通过，可以开跑。")


if __name__ == "__main__":
    main()
