"""OpenAI-only LLM client with disk cache, retry, cost ledger, cross-process daily cap,
per-call jsonl log, multi-sample (n=) support, and a bounded thread pool.

Adapted from night_shift/common/llm.py (same author, same machine). OpenAI-only by
project constraint: this study runs on OpenAI models exclusively.
"""
from __future__ import annotations
import os, json, time, hashlib, threading, pathlib, random, re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

_ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = _ROOT / "cache"
LOG_DIR = _ROOT / "logs"
RESULTS_DIR = _ROOT / "results"
for _d in (CACHE_DIR, LOG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


def _load_env():
    p = _ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# USD per 1M tokens: (input, output). OpenAI list prices, recorded 2026-08-28.
# NOTE: verified against the platform pricing page before the final budget table.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano":  (0.10, 0.40),
    "gpt-4.1-mini":  (0.40, 1.60),
    "gpt-4.1":       (2.00, 8.00),
    "gpt-4o-mini":   (0.15, 0.60),
    "gpt-4o":        (2.50, 10.00),
    "gpt-5-nano":    (0.05, 0.40),
    "gpt-5-mini":    (0.25, 2.00),
    "gpt-5":         (1.25, 10.00),
    "gpt-5.4-nano":  (0.05, 0.40),
    "gpt-5.4-mini":  (0.25, 2.00),
    "gpt-5.4":       (1.25, 10.00),
    # Gemini Flash 系列
    "gemini-3.7-flash":      (0.30, 2.50),
    "gemini-3.5-flash":      (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash":      (0.30, 2.50),
    # Anthropic Claude（官方公开价，美元 / 百万 token）
    "claude-haiku-4-5-20251001":  (1.00,  5.00),
    "claude-sonnet-5":            (2.00, 10.00),
    "claude-opus-5":              (5.00, 25.00),
    "claude-fable-5":             (10.00, 50.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-opus-4-5-20251101":   (5.00, 25.00),
}
# OpenAI reasoning models: no `temperature`, no `top_p`, use max_completion_tokens.
REASONING_PREFIXES = ("gpt-5", "o3", "o4")
REASONING_MIN_TOKENS = int(os.environ.get("REASONING_MIN_TOKENS", "1600"))


# ---- 开源权重模型：通过 vLLM 的 OpenAI 兼容端点接入。
# 注册方式：环境变量 LOCAL_MODELS='name=http://host:port/v1,name2=...'
# 模型名以 "local/" 开头即走本地端点；计价为 0（GPU 时间不计入美元上限），
# 但 token 仍照常记账，因此成本图上它们出现在 0 美元处而不是缺失。
LOCAL_PREFIX = "local/"

# ---- Google Gemini：官方 OpenAI 兼容端点，复用同一个客户端栈。
GEMINI_BASE = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")


def is_gemini(model: str) -> bool:
    return model.startswith("gemini-")


# ---- Anthropic Claude：官方 OpenAI 兼容端点。
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

# ---- OpenRouter：同一批模型的第二条路由。
# 只替换「客户端 + 线上模型名」，参数处理仍按底层厂商走（gemini- 依旧丢 seed、
# claude- 依旧不发 response_format），所以 is_gemini / is_claude 的判定保持不变。
# 模型名不变意味着缓存键与结果里的 model 字段也不变 —— 直连与 OpenRouter 的响应
# 可以共用缓存。这一点经过实测：MedQA CoT 逐题 100% 一致，面板配置下
# phi 0.917(直连) vs 0.931(OpenRouter)。
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MAP = {
    "gemini-3.7-flash": "google/gemini-3.7-flash",
    "gemini-3.5-flash-lite": "google/gemini-3.5-flash-lite",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "claude-opus-5": "anthropic/claude-opus-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
    # 第四类生态：中国实验室。跨 OpenAI/Google/Anthropic 只是跨三家很像的美国实验室，
    # 这四家的语料、训练目标与对齐流程都不同，是对「独立性买不到」最硬的检验。
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "qwen3.8-flash": "qwen/qwen3.8-flash",
    "qwen3.8-max": "qwen/qwen3.8-max",
    # glm-latest 是浮动别名（当前指向 z-ai/glm-5.3）。论文需要固定的模型身份，
    # 别名随时可能换底座，所以钉死到具体版本。
    "glm-5.3": "z-ai/glm-5.3",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
}
# 只有直连拿不到的模型默认走 OpenRouter；其余靠 OPENROUTER_MODELS 显式开启，
# 免得无意中把已经跑好的网格换了路由。
OPENROUTER_ALWAYS = {"gemini-3.1-pro", "claude-opus-5",
                     "deepseek-v4-flash", "deepseek-v4-pro",
                     "qwen3.8-flash", "qwen3.8-max",
                     "glm-5.3", "glm-5.3-flash"}
# 同一个模型走两条路单价不同（gemini-3.7-flash 直连 0.30/2.50，OpenRouter 0.75/3.75），
# 计费必须跟着路由走，否则花费统计会系统性偏低。
OPENROUTER_PRICING = {
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro": (2.00, 12.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "deepseek-v4-flash": (0.081, 0.162),
    "deepseek-v4-pro": (1.60, 3.20),
    "qwen3.8-flash": (0.15, 0.47),
    "qwen3.8-max": (2.00, 6.00),
    "glm-5.3": (1.40, 4.40),
    "glm-5.3-flash": (0.075, 0.25),
}


def _openrouter_models() -> set[str]:
    v = os.environ.get("OPENROUTER_MODELS", "")
    return OPENROUTER_ALWAYS | {x.strip() for x in v.split(",") if x.strip()}


def is_openrouter(model: str) -> bool:
    return model in _openrouter_models() and model in OPENROUTER_MAP


def wire_name(model: str) -> str:
    """发给端点的模型名。"""
    if is_openrouter(model):
        return OPENROUTER_MAP[model]
    return model[len(LOCAL_PREFIX):] if is_local(model) else model


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def _local_registry() -> dict[str, str]:
    reg = {}
    for item in os.environ.get("LOCAL_MODELS", "").split(","):
        item = item.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            reg[k.strip()] = v.strip()
    return reg


def is_local(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)


def local_endpoint(model: str) -> str:
    name = model[len(LOCAL_PREFIX):]
    reg = _local_registry()
    if name in reg:
        return reg[name]
    base = os.environ.get("VLLM_BASE_URL")
    if base:
        return base
    raise RuntimeError(
        f"no endpoint for {model}: set LOCAL_MODELS='{name}=http://host:port/v1' or VLLM_BASE_URL")


def is_reasoning(model: str) -> bool:
    """仅指 OpenAI 的推理系列（改用 max_completion_tokens / reasoning_effort）。
    Gemini 与本地模型走标准参数。"""
    return (not is_local(model)) and (not is_gemini(model)) and (not is_claude(model)) \
        and model.startswith(REASONING_PREFIXES)


@dataclass
class Usage:
    calls: int = 0
    samples: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_hits: int = 0
    cost_usd: float = 0.0


class Ledger:
    def __init__(self):
        self._lock = threading.Lock()
        self.by_model: dict[str, Usage] = {}

    def add(self, model, inp, out, cached, n=1, reasoning=0):
        delta = 0.0
        with self._lock:
            u = self.by_model.setdefault(model, Usage())
            u.calls += 1
            u.samples += n
            if cached:
                u.cached_hits += 1
            else:
                u.input_tokens += inp
                u.output_tokens += out
                u.reasoning_tokens += reasoning
                pi, po = ((OPENROUTER_PRICING.get(model) if is_openrouter(model) else None)
                          or PRICING.get(model, (0.0, 0.0)))
                delta = inp / 1e6 * pi + out / 1e6 * po
                u.cost_usd += delta
        if delta:
            record_global_spend(delta, vendor=vendor_of(model))

    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.by_model.values())

    def report(self) -> str:
        h = f"{'model':16s} {'calls':>7s} {'cachd':>7s} {'samp':>7s} {'in_tok':>10s} {'out_tok':>10s} {'USD':>9s}"
        lines = [h]
        for m, u in sorted(self.by_model.items()):
            lines.append(f"{m:16s} {u.calls:7d} {u.cached_hits:7d} {u.samples:7d} "
                         f"{u.input_tokens:10d} {u.output_tokens:10d} {u.cost_usd:9.4f}")
        lines.append(f"{'TOTAL':16s} {'':7s} {'':7s} {'':7s} {'':10s} {'':10s} {self.total_cost():9.4f}")
        return "\n".join(lines)


LEDGER = Ledger()
DAILY_CAP_USD = float(os.environ.get("DAILY_CAP_USD", "25"))
# 思考挤掉输出时，预算最多翻几次。两次足以从 2000 涨到 8000。
# 只允许加倍一次。思考量是重尾分布：正常题目 p99.5 已被 32000 覆盖，
# 再往上是模型陷入循环的病态题目 —— 给多少烧多少，追不上。
# 实测 glm-5.3 出现过 68096/64000，再翻到 128000 就是纯浪费（一次约 $0.56），
# 而且这种调用要跑好几分钟，拖慢整体。宁可让它变成一条空答（约 +1%）。
MAX_BUDGET_GROWTH = int(os.environ.get("MAX_BUDGET_GROWTH", "1"))
# 厂商级累计上限（跨天）。此前 ANTHROPIC_CAP_USD 写在 .env 里但没有任何代码读它 ——
# 是一条死配置，实际只有 DAILY_CAP 在拦。
VENDOR_CAP_USD = {
    "anthropic": float(os.environ["ANTHROPIC_CAP_USD"]) if os.environ.get("ANTHROPIC_CAP_USD") else None,
    "google": float(os.environ["GOOGLE_CAP_USD"]) if os.environ.get("GOOGLE_CAP_USD") else None,
    "openai": float(os.environ["OPENAI_CAP_USD"]) if os.environ.get("OPENAI_CAP_USD") else None,
}

# --- cross-process daily spend journal (a per-process ledger is not a daily cap) ---
_SPEND_FILE = RESULTS_DIR / "daily_spend.json"
_spend_lock = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def vendor_of(model: str) -> str | None:
    """厂商级累计上限只对付费的外部厂商有意义。"""
    if is_local(model):
        return None
    if is_openrouter(model):
        # 按 OpenRouter id 的机构前缀分账，否则新厂商会全部记到 openai 的预算上
        org = OPENROUTER_MAP[model].split("/", 1)[0]
        return {"anthropic": "anthropic", "google": "google", "openai": "openai",
                "deepseek": "deepseek", "qwen": "qwen", "z-ai": "zai",
                "moonshotai": "moonshot"}.get(org, org)
    return "anthropic" if is_claude(model) else ("google" if is_gemini(model) else "openai")


def vendor_spend_usd(vendor: str) -> float:
    """跨天累计（与 DAILY_CAP 的按天口径不同：厂商预算是一次性的总盘子）。"""
    try:
        return float(json.loads(_SPEND_FILE.read_text()).get(f"vendor:{vendor}", 0.0))
    except Exception:
        return 0.0


def global_spend_usd() -> float:
    try:
        return float(json.loads(_SPEND_FILE.read_text()).get(_today(), 0.0))
    except Exception:
        return 0.0


def record_global_spend(amount: float, vendor: str | None = None) -> float:
    """R1 #18: the in-process lock is not enough. With 9 concurrent runner processes the
    read-modify-write on the shared journal races and silently DROPS spend (observed: the
    file reset to $0.02 while true spend was $12+). Take an OS-level exclusive lock."""
    if amount <= 0:
        return global_spend_usd()
    import fcntl
    with _spend_lock:
        lockf = _SPEND_FILE.with_suffix(".lock")
        lockf.touch(exist_ok=True)
        with open(lockf, "r+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                try:
                    d = json.loads(_SPEND_FILE.read_text())
                except Exception:
                    d = {}
                d[_today()] = round(float(d.get(_today(), 0.0)) + amount, 6)
                if vendor:
                    vk = f"vendor:{vendor}"
                    d[vk] = round(float(d.get(vk, 0.0)) + amount, 6)
                tmp = _SPEND_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=1))
                tmp.replace(_SPEND_FILE)
                return d[_today()]
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)


class BudgetExceeded(RuntimeError):
    pass


_client = None
_gemini_client = None
_claude_client = None
_or_client = None
_local_clients: dict[str, Any] = {}
_json_off: dict[str, bool] = {}   # 端点拒绝 response_format 时记下，之后不再尝试
_temp_off: dict[str, bool] = {}   # 端点拒绝自定义 temperature 时记下（Claude 5 系列）
_client_lock = threading.Lock()


def _openai(model: str | None = None):
    """按模型名路由：OpenAI 官方 / Gemini 兼容端点 / 本地 vLLM 端点。"""
    global _client, _gemini_client, _claude_client, _or_client
    if model and is_openrouter(model):
        with _client_lock:
            if _or_client is None:
                from openai import OpenAI
                k = os.environ.get("OPENROUTER_API_KEY")
                if not k:
                    raise RuntimeError("OPENROUTER_API_KEY 未设置")
                _or_client = OpenAI(base_url=OPENROUTER_BASE, api_key=k,
                                    timeout=600.0, max_retries=0)
            return _or_client
    if model and is_claude(model):
        with _client_lock:
            if _claude_client is None:
                from openai import OpenAI
                k = os.environ.get("ANTHROPIC_API_KEY")
                if not k:
                    raise RuntimeError("ANTHROPIC_API_KEY 未设置")
                _claude_client = OpenAI(base_url=ANTHROPIC_BASE, api_key=k,
                                        timeout=600.0, max_retries=0)
            return _claude_client
    if model and is_gemini(model):
        with _client_lock:
            if _gemini_client is None:
                from openai import OpenAI
                k = os.environ.get("GEMINI_API_KEY")
                if not k:
                    raise RuntimeError("GEMINI_API_KEY 未设置")
                _gemini_client = OpenAI(base_url=GEMINI_BASE, api_key=k,
                                        timeout=300.0, max_retries=0)
            return _gemini_client
    if model and is_local(model):
        url = local_endpoint(model)
        with _client_lock:
            if url not in _local_clients:
                from openai import OpenAI
                _local_clients[url] = OpenAI(base_url=url, api_key="EMPTY",
                                             timeout=600.0, max_retries=0)
            return _local_clients[url]
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(timeout=240.0, max_retries=0)
        return _client


# 32 in-flight produced 429s on gpt-5-nano (8 lost episodes in the pilot); 18 is safe.
_INFLIGHT = threading.Semaphore(int(os.environ.get("LLM_MAX_INFLIGHT", "18")))
_calllog_lock = threading.Lock()
CALL_LOG = LOG_DIR / f"calls_{time.strftime('%Y%m%d')}.jsonl"


def _log_call(rec: dict):
    with _calllog_lock:
        with CALL_LOG.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _cache_key(model, system, messages, temperature, max_tokens, seed, n, effort, jsonmode, tag) -> str:
    payload = json.dumps({"model": model, "sys": system, "msgs": messages, "t": temperature,
                          "mt": max_tokens, "seed": seed, "n": n, "e": effort,
                          "j": jsonmode, "x": tag}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> pathlib.Path:
    d = CACHE_DIR / key[:2]
    d.mkdir(exist_ok=True)
    return d / f"{key}.json"


def chat(model: str, messages: list[dict], system: str | None = None,
         temperature: float | None = None, max_tokens: int = 1200,
         seed: int | None = None, n: int = 1, effort: str | None = None,
         json_mode: bool = False, use_cache: bool = True,
         max_retries: int = 8, tag: str = "") -> dict:
    """Returns {'texts': [str]*n, 'text': texts[0], 'input_tokens', 'output_tokens',
    'reasoning_tokens', 'model', 'cached'}."""
    key = _cache_key(model, system, messages, temperature, max_tokens, seed, n, effort, json_mode, tag)
    cp = _cache_path(key)
    if use_cache and cp.exists():
        try:
            r = json.loads(cp.read_text())
            LEDGER.add(model, r["input_tokens"], r["output_tokens"], cached=True,
                       n=len(r["texts"]), reasoning=r.get("reasoning_tokens", 0))
            r["cached"] = True
            return r
        except Exception:
            pass

    g = 0.0 if is_local(model) else global_spend_usd()
    if g > DAILY_CAP_USD:
        raise BudgetExceeded(f"cross-process day spend ${g:.2f} exceeds cap ${DAILY_CAP_USD}")
    ven = vendor_of(model)
    cap = VENDOR_CAP_USD.get(ven)
    if cap is not None:
        vs = vendor_spend_usd(ven)
        if vs > cap:
            raise BudgetExceeded(
                f"{ven} cumulative spend ${vs:.2f} exceeds cap ${cap:.2f}")

    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    last_err = None
    _grow = 0          # 因思考挤掉输出而加倍预算的次数
    for attempt in range(max_retries):
        try:
            wire = wire_name(model)
            kw: dict[str, Any] = dict(model=wire, messages=msgs)
            # Gemini 的 OpenAI 兼容层不认 seed，也不开 multiple candidates：
            #   seed  -> 400 Unknown name "seed"
            #   n>1   -> 400 Multiple candidates is not enabled for this model
            # 因此 n>1 改为顺序多次采样（温度提供随机性），seed 直接丢弃。
            # 两件事必须分开判定：
            #   多候选(n>1)  —— Gemini/Claude 兼容端点不开，OpenRouter 上 deepseek /
            #                   qwen / z-ai 也一律不支持（2026-09-01 逐个探过），
            #                   统一退化为顺序多次采样，温度提供随机性。
            #   seed         —— Gemini/Claude 不认，但 OpenRouter 上那几家认，
            #                   所以不能跟着 n 一起丢，否则白白损失可复现性。
            multi_n_ok = not (is_gemini(model) or is_claude(model) or is_openrouter(model))
            gem = is_gemini(model) or is_claude(model)
            if n > 1 and multi_n_ok:
                kw["n"] = n
            if is_reasoning(model):
                # reasoning tokens are drawn from the same budget: too small a cap returns "".
                kw["max_completion_tokens"] = max(max_tokens, REASONING_MIN_TOKENS)
                if effort:
                    kw["reasoning_effort"] = effort
            else:
                kw["max_tokens"] = max_tokens
                # Claude 5 系列（fable/opus/sonnet-5）自适应思考常开，只接受
                # temperature=1.0，其他值一律 "`temperature` is deprecated for this
                # model"。Haiku 4.5 仍接受自定义值，所以按端点响应动态降级而不是写死。
                if temperature is not None and not _temp_off.get(model):
                    kw["temperature"] = temperature
                if seed is not None and not gem:
                    kw["seed"] = seed
            # Claude 的 OpenAI 兼容端点只接受 response_format=json_schema，不接受
            # json_object。提示词里已经要求 JSON，parse_opinion 又有正则回退
            # （见 panels/base.py），因此直接不发这个字段。
            if json_mode and not is_claude(model) and not (is_local(model)
                                                           and _json_off.get(model)):
                kw["response_format"] = {"type": "json_object"}
            t0 = time.time()
            if n > 1 and not multi_n_ok:
                texts, inp, out, rtok = [], 0, 0, 0
                for _ in range(n):
                    with _INFLIGHT:
                        rr = _openai(model).chat.completions.create(**kw)
                    texts.append(rr.choices[0].message.content or "")
                    inp += rr.usage.prompt_tokens; out += rr.usage.completion_tokens
                    _ctd = getattr(rr.usage, "completion_tokens_details", None)
                    rtok += (getattr(_ctd, "reasoning_tokens", 0) or 0) if _ctd else 0
            else:
                with _INFLIGHT:
                    resp = _openai(model).chat.completions.create(**kw)
                texts = [(c.message.content or "") for c in resp.choices]
                inp, out = resp.usage.prompt_tokens, resp.usage.completion_tokens
                rtok = 0
                ctd = getattr(resp.usage, "completion_tokens_details", None)
                if ctd is not None:
                    rtok = getattr(ctd, "reasoning_tokens", 0) or 0
            # 思考模型的截断兜底：思考与可见输出共用 max_tokens，思考吃满就返回空串——
            # 这种调用照样按 token 计费却拿不到答案，是纯粹的浪费。这里不靠事先把预算
            # 猜准，而是检测到就加倍重试，最多两次（400 -> 800 -> 1600 这样翻）。
            _blank = sum(1 for x in texts if not x.strip())
            if (_blank and rtok >= 0.7 * max_tokens
                    and _grow < MAX_BUDGET_GROWTH and not is_local(model)):
                _grow += 1
                max_tokens = max_tokens * 2
                print(f"   [预算不足] {model} 思考 {rtok}/{max_tokens//2} tok 挤掉了输出，"
                      f"加倍到 {max_tokens} 重试", flush=True)
                continue

            r = {"texts": texts, "text": texts[0], "input_tokens": inp, "output_tokens": out,
                 "reasoning_tokens": rtok, "model": model, "cached": False}
            if use_cache:
                cp.write_text(json.dumps(r, ensure_ascii=False))
            LEDGER.add(model, inp, out, cached=False, n=len(texts), reasoning=rtok)
            pi, po = ((OPENROUTER_PRICING.get(model) if is_openrouter(model) else None)
                      or PRICING.get(model, (0.0, 0.0)))
            _log_call({"ts": round(t0, 1), "model": model, "tag": tag, "n": n, "seed": seed,
                       "effort": effort, "in": inp, "out": out, "rsn": rtok,
                       "usd": round(inp / 1e6 * pi + out / 1e6 * po, 6),
                       "latency": round(time.time() - t0, 2)})
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # vLLM 端点未必开启 guided decoding；这时退回自由文本，
            # parse_opinion 的正则回退足以处理（见 panels/base.py）。
            # 并发下多个请求会同时撞到这个错误：先返回的那个设标志，后返回的
            # 若还检查 "not _temp_off" 就会落到下面的 raise。所以无条件捕获重试。
            if is_claude(model) and "temperature" in msg and "deprecat" in msg.lower():
                _temp_off[model] = True
                continue
            if is_local(model) and json_mode and not _json_off.get(model) and any(
                    s in msg.lower() for s in ("response_format", "guided", "json_object",
                                               "structured output")):
                _json_off[model] = True
                continue
            if any(s in msg for s in ("invalid_request", "does not exist", "invalid_api_key",
                                      "model_not_found", "Unsupported parameter",
                                      "Unsupported value", "unsupported_value",
                                      "INVALID_ARGUMENT", "Unknown name",
                                      "Multiple candidates is not enabled")):
                raise
            # 每日配额型 429 与限流型 429 必须分开处理：前者退避再多次也没用，
            # 而每次重试仍然计入配额。Gemini 会给出 quotaId 与 retryDelay，
            # 据此立刻放弃，避免像 2026-08-31 那次把几小时和剩余配额烧在重试上。
            if "429" in msg:
                m_delay = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", msg)
                per_day = ("PerDay" in msg or "per_day" in msg
                           or (m_delay and int(m_delay.group(1)) > 900))
                if per_day:
                    raise RuntimeError(
                        f"daily quota exhausted for {model}; not retrying "
                        f"(retry after ~{int(m_delay.group(1))//3600 if m_delay else '?'}h): {msg[:200]}")
            # 限流型 429 需要比普通错误长得多的抖动退避，服务端给的 Retry-After 优先。
            if "429" in msg or "rate limit" in msg.lower():
                wait = 20.0
                m = re.search(r"try again in ([0-9.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 1.0
                time.sleep(min(wait * (attempt + 1) + random.random() * 3, 120))
            else:
                time.sleep(min(2 ** attempt + random.random(), 45))
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")


_MAX_WORKERS = int(os.environ.get("LLM_WORKERS", "8"))


def pmap(fn: Callable, items: Sequence, workers: int | None = None) -> list:
    """Ordered parallel map with a bounded pool."""
    w = workers or _MAX_WORKERS
    if w <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, items))
