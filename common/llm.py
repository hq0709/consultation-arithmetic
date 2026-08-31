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
    return (not is_local(model)) and (not is_gemini(model)) \
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
                pi, po = PRICING.get(model, (0.0, 0.0))
                delta = inp / 1e6 * pi + out / 1e6 * po
                u.cost_usd += delta
        if delta:
            record_global_spend(delta)

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

# --- cross-process daily spend journal (a per-process ledger is not a daily cap) ---
_SPEND_FILE = RESULTS_DIR / "daily_spend.json"
_spend_lock = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def global_spend_usd() -> float:
    try:
        return float(json.loads(_SPEND_FILE.read_text()).get(_today(), 0.0))
    except Exception:
        return 0.0


def record_global_spend(amount: float) -> float:
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
_local_clients: dict[str, Any] = {}
_json_off: dict[str, bool] = {}   # 端点拒绝 response_format 时记下，之后不再尝试
_client_lock = threading.Lock()


def _openai(model: str | None = None):
    """按模型名路由：OpenAI 官方 / Gemini 兼容端点 / 本地 vLLM 端点。"""
    global _client, _gemini_client
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

    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    last_err = None
    for attempt in range(max_retries):
        try:
            wire = model[len(LOCAL_PREFIX):] if is_local(model) else model
            kw: dict[str, Any] = dict(model=wire, messages=msgs)
            # Gemini 的 OpenAI 兼容层不认 seed，也不开 multiple candidates：
            #   seed  -> 400 Unknown name "seed"
            #   n>1   -> 400 Multiple candidates is not enabled for this model
            # 因此 n>1 改为顺序多次采样（温度提供随机性），seed 直接丢弃。
            gem = is_gemini(model)
            if n > 1 and not gem:
                kw["n"] = n
            if is_reasoning(model):
                # reasoning tokens are drawn from the same budget: too small a cap returns "".
                kw["max_completion_tokens"] = max(max_tokens, REASONING_MIN_TOKENS)
                if effort:
                    kw["reasoning_effort"] = effort
            else:
                kw["max_tokens"] = max_tokens
                if temperature is not None:
                    kw["temperature"] = temperature
                if seed is not None and not gem:
                    kw["seed"] = seed
            if json_mode and not (is_local(model) and _json_off.get(model)):
                kw["response_format"] = {"type": "json_object"}
            t0 = time.time()
            if gem and n > 1:
                texts, inp, out, rtok = [], 0, 0, 0
                for _ in range(n):
                    with _INFLIGHT:
                        rr = _openai(model).chat.completions.create(**kw)
                    texts.append(rr.choices[0].message.content or "")
                    inp += rr.usage.prompt_tokens; out += rr.usage.completion_tokens
            else:
                with _INFLIGHT:
                    resp = _openai(model).chat.completions.create(**kw)
                texts = [(c.message.content or "") for c in resp.choices]
                inp, out = resp.usage.prompt_tokens, resp.usage.completion_tokens
                rtok = 0
                ctd = getattr(resp.usage, "completion_tokens_details", None)
                if ctd is not None:
                    rtok = getattr(ctd, "reasoning_tokens", 0) or 0
            r = {"texts": texts, "text": texts[0], "input_tokens": inp, "output_tokens": out,
                 "reasoning_tokens": rtok, "model": model, "cached": False}
            if use_cache:
                cp.write_text(json.dumps(r, ensure_ascii=False))
            LEDGER.add(model, inp, out, cached=False, n=len(texts), reasoning=rtok)
            pi, po = PRICING.get(model, (0.0, 0.0))
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
            # 429s need a much longer, jittered backoff than transient errors, and the
            # server's own Retry-After beats any guess.
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
