"""NMI 的四类错误分类，落到医学 MCQA 上（NMI §4.4 Error Taxonomy）。

NMI 四类：Logical Contradiction / Numerical Drift / Context Omission /
Coordination Failure(MAS-specific)。我们保留同样的四类，给出多选题上可自动检测的
医学对应，并明确每条检测规则。
"""
from __future__ import annotations
import re, collections, math

_NUM = re.compile(r"\d+(?:\.\d+)?")
_STOP = set("the a an of and or to in for with on at by is are was were be been being that this "
            "these those it its as from than then which who whom what when where how why not no "
            "patient most likely best next step should would could may can his her their".split())


def _content(txt: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", str(txt).lower()) if w not in _STOP}


def logical_contradiction(op, item) -> bool:
    """理由所论证的选项与最终答案不一致。"""
    ans = op.get("answer")
    if ans not in item["options"]:
        return False
    r = _content(op.get("reason", ""))
    if len(r) < 4:
        return False
    ov = {k: len(r & _content(v)) for k, v in item["options"].items()}
    best = max(ov, key=ov.get)
    return best != ans and ov[best] >= ov[ans] + 2


def numerical_drift(op, item) -> bool:
    """理由中出现题干与选项都没有的数值（剂量/阈值臆造）。"""
    stem_nums = set(_NUM.findall(item["stem"])) | {
        n for v in item["options"].values() for n in _NUM.findall(str(v))}
    r_nums = set(_NUM.findall(str(op.get("reason", ""))))
    novel = {n for n in r_nums if n not in stem_nums and float(n) not in (0, 1, 2)}
    return len(novel) >= 2


def context_omission(op, item) -> bool:
    """理由完全没有引用题干的判别性内容词。"""
    r = _content(op.get("reason", ""))
    if len(r) < 3:
        return True
    return len(r & _content(item["stem"])) == 0


def coordination_failure(ep) -> bool:
    """MAS 特有：第 0 轮多数已正确，最终却输出错误（协作把对的改错）。"""
    rounds = ep.get("rounds") or []
    if len(rounds) < 2:
        return False
    gold = ep.get("gold")
    first = [o.get("answer") for o in rounds[0]]
    c = collections.Counter([a for a in first if a])
    if not c:
        return False
    top, n = c.most_common(1)[0]
    return top == gold and n > len(first) / 2 and ep.get("pred") != gold


def classify(eps, items_by_qid):
    out = collections.Counter(); tot = 0
    for ep in eps:
        it = items_by_qid.get(ep["qid"])
        if not it:
            continue
        tot += 1
        r0 = (ep.get("rounds") or [[]])[0]
        if any(logical_contradiction(o, it) for o in r0):
            out["logical_contradiction"] += 1
        if any(numerical_drift(o, it) for o in r0):
            out["numerical_drift"] += 1
        if any(context_omission(o, it) for o in r0):
            out["context_omission"] += 1
        if coordination_failure(ep):
            out["coordination_failure"] += 1
    if not tot:
        return {}
    d = {k: v / tot for k, v in out.items()}
    d["n"] = tot
    return d


def information_gain(ep, gold=None):
    """NMI Appendix D.5 的精确定义（贝叶斯后验方差缩减），不是香农熵减：

        DeltaI = 0.5 * log( Var[Y | s_pre] / Var[Y | s_post] ),   Var[Y|s] = p(s)(1-p(s))

    Y in {0,1} 是任务成功指示；s_pre 是协调前状态、s_post 是协调后状态。NMI 用 K=10 条
    温度 0.7 的推理轨迹做蒙特卡洛估计 p(s)。在 MCQA 上我们已经天然有一组样本：panel 的
    N 个意见就是 N 条独立轨迹，p(s) = 该轮给出正确答案的比例。这避免了额外采样成本，
    且与他们的估计量同构。
    """
    rounds = ep.get("rounds") or []
    if len(rounds) < 2:
        return None
    g = gold if gold is not None else ep.get("gold")

    def var(ops):
        a = [o.get("answer") for o in ops if o.get("answer")]
        if not a:
            return None
        k = sum(1 for x in a if x == g)
        # Laplace 平滑 (k+1)/(n+2)。不能用硬夹逼：Centralized 的末轮只有 orchestrator
        # 一条意见，夹逼会把 p 恒定压到 0.5、Var 恒为 0.25，使 DeltaI 结构性为负。
        p = (k + 1) / (len(a) + 2)
        return p * (1 - p)
    v0, v1 = var(rounds[0]), var(rounds[-1])
    if not v0 or not v1:
        return None
    return 0.5 * math.log(v0 / v1)


def answer_entropy_drop(ep):
    """辅助量：答案分布的香农熵减 (bits)。与 information_gain 一起报告。"""
    rounds = ep.get("rounds") or []
    if len(rounds) < 2:
        return None

    def H(ops):
        a = [o.get("answer") for o in ops if o.get("answer")]
        if not a:
            return 0.0
        c = collections.Counter(a); n = len(a)
        return -sum((v / n) * math.log2(v / n) for v in c.values())
    return H(rounds[0]) - H(rounds[-1])
