# 实验文档

换服务器后按本文操作。**第 1 节必须全绿再进第 2 节。**

---

## 1. 落地检查

```bash
pip install -r requirements.txt
cp .env.example .env && $EDITOR .env       # 填 OPENAI_API_KEY 与 DAILY_CAP_USD
python3 data/build_datasets.py             # 重建题目（本仓库不转载基准数据）
python3 scripts/preflight.py
```

预检的每一项以及失败时的含义：

| 检查 | 失败通常意味着 |
|---|---|
| python 版本 | 需要 3.10+ |
| 依赖 | `pip install -r requirements.txt` 没跑 |
| `.env` | 文件不存在，或 key 名写错 |
| 题目文件 | `build_datasets.py` 没跑，或上游数据源变了（三个文件必须各 **250** 条） |
| 冒烟测试 | key 无效 / 无该模型权限 / **推理模型返回空串**（见下） |

> **推理模型返回空串**是这个项目最容易踩的坑：gpt-5 系列的推理 token 与输出 token
> 共用预算，`max_tokens` 给小了会把额度全部用在推理上、返回 `""`。
> `REASONING_MIN_TOKENS=1600` 是实测下限，不要调低。

跑完后核对花费口径：

```bash
python3 scripts/true_spend.py      # 从逐次调用日志重算，不信任聚合计数器
```

---

## 2. 闭源臂（已完成，仅需复现时执行）

已产出 `results/G_{T1,T2,T3}_{medxpertqa,medagentsbench,medqa}.jsonl`，
共 180 个多智能体配置 / 63,499 episode，真实花费 **$39**。

```bash
bash scripts/run_nmi_grid.sh            # 完整网格；有缓存时大部分是命中
```

三个能力档：

| 档 | 模型 | effort | 能力指数 I | MedXpertQA |
|---|---|---|---|---|
| T1 | `gpt-4.1-nano` | — | 34.0 | 15.6% |
| T2 | `gpt-5-nano` | low | 50.8 | 32.0% |
| T3 | `gpt-5-mini` | low | 59.2 | 41.6% |

并发上限：`LLM_MAX_INFLIGHT=18`。实测 32 并发会在 gpt-5-nano 上触发 429 并丢 episode。

---

## 3. 开源权重臂（待跑）

### 目的

把多样性阶梯从「跨家族」推到「**跨生态**」。当前论文 Table 5：

| 多样性来源 | φ | φ/φ_max | N_eff@9 |
|---|---|---|---|
| 同模型 + 不同专科提示 | 0.734 | 0.766 | 1.31 |
| 不同 checkpoint · 同家族 | 0.437 | 0.536 | 2.00 |
| 不同 checkpoint · 跨家族（4o/4.1/5） | 0.308 | 0.450 | 2.60 |
| *完全独立* | 0 | 0 | 9.00 |

开源模型由**不同机构、不同数据、不同目标**训练，是比 OpenAI 内部三家族更彻底的
跨生态。要回答的问题：**离开一个生态之后，φ/φ_max 会掉到 0.45 以下吗？**

### 关键设计要求：能力匹配

**这一条决定实验成不成立。** 两个二元错误指示变量在边缘分布不同时，φ 有数学上限

    φ_max = sqrt( p(1-q) / (q(1-p)) ),   p ≤ q 为两者错误率

所以把一个强模型和一个弱模型放在一起，原始 φ 会**机械地**变低，与推理是否真的
不同无关。论文里已经证明：控制住错误率差之后，「混能力能去相关」这个效应**完全消失**
（β 从 −0.601 变成 +0.156，置信区间跨零）。

因此：
- 分析**必须**看 `φ/φ_max`，不能看原始 φ；`experiments/phi_decomposition.py` 已经这么做
- 最有价值的配对是**能力相当的跨生态对**。先测各开源模型的准确率，再挑与
  OpenAI 阶梯（15.6 / 32.0 / 41.6 on MedXpertQA）最接近的去配对

### 步骤

**(a) 起端点**，每个模型一个：

```bash
bash scripts/serve_local.sh medgemma-4b 8010 0        # <名字> <端口> <GPU>
bash scripts/serve_local.sh lingshu-7b  8011 1,2      # 逗号分隔 = 张量并行
```

显存：7B bf16 约需 15 GB 权重 + KV 缓存，24 GB 单卡够用；不够就用两张卡（TP=2）
或换 AWQ/GPTQ 量化权重。

**(b) 注册到 `.env`**（键名必须与 `--served-model-name` 完全一致）：

```
LOCAL_MODELS=medgemma-4b=http://localhost:8010/v1,lingshu-7b=http://localhost:8011/v1
```

**(c) 验证**：

```bash
python3 scripts/preflight.py --local
```

这一步会真的发一条临床问题过去并跑 `parse_opinion`。**解析失败是这条臂的头号风险**：
4–7B 的模型未必遵守 JSON 格式。解析器有正则回退（`answer is X` / `(X)`），
但如果预检报解析失败，先手工看 10 条原始输出再决定改提示还是换模型。

**(d) 跑**：

```bash
bash scripts/run_open_models.sh                       # 默认全部模型 × 全部 benchmark
MODELS="medgemma-4b lingshu-7b" bash scripts/run_open_models.sh   # 只跑指定的
```

产出 `results/OPEN_<model>_<bench>.jsonl`，每个 250 条。只跑单医生基线
（`--arches cot --Ns 1`）—— 算两两 φ 不需要真的组面板，这是最省的做法。

**(e) 分析**：

```bash
python3 experiments/phi_decomposition.py
```

会自动吸收 `OPEN_*.jsonl`，并在有跨生态数据时自动在回归里加入「同生态」项。
输出 `results/phi_decomposition.json`。

### 预期规模

| 项 | 量 |
|---|---|
| 生成次数 | 模型数 × 3 benchmark × 250 题 |
| GPU 时间 | 7B 模型每个 benchmark 约 20–40 分钟（vLLM 批处理，`--workers 16`） |
| 美元成本 | **0**（本地模型不计入日上限，但 token 仍照常记账） |

### 已在本地缓存的模型

`microsoft/llava-med-v1.5-mistral-7b` · `google/medgemma-4b-it` ·
`FreedomIntelligence/HuatuoGPT-Vision-7B` · `Qwen/Qwen2.5-7B-Instruct`
（Lingshu 需另行下载。）

> **注意**：LLaVA-Med 与 HuatuoGPT-Vision 是视觉语言模型，而本项目三个 benchmark
> 全是纯文本。它们能接受纯文本输入，但 LLaVA-Med 是 2023 年的 Mistral-7B 微调，
> 在 10 选项的 MedXpertQA 上很可能接近随机（10%）。接近随机的成员会把原始 φ 压低，
> 这**正是上面说的 φ_max 假象**——所以务必看归一化值，并优先做能力匹配的配对。

---

## 4. 分析脚本一览

跑完实验后，下面这些脚本产出论文里的每一个数字：

| 脚本 | 产出 |
|---|---|
| `experiments/analyze.py` | 主结果表、显著性检验 |
| `experiments/independence.py` | φ、N_eff、四个可证伪预测 |
| `experiments/phi_decomposition.py` | φ_max 修正、家族/生态/能力分解 |
| `experiments/aggregation_ceiling.py` | 意见对四分解、预言机上限、捕获率 |
| `experiments/ceiling_numbers.py` | 全文引用的权威数字（**口径唯一来源**） |
| `experiments/selection_signals.py` | 五种路由信号的判别力 |
| `experiments/window_boundary.py` | 窗口边界的可识别性 |
| `experiments/scaling_law.py` | 标度律拟合、决策边界 |
| `experiments/robustness.py` | 自助、正态性、异方差、嵌套模型 |
| `experiments/fig_*.py` | 全部图（样式统一在 `experiments/vizstyle.py`） |

> 引用任何「可用空间 / 捕获率」的数字时，一律以 `ceiling_numbers.py` 的输出为准。
> 这些量对预言机的定义敏感，曾经因为三处用了不同定义而出现过 25% / 11–43% / 17% 的矛盾。

---

## 5. 复现校验

改动代码后，用下面这条确认没有把论文里的数字改坏：

```bash
python3 experiments/ceiling_numbers.py     # 单医生 48.0 / 预言机 60.7 / 最好架构 50.0
python3 experiments/independence.py        # phi 0.734（不含 Hybrid）
python3 experiments/aggregation_ceiling.py # 一致·都错 40.3%
```

图的字号必须按最终排版宽度出图：ACL 双栏 `\columnwidth`=3.15in、`\textwidth`=6.30in。
出大图再让 LaTeX 缩会把图内字号压到 4–5pt（正文是 10pt）。
