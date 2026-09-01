# Consultation Arithmetic — 多智能体医疗会诊的实验代码

> 论文（LaTeX / 图 / 表）在独立仓库：**[hq0709/medical-agent-scaling](https://github.com/hq0709/medical-agent-scaling)**
> 本仓库只放代码、分析与实验文档。

一句话结论：**在医学里，一致就是证据。语言模型面板保留了一致，丢掉了赋予它意义的独立性。**

九人面板在 70.1% 的题目上异口同声，而其中 39.8% 是错的；讨论把一致率推到 98.5%，
一致时的正确率反而降到 54.3%。成因可测：成员两两误差相关 φ = 0.79，九个成员只值
**1.22 个有效独立意见**。

已完成 **420 个多智能体配置 / 144,499 个 episode**，横跨 3 家厂商 7 个模型、
3 个医疗 benchmark、5 种架构、面板规模 N ∈ {1,3,5,7,9}，每个 cell 都配等预算的
单模型对照。多样性分析另加 6 个模型，共覆盖六个独立训练的生态
（OpenAI / Google / Anthropic / DeepSeek / 阿里 / 智谱）。真实 API 花费约 **$579**。

---

## 快速开始

```bash
git clone <this repo> && cd consultation_saturation
pip install -r requirements.txt
cp .env.example .env          # 填入 OPENAI_API_KEY
python3 data/build_datasets.py        # 重建题目文件（本仓库不转载基准数据）
python3 scripts/preflight.py          # 必须全绿再开跑
```

预检覆盖：Python 版本、依赖、密钥、目录、题目条数，以及三个模型的真实冒烟调用。
任何一项红色都不要开始正式实验 —— 详见 `docs/EXPERIMENTS.md`。

---

## 目录

| 路径 | 内容 |
|---|---|
| `common/llm.py` | LLM 调用层：缓存 · 重试 · 计价 · 跨进程日上限 · `n=` 多样本 · 并行 map · **开源权重端点** |
| `panels/` | `roles` 专科路由 · `base` 提示与解析 · `architectures` 五种架构 |
| `experiments/` | 网格运行、各项分析、全部绘图 |
| `mechanisms/` | 协调指标与错误分类 |
| `scripts/` | 预检、启动 vLLM、批量运行、花费核账 |
| `data/build_datasets.py` | 从上游重建三个 benchmark 的 250 题子集 |
| `results/*.json` | 聚合结果（论文正文直接引用的数字） |
| `docs/EXPERIMENTS.md` | **实验文档：已跑什么、下一步跑什么、怎么跑** |

原始 episode（`results/*.jsonl`，353 MB）与 LLM 缓存（`cache/`，441 MB）不入库，
可由代码重建；`.env` 与基准数据同样排除。

---

## 两条实验臂

**闭源臂（已完成）** — OpenAI 的 6 个 checkpoint，横跨 4o / 4.1 / 5 三次独立训练。

**开源权重臂（待跑）** — MedGemma、Lingshu、LLaVA-Med、HuatuoGPT、Qwen2.5，
经 vLLM 的 OpenAI 兼容端点接入，代码里以 `local/<name>` 引用。
目的：把多样性阶梯从「跨家族」推到「**跨生态**」，即由不同机构、不同数据、
不同目标训练出来的模型之间的误差相关性。见 `docs/EXPERIMENTS.md` 第 3 节。

---

## 一个必须知道的成本机关

每个 agent 的第 0 轮意见缓存在 `(item, specialty_i, seed_base, i)` 上，
**不含 N、也不含架构名**。因此 N ∈ {1,3,5,7,9} 只花 9 次意见而不是 25 次，
架构之间大量复用彼此的调用。

> 永远不要把 `N` 或架构名写进第 0 轮的缓存 tag —— 那会静默地把花费翻好几倍。
> 反过来：推理模型会在进缓存键之前丢掉 `seed`，所以 seed **必须**写进 tag，
> 否则多跑的 seed 全是同一份缓存的克隆（实测 seed 1–3 在 gpt-5-nano 上给出 1000/1000 相同预测）。
