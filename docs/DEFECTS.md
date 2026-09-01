# 全文缺陷清单（2026-09-01 审计）

> **甲（数据）已全部修完，见文末「修复记录」。乙丙丁（结构 / 核心观点 / 句子）未动，
> 等标题与核心观点定下来之后一起做。**

## 甲 · 数据（先修这个，否则改写作是白改）

**A1 捕获率 κ 有三套互相矛盾的数字，且论文用的是孤儿文件。**
`results/capture_by_arch.json` 停在 8-29（180 配置网格），已经没有任何脚本生成它。
正文 §4.6、结论、Fig.10 图注引用的一套，连这个旧文件都对不上：

| 架构 | 论文正文 | 旧 JSON(8-29) | 新 JSON(9-01, 420配置) |
|---|---|---|---|
| Independent | 0.6 | 1.81 | **−12.43** |
| Self-consistency | 4.7 | 4.67 | 3.74 |
| Centralized | 6.5 | 0.59 | 4.15 |
| Decentralized | 8.6 | 9.66 | 3.69 |
| Hybrid | 8.8 | 1.19 | **−5.79** |

新网格上两个架构的符号翻转成负数。"no aggregation rule reaches 9" 这句在新数据下是错的说法——
真实情况更强："两个架构是负的"。第二根支柱现在建在错数字上。

**A2 Figure 10 的柱子和它自己的图注矛盾。**
`fig_ceiling.py` 读 `headroom_decomp.json`（8-29：single 48.0 / sc_oracle 59.69 / panel_oracle 60.71），
图注写的是 62.1 / 62.2 / 单医生 52.9。图画的是旧网格，字写的是新网格。
"99% 是采样、1% 是专科名册"这个结论句直接建在这张图上。

**A3 一致性数字两套并存。**
摘要 / 引言 / README：70.1% 一致率、错 39.8%（= independent r0 N=9）。
§4.7：66.3% → 98.3%，P(错|一致) 46.5%（= 全配置合并）。
同一个命名量两个值，读者无法调和。一致率还有 98.5 / 98.3 / 98 三种写法。

**A4 六个结果文件停留在 180 配置网格，其中五个是孤儿。**
生成它们的分析脚本已经不在仓库里了 —— 分析流水线重写过（`ceiling_numbers.py` 等），
旧的生产者被删掉，输出留在原地，论文继续引用。无生产者的有：
`capture_by_arch` `confidence_discrimination` `panel_vs_sc_oracle` `cost_normalized`
`gain_by_config`；`headroom_decomp` 只被 `fig_ceiling.py` 读、没人写；
`robustness.py` 还在，重跑即可。
这些分析全部只读 `results/*.jsonl`，不碰 API，重算免费。

`confidence_discrimination` `panel_vs_sc_oracle` `headroom_decomp` `cost_normalized`
`robustness` `gain_by_config` 全部 8-29 或更早。
安全节（86.0/82.6/AUC 0.597）、成本节、稳健性节整段建在旧网格上。

## 乙 · 结构

**B1 结果节的骨架是别人的，不是我们论点的。**
7,246 词、30 个 `\paragraph`。最强的三节（§4.5 前提 / §4.6 天花板 / §4.7 安全）排在第 4、5、6 位，
前面近 3,000 词是从 NMI 复现继承的 scaling-law 拟合、coordination efficiency、information transfer。
那是复现别人框架时的骨架。读者读到我们真正的论点时已经走了。

**B2 全文最好的一句话在结论里。**
"A consultation is worth convening only if somebody in the room is right when the others are wrong,
and the room can tell who." 这是整篇的骨架，却出现在第 6 节。

**B3 引言没有路线图段落。** 对标文章有，且明确交代每节讲什么。

**B4 §3.3 定义七个量，五个在论证里几乎不出现。**
turns / message density / redundancy / overhead / coordination efficiency / error absorption。
真正承载论证的只有 φ、N_eff、κ。对标文章只定义它要用的。

**B5 结论引入正文没有的结果。** "capability 放在推理 agent 上比放在监督 agent 上重要 4.1×"。

## 丙 · 核心观点与命名

**C1 现象没有名字。** 对标文章给了 "lost in conversation"，标题、现象名、结论句同一串词。我们没有。

**C2 核心观点全文四种说法。** 摘要、引言、结论、README 各写一套，没有一句是重复出现的。

**C3 标题。** "Rethinking" 是弱开场，承诺的是"重新考虑"而不是一个发现；agent 不突出。

**C4 我今天写的 hollow consensus 只到第二层。** 见下面"核心观点的四层"。

## 丁 · 句子

**D1 文学性倒装难以复述。** "supplies the form of that evidence and none of its content"、
"not a disappointment but a hazard"。漂亮，但读者复述不出来。对标文章的句子是
"All LLMs exhibit very high unreliability in multi-turn settings, regardless of aptitude."

**D2 Related work 是引用堆砌不是叙事。** 847 词四段，其中一句 60 词塞了 6 个引用。
对标文章的 related work 讲的是一段历史。

**D3 机制没有枚举。** 对标文章用 (1)(2)(3)(4) 把四个机制摆出来。我们埋在从句里。


---

# 修复记录（2026-09-01）

## 甲全部完成，并在过程中查出四个清单上没有的问题

| | 问题 | 处理 |
|---|---|---|
| A1 | 捕获率 κ 三套数字 | 全文改引 `ceiling_numbers.json`。新网格上 Independent **−12.4**、Hybrid **−5.8**，两个架构低于「随机挑一个成员」。图注 "no rule reaches 9" 改为 "no rule reaches 5, and two fall below zero" |
| A2 | Fig.10 图与图注不同源 | headroom 分解并入 `ceiling_numbers.py`，`fig_ceiling.py` 改读同一份。`62.1 / 62.2 / +0.06pp` 在新网格上完全复现，所以图注本来就是对的，错的是图 |
| A3 | 一致性数字两套 | 全文统一到 **N=9**。摘要/引言 98.5→98.3，§4.7 从 N≥3 改为 N=9（69.5% → 98.3%，P(错\|一致) 39.7% / 45.7%） |
| A4 | 六个结果文件停在 180 网格 | 新建 `experiments/refresh_stale.py` 重算置信度判别力、一致率、成本三项；七个孤儿 JSON 移入 `_superseded/` |

### 过程中新查出的

**N1 「讨论把置信差距压缩 45%」不成立。** 在旧的 9 文件网格上是 42%（复现得到 3.44→2.01），
但那是 **OpenAI 专有效应**：gpt 三个模型压缩 59–78%，Google 与 Anthropic 全部为负（讨论反而
拉大差距）。七模型 N=9 合并后是 3.1→2.4。已改写 §4.7、结论、ethics、Fig.9 图注。

**N2 `robustness.py` / `make_appendix_tables.py` / `specialty_match.py` 还在用 `glob("G_*.jsonl")`。**
它们只看得见旧的 9 个文件，扩网格之后数字纹丝不动。已切到 `main_grid()`。
（`phi_decomposition.py` 的宽 glob 是故意的 —— 它要跨全部厂商配对，保持不动。）

**N3 Hybrid 转诊率 12.4% 是旧数，实际 20.0%。** 最高的 cell 也从
「gpt-5-nano / MedAgentsBench 40.4%」变成「claude-sonnet-5 / MedXpertQA 76.8%」。

**N4 Fig.9 与 tab:safety 用了两个不同的置信度定义**（图取最后一个 ≥2 人的轮次，表取系统
真正输出的那一轮）。Centralized 差 0.4 点。统一到后者 —— 临床医生看到的是编排者的裁决。

## 已核对为**当前**、无需改动的

`tab:neff`（φ=0.794 / 0.990、N_eff 1.16–1.43）· `tab:signals`（AUC 0.570–0.604）·
`tab:ceiling`（四个 P_SA 分箱）· `tab:ppd`（九行全部精确复现）· 稳健性全段（σ̂、Lasso、
嵌套 CV R²、panel size −0.006）· 多样性阶梯（2.60 / 2.29 / 2.31 / 2.02）· 规模（21 文件 /
7 模型 / 144,499 episodes / 420 配置）


---

# 第二轮：删掉站不住的分析（2026-09-01，用户："不好的部分的数据最好别留着"）

**N5 `make_paper_tables.py` 也是 `--glob "G_*.jsonl"` 默认。** 第四个。协调指标表因此只覆盖
三个 OpenAI 模型（k=45），而 §4.4 正文把这些数字当成全局事实陈述。已切到 `main_grid()`
（k=105）。随之更新：轮数 7.5/6.5/1.9 → 6.8/6.2/2.1、幂律 0.85/0.770/0.299 → 0.84/0.777/0.327、
消息密度 0.462−0.045ln(c)/R²0.056 → 0.504−0.028ln(c)/R²0.025、Kendall 两两 (−0.200,+0.400,0)
→ (0,+0.200,0)、每千 token 成功数与 4.3× → 3.7×、冗余度分解 0.426/0.377/0.227 →
0.392/0.351/0.206。

**N6 删掉「误差吸收率」。** 代数上
`absorb = (E_SAS − E_MAS)/E_SAS = 增益 / (1 − P_SA)` —— 它是准确率增益的单调重标，
不是一个独立的协调测量。论文把它列进 §3.3 的七个量、在 §4.4 单独报一段、还在 Fig.6 占一格，
等于把同一个数报三遍并从中读出第二个"发现"。Independent 在新网格上是 −0.1%，
Fig.6 那一格的 `ylim(0,27)` 还会把负值直接切掉。已从 §3.3、§4.4、Fig.6 全部移除
（Fig.6 从 2×2 改为 1×3）。

**N7 删掉「信息增益」。** 附录定义的是后验方差缩减，但报出来的三个数与该定义在两个网格上
都对不上。Hybrid 的 +1.46 精确复现为「旧 9 文件网格上的**熵下降**」—— 另一个量。
Decentralized 的 +0.90 和 Centralized 的 +0.24 在任何组合下都复现不出来，而且 Centralized
根本没有 `entropy_final` 字段，熵版本对它不存在。按定义重算是 +0.19 / +0.17 / −0.23，
但 Centralized 的值由「末轮只有编排者一条意见」结构性决定（附录自己就警告过这点），
与其余架构不可比。整段与附录 A.5 一并删除，§4.4 标题去掉 "and information transfer"。
