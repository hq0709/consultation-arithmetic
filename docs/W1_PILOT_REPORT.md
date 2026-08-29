# W1 Pilot Report — Plan §5 gate

**Run:** 200 MedXpertQA items × {N=1,5,9} × architectures {①independent, ②discussion} × T2
(`gpt-5-nano`, effort=low, temp n/a) × 1 seed, plus all §3.3 baselines. 2,800 episodes, 0 lost.
**Actual spend: $1.70.** (Whole project to date, including 6-model tier probe, difficulty
tagging, θ calibration and a GPT-5.4 code review: **$2.16**.)

## 1. Aggregate result — flat, and *nothing* is significant

| architecture | N=1 | N=5 | N=9 | $/item at N=9 |
|---|---|---|---|---|
| single zero-shot | 33.0% | – | – | $0.00015 |
| single CoT | 32.0% | – | – | $0.00020 |
| ① independent panel (vote) | 34.5% | 34.0% | **32.5%** | $0.00164 |
| ② panel discussion | 34.5% | 36.5% | 35.5% | $0.00322 |
| self-consistency (k = N) | 32.5% | 34.0% | 35.0% | $0.00168 |

Wilson 95% CIs are ±6–7pp and overlap everywhere. **Every paired McNemar test is
Holm-p = 1.00.** At n=200 this pilot cannot resolve the effects in play — which is exactly
what a pilot is for (§3).

Accuracy-per-dollar is strictly ordered and *not* ambiguous: zero-shot 2.26 > single CoT 1.58
> ① N=5 0.38 > ① N=9 0.20 > ② N=9 0.11 > SC k=27 0.07. **Consultation costs 9–21× more per
question and buys nothing measurable at this scale.**

## 2. The finding the aggregate was hiding — the sign of the effect flips with difficulty

Strata from the plan's §3.5 tagger (3 mid-tier models × k=5 pass rate):
easy n=13, medium n=62, hard n=125.

| stratum | ① N=1 | ① N=5 | ① N=9 | gain N=9 vs N=1 |
|---|---|---|---|---|
| easy | 84.6% | 100.0% | 100.0% | **+15.4pp** |
| medium | 72.6% | 75.8% | 75.8% | +3.2pp |
| **hard** | **10.4%** | **6.4%** | **4.0%** | **−6.4pp (p=0.04)** |

This is the **only** nominally significant contrast in the entire pilot, and it is a *harm*.
Adding independent experts degrades hard-case accuracy monotonically. Two further structures:

- **Discussion protects against the harm that voting causes.** On hard items ② holds
  10.4 → 10.4 → 8.8% where ① collapses 10.4 → 6.4 → 4.0%. Majority voting amplifies
  correlated errors; letting a correct minority argue back partly prevents that.
- **On hard items every method is at or below the 10% random baseline** for 10-option MCQ
  (① N=9 reaches 4.0%). This is systematic distractor attraction, not noise-limited guessing.

At round 0, **48.5% of 5-expert panels are already unanimous** while panel accuracy is 34% —
unanimity is not evidence of correctness. That is the correlated-error mechanism the paper
needs, visible in the pilot.

> ⚠️ **Stratum confound to fix before the full grid.** The difficulty tagger includes
> `gpt-5-nano`, the model being evaluated, so the "hard" stratum is partly *defined* by this
> model failing. This inflates the *level* of the hard-stratum harm. It does **not** explain
> the *slope*: N=1 and N=9 are compared within the same items. Fix: tag difficulty with models
> disjoint from the evaluation ladder, or leave-one-out.

## 3. Power — what the full grid actually needs

Median discordance is only **8.5%** because the nested design couples panels (common random
numbers). That is the variance reduction the reviewer credited, and it makes the paired tests
far more efficient than unpaired ones.

| items | minimum detectable difference (80% power) |
|---|---|
| 200 (this pilot) | 5.7 pp |
| 500 | 3.6 pp |
| **1,000** | **2.6 pp** |
| 2,000 | 1.8 pp |

Required n at the observed effect sizes:

| contrast | observed | n needed |
|---|---|---|
| ② discussion vs ① independent @ N=9 | +3.0pp | **521** |
| ① independent N=9 vs $-matched SC k=9 | −2.5pp | **563** |
| ② discussion vs ① independent @ N=5 | +2.5pp | **688** |
| SC k=9 vs k=1 | +2.5pp | 1,191 |
| ② N=5 vs N=1 | +2.0pp | 1,567 |
| ① N=9 vs N=1 | −2.0pp | 2,352 |

**Architecture contrasts are affordable (~500–700 items); N-scaling contrasts inside one
architecture are not (1,500–27,000).** The headline dose–response curve is only resolvable by
pooling all three benchmarks × 3 tiers × 3 seeds in the mixed-effects model — which is exactly
what plan §4.5 specifies. Per-benchmark N-curves will be descriptive, not individually powered:
say so in the paper rather than over-claiming.

## 4. Calibrated budget for the full grid

Measured, not guessed: $0.0085/item for the pilot's 14 cells; adding N=3,7, tiered ③ and
debate ④ brings a full tier-seed pass to **$0.021/item**.

| Scope | Est. cost | Est. wall-clock |
|---|---|---|
| **A. Full plan §3.1 grid** — N{1,3,5,7,9} × arch{①②③④} × T1,T2 × 3 benchmarks (2,394 items) × 3 seeds, + T3 slice + heterogeneity + temperature | **~$320** | ~120 h |
| **B. Plan §6 degraded grid** — drop N=7, run ④ at N=5 only, everything else identical | **~$190** | ~50 h |
| C. Powered-contrast-only — the six contrasts above at 1,000 items, T2+T3, 3 seeds | ~$60 | ~15 h |

All three are far under the plan's $1,500–3,000 estimate: nested caching plus the cheap OpenAI
ladder bought roughly a 5–10× reduction. **Wall-clock, not dollars, is the binding constraint**
— gpt-5-nano tops out near 250 calls/min before 429s (8 episodes were lost to this in the
pilot and recovered by the resume pass).
