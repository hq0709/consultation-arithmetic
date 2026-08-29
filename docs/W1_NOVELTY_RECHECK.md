# W1 Day-1 Mandatory Novelty Re-check — Project 4 (Consultation Saturation)

**Date:** 2026-08-28 · **Verdict: PROCEED.** No medicine-specific N-scaling / dose–response
curve paper found. The general-domain space has tightened; the medical lane is still open.

## Queries run (per plan §0)
`number of agents scaling medical diagnosis` · `multi-agent saturation clinical LLM` ·
`team size LLM consultation medicine` · `collaboration threshold medical` ·
`dose response agents diagnosis` · plus the current TeamMedAgents version.

## What is now taken (must be cited, must not be re-claimed)

| Work | What it establishes | Why we are still distinct |
|---|---|---|
| **Scaling Behavior of Single LLM-Driven MAS** ([2606.00655](https://arxiv.org/abs/2606.00655)) | **Inverted-U with agent count** — coordination overhead dominates past an optimum. General domains. | Our inverted-U claim must be framed as *medicine-specific transfer + difficulty stratification*, not as discovery of the shape. |
| **Understanding Agent Scaling via Diversity** ([2602.03794](https://arxiv.org/abs/2602.03794)) | "fast-then-slow"; 2 diverse agents ≈ 16 homogeneous. Diversity is the mechanism. | Directly pre-empts a naive version of our RQ5. We must run the *within-vendor family* heterogeneity arm and the discussion-round diversity-collapse trajectory, which they do not. |
| **Capable LMs outgrow collaboration** (NMI 2026, [s42256-026-01268-y](https://www.nature.com/articles/s42256-026-01268-y)) | ~45% solo-success collaboration threshold, 1–9 agents. **Medicine excluded.** | RQ2 is exactly the medical transfer test. Unchanged. |
| **TeamMedAgents v3** ([2508.08115v3](https://arxiv.org/abs/2508.08115), Mar 2026, retitled *Pareto-Efficient Multi-Agent Medical Reasoning*) | Now sweeps **team sizes 2–5** and adds a Pareto/efficiency framing. | Closer than v1 was. Our N reaches 9 (+15 far end), covers 4 architectures, and adds the budget-matched self-consistency control they do not run. **Cite as the closest medical prior; never claim "first team-size ablation in medicine" — claim first *curve* (N≥7) and first *saturation mechanism*.** |
| **Orchestrated multi-agents under clinical-scale workloads** ([s44401-026-00077-0](https://www.nature.com/articles/s44401-026-00077-0)) | Multi-agent degrades more slowly than single-agent as *task volume* scales. | Different axis (workload, not panel size). Complementary. |
| **ConfAgents** ([2508.04915](https://arxiv.org/abs/2508.04915)) | Conformal-guided cost-efficient medical multi-agent; notes ~50× time overhead for marginal gain. | Supports RQ4's economics premise; it optimizes, we chart. |

## Positioning adjustments forced by this re-check
1. **Drop any "we discover non-monotonicity" phrasing.** 2606.00655 owns the inverted-U in general
   domains. Our headline is the *medical* dose–response curve + *who* it inverts for (hard stratum)
   + *why* (clinical panel mechanisms) + *what it costs*.
2. **Elevate the budget-matched self-consistency control** (§3.3) to a co-headline. Nothing found
   runs it in medicine; it is the sharpest surviving blade.
3. **Elevate tiered referral (③)** — referral appropriateness and triage cost-saving are clinical
   endpoints absent from every paper above.
4. Re-run all queries the week before submission (plan §0).

---
# W1 Calibration Finding — the tiered-referral gate does not work as specified

`scripts/calibrate_theta.py --items data/medxpertqa_dev100.jsonl --model gpt-5-nano --effort low`
(100 dev items, disjoint from every evaluation sample; $0.019)

Generalist accuracy **30.0%**. Self-reported confidence is clustered in 70–89 and is **almost
uninformative about correctness**:

| confidence bin | n | accuracy |
|---|---|---|
| 60–69 | 12 | 25.0% |
| 70–79 | 32 | 31.2% |
| 80–89 | 49 | 32.7% |
| 90–99 | 4 | 25.0% |

Best achievable referral gate: **Youden J = 0.071** (θ=70, refer 15%, TPR 0.171, FPR 0.100).
At the ~50%-referral operating point (θ=80): J = 0.052.

**Consequence for architecture ③.** The plan's "referral appropriateness" endpoint has a near-null
answer *before the panel is even involved*: a confidence-triggered gate cannot select the cases the
generalist gets wrong. Frozen decision: **θ = 80** (the operating point that maximises J among
thresholds that actually exercise the architecture, referral rate 47%). The full calibration curve
is reported; the near-chance gate is a result, not a bug, and it sharpens the paper's claim —
*any benefit of tiered referral comes from the panel, not from the triage.*

Not added to the run (would be a new experiment beyond the plan; flagged for the user):
an **agreement-based** gate — sample the generalist k=3 and refer when not unanimous — is the
obvious repair and would likely make ③ work. Recommend adding it as a §3.2③ variant.
