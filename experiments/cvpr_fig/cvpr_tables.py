"""Supplementary tables of the CVPR version: one table per experiment, never split.

  python experiments/cvpr_fig/cvpr_tables.py      # -> paper/cvpr/sec/supp_tables.tex

Reads results/imaging_v2_{A,B,C,M,D}.json (experiments/imaging_v2_analysis.py). CVPR vocabulary
(agents, Independent / Decentralized MAS, perception and reasoning modules, in-context examples).
Cells print point estimates only (the paper shows no intervals). Table S1 (validation)
is written by hand in sec/X_suppl.tex, so these tables start at S2.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
J = {k: json.loads((R / f"imaging_v2_{k}.json").read_text()) for k in "ABCMD"}
A, B, C, M, D = (J[k] for k in "ABCMD")
MODELS = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001", "qwen3.8-flash"]
API3 = MODELS[:3]
NAME = {"gpt-5-mini": "GPT-5 mini", "gemini-3.7-flash": "Gemini 3.7 Flash", "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "qwen3.8-flash": "Qwen 3.8 Flash", "medgemma-4b": "MedGemma 4B", "qwen3-vl-8b": "Qwen3-VL 8B",
        "qwen2-vl-7b": "Qwen2-VL 7B", "huatuogpt-vision-7b": "HuatuoGPT-Vision 7B", "llava-1.5-7b": "LLaVA-1.5 7B"}
SETS = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
OW = ["medgemma-4b", "qwen3-vl-8b", "qwen2-vl-7b", "huatuogpt-vision-7b", "llava-1.5-7b"]


def num(v, nd, sign):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "")


def c(ci, nd=1, sign=False):
    """The point estimate (an [est, lo, hi] triple prints its estimate)."""
    if ci is None:
        return "--"
    if isinstance(ci, (int, float)):
        return num(ci, nd, sign)
    e, lo, hi = ci
    if e == lo == hi == 0:                       # term not identified (design not run on this data set)
        return "--"
    return num(e, nd, sign)                      # point estimate only: the paper shows no intervals


TABLES = []


def tabular(head, rows, key, group_rule):
    spec = "l" * key + "c" * (len(head) - key)
    L = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join("\\makecell{" + h + "}" for h in head) + " \\\\",
         "\\midrule"]
    for i, r in enumerate(rows):
        if group_rule and i and i % group_rule == 0:
            L.append("\\midrule")
        L.append(" & ".join(r) + " \\\\")
    return L + ["\\bottomrule", "\\end{tabular}"]


def table(label, caption, head, rows, key=2, star=True, group_rule=4, title=None, extra=(), sep=3.2):
    """One float per experiment; `extra` adds further blocks (title, head, rows, key, group_rule) below the first."""
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[!htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{%.1fpt}" % sep,
         f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    for k, (t, h, r, ky, gr) in enumerate([(title, head, rows, key, group_rule)] + list(extra)):
        if k:
            L.append("\\par\\medskip")
        if t:
            L.append("{\\footnotesize " + t + "}\\par\\smallskip")
        L += tabular(h, r, ky, gr)
    L += [f"\\end{{{env}}}", ""]
    TABLES.append("\n".join(L))


def by_set_model(models=MODELS):
    for s, sl in SETS.items():
        for k, m in enumerate(models):
            yield s, (sl if k == 0 else ""), m


# S2 multi-agent systems with original images
rows = []
for s, sl, m in by_set_model():
    d = A[f"{m}|{s}"]
    p, u = d["paired"], d["discussion"]["unanimity_final"]
    rows.append([sl, NAME[m], c(d["real"]["single"][0]), c(d["real"]["reader_mean"][0]), c(d["real"]["panel9"][0]),
                 c(d["discussion"]["acc9"][0]), c(p["panel9_minus_single"], sign=True), c(p["disc9_minus_single"], sign=True),
                 c(p["reader_mean_minus_single"], sign=True), c(p["panel9_minus_reader_mean"], sign=True),
                 c(d["real"]["phi"], 3), num(d["real"]["neff"], 2, False), c(d["discussion"]["phi_last"], 3),
                 num(u["unanimity"], 1, False), num(u["wrong_given_unanimous"], 1, False)])
table("tab:s-mas", "Multi-agent systems of 9 agents with original images (250 items per data set): accuracy (\\%), "
      "multi-agent gain (pp), its split into mean agent minus single agent and majority minus mean agent, error correlation "
      "$\\phi$ of initial and final answers, $N_{\\mathrm{eff}}$, and the final answers of the Decentralized MAS that are "
      "unanimous and, among those, wrong (\\%).",
      ["Data set", "Model", "Single\\\\agent", "Mean\\\\agent", "Indep.", "Decentr.", "Indep.\\\\$-$ single",
       "Decentr.\\\\$-$ single", "Mean agent\\\\$-$ single", "Majority $-$\\\\mean agent", "$\\phi$\\\\initial",
       "$N_{\\mathrm{eff}}$", "$\\phi$\\\\final", "Unani-\\\\mous", "Wrong if\\\\unanimous"], rows, sep=2.4)

# S3 reduced resolution and blank images
rows = []
for s, sl, m in by_set_model():
    d = A[f"{m}|{s}"]
    lv = [d[k] for k in ("real", "res256", "res128", "blank")]
    rows.append([sl, NAME[m]] + [c(x["single"]) for x in lv] + [c(x["phi"], 3) for x in lv] +
                [c(d["paired"]["single_real_minus_blank"], sign=True), num(d["blank"]["abstention_pct"], 1, False),
                 c(d["blank"].get("phi_answered"), 3)])
table("tab:s-resolution", "Reduced resolution and blank images: single-agent accuracy (\\%) and error correlation $\\phi$ of the "
      "Independent MAS of 9 agents with original images ($\\le$1024 pixels), images capped at 256 and 128 pixels, and blank "
      "images (uniform gray, same size); single-agent accuracy lost with the blank image (pp); unparsed answers with the "
      "blank image (\\%) and $\\phi$ recomputed on items both agents answered.",
      ["Data set", "Model", "Acc.\\\\original", "Acc.\\\\256 px", "Acc.\\\\128 px", "Acc.\\\\blank",
       "$\\phi$\\\\original", "$\\phi$\\\\256 px", "$\\phi$\\\\128 px", "$\\phi$\\\\blank",
       "Original $-$\\\\blank", "Unparsed,\\\\blank", "$\\phi$ blank,\\\\answered"], rows)

# S4 answer cross-tabulation and shared wrong answers
rows = []
for s, sl, m in by_set_model():
    p = A[f"{m}|{s}"]["prior_decomposition"]
    rows.append([sl, NAME[m], num(p["prior_right_stays_right"], 1, False), num(p["image_corrects_prior"], 1, False),
                 num(p["image_misleads_prior"], 1, False), num(p["prior_wrong_stays_wrong"], 1, False),
                 str(p["consensus_errors"]), c(p["consensus_errors_matching_blank_majority"])])
table("tab:s-prior", "Agent answers of the Independent MAS of 9 cross-tabulated by correctness with the original and the blank "
      "image (\\% of answers), and shared wrong answers (at least 7 of 9 agents wrong) whose majority answer matches the "
      "blank-image majority.",
      ["Data set", "Model", "Right with\\\\both", "Right only\\\\with original", "Wrong only\\\\with original", "Wrong with\\\\both",
       "Shared wrong\\\\answers ($n$)", "Matching blank-\\\\image majority (\\%)"], rows, star=True)

# S5 occlusion and paired images
rows = []
for m in API3:
    t, p = A[f"{m}|slake_target"], A[f"{m}|slake_pairs"]
    rows.append([NAME[m], "proprietary", c(t["panel9_occ_target_reader_flip"]), c(t["panel9_occ_control_reader_flip"]),
                 c(t["panel9_region_dependence"], sign=True), c(p["single_both_correct"]), c(p["single_answers_differ"])])
for m in OW:
    t, p = D[f"{m}|slake_target"], D[f"{m}|slake_pairs"]
    rows.append([NAME[m], "open-weight", c(t["flip_target"]), c(t["flip_control"]), c(t["region_dependence_flip"], sign=True),
                 c(p["both_correct"]), c(p["answers_differ"])])
table("tab:s-region", "Region occlusion (SLAKE-target, 250 items) and paired images (SLAKE-pairs, 200 pairs). Answer-change "
      "rates under target and control occlusion (\\%; proprietary models: agents of the Independent MAS of 9; open-weight "
      "models: one forward pass), occlusion sensitivity (target minus control, pp), pair accuracy and the share of pairs "
      "answered differently (\\%; single agent).",
      ["Model", "", "Target\\\\change", "Control\\\\change", "Occlusion\\\\sensitivity", "Pair\\\\accuracy", "Answers\\\\differ"],
      rows, star=False, group_rule=3)

# S6 conformity in debate
rows = []
for s, sl, m in by_set_model():
    k = A[f"{m}|{s}"]["discussion"]["conformity"]
    rows.append([sl, NAME[m], c(k["right_to_wrong_minority"]), c(k["right_to_wrong_majority"]),
                 c(k["right_to_wrong_image_dependent"]), c(k["right_to_wrong_prior_consistent"]),
                 c(k["right_to_wrong_difference"], sign=True), c(k["image_dependent_correct_in_minority_pct"]),
                 c(k["prior_consistent_correct_in_minority_pct"])])
table("tab:s-conformity", "Conformity in the Decentralized MAS of 9 agents: share of correct initial answers that are wrong "
      "after debate (\\%), for answers in the initial minority or majority, and for answers that differ from or match the "
      "same agent's blank-image answer (difference in pp); share of correct initial answers in the initial minority among "
      "those that differ from or match the blank-image answer (\\%).",
      ["Data set", "Model", "Initial\\\\minority", "Initial\\\\majority", "Differs\\\\from blank", "Matches\\\\blank",
       "Difference", "In minority,\\\\differs", "In minority,\\\\matches"], rows, star=True)

# S7 perception and reasoning, with report coverage on SLAKE
rows = []
for s, sl, r in by_set_model(API3):
    row = B[s][r]
    rows.append([sl, NAME[r]] + [c(row[f"{p}|own"]["reader_acc"]) for p in API3] + [c(row["direct"]["reader_acc"])] +
                [c(row[f"{p}|own"]["reader_acc_minus_image"], sign=True) for p in API3] +
                [c(row["direct"]["phi"], 3), c(row[f"{r}|own"]["phi"], 3), c(row[f"{r}|shared"]["phi"], 3)])
cov = [[NAME[r]] + [num(B["slake"][r][f"{p}|own"][k], 1, False) for p in API3
                    for k in ("all_reports_miss_target_pct", "errors_report_misses_target_pct")] for r in API3]
SHORT = {"gpt-5-mini": "GPT-5 mini", "gemini-3.7-flash": "Gemini", "claude-haiku-4-5-20251001": "Claude"}
table("tab:s-report", "Perception and reasoning. (a) Mean agent accuracy (\\%) of each reasoning module when every agent reads "
      "its own query-agnostic report from the named perception module, or reads the image (self-consistency samples of the "
      "reasoning module); the change from the image (pp); error correlation $\\phi$ of 9 agents reading the image, their "
      "own reports and one shared report (perception and reasoning module the same model). (b) SLAKE: reports that do not "
      "mention the queried target, among all reports and among the reports behind wrong answers (\\%).",
      ["Data set", "Reasoning\\\\module"] + [f"Reports by\\\\{SHORT[p]}" for p in API3] + ["Image"] +
      [f"{SHORT[p]} reports\\\\$-$ image" for p in API3] + ["$\\phi$\\\\image", "$\\phi$ own\\\\reports",
                                                          "$\\phi$ shared\\\\report"],
      rows, group_rule=3, title="(a) Accuracy and error correlation",
      extra=[("(b) Target absent from the report, SLAKE", ["Reasoning\\\\module"] +
              [f"{SHORT[p]} reports,\\\\{w}" for p in API3 for w in ("all", "behind errors")], cov, 1, 0)], sep=2.0)

# S8 matched contrasts
SHOW = [("role", "Role"), ("rendering", "Contrast adjustment"), ("reference cases", "In-context examples"),
        ("report", "Report"), ("perceiver", "Perception module"), ("reasoner", "Reasoning module"), ("model", "Model")]
rows = []
for m in MODELS:
    rows.append([f"Self-consistency $\\phi$, {NAME[m]}"] + [c(((C[s]["contrasts"].get("sampling only") or {}).get(m)), 3)
                                                          for s in SETS])
for k, lab in SHOW:
    rows.append([f"$\\Delta\\phi$, {lab}"] + [c((C[s]["contrasts"].get(k) or {}).get("mean"), 3, sign=True) for s in SETS])
table("tab:s-contrasts", "Matched contrasts of error correlation: $\\phi$ of two self-consistency agents of one model, and the "
      "mean change $\\Delta\\phi$ over models (or model pairs) when two agents do not share the named component. In-context "
      "examples were run on VQA-RAD and SLAKE.", ["", "VQA-RAD", "SLAKE", "MedXpertQA-MM"], rows, key=1, star=False)

# S9 regression summary
TERMS = [("different perceiver", "Perception\\\\module"), ("image vs report", "Image vs\\\\report"),
         ("different rendering", "Contrast\\\\adjustment"), ("different role", "Role"),
         ("different reference cases", "In-context\\\\examples"), ("different report", "Report")]
rows = [[sl] + [c((C[s]["terms"] or {}).get(t), 3, sign=True) for t, _ in TERMS] +
        [c((C[s].get("different_reasoner") or {}).get("mean"), 3, sign=True)] for s, sl in SETS.items()]
table("tab:s-regression", "Additive regression of pairwise error correlation on the components two agents do not share (one "
      "intercept per pair of reasoning modules), a secondary summary of Tab.~\\ref{tab:s-contrasts}.",
      ["Data set"] + [h for _, h in TERMS] + ["Reasoning\\\\module"], rows, key=1, group_rule=0)

# S10 in-context examples
CONDS = ["none", "shared-image", "private-image", "shared-text", "private-text"]
CL = {"none": "None", "shared-image": "Shared,\\\\images", "private-image": "Private,\\\\images",
      "shared-text": "Shared,\\\\text", "private-text": "Private,\\\\text"}
rows = []
for s in ("vqarad", "slake"):
    for k, m in enumerate(API3):
        d = M[f"{m}|{s}"]
        sl = SETS[s] if k == 0 else ""
        single = {"none": d.get("single_none"), "shared-image": d.get("single_shared-image"),
                  "shared-text": d.get("single_shared-text")}
        rows.append([sl, NAME[m], "Single agent"] + [c(single.get(x)) for x in CONDS])
        rows.append(["", "", "MAS accuracy"] + [c(d.get("panel_" + x)) for x in CONDS])
        rows.append(["", "", "$\\phi$"] + [c(d.get("phi_" + x), 3) for x in CONDS])
table("tab:s-icl", "In-context examples (VQA-RAD and SLAKE): single-agent accuracy, accuracy of the Independent MAS of 9 "
      "agents (\\%) and its error correlation $\\phi$ without examples, with shared examples (the same 3 for every agent) "
      "and with private examples (different examples per agent), given with their images or as text. The single agent "
      "received shared examples only.",
      ["Data set", "Model", ""] + [CL[x] for x in CONDS], rows, key=3, star=True, group_rule=3)

# S11 open-weight models, with error correlation between models
rows = []
for m in OW:
    for k, s in enumerate(SETS):
        d = D[f"{m}|{s}"]
        rows.append([NAME[m] if k == 0 else "", SETS[s], c(d["acc_real"]), c(d["acc_blank"]), c(d.get("acc_res256")),
                     c(d.get("acc_res128")), c(d["visual_reliance"], 2)])
pairs = []
for i, a in enumerate(OW):
    for b in OW[i + 1:]:
        x, y = sorted((a, b))
        cells = [D["cross_model_pairs"][f"{s}|{x}|{y}"] for s in SETS]
        pairs.append([f"{NAME[a]} -- {NAME[b]}", "yes" if cells[0]["shared_encoder"] else ""] +
                     [num(e[k], 3, False) for e in cells for k in ("phi", "phi_over_max")])
table("tab:s-openweight", "Open-weight models (one forward pass per item). (a) Accuracy (\\%) with original, blank, 256-pixel "
      "and 128-pixel images, and mean visual reliance ($|\\Delta$ log-odds of yes$|$ between original and blank image on "
      "yes/no items; Kullback--Leibler divergence on MedXpertQA-MM). (b) Error correlation $\\phi$ between two models and "
      "$\\phi/\\phi_{\\max}$, its ratio to the maximum given the two marginal error rates; HuatuoGPT-Vision 7B and LLaVA-1.5 "
      "7B share the CLIP ViT-L/14-336 encoder.",
      ["Model", "Data set", "Original", "Blank", "256 px", "128 px", "Visual\\\\reliance"], rows, star=True, group_rule=3,
      title="(a) Accuracy and visual reliance",
      extra=[("(b) Error correlation between models", ["Model pair", "Shared\\\\encoder"] +
              [f"{SETS[s]},\\\\{w}" for s in SETS for w in ("$\\phi$", "$\\phi/\\phi_{\\max}$")], pairs, 1, 0)])

out = ROOT / "paper" / "cvpr" / "sec" / "supp_tables.tex"
out.write_text("% Generated by experiments/cvpr_fig/cvpr_tables.py. Do not edit by hand.\n\n" + "\n".join(TABLES))
print(out, len(TABLES), "tables (S2 to S%d)" % (len(TABLES) + 1))
