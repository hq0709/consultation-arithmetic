"""Imaging items for the radiology reading-panel experiments.

Schema as build_datasets.py, plus  images: [repo-relative paths], domain: "radiology".
Outputs (git-ignored, like every data/*.jsonl): data/vqarad_closed_250.jsonl,
data/mxqa_mm_rad_250.jsonl, data/slake_closed_250.jsonl; images under data_local/.

  VQA-RAD   test split, closed questions with a yes/no gold answer. The option order is
            shuffled per item (seeded by qid) so "yes" is not always (A).
  MedXpertQA-MM  test split, items whose every image is a radiology modality. Modality is
            labelled per image by gpt-4.1-nano (temperature 0, cached). It calls some drawings
            radiographs (2 of 24 in a visual spot check), so every candidate image is then
            confirmed as an actual acquisition by claude-haiku-4.5, a model outside the
            evaluated set. Labels: data_local/medxpertqa_mm/modality.json.
  SLAKE     English test split, closed questions with a yes/no gold answer (official
            BoKelvin/SLAKE release; images from its imgs.zip). Options shuffled as for VQA-RAD.
  SLAKE-target (S4)  English closed yes/no questions about an organ or a lesion that is present in
            the image (from detection.json boxes), test + train, disjoint from the SLAKE-250 set.
            Each item carries target_box (union of the matched boxes, expanded 10% per side, at most
            40% of the image) and control_box (same size, no overlap, at least 70% inside the body).
  SLAKE-pairs (S5)   the same question on two images with opposite gold answers, 200 pairs,
            at most 5 pairs per question template.
All are sampled with the same stratified shuffle as the text benchmarks (seed 42).
"""
import ast, collections, glob, hashlib, io, json, pathlib, random, re, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from data.build_datasets import write, stratified_shuffle  # noqa: E402

LOCAL = ROOT / "data_local"
VQARAD_PARQUET = sorted((pathlib.Path.home() / ".cache/huggingface/hub/datasets--flaviagiammarino--vqa-rad")
                        .glob("snapshots/*/data/test-*.parquet"))
MM = LOCAL / "medxpertqa_mm"
RADIOLOGY = {"radiograph", "CT", "MRI", "ultrasound", "nuclear medicine", "angiography/fluoroscopy",
             "mammography"}
MODALITIES = sorted(RADIOLOGY) + ["pathology/histology", "endoscopy", "clinical photograph",
                                  "ECG/waveform", "chart/diagram/table", "other"]
N = 250


def build_vqarad():
    import pandas as pd
    df = pd.read_parquet(VQARAD_PARQUET[0])
    out_dir = LOCAL / "vqarad/images"; out_dir.mkdir(parents=True, exist_ok=True)
    items, seen = [], set()
    for i, r in df.iterrows():
        ans = str(r["answer"]).strip().lower()
        if ans not in ("yes", "no"):
            continue
        raw = r["image"]["bytes"]
        h = hashlib.sha256(raw).hexdigest()[:16]
        key = (h, r["question"].strip().lower())
        if key in seen:                       # the test split repeats a few (image, question) pairs
            continue
        seen.add(key)
        img = out_dir / f"{h}.png"
        if not img.exists():
            from PIL import Image
            Image.open(io.BytesIO(raw)).convert("RGB").save(img)
        qid = f"vqarad/test-{i}"
        opts = ["Yes", "No"]
        random.Random(qid).shuffle(opts)
        options = dict(zip("AB", opts))
        gold = next(k for k, v in options.items() if v.lower() == ans)
        items.append({"qid": qid, "bench": "vqarad", "stem": r["question"].strip(), "options": options,
                      "answer": gold, "images": [str(img.relative_to(ROOT))], "domain": "radiology",
                      "meta": {"gold_text": ans, "image": h}})
    items = stratified_shuffle(items, lambda r: r["meta"]["gold_text"])
    write("vqarad_closed_250.jsonl", items[:N])
    return len(items)


def classify_modality(paths):
    from common.llm import chat, pmap
    sys_msg = ("You label medical images by modality. Reply with JSON only: "
               '{"modality": "<one of: ' + ", ".join(MODALITIES) + '>"}')

    def one(p):
        r = chat("gpt-4.1-nano", [{"role": "user", "content": "What is the modality of this image?"}],
                 system=sys_msg, temperature=0.0, seed=7, max_tokens=30, json_mode=True,
                 tag="modality", images=[p])
        try:
            m = json.loads(r["text"]).get("modality", "other")
        except Exception:
            m = "other"
        return m if m in MODALITIES else "other"
    return dict(zip(paths, pmap(one, paths, workers=8)))


VERIFIER = "claude-haiku-4-5-20251001"


def verify_radiological(paths):
    from common.llm import chat, pmap
    sys_msg = ("Is this image an actual radiological acquisition: a radiograph, CT, MRI, ultrasound, "
               "nuclear medicine scan, angiogram or fluoroscopy, or mammogram? Drawings, illustrations, "
               "schematic or annotated diagrams, clinical photographs, pathology slides, charts and "
               'tables do not count. Reply with JSON only: {"radiological": true or false}')

    def one(p):
        r = chat(VERIFIER, [{"role": "user", "content": "Classify the attached image."}], system=sys_msg,
                 temperature=0.0, max_tokens=30, json_mode=True, tag="radverify", images=[p])
        try:
            return bool(json.loads(r["text"].strip().strip("`").removeprefix("json")).get("radiological"))
        except Exception:
            return "true" in r["text"].lower()
    return dict(zip(paths, pmap(one, paths, workers=8)))


def build_mxqa_mm():
    img_dir = MM / "images"
    if not img_dir.exists():
        with zipfile.ZipFile(MM / "images.zip") as z:
            z.extractall(MM)
    rows = [json.loads(l) for l in open(MM / "MM/test.jsonl")]
    find = {p.name: p for p in (MM / "images").rglob("*") if p.is_file()}
    paths = sorted({str(find[f].relative_to(ROOT)) for r in rows for f in _list(r["images"])})
    mod = classify_modality(paths)
    cand = sorted({p for r in rows for p in [str(find[f].relative_to(ROOT)) for f in _list(r["images"])]
                   if mod[p] in RADIOLOGY})
    ok = verify_radiological(cand)
    (MM / "modality.json").write_text(json.dumps({p: {"modality": mod[p], "radiological": ok.get(p)}
                                                  for p in paths}, indent=1))
    items = []
    for r in rows:
        ims = [str(find[f].relative_to(ROOT)) for f in _list(r["images"])]
        if not ims or not all(mod[p] in RADIOLOGY and ok.get(p) for p in ims):
            continue
        stem = r["question"].split("\nAnswer Choices:")[0].strip()
        opts = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
        items.append({"qid": f"mxqmm/{r['id']}", "bench": "mxqa_mm_rad", "stem": stem, "options": opts,
                      "answer": r["label"], "images": ims, "domain": "radiology",
                      "meta": {"body_system": r["body_system"], "task": r["medical_task"],
                               "type": r["question_type"], "modality": [mod[p] for p in ims]}})
    items = stratified_shuffle(items, lambda r: (r["meta"]["body_system"], r["meta"]["task"]))
    write("mxqa_mm_rad_250.jsonl", items[:N])
    c = collections.Counter(m for p in paths for m in [mod[p]])
    rejected = sum(1 for p in cand if not ok[p])
    print(f"  verifier rejected {rejected} of {len(cand)} images labelled radiological")
    return len(rows), len(items), dict(c)


def build_slake():
    import glob
    src = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/datasets--BoKelvin--SLAKE"
                                "/snapshots/*/test.json")))[0]
    d = LOCAL / "slake"
    if not (d / "imgs").exists():
        with zipfile.ZipFile(d / "imgs.zip") as z:
            z.extractall(d)
    items = []
    for r in json.load(open(src)):
        ans = r["answer"].strip().lower()
        if r["q_lang"] != "en" or r["answer_type"] != "CLOSED" or ans not in ("yes", "no"):
            continue
        img = d / "imgs" / r["img_name"]
        qid = f"slake/test-{r['qid']}"
        opts = ["Yes", "No"]
        random.Random(qid).shuffle(opts)
        options = dict(zip("AB", opts))
        gold = next(k for k, v in options.items() if v.lower() == ans)
        items.append({"qid": qid, "bench": "slake", "stem": r["question"].strip(), "options": options,
                      "answer": gold, "images": [str(img.relative_to(ROOT))], "domain": "radiology",
                      "meta": {"gold_text": ans, "modality": r["modality"], "location": r["location"],
                               "content": r.get("content_type")}})
    items = stratified_shuffle(items, lambda r: (r["meta"]["gold_text"], r["meta"]["modality"]))
    write("slake_closed_250.jsonl", items[:N])
    return len(items)


SLAKE_JSON = lambda split: sorted(glob.glob(str(
    pathlib.Path.home() / f".cache/huggingface/hub/datasets--BoKelvin--SLAKE/snapshots/*/{split}.json")))[0]
ORGAN_WORDS = ["liver", "lung", "heart", "kidney", "spleen", "bladder", "colon", "stomach", "rectum",
               "esophagus", "duodenum", "small bowel", "spinal cord", "trachea", "larynx", "parotid",
               "mandible", "temporal lobe", "brain stem", "femoral head", "eye", "tooth"]


def _slake_rows(splits=("test", "train")):
    import glob as _g  # noqa: F401
    out = []
    for s in splits:
        for r in json.load(open(SLAKE_JSON(s))):
            a = r["answer"].strip().lower()
            if r["q_lang"] == "en" and r["answer_type"] == "CLOSED" and a in ("yes", "no"):
                out.append(dict(r, split=s, gold_text=a))
    return out


def _yesno_item(qid, bench, r, extra_meta=None):
    img = LOCAL / "slake/imgs" / r["img_name"]
    opts = ["Yes", "No"]
    random.Random(qid).shuffle(opts)
    options = dict(zip("AB", opts))
    gold = next(k for k, v in options.items() if v.lower() == r["gold_text"])
    meta = {"gold_text": r["gold_text"], "modality": r["modality"], "content": r.get("content_type"),
            "slake_qid": r["qid"], "split": r["split"]}
    meta.update(extra_meta or {})
    return {"qid": qid, "bench": bench, "stem": r["question"].strip(), "options": options, "answer": gold,
            "images": [str(img.relative_to(ROOT))], "domain": "radiology", "meta": meta}


def _target(r):
    """-> (union box [x0,y0,x1,y1], label) of the region the question asks about, or None."""
    q = r["question"].lower().replace("brian", "brain")
    det_p = LOCAL / "slake/imgs" / pathlib.Path(r["img_name"]).parent / "detection.json"
    if not det_p.exists():
        return None
    boxes = [(k, v) for d in json.load(open(det_p)) for k, v in d.items()]
    organ = [w for w in ORGAN_WORDS if re.search(rf"\b{w}s?\b", q)]
    is_organ_lab = lambda k: any(w in k.lower() for w in ORGAN_WORDS)
    to_xyxy = lambda v: [v[0], v[1], v[0] + v[2], v[1] + v[3]]
    if q.startswith("does the picture contain") and organ:
        if r["gold_text"] != "yes":
            return None
        sel = [to_xyxy(v) for k, v in boxes if any(w in k.lower() for w in organ)]
        lab = organ[0]
    elif any(w in q for w in ("healthy", "normal", "abnormal")):
        healthy_form = ("healthy" in q or ("normal" in q and "abnormal" not in q))
        present = (r["gold_text"] == "no") if healthy_form else (r["gold_text"] == "yes")
        if not present:
            return None
        les = [to_xyxy(v) for k, v in boxes if not is_organ_lab(k)]
        if organ:
            org = [to_xyxy(v) for k, v in boxes if any(w in k.lower() for w in organ)]
            inside = [b for b in les if any(b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3] for o in org)]
            les = inside or les
        sel, lab = les, "lesion"
    else:
        return None
    if not sel:
        return None
    return [min(b[0] for b in sel), min(b[1] for b in sel), max(b[2] for b in sel), max(b[3] for b in sel)], lab


def _boxes(r, box):
    import numpy as np
    from PIL import Image
    im = Image.open(LOCAL / "slake/imgs" / r["img_name"]).convert("L"); W, H = im.size
    x0, y0, x1, y1 = box; mw, mh = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    tb = [max(0, x0 - mw), max(0, y0 - mh), min(W, x1 + mw), min(H, y1 + mh)]
    bw, bh = tb[2] - tb[0], tb[3] - tb[1]
    if bw * bh > 0.4 * W * H or bw < 4 or bh < 4:
        return None
    a = np.asarray(im)
    cands = []
    step = max(4, min(W, H) // 16)
    for cy in range(0, int(H - bh) + 1, step):
        for cx in range(0, int(W - bw) + 1, step):
            c = [cx, cy, cx + bw, cy + bh]
            if c[0] < tb[2] and tb[0] < c[2] and c[1] < tb[3] and tb[1] < c[3]:
                continue
            if (a[int(c[1]):int(c[3]), int(c[0]):int(c[2])] > 15).mean() >= 0.7:
                cands.append(c)
    if not cands:
        return None
    return tb, random.Random(r["img_name"] + r["question"]).choice(cands)


def build_slake_target():
    used = {json.loads(l)["meta"]["slake_qid"] for l in open(ROOT / "data/slake_closed_250.jsonl")
            if "slake_qid" in json.loads(l).get("meta", {})}
    used |= {int(json.loads(l)["qid"].split("-")[-1]) for l in open(ROOT / "data/slake_closed_250.jsonl")}
    items = []
    for r in _slake_rows():
        if r["qid"] in used or r.get("content_type") not in ("Organ", "Abnormality"):
            continue
        tg = _target(r)
        if not tg:
            continue
        bx = _boxes(r, tg[0])
        if not bx:
            continue
        it = _yesno_item(f"slakeT/{r['split']}-{r['qid']}", "slake_target", r,
                         {"target_label": tg[1], "kind": "organ" if tg[1] != "lesion" else "lesion"})
        it["target_box"], it["control_box"] = [round(v, 1) for v in bx[0]], [round(v, 1) for v in bx[1]]
        items.append(it)
    items = stratified_shuffle(items, lambda r: (r["meta"]["kind"], r["meta"]["modality"]))
    write("slake_target_250.jsonl", items[:N])
    return len(items)


def build_slake_pairs(n_pairs=200, per_template=5):
    norm = lambda q: re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
    by = collections.defaultdict(lambda: {"yes": [], "no": []})
    for r in _slake_rows():
        by[norm(r["question"])][r["gold_text"]].append(r)
    pairs = []
    for q, v in sorted(by.items()):
        ys = sorted(v["yes"], key=lambda r: r["img_name"]); ns = sorted(v["no"], key=lambda r: r["img_name"])
        rng = random.Random(q); rng.shuffle(ys); rng.shuffle(ns)
        seen, k = set(), 0
        for a, b in zip(ys, ns):
            if a["img_name"] == b["img_name"] or a["img_name"] in seen or b["img_name"] in seen or k >= per_template:
                continue
            seen |= {a["img_name"], b["img_name"]}; k += 1
            pairs.append((q, a, b))
    pairs = stratified_shuffle(pairs, lambda p: p[0])[:n_pairs]
    items = []
    for i, (q, a, b) in enumerate(pairs):
        for side, r in (("a", a), ("b", b)):
            items.append(_yesno_item(f"slakeP/{i:03d}-{side}", "slake_pairs", r, {"pair": i, "template": q}))
    write("slake_pairs_200.jsonl", items)
    return len(pairs), len({p[0] for p in pairs})


def _list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


if __name__ == "__main__":
    print("VQA-RAD closed yes/no items available:", build_vqarad())
    n_all, n_rad, mods = build_mxqa_mm()
    print(f"MedXpertQA-MM: {n_rad} of {n_all} items have only radiology images; image modalities: {mods}")
    print("SLAKE closed yes/no English items available:", build_slake())
    print("SLAKE-target items available:", build_slake_target())
    print("SLAKE pairs (pairs, templates):", build_slake_pairs())
