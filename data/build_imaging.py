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
All are sampled with the same stratified shuffle as the text benchmarks (seed 42).
"""
import ast, collections, hashlib, io, json, pathlib, random, sys, zipfile

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


def _list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


if __name__ == "__main__":
    print("VQA-RAD closed yes/no items available:", build_vqarad())
    n_all, n_rad, mods = build_mxqa_mm()
    print(f"MedXpertQA-MM: {n_rad} of {n_all} items have only radiology images; image modalities: {mods}")
    print("SLAKE closed yes/no English items available:", build_slake())
