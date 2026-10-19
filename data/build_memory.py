"""Reference-case memory for experiment C4 (docs/IMAGING_PROTOCOL.md section 6).

Library: closed yes/no questions on VQA-RAD and SLAKE (English) training images, one entry per
image with all its closed QAs. Images identical to a test image (SHA-256) are dropped; for each
query, library images are dropped as the same image, scan or patient when their 64-bit perceptual
hash is within Hamming distance 8 of the query's, or their CLIP cosine is >= 0.98. (V3: CLIP cosines
between ordinary radiology images sit at 0.91-0.95, so 0.95 removed similar cases, not copies;
pHash alone missed same-slice CT pairs and same-patient chest films at cosine 0.99.)
The datasets carry no patient IDs, so the cosine bound is the patient-level safeguard.
Retrieval: CLIP ViT-L/14-336 image embeddings (local GPU), the 27 most similar library images,
shuffled with a seed derived from the qid and split into 9 sets of 3 (shared memory uses set 0,
private memory gives reader i set i, so both conditions draw from the same similarity range).
Each reference image carries the one closed QA whose question is most similar to the query
question (TF-IDF cosine).
Writes data/vqarad_closed_250_mem.jsonl and data/slake_closed_250_mem.jsonl (items + "refs") and
data_local/memory/retrieval_stats.json.
"""
import collections, glob, hashlib, io, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np

LOCAL = ROOT / "data_local"
MEM = LOCAL / "memory"; MEM.mkdir(parents=True, exist_ok=True)
K, SETS, PER, PHASH_DUP, SAME_SCAN = 27, 9, 3, 8, 0.98
HF = pathlib.Path.home() / ".cache/huggingface/hub"


def sha(path):
    from PIL import Image
    b = io.BytesIO(); Image.open(path).convert("RGB").save(b, format="PNG")
    return hashlib.sha256(b.getvalue()).hexdigest()


def phash(path):
    """64-bit DCT perceptual hash (32x32 greyscale, 8x8 low-frequency block, median threshold)."""
    from PIL import Image
    from scipy.fft import dctn
    a = np.asarray(Image.open(ROOT / path).convert("L").resize((32, 32), Image.LANCZOS), float)
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    return d > np.median(d[1:])


def library():
    import pandas as pd
    lib = collections.defaultdict(list)                     # image path -> [(question, answer)]
    df = pd.read_parquet(sorted(HF.glob("datasets--flaviagiammarino--vqa-rad/snapshots/*/data/train-*.parquet"))[0])
    out = MEM / "vqarad_train"; out.mkdir(exist_ok=True)
    for _, r in df.iterrows():
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        raw = r["image"]["bytes"]
        p = out / f"{hashlib.sha256(raw).hexdigest()[:16]}.png"
        if not p.exists():
            from PIL import Image
            Image.open(io.BytesIO(raw)).convert("RGB").save(p)
        lib[str(p.relative_to(ROOT))].append((r["question"].strip(), a.capitalize()))
    tr = json.load(open(sorted(HF.glob("datasets--BoKelvin--SLAKE/snapshots/*/train.json"))[0]))
    for r in tr:
        a = r["answer"].strip().lower()
        if r["q_lang"] == "en" and r["answer_type"] == "CLOSED" and a in ("yes", "no"):
            p = LOCAL / "slake/imgs" / r["img_name"]
            lib[str(p.relative_to(ROOT))].append((r["question"].strip(), a.capitalize()))
    return {k: sorted(set(v)) for k, v in lib.items()}


def embed(paths, device="cuda", bs=32):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(ROOT / p).convert("RGB") for p in paths[i:i + bs]]
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, torch.float16)
            f = model.get_image_features(pixel_values=x)
            f = getattr(f, "pooler_output", f)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    lib = library()
    sets = {"vqarad_closed_250": [json.loads(l) for l in open(ROOT / "data/vqarad_closed_250.jsonl")],
            "slake_closed_250": [json.loads(l) for l in open(ROOT / "data/slake_closed_250.jsonl")]}
    test_hash = {sha(ROOT / it["images"][0]) for its in sets.values() for it in its}
    lib_paths = sorted(p for p in lib if sha(ROOT / p) not in test_hash)
    dropped_identical = len(lib) - len(lib_paths)
    q_paths = sorted({it["images"][0] for its in sets.values() for it in its})
    E_lib, E_q = embed(lib_paths), embed(q_paths)
    H_lib = np.array([phash(p) for p in lib_paths]); H_q = {p: phash(p) for p in q_paths}
    qi = {p: i for i, p in enumerate(q_paths)}
    tfidf = TfidfVectorizer().fit([q for v in lib.values() for q, _ in v] + [it["stem"] for its in sets.values() for it in its])
    stats = {"library_images": len(lib), "dropped_identical": dropped_identical, "near_dup_dropped": 0, "cos": []}
    for name, items in sets.items():
        with open(ROOT / f"data/{name}_mem.jsonl", "w") as f:
            for it in items:
                sims = E_lib @ E_q[qi[it["images"][0]]]
                dup = ((H_lib != H_q[it["images"][0]]).sum(1) <= PHASH_DUP) | (sims >= SAME_SCAN)
                order = [j for j in np.argsort(-sims) if not dup[j]]
                stats["near_dup_dropped"] += int(dup.sum())
                top = order[:K]
                stats["cos"] += [float(sims[j]) for j in top]
                random.Random(it["qid"]).shuffle(top)
                qv = tfidf.transform([it["stem"]])
                refs = []
                for s in range(SETS):
                    group = []
                    for j in top[s * PER:(s + 1) * PER]:
                        qas = lib[lib_paths[j]]
                        sc = (tfidf.transform([q for q, _ in qas]) @ qv.T).toarray().ravel()
                        q, a = qas[int(np.argmax(sc))]
                        group.append({"image": lib_paths[j], "question": q, "answer": a, "cos": round(float(sims[j]), 3)})
                    refs.append(group)
                f.write(json.dumps(dict(it, refs=refs), ensure_ascii=False) + "\n")
        print(f"  wrote {name}_mem.jsonl")
    c = np.array(stats.pop("cos"))
    stats.update(cos_mean=round(float(c.mean()), 3), cos_p10=round(float(np.percentile(c, 10)), 3),
                 cos_p90=round(float(np.percentile(c, 90)), 3))
    (MEM / "retrieval_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
