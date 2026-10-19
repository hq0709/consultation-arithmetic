"""Experiment D (docs/IMAGING_PROTOCOL.md): option probabilities of local open-weight VLMs.

For every item and image condition, one forward pass with a fixed prompt; the next-token logits
of the option letters (best of "A"/" A" variants) are softmaxed over the options. No sampling.
Image conditions use the same spec grammar as the API runs (common/llm.py: blank, res256, res128,
inv, clahe, occ=...), rendered with the same code, so every reader sees identical pixels.

  python experiments/openweight_probe.py --model qwen3-vl-8b --items data/vqarad_closed_250.jsonl \
      --conditions real,blank --out results/OW_qwen3-vl-8b_vqarad.jsonl
  python experiments/openweight_probe.py --model llava-med --smoke

Models whose code lives in a LLaVA fork (llava-med, huatuogpt-vision) must each run in their own
process: both forks are importable only as the package `llava`.
"""
import argparse, json, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
CK = pathlib.Path("/home/myid/hj67104/medical_vlm_study/checkpoints")
VLM = pathlib.Path("/home/myid/hj67104/medical_vlm_study")
PROMPT = ("You are an attending general radiologist. Answer the multiple-choice question about the "
          "attached image.\n\n{q}\n\nOptions:\n{opts}\n\nAnswer with the option letter only.")
# Yes/no items are scored on the answer words, not on option letters: the LLaVA family puts most of
# its mass on "A" whatever the image shows (smoke test: LLaVA-Med p(A) = 0.96-1.00 on every item).
PROMPT_YN = ("You are an attending general radiologist. Answer the question about the attached image."
             "\n\n{q}\n\nAnswer yes or no.")
MODELS = {
    # name: (loader, path or hub id, vision encoder family for the shared-encoder comparison)
    "qwen3-vl-8b": ("hf", str(CK / "Qwen3-VL-8B-Instruct"), "qwen3-vit"),
    "qwen2-vl-7b": ("hf", str(CK / "Qwen2-VL-7B-Instruct"), "qwen2-vit"),
    "medgemma-4b": ("hf", str(sorted(pathlib.Path.home().glob(
        ".cache/huggingface/hub/models--google--medgemma-4b-it/snapshots/*/config.json"))[0].parent), "siglip"),
    "llava-1.5-7b": ("hf", "llava-hf/llava-1.5-7b-hf", "clip-vit-l14-336"),
    "llava-med": ("llava", str(CK / "llava-med-v1.5-mistral-7b"), "clip-vit-l14-336"),
    "huatuogpt-vision-7b": ("huatuo", str(CK / "huatuogpt-vision-7b"), "clip-vit-l14-336"),
}


def render(spec):
    """PIL image for an image spec, rendered exactly as the API runs render it (minus JPEG)."""
    from PIL import Image
    from common.llm import _render, IMAGE_MAX_SIDE  # noqa: F401
    *ops, path = spec.split(":")
    p = path if os.path.isabs(path) else str(ROOT / path)
    with Image.open(p) as im:
        return _render(im.convert("RGB"), ops)


def letter_ids(tok, letter):
    ids = set()
    variants = [letter, " " + letter]
    if letter.lower() in ("yes", "no"):
        variants = [w for c in (letter.lower(), letter.capitalize()) for w in (c, " " + c)]
    for v in variants:
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            ids.add(e[0])
    return sorted(ids)


class HFChat:
    def __init__(self, path, device):
        import torch, transformers
        from transformers import AutoProcessor, AutoConfig
        arch = AutoConfig.from_pretrained(path).architectures[0]
        cls = getattr(transformers, arch, None) or transformers.AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = cls.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.tok = self.proc.tokenizer
        self.device = device

    def logits(self, images, text):
        import torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=[prompt], images=images or None, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits[0, -1].float()


class LlavaFork:
    """LLaVA-Med (original LLaVA code) and HuatuoGPT-Vision (its own LLaVA fork)."""

    def __init__(self, kind, path, device):
        import torch
        self.kind, self.device = kind, device
        if kind == "llava":
            sys.path.insert(0, str(VLM / "LLaVA"))
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            # one visible GPU per process: the builder's default device_map="auto" puts it all there
            self.tok, self.model, self.ip, _ = load_pretrained_model(path, None, get_model_name_from_path(path))
            self.conv_mode = "mistral_instruct"
        else:
            sys.path.insert(0, str(VLM / "HuatuoGPT-Vision"))
            from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
            from transformers import AutoTokenizer
            from safetensors import safe_open
            # transformers 5 forbids the fork's nested CLIP load inside __init__: build the language
            # model first, then the base CLIP tower, then overwrite it with the checkpoint's own
            # (fine-tuned) vision weights, exactly the tensors the original loader ends up using.
            self.model = LlavaQwen2ForCausalLM.from_pretrained(path, init_vision_encoder_from_ckpt=False,
                                                               torch_dtype=torch.bfloat16)
            vt = self.model.get_vision_tower()
            vt.vision_tower_name = str(pathlib.Path(path) / "vit/clip_vit_large_patch14_336")
            vt.load_model()
            pre = "model.vision_tower.vision_tower."
            idx = json.load(open(pathlib.Path(path) / "model.safetensors.index.json"))["weight_map"]
            sd = {}
            for shard in sorted({f for k, f in idx.items() if k.startswith(pre)}):
                with safe_open(str(pathlib.Path(path) / shard), framework="pt") as f:
                    sd.update({k[len(pre):]: f.get_tensor(k) for k in f.keys() if k.startswith(pre)})
            target = set(vt.vision_tower.state_dict())
            if not any(k.startswith("vision_model.") for k in target):   # transformers 5 flattened CLIPVisionModel
                sd = {k[len("vision_model."):] if k.startswith("vision_model.") else k: v for k, v in sd.items()}
            vt.vision_tower.load_state_dict(sd, strict=True)
            self.model.to(device=device, dtype=torch.bfloat16).eval()
            vt.to(device=device, dtype=torch.bfloat16)
            self.ip = vt.image_processor
            self.tok = AutoTokenizer.from_pretrained(path)
            self.conv_mode = "huatuo"
        self.model.eval()

    def logits(self, images, text):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        q = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + text
        if self.conv_mode == "huatuo":                 # its fork has no llava.mm_utils
            return self._huatuo_logits(images, q)
        from llava.mm_utils import tokenizer_image_token, process_images
        if True:
            from llava.conversation import conv_templates
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], q); conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        kw = {}
        if images:
            px = process_images(images, self.ip, self.model.config)
            px = px.to(self.device, dtype=self.model.dtype) if not isinstance(px, list) else [x.to(self.device, dtype=self.model.dtype) for x in px]
            kw = {"images": px, "image_sizes": [im.size for im in images]}
        with torch.no_grad():
            return self.model(input_ids=ids, **kw).logits[0, -1].float()


def _huatuo_tokenize(tok, prompt, image_token_index):
    """HuatuoGPT-Vision/cli.py::tokenizer_image_token."""
    import torch
    chunks = [tok(c, add_special_tokens=False).input_ids for c in prompt.split("<image>")]
    ids = []
    for i, c in enumerate(chunks):
        if i:
            ids.append(image_token_index)
        ids.extend(c)
    return torch.tensor(ids, dtype=torch.long)


def _huatuo_logits(self, images, q):
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from PIL import Image
    prompt = f"<|user|>\n{q.strip()}\n<|assistant|>\n"
    ids = _huatuo_tokenize(self.tok, prompt, IMAGE_TOKEN_INDEX).unsqueeze(0).to(self.device)
    kw = {}
    if images:
        bg = tuple(int(x * 255) for x in self.ip.image_mean)
        ts = []
        for im in images:
            w, h = im.size
            s = Image.new(im.mode, (max(w, h),) * 2, bg); s.paste(im, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
            ts.append(self.ip.preprocess(s, return_tensors="pt")["pixel_values"][0])
        kw = {"images": torch.stack(ts).to(self.device, dtype=torch.bfloat16)}
    with torch.no_grad():
        return self.model(input_ids=ids, **kw).logits[0, -1].float()


LlavaFork._huatuo_logits = _huatuo_logits


def load(name, device):
    kind, path, _ = MODELS[name]
    return HFChat(path, device) if kind == "hf" else LlavaFork(kind, path, device)


def is_yesno(item):
    return sorted(v.lower() for v in item["options"].values()) == ["no", "yes"]


def option_probs(m, item, spec_images):
    """-> {option letter: probability}; yes/no items are scored on the words yes/no."""
    import torch
    letters = list(item["options"])
    ims = [render(s) for s in spec_images]
    tok = getattr(m, "tok")
    if is_yesno(item):
        lg = m.logits(ims, PROMPT_YN.format(q=item["stem"]))
        sc = torch.stack([lg[letter_ids(tok, item["options"][L])].max() for L in letters])
    else:
        opts = "\n".join(f"({k}) {v}" for k, v in item["options"].items())
        lg = m.logits(ims, PROMPT.format(q=item["stem"], opts=opts))
        sc = torch.stack([lg[letter_ids(tok, L)].max() for L in letters])
    return dict(zip(letters, torch.softmax(sc, 0).tolist()))


def specs_for(item, cond):
    imgs = item.get("images") or []
    if cond == "real":
        return imgs
    if cond in ("occ_target", "occ_control"):
        b = item["target_box" if cond == "occ_target" else "control_box"]
        return [f"occ={','.join(f'{v:.0f}' for v in b)}:{imgs[0]}"] + imgs[1:]
    return [f"{cond}:{x}" for x in imgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--items", default="data/vqarad_closed_250.jsonl")
    ap.add_argument("--conditions", default="real")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="load the model and score 5 items, real and blank")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / a.items)][: (5 if a.smoke else a.limit)]
    conds = ["real", "blank"] if a.smoke else a.conditions.split(",")
    t0 = time.time()
    m = load(a.model, a.device)
    print(f"[{a.model}] loaded in {time.time() - t0:.0f}s", flush=True)
    done = set()
    if a.out and (ROOT / a.out).exists():
        done = {(r["qid"], r["condition"]) for r in map(json.loads, open(ROOT / a.out))}
    fh = open(ROOT / a.out, "a") if a.out else None
    n = c = 0
    for it in items:
        for cond in conds:
            if (it["qid"], cond) in done:
                continue
            p = option_probs(m, it, specs_for(it, cond))
            pred = max(p, key=p.get)
            rec = {"qid": it["qid"], "bench": it["bench"], "model": a.model, "condition": cond,
                   "probs": p, "pred": pred, "gold": it["answer"], "correct": int(pred == it["answer"]),
                   "encoder": MODELS[a.model][2]}
            if fh:
                fh.write(json.dumps(rec) + "\n")
            if cond == "real":
                n += 1; c += rec["correct"]
            if a.smoke:
                print(f"  {it['qid']:<18} {cond:<6} pred {pred} gold {it['answer']} p={ {k: round(v, 2) for k, v in p.items()} }")
    if fh:
        fh.close()
    print(f"[{a.model}] real-image accuracy {c}/{n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
