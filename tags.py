import re
from typing import List
import torch
import streamlit as st
from wordfreq import zipf_frequency
from PIL import Image
from clip_model import clip_model, preprocess, tokenizer, device

HAS_CI = False
try:
    from clip_interrogator import Config as CIConfig, Interrogator
    HAS_CI = True
except Exception:
    HAS_CI = False

@st.cache_resource(show_spinner=False)
def load_interrogator():
    if not HAS_CI:
        return None
    cfg = CIConfig(
        clip_model_name="ViT-B-32/laion2b_s34b_b79k",
        caption_model_name="blip-base",
        device=device,
        quiet=True,
    )
    cfg.use_flavors = True
    return Interrogator(cfg)

interrogator = load_interrogator()

def trim_3plus_repeats(tag: str) -> str:
    words = re.findall(r"[^\W\d_]+", tag.lower())
    kept, counts = [], {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
        if counts[w] <= 2:
            kept.append(w)
    return " ".join(kept)

@torch.inference_mode()
def auto_tags_for_image(
    pil: Image.Image,
    MAX_TAGS: int = 6,
    THRESH_IMG: float = 1.0,
    THRESH_TAG: float = 0.7,
):
    def trim_3plus_repeats(tag: str) -> str:
        words = re.findall(r"[^\W\d_]+", tag.lower())
        kept, counts = [], {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if counts[w] <= 2:
                kept.append(w)
        return " ".join(kept)

    prompt = interrogator.interrogate_fast(pil)
    raw_tags = [t.strip() for t in re.split(r",|\band\b", prompt) if t.strip()]
    raw_tags = [trim_3plus_repeats(t) for t in raw_tags]

    def keep(tag: str) -> bool:
        return any(zipf_frequency(w, "en") >= 4.1 for w in re.findall(r"[a-zA-Z']+", tag))

    all_tags = [t for t in raw_tags if keep(t)]
    if not all_tags:
        return []

    seed = all_tags[: min(4, len(all_tags))]
    rest = all_tags[len(seed):]

    img_t = preprocess(pil).unsqueeze(0).to(device)
    E_img = clip_model.encode_image(img_t)
    E_img = torch.nn.functional.normalize(E_img, dim=-1)
    tok_all = tokenizer(all_tags).to(device)
    tag_emb = clip_model.encode_text(tok_all)
    tag_emb = torch.nn.functional.normalize(tag_emb, dim=-1)

    idx = {t: i for i, t in enumerate(all_tags)}
    def v(t): return tag_emb[idx[t]]

    kept = list(seed)

    relax_schedule = [
        (THRESH_IMG, THRESH_TAG),
        (0.95, 0.65),
        (0.90, 0.60),
        (0.85, 0.55),
        (0.80, 0.50),
    ]

    def try_add(tlist, th_img, th_tag):
        added = 0
        for t in tlist:
            if t in kept:
                continue
            if len(kept) >= MAX_TAGS:
                break
            vt = v(t)
            sim_img = float(E_img @ vt.T)
            sim_tag = max(float(vt @ v(k).T) for k in kept) if kept else 0.0
            if sim_img >= th_img and sim_tag >= th_tag:
                kept.append(t)
                added += 1
        return added

    for th_img, th_tag in relax_schedule:
        if len(kept) >= MAX_TAGS:
            break
        try_add(rest, th_img, th_tag)

    if len(kept) < MAX_TAGS and rest:
        sims = [(t, float(E_img @ v(t).T)) for t in rest if t not in kept]
        sims.sort(key=lambda x: x[1], reverse=True)
        for t, _ in sims:
            if len(kept) >= MAX_TAGS:
                break
            kept.append(t)

    return kept[:MAX_TAGS]
