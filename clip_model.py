from typing import List
import numpy as np
import torch
from PIL import Image
import open_clip
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device

clip_model, preprocess, tokenizer, device = load_clip()

def norm_torch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def embed_text(texts: List[str]) -> np.ndarray:
    OUT = []
    bs = 64
    for i in range(0, len(texts), bs):
        tok = tokenizer(texts[i:i+bs]).to(device)
        feats = clip_model.encode_text(tok)
        feats = norm_torch(feats).detach().cpu().numpy().astype("float32")
        OUT.append(feats)
    return np.vstack(OUT)

def embed_image(pil: Image.Image) -> np.ndarray:
    t = preprocess(pil).unsqueeze(0).to(device)
    feats = clip_model.encode_image(t)
    feats = norm_torch(feats).detach().cpu().numpy().astype("float32")
    return feats[0]
