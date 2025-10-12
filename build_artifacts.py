#!/usr/bin/env python3
# build_artifacts.py — merged best-of-both
import os, re, json, random, pickle, pathlib, sys
from typing import List, Dict, Tuple

import numpy as np
import torch
import open_clip
from sklearn.cluster import MiniBatchKMeans

# ---------------- Config ----------------
LYRICS_DIR = pathlib.Path("lyrics_3.6k")   # each file is one song JSON
ART        = pathlib.Path("artifacts")
MODEL      = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SEED       = 42

BS         = 128        # text batch size
MAX_LYR    = 5000       # match the app truncation
MIN_CHARS  = 40         # skip very short lyrics

# ---------------- Seed & fs ----------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
ART.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def clean_text(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s).strip()

def load_song_json(p: pathlib.Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    sid    = str(obj.get("id") or p.stem)
    title  = clean_text(obj.get("title") or "")
    artist = clean_text(obj.get("artist") or "")
    lyrics = (obj.get("lyrics") or "").strip()

    # match the app’s preprocessing:
    lyrics = re.sub(r"\[.*?]", "", lyrics)   # drop [chorus], [verse], …
    lyrics = re.sub(r"\s+", " ", lyrics).lower()
    lyrics = lyrics[:MAX_LYR]

    return {"id": sid, "title": title, "artist": artist, "lyrics": lyrics}

def prepare_texts(records: List[Dict]) -> List[str]:
    # exact template used in app.py
    return [f'{r["title"]} by {r["artist"]}. Lyrics: {r["lyrics"]}' for r in records]

@torch.inference_mode()
def embed_texts(texts: List[str], model, tokenizer, device) -> np.ndarray:
    OUT = []
    for i in range(0, len(texts), BS):
        tok = tokenizer(texts[i:i+BS]).to(device)
        feats = model.encode_text(tok)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        OUT.append(feats.detach().cpu().numpy().astype("float32"))
    if not OUT:
        return np.zeros((0, 512), dtype="float32")
    X = np.vstack(OUT)  # (N, 512)
    # extra numeric safety on numpy side
    X /= np.maximum(1e-8, np.linalg.norm(X, axis=1, keepdims=True))
    return X.astype("float32")

def pick_k(n: int) -> int:
    # robust cluster count for small/large datasets
    return max(4, min(256, int(np.sqrt(max(1, n)))))

# ---------------- Main ----------------
def main():
    # ---------- collect ----------
    files = sorted([p for p in LYRICS_DIR.rglob("*.json") if p.is_file()])
    if not files:
        raise SystemExit("No *.json files found in starter_lyrics/")

    records = []
    for p in files:
        try:
            r = load_song_json(p)
        except Exception as e:
            print(f"!! Skipping {p}: {e}")
            continue
        if len(r["lyrics"]) < MIN_CHARS:
            print(f"!! Skipping short lyrics: {p.name}")
            continue
        records.append(r)

    N = len(records)
    if N == 0:
        raise SystemExit("No valid songs loaded (all were empty/invalid).")

    print(f"Loaded {N} songs from {LYRICS_DIR}/")

    # optional: JSONL for quick inspection
    with open(ART / "songs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- OpenCLIP text encoder (match app) ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP: {MODEL} / {PRETRAINED} on {device} …")
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL)

    # ---------- encode ----------
    texts = prepare_texts(records)
    print(f"Encoding {len(texts)} texts to 512-D CLIP space …")
    X = embed_texts(texts, model, tokenizer, device)
    if X.ndim != 2 or X.shape[1] != 512:
        raise SystemExit(f"Unexpected embedding shape {X.shape}; expected (N, 512)")

    ids  = [r["id"] for r in records]
    meta = [{"id": r["id"], "title": r["title"], "artist": r["artist"]} for r in records]

    # ---------- 2-level k-means (residual) ----------
    K1 = pick_k(N)
    K2 = pick_k(N)
    print(f"Fitting k-means: L1={K1}, L2={K2}")

    km1 = MiniBatchKMeans(n_clusters=K1, batch_size=8192, n_init=10, random_state=SEED)
    c1  = km1.fit_predict(X)
    residual = X - km1.cluster_centers_[c1]

    km2 = MiniBatchKMeans(n_clusters=K2, batch_size=8192, n_init=10, random_state=SEED)
    c2  = km2.fit_predict(residual)

    # { song_id (str) : (b1, b2) }
    bucket_map: Dict[str, Tuple[int, int]] = {
        str(ids[i]): (int(c1[i]), int(c2[i])) for i in range(N)
    }

    # ---------- save ----------
    np.save(ART / "embeddings.npy", X.astype("float32"))
    with open(ART / "ids.json", "w", encoding="utf-8") as f:  json.dump(ids, f)
    with open(ART / "meta.json", "w", encoding="utf-8") as f: json.dump(meta, f)
    with open(ART / "km1.pkl", "wb") as f: pickle.dump(km1, f)
    with open(ART / "km2.pkl", "wb") as f: pickle.dump(km2, f)
    with open(ART / "bucket_map.pkl", "wb") as f: pickle.dump(bucket_map, f)

    print("\nArtifacts written to ./artifacts:")
    for name in ["embeddings.npy","ids.json","meta.json","km1.pkl","km2.pkl","bucket_map.pkl","songs.jsonl"]:
        p = ART / name
        if p.exists():
            print(" -", p)

if __name__ == "__main__":
    main()

