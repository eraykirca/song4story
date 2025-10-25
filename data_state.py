import json, pickle, pathlib, random, secrets
from typing import Dict, Optional, Tuple, List
import numpy as np
import faiss
import torch
import streamlit as st

ARTIFACTS_DIR = "artifacts"
TOPK          = 50
SEED          = 42
USE_ROUTER_TOPB = 3
MMR_ALPHA       = 0.65
ARTIST_CAP      = 1
JITTER_EPS      = 0.003

random.seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)

st.session_state.setdefault("rng_seed", secrets.randbits(32))
rng = np.random.default_rng(st.session_state["rng_seed"])

st.session_state.setdefault("taste_seen_rows", set())
st.session_state.setdefault("taste_seen_artists", set())

def load_jsonl_meta(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

@st.cache_resource(show_spinner=False)
def load_artifacts():
    art = pathlib.Path(ARTIFACTS_DIR)
    have_full = all([
        (art / "embeddings.npy").exists(),
        (art / "ids.json").exists(),
        (art / "meta.json").exists(),
    ])
    if have_full:
        emb = np.load(art/"embeddings.npy").astype("float32")
        with open(art/"ids.json","r",encoding="utf-8") as f: ids = json.load(f)
        with open(art/"meta.json","r",encoding="utf-8") as f: meta = json.load(f)
        bucket_map = None
        if (art/"bucket_map.pkl").exists():
            with open(art/"bucket_map.pkl","rb") as f:
                bucket_map = pickle.load(f)
        return emb, ids, meta, bucket_map

embeddings, song_ids, song_meta, bucket_map = load_artifacts()

# --- Load k-means + cluster maps ---
KM1, KM2 = None, None
try:
    with open(pathlib.Path(ARTIFACTS_DIR) / "km1.pkl", "rb") as f:
        KM1 = pickle.load(f)
    with open(pathlib.Path(ARTIFACTS_DIR) / "km2.pkl", "rb") as f:
        KM2 = pickle.load(f)
except Exception:
    KM1 = KM2 = None

# Build: row -> b1 and members per cluster
row2b1 = [None] * len(song_meta)
cluster_members: Dict[int, list] = {}
if bucket_map is not None:
    for row, m in enumerate(song_meta):
        sid = str(m["id"])
        if sid in bucket_map:
            b1 = bucket_map[sid][0] if isinstance(bucket_map[sid], (list, tuple)) else bucket_map[sid]
            row2b1[row] = int(b1)
            cluster_members.setdefault(int(b1), []).append(row)

# Fallback infer b1 from centers if no bucket_map
if bucket_map is None and KM1 is not None:
    centers = KM1.cluster_centers_.astype("float32")
    centers /= np.maximum(1e-8, np.linalg.norm(centers, axis=1, keepdims=True))
    ip = embeddings @ centers.T
    inferred = np.argmax(ip, axis=1)
    for row, b1 in enumerate(inferred):
        row2b1[row] = int(b1)
        cluster_members.setdefault(int(b1), []).append(row)

# Archetypes
archetype_row = {}
if KM1 is not None and cluster_members:
    C = KM1.cluster_centers_.astype("float32")
    C /= np.maximum(1e-8, np.linalg.norm(C, axis=1, keepdims=True))
    E = embeddings
    for b1, rows in cluster_members.items():
        if not rows: continue
        cos = (E[rows] @ C[b1])
        archetype_row[b1] = rows[int(np.argmax(cos))]

CENTERS_NORM = None
if KM1 is not None:
    CENTERS_NORM = KM1.cluster_centers_.astype("float32")
    CENTERS_NORM /= np.maximum(1e-8, np.linalg.norm(CENTERS_NORM, axis=1, keepdims=True))

# Auto-tuned taste params
def _compute_taste_params():
    N = len(embeddings)
    sizes = np.array([len(v) for v in cluster_members.values()]) if cluster_members else np.array([N])
    C = int(len(sizes))
    avg = float(sizes.mean()) if sizes.size > 0 else float(N)

    p = sizes / max(1, sizes.sum())
    entropy = float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())
    norm_entropy = (entropy / np.log2(C)) if C > 1 else 1.0

    base = max(10, int(round(np.sqrt(max(1, N)) * 0.55)))
    roomy_scale = float(np.clip(avg / 40.0, 0.75, 1.5))
    k_early = int(np.clip(base * roomy_scale, 10, 48))
    k_late  = int(np.clip(k_early * 1.3,     12, 96))

    cap_early = 1
    cap_mid   = 1
    if avg >= 80 and norm_entropy >= 0.85:
        cap_late = 3
    elif avg >= 35 and norm_entropy >= 0.70:
        cap_late = 2
    else:
        cap_late = 1

    return k_early, k_late, cap_early, cap_mid, cap_late

import numpy as np
K_EACH_EARLY, K_EACH_LATE, CAP_EARLY, CAP_MID, CAP_LATE = _compute_taste_params()

DIM = embeddings.shape[1]
id2row = {str(m["id"]): i for i, m in enumerate(song_meta)}

@st.cache_resource(show_spinner=False)
def build_faiss_ip(vecs: np.ndarray):
    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 128
    index.add(vecs.astype("float32"))
    return index

faiss_index = build_faiss_ip(embeddings)
