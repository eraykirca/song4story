import json, pickle, pathlib
import numpy as np
import streamlit as st

art = pathlib.Path("artifacts")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load vectors + metadata + bucket_map (if present). Else build from starter_lyrics/*.json."""
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
                bucket_map = pickle.load(f)   # dict: id -> (b1,b2)
        return emb, ids, meta, bucket_map

embeddings, song_ids, song_meta, bucket_map = load_artifacts()

# --- Load k-means + cluster maps (Level-1 clusters) ---
KM1, KM2 = None, None
try:
    with open(pathlib.Path("artifacts") / "km1.pkl", "rb") as f:
        KM1 = pickle.load(f)
    with open(pathlib.Path("artifacts") / "km2.pkl", "rb") as f:
        KM2 = pickle.load(f)
except Exception:
    KM1 = KM2 = None

# Build: row -> b1 and members per cluster
row2b1 = [None] * len(song_meta)
cluster_members = {}  # b1 -> [rows]
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

# Precompute archetype (closest to centroid) per b1
archetype_row = {}
if KM1 is not None and cluster_members:
    C = KM1.cluster_centers_.astype("float32")
    C /= np.maximum(1e-8, np.linalg.norm(C, axis=1, keepdims=True))
    E = embeddings  # already normalized
    for b1, rows in cluster_members.items():
        if not rows:
            continue
        cos = (E[rows] @ C[b1])
        archetype_row[b1] = rows[int(np.argmax(cos))]

# Normalized L1 centers (for top-B routing)
CENTERS_NORM = None
if KM1 is not None:
    CENTERS_NORM = KM1.cluster_centers_.astype("float32")
    CENTERS_NORM /= np.maximum(1e-8, np.linalg.norm(CENTERS_NORM, axis=1, keepdims=True))