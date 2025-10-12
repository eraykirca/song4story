import os, io, json, glob, pickle, random, pathlib, re
from typing import List, Dict, Optional, Tuple
from wordfreq import zipf_frequency
import numpy as np
from PIL import Image

import streamlit as st
import faiss
import torch
import open_clip
import streamlit.components.v1 as components
import secrets


import urllib.parse
ART = pathlib.Path("artifacts")

# --- Compact Spotify play button helpers ---
import urllib.parse

# --- Spotify embed helpers ---
# --- Spotify embed helpers (standard size) ---
def spotify_embed_html(track_id: str, width: int = 300, height: int = 152) -> str:
    """Standard Spotify embed card with title, artist, and album art."""
    return f"""
    <div style="display:flex;justify-content:center;margin:8px 0 6px;">
        <iframe style="border-radius:12px"
                src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                width="{width}" height="{height}" frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"></iframe>
    </div>
    """

def spotify_search_link(title: str, artist: str) -> str:
    import urllib.parse
    q = urllib.parse.quote_plus(f"{title} {artist}")
    return f"https://open.spotify.com/search/{q}"

def render_song_card(song_meta_obj: dict, pick_key: str, show_pick: bool = True):
    sid = str(song_meta_obj.get("id", ""))
    spid = SPOTIFY_IDS.get(sid)

    if spid:
        st.markdown(spotify_embed_html(spid), unsafe_allow_html=True)
    else:
        # Fallback if ID missing
        st.link_button("▶ Open in Spotify",
                       spotify_search_link(song_meta_obj["title"], song_meta_obj["artist"]))

    if show_pick:
        if st.button("Pick", key=pick_key):
            return True
    return False



def load_spotify_ids() -> dict:
    """Return {song_id: spotify_id} from artifacts/songs.jsonl."""
    import json
    from pathlib import Path
    ARTIFACTS = Path("artifacts")
    fp = ARTIFACTS / "songs.jsonl"
    out = {}
    if not fp.exists():
        return out
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("id", "")).strip()
            spid = (obj.get("spotify_id") or "").strip()
            if sid and spid:
                out[sid] = spid
    return out

SPOTIFY_IDS = load_spotify_ids()


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()  # old Streamlit fallback

# Config
ARTIFACTS_DIR = "artifacts"
TOPK          = 50                    # candidates before top-N
SEED          = 42
USE_ROUTER_TOPB = 3      # search top-B clusters instead of just 1 (set 1 to keep old behavior)
MMR_ALPHA       = 0.65   # 1.0 = pure relevance, 0.0 = pure diversity
ARTIST_CAP      = 1      # max results per artist in a single list
JITTER_EPS      = 0.003  # small random tie-break on scores

random.seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)

# Per-session RNG (stable randomness per browser session)
st.session_state.setdefault("rng_seed", secrets.randbits(32))
rng = np.random.default_rng(st.session_state["rng_seed"])

# Global memory for the taste warm-up across rounds (this session)
st.session_state.setdefault("taste_seen_rows", set())
st.session_state.setdefault("taste_seen_artists", set())

# Try to import clip-interrogator for tags (optional)
HAS_CI = False
try:
    from clip_interrogator import Config as CIConfig, Interrogator
    HAS_CI = True
except Exception:
    HAS_CI = False

# ---------- Model (shared for text & images) ----------
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

# ---------- Data loading ----------
def load_jsonl_meta(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load vectors + metadata + bucket_map (if present). Else build from starter_lyrics/*.json."""
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
                bucket_map = pickle.load(f)   # dict: id -> (b1,b2)
        return emb, ids, meta, bucket_map

embeddings, song_ids, song_meta, bucket_map = load_artifacts()

# --- Load k-means + cluster maps (Level-1 clusters) ---
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


# --- Auto-tune taste-test params based on dataset/cluster stats ---
def _compute_taste_params():
    N = len(embeddings)
    sizes = np.array([len(v) for v in cluster_members.values()]) if cluster_members else np.array([N])
    C = int(len(sizes))
    avg = float(sizes.mean()) if sizes.size > 0 else float(N)

    # normalized entropy of cluster sizes (1=balanced, 0=imbalanced)
    p = sizes / max(1, sizes.sum())
    entropy = float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())
    norm_entropy = (entropy / np.log2(C)) if C > 1 else 1.0

    # neighbor breadth ~ sqrt(N), scaled by "roominess" (avg songs/cluster)
    base = max(10, int(round(np.sqrt(max(1, N)) * 0.55)))
    roomy_scale = float(np.clip(avg / 40.0, 0.75, 1.5))
    k_early = int(np.clip(base * roomy_scale, 10, 48))     # rounds 2–3
    k_late  = int(np.clip(k_early * 1.3,     12, 96))     # round 4+

    # per-cluster caps: relax only if clusters are big & balanced
    cap_early = 1
    cap_mid   = 1
    if avg >= 80 and norm_entropy >= 0.85:
        cap_late = 3
    elif avg >= 35 and norm_entropy >= 0.70:
        cap_late = 2
    else:
        cap_late = 1

    return k_early, k_late, cap_early, cap_mid, cap_late

K_EACH_EARLY, K_EACH_LATE, CAP_EARLY, CAP_MID, CAP_LATE = _compute_taste_params()

DIM = embeddings.shape[1]

# quick ID→row index map
id2row = {str(m["id"]): i for i, m in enumerate(song_meta)}

# ---------- FAISS index ----------
@st.cache_resource(show_spinner=False)
def build_faiss_ip(vecs: np.ndarray):
    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 128    # better recall at query time
    index.add(vecs.astype("float32"))
    return index


faiss_index = build_faiss_ip(embeddings)

# ---------- TIGER-lite router ----------
def route_bucket(x: np.ndarray) -> Optional[Tuple[int,int]]:
    """Return (b1,b2) if bucket_map exists, else None."""
    if bucket_map is None:
        return None
    # simple L1 only or (b1,b2) if you stored 2-level routes
    # Here we find the nearest by comparing to a representative mapping:
    # bucket_map was saved as: {song_id: (b1,b2)} — we’ll approximate by using
    # the bucket of the nearest song to x among a random sample.
    sample = np.random.choice(len(embeddings), size=min(1000, len(embeddings)), replace=False)
    D,I = faiss_index.search(x[None,:].astype("float32"), 1)
    # nearest existing vector’s id → bucket
    nearest_idx = int(I[0][0])
    sid = str(song_meta[nearest_idx]["id"])
    return bucket_map.get(sid)

def search(x: np.ndarray, n: int = 5, k: int = TOPK) -> List[Dict]:
    """
    Top-B cluster routing + MMR diversity + artist cap + tiny jitter.
    """
    q = x.astype("float32").reshape(-1)
    cand_rows, cand_scores = [], []

    def pack(rows, scores):
        out = []
        for r, sc in zip(rows, scores):
            m = song_meta[int(r)].copy()
            m["clip_score"] = float(sc)
            out.append(m)
        return out

    # ---- collect candidates from top-B L1 clusters (if available)
    if CENTERS_NORM is not None and cluster_members:
        B = max(1, int(USE_ROUTER_TOPB))
        sims = (CENTERS_NORM @ q)                           # (K1,)
        topB = np.argsort(-sims)[:B]

        # per-cluster depth: pull more than needed; MMR will downselect
        per = max(int(np.ceil(k / B)) * 2, n * 2)
        for b1 in topB:
            rows = cluster_members.get(int(b1), [])
            if not rows:
                continue
            subv = embeddings[rows]
            sub = faiss.IndexFlatIP(DIM)
            sub.add(subv)
            D, I = sub.search(q[None, :], min(per, len(rows)))
            for sc, idx in zip(D[0], I[0]):
                cand_rows.append(rows[int(idx)])
                cand_scores.append(float(sc))
    else:
        # fallback: global top-k
        D, I = faiss_index.search(q[None, :], k)
        cand_rows = [int(i) for i in I[0]]
        cand_scores = [float(s) for s in D[0]]

    if not cand_rows:
        return []

    # ---- de-dup (keep first hit for each row)
    seen, rows_u, scores_u = set(), [], []
    for r, sc in zip(cand_rows, cand_scores):
        if r in seen:
            continue
        seen.add(r)
        rows_u.append(int(r))
        scores_u.append(float(sc))

    # ---- tiny jitter to break ties
    if JITTER_EPS > 0:
        jitter = (np.random.rand(len(scores_u)) * 2 - 1) * float(JITTER_EPS)
        scores_perturbed = (np.asarray(scores_u, dtype=np.float32) + jitter).tolist()
    else:
        scores_perturbed = scores_u

    # ---- MMR diversify, then apply artist cap
    mmr_rows = _mmr_select(rows_u, scores_perturbed, take_m=max(n * 3, n + 3), alpha=MMR_ALPHA)

    final_rows, artist_counts = [], {}
    for r in mmr_rows:
        artist = song_meta[int(r)]["artist"]
        if artist and artist_counts.get(artist, 0) >= ARTIST_CAP:
            continue
        final_rows.append(int(r))
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if len(final_rows) >= n:
            break

    # backfill if short (ignore artist cap)
    if len(final_rows) < n:
        for r in rows_u:
            if r not in final_rows:
                final_rows.append(int(r))
                if len(final_rows) >= n:
                    break

    # return with original (non-jittered) FAISS scores
    score_map = {r: s for r, s in zip(rows_u, scores_u)}
    return pack(final_rows, [score_map.get(r, 0.0) for r in final_rows])


# ---------- Auto-tags via clip-interrogator (optional) ----------
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
        if counts[w] <= 2:  # allow at most 2 repeats of each word
            kept.append(w)
    return " ".join(kept)

@torch.inference_mode()
def auto_tags_for_image(
    pil: Image.Image,
    MAX_TAGS: int = 6,
    THRESH_IMG: float = 1.0,
    THRESH_TAG: float = 0.7,
):
    """
    Generate up to MAX_TAGS filtered tags.
    - squash 3+ repeats per tag
    - drop likely personal names (no common words)
    - seed with first 4 tags, then add similar tags
    - progressively relax thresholds to try to reach MAX_TAGS
    - if still short, fill by top image similarity
    Returns: List[str] (kept tags only)
    """

    def trim_3plus_repeats(tag: str) -> str:
        words = re.findall(r"[^\W\d_]+", tag.lower())
        kept, counts = [], {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if counts[w] <= 2:
                kept.append(w)
        return " ".join(kept)

    # 1) raw prompt -> raw tag list (we only *use* filtered tags)
    prompt = interrogator.interrogate_fast(pil)
    raw_tags = [t.strip() for t in re.split(r",|\band\b", prompt) if t.strip()]
    raw_tags = [trim_3plus_repeats(t) for t in raw_tags]

    # 2) drop likely names (keep if any common word, Zipf >= 4.1)
    def keep(tag: str) -> bool:
        return any(zipf_frequency(w, "en") >= 4.1 for w in re.findall(r"[a-zA-Z']+", tag))

    all_tags = [t for t in raw_tags if keep(t)]
    if not all_tags:
        return []

    # 3) seed with first up to 4
    seed = all_tags[: min(4, len(all_tags))]
    rest = all_tags[len(seed):]

    # 4) embeddings
    img_t = preprocess(pil).unsqueeze(0).to(device)
    E_img = clip_model.encode_image(img_t)
    E_img = torch.nn.functional.normalize(E_img, dim=-1)
    tok_all = tokenizer(all_tags).to(device)
    tag_emb = clip_model.encode_text(tok_all)
    tag_emb = torch.nn.functional.normalize(tag_emb, dim=-1)

    idx = {t: i for i, t in enumerate(all_tags)}
    def v(t): return tag_emb[idx[t]]

    kept = list(seed)

    # 5) try to add more with progressive relaxation
    relax_schedule = [
        (THRESH_IMG, THRESH_TAG),     # e.g. (1.0, 0.7)
        (0.95, 0.65),
        (0.90, 0.60),
        (0.85, 0.55),
        (0.80, 0.50),
    ]

    # helper to attempt adding with thresholds
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

    # 6) if still short, fill by top image similarity (no more filtering)
    if len(kept) < MAX_TAGS and rest:
        sims = [(t, float(E_img @ v(t).T)) for t in rest if t not in kept]
        sims.sort(key=lambda x: x[1], reverse=True)
        for t, _ in sims:
            if len(kept) >= MAX_TAGS:
                break
            kept.append(t)

    # cap to MAX_TAGS
    return kept[:MAX_TAGS]

# ---------- Taste model (no training) ----------
class TasteState:
    TARGET_PICKS = 5  # how many choices the user should make

    def __init__(self):
        self.round = 0
        self.history = []    # list of tuples: (choice_idx, candidate_rows)
        self.likes = []      # row indices user chose
        self.dislikes = []   # row indices not chosen in each round
        self.seen = set()    # rows shown to the user (avoid repeats)
        self.current_cands = None  # the 5 rows currently on screen

    def picks_done(self) -> int:
        return len(self.likes)

    def picks_left(self) -> int:
        return max(0, self.TARGET_PICKS - self.picks_done())

    def done(self) -> bool:
        return self.picks_done() >= self.TARGET_PICKS

    def user_vector(self) -> Optional[np.ndarray]:
        if not self.likes and not self.dislikes:
            return None
        like_vecs = embeddings[self.likes] if self.likes else None
        dislike_vecs = embeddings[self.dislikes] if self.dislikes else None
        if like_vecs is None:
            return None
        luv = like_vecs.mean(0)
        if dislike_vecs is not None and len(self.dislikes) > 0:
            dv = dislike_vecs.mean(0)
            uv = luv - 0.25 * dv
        else:
            uv = luv
        uv = uv / max(1e-8, np.linalg.norm(uv))
        return uv.astype("float32")

def _faiss_neighbors(seed_rows, k_each=10, exclude=None):
    if not seed_rows:
        return []
    exclude = set(exclude or [])
    q = embeddings[np.array(seed_rows)]
    D, I = faiss_index.search(q.astype("float32"), k_each + 1)
    out = []
    for rlist in I:
        for r in rlist:
            if r == -1:
                continue
            if int(r) in seed_rows or int(r) in exclude:
                continue
            out.append(int(r))
    seen = set(); uniq = []
    for r in out:
        if r not in seen:
            seen.add(r); uniq.append(r)
    return uniq

def _widen_neighbors(seed_rows, base_k, used, min_unique, hard_cap=128):
    """Call _faiss_neighbors, widening k_each until we have enough unique rows or hit a cap."""
    if not seed_rows:
        return []
    k = int(base_k)
    neigh = _faiss_neighbors(seed_rows, k_each=k, exclude=used)
    while len(neigh) < min_unique and k < hard_cap:
        k = min(k * 2, hard_cap)
        neigh = _faiss_neighbors(seed_rows, k_each=k, exclude=used)
    return neigh

def _dislike_vector(use_last_round=True):
    """
    Mean embedding of dislikes. If use_last_round and present, use only
    the 5 from the last round (stronger 'push-away'); otherwise all dislikes.
    Returns None if no dislikes.
    """
    rows = []
    if use_last_round and "last_round_rows" in st.session_state:
        rows = [r for r in st.session_state["last_round_rows"] if r in taste.dislikes]
    if not rows:
        rows = list(set(taste.dislikes))
    if not rows:
        return None
    V = embeddings[np.array(rows)]
    v = V.mean(0).astype("float32")
    v /= max(1e-8, np.linalg.norm(v))
    return v

def _clusters_sorted_by_similarity_to_vec(vec):
    """(b1, sim) high→low for an arbitrary vector."""
    if vec is None or KM1 is None or not cluster_members:
        return []
    C = KM1.cluster_centers_.astype("float32")
    C /= np.maximum(1e-8, np.linalg.norm(C, axis=1, keepdims=True))
    sims = (C @ vec.astype("float32"))
    order = np.argsort(-sims)
    return [(int(b1), float(sims[ int(b1) ])) for b1 in order]


def _user_vector_or_centroid():
    uv = taste.user_vector()
    if uv is not None:
        return uv
    if archetype_row:
        rows = list(archetype_row.values())
        m = embeddings[rows].mean(0).astype("float32")
        return m / max(1e-8, np.linalg.norm(m))
    return None

def _clusters_sorted_by_similarity_to(vec):
    if KM1 is None or not cluster_members or vec is None:
        return []
    C = KM1.cluster_centers_.astype("float32")
    C /= np.maximum(1e-8, np.linalg.norm(C, axis=1, keepdims=True))
    sims = (C @ vec.astype("float32"))
    order = np.argsort(-sims)
    return [(int(b1), float(sims[b1])) for b1 in order]

def _sample_diverse_from_clusters(k, used_rows):
    picks = []
    used_rows = set(used_rows)
    for b1, row in archetype_row.items():
        if len(picks) >= k:
            break
        if row not in used_rows:
            picks.append(row)
    return picks

def _sample_from_far_clusters(k, uv, used_rows):
    if uv is None or KM1 is None or not cluster_members:
        return []
    far_first = list(reversed(_clusters_sorted_by_similarity_to(uv)))
    picks, used_rows = [], set(used_rows)
    for b1, _ in far_first:
        if len(picks) >= k:
            break
        cand = archetype_row.get(b1, None)
        if cand is not None and cand not in used_rows:
            picks.append(cand)
        else:
            for r in cluster_members[b1]:
                if r not in used_rows:
                    picks.append(r); break
    return picks[:k]

def _ensure_per_cluster_diversity(rows, max_per_cluster=1):
    kept, count = [], {}
    for r in rows:
        b1 = row2b1[r] if r < len(row2b1) else None
        if b1 is None:
            kept.append(r); continue
        c = count.get(b1, 0)
        if c < max_per_cluster:
            kept.append(r); count[b1] = c + 1
    return kept

def _mmr_select(candidate_rows, candidate_scores, take_m, alpha=MMR_ALPHA):
    """
    Greedy Maximal Marginal Relevance:
    score = alpha * relevance - (1-alpha) * max_sim_to_picked
    Returns rows in selected order (up to take_m).
    """
    if not candidate_rows:
        return []
    rows = list(candidate_rows)
    rel  = np.asarray(candidate_scores, dtype=np.float32)

    E = embeddings[np.array(rows)]                  # (M, D), normalized
    S = (E @ E.T).astype(np.float32)                # (M, M) cosine sims

    picked = []
    unused = list(range(len(rows)))

    # start: best relevance
    j0 = int(np.argmax(rel))
    picked.append(j0)
    unused.remove(j0)

    while len(picked) < min(take_m, len(rows)):
        best_j, best_val = None, -1e9
        for j in unused:
            div = float(S[j, picked].max()) if picked else 0.0
            val = alpha * float(rel[j]) - (1.0 - alpha) * div
            if val > best_val:
                best_val, best_j = val, j
        picked.append(best_j)
        unused.remove(best_j)

    return [rows[j] for j in picked]

def _artist(row_idx):
    return song_meta[int(row_idx)]["artist"]

def _pick_cluster_rep(b1, blocked_rows, top_m=3):
    """
    Pick a representative from cluster b1:
    - Score rows by cosine to the L1 centroid
    - Take top_m, then randomly choose 1 with rng (to vary across sessions)
    - Skip rows already blocked
    """
    rows = cluster_members.get(int(b1), [])
    if not rows:
        return None

    # centroid vector (normalized once in CENTERS_NORM)
    c = CENTERS_NORM[int(b1)] if CENTERS_NORM is not None else None
    if c is None:
        return None

    E = embeddings[np.array(rows)]            # normalized
    sims = (E @ c).astype(np.float32)         # higher = closer to centroid
    order = np.argsort(-sims)

    picks = []
    for idx in order:
        r = rows[int(idx)]
        if r in blocked_rows:
            continue
        picks.append(r)
        if len(picks) >= top_m:
            break

    if not picks:
        return None

    # random choice among the top_m to avoid same rep each launch
    return int(rng.choice(picks))


def policy_sample_candidates(k=5):
    """
    Taste sampler with intent-aware branch:
      - If user pressed "I don’t like any of these":
          * If 0 likes: push AWAY from last round → far clusters, avoid last clusters.
          * If ≥1 like: pull TOWARD likes → neighbors first, then near clusters.
      - Otherwise: use the existing round-based logic (unchanged below).
    """
    # exhaustion guard (yours)
    blocked = set(taste.seen) | set(taste.likes) | set(taste.dislikes)
    if len(embeddings) - len(blocked) < k:
        taste.seen = set(taste.likes) | set(taste.dislikes)

    used_rows = set(taste.seen) | set(taste.likes) | set(taste.dislikes) | set(st.session_state["taste_seen_rows"])
    used_artists = set(st.session_state["taste_seen_artists"])

    uv = _user_vector_or_centroid()
    round_cap = 1 if taste.round <= 3 else 2  # artist cap per round
    per_artist = {}

    def _accept(r):
        a = _artist(r)
        if a and per_artist.get(a, 0) >= round_cap:
            return False
        return True

    def _commit(rows):
        for r in rows:
            a = _artist(r)
            per_artist[a] = per_artist.get(a, 0) + 1
        st.session_state["taste_seen_rows"].update(int(r) for r in rows)
        st.session_state["taste_seen_artists"].update(_artist(r) for r in rows)

    # -------- NEW: react only if the user pressed "none" on the previous round
    pressed_none = st.session_state.get("pressed_none_last_round", False)
    if pressed_none:
        cands = []

        # A) If 0 likes so far → push AWAY from last round's 5
        if taste.picks_done() == 0 and KM1 is not None and cluster_members:
            # clusters to avoid: those that produced the disliked 5 we just showed
            avoid_clusters = set()
            for r in st.session_state.get("last_round_rows", []):
                b = row2b1[r] if r < len(row2b1) else None
                if b is not None:
                    avoid_clusters.add(int(b))

            dv = _dislike_vector(use_last_round=True)
            # far clusters first (least similar to dislike vector)
            far = [b for (b, _) in reversed(_clusters_sorted_by_similarity_to_vec(dv))] if dv is not None else list(cluster_members.keys())
            far = list(rng.permutation(far)) if len(far) > 1 else far

            for b1 in far:
                if len(cands) >= k:
                    break
                if int(b1) in avoid_clusters:
                    continue
                rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                if rep is None or rep in used_rows or not _accept(rep):
                    continue
                cands.append(int(rep))

            # backfill if needed from any other clusters (shuffled)
            if len(cands) < k and cluster_members:
                rest = [b for b in cluster_members.keys() if b not in avoid_clusters]
                rest = list(rng.permutation(rest))
                for b1 in rest:
                    if len(cands) >= k:
                        break
                    rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                    if rep is None or rep in used_rows or not _accept(rep):
                        continue
                    cands.append(int(rep))

            _commit(cands)
            st.session_state["pressed_none_last_round"] = False  # consume flag
            return np.array(cands[:k], dtype=int)

        # B) If ≥1 like → pull TOWARD: neighbors first, then near clusters
        like_seed = taste.likes[-5:] if taste.likes else []
        neigh = _faiss_neighbors(like_seed, k_each=K_EACH_EARLY, exclude=used_rows) if like_seed else []
        if neigh:
            neigh = list(rng.permutation(neigh))

        cands = []
        for r in neigh:
            if len(cands) >= k:
                break
            if r in used_rows or not _accept(r):
                continue
            cands.append(int(r))

        if len(cands) < k and uv is not None and KM1 is not None:
            near = [b for (b, _) in _clusters_sorted_by_similarity_to(uv)]  # high→low sim
            for b1 in near:
                if len(cands) >= k:
                    break
                rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                if rep is None or rep in used_rows or not _accept(rep):
                    continue
                cands.append(int(rep))

        # final fallback: any unseen
        if len(cands) < k:
            all_rows = np.setdiff1d(np.arange(len(embeddings)),
                                    np.fromiter(used_rows | set(cands), int, count=len(used_rows | set(cands))) if used_rows else [])
            all_rows = list(rng.permutation(all_rows))
            for r in all_rows:
                if len(cands) >= k:
                    break
                if not _accept(r):
                    continue
                cands.append(int(r))

        _commit(cands)
        st.session_state["pressed_none_last_round"] = False  # consume flag
        return np.array(cands[:k], dtype=int)

    # -------- NO BUTTON: fall back to your existing logic (UNCHANGED BELOW)

    # -------- Rounds 0–1: pick random clusters' representatives
    if taste.round <= 1 and CENTERS_NORM is not None and cluster_members:
        cands = []
        cluster_ids = list(cluster_members.keys())
        cluster_ids = list(rng.permutation(cluster_ids))  # random order each session

        for b1 in cluster_ids:
            if len(cands) >= k:
                break
            rep = _pick_cluster_rep(b1, blocked_rows=used_rows, top_m=3)
            if rep is None:
                continue
            if rep in used_rows:
                continue
            if not _accept(rep):
                continue
            cands.append(int(rep))

        # if short, fill from far clusters (least similar to uv) or any unseen
        if len(cands) < k:
            if uv is not None and KM1 is not None:
                order = _clusters_sorted_by_similarity_to(uv)  # high->low
                far = [b for (b, _) in reversed(order)]
            else:
                far = cluster_ids
            for b1 in far:
                if len(cands) >= k:
                    break
                rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                if rep is None or rep in used_rows or not _accept(rep):
                    continue
                cands.append(int(rep))

        _commit(cands)
        return np.array(cands[:k], dtype=int)

    # -------- Rounds 2–3: neighbors + diverse fillers
    if taste.round <= 3:
        like_seed = taste.likes[-3:] if taste.likes else []
        neigh = _faiss_neighbors(like_seed, k_each=K_EACH_EARLY, exclude=used_rows) if like_seed else []
        if neigh:
            neigh = list(rng.permutation(neigh))

        cands = []
        for r in neigh:
            if len(cands) >= min(2, k):
                break
            if r in used_rows or not _accept(r):
                continue
            cands.append(int(r))

        cluster_ids = list(cluster_members.keys())
        cluster_ids = list(rng.permutation(cluster_ids))
        for b1 in cluster_ids:
            if len(cands) >= k:
                break
            rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
            if rep is None or rep in used_rows or not _accept(rep):
                continue
            cands.append(int(rep))

        _commit(cands)
        return np.array(cands[:k], dtype=int)

    # -------- Round 4+: exploit + wildcard
    like_seed = taste.likes[-5:] if taste.likes else []
    neigh = _faiss_neighbors(like_seed, k_each=K_EACH_LATE, exclude=used_rows) if like_seed else []
    if neigh:
        neigh = list(rng.permutation(neigh))

    cands = []
    for r in neigh:
        if len(cands) >= min(3, k):
            break
        if r in used_rows or not _accept(r):
            continue
        cands.append(int(r))

    if uv is not None and KM1 is not None and cluster_members:
        far = [b for (b, _) in reversed(_clusters_sorted_by_similarity_to(uv))]
        for b1 in far:
            if len(cands) >= k:
                break
            rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
            if rep is None or rep in used_rows or not _accept(rep):
                continue
            cands.append(int(rep))

    if len(cands) < k:
        all_rows = np.setdiff1d(np.arange(len(embeddings)),
                                np.fromiter(used_rows | set(cands), int, count=len(used_rows | set(cands))) if used_rows else [])
        all_rows = list(rng.permutation(all_rows))
        for r in all_rows:
            if len(cands) >= k:
                break
            if not _accept(r):
                continue
            cands.append(int(r))

    _commit(cands)
    return np.array(cands[:k], dtype=int)

def sample_unseen_candidates(k: int = 5) -> np.ndarray:
    # Block songs we've already shown or labeled
    blocked = set(taste.seen) | set(taste.likes) | set(taste.dislikes)
    all_rows = np.arange(len(embeddings))
    if blocked:
        blocked_arr = np.fromiter(blocked, dtype=int)
        available = np.setdiff1d(all_rows, blocked_arr, assume_unique=False)
    else:
        available = all_rows

    # If we run out, reset 'seen' but keep likes/dislikes blocked
    if len(available) < k:
        taste.seen = set(taste.likes) | set(taste.dislikes)
        blocked = set(taste.seen)
        blocked_arr = np.fromiter(blocked, dtype=int) if blocked else np.array([], dtype=int)
        available = np.setdiff1d(all_rows, blocked_arr, assume_unique=False)

    # Sample without replacement
    k_eff = min(k, len(available))
    return np.random.choice(available, size=k_eff, replace=False)

if "taste" not in st.session_state:
    st.session_state.taste = TasteState()
taste = st.session_state.taste

def _reset_step1():
    """
    Hard reset of Step 1 (taste warm-up), like a page refresh for this section.
    - Drop the TasteState object, so the 'if "taste" not in st.session_state' code rebuilds it.
    - Drop the per-session RNG seed, so cluster order + picks change.
    - Drop the Step-1 seen caches (rows/artists).
    """
    for k in ("taste", "taste_seen_rows", "taste_seen_artists", "rng_seed"):
        if k in st.session_state:
            del st.session_state[k]
    # also clear any persisted "current_cands" for the in-progress round
    if "taste" in st.session_state and hasattr(st.session_state.taste, "current_cands"):
        st.session_state.taste.current_cands = None
    rerun()


# ---------- UI ----------
st.set_page_config(page_title="Song4Story", page_icon="🎵")

st.title("Song4Story — pick the best song for your IG story!")
st.header(f"Your music taste + Your image = Your song!")
st.caption(f"by Eray Kırca")
st.header("Step 1 — Quick taste warm-up (5 rounds)")
if not taste.done():
    # Persist the 5 candidates until the user clicks
    if taste.current_cands is None:
        taste.current_cands = policy_sample_candidates(k=5)

    cand_rows = np.array(taste.current_cands)
    st.session_state.setdefault("pressed_none_last_round", False)

    # NEW: remember this round’s rows
    st.session_state["last_round_rows"] = [int(r) for r in cand_rows]
    picked = None

    # Progress text: "X selected — need Y more"
    st.caption(f"{taste.picks_done()} selected — need {taste.picks_left()} more")

    left_col, right_col = st.columns(2)

    # Left column: first 3 songs stacked
    with left_col:
        for j in range(min(3, len(cand_rows))):
            row = int(cand_rows[j])
            m = song_meta[row]

            sid = str(m.get("id", ""))
            spid = SPOTIFY_IDS.get(sid)

            if spid:
                st.markdown(spotify_embed_html(spid), unsafe_allow_html=True)
            else:
                st.link_button("Open in Spotify", spotify_search_link(m["title"], m["artist"]))

            if st.button("Pick", key=f"pick_left_{taste.picks_done()}_{j}"):
                picked = j

    # Right column: remaining (up to 2) stacked
    with right_col:
        # Right column: remaining (up to 2) stacked
        for j in range(3, min(5, len(cand_rows))):
            row = int(cand_rows[j])
            m = song_meta[row]

            sid = str(m.get("id", ""))
            spid = SPOTIFY_IDS.get(sid)

            if spid:
                st.markdown(spotify_embed_html(spid), unsafe_allow_html=True)
            else:
                st.link_button("▶ Open in Spotify", spotify_search_link(m["title"], m["artist"]))

            if st.button("Pick", key=f"pick_right_{taste.picks_done()}_{j}"):
                picked = j

        if st.button("I don’t like any of these", key=f"none_round_{taste.round}"):
            st.session_state["pressed_none_last_round"] = True  # <--- NEW
            # label all 5 as dislikes, mark seen, advance round
            for r in cand_rows:
                taste.dislikes.append(int(r))
            taste.seen.update(int(r) for r in cand_rows)
            taste.round += 1
            taste.current_cands = None
            rerun()  # refresh UI

        if st.button("Start Over", key="start_over_btn"):
            _reset_step1()  # <--- NEW reset

        # ▼▼ Explainer sits in the remaining vertical space (collapsible) ▼▼
        with st.expander("How does this taste picker work?"):
            st.markdown(
                """
    - **5 rounds × 5 songs**. In each round, pick **one** you like.
    - We build your **taste vector** by averaging embeddings of your picks (normalized).
    - The other 4 shown in the round act as **soft negatives** (we nudge away from them).
      """
            )

    # Handle the selection
    if picked is not None:
        st.session_state["pressed_none_last_round"] = False  # <--- NEW
        choice_row = int(cand_rows[picked])
        taste.likes.append(choice_row)
        for r in cand_rows:
            if int(r) != choice_row:
                taste.dislikes.append(int(r))

        # Mark all 5 as seen, advance round, and prepare next 5
        taste.seen.update(int(r) for r in cand_rows)
        taste.round += 1
        taste.current_cands = None  # force a fresh sample on next render
        rerun()

else:
    st.success("Taste warm-up complete")

st.divider()

# --- helpers/state for stable auto-tags ---
import hashlib, io

def _img_key(pil_img):
    b = io.BytesIO()
    pil_img.save(b, format="PNG")
    return hashlib.md5(b.getvalue()).hexdigest()

st.session_state.setdefault("img_key", None)
st.session_state.setdefault("auto_tags", [])

st.header("Step 2 — Upload your image")
img_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg","jpeg","png"])
img = None

if img_file:
    img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
    st.image(img, caption="Your image")

    key = _img_key(img)
    if st.session_state["img_key"] != key:
        with st.spinner("Generating tags…"):
            kept_tags = auto_tags_for_image(
                img,
                MAX_TAGS=6,
                THRESH_IMG=1.0,
                THRESH_TAG=0.7,
            )
        st.session_state["img_key"] = key
        st.session_state["auto_tags"] = kept_tags

auto_tags = st.session_state["auto_tags"]

if auto_tags:
    st.info("**Auto tags:** " + ", ".join(auto_tags))
else:
    st.warning("No tags generated. Upload your image to start.")

st.header("Step 3 — (Optional) Add your own tags")
custom_tags = st.text_input("Comma-separated tags (e.g., cozy, rainy night, indie)")

def _as_vec(x, name):
    """Coerce to 1-D float32 array and sanity-check length."""
    if x is None:
        return None
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape[0] != DIM:
        st.warning(f"{name} vector has wrong dim {v.shape[0]} (expected {DIM}); skipping it.")
        return None
    return v

st.header("Step 4 — Weights & search")
w_img  = st.slider("Image weight", 0.0, 1.0, 0.50, 0.05)
w_tags = st.slider("Tags weight",  0.0, 1.0, 0.20, 0.05)
w_user = st.slider("Taste weight", 0.0, 1.0, 0.30, 0.05)
num_show = st.radio("How many to show", [2, 4, 6], index=1, horizontal=True, key="num_show_radio")

if st.button("Suggest songs"):
    if img is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Embedding & searching…"):
            # image vec
            img_vec = _as_vec(embed_image(img), "image")
            vecs = []
            if img_vec is not None and w_img > 0:
                vecs.append(w_img * img_vec)

            # merge auto + custom tags, dedupe
            tag_list = []
            if auto_tags:
                tag_list += auto_tags
            if custom_tags.strip():
                tag_list += [t.strip() for t in custom_tags.split(",") if t.strip()]
            seen = set()
            tag_list = [t for t in tag_list if not (t in seen or seen.add(t))]

            # tag vec (mean of tag embeddings)
            if tag_list and w_tags > 0:
                tag_mat = embed_text(tag_list)           # (T, DIM)
                tag_vec = tag_mat.mean(0)                # (DIM,)
                tag_vec /= max(1e-8, np.linalg.norm(tag_vec))
                tag_vec = _as_vec(tag_vec, "tags")
                if tag_vec is not None:
                    vecs.append(w_tags * tag_vec)

            # user taste vec
            user_vec = taste.user_vector()
            user_vec = _as_vec(user_vec, "taste")
            if user_vec is not None and w_user > 0:
                vecs.append(w_user * user_vec)

            if not vecs:
                st.error("No signals to search with. Add an image and/or tags.")
            else:
                # robust stack + sum
                vecs = [v for v in vecs if v is not None]
                V = np.stack(vecs, axis=0)               # (N, DIM)
                q = V.sum(axis=0)
                q = q / max(1e-8, np.linalg.norm(q))
                picks = search(q.astype("float32"), n=num_show, k=TOPK)

                # --- Show Recommendations ---
                st.subheader("Recommendations")

                if not picks:
                    st.info("No recommendations yet. Adjust weights or try again.")
                else:
                    recs = picks[:num_show]  # defensive slice (2/4/6)

                    # Render as 2 per row:
                    # 2  -> 1 row (2 side-by-side)
                    # 4  -> 2 rows (2x2)
                    # 6  -> 3 rows (2x2x2)
                    for start in range(0, len(recs), 2):
                        c1, c2 = st.columns(2)
                        row_items = recs[start:start + 2]

                        with c1:
                            m = row_items[0]
                            sid = str(m.get("id", ""))
                            spid = SPOTIFY_IDS.get(sid)
                            if spid:
                                st.markdown(spotify_embed_html(spid), unsafe_allow_html=True)
                            else:
                                st.link_button("Open in Spotify", spotify_search_link(m["title"], m["artist"]))

                        if len(row_items) > 1:
                            with c2:
                                m = row_items[1]
                                sid = str(m.get("id", ""))
                                spid = SPOTIFY_IDS.get(sid)
                                if spid:
                                    st.markdown(spotify_embed_html(spid), unsafe_allow_html=True)
                                else:
                                    st.link_button("Open in Spotify", spotify_search_link(m["title"], m["artist"]))

st.divider()
with st.expander("Advanced"):
    if st.button("Reset caches (tagger/models)"):
        st.cache_resource.clear()
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        st.rerun()
    st.caption("Unified CLIP space (ViT-B/32), TIGER-lite routing, and quick taste learning (no training).")
    st.caption(f"CI loaded: {HAS_CI}, Interrogator ready: {interrogator is not None}")
    st.caption("Tip: first run may be slow due to model downloads & cache warm-up. Subsequent runs are fast.")
