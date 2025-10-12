from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
import streamlit as st
from .artifacts import (
embeddings, song_meta, KM1, CENTERS_NORM, cluster_members
)
import random
import torch
import secrets

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
@st.cache_resource(show_spinner=False)
def build_faiss_ip(vecs: np.ndarray):
    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 128    # better recall at query time
    index.add(vecs.astype("float32"))
    return index


faiss_index = build_faiss_ip(embeddings)

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