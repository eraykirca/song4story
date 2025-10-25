from typing import Optional, List, Tuple
import numpy as np
import streamlit as st
from data_state import (
    embeddings, song_meta, KM1, cluster_members, row2b1, archetype_row,
    CENTERS_NORM, K_EACH_EARLY, K_EACH_LATE, faiss_index
)
import data_state as ds
import faiss

class TasteState:
    TARGET_PICKS = 5

    def __init__(self):
        self.round = 0
        self.history = []
        self.likes = []
        self.dislikes = []
        self.seen = set()
        self.current_cands = None

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
    if not seed_rows:
        return []
    k = int(base_k)
    neigh = _faiss_neighbors(seed_rows, k_each=k, exclude=used)
    while len(neigh) < min_unique and k < hard_cap:
        k = min(k * 2, hard_cap)
        neigh = _faiss_neighbors(seed_rows, k_each=k, exclude=used)
    return neigh

def _dislike_vector(use_last_round=True):
    rows = []
    from app_hooks import get_taste  # late import to access session state holder
    taste = get_taste()
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
    if vec is None or KM1 is None or not cluster_members:
        return []
    C = KM1.cluster_centers_.astype("float32")
    C /= np.maximum(1e-8, np.linalg.norm(C, axis=1, keepdims=True))
    sims = (C @ vec.astype("float32"))
    order = np.argsort(-sims)
    return [(int(b1), float(sims[ int(b1) ])) for b1 in order]

def _user_vector_or_centroid():
    from app_hooks import get_taste
    taste = get_taste()
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

def _artist(row_idx):
    return song_meta[int(row_idx)]["artist"]

def _pick_cluster_rep(b1, blocked_rows, top_m=3):
    rows = cluster_members.get(int(b1), [])
    if not rows:
        return None
    c = CENTERS_NORM[int(b1)] if CENTERS_NORM is not None else None
    if c is None:
        return None
    E = embeddings[np.array(rows)]
    sims = (E @ c).astype(np.float32)
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
    return int(ds.rng.choice(picks))

def policy_sample_candidates(k=5):
    from app_hooks import get_taste
    taste = get_taste()
    blocked = set(taste.seen) | set(taste.likes) | set(taste.dislikes)
    if len(embeddings) - len(blocked) < k:
        taste.seen = set(taste.likes) | set(taste.dislikes)

    used_rows = set(taste.seen) | set(taste.likes) | set(taste.dislikes) | set(st.session_state["taste_seen_rows"])
    used_artists = set(st.session_state["taste_seen_artists"])

    uv = _user_vector_or_centroid()
    round_cap = 1 if taste.round <= 3 else 2
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

    pressed_none = st.session_state.get("pressed_none_last_round", False)
    if pressed_none:
        cands = []
        if taste.picks_done() == 0 and KM1 is not None and cluster_members:
            avoid_clusters = set()
            for r in st.session_state.get("last_round_rows", []):
                b = row2b1[r] if r < len(row2b1) else None
                if b is not None:
                    avoid_clusters.add(int(b))

            dv = _dislike_vector(use_last_round=True)
            far = [b for (b, _) in reversed(_clusters_sorted_by_similarity_to_vec(dv))] if dv is not None else list(cluster_members.keys())
            far = list(ds.rng.permutation(far)) if len(far) > 1 else far

            for b1 in far:
                if len(cands) >= k:
                    break
                if int(b1) in avoid_clusters:
                    continue
                rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                if rep is None or rep in used_rows or not _accept(rep):
                    continue
                cands.append(int(rep))

            if len(cands) < k and cluster_members:
                rest = [b for b in cluster_members.keys() if b not in avoid_clusters]
                rest = list(ds.rng.permutation(rest))
                for b1 in rest:
                    if len(cands) >= k:
                        break
                    rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                    if rep is None or rep in used_rows or not _accept(rep):
                        continue
                    cands.append(int(rep))

            _commit(cands)
            st.session_state["pressed_none_last_round"] = False
            return np.array(cands[:k], dtype=int)

        like_seed = taste.likes[-5:] if taste.likes else []
        neigh = _faiss_neighbors(like_seed, k_each=K_EACH_EARLY, exclude=used_rows) if like_seed else []
        if neigh:
            neigh = list(ds.rng.permutation(neigh))

        cands = []
        for r in neigh:
            if len(cands) >= k:
                break
            if r in used_rows or not _accept(r):
                continue
            cands.append(int(r))

        if len(cands) < k and uv is not None and KM1 is not None:
            near = [b for (b, _) in _clusters_sorted_by_similarity_to(uv)]
            for b1 in near:
                if len(cands) >= k:
                    break
                rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
                if rep is None or rep in used_rows or not _accept(rep):
                    continue
                cands.append(int(rep))

        if len(cands) < k:
            all_rows = np.setdiff1d(np.arange(len(embeddings)),
                                    np.fromiter(used_rows | set(cands), int, count=len(used_rows | set(cands))) if used_rows else [])
            all_rows = list(ds.rng.permutation(all_rows))
            for r in all_rows:
                if len(cands) >= k:
                    break
                if not _accept(r):
                    continue
                cands.append(int(r))

        _commit(cands)
        st.session_state["pressed_none_last_round"] = False
        return np.array(cands[:k], dtype=int)

    if taste.round <= 1 and CENTERS_NORM is not None and cluster_members:
        cands = []
        cluster_ids = list(cluster_members.keys())
        cluster_ids = list(ds.rng.permutation(cluster_ids))
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

        if len(cands) < k:
            if uv is not None and KM1 is not None:
                order = _clusters_sorted_by_similarity_to(uv)
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

    if taste.round <= 3:
        like_seed = taste.likes[-3:] if taste.likes else []
        neigh = _faiss_neighbors(like_seed, k_each=K_EACH_EARLY, exclude=used_rows) if like_seed else []
        if neigh:
            neigh = list(ds.rng.permutation(neigh))

        cands = []
        for r in neigh:
            if len(cands) >= min(2, k):
                break
            if r in used_rows or not _accept(r):
                continue
            cands.append(int(r))

        cluster_ids = list(cluster_members.keys())
        cluster_ids = list(ds.rng.permutation(cluster_ids))
        for b1 in cluster_ids:
            if len(cands) >= k:
                break
            rep = _pick_cluster_rep(b1, blocked_rows=used_rows | set(cands), top_m=3)
            if rep is None or rep in used_rows or not _accept(rep):
                continue
            cands.append(int(rep))

        _commit(cands)
        return np.array(cands[:k], dtype=int)

    like_seed = taste.likes[-5:] if taste.likes else []
    neigh = _faiss_neighbors(like_seed, k_each=K_EACH_LATE, exclude=used_rows) if like_seed else []
    if neigh:
        neigh = list(ds.rng.permutation(neigh))

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
        all_rows = list(ds.rng.permutation(all_rows))
        for r in all_rows:
            if len(cands) >= k:
                break
            if not _accept(r):
                continue
            cands.append(int(r))

    _commit(cands)
    return np.array(cands[:k], dtype=int)

def sample_unseen_candidates(k: int = 5) -> np.ndarray:
    from app_hooks import get_taste
    taste = get_taste()
    blocked = set(taste.seen) | set(taste.likes) | set(taste.dislikes)
    all_rows = np.arange(len(embeddings))
    if blocked:
        blocked_arr = np.fromiter(blocked, dtype=int)
        available = np.setdiff1d(all_rows, blocked_arr, assume_unique=False)
    else:
        available = all_rows

    if len(available) < k:
        taste.seen = set(taste.likes) | set(taste.dislikes)
        blocked = set(taste.seen)
        blocked_arr = np.fromiter(blocked, dtype=int) if blocked else np.array([], dtype=int)
        available = np.setdiff1d(all_rows, blocked_arr, assume_unique=False)

    k_eff = min(k, len(available))
    return np.random.choice(available, size=k_eff, replace=False)






