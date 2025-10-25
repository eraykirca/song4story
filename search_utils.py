from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from data_state import (
    TOPK, USE_ROUTER_TOPB, MMR_ALPHA, ARTIST_CAP, JITTER_EPS,
    embeddings, song_meta, cluster_members, CENTERS_NORM, DIM, faiss_index
)

def route_bucket(x: np.ndarray) -> Optional[Tuple[int,int]]:
    if False:  # kept placeholder to preserve function body when moved (no behavior change)
        pass
    # needs KM/bucket map. they’re accessed in app through data_state
    from data_state import bucket_map, song_meta as _meta
    if bucket_map is None:
        return None
    D, I = faiss_index.search(x[None,:].astype("float32"), 1)
    nearest_idx = int(I[0][0])
    sid = str(_meta[nearest_idx]["id"])
    return bucket_map.get(sid)

def _mmr_select(candidate_rows, candidate_scores, take_m, alpha=MMR_ALPHA):
    if not candidate_rows:
        return []
    rows = list(candidate_rows)
    rel  = np.asarray(candidate_scores, dtype=np.float32)
    E = embeddings[np.array(rows)]
    S = (E @ E.T).astype(np.float32)
    picked = []
    unused = list(range(len(rows)))
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
    q = x.astype("float32").reshape(-1)
    cand_rows, cand_scores = [], []

    def pack(rows, scores):
        out = []
        for r, sc in zip(rows, scores):
            m = song_meta[int(r)].copy()
            m["clip_score"] = float(sc)
            out.append(m)
        return out

    if CENTERS_NORM is not None and cluster_members:
        B = max(1, int(USE_ROUTER_TOPB))
        sims = (CENTERS_NORM @ q)
        topB = np.argsort(-sims)[:B]
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
        D, I = faiss_index.search(q[None, :], k)
        cand_rows = [int(i) for i in I[0]]
        cand_scores = [float(s) for s in D[0]]

    if not cand_rows:
        return []

    seen, rows_u, scores_u = set(), [], []
    for r, sc in zip(cand_rows, cand_scores):
        if r in seen:
            continue
        seen.add(r)
        rows_u.append(int(r))
        scores_u.append(float(sc))

    if JITTER_EPS > 0:
        jitter = (np.random.rand(len(scores_u)) * 2 - 1) * float(JITTER_EPS)
        scores_perturbed = (np.asarray(scores_u, dtype=np.float32) + jitter).tolist()
    else:
        scores_perturbed = scores_u

    mmr_rows = _mmr_select(rows_u, scores_perturbed, take_m=max(n * 3, n + 3))

    final_rows, artist_counts = [], {}
    for r in mmr_rows:
        artist = song_meta[int(r)]["artist"]
        if artist and artist_counts.get(artist, 0) >= ARTIST_CAP:
            continue
        final_rows.append(int(r))
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if len(final_rows) >= n:
            break

    if len(final_rows) < n:
        for r in rows_u:
            if r not in final_rows:
                final_rows.append(int(r))
                if len(final_rows) >= n:
                    break

    score_map = {r: s for r, s in zip(rows_u, scores_u)}
    return pack(final_rows, [score_map.get(r, 0.0) for r in final_rows])

