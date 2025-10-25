import io, pathlib, secrets, numpy as np
from PIL import Image
import streamlit as st

# Global memory for the taste warm-up across rounds (this session)
st.session_state.setdefault("taste_seen_rows", set())
st.session_state.setdefault("taste_seen_artists", set())

from assets import ensure_assets
ensure_assets()
from spotify_helpers import spotify_embed_html, spotify_search_link
from spotify_helpers import SPOTIFY_IDS
from clip_model import embed_text, embed_image
from data_state import (
    TOPK, rng, embeddings, song_meta, DIM
)
from search_utils import search
from tags import auto_tags_for_image, HAS_CI, interrogator
from app_hooks import get_taste
from taste import policy_sample_candidates

def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

st.set_page_config(page_title="Song4Story", page_icon="🎵")
taste = get_taste()

st.title("Song4Story — pick the best song for your IG story!")
st.header(f"Your music taste + Your image = Your song!")
st.caption(f"by Eray Kırca")

st.header("Step 1 — Quick taste warm-up (5 rounds)")
if not taste.done():
    if taste.current_cands is None:
        taste.current_cands = policy_sample_candidates(k=5)

    cand_rows = np.array(taste.current_cands)
    st.session_state.setdefault("pressed_none_last_round", False)
    st.session_state["last_round_rows"] = [int(r) for r in cand_rows]
    picked = None

    st.caption(f"{taste.picks_done()} selected — need {taste.picks_left()} more")

    left_col, right_col = st.columns(2)

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

    with right_col:
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
            st.session_state["pressed_none_last_round"] = True
            for r in cand_rows:
                taste.dislikes.append(int(r))
            taste.seen.update(int(r) for r in cand_rows)
            taste.round += 1
            taste.current_cands = None
            rerun()

        if st.button("Start Over", key="start_over_btn"):
            for k in (
                "taste",                   # TasteState object
                "taste_seen_rows",         # per-round dedupe cache
                "taste_seen_artists",
                "pressed_none_last_round", # 'none of these' flag
                "last_round_rows",         # last 5 shown
                "rng_seed",                # we'll recreate below
            ):
                st.session_state.pop(k, None)

            # New seed + resync the module RNG that taste.py reads
            st.session_state["rng_seed"] = secrets.randbits(32)
            import data_state as ds
            ds.rng = np.random.default_rng(st.session_state["rng_seed"])

            # also clear any persisted in-progress candidates, if present
            if "taste" in st.session_state and hasattr(st.session_state.taste, "current_cands"):
                st.session_state.taste.current_cands = None
            rerun()

        with st.expander("How does this taste picker work?"):
            st.markdown(
                """
- **5 rounds × 5 songs**. In each round, pick **one** you like.
- We build your **taste vector** by averaging embeddings of your picks (normalized).
- The other 4 shown in the round act as **soft negatives** (we nudge away from them).
                """
            )

    if picked is not None:
        st.session_state["pressed_none_last_round"] = False
        choice_row = int(cand_rows[picked])
        taste.likes.append(choice_row)
        for r in cand_rows:
            if int(r) != choice_row:
                taste.dislikes.append(int(r))
        taste.seen.update(int(r) for r in cand_rows)
        taste.round += 1
        taste.current_cands = None
        rerun()
else:
    st.success("Taste warm-up complete")

st.divider()

# --- helpers/state for stable auto-tags ---
import hashlib
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
    if x is None:
        return None
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape[0] != DIM:
        st.warning(f"{name} vector has wrong dim {v.shape[0]} (expected {DIM}); skipping it.")
        return None
    return v

st.header("Step 4 — Weights & search")
w_img  = st.slider("Image weight", 0.0, 1.0, 0.50, 0.05)
w_tags = st.slider("Tags weight",  0.0, 1.0, 0.25, 0.05)
w_user = st.slider("Taste weight", 0.0, 1.0, 0.25, 0.05)
num_show = st.radio("How many to show", [2, 4, 6], index=1, horizontal=True, key="num_show_radio")

if st.button("Suggest songs"):
    if img is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Embedding & searching…"):
            vecs = []
            img_vec = _as_vec(embed_image(img), "image")
            if img_vec is not None and w_img > 0:
                vecs.append(w_img * img_vec)

            tag_list = []
            if auto_tags:
                tag_list += auto_tags
            if custom_tags.strip():
                tag_list += [t.strip() for t in custom_tags.split(",") if t.strip()]
            seen = set()
            tag_list = [t for t in tag_list if not (t in seen or seen.add(t))]

            if tag_list and w_tags > 0:
                tag_mat = embed_text(tag_list)
                tag_vec = tag_mat.mean(0)
                tag_vec /= max(1e-8, np.linalg.norm(tag_vec))
                tag_vec = _as_vec(tag_vec, "tags")
                if tag_vec is not None:
                    vecs.append(w_tags * tag_vec)

            user_vec = taste.user_vector()
            user_vec = _as_vec(user_vec, "taste")
            if user_vec is not None and w_user > 0:
                vecs.append(w_user * user_vec)

            if not vecs:
                st.error("No signals to search with. Add an image and/or tags.")
            else:
                V = np.stack(vecs, axis=0)
                q = V.sum(axis=0)
                q = q / max(1e-8, np.linalg.norm(q))
                picks = search(q.astype("float32"), n=num_show, k=TOPK)

                st.subheader("Recommendations")
                if not picks:
                    st.info("No recommendations yet. Adjust weights or try again.")
                else:
                    recs = picks[:num_show]
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











