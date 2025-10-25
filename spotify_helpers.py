import streamlit as st
import urllib.parse
import spotify_ids

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
        st.link_button("▶ Open in Spotify",
                       spotify_search_link(song_meta_obj["title"], song_meta_obj["artist"]))

    if show_pick:
        if st.button("Pick", key=pick_key):
            return True
    return False

SPOTIFY_IDS = spotify_ids.load_spotify_ids()