# song4story/core/spotify.py
from pathlib import Path
import json


ARTIFACTS = Path("artifacts")

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
