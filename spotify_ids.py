from pathlib import Path
import json

def load_spotify_ids() -> dict:
    """Return {song_id: spotify_id} from artifacts/songs.jsonl."""
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


