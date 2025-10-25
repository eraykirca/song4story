import os, hashlib, tarfile, pathlib, tempfile, urllib.request

ROOT = pathlib.Path(__file__).parent.resolve()
CACHE_DIR  = ROOT / "cache"
MODELS_DIR = ROOT / "models"

ASSETS = [
    {
        "name": "cache_vitb_bundle.tar.gz",
        "url": "https://github.com/eraykirca/song4story/releases/download/v1-assets/cache_vitb_bundle.tar.gz",
        "sha256": "7b56ee9abb1bcd9fcc7b8c9bd26dac7a91a29a0a6eb2a875f0459684c9eb23c9"
    },
    {
        "name": "models_bundle.tar.gz",
        "url": "https://github.com/eraykirca/song4story/releases/download/v1-assets/models_bundle.tar.gz",
        "sha256": "8ea38c52fa9be9c6c6fc6f70f6d7255d7e652db26a9b4f01e0c10b6ce53f55b0"
    }
]

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)

def ensure_assets():
    if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()) and MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        return
    for spec in ASSETS:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td) / spec["name"]
            _download(spec["url"], tmp)
            if spec.get("sha256"):
                digest = _sha256(tmp)
                if digest != spec["sha256"]:
                    raise RuntimeError(f"Checksum failed for {spec['name']}: expected {spec['sha256']} got {digest}")
            with tarfile.open(tmp, "r:gz") as tar:
                tar.extractall(path=ROOT)