from __future__ import annotations

import hashlib
import hmac
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from .config import settings

# A local secret would come from a secret manager in prod. For the MVP the
# "signature" only proves the URL was minted by us; it is not a security boundary.
_SIGNING_SECRET = b"spin360-local-dev-signing-key"


class LocalObjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError("path traversal blocked")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        self._abs(key).write_bytes(data)
        return self.url_for(key)

    def put_file(self, key: str, src: Path) -> str:
        shutil.copyfile(src, self._abs(key))
        return self.url_for(key)

    def open(self, key: str, mode: str = "rb") -> BinaryIO:
        return open(self._abs(key), mode)

    def path(self, key: str) -> Path:
        """Local-only escape hatch so tools that need a real path (ffmpeg,
        Blender) can operate on the file. Real backends would download to temp."""
        return self._abs(key)

    def url_for(self, key: str, ttl_s: int = 3600) -> str:
        exp = int(time.time()) + ttl_s
        sig = hmac.new(_SIGNING_SECRET, f"{key}:{exp}".encode(), hashlib.sha256).hexdigest()[:16]
        return f"local://{key}?exp={exp}&sig={sig}"

    def delete(self, key: str) -> None:
        p = self._abs(key)
        if p.exists():
            p.unlink()

    def gc(self, older_than_hours: int | None = None) -> int:
        """Retention/deletion policy. Returns count removed."""
        older_than_hours = older_than_hours or settings.artifact_retention_hours
        cutoff = time.time() - older_than_hours * 3600
        removed = 0
        for p in self.root.rglob("*"):
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        return removed


def key_to_path(url_or_key: str) -> Path:
    """Resolve a `local://key?...` url (or bare key) back to a local path."""
    key = url_or_key
    if url_or_key.startswith("local://"):
        key = url_or_key[len("local://"):].split("?", 1)[0]
    return store.path(key)


store = LocalObjectStore(settings.artifacts_dir)
