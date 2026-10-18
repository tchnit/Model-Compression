"""File hashes, sizes and reproducible artifact inventories."""
import hashlib
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact_info(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Artifact rỗng: {path}")
    return {"path": str(path), "size_mib": sum(p.stat().st_size for p in files) / 2**20,
            "files": [{"name": p.name if path.is_file() else str(p.relative_to(path)),
                       "sha256": sha256(p)} for p in files]}


def get_dir_size_mib(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)

