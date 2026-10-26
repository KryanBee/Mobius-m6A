"""Small portable IO and provenance helpers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["source_id", "transcript_id", "transcript_position"]

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace a metadata file atomically and clean staging files on failure."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_info(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    code = {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "sonic").glob("*.py")) + sorted(ROOT.glob("*.py"))}
    result = {"python": platform.python_version(), "code_sha256": code,
              "packages": {d: importlib.metadata.version(d) for d in
                           ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost-cpu", "torch")}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


@contextmanager
def new_directory(destination):
    """Publish complete artifacts together; refuse existing destinations.

    The lock also prevents two copies of our extractor from racing to publish.
    A crash can leave a .lock/.partial directory, which must be inspected before
    cleanup. Consumers require manifest.json and never read partial directories.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".lock")
    with lock.open("x"):
        try:
            if destination.exists():
                raise FileExistsError(f"Use a new output directory: {destination}")
            with tempfile.TemporaryDirectory(prefix=".partial-", dir=destination.parent) as tmp:
                yield Path(tmp)
                os.rename(tmp, destination)
        finally:
            lock.unlink(missing_ok=True)
