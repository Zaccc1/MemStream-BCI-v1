"""I/O, integrity checks and calibration loading for the demo."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys


def long_path(path):
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


ROOT = long_path(Path(__file__).parent.parent)


def read_json(path):
    return json.loads(long_path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with long_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_asset(relative):
    manifest = read_json(ROOT / "data/assets.json")
    entry = manifest[relative]
    path = ROOT / relative
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Bundled asset failed integrity check: {relative}")
    return path


def load_scientific_source():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    sys.path.insert(0, str(ROOT / "source/benchmarks/ibl"))


def load_calibration(mode):
    # Only the fixed, hash-checked, author-provided pickle is allowed here.
    # Python pickle is executable: do not substitute an untrusted model file.
    path = verified_asset(f"data/calibration_{mode}.pkl.gz")
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)
