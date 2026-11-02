"""Narzędzia niezależne od PyTorch, używane również przez raport."""

import csv
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = {"sr4": ("sr", 4), "sr8": ("sr", 8),
              "dn001": ("denoise", 0.01), "dn003": ("denoise", 0.03)}


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Standardowy JSON: brak NaN/Infinity. Inf PSNR zapisujemy jako tekst.
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Brak wierszy do zapisania.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_seed(seed, name, condition):
    """Stały seed niezależny od hash(), kolejności i liczby procesów."""
    value = f"{seed}/{name}/{condition}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")
