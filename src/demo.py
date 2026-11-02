"""Mały, samowystarczalny zbiór do próby technicznej; nie zastępuje DIV2K."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .common import dump_json, sha256


def make_demo(root, seed=123):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Katalog demo musi być pusty: {root}")
    rng = np.random.default_rng(seed)
    manifest = {"dataset": "SYNTHETIC-SMOKE", "size": 256,
                "preprocessing": "Procedural RGB gradients, shapes and textures, 256x256",
                "test_protocol": "synthetic-smoke", "splits": {}}
    for split, count in (("train", 8), ("val", 2), ("test", 3)):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for index in range(count):
            y, x = np.mgrid[0:256, 0:256].astype(np.float32) / 255
            frequency = float(rng.uniform(1, 8))
            image = np.stack([x, y, 0.5 + 0.25 * np.sin(frequency * x * 6.28)], axis=-1)
            image = np.clip(image * rng.uniform(0.4, 0.9, (1, 1, 3)) + rng.uniform(0, 0.1), 0, 1)
            pil = Image.fromarray((image * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil)
            for _ in range(12):
                left, top = rng.integers(0, 220, size=2).tolist()
                color = tuple(rng.integers(10, 245, size=3).tolist())
                draw.rectangle((left, top, left + int(rng.integers(6, 35)),
                                top + int(rng.integers(6, 35))), fill=color)
            path = folder / f"{split}_{index:03d}.png"
            pil.save(path)
            digest = sha256(path)
            records.append({"file": path.name, "source_sha256": digest, "prepared_sha256": digest})
        manifest["splits"][split] = records
    dump_json(root / "manifest.json", manifest)
    print(f"Utworzono {root}: syntetyczny test techniczny, nie DIV2K.")
