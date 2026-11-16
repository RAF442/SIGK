"""RAISE download, physical degradation, paired images and training patches.

Shared file/provenance helpers live here to keep the project to a few modules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
# Set project-local caches before importing PyTorch or matplotlib.
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PIPELINE_VERSION = 1


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_rows(args):
    with open(args.csv, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if r["Device"].strip() == "Nikon D7000" and "indoor" in r["Keywords"].lower()]
    if len(rows) < 100:
        raise ValueError("CSV musi zawierac co najmniej 100 zdjec Indoor / Nikon D7000.")
    rows = rows[:100]
    names = [r["File"] for r in rows]
    if len(set(names)) != 100 or any(Path(n).name != n or not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Identyfikatory zdjec musza byc unikalnymi, bezpiecznymi nazwami plikow.")
    return rows


def protocol(args):
    rows = select_rows(args)
    # Fixed seed for the split, independent of model/noise seed.
    valid = set(np.random.default_rng(2026).choice(80, size=8, replace=False).tolist())
    items = [{"id": r["File"], "index": i, "split": "test" if i >= 80 else "val" if i in valid else "train"} for i, r in enumerate(rows)]
    settings = {
        "version": PIPELINE_VERSION, "csv_sha256": sha256(args.csv),
        "ev": args.ev, "alpha": args.alpha, "sigma": args.sigma,
        "gamma": args.gamma, "seed": args.seed, "half_size": True,
        "gt": "raw.postprocess(use_camera_wb=True, half_size=True)",
        "items": items,
    }
    settings["signature"] = digest(settings)
    return settings


def http_session():
    session = requests.Session()
    session.headers["User-Agent"] = "SIGK-educational-project/1.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
    session.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
    return session


def nef_valid(path):
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False
    with path.open("rb") as stream:
        return stream.read(4) in (b"II*\x00", b"MM\x00*")


def download(args):
    raw_dir = args.data / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with http_session() as session:
        for row in tqdm(select_rows(args), desc="Pobieranie RAW", unit="zdj"):
            path = raw_dir / (row["File"] + ".NEF")
            if nef_valid(path):
                continue
            original = row["NEF"].strip()
            # The CSV uses an old IP. The official hostname is tried as a fallback.
            candidates = ["https://loki.disi.unitn.it" + urlparse(original).path, original,
                          "http://loki.disi.unitn.it" + urlparse(original).path]
            errors = []
            part = path.with_suffix(".NEF.part")
            for url in dict.fromkeys(candidates):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with session.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
                        if response.status_code == 416:
                            # A completed .part may remain after an interrupted rename.
                            if nef_valid(part):
                                import rawpy
                                with rawpy.imread(str(part)):
                                    pass
                                part.replace(path)
                                break
                            part.unlink(missing_ok=True)
                            raise ValueError("Nieprawidlowy zakres pobierania; ponow uruchomienie.")
                        response.raise_for_status()
                        append = response.status_code == 206 and offset > 0
                        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Serwer zwrocil nieprawidlowy Content-Range.")
                        expected = int(response.headers.get("Content-Length", 0))
                        received = 0
                        with part.open("ab" if append else "wb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                stream.write(chunk)
                                received += len(chunk)
                        if expected and received != expected:
                            raise ValueError("Niepelny plik RAW.")
                    if not nef_valid(part):
                        part.unlink(missing_ok=True)
                        raise ValueError("Odpowiedz nie jest plikiem NEF (byc moze strona bledu).")
                    import rawpy
                    with rawpy.imread(str(part)):
                        pass
                    part.replace(path)
                    break
                except (requests.RequestException, ValueError, OSError) as exc:
                    errors.append(f"{url}: {exc}")
            else:
                raise RuntimeError("Nie mozna pobrac " + row["File"] + ". Mozesz umiescic NEF w data/raw i ponowic.\n" + "\n".join(errors))


def reduce_exposure(image, ev_shift=-3.0):
    return np.asarray(image, dtype=np.float32) * np.float32(2.0 ** ev_shift)


def add_physical_noise(image, rng, alpha=0.001, sigma=0.005):
    shot = rng.poisson(np.maximum(image, 0) / alpha) * alpha
    return np.clip(shot + rng.normal(0.0, sigma, image.shape), 0.0, 1.0).astype(np.float32)


def apply_gamma_correction(image, gamma=2.2):
    return np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma).astype(np.float32)


def convert_to_8bit(image):
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def atomic_png(array, path):
    temp = path.with_suffix(".png.tmp")
    Image.fromarray(array).save(temp, format="PNG")
    temp.replace(path)


def prepare(args):
    import rawpy
    meta = protocol(args)
    path = args.data / "prepared" / "manifest.json"
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("gt", "input"):
        (folder / name).mkdir(exist_ok=True)
    if path.exists() and load_json(path)["signature"] != meta["signature"]:
        raise RuntimeError("Parametry danych roznia sie od data/prepared/manifest.json. Uzyj osobnego --data i --results dla nowego eksperymentu.")
    # Mark provenance before writing pairs, allowing interrupted preparation to resume.
    save_json(path, meta)
    for item in tqdm(meta["items"], desc="Przygotowanie par", unit="zdj"):
        name = item["id"]
        gt_path, dark_path = folder / "gt" / f"{name}.png", folder / "input" / f"{name}.png"
        if gt_path.exists() and dark_path.exists():
            try:
                for image_path in (gt_path, dark_path):
                    with Image.open(image_path) as im:
                        im.verify()
                continue
            except OSError:
                pass
        source = args.data / "raw" / f"{name}.NEF"
        if not nef_valid(source):
            raise FileNotFoundError(f"Brak poprawnego RAW: {source}. Najpierw wykonaj download.")
        with rawpy.imread(str(source)) as raw:
            # Preserve the exact target-generation settings from the assignment.
            gt = raw.postprocess(use_camera_wb=True, half_size=True)
            linear = raw.postprocess(use_camera_wb=True, gamma=(1, 1), no_auto_bright=True,
                                     output_bps=16, half_size=True).astype(np.float32) / 65535.0
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, item["index"]]))
        noisy = add_physical_noise(reduce_exposure(linear, args.ev), rng, args.alpha, args.sigma)
        low = convert_to_8bit(apply_gamma_correction(noisy, args.gamma))
        if gt.shape != low.shape:
            raise ValueError(f"Niezgodne wymiary pary: {name}")
        atomic_png(gt, gt_path)
        atomic_png(low, dark_path)
    print("Gotowe: 72 train / 8 val / 20 test. Szum testowy pozostaje staly.")


def require_prepared(args):
    expected = protocol(args)
    path = args.data / "prepared" / "manifest.json"
    if not path.exists() or load_json(path)["signature"] != expected["signature"]:
        raise RuntimeError("Brak zgodnych danych. Najpierw uruchom prepare (lub all).")
    for item in expected["items"]:
        for kind in ("gt", "input"):
            if not (path.parent / kind / (item["id"] + ".png")).exists():
                raise FileNotFoundError("Niepelne dane. Ponow prepare lub all.")
    return expected


def read_pair(args, name):
    result = []
    for kind in ("input", "gt"):
        with Image.open(args.data / "prepared" / kind / f"{name}.png") as image:
            result.append(np.array(image.convert("RGB")))
    return result


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0


class Patches(Dataset):
    """Cache uint8 pairs once; random crops never cross image-level splits."""
    def __init__(self, args, items, training=True):
        self.pairs = [read_pair(args, item["id"]) for item in tqdm(items, desc="Wczytywanie obrazow", leave=False)]
        self.patch = args.patch
        self.seed = args.seed
        self.epoch = 0
        self.training = training
        self.length = args.steps * args.batch if training else len(items) * 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, int(self.training)]))
        pair = self.pairs[int(rng.integers(len(self.pairs)))] if self.training else self.pairs[index // 4]
        h, w = pair[0].shape[:2]
        p = self.patch
        if min(h, w) < p:
            raise ValueError(f"Fragment {p} jest wiekszy od obrazu {h}x{w}.")
        y, x = int(rng.integers(h - p + 1)), int(rng.integers(w - p + 1))
        crops = [im[y:y+p, x:x+p] for im in pair]
        if self.training:
            rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
            crops = [np.rot90(im, rotation) for im in crops]
            if flip:
                crops = [im[:, ::-1] for im in crops]
        return tuple(tensor(im) for im in crops)

