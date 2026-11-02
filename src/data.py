"""Pobieranie, jawny podział DIV2K i degradacje RGB [0, 1]."""

import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import CONDITIONS, dump_json, image_seed, load_json, sha256

URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def download(root):
    """Pobiera tylko publiczne archiwa HR. Nie uruchamia się przy treningu."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for url in URLS.values():
        target = root / url.rsplit("/", 1)[1]
        if target.exists() and zipfile.is_zipfile(target):
            print(f"Archiwum już istnieje: {target}", flush=True)
            continue
        part = target.with_suffix(".zip.part")
        print(f"Pobieranie {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "SIGK-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as source, part.open("wb") as out:
            total = int(source.headers.get("Content-Length", 0))
            received, last = 0, 0
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
                received += len(chunk)
                if received - last >= 100 * 1024 * 1024:
                    print(f"  {received / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                    last = received
        if not zipfile.is_zipfile(part):
            raise ValueError(f"Pobrany plik nie jest ZIP: {part}")
        part.replace(target)


def extract_archive(archive, destination):
    """Ochrona przed wyjściem ścieżki ZIP poza wskazany katalog."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            resolved = (destination / info.filename).resolve()
            if not resolved.is_relative_to(destination):
                raise ValueError(f"Niebezpieczna ścieżka ZIP: {info.filename}")
        zf.extractall(destination)  # odczyt ZIP sprawdza CRC każdego pliku


def official_split():
    return {"train": [f"{i:04d}.png" for i in range(1, 721)],
            "val": [f"{i:04d}.png" for i in range(721, 801)],
            "test": [f"{i:04d}.png" for i in range(801, 901)]}


def prepare(raw, output, test_dir=None):
    import cv2

    raw, output = Path(raw), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Katalog docelowy musi być pusty: {output}")
    for split in ("train", "valid"):
        folder = raw / f"DIV2K_{split}_HR"
        if not folder.exists():
            archive = raw / f"DIV2K_{split}_HR.zip"
            if split == "valid" and test_dir:
                continue
            if not archive.exists():
                raise FileNotFoundError(f"Brak {folder} oraz {archive}. Użyj download.")
            extract_archive(archive, raw)
    plan = official_split()
    sources = {"train": raw / "DIV2K_train_HR", "val": raw / "DIV2K_train_HR",
               "test": Path(test_dir) if test_dir else raw / "DIV2K_valid_HR"}
    if test_dir:
        plan["test"] = sorted(p.name for p in sources["test"].glob("*.png"))
        if not plan["test"]:
            raise ValueError("Własny zbiór testowy musi zawierać pliki PNG HR.")
    # Sprawdź komplet źródeł przed pierwszym zapisem.
    for split, names in plan.items():
        for name in names:
            if not (sources[split] / name).is_file():
                raise FileNotFoundError(sources[split] / name)
    manifest = {"dataset": "DIV2K", "size": 256,
                "preprocessing": "RGB; whole image resize 256x256; OpenCV INTER_AREA",
                "test_protocol": "custom-HR" if test_dir else "public-validation-proxy",
                "splits": {}, "source_url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/"}
    seen_sources = set()
    for split, names in plan.items():
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for name in names:
            source = sources[split] / name
            digest = sha256(source)
            if digest in seen_sources:
                raise ValueError(f"Powtórzony obraz źródłowy między podziałami: {source}")
            seen_sources.add(digest)
            with Image.open(source) as image:
                array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            array = cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
            dest = folder / name
            Image.fromarray(array).save(dest)
            records.append({"file": name, "source_sha256": digest,
                            "prepared_sha256": sha256(dest)})
        manifest["splits"][split] = records
        print(f"{split}: {len(records)} obrazów", flush=True)
    dump_json(output / "manifest.json", manifest)


def verify_data(root):
    root = Path(root)
    manifest = load_json(root / "manifest.json")
    hashes = set()
    for split in ("train", "val", "test"):
        records = manifest["splits"][split]
        if not records:
            raise ValueError(f"Pusty podział {split}")
        expected = {r["file"] for r in records}
        if {p.name for p in (root / split).glob("*.png")} != expected:
            raise ValueError(f"Zawartość {split} nie zgadza się z manifestem.")
        for record in records:
            path = root / split / record["file"]
            if sha256(path) != record["prepared_sha256"]:
                raise ValueError(f"Zmieniony obraz: {path}")
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != "RGB":
                    raise ValueError(f"Nieprawidłowy obraz: {path}")
            if record["source_sha256"] in hashes:
                raise ValueError("Przeciek danych: powtórzony obraz źródłowy.")
            hashes.add(record["source_sha256"])
    return manifest


@lru_cache(maxsize=900)
def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def degrade(clean, condition, rng):
    import cv2

    task, value = CONDITIONS[condition]
    if task == "sr":
        size = 256 // value
        low = cv2.resize(clean, (size, size), interpolation=cv2.INTER_AREA)
        base = cv2.resize(low, (256, 256), interpolation=cv2.INTER_CUBIC)
        return np.clip(low, 0, 1), np.clip(base, 0, 1)
    noisy = np.clip(clean + rng.normal(0, value, clean.shape), 0, 1).astype(np.float32)
    return noisy, noisy.copy()


def fixed_degradation(clean, condition, seed, filename):
    return degrade(clean, condition, np.random.default_rng(image_seed(seed, filename, condition)))


def bilateral(image, sigma, color_factor=2.0, spatial=3.0):
    from skimage.restoration import denoise_bilateral

    return np.clip(denoise_bilateral(image, sigma_color=sigma * color_factor,
                                   sigma_spatial=spatial, channel_axis=-1,
                                   mode="reflect"), 0, 1).astype(np.float32)
