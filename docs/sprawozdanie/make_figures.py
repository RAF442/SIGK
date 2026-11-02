"""Rysunki do sprawozdania. Korzysta tylko z gotowych plików w examples/ i runs/,
nie uruchamia sieci (wyniki sieci to zapisane wyjścia komendy predict).

Uruchomienie z katalogu projektu: python docs/sprawozdanie/make_figures.py
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import bilateral, read_rgb  # noqa: E402

EXAMPLES = ROOT / "examples"
OUT = Path(__file__).resolve().parent / "fig"
# Wycinek z kaskiem: dużo krawędzi, napis i drobne tekstury.
CROP = (slice(56, 152), slice(96, 192))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def load(name):
    return read_rgb(str(EXAMPLES / name))


def psnr(reference, image):
    value = -10 * np.log10(np.mean((reference.astype(np.float64) - image) ** 2))
    return f"{value:.2f}".replace(".", ",") + " dB"


def decimal(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def nearest(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)


def bicubic(image):
    return np.clip(cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC), 0, 1)


def show(ax, image, title, crop=False):
    ax.imshow(image[CROP] if crop else image, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def degradations(reference):
    fig, axes = plt.subplots(1, 5, figsize=(6.3, 1.6))
    show(axes[0], reference, "referencja 256×256")
    show(axes[1], nearest(load("input_sr4.png")), "SR ×4: 64×64")
    show(axes[2], nearest(load("input_sr8.png")), "SR ×8: 32×32")
    show(axes[3], load("input_dn001.png"), "szum σ = 0,01")
    show(axes[4], load("input_dn003.png"), "szum σ = 0,03")
    fig.savefig(OUT / "degradacje.pdf")
    plt.close(fig)


def super_resolution(reference):
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.4))
    for row, (condition, label) in enumerate((("sr4", "×4"), ("sr8", "×8"))):
        low = load(f"input_{condition}.png")
        network = load(f"output_{condition}.png")
        base = bicubic(low)
        show(axes[row, 0], nearest(low), f"wejście {label} (nearest)", crop=True)
        show(axes[row, 1], base, f"bicubic, {psnr(reference, base)}", crop=True)
        show(axes[row, 2], network, f"ResidualNet, {psnr(reference, network)}", crop=True)
        show(axes[row, 3], reference, "referencja", crop=True)
    fig.savefig(OUT / "sr_porownanie.pdf")
    plt.close(fig)


def denoising(reference, parameters):
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.4))
    results = {}
    for row, (condition, sigma) in enumerate((("dn001", 0.01), ("dn003", 0.03))):
        noisy = load(f"input_{condition}.png")
        network = load(f"output_{condition}.png")
        filtered = bilateral(noisy, sigma, **parameters[condition])
        results[condition] = (noisy, filtered, network)
        label = decimal(sigma, 2)
        show(axes[row, 0], noisy, f"σ = {label}, {psnr(reference, noisy)}", crop=True)
        show(axes[row, 1], filtered, f"bilateral, {psnr(reference, filtered)}", crop=True)
        show(axes[row, 2], network, f"ResidualNet, {psnr(reference, network)}", crop=True)
        show(axes[row, 3], reference, "referencja", crop=True)
    fig.savefig(OUT / "dn_porownanie.pdf")
    plt.close(fig)

    # Mapy błędu dla silniejszego szumu, na całym obrazie.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35))
    for ax, image, title in zip(axes, results["dn003"],
                                ("wejście zaszumione", "bilateral", "ResidualNet")):
        error = np.abs(reference - image).mean(axis=2)
        handle = ax.imshow(error, cmap="magma", vmin=0, vmax=0.08)
        ax.set_title(f"{title}\nśredni |błąd| = {decimal(error.mean(), 4)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(handle, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(OUT / "dn_mapy_bledu.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    reference = load("reference.png")
    tuned = json.loads((ROOT / "runs/main/bilateral.json").read_text(encoding="utf-8"))
    degradations(reference)
    super_resolution(reference)
    denoising(reference, tuned["parameters"])
    print(f"Zapisano rysunki w {OUT}")


if __name__ == "__main__":
    main()
