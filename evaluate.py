"""Histogram baseline, image reconstruction, metrics, figures and diagnostics."""
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timezone

from data import (
    add_physical_noise, apply_gamma_correction, atomic_png, convert_to_8bit,
    digest, load_json, protocol, read_pair, reduce_exposure, require_prepared,
    save_json, sha256, tensor,
)
from train import VARIANTS, device_for, load_checkpoint, seed_all, train_config

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from pytorch_msssim import ssim

from model import UNet

METRICS = ("psnr", "ssim", "lpips")


def baseline(image):
    """Global histogram equalization in YCbCr luminance, unchanged chroma."""
    from PIL import ImageOps
    y, cb, cr = Image.fromarray(image).convert("YCbCr").split()
    return np.array(Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB"))


@torch.inference_mode()
def predict(model, image, device, tile=512, halo=64):
    """Tile cores have context on all sides; no resizing of the evaluated image."""
    h, w = image.shape[:2]
    result = np.empty((h, w, 3), dtype=np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            top, left, bottom, right = max(0, y-halo), max(0, x-halo), min(h, y2+halo), min(w, x2+halo)
            patch = tensor(image[top:bottom, left:right]).unsqueeze(0).to(device)
            pred = model(patch)[0].permute(1, 2, 0).cpu().numpy()
            result[y:y2, x:x2] = pred[y-top:y2-top, x-left:x2-left]
    return convert_to_8bit(result)


class Measures:
    def __init__(self, device):
        import lpips
        self.device = device
        print("Ladowanie LPIPS-AlexNet (przy pierwszym uruchomieniu pobierane sa oficjalne wagi).")
        self.lpips = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()

    @torch.inference_mode()
    def __call__(self, prediction, target):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        a, b = prediction.astype(np.float32) / 255, target.astype(np.float32) / 255
        value = self.lpips(tensor(prediction).unsqueeze(0).to(self.device) * 2 - 1,
                           tensor(target).unsqueeze(0).to(self.device) * 2 - 1).item()
        return {"psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
                "ssim": float(structural_similarity(b, a, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                                    sigma=1.5, use_sample_covariance=False)),
                "lpips": float(value)}


def write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def evaluate(args):
    meta = require_prepared(args)
    device = device_for(args)
    config_path = args.results / "config.json"
    if not config_path.exists():
        raise RuntimeError("Najpierw wykonaj trening.")
    run = load_json(config_path)
    if run["signature"] != digest(train_config(args, meta)):
        raise RuntimeError("Opcje ewaluacji musza odpowiadac opcjom treningu.")
    models, checkpoints = {}, {}
    for variant in VARIANTS:
        folder = args.results / variant
        if not (folder / "last.pt").exists() or not load_checkpoint(folder / "last.pt").get("complete"):
            raise RuntimeError(f"Trening {variant} jeszcze nie jest zakonczony.")
        checkpoint = load_checkpoint(folder / "best.pt")
        if checkpoint["signature"] != run["signature"]:
            raise RuntimeError("Niezgodne wagi modelu.")
        model = UNet(args.width, args.ev, args.gamma).to(device)
        model.load_state_dict(checkpoint["model"])
        models[variant] = model.eval()
        checkpoints[variant] = {"epoch": checkpoint["epoch"], "sha256": sha256(folder / "best.pt")}
    evaluation_config = {"signature": run["signature"], "checkpoints": checkpoints, "tile": args.tile, "halo": 64,
                         "metrics": "PSNR RGB; SSIM RGB gaussian sigma1.5; LPIPS-AlexNet v0.1 full image; uint8"}
    output = args.results / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_path = output / "protocol.json"
    if evaluation_path.exists() and load_json(evaluation_path) != evaluation_config:
        raise RuntimeError("Wyniki ewaluacji maja inna konfiguracje. Uzyj nowego --results.")
    save_json(evaluation_path, evaluation_config)
    measures = None
    rows, details = [], []
    for item in tqdm([r for r in meta["items"] if r["split"] == "test"], desc="Test 20 zdjec"):
        name = item["id"]
        cached = output / f"{name}.json"
        expected_pngs = [output / method / f"{name}.png" for method in ("baseline", *VARIANTS)]
        if cached.exists() and all(p.exists() for p in expected_pngs):
            saved = load_json(cached)
            rows.extend(saved["rows"])
            details.append(saved["details"])
            continue
        if measures is None:
            measures = Measures(device)
        x, y = read_pair(args, name)
        predictions = {"input": x, "baseline": baseline(x)}
        predictions.update({key: predict(model, x, device, args.tile) for key, model in models.items()})
        image_rows = []
        for method, prediction in predictions.items():
            metrics = measures(prediction, y)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError(f"Nieprawidlowa metryka: {name}, {method}.")
            image_rows.append({"image": name, "method": method, **metrics})
            if method != "input":
                folder = output / method
                folder.mkdir(exist_ok=True)
                atomic_png(prediction, folder / f"{name}.png")
        details_item = {"image": name, "mean_luminance": float(np.mean(y @ np.array([0.2126, 0.7152, 0.0722])) / 255),
                        "target_clipped_fraction": float(np.mean(np.any(y >= 254, axis=2))),
                        "network_clipped_fraction": float(np.mean(np.any(predictions['l1_ssim'] >= 254, axis=2))),
                        "channel_mae": np.mean(np.abs(predictions['l1_ssim'].astype(float) - y), axis=(0, 1)).tolist()}
        save_json(cached, {"rows": image_rows, "details": details_item})
        rows.extend(image_rows)
        details.append(details_item)
    write_csv(output / "per_image.csv", rows)
    summary = {}
    for method in ("input", "baseline", *VARIANTS):
        values = [r for r in rows if r["method"] == method]
        summary[method] = {metric: {"mean": float(np.mean([r[metric] for r in values])),
                                    "std": float(np.std([r[metric] for r in values], ddof=1))} for metric in METRICS}
    save_json(output / "summary.json", {"complete": True, "count": 20, "methods": summary,
              "details": details, "checkpoints": checkpoints, "signature": run["signature"],
              "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("\nMetoda           PSNR         SSIM        LPIPS")
    for method, values in summary.items():
        print(f"{method:12s} {values['psnr']['mean']:10.3f} {values['ssim']['mean']:12.4f} {values['lpips']['mean']:12.4f}")
    create_figures(args)


def create_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = args.results / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for variant in VARIANTS:
        history = load_json(args.results / variant / "history.json")
        for ax, key, title in zip(axes, ("loss", "val_psnr", "val_ssim"), ("Strata treningowa", "PSNR walidacji [dB]", "SSIM walidacji")):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=variant)
            ax.set(xlabel="Epoka", title=title)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "learning.png", dpi=170)
    plt.close(fig)
    with (args.results / "evaluation" / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["image"] for r in rows if r["method"] == "baseline"]
    values = {(r["image"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, metric, label in zip(axes, METRICS, ("PSNR [dB]", "SSIM", "LPIPS")):
        for method in ("baseline", *VARIANTS):
            ax.plot(range(81, 101), [float(values[n, method][metric]) for n in names], marker=".", linewidth=1, label=method)
        ax.set(xlabel="Numer obrazu w CSV", title=label)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "test_metrics.png", dpi=170)
    plt.close(fig)
    summary = load_json(args.results / "evaluation" / "summary.json")
    differences = {n: float(values[n, "l1_ssim"]["psnr"]) - float(values[n, "baseline"]["psnr"]) for n in names}
    cases = [(names[0], "Pierwszy obraz testowy"), (names[len(names)//2], "Srodkowy obraz testowy"),
             (max(differences, key=differences.get), "Najwieksza przewaga PSNR nad baseline"),
             (min(differences, key=differences.get), "Najmniejsza przewaga PSNR nad baseline"),
             (min(summary["details"], key=lambda d: d["mean_luminance"])["image"], "Najciemniejszy obraz referencyjny"),
             (max(summary["details"], key=lambda d: d["target_clipped_fraction"])["image"], "Najwiecej nasyconych pikseli referencji")]
    unique = []
    for name, reason in cases:
        if name in [r["image"] for r in unique]:
            next(r for r in unique if r["image"] == name)["reason"] += "; " + reason.lower()
            continue
        low, gt = read_pair(args, name)
        def read_prediction(method):
            with Image.open(args.results / "evaluation" / method / f"{name}.png") as image:
                return np.array(image)
        images = [low, read_prediction("baseline"), read_prediction("l1_ssim"), gt]
        # Deterministic detail crop: highest luminance variance on a 4x4 grid of target.
        h, w = gt.shape[:2]
        p = min(256, h, w)
        coords = [(int(y), int(x)) for y in np.linspace(0, h-p, 4) for x in np.linspace(0, w-p, 4)]
        cy, cx = max(coords, key=lambda c: float(np.var(gt[c[0]:c[0]+p, c[1]:c[1]+p].mean(axis=2))))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for j, (im, title) in enumerate(zip(images, (f"Wejscie {args.ev:g} EV", "Histogram", "U-Net L1 + SSIM", "Referencja"))):
            axes[0, j].imshow(im)
            axes[0, j].add_patch(plt.Rectangle((cx, cy), p, p, fill=False, edgecolor="#ffbf00", linewidth=1.2))
            axes[0, j].set_title(title)
            axes[1, j].imshow(im[cy:cy+p, cx:cx+p])
            for ax in axes[:, j]:
                ax.axis("off")
        fig.tight_layout(pad=0.5)
        filename = f"case_{name}.jpg"
        fig.savefig(figures / filename, dpi=170, pil_kwargs={"quality": 92})
        plt.close(fig)
        unique.append({"image": name, "reason": reason, "figure": filename, "crop": [cx, cy, p]})
    save_json(figures / "cases.json", unique)


def check(args):
    device = device_for(args)
    seed_all(args.seed)
    print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, urzadzenie: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), "architektury:", torch.cuda.get_arch_list())
    model = UNet(args.width, args.ev, args.gamma).to(device)
    x = torch.rand(1, 3, 65, 79, device=device)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    loss = (y - x).abs().mean() + 0.2 * (1 - ssim(y, x, data_range=1.0))
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    clean = np.full((256, 256, 3), 0.5, dtype=np.float32)
    dark = reduce_exposure(clean)
    assert np.allclose(dark, 0.0625)
    noisy = add_physical_noise(dark, np.random.default_rng(42))
    assert abs(float(noisy.mean()) - 0.0625) < 0.001
    assert abs(float(noisy.var()) - (0.001 * 0.0625 + 0.005**2)) < 0.000005
    assert convert_to_8bit(np.array([0.0, 0.5, 1.0])).tolist() == [0, 128, 255]
    assert np.allclose(apply_gamma_correction(clean) ** args.gamma, clean)
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert baseline(np.repeat(a[..., None], 3, axis=2)).shape == (16, 16, 3)
    assert {k: sum(r['split'] == k for r in protocol(args)['items']) for k in ('train', 'val', 'test')} == {'train': 72, 'val': 8, 'test': 20}
    print(f"OK: model ({sum(p.numel() for p in model.parameters()):,} parametrow), gradient, szum, EV, gamma, kwantyzacja, baseline i podzial danych.")
    if args.with_lpips:
        measures = Measures(device)
        image = np.random.default_rng(1).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        # Identity PSNR is infinite mathematically; this test examines LPIPS only.
        with torch.inference_mode():
            t = tensor(image).unsqueeze(0).to(device) * 2 - 1
            value = measures.lpips(t, t).item()
        assert abs(value) < 1e-6
        print("OK: LPIPS identycznych obrazow =", value)

