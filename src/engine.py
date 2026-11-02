"""Trening, walidacja, ewaluacja i odtwarzanie pojedynczego obrazu."""

import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import CONDITIONS, dump_json, image_seed, load_json, sha256, write_csv
from .data import bilateral, degrade, fixed_degradation, read_rgb, save_rgb, verify_data
from .metrics import Metrics
from .models import Restorer, load_model


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna. Zainstaluj PyTorch CUDA albo wybierz --device cpu.")
    return torch.device(name)


def environment():
    packages = ("torch", "torchvision", "numpy", "Pillow", "opencv-python-headless",
                "scikit-image", "lpips", "matplotlib")
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result["cuda"] = torch.version.cuda
    return result


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


class Images(Dataset):
    def __init__(self, root, split, condition, seed=42, repeat=1, augment=False, torch_noise=False):
        self.files = sorted((Path(root) / split).glob("*.png"))
        if not self.files:
            raise ValueError(f"Brak obrazów: {root}/{split}")
        self.condition, self.seed = condition, seed
        self.repeat, self.augment, self.epoch = repeat, augment, 0
        self.torch_noise = torch_noise

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        clean = read_rgb(str(path)).copy()
        if self.augment:
            rng = np.random.default_rng(image_seed(self.seed, f"{self.epoch}/{index}", self.condition))
            clean = np.rot90(clean, int(rng.integers(4)))
            if rng.random() < 0.5:
                clean = np.fliplr(clean)
            clean = np.ascontiguousarray(clean)
            if self.torch_noise:
                # Szum powstanie w jednym batchu na urządzeniu, bez kopii CPU->GPU wejścia i base.
                target = to_tensor(clean)
                return target, target, target
            degraded, base = degrade(clean, self.condition, rng)
        else:
            degraded, base = fixed_degradation(clean, self.condition, self.seed, path.name)
        return to_tensor(degraded), to_tensor(base), to_tensor(clean)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    values = []
    for image, base, target in loader:
        target = target.to(device)
        prediction = model(image.to(device), base.to(device)).clamp(0, 1)
        mse = (prediction - target).square().mean((1, 2, 3))
        values.extend((-10 * torch.log10(mse.clamp_min(1e-15))).cpu().tolist())
    return float(np.mean(values))


def train(args):
    device = select_device(args.device)
    precision = getattr(args, "precision", "fp32")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 wymaga zgodnej karty CUDA. Wybierz --precision fp32.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = verify_data(args.data)
    noise_backend = getattr(args, "training_noise", "auto")
    if noise_backend == "auto":
        noise_backend = "torch" if device.type == "cuda" and args.condition.startswith("dn") else "numpy"
    if not args.condition.startswith("dn"):
        noise_backend = "numpy"
    folder = Path(args.runs) / args.condition
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise ValueError(f"Katalog zawiera poprzedni trening: {folder}. Użyj --resume lub innego --runs.")
    folder.mkdir(parents=True, exist_ok=True)
    architecture = {"width": args.width, "blocks": args.blocks}
    model = Restorer(args.condition, **architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                          eta_min=args.lr * 0.01)
    config = {"condition": args.condition, "architecture": architecture,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "loss": args.loss, "seed": args.seed, "repeat": args.repeat,
              "precision": precision,
              "training_noise": noise_backend,
              "manifest_sha256": sha256(Path(args.data) / "manifest.json")}
    start, best, history = 0, -float("inf"), []
    if args.resume:
        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        stored_config = dict(checkpoint["config"])
        stored_config.setdefault("precision", "fp32")
        stored_config.setdefault("training_noise", "numpy")
        if stored_config != config:
            raise ValueError("Resume wymaga identycznych danych i parametrów treningu.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Stan Adam musi trafić na to samo urządzenie co parametry.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, history = checkpoint["epoch"], checkpoint["best_psnr"], checkpoint["history"]
    dump_json(folder / "config.json", {**config, "environment": environment(),
                                      "dataset": manifest["dataset"]})
    training = Images(args.data, "train", args.condition, args.seed, args.repeat, True,
                      torch_noise=noise_backend == "torch")
    validation = Images(args.data, "val", args.condition, args.seed)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    criterion = torch.nn.MSELoss() if args.loss == "mse" else torch.nn.L1Loss()
    for epoch in range(start, args.epochs):
        started = time.perf_counter()
        training.epoch = epoch
        # Każda epoka ma własne tasowanie: wznowienie odtwarza kolejność.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        noise_generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                            generator=generator, num_workers=args.workers)
        model.train()
        loss_sum = 0.0
        for image, base, target in loader:
            target = target.to(device)
            if training.torch_noise:
                noise = torch.randn(target.shape, generator=noise_generator, device=device,
                                    dtype=target.dtype) * CONDITIONS[args.condition][1]
                image = (target + noise).clamp(0, 1)
                base = image
            else:
                image, base = image.to(device), base.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16"):
                prediction = model(image, base)
                loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Niefinitywna strata: trening przerwany.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * image.size(0)
        psnr = validate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(training),
               "val_PSNR": psnr, "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started}
        history.append(row)
        improved = psnr > best
        best = max(best, psnr)
        scheduler.step()
        checkpoint = {"format_version": 1, "condition": args.condition,
                      "architecture": architecture, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "config": config, "epoch": epoch + 1, "best_psnr": best, "history": history}
        # Zapis atomowy, aby przerwanie nie zniszczyło poprzedniego checkpointu.
        for name in (["best.pt", "last.pt"] if improved else ["last.pt"]):
            temporary = folder / (name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(folder / name)
        write_csv(folder / "history.csv", history)
        print(f"{args.condition} | {epoch + 1}/{args.epochs} | loss {row['train_loss']:.6f}"
              f" | val PSNR {psnr:.3f} dB | {row['seconds']:.1f} s", flush=True)
    # Odtwórz CSV również po wznowieniu już ukończonego treningu.
    if history:
        write_csv(folder / "history.csv", history)


@torch.inference_mode()
def restore(model, image, base, device):
    result = model(to_tensor(image)[None].to(device), to_tensor(base)[None].to(device))
    return result[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def tune(args):
    """Dobór bilateral wyłącznie na wewnętrznym zbiorze walidacyjnym."""
    from .metrics import errors

    verify_data(args.data)
    rows, chosen = [], {}
    for condition in ("dn001", "dn003"):
        sigma = CONDITIONS[condition][1]
        best_psnr = -float("inf")
        for factor in (1.0, 2.0, 3.0):
            for spatial in (1.0, 3.0):
                scores = []
                for path in sorted((Path(args.data) / "val").glob("*.png")):
                    clean = read_rgb(str(path))
                    noisy, _ = fixed_degradation(clean, condition, args.seed, path.name)
                    result = bilateral(noisy, sigma, factor, spatial)
                    scores.append(errors(clean, result)["PSNR"])
                psnr = float(np.mean(scores))
                rows.append({"condition": condition, "color_factor": factor,
                             "spatial": spatial, "val_PSNR": psnr})
                if psnr > best_psnr:
                    best_psnr = psnr
                    chosen[condition] = {"color_factor": factor, "spatial": spatial}
                print(f"Bilateral {condition}: color={factor}, spatial={spatial}, PSNR={psnr:.3f}", flush=True)
    dump_json(args.output, {"seed": args.seed, "split": "val", "parameters": chosen,
                            "manifest_sha256": sha256(Path(args.data) / "manifest.json")})
    write_csv(Path(args.output).with_suffix(".csv"), rows)


def evaluate(args):
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    manifest = verify_data(args.data)
    manifest_hash = sha256(Path(args.data) / "manifest.json")
    if args.split == "test" and manifest["test_protocol"] == "public-validation-proxy":
        print("UWAGA: publiczne DIV2K validation HR jako test zastępczy; to nie oficjalny test HR.", flush=True)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Wybierz pusty katalog wyników: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tuned = load_json(args.bilateral_config) if args.bilateral_config else None
    if tuned and (tuned["manifest_sha256"] != manifest_hash or tuned["seed"] != args.seed):
        raise ValueError("Parametry bilateral pochodzą z innych danych lub innego ziarna.")
    models = {}
    checkpoints = {}
    for condition in args.conditions:
        path = Path(args.runs) / condition / "best.pt"
        model, checkpoint = load_model(path, device)
        if checkpoint["condition"] != condition:
            raise ValueError(f"Niewłaściwy checkpoint dla {condition}")
        if checkpoint["config"]["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Checkpoint {condition} trenowano na innych danych.")
        models[condition] = model
        checkpoints[condition] = {"sha256": sha256(path), "epoch": checkpoint["epoch"],
                                  "architecture": checkpoint["architecture"],
                                  "training_config": checkpoint["config"],
                                  "validation_PSNR": checkpoint["best_psnr"]}
    metrics = Metrics(device, args.lpips_net, args.skip_lpips, args.sne_definition)
    paths = sorted((Path(args.data) / args.split).glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]
    meta = {"dataset": manifest["dataset"], "test_protocol": manifest["test_protocol"],
            "split": args.split, "image_count": len(paths), "full_split": not bool(args.limit),
            "manifest_sha256": manifest_hash, "seed": args.seed, "conditions": args.conditions,
            "checkpoints": checkpoints, "lpips_net": None if args.skip_lpips else args.lpips_net,
            "sne_definition": args.sne_definition,
            "primary_metrics_complete": not args.skip_lpips,
            "complete_metrics": not args.skip_lpips and args.sne_definition == "sse",
            "bilateral": tuned["parameters"] if tuned else "fixed: color_factor=2; spatial=3",
            "environment": environment(),
            "metric_protocol": "RGB [0,1]; full 256x256; mean per-image metrics; no border crop"}
    dump_json(output / "metadata.json", meta)
    rows = []
    for condition in args.conditions:
        task, value = CONDITIONS[condition]
        parameters = tuned["parameters"][condition] if tuned and task == "denoise" else {}
        label = f"bicubic OpenCV x{value}" if task == "sr" else f"bilateral sigma={value:g}"
        network_label = (f"ResidualNet SR {256 // value}x{256 // value}->256x256 (x{value})"
                         if task == "sr" else f"ResidualNet denoise sigma={value:g}")
        for index, path in enumerate(paths):
            clean = read_rgb(str(path))
            image, base = fixed_degradation(clean, condition, args.seed, path.name)
            baseline = base if task == "sr" else bilateral(image, value, **parameters)
            prediction = restore(models[condition], image, base, device)
            candidates = [(label, baseline), (network_label, prediction)]
            if task == "denoise":
                candidates.insert(0, (f"noisy sigma={value:g}", image))
            for method, result in candidates:
                rows.append({"condition": condition, "image": path.name, "method": method,
                             **metrics(clean, result)})
            if index < args.examples:
                from .report import comparison
                comparison(output / "images" / f"{condition}_{path.stem}.png",
                           clean, base if task == "sr" else image, baseline, prediction,
                           condition, label)
            print(f"{condition}: {index + 1}/{len(paths)} {path.name}", flush=True)
    # Ujednolicona kolejność kolumn również dla starszego API pomijającego SNE.
    for row in rows:
        row.setdefault("SNE", None)
    write_csv(output / "per_image.csv", rows)
    summary = aggregate(rows)
    write_csv(output / "summary.csv", summary)
    dump_json(output / "summary.json", summary)
    from .report import make_report
    make_report(output, args.runs)
    print(f"Raport: {(output / 'raport.html').resolve()}", flush=True)


def aggregate(rows):
    result = []
    groups = sorted({(row["condition"], row["method"]) for row in rows})
    for condition, method in groups:
        samples = [r for r in rows if (r["condition"], r["method"]) == (condition, method)]
        row = {"condition": condition, "method": method, "images": len(samples)}
        for metric in ("PSNR", "SSIM", "LPIPS", "SNE", "MSE", "SSE"):
            values = [r[metric] for r in samples if r.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def predict(args):
    import cv2

    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    condition = checkpoint["condition"]
    task, value = CONDITIONS[condition]
    image = read_rgb(str(Path(args.input).resolve()))
    expected = (256 // value, 256 // value) if task == "sr" else (256, 256)
    if image.shape[:2] != expected:
        raise ValueError(f"Model {condition} wymaga wejścia {expected[1]}x{expected[0]} px. "
                         "Podaj już zdegradowany obraz; predict nie dodaje szumu.")
    base = np.clip(cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC), 0, 1) if task == "sr" else image
    save_rgb(args.output, restore(model, image, base, device))
    print(f"Zapisano {args.output}")
