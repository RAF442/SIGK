"""Run controlled U-Net experiments on the already prepared RAISE data.

python experiments.py                 # screen, confirm and evaluate finalists
python experiments.py --stage train   # screening + repeated-seed validation
python experiments.py --stage evaluate
python report.py                      # include completed study in the single PDF

Each ablation changes one factor versus L1 + 0.2*SSIM, logit-output U-Net.
The test split is accessed only after validation-based selection is saved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

from data import ROOT, Patches, atomic_png, digest, load_json, read_pair, require_prepared, save_json, sha256
from main import parser as project_parser
from train import device_for, load_checkpoint, save_checkpoint, seed_all, train_config, validate
from evaluate import Measures, predict
from unet_variants import build_model

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

METRICS = ("psnr", "ssim", "lpips")


def specifications():
    base = {"output": "logit", "activation": "leaky_relu", "residual": False, "attention": False,
            "loss": "l1", "ssim_weight": 0.2, "scheduler": "cosine"}
    choices = [
        ("reference", "Punkt odniesienia", "Model oryginalny", {}),
        ("output_direct", "Wyjście", "Bezpośrednie RGB", {"output": "direct"}),
        ("output_rgb_residual", "Wyjście", "Poprawka w RGB", {"output": "rgb_residual"}),
        ("lr_plateau", "Uczenie", "LR po plateau", {"scheduler": "plateau"}),
        ("loss_mse", "Strata", "MSE + 0,2 SSIM", {"loss": "mse"}),
        ("loss_charbonnier", "Strata", "Charbonnier + 0,2 SSIM", {"loss": "charbonnier"}),
        ("ssim_005", "Waga SSIM", "L1 + 0,05 SSIM", {"ssim_weight": 0.05}),
        ("ssim_05", "Waga SSIM", "L1 + 0,5 SSIM", {"ssim_weight": 0.5}),
        ("activation_prelu", "Aktywacja", "PReLU", {"activation": "prelu"}),
        ("activation_silu", "Aktywacja", "SiLU", {"activation": "silu"}),
        ("residual_blocks", "Bloki", "Bloki resztkowe", {"residual": True}),
        ("channel_attention", "Uwaga", "Uwaga kanałowa SE", {"attention": True}),
    ]
    return [{"id": key, "group": group, "label": label, **base, **change} for key, group, label, change in choices]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def loss_value(prediction, target, spec):
    difference = prediction.float() - target.float()
    if spec["loss"] == "mse":
        point = difference.square().mean()
    elif spec["loss"] == "charbonnier":
        point = (difference.square() + 1e-6).sqrt().mean()
    else:
        point = difference.abs().mean()
    return point + spec["ssim_weight"] * (1 - ssim(prediction.float(), target.float(), data_range=1.0))


def protocol(args, seeds):
    meta = require_prepared(args)
    hyper = {key: getattr(args, key) for key in ("width", "ev", "gamma", "epochs", "steps", "batch", "patch", "lr", "patience", "seed", "tile", "no_amp")}
    value = {"version": 1, "data_signature": meta["signature"], "hyperparameters": hyper,
             "specifications": specifications(), "confirmation_seeds": seeds, "shortlist_size": 2,
             "selection": "mean full-image validation PSNR, first screen then three training seeds",
             "checkpoint_selection": f"PSNR on 32 fixed validation crops; patience={args.patience}; epochs<={args.epochs}",
             "noise_seed": args.seed, "torch": str(torch.__version__),
             "source_sha256": {name: sha256(ROOT/name) for name in
                               ("experiments.py", "unet_variants.py", "model.py", "data.py", "train.py", "evaluate.py")}}
    value["signature"] = digest(value)
    return value, meta


class Study:
    def __init__(self, args, folder, seeds):
        self.args, self.folder, self.seeds = args, folder, seeds
        self.protocol, self.meta = protocol(args, seeds)
        self.device = device_for(args)
        self.specs = {s["id"]: s for s in specifications()}
        self.datasets = None
        self.measures = None
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/"protocol.json"
        if path.exists() and load_json(path) != self.protocol:
            raise RuntimeError("Konfiguracja badan ulegla zmianie. Wybierz osobny --study.")
        save_json(path, self.protocol)

    def run_folder(self, spec, seed):
        return self.folder/"runs"/f"{spec['id']}_s{seed}"

    def run_signature(self, spec, seed):
        return digest({"protocol": self.protocol["signature"], "spec": spec, "training_seed": seed})

    def model(self, spec, seed):
        seed_all(seed)
        return build_model(self.args, spec).to(self.device)

    def original_reference(self, spec, seed, folder, signature):
        if spec["id"] != "reference" or seed != self.args.seed:
            return False
        legacy = self.args.results
        if not (legacy/"config.json").exists() or not (legacy/"l1_ssim"/"last.pt").exists():
            return False
        run = load_json(legacy/"config.json")
        if run["config"] != train_config(self.args, self.meta):
            return False
        last = load_checkpoint(legacy/"l1_ssim"/"last.pt")
        if not last.get("complete") or last["signature"] != run["signature"]:
            return False
        best = load_checkpoint(legacy/"l1_ssim"/"best.pt")
        model = self.model(spec, seed)
        model.load_state_dict(best["model"], strict=True)
        save_checkpoint(folder/"best.pt", {"model": best["model"], "epoch": best["epoch"], "signature": signature})
        save_json(folder/"history.json", last["history"])
        save_json(folder/"training.json", {"complete": True, "signature": signature,
                  "parameters": sum(p.numel() for p in model.parameters()), "epochs": last["epoch"],
                  "best_epoch": best["epoch"], "training_seconds": sum(r["seconds"] for r in last["history"]),
                  "reused_original": True, "original_checkpoint_sha256": sha256(legacy/"l1_ssim"/"best.pt")})
        print("reference: wykorzystuje ukonczony oryginalny trening.", flush=True)
        return True

    def training(self, spec, seed):
        args, device = self.args, self.device
        folder = self.run_folder(spec, seed)
        folder.mkdir(parents=True, exist_ok=True)
        signature = self.run_signature(spec, seed)
        done = folder/"training.json"
        if done.exists():
            if load_json(done)["signature"] != signature:
                raise RuntimeError("Niezgodne pochodzenie treningu.")
            if (folder/"best.pt").exists():
                return
        if self.original_reference(spec, seed, folder, signature):
            return
        if self.datasets is None:
            self.datasets = (Patches(args, [i for i in self.meta["items"] if i["split"] == "train"]),
                             Patches(args, [i for i in self.meta["items"] if i["split"] == "val"], training=False))
        training, validation = self.datasets
        training.seed = seed
        train_loader = DataLoader(training, batch_size=args.batch, num_workers=0, pin_memory=device.type == "cuda")
        val_loader = DataLoader(validation, batch_size=args.batch, num_workers=0)
        model = self.model(spec, seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if spec["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=args.lr*0.05)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.05)
        amp = device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        history, best, stale, first = [], -math.inf, 0, 1
        if (folder/"last.pt").exists():
            saved = load_checkpoint(folder/"last.pt")
            if saved["signature"] != signature:
                raise RuntimeError("Checkpoint eksperymentu ma inna sygnature.")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            history, best, stale, first = saved["history"], saved["best"], saved["stale"], saved["epoch"]+1
        print(f"\n{spec['id']} / seed={seed}: start epoki {first}, {sum(p.numel() for p in model.parameters()):,} parametrow", flush=True)
        for epoch in range(first, args.epochs+1):
            if stale >= args.patience:
                break
            start = time.perf_counter()
            seed_all(seed+epoch)
            training.epoch = epoch
            model.train()
            total = 0.0
            skipped = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    prediction = model(x)
                loss = loss_value(prediction, y, spec)
                if not torch.isfinite(loss):
                    raise RuntimeError("Niefinitywna strata eksperymentu.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale)
                total += loss.item()
            psnr, structural = validate(model, val_loader, device)
            if spec["scheduler"] == "plateau":
                scheduler.step(psnr)
            else:
                scheduler.step()
            row = {"epoch": epoch, "loss": total/len(train_loader), "val_psnr": psnr, "val_ssim": structural,
                   "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-start, "amp_skipped_steps": skipped}
            history.append(row)
            improved = psnr > best
            best, stale = (psnr, 0) if improved else (best, stale+1)
            if improved:
                save_checkpoint(folder/"best.pt", {"model": model.state_dict(), "epoch": epoch, "signature": signature})
            save_checkpoint(folder/"last.pt", {"signature": signature, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "history": history, "epoch": epoch, "best": best, "stale": stale})
            save_json(folder/"history.json", history)
            if improved or epoch % 5 == 0 or stale >= args.patience:
                print(f"  epoka {epoch:2}: val PSNR={psnr:.3f}, SSIM={structural:.4f}, lr={row['lr']:.6f}, {row['seconds']:.1f}s, AMP skip={skipped}", flush=True)
        best_state = load_checkpoint(folder/"best.pt")
        save_json(done, {"complete": True, "signature": signature, "parameters": sum(p.numel() for p in model.parameters()),
                        "epochs": len(history), "best_epoch": best_state["epoch"],
                        "training_seconds": sum(r["seconds"] for r in history), "reused_original": False})
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def assess(self, spec, seed, split):
        folder = self.run_folder(spec, seed)
        training = load_json(folder/"training.json")
        checkpoint_hash = sha256(folder/"best.pt")
        output = folder/split
        output.mkdir(exist_ok=True)
        summary_path = output/"summary.json"
        signature = digest({"run": training["signature"], "checkpoint": checkpoint_hash, "split": split})
        if summary_path.exists():
            saved = load_json(summary_path)
            if saved["signature"] != signature:
                raise RuntimeError("Niezgodne wyniki oceny eksperymentu.")
            return saved
        if self.measures is None:
            self.measures = Measures(self.device)
        model = self.model(spec, seed)
        state = load_checkpoint(folder/"best.pt")
        if state["signature"] != training["signature"]:
            raise RuntimeError("Niezgodny checkpoint eksperymentu.")
        model.load_state_dict(state["model"])
        model.eval()
        rows = []
        for item in [i for i in self.meta["items"] if i["split"] == split]:
            cached = output/(item["id"]+".json")
            if cached.exists() and load_json(cached).get("signature") == signature:
                rows.append(load_json(cached)["row"])
                continue
            x, y = read_pair(self.args, item["id"])
            prediction = predict(model, x, self.device, self.args.tile)
            values = self.measures(prediction, y)
            if not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError("Nieprawidlowa metryka w eksperymencie.")
            row = {"image": item["id"], **values}
            if split == "test" and seed == self.seeds[0]:
                atomic_png(prediction, output/(item["id"]+".png"))
            save_json(cached, {"signature": signature, "row": row})
            rows.append(row)
        values = {m: {"mean": float(np.mean([r[m] for r in rows])), "std": float(np.std([r[m] for r in rows], ddof=1))} for m in METRICS}
        summary = {"signature": signature, "count": len(rows), "metrics": values, "rows": rows}
        write_rows(output/"per_image.csv", rows)
        save_json(summary_path, summary)
        print(f"  {spec['id']} seed={seed} {split}: PSNR={values['psnr']['mean']:.3f}, SSIM={values['ssim']['mean']:.4f}, LPIPS={values['lpips']['mean']:.4f}", flush=True)
        del model
        return summary

    def screen(self):
        rows = []
        for spec in self.specs.values():
            self.training(spec, self.seeds[0])
            assessment = self.assess(spec, self.seeds[0], "val")
            training = load_json(self.run_folder(spec, self.seeds[0])/"training.json")
            rows.append({"id": spec["id"], "group": spec["group"], "label": spec["label"],
                         **{m: assessment["metrics"][m]["mean"] for m in METRICS},
                         "epochs": training["epochs"], "best_epoch": training["best_epoch"],
                         "parameters": training["parameters"], "training_minutes": training["training_seconds"]/60})
            write_rows(self.folder/"screening.csv", rows)
            save_json(self.folder/"screening.json", rows)
        shortlist = [r["id"] for r in sorted(rows, key=lambda r: r["psnr"], reverse=True) if r["id"] != "reference"][:2]
        save_json(self.folder/"shortlist.json", {"signature": self.protocol["signature"], "candidates": shortlist, "criterion": "validation PSNR only"})
        return shortlist

    def confirm(self, shortlist):
        rows = []
        for key in ["reference", *shortlist]:
            spec = self.specs[key]
            assessments = []
            for seed in self.seeds:
                self.training(spec, seed)
                assessments.append(self.assess(spec, seed, "val"))
            row = {"id": key, "label": spec["label"], "seeds": self.seeds,
                   "metrics": {m: {"mean": float(np.mean([a["metrics"][m]["mean"] for a in assessments])),
                                   "std": float(np.std([a["metrics"][m]["mean"] for a in assessments], ddof=1))} for m in METRICS}}
            rows.append(row)
            save_json(self.folder/"confirmation.json", rows)
        # The reference is eligible: report a negative experiment if no change helps.
        winner = max(rows, key=lambda r: r["metrics"]["psnr"]["mean"])["id"]
        selection = {"signature": self.protocol["signature"], "winner": winner,
                     "finalists": [r["id"] for r in rows], "criterion": "mean full-image validation PSNR over three seeds",
                     "selected_utc": datetime.now(timezone.utc).isoformat()}
        path = self.folder/"selection.json"
        if path.exists():
            old = load_json(path)
            if any(old[k] != selection[k] for k in ("signature", "winner", "finalists", "criterion")):
                raise RuntimeError("Zapisany wybor na walidacji ulegl zmianie; nie wolno dostrajac po tescie.")
        else:
            save_json(path, selection)
        print("\nWybrany wylacznie na walidacji:", winner, flush=True)

    def test(self):
        selection = load_json(self.folder/"selection.json")
        if selection["signature"] != self.protocol["signature"]:
            raise RuntimeError("Brak zgodnego wyboru modelu na walidacji.")
        rows = []
        for key in selection["finalists"]:
            assessments = [self.assess(self.specs[key], seed, "test") for seed in self.seeds]
            rows.append({"id": key, "label": self.specs[key]["label"], "seeds": self.seeds,
                         "metrics": {m: {"mean": float(np.mean([a["metrics"][m]["mean"] for a in assessments])),
                                         "std": float(np.std([a["metrics"][m]["mean"] for a in assessments], ddof=1))} for m in METRICS}})
        result = {"complete": True, "signature": self.protocol["signature"], "winner": selection["winner"],
                  "seeds": self.seeds, "test_count": 20, "screening": load_json(self.folder/"screening.json"),
                  "confirmation": load_json(self.folder/"confirmation.json"), "test": rows,
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        save_json(self.folder/"summary.json", result)
        self.figures(result)

    def figures(self, result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        folder = self.folder/"figures"
        folder.mkdir(exist_ok=True)
        rows = sorted(result["screening"], key=lambda r: r["psnr"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([r["label"] for r in rows], [r["psnr"] for r in rows], color=["#17324d" if r["id"] == "reference" else "#087f8c" for r in rows])
        ax.set(xlabel="PSNR walidacji [dB]", title="Jedna zmiana względem U-Neta odniesienia / seed 42")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(folder/"screening.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, metric in zip(axes, METRICS):
            values = result["test"]
            ax.bar(range(len(values)), [r["metrics"][metric]["mean"] for r in values],
                   yerr=[r["metrics"][metric]["std"] for r in values], capsize=4, color="#087f8c")
            ax.set_xticks(range(len(values)), [r["label"] for r in values], rotation=15, ha="right", fontsize=8)
            ax.set_title(metric.upper() + (" ↑" if metric != "lpips" else " ↓"))
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(folder/"final_test.png", dpi=170)
        plt.close(fig)
        comparisons = []
        winner = result["winner"]
        first_seed = self.seeds[0]
        a = load_json(self.run_folder(self.specs["reference"], first_seed)/"test"/"summary.json")["rows"]
        b = load_json(self.run_folder(self.specs[winner], first_seed)/"test"/"summary.json")["rows"]
        by_name = {r["image"]: r for r in a}
        gain = {r["image"]: r["psnr"]-by_name[r["image"]]["psnr"] for r in b}
        names = list(dict.fromkeys((a[0]["image"], max(gain, key=gain.get), min(gain, key=gain.get))))
        for name in names:
            x, gt = read_pair(self.args, name)
            def restored(key):
                with Image.open(self.run_folder(self.specs[key], first_seed)/"test"/(name+".png")) as im:
                    return np.array(im)
            images = [x, restored("reference"), restored(winner), gt]
            h, w = gt.shape[:2]
            size = min(256, h, w)
            points = [(int(y), int(x)) for y in np.linspace(0, h-size, 4) for x in np.linspace(0, w-size, 4)]
            cy, cx = max(points, key=lambda p: np.var(gt[p[0]:p[0]+size, p[1]:p[1]+size].mean(axis=2)))
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for column, (im, title) in enumerate(zip(images, ("Wejście -3 EV", "Oryginalny U-Net", self.specs[winner]["label"], "Referencja"))):
                axes[0, column].imshow(im)
                axes[0, column].set_title(title)
                axes[0, column].add_patch(plt.Rectangle((cx, cy), size, size, fill=False, edgecolor="#ffbf00", linewidth=1.2))
                axes[1, column].imshow(im[cy:cy+size, cx:cx+size])
                for ax in axes[:, column]:
                    ax.axis("off")
            fig.tight_layout(pad=.5)
            filename = f"comparison_{name}.jpg"
            fig.savefig(folder/filename, dpi=170, pil_kwargs={"quality": 92})
            plt.close(fig)
            comparisons.append({"image": name, "figure": filename, "psnr_gain": gain[name]})
        save_json(folder/"comparisons.json", comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--study", type=Path, default=ROOT/"results"/"experiments")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    options = parser.parse_args()
    args = project_parser().parse_args(["train", "--device", options.device])
    seeds = [42, 43, 44]
    try:
        study = Study(args, options.study, seeds)
        if options.stage in ("all", "train"):
            study.confirm(study.screen())
        if options.stage in ("all", "evaluate"):
            study.test()
        print("Badania zakonczone. Raport: python report.py", flush=True)
    except KeyboardInterrupt:
        print("Przerwano. Ponow identyczne polecenie; zapisane etapy zostana wznowione.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"BLAD: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
