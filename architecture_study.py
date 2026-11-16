"""Controlled comparison of U-Net, ResU-Net, small NAFNet and small MIRNet.

python architecture_study.py                 # train, validate, test, figures
python architecture_study.py --stage train
python architecture_study.py --stage evaluate
python report.py                            # update the single report PDF

All architectures use direct RGB, L1 + 0.2*(1-SSIM), the same data, crops,
optimizer, training limit and three seeds. No pretrained reconstruction weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

from data import ROOT, digest, load_json, read_pair, require_prepared, save_json, sha256, tensor
from main import parser as project_parser
from train import device_for, load_checkpoint, save_checkpoint, seed_all
from experiments import Study
from architectures import build_architecture

import numpy as np
from PIL import Image
import torch

METRICS = ("psnr", "ssim", "lpips")


def specifications():
    return [{"id": name, "architecture": name, "label": label, "width": width,
             "output": "direct", "loss": "l1", "ssim_weight": 0.2, "scheduler": "cosine"}
            for name, label, width in (("unet", "U-Net RGB", 24), ("resunet", "ResU-Net RGB", 24),
                                       ("nafnet", "NAFNet-small", 30), ("mirnet", "MIRNet-small", 28))]


def comparison_protocol(args, seeds):
    meta = require_prepared(args)
    value = {"version": 1, "data_signature": meta["signature"], "seeds": seeds,
             "hyperparameters": {k: getattr(args, k) for k in ("width", "ev", "gamma", "epochs", "steps", "batch", "patch", "lr", "patience", "seed", "tile", "no_amp")},
             "specifications": specifications(), "selection": "mean full-image validation PSNR across three seeds",
             "checkpoint_selection": "PSNR on 32 fixed validation crops; stop after 15 unimproved epochs",
             "common_output": "zero-initialized head, direct RGB sigmoid; no image-level residual shortcut",
             "nafnet": {"encoder_blocks": [1, 1, 2], "middle_blocks": 1, "decoder_blocks": [1, 1, 1],
                         "layernorm": "native PyTorch channel norm in float32", "TLC": False},
             "mirnet": {"residual_groups": 1, "MSRB_per_group": 1, "scales": 3, "columns": 2,
                         "shared_DAU_across_columns": True, "antialias": "3x3 binomial blur"},
             "evaluation": "uint8 RGB; 512 core / 64 halo; identical PSNR, SSIM and LPIPS-AlexNet",
             "benchmark": "batch1 patch256 float32, 5 warmups and 20 timed forwards, trained seed42 model",
             "torch": str(torch.__version__), "source_sha256": {n: sha256(ROOT/n) for n in
                 ("architecture_study.py", "architectures.py", "experiments.py", "unet_variants.py", "model.py", "data.py", "train.py", "evaluate.py")}}
    value["signature"] = digest(value)
    return value, meta


class ArchitectureStudy(Study):
    def __init__(self, args, folder, seeds=(42, 43, 44)):
        self.args, self.folder, self.seeds = args, folder, list(seeds)
        self.protocol, self.meta = comparison_protocol(args, self.seeds)
        self.device = device_for(args)
        self.specs = {s["id"]: s for s in specifications()}
        self.datasets = self.measures = None
        folder.mkdir(parents=True, exist_ok=True)
        p = folder/"protocol.json"
        if p.exists() and load_json(p) != self.protocol:
            raise RuntimeError("Zmieniono konfiguracje porownania; wybierz nowy --study.")
        save_json(p, self.protocol)

    def model(self, spec, seed):
        seed_all(seed)
        return build_architecture(self.args, spec).to(self.device)

    def previous_folder(self, seed):
        return self.args.results/"experiments"/"runs"/f"output_direct_s{seed}"

    def original_reference(self, spec, seed, folder, signature):
        if spec["architecture"] != "unet":
            return False
        previous_study = self.args.results/"experiments"
        previous = self.previous_folder(seed)
        if not (previous/"training.json").exists() or not (previous_study/"protocol.json").exists():
            return False
        protocol = load_json(previous_study/"protocol.json")
        if protocol["data_signature"] != self.protocol["data_signature"] or protocol["hyperparameters"] != self.protocol["hyperparameters"]:
            return False
        if any(sha256(ROOT/name) != value for name, value in protocol["source_sha256"].items()):
            return False
        old = load_json(previous/"training.json")
        best = load_checkpoint(previous/"best.pt")
        if not old["complete"] or best["signature"] != old["signature"]:
            raise RuntimeError("Niespojny poprzedni trening U-Neta.")
        model = self.model(spec, seed)
        model.load_state_dict(best["model"], strict=True)
        save_checkpoint(folder/"best.pt", {"model": best["model"], "epoch": best["epoch"], "signature": signature})
        save_json(folder/"history.json", load_json(previous/"history.json"))
        save_json(folder/"training.json", {**old, "signature": signature, "parameters": sum(p.numel() for p in model.parameters()),
                                          "reused_previous": True, "previous_checkpoint_sha256": sha256(previous/"best.pt")})
        print(f"U-Net seed={seed}: wykorzystuje poprzedni ukonczony trening RGB.", flush=True)
        return True

    def assess(self, spec, seed, split):
        folder = self.run_folder(spec, seed)
        record = load_json(folder/"training.json")
        previous = self.previous_folder(seed)
        if record.get("reused_previous") and (previous/split/"summary.json").exists():
            old = load_json(previous/split/"summary.json")
            old_record = load_json(previous/"training.json")
            source_hash = sha256(previous/"best.pt")
            expected = digest({"run": old_record["signature"], "checkpoint": source_hash, "split": split})
            if old["signature"] != expected or source_hash != record["previous_checkpoint_sha256"]:
                raise RuntimeError("Zmieniono poprzednie wyniki U-Neta.")
            signature = digest({"run": record["signature"], "checkpoint": sha256(folder/"best.pt"), "split": split})
            output = folder/split
            output.mkdir(exist_ok=True)
            summary = {**old, "signature": signature, "reused_previous": True}
            save_json(output/"summary.json", summary)
            if split == "test" and seed == self.seeds[0]:
                for row in old["rows"]:
                    source = previous/split/(row["image"]+".png")
                    target = output/source.name
                    if source.exists() and not target.exists():
                        shutil.copyfile(source, target)
            return summary
        return super().assess(spec, seed, split)

    def aggregate(self, key, split):
        assessed = [load_json(self.run_folder(self.specs[key], seed)/split/"summary.json") for seed in self.seeds]
        training = [load_json(self.run_folder(self.specs[key], seed)/"training.json") for seed in self.seeds]
        return {"id": key, "label": self.specs[key]["label"], "width": self.specs[key]["width"],
                "parameters": training[0]["parameters"], "seeds": self.seeds,
                "epochs": [r["epochs"] for r in training], "best_epochs": [r["best_epoch"] for r in training],
                "training_minutes_mean": float(np.mean([r["training_seconds"]/60 for r in training])),
                "reused_previous": all(r.get("reused_previous", False) for r in training),
                "metrics": {m: {"mean": float(np.mean([a["metrics"][m]["mean"] for a in assessed])),
                                "std": float(np.std([a["metrics"][m]["mean"] for a in assessed], ddof=1))} for m in METRICS}}

    def train_all(self):
        rows = []
        for key, spec in self.specs.items():
            for seed in self.seeds:
                self.training(spec, seed)
                self.assess(spec, seed, "val")
            rows.append(self.aggregate(key, "val"))
            save_json(self.folder/"validation.json", rows)
        winner = max(rows, key=lambda r: r["metrics"]["psnr"]["mean"])["id"]
        selection = {"signature": self.protocol["signature"], "winner": winner, "architectures": list(self.specs),
                     "criterion": self.protocol["selection"], "selected_utc": datetime.now(timezone.utc).isoformat()}
        path = self.folder/"selection.json"
        if path.exists():
            old = load_json(path)
            if any(old[k] != selection[k] for k in ("signature", "winner", "architectures", "criterion")):
                raise RuntimeError("Wybor architektury ulegl zmianie po tescie.")
        else:
            save_json(path, selection)
        print("\nArchitektura wybrana na walidacji:", winner, flush=True)

    @torch.inference_mode()
    def benchmark(self):
        path = self.folder/"benchmark.json"
        if path.exists():
            return load_json(path)
        item = next(i for i in self.meta["items"] if i["split"] == "val")
        image, _ = read_pair(self.args, item["id"])
        x = tensor(image[:256, :256]).unsqueeze(0).to(self.device)
        rows = []
        for key, spec in self.specs.items():
            model = self.model(spec, self.seeds[0]).eval()
            model.load_state_dict(load_checkpoint(self.run_folder(spec, self.seeds[0])/"best.pt")["model"])
            for _ in range(5):
                model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            milliseconds = []
            for _ in range(20):
                start = time.perf_counter()
                model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                milliseconds.append((time.perf_counter()-start)*1000)
            rows.append({"id": key, "median_ms": float(np.median(milliseconds)),
                         "p90_ms": float(np.percentile(milliseconds, 90)),
                         "activation_mib": (torch.cuda.max_memory_allocated()-baseline)/1024**2 if self.device.type == "cuda" else None})
            del model
        save_json(path, rows)
        return rows

    def test_all(self):
        selection = load_json(self.folder/"selection.json")
        if selection["signature"] != self.protocol["signature"]:
            raise RuntimeError("Najpierw wykonaj wybor na walidacji.")
        benchmark = self.benchmark()
        rows = []
        for key, spec in self.specs.items():
            for seed in self.seeds:
                self.assess(spec, seed, "test")
            rows.append(self.aggregate(key, "test"))
        result = {"complete": True, "signature": self.protocol["signature"], "winner": selection["winner"],
                  "seeds": self.seeds, "test_count": 20, "validation": load_json(self.folder/"validation.json"),
                  "test": rows, "benchmark": benchmark, "finished_utc": datetime.now(timezone.utc).isoformat()}
        save_json(self.folder/"summary.json", result)
        self.figures(result)

    def figures(self, result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        folder = self.folder/"figures"
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, metric in zip(axes, METRICS):
            ax.bar(range(4), [r["metrics"][metric]["mean"] for r in result["test"]],
                   yerr=[r["metrics"][metric]["std"] for r in result["test"]], capsize=4, color="#087f8c")
            ax.set_xticks(range(4), [r["label"] for r in result["test"]], rotation=20, ha="right", fontsize=8)
            ax.set_title(metric.upper()+ (" ↑" if metric != "lpips" else " ↓"))
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(folder/"test_metrics.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (key, spec) in zip(axes.flat, self.specs.items()):
            for seed in self.seeds:
                h = load_json(self.run_folder(spec, seed)/"history.json")
                ax.plot([r["epoch"] for r in h], [r["val_psnr"] for r in h], label=f"seed {seed}")
            ax.set(title=spec["label"], xlabel="Epoka", ylabel="PSNR fragmentów [dB]")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/"learning.png", dpi=170)
        plt.close(fig)
        first_seed = self.seeds[0]
        per_model = {key: {r["image"]: r for r in load_json(self.run_folder(spec, first_seed)/"test"/"summary.json")["rows"]}
                     for key, spec in self.specs.items()}
        winner = result["winner"]
        gains = {name: per_model[winner][name]["psnr"]-r["psnr"] for name, r in per_model["unet"].items()}
        names = list(dict.fromkeys((next(iter(gains)), "r35f565f8t", max(gains, key=gains.get), min(gains, key=gains.get))))
        names = [n for n in names if n in gains]
        cases = []
        for name in names:
            x, target = read_pair(self.args, name)
            predictions = []
            for key, spec in self.specs.items():
                with Image.open(self.run_folder(spec, first_seed)/"test"/(name+".png")) as im:
                    predictions.append(np.array(im))
            images = [x, *predictions, target]
            h, w = target.shape[:2]
            size = min(256, h, w)
            points = [(int(y), int(x)) for y in np.linspace(0, h-size, 4) for x in np.linspace(0, w-size, 4)]
            cy, cx = max(points, key=lambda p: np.var(target[p[0]:p[0]+size, p[1]:p[1]+size].mean(axis=2)))
            fig, axes = plt.subplots(2, 6, figsize=(18, 7))
            titles = ["Wejście -3 EV", *[s["label"] for s in self.specs.values()], "Referencja"]
            for column, (im, title) in enumerate(zip(images, titles)):
                axes[0, column].imshow(im)
                axes[0, column].set_title(title, fontsize=13)
                axes[0, column].add_patch(plt.Rectangle((cx, cy), size, size, fill=False, edgecolor="#ffbf00", linewidth=1.2))
                axes[1, column].imshow(im[cy:cy+size, cx:cx+size])
                for ax in axes[:, column]:
                    ax.axis("off")
            fig.tight_layout(pad=.6)
            filename = f"architecture_{name}.jpg"
            fig.savefig(folder/filename, dpi=170, pil_kwargs={"quality": 93})
            plt.close(fig)
            cases.append({"image": name, "figure": filename, "winner_psnr_gain": gains[name],
                          "metrics": {key: per_model[key][name] for key in self.specs}})
        save_json(folder/"comparisons.json", cases)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--study", type=Path, default=ROOT/"results"/"architectures")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    options = p.parse_args()
    args = project_parser().parse_args(["train", "--device", options.device])
    try:
        study = ArchitectureStudy(args, options.study)
        if options.stage in ("all", "train"):
            study.train_all()
        if options.stage in ("all", "evaluate"):
            study.test_all()
        print("Porownanie zakonczone. Raport: python report.py", flush=True)
    except KeyboardInterrupt:
        print("Przerwano. Ponow identyczne polecenie, aby wznowic.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"BLAD: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
