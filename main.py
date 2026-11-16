"""SIGK 2: run the low-light enhancement experiment from the command line.

Run: python main.py all
Other stages: check, download, prepare, train, evaluate.
Use --help for options. Generate the PDF afterwards with python report.py.

Implementation: data.py (input pairs), train.py (learning), evaluate.py (results).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from data import ROOT, download, prepare
from train import train
from evaluate import check, evaluate

import requests


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("all", "check", "download", "prepare", "train", "evaluate"))
    p.add_argument("--csv", type=Path, default=ROOT / "RAISE_258.csv")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps", type=int, default=80, help="Liczba batchy na epoke")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ev", type=float, default=-3.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--sigma", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--with-lpips", action="store_true", help="check: pobierz wagi i sprawdz LPIPS")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if any(getattr(args, k) <= 0 for k in ("epochs", "steps", "batch", "patch", "width", "patience", "tile", "lr", "alpha", "gamma")) or args.sigma < 0 or args.seed < 0 or args.ssim_weight < 0:
        p.error("Rozmiary, liczby krokow, lr, alpha i gamma musza byc dodatnie; sigma, seed i waga SSIM nieujemne.")
    if args.patch < 16 or args.patch % 8 or args.tile % 8:
        p.error("patch >= 16; patch i tile musza byc wielokrotnosciami 8.")
    stages = {"check": check, "download": download, "prepare": prepare, "train": train, "evaluate": evaluate}
    start = time.perf_counter()
    try:
        for name in ("check", "download", "prepare", "train", "evaluate") if args.stage == "all" else (args.stage,):
            print(f"\n=== {name.upper()} ===", flush=True)
            stages[name](args)
    except KeyboardInterrupt:
        print("\nPrzerwano. Ponow to samo polecenie; trening wroci do ostatniej zakonczonej epoki.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, requests.RequestException) as exc:
        print(f"\nBLAD: {exc}", file=sys.stderr)
        return 1
    print(f"\nUkonczono w {(time.perf_counter()-start)/60:.1f} min.")
    if args.stage in ("all", "evaluate"):
        print('Dane do sprawozdania gotowe. Uruchom: python report.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
