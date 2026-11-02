"""Interfejs: python -m src --help. Importy ciężkich bibliotek są opóźnione."""

import argparse
import os
from pathlib import Path

from .common import CONDITIONS


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Wartość musi być dodatnia.")
    return result


def positive_float(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("Wartość musi być dodatnia i skończona.")
    return result


def parser():
    p = argparse.ArgumentParser(description="SIGK: super-resolution i odszumianie DIV2K")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--raw", default="data/raw")
    d = sub.add_parser("prepare")
    d.add_argument("--raw", default="data/raw")
    d.add_argument("--data", default="data/div2k256")
    d.add_argument("--test-dir")
    d = sub.add_parser("demo-data")
    d.add_argument("--data", default="data/demo")
    d.add_argument("--seed", type=int, default=123)
    d = sub.add_parser("verify")
    d.add_argument("--data", default="data/div2k256")
    t = sub.add_parser("train")
    t.add_argument("--condition", required=True, choices=CONDITIONS)
    t.add_argument("--data", default="data/div2k256")
    t.add_argument("--runs", default="runs/main")
    t.add_argument("--epochs", type=positive_int, default=60)
    t.add_argument("--batch-size", type=positive_int, default=8)
    t.add_argument("--width", type=positive_int, default=32)
    t.add_argument("--blocks", type=positive_int, default=6)
    t.add_argument("--repeat", type=positive_int, default=4)
    t.add_argument("--lr", type=positive_float, default=0.001)
    t.add_argument("--loss", choices=("mse", "l1"), default="mse")
    t.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    t.add_argument("--training-noise", choices=("auto", "numpy", "torch"), default="auto")
    t.add_argument("--workers", type=int, choices=range(0, 17), default=0)
    t.add_argument("--resume", action="store_true")
    b = sub.add_parser("tune-baseline")
    b.add_argument("--data", default="data/div2k256")
    b.add_argument("--output", default="runs/main/bilateral.json")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", default="data/div2k256")
    e.add_argument("--runs", default="runs/main")
    e.add_argument("--output", default="runs/evaluation")
    e.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    e.add_argument("--split", choices=("val", "test"), default="test")
    e.add_argument("--limit", type=positive_int)
    e.add_argument("--examples", type=positive_int, default=5)
    e.add_argument("--bilateral-config")
    e.add_argument("--lpips-net", choices=("squeeze", "alex", "vgg"), default="squeeze")
    e.add_argument("--skip-lpips", action="store_true")
    e.add_argument("--sne-definition", choices=("sse",), default="sse")
    i = sub.add_parser("predict")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", required=True)
    r = sub.add_parser("report")
    r.add_argument("--output", default="runs/evaluation")
    r.add_argument("--runs", default="runs/main")
    r.add_argument("--markdown", action="store_true")
    for command in (t, b, e):
        command.add_argument("--seed", type=int, default=42)
    for command in (t, e, i):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in (t, e):
        command.add_argument("--threads", type=positive_int, default=4)
    sub.add_parser("doctor")
    return p


def main():
    # Cache wag i fontów wewnątrz projektu, także w środowiskach bez zapisu do HOME.
    project = Path(__file__).resolve().parent.parent
    os.environ.setdefault("TORCH_HOME", str(project / ".cache" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".cache" / "matplotlib"))
    p = parser()
    args = p.parse_args()
    try:
        if args.command == "download":
            from .data import download
            download(args.raw)
        elif args.command == "prepare":
            from .data import prepare
            prepare(args.raw, args.data, args.test_dir)
        elif args.command == "verify":
            from .data import verify_data
            manifest = verify_data(args.data)
            print({k: len(v) for k, v in manifest["splits"].items()})
        elif args.command == "demo-data":
            from .demo import make_demo
            make_demo(args.data, args.seed)
        elif args.command == "report":
            from .report import make_report
            make_report(args.output, args.runs, write_markdown=args.markdown)
            print(Path(args.output, "raport.html").resolve())
        elif args.command == "doctor":
            from .engine import environment
            import json
            print(json.dumps(environment(), ensure_ascii=False, indent=2))
        else:
            from .engine import evaluate, predict, train, tune
            {"train": train, "evaluate": evaluate, "predict": predict, "tune-baseline": tune}[args.command](args)
    except ModuleNotFoundError as error:
        p.exit(1, f"Brak zależności: {error}. Zainstaluj: python -m pip install -r requirements.txt\n")
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        p.exit(1, f"Błąd: {error}\n")
