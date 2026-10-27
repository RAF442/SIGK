"""Przygotuj wejścia i wyniki predict do prezentacji, po wytrenowaniu 4 modeli."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))

from sigk.common import CONDITIONS
from sigk.data import fixed_degradation, read_rgb, save_rgb
from sigk.engine import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/div2k256"))
    parser.add_argument("--runs", default=str(ROOT / "runs/main"))
    parser.add_argument("--output", default=str(ROOT / "examples"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.data) / "val/0721.png"
    clean = read_rgb(str(source))
    save_rgb(output / "reference.png", clean)
    for condition in CONDITIONS:
        degraded, _ = fixed_degradation(clean, condition, 42, source.name)
        input_path = output / f"input_{condition}.png"
        save_rgb(input_path, degraded)
        predict(argparse.Namespace(device=args.device,
                                  checkpoint=str(Path(args.runs) / condition / "best.pt"),
                                  input=str(input_path), output=str(output / f"output_{condition}.png")))
    (output / "README.md").write_text(
        "# Przykłady działania predict\n\n"
        "Referencja: DIV2K train HR 0721, wewnętrzny podział val, resize RGB 256x256.\n"
        "Dane służą do prezentacji działania; nie są końcowym zbiorem testowym.\n\n"
        "Wejścia input_sr4/input_sr8 mają rozmiary 64/32 px, a input_dn001/input_dn003 "
        "256 px i odpowiednio szum sigma 0.01/0.03 (seed=42). Wszystkie zapisano jako PNG 8-bit. "
        "Wyjścia output_* pochodzą z komendy predict na zapisanych wejściach. "
        "Metryki końcowego raportu obliczono osobno na danych float przed zapisem PNG.\n\n"
        "Przykładowa komenda z katalogu projektu:\n\n"
        "```powershell\n.venv/Scripts/python.exe -m sigk predict --checkpoint runs/main/sr4/best.pt "
        "--input examples/input_sr4.png --output examples/my_result.png\n```\n\n"
        "Źródło i licencja akademicka: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
