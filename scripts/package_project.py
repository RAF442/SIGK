"""Spakuj źródła, wytrenowane modele i końcowy raport, bez DIV2K i .venv."""

from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    if not (root / "runs/evaluation/raport.html").is_file():
        raise SystemExit("Najpierw zakończ ewaluację w runs/evaluation.")
    files = []
    for name in ("README.md", "STATUS.md", "pyproject.toml", "requirements.txt",
                 "requirements-lock.txt", "SIGK___Projekt_1.pdf", ".gitignore"):
        files.append(root / name)
    for folder in ("sigk", "tests", "scripts", "docs", "examples", "runs/main", "runs/evaluation"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".tmp")
    output = root / "output/SIGK_projekt.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Uszkodzony plik w archiwum: {bad}")
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
