"""Statyczny raport HTML/Markdown oraz figury gotowe do prezentacji."""

import csv
import html
import math
from pathlib import Path

from .common import load_json


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def comparison(path, clean, degraded, baseline, prediction, condition, baseline_label):
    import numpy as np
    plt = pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), constrained_layout=True)
    images = (clean, degraded, baseline, prediction)
    labels = ("Referencja 256 x 256", "Wejście (SR: bicubic)", baseline_label, "ResidualNet")
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[0, i].imshow(image, vmin=0, vmax=1)
        axes[0, i].set_title(label, fontsize=10)
        axes[0, i].axis("off")
        if i == 0:
            axes[1, i].imshow(clean[80:176, 80:176])
            axes[1, i].set_title("Referencja: wycinek 96 x 96", fontsize=9)
        else:
            error = np.mean(np.abs(clean - image), axis=2)
            axes[1, i].imshow(error, cmap="magma", vmin=0, vmax=0.1)
            axes[1, i].set_title("Średni błąd |RGB|, skala 0-0.1", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle(condition)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def fmt(value):
    if value is None or value == "":
        return "brak"
    number = float(value)
    return f"{number:.5f}" if math.isfinite(number) else str(number)


def conclusions(rows):
    result = []
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        neural = next(r for r in group if r["method"].startswith("ResidualNet"))
        baseline = next(r for r in group if not r["method"].startswith(("ResidualNet", "noisy")))
        delta = float(neural["PSNR"]) - float(baseline["PSNR"])
        ssim_delta = float(neural["SSIM"]) - float(baseline["SSIM"])
        psnr_text = f"{delta:+.3f} dB" if math.isfinite(delta) else "nieokreślona (wartości nieskończone)"
        text = (f"{condition}: zmiana PSNR sieci względem metody bazowej wynosi {psnr_text}, "
                f"a SSIM {ssim_delta:+.5f}.")
        if neural.get("LPIPS") is not None and baseline.get("LPIPS") is not None:
            difference = float(neural["LPIPS"]) - float(baseline["LPIPS"])
            text += f" Zmiana LPIPS: {difference:+.5f} (mniej = lepiej)."
        if neural.get("SNE") is not None and baseline.get("SNE") is not None:
            difference = float(neural["SNE"]) - float(baseline["SNE"])
            text += f" Zmiana SNE: {difference:+.5f} (mniej = lepiej)."
        if math.isfinite(delta):
            text += " Nie stwierdzono poprawy PSNR." if delta <= 0 else " Uzyskano poprawę średniego PSNR."
        result.append(text)
    return result


def make_report(output, runs="runs/main", write_markdown=False):
    output, runs = Path(output), Path(runs)
    meta = load_json(output / "metadata.json")
    rows = load_json(output / "summary.json")
    comments = conclusions(rows)
    warning = []
    if meta["dataset"] != "DIV2K":
        warning.append("PRÓBA TECHNICZNA NA DANYCH SYNTETYCZNYCH. Wyniki nie są wynikami DIV2K.")
    if meta["test_protocol"] == "public-validation-proxy":
        warning.append("Użyto publicznych DIV2K validation HR jako testu zastępczego. "
                       "Oficjalne DIV2K test HR nie są publicznie udostępnione.")
    if not meta["full_split"]:
        warning.append("Przetworzono ograniczoną liczbę obrazów. To nie pełna ewaluacja zbioru.")
    if not meta.get("primary_metrics_complete", meta["complete_metrics"]):
        warning.append("Pominięto LPIPS; raport nie zawiera kompletu wymaganych metryk.")
    if meta["split"] != "test":
        warning.append(f"Ewaluacja na podziale {meta['split']}; nie jest to wynik testowy.")
    if meta.get("sne_definition") != "sse":
        warning.append("Brak SNE zgodnego z wykładem. Slajd 41 definiuje SNE jako sumę kwadratów "
                       "błędów (SSE), bez normalizacji. Należy uzupełnić zapisane wyniki.")
    plots = []
    plt = pyplot()
    chart_folder = output / "charts"
    chart_folder.mkdir(exist_ok=True)
    for condition in meta["conditions"]:
        history_path = runs / condition / "history.csv"
        if history_path.exists():
            with history_path.open(encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
            epochs = [int(r["epoch"]) for r in history]
            axes[0].plot(epochs, [float(r["train_loss"]) for r in history])
            axes[0].set(xlabel="Epoka", ylabel="Strata treningowa", title=condition)
            axes[1].plot(epochs, [float(r["val_PSNR"]) for r in history])
            axes[1].set(xlabel="Epoka", ylabel="PSNR walidacji [dB]", title="Wybór checkpointu")
            path = chart_folder / f"{condition}_training.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            plots.append(path.relative_to(output).as_posix())
    figures = sorted(p.relative_to(output).as_posix() for p in (output / "images").glob("*.png"))
    columns = ("condition", "method", "images", "PSNR", "SSIM", "LPIPS", "SNE", "MSE")
    headers = ("Wariant", "Metoda", "N", "PSNR ↑ [dB]", "SSIM ↑", "LPIPS ↓", "SNE ↓ (=SSE)", "MSE ↓")
    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    markdown = ["# SIGK - wyniki eksperymentów", "", *[f"> {w}\n" for w in warning],
                f"Zbiór: {meta['dataset']}; podział: {meta['split']}; N={meta['image_count']}; seed={meta['seed']}.",
                "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [str(row[c]) if c in ("condition", "method", "images") else fmt(row.get(c)) for c in columns]
        table += "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"
        markdown.append("| " + " | ".join(cells) + " |")
    table += "</tbody></table>"
    protocol = (f"{meta['metric_protocol']}. LPIPS: {meta['lpips_net'] or 'pominięto'}. "
                f"Definicja zapisanych wyników SNE: {meta['sne_definition'] or 'brak'}. "
                "SNE (Squared Norm Error) = suma (referencja - wynik)^2 po wszystkich pikselach "
                "i kanałach RGB [0,1], zgodnie ze slajdem 41 wykładu. SNE = SSE = 256*256*3*MSE. "
                "SSIM: okno Gaussa, sigma=1.5, data_range=1, covariance population. "
                "Tabele zawierają średnie arytmetyczne metryk liczonych osobno dla każdego obrazu. "
                "Pełna konfiguracja i wersje bibliotek: metadata.json; wyniki jednostkowe: per_image.csv.")
    architecture = ("ResidualNet: Conv 3x3, bloki residualne Conv-ReLU-Conv ze skalą 0.1, "
                    "globalny skip cech i rekonstrukcja RGB. SR wykorzystuje PixelShuffle x2 "
                    "powtarzany 2/3 razy oraz dodaje poprawkę do OpenCV bicubic. "
                    "Odszumianie dodaje uczoną poprawkę do wejścia. Osobne wagi dla każdego wariantu. "
                    "Brak BatchNorm; zerowa inicjalizacja ostatniej warstwy. "
                    "Najlepszą epokę wybrano według PSNR wewnętrznej walidacji; test nie służy do doboru wag.")
    training_details = []
    for condition, checkpoint in meta.get("checkpoints", {}).items():
        config = checkpoint.get("training_config", {})
        arch = checkpoint["architecture"]
        text = (f"{condition}: {arch['width']} kanałów, {arch['blocks']} bloków; "
                f"wybrana epoka {checkpoint['epoch']}.")
        if config:
            text += (f" Plan: {config['epochs']} epok, repeat={config['repeat']}, "
                     f"batch={config['batch_size']}, loss={config['loss']}, lr={config['lr']}, "
                     f"precyzja treningu={config.get('precision', 'fp32')}, "
                     f"generator szumu={config.get('training_noise', 'numpy')}; walidacja FP32.")
        training_details.append(text)
    baseline_details = f"Parametry bilateral: {meta['bilateral']}."
    stylesheet = """
    :root{color-scheme:light}body{font:16px/1.6 system-ui,sans-serif;color:#172333;background:#eef2f6;margin:0}
    main{max-width:1220px;margin:32px auto;padding:36px;background:white;border-radius:16px}
    h1{font-size:32px;margin-top:0}h2{margin-top:36px}p,li{max-width:100ch}
    .notice{background:#fff2d8;border-left:4px solid #d48d00;padding:12px 18px;margin:14px 0}
    .table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}
    th{background:#17354d;color:white;text-align:left}th,td{padding:10px;border-bottom:1px solid #dce4eb}
    tr:nth-child(even){background:#f5f8fa}img{max-width:100%;height:auto;border:1px solid #e2e8f0;margin:12px 0}
    a{color:#1268aa}code{background:#edf2f6;padding:2px 5px}@media print{body{background:white}main{margin:0;padding:0}img{break-inside:avoid}}
    """
    body = ("<!doctype html><html lang='pl'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>SIGK - raport eksperymentów</title><style>{stylesheet}</style><main>"
            "<h1>Modyfikacja obrazów / SIGK</h1>"
            f"<p>Zbiór: <b>{html.escape(meta['dataset'])}</b> · podział {html.escape(meta['split'])} · "
            f"{meta['image_count']} obrazów na wariant · seed {meta['seed']}</p>"
            + "".join(f"<div class='notice'>{html.escape(w)}</div>" for w in warning)
            + "<h2>Wyniki pomiarów</h2><div class='table'>" + table + "</div>"
            + "<h2>Protokół</h2><p>" + html.escape(protocol) + "</p>"
            + "<h2>Architektura</h2><p>" + html.escape(architecture) + "</p>"
            + "<ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in training_details) + "</ul>"
            + "<p>" + html.escape(baseline_details) + "</p>"
            + "<h2>Wnioski z pomiarów</h2><ul>"
            + "".join(f"<li>{html.escape(c)}</li>" for c in comments) + "</ul>"
            + "<p>Są to porównania średnich na tym zbiorze, bez testu istotności statystycznej. "
              "Wynik PSNR nie przesądza o jakości percepcyjnej; należy również sprawdzić LPIPS i obrazy.</p>"
            + "<h2>Przebieg treningu</h2>"
            + "".join(f"<img loading='lazy' src='{html.escape(p, quote=True)}' alt='Trening {html.escape(Path(p).stem, quote=True)}'>" for p in plots)
            + "<h2>Porównania obrazów i mapy błędu</h2>"
            + "".join(f"<img loading='lazy' src='{html.escape(p, quote=True)}' alt='Porównanie {html.escape(Path(p).stem, quote=True)}'>" for p in figures)
            + "<h2>Dane źródłowe</h2><p><a href='summary.csv'>Tabela CSV</a> · "
              "<a href='per_image.csv'>Pomiary każdego obrazu</a> · "
              "<a href='metadata.json'>Konfiguracja</a></p></main></html>")
    (output / "raport.html").write_text(body, encoding="utf-8")
    markdown += ["", "## Protokół", "", protocol, "", "## Architektura", "", architecture,
                 "", *[f"- {t}" for t in training_details], "", baseline_details,
                 "", "## Wnioski", "", *[f"- {c}" for c in comments], "", "## Wizualizacje", ""]
    markdown += [f"![{Path(p).stem}]({p})\n" for p in plots + figures]
    if write_markdown:
        (output / "raport.md").write_text("\n".join(markdown), encoding="utf-8")
