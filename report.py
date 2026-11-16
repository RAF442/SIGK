"""Generate the sole documentation artifact with LaTeX: sprawozdanie.pdf.

python report.py                  # requires completed real experiments
python report.py --draft          # methodology only, explicitly without results
python report.py --authors "Imie Nazwisko, Imie Nazwisko"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

from report_latex import LatexReport

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"results"/".matplotlib"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def study_learning_figures(folder, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {r["id"]: r["label"] for r in rows}
    histories = {key: load(folder/"runs"/(key+"_s42")/"history.json") for key in labels}
    panels = [
        ("Sposób wyjścia", ("output_direct", "output_rgb_residual")),
        ("Harmonogram uczenia", ("lr_plateau",)),
        ("Funkcja straty", ("loss_mse", "loss_charbonnier")),
        ("Waga SSIM", ("ssim_005", "ssim_05")),
        ("Aktywacja", ("activation_prelu", "activation_silu")),
        ("Bloki i uwaga", ("residual_blocks", "channel_attention")),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax, (title, keys) in zip(axes.flat, panels):
        for key in ("reference", *keys):
            h = histories[key]
            ax.plot([r["epoch"] for r in h], [r["val_psnr"] for r in h], label=labels[key],
                    linewidth=1.2, **({"color": "#17324d"} if key == "reference" else {}))
        ax.set(title=title, xlabel="Epoka", ylabel="PSNR fragmentów [dB]")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder/"figures"/"learning_groups.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 2.5))
    for key in ("reference", "lr_plateau"):
        h = histories[key]
        ax.plot([r["epoch"] for r in h], [r["lr"] for r in h], label=labels[key])
    ax.set(xlabel="Epoka", ylabel="Learning rate", title="Harmonogramy learning rate")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/"figures"/"learning_rates.png", dpi=170)
    plt.close(fig)


def build(args):
    document = LatexReport(args.authors)
    story = document.parts
    p, heading, table, figure = document.paragraph, document.heading, document.table, document.figure
    width = 509.27559  # Relative column proportions retained from the result tables.

    run_path = args.results / "config.json"
    summary_path = args.results / "evaluation" / "summary.json"
    complete = summary_path.exists() and run_path.exists()
    if complete:
        run, summary = load(run_path), load(summary_path)
        complete = summary.get("complete") and summary.get("count") == 20 and summary.get("signature") == run.get("signature")
    if not complete and not args.draft:
        raise RuntimeError("Brak kompletnych wynikow 20 zdjec. Uruchom main.py all. Do wersji metodologicznej sluzy --draft.")
    # --draft is deliberately a methodology report, even if old results exist.
    complete = complete and not args.draft
    default = {"seed": 42, "width": 24, "ev": -3.0, "gamma": 2.2, "epochs": 80, "steps": 80,
               "batch": 8, "patch": 256, "lr": 0.0002, "ssim_weight": 0.2, "patience": 15}
    config = run["config"] if complete else default
    metadata = run["protocol"] if complete else {"alpha": 0.001, "sigma": 0.005}
    rows, cases = [], []
    if complete:
        with (args.results / "evaluation" / "per_image.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(r["id"], method) for r in run["protocol"]["items"] if r["split"] == "test" for method in ("input", "baseline", "l1", "l1_ssim")}
        if len(rows) != 80 or {(r["image"], r["method"]) for r in rows} != expected:
            raise RuntimeError("Niepelna lub niespojna tabela wynikow.")
        for variant in ("l1", "l1_ssim"):
            if file_hash(args.results / variant / "best.pt") != summary["checkpoints"][variant]["sha256"]:
                raise RuntimeError("Wagi zmienily sie od ewaluacji. Powtorz ocene.")
        cases_path = args.results / "figures" / "cases.json"
        if not cases_path.exists():
            raise RuntimeError("Brak ilustracji. Ponow main.py evaluate z opcjami treningu.")
        cases = load(cases_path)

    document.cover(complete)
    if complete:
        p("<b>Status: zakończone eksperymenty.</b> Tabele i ilustracje pochodzą z zapisanej ewaluacji 20 zdjęć testowych. Główną metodą sieciową jest z góry ustalony wariant L1 + SSIM; wariant L1 stanowi eksperyment porównawczy.")
        if ((args.experiments or args.results/"experiments")/"summary.json").exists():
            p("Dokument obejmuje również rozwinięcie projektu: 12 kontrolowanych konfiguracji U-Neta oraz potwierdzenie finalistów z trzema ziarnami treningu. Pierwsza część zachowuje wyniki pierwotnego projektu; rozdziały 7-9 przedstawiają nowe próby, wybór na walidacji i końcowe porównanie testowe.")
        if ((args.architectures or args.results/"architectures")/"summary.json").exists():
            p("Kolejna część porównuje U-Net, ResU-Net, mały NAFNet i zmniejszony MIRNet przy wspólnym wyjściu RGB, stracie i budżecie treningu. Dla każdej architektury wykonano ocenę z trzema ziarnami. Wyniki tego porównania znajdują się w rozdziałach 10-13.")
    else:
        table([["STATUS OPRACOWANIA"], ["Wersja metodologiczna przed uruchomieniem obliczeń. Brak wyników treningu i testu. Dokument nie stanowi końcowego sprawozdania z eksperymentów."]], [width])
        p("Po zakończeniu <b>python main.py all</b> polecenie <b>python report.py</b> zastąpi ten plik pełnym sprawozdaniem, zawierającym rzeczywiste metryki, wykresy, rekonstrukcje i analizę. Nie wstawiono przykładowych ani szacowanych wartości metryk.")
    document.start_body()
    heading("1. Cel i wymagania")
    p("Celem jest odtworzenie prawidłowo naświetlonego obrazu na podstawie wejścia o obniżonej ekspozycji i z fizycznie motywowanym szumem. Projekt obejmuje przygotowanie danych RAW, trening własnej sieci w PyTorch i porównanie z globalnym wyrównaniem histogramu jasności.")
    table([["Element", "Realizacja"],
           ["Dane", "Pierwsze 100 pozycji Indoor / Nikon D7000 w kolejności CSV RAISE_258.csv."],
           ["Podział", "Pozycje 81-100: test. Z pozycji 1-80: 72 treningowe i 8 walidacyjnych."],
           ["Degradacja", "Ekspozycja -3 EV, szum Poissona i Gaussa w przestrzeni liniowej, następnie gamma."],
           ["Porównanie", "Baseline histogramowy, U-Net L1 oraz U-Net L1 + SSIM."],
           ["Ocena", "PSNR, SSIM i LPIPS na wszystkich 20 obrazach testowych."]], [100, width-100])
    p("Źródło obrazów: RAISE, Uniwersytet w Trydencie [1].", "SmallText")

    document.page_break()
    heading("2. Dane i model degradacji")
    p("Odczyt CSV zachowuje kolejność wierszy. Filtrowanie odbywa się po aparacie Nikon D7000 i obecności kategorii Indoor. Dostarczony CSV zawiera 633 pasujące pozycje; wykorzystywanych jest wyłącznie pierwszych 100. Obrazy nie są sortowane po nazwach ani losowane przed wydzieleniem ostatnich 20 pozycji.")
    p("Pliki NEF są przetwarzane przez rawpy/LibRaw [2]. Obraz referencyjny generuje dokładnie wywołanie z instrukcji: <b>raw.postprocess(use_camera_wb=True, half_size=True)</b>. Rozdzielczość jest zmniejszona dwukrotnie w obu wymiarach przez przetwarzanie RAW, bez późniejszego skalowania podczas testu.")
    p("Wejście powstaje z osobnego wywołania z gamma=(1, 1), no_auto_bright=True i output_bps=16. Wynik jest dzielony przez 65535. Dla obrazu liniowego L wykonywane są kolejno:")
    table([["Etap", "Równanie / parametr"],
           ["Ekspozycja", r"<math>D = L\,2^{EV}</math>; " + f"<math>EV = {config['ev']:g}</math>, dla -3 EV: <math>D = L/8</math>."],
           ["Szum fotonowy", r"<math>P = \alpha\,\operatorname{Poisson}(D/\alpha)</math>, " + f"<math>\u03b1 = {metadata['alpha']:g}</math>."],
           ["Szum odczytu", r"<math>N = \operatorname{clip}(P + \mathcal{N}(0,\sigma^2),0,1)</math>, " + f"<math>\u03c3 = {metadata['sigma']:g}</math>."],
           ["Gamma", r"<math>S = N^{1/\gamma}</math>, " + f"<math>\u03b3 = {config['gamma']:g}</math>."],
           ["Zapis", r"<math>\operatorname{round}(255\,S)</math>, RGB uint8, bezstratny PNG."]], [105, width-105])
    p("Przed obcięciem wartości model ma średnią D i wariancję αD + σ². Oddaje zależność szumu fotonowego od ilości światła oraz stały składnik odczytu. Jest to uproszczenie z polecenia, stosowane do zdemozajkowanego liniowego RGB: nie jest kalibracją konkretnej matrycy ani symulacją pełnego toru Bayera.")
    p("Obraz docelowy zachowuje domyślne automatyczne rozjaśnianie i krzywą tonalną rawpy, natomiast wejście korzysta z uproszczonej potęgi 1/2,2. W konsekwencji sieć uczy się również dopasowania jasności i tonów do tak zdefiniowanej referencji. Nie utożsamiamy tej operacji z samym odszumianiem ani z dokładną funkcją transferu sRGB.")
    p("Każde zdjęcie otrzymuje generator szumu zależny od ziarna i pozycji w CSV. Te same pary są używane w obu wariantach. Szum walidacyjny i testowy pozostaje stały. Zbiór jest syntetyczny; wyniki nie dowodzą skuteczności na rzeczywistych nocnych zdjęciach z innych aparatów.")

    document.page_break()
    heading("3. Sieć i trening")
    p(f"Zastosowano residualny U-Net z trzema redukcjami rozdzielczości. Liczba kanałów na poziomach enkodera wynosi {config['width']}, {config['width']*2}, {config['width']*4} i {config['width']*8}. Każdy blok zawiera dwie konwolucje 3 × 3 oraz LeakyReLU(0,1). Redukcję realizuje average pooling, a dekoder interpolacja biliniowa i konkatenacja cech z odpowiadającego poziomu enkodera. Nie użyto wag wstępnie wytrenowanego modelu rekonstrukcji.")
    p(r"Sieć przewiduje poprawkę w przestrzeni logitów do obrazu rozjaśnionego współczynnikiem <math>2^{-EV/\gamma}</math>. Obraz bazowy jest ograniczany do [0,01; 0,99], a wynik funkcją sigmoid do [0; 1]. Końcowa konwolucja 1 × 1 ma początkowo zerowe wagi, więc startuje od prostego rozjaśnienia. Rozmiary niepodzielne przez osiem są dopełniane i przycinane po predykcji.")
    table([["Parametr", "Wartość"],
           ["Fragment treningowy / batch", f"{config['patch']} × {config['patch']} / {config['batch']}"],
           ["Limit treningu", f"{config['epochs']} epok, {config['steps']} batchy na epokę"],
           ["Optymalizacja", f"Adam, początkowy lr = {config['lr']:g}; cosine annealing do 5% lr"],
           ["Regularność obliczeń", "Gradient clipping do normy 1; mixed precision na CUDA, jeśli włączone"],
           ["Augmentacja", "Wspólne dla pary losowe wycinki, obroty o wielokrotność 90° i odbicia"],
           ["Wybór checkpointu", "Największy średni PSNR 32 stałych fragmentów walidacji (4 na zdjęcie)"],
           ["Wczesne zatrzymanie", f"{config['patience']} epok bez poprawy PSNR walidacji"],
           ["Ziarna", f"Trening/szum: {config['seed']}; losowanie podziału walidacji: 2026"]], [145, width-145])
    p("Eksperyment A minimalizuje średni błąd bezwzględny, a eksperyment B dodaje składnik strukturalny:")
    story.append(r"\begin{align}\mathcal{L}_A &= \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i-y_i|,\\"
                 r"\mathcal{L}_B &= \mathcal{L}_A+\lambda\bigl(1-\operatorname{SSIM}(\hat y,y)\bigr).\end{align}")
    p(f"Przyjęto λ = {config['ssim_weight']:g}. SSIM w stracie jest obliczany przez pytorch-msssim w float32. Wykresy wartości strat obu wariantów mają różne funkcje celu i nie służą bezpośredniemu porównaniu jakości.")
    p("Oba treningi rozpoczynają się od tej samej inicjalizacji, korzystają z tego samego podziału oraz odpowiadających sobie fragmentów i augmentacji. Test nie wpływa na wybór checkpointu ani moment zatrzymania. Wariant B jest z góry główną metodą raportu, nawet jeżeli wariant A uzyska lepszy wynik testowy.")
    if complete:
        p("Środowisko wykonania: " + escape(run["gpu"]) + "; PyTorch " + escape(config["torch"]) + ".", "SmallText")
        table([["Wariant", "Zakończone epoki", "Wybrana epoka", "Czas epok [min]"]] + [
            [v, len(load(args.results/v/"history.json")), summary["checkpoints"][v]["epoch"],
             f"{sum(r['seconds'] for r in load(args.results/v/'history.json'))/60:.1f}"] for v in ("l1", "l1_ssim")], [100, 130, 120, width-350])

    document.page_break()
    heading("4. Protokół oceny i uruchomienie")
    p("Baseline wykonuje globalne wyrównanie histogramu składowej Y w przestrzeni YCbCr z zachowaniem składowych chrominancji, następnie wraca do RGB. Wykorzystuje ImageOps.equalize z Pillow. Jest to prosta, deterministyczna metoda bez treningu i bez doboru parametrów na teście.")
    tile = load(args.results / "evaluation" / "protocol.json")["tile"] if complete else 512
    p(f"Każdy model rekonstruuje całe obrazy testowe kafelkami o rdzeniu {tile} × {tile} i otoczeniu 64 pikseli. Do wyniku trafia centralny rdzeń; kontekst ogranicza artefakty granic. Obraz nie jest przeskalowywany. Wszystkie metody są oceniane po identycznej kwantyzacji do RGB uint8, w zakresie [0; 1].")
    story.append(r"\begin{equation}\operatorname{PSNR}=10\log_{10}\!\left(\frac{1}{\operatorname{MSE}}\right)\;[\mathrm{dB}].\end{equation}")
    p("MSE obejmuje wszystkie piksele i kanały RGB w zakresie [0; 1]. Wyższy PSNR jest lepszy. SSIM jest średnią dla RGB, z gaussowskim oknem (σ=1,5) i populacyjną kowariancją, zgodnie z ustawieniami skimage.metrics.structural_similarity [3]. Wyższy SSIM jest lepszy. LPIPS-AlexNet v0.1 używa oficjalnych wag i całego obrazu z wejściem RGB przeskalowanym do [-1; 1] [4]. Niższy LPIPS jest lepszy.")
    p("Raport podaje średnią arytmetyczną metryk liczonych osobno dla 20 zdjęć. Odchylenie standardowe próby opisuje zmienność między obrazami, nie przedział ufności. Walidacyjny SSIM z pytorch-msssim służy do monitorowania; końcowy SSIM jest liczony identycznie dla wszystkich metod w skimage.")
    p("Uruchomienie w Windows", "Heading2")
    p("Otwórz PowerShell w folderze projektu. Skrypt startowy przygotuje lokalne .venv i przypięte zależności. PyTorch korzysta z wydania CUDA 12.8. W razie blokady skryptów można użyć poniższego polecenia, które zmienia zasady wykonania tylko dla uruchamianego procesu:")
    p("powershell -ExecutionPolicy Bypass -File .\\start.ps1 all", "CodeText")
    p("Po obliczeniach wygeneruj dokument z wynikami, opcjonalnie podając autorów:")
    p("Skład PDF wymaga XeLaTeX z TeX Live: lokalnie w Linux albo w dystrybucji Ubuntu WSL na Windows. Generator tworzy źródło LaTeX, kompiluje dokument trzykrotnie dla spisu treści i numeracji, a następnie usuwa pliki pomocnicze. Autorzy projektu są ustawieni domyślnie; opcja --authors pozwala ich zmienić.", "SmallText")
    p('powershell -ExecutionPolicy Bypass -File .\\start.ps1 report<br/>lub: .\\.venv\\Scripts\\python.exe report.py --authors &quot;Imię Nazwisko, Imię Nazwisko&quot;', "CodeText")
    p("Jeżeli środowisko jest już aktywne: python main.py all, następnie python report.py. Etapy można uruchamiać oddzielnie: check, download, prepare, train, evaluate. Polecenie check --with-lpips sprawdza również pobieranie wag metryki. Opcje są dostępne pod --help.")
    p("Ponowienie identycznego polecenia pomija gotowe pliki i zakończone treningi. Pobieranie kontynuuje plik .part, jeśli serwer obsługuje Range. Trening wraca do ostatniej zakończonej epoki; kroki przerwanej epoki są powtarzane. Ewaluacja zapisuje postęp po każdym zdjęciu. Zmiana konfiguracji wymaga osobnego --results, a zmiana degradacji także osobnego --data, co chroni przed mieszaniem wyników.")
    p("Repozytorium zawiera kod, requirements.txt, CSV, skrypt startowy, .gitignore i ten dokument. Foldery data/, results/ oraz .venv/ pozostają lokalne. Zachowują dane, wagi i wyniki potrzebne do wznowienia pracy oraz odtworzenia raportu. Nie są dodatkową dokumentacją.")

    document.page_break()
    heading("5. Wyniki i analiza")
    if not complete:
        p("<b>Eksperymenty nie zostały jeszcze wykonane.</b> Nie można obecnie ocenić jakości rekonstrukcji ani stwierdzić przewagi sieci nad wyrównaniem histogramu. Tabela zostanie wypełniona wyłącznie na podstawie kompletnej ewaluacji ostatnich 20 obrazów.")
        table([["Metoda", "PSNR ↑", "SSIM ↑", "LPIPS ↓"], ["baseline", "brak pomiaru", "brak pomiaru", "brak pomiaru"],
               ["network: L1 + SSIM", "brak pomiaru", "brak pomiaru", "brak pomiaru"]], [170, 110, 110, width-390])
        p("Pełna wersja zawiera również eksperyment L1, wykresy uczenia, wyniki każdego zdjęcia oraz porównania wizualne wejścia, histogramu, sieci i referencji. Przykłady wybierane są według jawnych kryteriów: pierwszy i środkowy obraz testu, największa i najmniejsza przewaga PSNR nad baseline, najciemniejsza referencja oraz referencja z największym udziałem nasyconych pikseli.")
        p("Plan analizy przypadków szczególnych", "Heading2")
        p("W cieniach należy sprawdzić, czy rozjaśnienie nie wzmacnia ziarnistości i kolorowych plam, a wygładzanie nie usuwa faktur. Dla jasnych obszarów trzeba porównać utratę szczegółów oraz udział pikseli bliskich nasyceniu. W kwestii koloru należy zestawić neutralne powierzchnie, kontrast i błędy kanałów RGB. Automatyczna część raportu poda liczby, a ilustracje pozwolą zweryfikować ich znaczenie wizualne.")
    else:
        methods = summary["methods"]
        p("Tabela 1. Główne porównanie wymagane w poleceniu. Średnia ± odchylenie standardowe dla 20 zdjęć.", "SmallText")
        def metric_row(label, key):
            return [label] + [f"{methods[key][m]['mean']:.3f} ± {methods[key][m]['std']:.3f}" for m in ("psnr", "ssim", "lpips")]
        table([["Metoda", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"], metric_row("baseline", "baseline"), metric_row("network: L1 + SSIM", "l1_ssim")], [155, 120, 115, width-390])
        p("Tabela 2. Dodatkowe punkty odniesienia i eksperyment z funkcją straty.", "SmallText")
        table([["Metoda", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"], metric_row("Wejście bez rekonstrukcji", "input"), metric_row("U-Net L1", "l1")], [155, 120, 115, width-390])
        indexed = {(r["image"], r["method"]): r for r in rows}
        names = [r["id"] for r in run["protocol"]["items"] if r["split"] == "test"]
        differences = {m: methods["l1_ssim"][m]["mean"] - methods["baseline"][m]["mean"] for m in ("psnr", "ssim", "lpips")}
        wins = {m: sum((float(indexed[n,"l1_ssim"][m]) < float(indexed[n,"baseline"][m])) if m == "lpips" else (float(indexed[n,"l1_ssim"][m]) > float(indexed[n,"baseline"][m])) for n in names) for m in ("psnr", "ssim", "lpips")}
        p(f"Różnica średnich network - baseline wynosi {differences['psnr']:+.3f} dB dla PSNR, {differences['ssim']:+.4f} dla SSIM oraz {differences['lpips']:+.4f} dla LPIPS. Dodatnie różnice PSNR i SSIM oraz ujemna różnica LPIPS oznaczają poprawę. Sieć uzyskała lepszy wynik na {wins['psnr']}/20 zdjęć według PSNR, {wins['ssim']}/20 według SSIM i {wins['lpips']}/20 według LPIPS.")
        directions = [differences['psnr'] > 0, differences['ssim'] > 0, differences['lpips'] < 0]
        if all(directions):
            p("Sieć przewyższa baseline według wszystkich trzech średnich metryk na tym zbiorze testowym. Wniosek dotyczy dokładnie przyjętej degradacji i podziału danych; nie dowodzi przewagi w dowolnej rzeczywistej scenie nocnej.")
        elif any(directions):
            p("Wyniki nie wskazują jednolitej przewagi we wszystkich metrykach. Trzeba rozpatrywać osobno zgodność pikselową, strukturę i odległość percepcyjną, a kompromisy sprawdzić na zestawieniach obrazów.")
        else:
            p("Nie uzyskano przewagi sieci nad baseline według średnich metryk. Wynik należy traktować jako ograniczenie obecnej konfiguracji; poprawy wymagać mogą trening, funkcja straty lub dopasowanie toru tonalnego do referencji.")
        delta_l1 = methods['l1_ssim']['psnr']['mean'] - methods['l1']['psnr']['mean']
        p(f"Dodanie składnika SSIM zmieniło średni PSNR względem treningu L1 o {delta_l1:+.3f} dB. Porównanie pozostałych metryk w tabelach pokazuje, czy zmiana funkcji celu poprawia strukturę i percepcję kosztem zgodności pikselowej.")
        clipped_gt = sum(d['target_clipped_fraction'] for d in summary['details']) / 20
        clipped_net = sum(d['network_clipped_fraction'] for d in summary['details']) / 20
        channels = [sum(d['channel_mae'][i] for d in summary['details']) / 20 for i in range(3)]
        p(f"Średni udział pikseli z co najmniej jednym kanałem ≥254 wynosi {100*clipped_gt:.2f}% w referencji i {100*clipped_net:.2f}% w rekonstrukcji. Jest to wskaźnik bliskości nasycenia, a nie bezpośrednia miara odzyskanych szczegółów. Średnie błędy bezwzględne kanałów R/G/B wynoszą {channels[0]:.2f}/{channels[1]:.2f}/{channels[2]:.2f} w skali 0-255. Same wartości MAE nie określają kierunku zafarbu.")
        document.page_break()
        heading("6. Przebieg uczenia i rozrzut wyników")
        figure(args.results/"figures"/"learning.png", "Rys. 1. Straty treningowe i metryki na stałych fragmentach walidacyjnych.", 210)
        figure(args.results/"figures"/"test_metrics.png", "Rys. 2. Wyniki poszczególnych zdjęć testowych; oś pozioma zachowuje numery w CSV.", 210)
        p("Checkpointy wybrano wyłącznie na walidacji. Wykres testowy przedstawia zmienność jakości pomiędzy scenami, nie służy do ponownego wyboru hiperparametrów. Czas w tabeli treningu jest sumą zapisanych czasów ukończonych epok i nie obejmuje pobierania danych ani przerw w pracy.")
        for number, case in enumerate(cases, 3):
            document.page_break()
            heading("Porównanie wizualne / " + escape(case['image']))
            p(escape(case['reason']) + ".", "SmallText")
            figure(args.results/"figures"/case['figure'], f"Rys. {number}. Całe obrazy i odpowiadający im detal. Żółta ramka wskazuje powiększony fragment.", 360)
            name = case['image']
            table([["Metoda", "PSNR [dB]", "SSIM", "LPIPS"]] + [
                [label] + [f"{float(indexed[name, method][m]):.4f}" for m in ("psnr", "ssim", "lpips")]
                for label, method in (("Histogram", "baseline"), ("U-Net L1 + SSIM", "l1_ssim"))], [155, 120, 115, width-390])
            p("Wszystkie kolumny mają ten sam obszar kadru i wspólną skalę RGB 0-255, bez automatycznego rozciągania kontrastu. Fragment wybrano deterministycznie według największej wariancji jasności referencji w siatce 4 × 4, bez używania predykcji do wyboru detalu. Należy porównać zachowanie faktur, kolorowe plamy w cieniach, neutralność barw oraz szczegóły jasnych powierzchni.", "SmallText")

        document.page_break()
        heading("Wyniki poszczególnych zdjęć")
        p("W każdej komórce podano baseline / network L1 + SSIM. Wartości w tabelach głównych pochodzą z tych samych 20 obrazów.", "SmallText")
        detail_rows = [["Obraz", "PSNR [dB]", "SSIM", "LPIPS"]]
        for name in names:
            detail_rows.append([name] + [f"{float(indexed[name,'baseline'][m]):.3f} / {float(indexed[name,'l1_ssim'][m]):.3f}" for m in ("psnr", "ssim", "lpips")])
        table(detail_rows, [125, 130, 125, width-380], small=True)

    study_folder = args.experiments or args.results / "experiments"
    study_path = study_folder / "summary.json"
    if complete and study_path.exists():
        study = load(study_path)
        study_protocol = load(study_folder / "protocol.json")
        selection = load(study_folder / "selection.json")
        if not study.get("complete") or study["signature"] != study_protocol["signature"] or study["signature"] != selection["signature"]:
            raise RuntimeError("Niepelna lub niespojna seria eksperymentow U-Neta.")
        if study_protocol["data_signature"] != run["protocol"]["signature"]:
            raise RuntimeError("Eksperymenty korzystaja z innego protokolu danych.")
        base = next(r for r in study["screening"] if r["id"] == "reference")
        document.page_break()
        heading("7. Eksperymenty kontrolowane U-Neta")
        p("Po wykonaniu podstawowego projektu przeprowadzono serię 12 konfiguracji U-Neta. Każda nowa konfiguracja zmienia dokładnie jeden czynnik względem modelu L1 + 0,2 · (1 - SSIM): sposób wyjścia, harmonogram learning rate, funkcję straty, wagę SSIM, aktywację, połączenia resztkowe albo uwagę kanałową. Nie łączono zwycięskich zmian po obejrzeniu testu.")
        p("Pozostawiono te same 72 obrazy treningowe, 8 walidacyjnych i 20 testowych, tę samą degradację -3 EV, rozmiar fragmentu 256 × 256, batch 8 oraz limit 80 epok po 80 kroków. Checkpoint każdej próby wybierano na podstawie PSNR 32 stałych fragmentów walidacyjnych, z zatrzymaniem po 15 epokach bez poprawy. Następnie oceniano wybrany checkpoint na wszystkich 8 pełnych obrazach walidacyjnych.")
        p("Pierwszy przegląd konfiguracji wykonano dla ziarna treningu 42. Model odniesienia wykorzystuje zapisany wcześniej trening: test zgodności potwierdził identyczną strukturę wag i wynik predykcji. Pozostałe próby zaczynały od tej samej inicjalizacji wspólnych konwolucji i tych samych wycinków. Dodatkowe parametry bloków resztkowych lub uwagi inicjalizowano od nowa. Wszystkie liczby w poniższej tabeli dotyczą walidacji, a nie zbioru testowego.")
        study_rows = [["Konfiguracja", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓", "ΔPSNR"]]
        for r in study["screening"]:
            study_rows.append([r["label"], f"{r['psnr']:.3f}", f"{r['ssim']:.4f}", f"{r['lpips']:.4f}", f"{r['psnr']-base['psnr']:+.3f}"])
        table(study_rows, [189, 85, 80, 80, width-434], small=True)
        p("ΔPSNR oznacza różnicę względem modelu odniesienia na tych samych 8 pełnych obrazach walidacyjnych. Strzałki wskazują pożądany kierunek zmiany. Jest to wstępny przegląd z jednym ziarnem, więc nie służy samodzielnie do wnioskowania o stabilności przewagi.", "SmallText")

        document.page_break()
        heading("Interpretacja przeglądu konfiguracji")
        groups = [
            ("Sposób tworzenia wyjścia", ("output_direct", "output_rgb_residual")),
            ("Sterowanie learning rate", ("lr_plateau",)),
            ("Funkcja straty", ("loss_mse", "loss_charbonnier")),
            ("Waga składnika SSIM", ("ssim_005", "ssim_05")),
            ("Funkcja aktywacji", ("activation_prelu", "activation_silu")),
            ("Bloki resztkowe i uwaga", ("residual_blocks", "channel_attention")),
        ]
        screened = {r["id"]: r for r in study["screening"]}
        for title, keys in groups:
            p(title, "Heading2")
            for key in keys:
                r = screened[key]
                p(f"{escape(r['label'])}: zmiana względem odniesienia wynosi {r['psnr']-base['psnr']:+.3f} dB PSNR, "
                  f"{r['ssim']-base['ssim']:+.4f} SSIM i {r['lpips']-base['lpips']:+.4f} LPIPS.")
        p("Są to osobne próby, a nie kolejne ulepszenia jednej stale modyfikowanej sieci. Dodatnie różnice PSNR/SSIM i ujemna LPIPS wskazują poprawę danej miary. Różne miary mogą wskazywać inne kompromisy. Tylko dwa najlepsze nowe warianty według z góry ustalonego PSNR przechodzą do potwierdzenia z dodatkowymi ziarnami.")
        p("Kryterium wyboru epoki jest jednakowe dla wszystkich strat: PSNR. Większa waga SSIM może prowadzić do wyższego SSIM w późniejszej epoce, która nie zostanie wybrana, jeśli jej PSNR jest niższy. Wyniki wag SSIM dotyczą zatem modeli wybranych według PSNR i nie rozstrzygają, jaki wynik dałby wybór checkpointu bezpośrednio według SSIM.", "SmallText")

        document.page_break()
        heading("Co dokładnie zmieniono")
        table([
            ["Czynnik", "Definicja eksperymentu"],
            ["Wyjście bezpośrednie", "Sigmoid na wyniku końcowej konwolucji przewiduje czyste RGB, bez dodawania obrazu wejściowego."],
            ["Poprawka w RGB", "Wynik = clip(rozjaśnione wejście + poprawka, 0, 1). Rozjaśniona baza jest taka sama jak w oryginale; zastąpiono dodawanie w logitach dodawaniem w RGB."],
            ["Harmonogram LR", "ReduceLROnPlateau: tryb max dla PSNR walidacji, mnożnik 0,5, patience=4, minimum 0,05 początkowego lr. Zatrzymanie nadal po 15 epokach bez poprawy."],
            ["Funkcja straty", "Składnik L1 zastąpiono MSE albo Charbonnierem mean(sqrt((ŷ-y)² + 10^-6)). Składnik SSIM zachowuje wagę 0,2. Zmiana rodzaju błędu zmienia także względną skalę obu składników."],
            ["Waga SSIM", "Zmieniono wyłącznie λ z 0,2 na 0,05 lub 0,5 w stracie L1 + λ(1-SSIM)."],
            ["Aktywacje", "Wewnątrz bloków LeakyReLU(0,1) zastąpiono PReLU z osobnym uczonym nachyleniem kanałów (start 0,1) albo SiLU. Końcowe wyjście pozostało oryginalne [7, 8]."],
            ["Bloki resztkowe", "Do wyjścia każdego bloku dodano jego wejście. Przy różnej liczbie kanałów ścieżka skrótu używa konwolucji 1 × 1. Zmieniane są mapy cech, a nie ogólna ścieżka wyjściowa."],
            ["Uwaga kanałowa", "Po każdym bloku dodano SE: global average pooling, redukcja kanałów (minimum 4, współczynnik 8), ReLU, przywrócenie kanałów i sigmoid. Cechy mnożone są przez uzyskane wagi [9]."],
        ], [113, width-113], small=True)
        p("Zmiany architektury mogą zwiększać liczbę parametrów. Nie wyrównywano jej dodatkową zmianą szerokości sieci, ponieważ wprowadziłoby to kolejny czynnik. Koszt i rozmiar modeli podano osobno.")
        p("Globalne uśrednianie w SE dotyczy aktualnego fragmentu podczas treningu i kafelka podczas rekonstrukcji. Nie jest to jedna wspólna waga wyznaczona z całego zdjęcia. Zależność uwagi od rozmiaru i zawartości kafelka stanowi ograniczenie tego wariantu przy zastosowanym sposobie inferencji.", "SmallText")

        document.page_break()
        heading("Koszt i przebieg eksperymentów")
        figure(study_folder/"figures"/"screening.png", "Przegląd konfiguracji według PSNR pełnych obrazów walidacyjnych; pojedyncze ziarno 42.", 260)
        table([["Konfiguracja", "Parametry", "Epoki", "Najlepsza epoka", "Czas [min]"]] + [
            [r["label"], f"{r['parameters']:,}", r["epochs"], r["best_epoch"], f"{r['training_minutes']:.2f}"]
            for r in study["screening"]], [183, 90, 55, 95, width-423], small=True)
        p("Czas jest sumą ukończonych epok, bez odczytu danych, pełnej ewaluacji i przerw. Czas modelu odniesienia pochodzi z pierwotnego treningu, który został wykorzystany ponownie. Różne liczby epok wynikają z tego samego kryterium wcześniejszego zatrzymania; porównanie jest ograniczone jednakowym maksymalnym budżetem, nie identycznym zużytym czasem.", "SmallText")

        study_learning_figures(study_folder, study["screening"])
        document.page_break()
        heading("Przebieg uczenia nowych wariantów")
        figure(study_folder/"figures"/"learning_groups.png", "Każdy panel porównuje model odniesienia z wariantami zmieniającymi wskazany czynnik. Ziarno 42.", 355)
        figure(study_folder/"figures"/"learning_rates.png", "Learning rate po zakończeniu danej epoki. Długość krzywej zależy od wcześniejszego zatrzymania.", 130)
        p("Na wykresach pokazano PSNR 32 stałych fragmentów walidacyjnych używanych do wyboru checkpointu. W tabelach eksperymentów podano natomiast ocenę pełnych ośmiu zdjęć. Krzywe tych dwóch procedur nie powinny być utożsamiane. Straty różnych rodzin mają różne skale, dlatego nie porównywano ich surowych wartości jako miary jakości.", "SmallText")

        document.page_break()
        heading("8. Powtórzenia i wybór konfiguracji")
        p("Dwa najlepsze nowe warianty według PSNR pełnej walidacji oraz model odniesienia sprawdzono z trzema ziarnami treningu: 42, 43 i 44. Zmieniano inicjalizację i treningowe wycinki; podział danych, obrazy wejściowe, realizacja szumu i walidacja pozostawały stałe. Odchylenia w tej części opisują zmienność pomiędzy trzema treningami, a nie rozrzut 8 lub 20 zdjęć.")
        def study_metric_row(r):
            return [r["label"]] + [f"{r['metrics'][m]['mean']:.3f} ± {r['metrics'][m]['std']:.3f}" for m in ("psnr", "ssim", "lpips")]
        table([["Pełna walidacja", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"]] + [study_metric_row(r) for r in study["confirmation"]], [170, 120, 110, width-400])
        chosen = next(r for r in study["confirmation"] if r["id"] == study["winner"])
        reference_validation = next(r for r in study["confirmation"] if r["id"] == "reference")
        validation_gain = chosen["metrics"]["psnr"]["mean"] - reference_validation["metrics"]["psnr"]["mean"]
        p(f"Wybrano: <b>{escape(chosen['label'])}</b>. Kryterium stanowił największy średni PSNR pełnej walidacji z trzech treningów. Różnica względem odniesienia wynosi {validation_gain:+.3f} dB. Model odniesienia również mógł wygrać; nie zakładano z góry, że nowa konfiguracja poprawi rezultat.")
        p("Wybór i lista finalistów zostały zapisane przed uruchomieniem nowej oceny testowej. Wszystkie trzy konfiguracje końcowe oceniono następnie na tych samych 20 zdjęciach. Wynik testowy nie zmienia wyboru wariantu. Ponieważ wyniki podstawowego projektu były już wcześniej znane, jest to rozwinięcie eksperymentu na istniejącym zbiorze, a nie niezależna walidacja na nowo zebranych scenach.")
        p("Uruchomienie i odtwarzanie", "Heading2")
        p(".\\.venv\\Scripts\\python.exe experiments.py<br/>.\\.venv\\Scripts\\python.exe report.py", "CodeText")
        p("Etap --stage train wykonuje przegląd i powtórzenia walidacyjne, a --stage evaluate uruchamia ocenę finalistów po zapisanym wyborze. Przerwane treningi wracają do ostatniej ukończonej epoki. Wyniki są w lokalnym, ignorowanym przez Git podfolderze results/experiments. Dotychczasowe modele i wyniki podstawowe zachowano.")

        document.page_break()
        heading("9. Końcowe wyniki eksperymentów")
        table([["Test: 20 zdjęć", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"]] + [study_metric_row(r) for r in study["test"]], [170, 120, 110, width-400])
        reference_test = next(r for r in study["test"] if r["id"] == "reference")
        chosen_test = next(r for r in study["test"] if r["id"] == study["winner"])
        delta = {m: chosen_test["metrics"][m]["mean"] - reference_test["metrics"][m]["mean"] for m in ("psnr", "ssim", "lpips")}
        p(f"Dla wariantu wybranego na walidacji różnica średnich testowych względem oryginalnego U-Neta wynosi: PSNR {delta['psnr']:+.3f} dB, SSIM {delta['ssim']:+.4f}, LPIPS {delta['lpips']:+.4f}. Poprawę oznacza wzrost PSNR i SSIM oraz spadek LPIPS. Liczby są średnimi z trzech treningów; każdy wynik treningu jest średnią z 20 zdjęć.")
        image_scores = {}
        for key in dict.fromkeys(("reference", study["winner"])):
            all_seeds = [load(study_folder/"runs"/f"{key}_s{seed}"/"test"/"summary.json")["rows"] for seed in study["seeds"]]
            image_scores[key] = {r["image"]: {m: sum(next(x[m] for x in rows if x["image"] == r["image"]) for rows in all_seeds)/len(all_seeds)
                                               for m in ("psnr", "ssim", "lpips")} for r in all_seeds[0]}
        wins = {m: sum((image_scores[study["winner"]][n][m] < image_scores["reference"][n][m]) if m == "lpips"
                       else (image_scores[study["winner"]][n][m] > image_scores["reference"][n][m])
                       for n in image_scores["reference"]) for m in ("psnr", "ssim", "lpips")}
        p(f"Po uśrednieniu wyników trzech ziaren osobno dla każdego zdjęcia wybrany wariant poprawia PSNR na {wins['psnr']}/20 obrazów, SSIM na {wins['ssim']}/20 i LPIPS na {wins['lpips']}/20. Lepsze średnie nie oznaczają przewagi w każdej scenie.")
        if study["winner"] == "reference":
            p("Żadna z potwierdzonych nowych konfiguracji nie przewyższyła odniesienia według głównego kryterium walidacyjnego. Eksperyment nie uzasadnia zastąpienia dotychczasowego modelu.")
        elif delta["psnr"] > 0 and delta["ssim"] > 0 and delta["lpips"] < 0:
            p("Wariant wybrany na walidacji poprawił wszystkie trzy średnie metryki na teście. Wynik wskazuje korzyść z tej konkretnej zmiany dla przyjętej degradacji. Nie dowodzi uniwersalnej przewagi dla innych aparatów, poziomów oświetlenia lub rodzajów szumu.")
        else:
            p("Zmiana wybrana na walidacji nie poprawiła jednocześnie wszystkich średnich metryk testowych. Jej korzyści i ograniczenia należy oceniać oddzielnie dla zgodności pikselowej, struktury i podobieństwa percepcyjnego; sam wzrost PSNR nie przesądza o lepszym odtworzeniu faktur.")
        if study["winner"] == "output_direct":
            p("Wybrane wyjście bezpośrednie nie przekazuje zaszumionego obrazu gotową ścieżką do końcowego wyniku. Sieć musi odtworzyć całe RGB z wydobytych cech. Hipotezą wyjaśniającą ewentualną poprawę jest łatwiejsze odrzucenie szumu, zamiast uczenia jego dokładnego znoszenia poprawką w logitach. Eksperyment potwierdza wynik dla tej konfiguracji; nie rozstrzyga przewagi wszystkich sieci bez połączenia resztkowego nad wszystkimi sieciami resztkowymi.")
        qualitative = None
        if (study_folder/"qualitative.json").exists():
            qualitative = load(study_folder/"qualitative.json")
            if qualitative["signature"] != study["signature"]:
                raise RuntimeError("Opis wizualny dotyczy innej serii badan.")
        figure(study_folder/"figures"/"final_test.png", "Końcowe metryki; słupki błędu pokazują odchylenie między trzema ziarnami treningu.", 200)
        if qualitative:
            p(escape(qualitative["overview"]))
        for case in load(study_folder/"figures"/"comparisons.json"):
            document.page_break()
            heading("Wynik eksperymentu / " + escape(case["image"]))
            p("Porównanie dla ziarna 42. Przykłady obejmują pierwszy obraz testowy oraz obrazy z największą i najmniejszą różnicą PSNR; powtarzające się identyfikatory pominięto.", "SmallText")
            figure(study_folder/"figures"/case["figure"], "Wejście, U-Net odniesienia, wariant wybrany na walidacji i referencja. Dolny rząd pokazuje ten sam fragment.", 370)
            p(f"Różnica PSNR nowego wariantu względem odniesienia na tym zdjęciu: {case['psnr_gain']:+.3f} dB. Fragment wybrano według wariancji jasności referencji, bez używania predykcji do wyboru kadru. Obrazy mają wspólną skalę RGB 0-255.", "SmallText")
            if qualitative and case["image"] in qualitative["cases"]:
                p(escape(qualitative["cases"][case["image"]]))

    architecture_folder = args.architectures or args.results/"architectures"
    if complete and (architecture_folder/"summary.json").exists():
        architecture = load(architecture_folder/"summary.json")
        architecture_protocol = load(architecture_folder/"protocol.json")
        architecture_selection = load(architecture_folder/"selection.json")
        if not architecture.get("complete") or len(architecture["test"]) != 4 or len(architecture["validation"]) != 4:
            raise RuntimeError("Niepelne porownanie architektur.")
        if architecture["signature"] != architecture_protocol["signature"] or architecture["signature"] != architecture_selection["signature"]:
            raise RuntimeError("Niespojne pochodzenie porownania architektur.")
        if architecture_protocol["data_signature"] != run["protocol"]["signature"]:
            raise RuntimeError("Porownanie architektur korzysta z innych danych.")
        labels = {r["id"]: r["label"] for r in architecture["test"]}
        document.page_break()
        heading("10. Porównanie architektur")
        p("Po badaniu pojedynczych zmian w U-Necie porównano cztery konstrukcje: U-Net, ResU-Net, mały NAFNet i zmniejszony MIRNet. Punktem odniesienia jest U-Net z bezpośrednim RGB, czyli wariant wybrany w poprzednim etapie. Wspólny sposób tworzenia wyjścia ogranicza mieszanie wpływu architektury ze stwierdzonym wcześniej wpływem gotowej ścieżki zaszumionego obrazu.")
        p("Wszystkie modele zwracają bezpośrednie RGB przez sigmoid. Końcowa konwolucja jest inicjalizowana zerami, więc każdy model początkowo zwraca 0,5. Zachowano L1 + 0,2 · (1 - SSIM), Adam z lr=0,0002, harmonogram kosinusowy do 5% lr, fragment 256 × 256, batch 8 i maksymalnie 80 epok po 80 kroków. Wcześniejsze zatrzymanie następuje po 15 epokach bez poprawy PSNR tych samych 32 fragmentów walidacyjnych.")
        table([
            ["Architektura", "Szerokość", "Budowa zastosowanego wariantu"],
            ["U-Net RGB", "24", "Trzy redukcje rozdzielczości, konwolucje 3 × 3, LeakyReLU, interpolacja biliniowa i konkatenacja cech. Bezpośrednie wyjście RGB."],
            ["ResU-Net RGB", "24", "Ta sama konstrukcja z dodawaniem wejścia do wyjścia każdego bloku cech. Przy zmianie kanałów konwolucja 1 × 1 w skrócie."],
            ["NAFNet-small", "30", "Trzy poziomy enkodera: 1/1/2 bloki NAF, jeden blok środkowy, dekoder 1/1/1. Normalizacja kanałowa, konwolucje depthwise, SimpleGate, uproszczona uwaga kanałowa, PixelShuffle i uczone skale β/γ [10]."],
            ["MIRNet-small", "28", "Jedna grupa resztkowa, jeden MSRB o trzech strumieniach rozdzielczości i dwóch kolumnach. DAU z uwagą przestrzenną i kanałową, wymiana cech pomiędzy skalami, SKFF, antyaliasing i resztkowe zmiany rozdzielczości [11]."],
        ], [104, 56, width-160], small=True)
        p("Są to warianty projektowe o zbliżonej liczbie parametrów, a nie pełne modele benchmarkowe ani próba odtworzenia wyników z publikacji. NAFNet i MIRNet mają wspólne z U-Netem bezpośrednie wyjście zamiast oryginalnego dodawania obrazu wejściowego. Nie zastosowano gotowych wag sieci rekonstrukcji. Wagi trzech U-Netów oraz ich zgodne wyniki wykorzystano ponownie z poprzedniego badania; pozostałe dziewięć modeli wytrenowano od początku.")
        p("Implementacja NAFNet używa natywnej normalizacji PyTorch w float32 i nie korzysta z wariantu NAFNetLocal/TLC. W MIRNet zachowano współdzielenie jednostki DAU między dwiema kolumnami danej skali; PReLU ma osobną instancję w każdej jednostce. Ograniczono liczbę grup i kanałów, zachowując charakterystyczne mechanizmy wieloskalowe.", "SmallText")

        document.page_break()
        heading("11. Walidacja i koszt obliczeń")
        p("Każdą architekturę oceniono z ziarnami treningu 42, 43 i 44. Obrazy, szum, podział 72/8/20 i fragmenty walidacyjne pozostały stałe. Po wyborze checkpointu na fragmentach oceniano wszystkie 8 pełnych obrazów walidacyjnych. Wybór architektury oparto wyłącznie na średnim PSNR pełnej walidacji z trzech treningów.")
        def architecture_row(r):
            return [r["label"]] + [f"{r['metrics'][m]['mean']:.3f} ± {r['metrics'][m]['std']:.3f}" for m in ("psnr", "ssim", "lpips")]
        table([["Pełna walidacja", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"]] + [architecture_row(r) for r in architecture["validation"]], [155, 120, 115, width-390])
        picked = next(r for r in architecture["validation"] if r["id"] == architecture["winner"])
        p("Architektura wybrana przed testem: <b>" + escape(picked["label"]) + "</b>. Odchylenia w tabelach opisują zmienność między trzema treningami, a nie przedział ufności i nie rozrzut pojedynczych zdjęć.")
        ranked_architectures = sorted(architecture["validation"], key=lambda r: r["metrics"]["psnr"]["mean"], reverse=True)
        second_architecture = ranked_architectures[1]
        selection_margin = picked["metrics"]["psnr"]["mean"] - second_architecture["metrics"]["psnr"]["mean"]
        if selection_margin < max(picked["metrics"]["psnr"]["std"], second_architecture["metrics"]["psnr"]["std"]):
            p(f"Przewaga walidacyjna nad {escape(second_architecture['label'])} wynosi tylko {selection_margin:.3f} dB i jest mniejsza od obserwowanej zmienności treningów. Reguła wskazuje zwycięzcę tej serii, ale trzy ziarna nie uzasadniają wniosku o trwałej przewadze jednej architektury.", "SmallText")
        benchmarks = {r["id"]: r for r in architecture["benchmark"]}
        table([["Model", "Parametry", "Epoki 42/43/44", "Wybrane epoki", "Śr. trening [min]", "Inferencja [ms]"]] + [
            [r["label"], f"{r['parameters']:,}", " / ".join(map(str,r["epochs"])), " / ".join(map(str,r["best_epochs"])),
             f"{r['training_minutes_mean']:.2f}", f"{benchmarks[r['id']]['median_ms']:.2f}"] for r in architecture["validation"]],
            [99, 79, 85, 85, 80, width-428], small=True)
        p("Czas treningu jest średnią sum czasów epok trzech przebiegów. Dla U-Neta pochodzi z wcześniejszego badania. Inferencję zmierzono ponownie na tym samym GPU dla wszystkich modeli: jeden fragment walidacyjny 256 × 256, batch 1, float32, 5 rozgrzewek i 20 pomiarów; podano medianę. Nie jest to czas przetworzenia całego zdjęcia. Podobna liczba parametrów nie oznacza podobnego kosztu obliczeń.", "SmallText")
        p("Ograniczenie porównania", "Heading2")
        p("Dla każdej sieci zastosowano ten sam przepis treningowy, bez osobnego przeszukiwania learning rate i funkcji straty. Wniosek dotyczy zatem małych wariantów w określonym budżecie i na 72 zdjęciach. Nie stanowi ogólnego rankingu potencjału pełnych architektur. Inne szerokości, dłuższy trening lub dostrojenie optymalizacji mogłyby zmienić kolejność.")
        p("Wspólna inferencja używa rdzenia 512 × 512 z otoczeniem 64 pikseli, bez skalowania obrazu. Mechanizmy globalnego uśredniania w NAFNet/MIRNet działają na aktualnym kafelku, a nie na całym zdjęciu. Zmiana rozmiaru kontekstu względem treningu może wpływać na ich zachowanie.")

        document.page_break()
        heading("12. Uczenie czterech architektur")
        figure(architecture_folder/"figures"/"learning.png", "PSNR na stałych fragmentach walidacyjnych. Każdy panel pokazuje trzy niezależne treningi jednej architektury.", 360)
        p("Długość krzywych wynika z jednakowego limitu i jednakowego kryterium wcześniejszego zatrzymania. Do testu trafiają checkpointy o najwyższym PSNR fragmentów walidacyjnych, a nie automatycznie ostatnie epoki. Pełną walidację stosuje się potem do wyboru architektury. Wykresów fragmentów nie należy utożsamiać ze średnimi pełnych zdjęć z tabel.")
        p("Weryfikacja implementacji", "Heading2")
        p("Sprawdzono poprawne wymiary wyjścia także dla nieparzystych rozdzielczości, ograniczenie RGB do [0; 1], skończone gradienty i trening AMP z batchem 8. Normalizację NAF porównano ze wzorem, blok NAF z zerowymi β/γ sprawdzono jako identyczność, a filtr antyaliasingu na stałym obrazie. Krótki test integracyjny objął wszystkie cztery architektury, wznawianie checkpointów, wybór na walidacji, metryki oraz generowanie ilustracji. Dane tymczasowego testu nie zostały użyte w wynikach projektu.")
        p("Odtworzenie porównania", "Heading2")
        p(".\\.venv\\Scripts\\python.exe architecture_study.py<br/>.\\.venv\\Scripts\\python.exe report.py", "CodeText")
        p("Opcja --stage train wykonuje trening i walidację, a --stage evaluate ocenę testową po zapisanym wyborze. Gotowe przebiegi są pomijane; przerwane wracają do ostatniej ukończonej epoki. Kod sieci jest w architectures.py, a kod porównania w architecture_study.py. Wyniki pozostają lokalnie w results/architectures, ignorowanym przez Git.")

        document.page_break()
        heading("13. Test porównania architektur")
        table([["Test: 20 zdjęć", "PSNR [dB] ↑", "SSIM ↑", "LPIPS ↓"]] + [architecture_row(r) for r in architecture["test"]], [155, 120, 115, width-390])
        p("Każda wartość to średnia metryk 20 zdjęć, następnie uśredniona między trzema treningami. Odchylenie dotyczy trzech treningów. Wyboru architektury nie zmieniono po obejrzeniu testu; wszystkie cztery są pokazane, również gdy któraś wypada słabiej.")
        reference_arch = next(r for r in architecture["test"] if r["id"] == "unet")
        selected_arch = next(r for r in architecture["test"] if r["id"] == architecture["winner"])
        delta_arch = {m: selected_arch["metrics"][m]["mean"]-reference_arch["metrics"][m]["mean"] for m in ("psnr","ssim","lpips")}
        if architecture["winner"] == "unet":
            p("Najlepszy PSNR walidacyjny zachował U-Net RGB. W tym budżecie i przy wspólnym przepisie treningowym nowe konstrukcje nie uzasadniły jego zastąpienia według ustalonego kryterium. Poszczególne miary testowe mogą wskazywać inne kompromisy, co widać w tabeli i na obrazach.")
        else:
            p(f"Dla architektury wybranej na walidacji ({escape(selected_arch['label'])}) różnica testowa względem U-Neta RGB wynosi {delta_arch['psnr']:+.3f} dB PSNR, {delta_arch['ssim']:+.4f} SSIM i {delta_arch['lpips']:+.4f} LPIPS. Poprawie odpowiada dodatnia różnica PSNR/SSIM oraz ujemna LPIPS.")
        figure(architecture_folder/"figures"/"test_metrics.png", "Metryki testowe czterech architektur; słupki błędu oznaczają odchylenie między trzema ziarnami.", 205)
        architecture_notes = None
        if (architecture_folder/"qualitative.json").exists():
            architecture_notes = load(architecture_folder/"qualitative.json")
            if architecture_notes["signature"] != architecture["signature"]:
                raise RuntimeError("Opis obrazow dotyczy innego porownania architektur.")
            p(escape(architecture_notes["overview"]))
        architecture_cases = load(architecture_folder/"figures"/"comparisons.json")
        for case in architecture_cases:
            document.page_break(landscape=True)
            heading("Architektury / " + escape(case["image"]))
            p("Ziarno 42. Wspólny kadr i skala RGB 0-255; u góry pełne obrazy, poniżej powiększenie.", "SmallText")
            figure(architecture_folder/"figures"/case["figure"], "Pełne obrazy i wspólny detal wskazany żółtą ramką. Fragment wybrano według wariancji jasności referencji.", 330)
            table([["Model", "PSNR [dB]", "SSIM", "LPIPS"]] + [
                [labels[key]]+[f"{case['metrics'][key][m]:.4f}" for m in ("psnr","ssim","lpips")]
                for key in ("unet","resunet","nafnet","mirnet")], [155, 120, 115, width-390], small=True)
        if architecture_notes:
            document.page_break()
            heading("Omówienie przykładów architektur")
            p("Przykłady obejmują pierwszy obraz testowy, wcześniej rozpoznany trudny witraż oraz największą i najmniejszą różnicę PSNR wybranej architektury względem U-Neta. Powtarzające się identyfikatory pominięto. Wybór przykładów nie służył do dobierania modelu.", "SmallText")
            for case in architecture_cases:
                if case["image"] in architecture_notes["cases"]:
                    p(escape(case["image"]), "Heading2")
                    p(escape(architecture_notes["cases"][case["image"]]))

    document.page_break()
    heading("Ograniczenia i źródła")
    p("Zaledwie 72 zdjęcia treningowe ograniczają różnorodność scen. Kolejne obrazy w CSV mogą przedstawiać podobne ujęcia; podział na poziomie zdjęć zapobiega przeciekowi fragmentów tego samego pliku, ale nie gwarantuje niezależności scen. Zachowano wymagany w poleceniu test na ostatnich 20 z pierwszych 100 zdjęć.")
    p("Ziarna oraz ustawienia cuDNN ograniczają losowość. Identyczne wyniki bitowe nie są gwarantowane po zmianie wersji bibliotek, sterownika, sprzętu ani pomiędzy CPU i GPU. Manifest danych i konfiguracja treningu blokują przypadkowe mieszanie niezgodnych eksperymentów. Raport nie zawiera wyników pomiarów innych niż rzeczywiście zapisane przez program.")
    p("Model degradacji nie obejmuje m.in. nieliniowości elektroniki, szumu pasmowego, hot pixels, kompresji ani zmian balansu bieli pomiędzy ujęciami. Obcięcie sygnału i kwantyzacja powodują nieodwracalną utratę informacji. Wygładzona powierzchnia może poprawić błąd pikselowy i jednocześnie zatrzeć drobne faktury, dlatego trzy metryki trzeba interpretować łącznie z obrazami.")
    references = [
        ('[1] RAISE - Raw Images Dataset for Digital Image Forensics. Strona pobierania.', 'https://loki.disi.unitn.it/RAISE/download.html'),
        ('[2] rawpy. Dokumentacja parametrów przetwarzania RAW.', 'https://letmaik.github.io/rawpy/api/rawpy.Params.html'),
        ('[3] scikit-image. Metryki peak_signal_noise_ratio i structural_similarity.', 'https://scikit-image.org/docs/stable/api/skimage.metrics.html'),
        ('[4] R. Zhang i in. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018. Oficjalna implementacja LPIPS.', 'https://github.com/richzhang/PerceptualSimilarity'),
        ('[5] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015. Inspiracja architektury.', 'https://arxiv.org/abs/1505.04597'),
        ('[6] PyTorch. Oficjalne instrukcje instalacji.', 'https://pytorch.org/get-started/locally/'),
        ('[7] K. He i in. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, 2015. PReLU.', 'https://arxiv.org/abs/1502.01852'),
        ('[8] S. Elfwing i in. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning, 2017. SiLU.', 'https://arxiv.org/abs/1702.03118'),
        ('[9] J. Hu i in. Squeeze-and-Excitation Networks. Uwaga kanałowa.', 'https://arxiv.org/abs/1709.01507'),
        ('[10] L. Chen i in. Simple Baselines for Image Restoration, ECCV 2022. NAFNet.', 'https://arxiv.org/abs/2204.04676'),
        ('[11] S. W. Zamir i in. Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. MIRNet.', 'https://arxiv.org/abs/2003.06792'),
        ('[12] Oficjalna implementacja NAFNet. Wariant projektu adaptuje bloki i rozmiar oraz zmienia wyjście.', 'https://github.com/megvii-research/NAFNet'),
        ('[13] Oficjalna implementacja MIRNet. Wariant projektu zmniejsza głębokość i szerokość oraz zmienia wyjście.', 'https://github.com/swz30/MIRNet'),
    ]
    for label, url in references:
        p(escape(label) + '<br/><link href="' + url + '" color="#087f8c">' + escape(url) + '</link>', "SmallText")
    document.compile(args.output, args.wsl_distro, args.keep_latex)
    print("Zapisano:", args.output)
    if not complete:
        print("WERSJA METODOLOGICZNA: brak wynikow. Po main.py all ponow report.py bez --draft.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT/"results")
    p.add_argument("--output", type=Path, default=ROOT/"sprawozdanie.pdf")
    p.add_argument("--authors", default="Kacper Preyzner, Rafał Zwierzyński")
    p.add_argument("--wsl-distro", default="Ubuntu", help="Dystrybucja WSL z XeLaTeX (Windows)")
    p.add_argument("--keep-latex", type=Path, default=None, help="Opcjonalnie zachowaj źródło i pliki kompilacji w podanym folderze")
    p.add_argument("--experiments", type=Path, default=None, help="Opcjonalny folder z zakonczona seria eksperymentow")
    p.add_argument("--architectures", type=Path, default=None, help="Opcjonalny folder z porownaniem architektur")
    p.add_argument("--draft", action="store_true")
    args = p.parse_args()
    try:
        build(args)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        p.exit(1, f"BLAD: {exc}\n")


if __name__ == "__main__":
    main()
