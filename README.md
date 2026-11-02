# SIGK - Projekt 1: modyfikacja obrazów

Projekt realizuje **dwa zadania** z dostarczonego PDF w Pythonie i PyTorch:

| Wariant | Wejście | Wyjście | Metoda bazowa |
| --- | --- | --- | --- |
| `sr4` | RGB 64 x 64 | RGB 256 x 256 | OpenCV `INTER_CUBIC` |
| `sr8` | RGB 32 x 32 | RGB 256 x 256 | OpenCV `INTER_CUBIC` |
| `dn001` | RGB 256 x 256, szum Gaussa sigma=0.01 | RGB 256 x 256 | skimage `denoise_bilateral` |
| `dn003` | RGB 256 x 256, szum Gaussa sigma=0.03 | RGB 256 x 256 | skimage `denoise_bilateral` |

Kod obejmuje przygotowanie DIV2K, architekturę residualną, trening, wybór checkpointu,
dobór parametrów bilateral, ewaluację, przetwarzanie własnego obrazu oraz raport
HTML i Markdown z tabelami, wykresami i obrazami. **Kod i syntetyczna próba działania
nie zastępują eksperymentów na DIV2K**. Wyniki i ograniczenia opisuje [sprawozdanie](docs/sprawozdanie/sprawozdanie.pdf).

Wykonano już pełną serię 4 modeli po 30 epok i ewaluację na 100 publicznych
obrazach validation HR. Rzeczywiste tabele i interpretacja są w
[sprawozdaniu](docs/sprawozdanie/sprawozdanie.pdf), a ilustracje w `runs/evaluation/raport.html`.
SNE zweryfikowano ze slajdem 41 wykładu: jest to suma kwadratów błędów, czyli SSE.
Kolumnę SNE uzupełniono z wcześniej zapisanych SSE, bez ponownego treningu.

Po uporządkowaniu repozytorium zachowano końcowe modele `best.pt` i raport HTML.
Usunięto katalog `runs/preliminary/`, cztery końcowe checkpointy `last.pt` oraz
`runs/evaluation/raport.md`. Próby wstępne opisane w dokumentacji są nadal dostępne
w historii Git, np. w commicie `5069b22`. Nowy trening ponownie tworzy `last.pt`,
a `report --markdown` tworzy dodatkową wersję Markdown. Wznowienie treningu wymaga
zachowanego lokalnie checkpointu `last.pt`; pliki `best.pt` wystarczają do predykcji.

## Instalacja (Windows / PowerShell)

Z katalogu projektu, z Pythonem 3.10 lub nowszym:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -X utf8 -m src doctor
```

Nie trzeba aktywować środowiska. Dla karty NVIDIA można najpierw zainstalować
zgodną parę `torch` / `torchvision` z CUDA, a potem pozostałe zależności.
Przykład zgodnej pary z oficjalnego indeksu CUDA 12.8:

```powershell
.venv/Scripts/python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Wynik `doctor` powinien pokazać nazwę GPU i wersję CUDA. Instalacja CPU również
działa; `--device auto` automatycznie wybiera dostępne urządzenie.
Komendy działają też na Linuxie po zastąpieniu `.venv/Scripts/python.exe` przez
`.venv/bin/python`. Pętle w poniższych przykładach są zapisane dla PowerShell.

## Szybka próba bez DIV2K

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python.exe -X utf8 -m src demo-data --data data/demo
foreach ($condition in @("sr4", "sr8", "dn001", "dn003")) {
    .venv/Scripts/python.exe -X utf8 -m src train --condition $condition --data data/demo --runs runs/smoke --epochs 2 --repeat 1 --width 8 --blocks 1 --batch-size 2 --device cpu
    if ($LASTEXITCODE -ne 0) { throw "Trening $condition zakonczony bledem" }
}
.venv/Scripts/python.exe -X utf8 -m src evaluate --data data/demo --runs runs/smoke --output runs/smoke-evaluation --examples 2 --device cpu
```

Polecenia tworzą 13 obrazów proceduralnych, trenują cztery małe modele po dwie epoki
i zapisują raport w `runs/smoke-evaluation/raport.html`. Pierwsze użycie LPIPS
pobiera wagi SqueezeNet (ok. 5 MB). Dodanie `--skip-lpips` do komendy `evaluate`
pomija ten download i wyraźnie oznacza niekompletne metryki. Potrzebne są puste
katalogi `data/demo`, `runs/smoke` i `runs/smoke-evaluation`, aby nie nadpisać wcześniejszych prób.

## Pełny eksperyment DIV2K

Pobranie archiwów HR to około 3.9 GB. Rozpakowane źródła, środowisko CUDA,
archiwa i wyniki wymagają orientacyjnie 15-20 GB miejsca.

```powershell
.venv/Scripts/python.exe -X utf8 -m src download
.venv/Scripts/python.exe -X utf8 -m src prepare
.venv/Scripts/python.exe -X utf8 -m src verify

foreach ($condition in @("sr4", "sr8", "dn001", "dn003")) {
    .venv/Scripts/python.exe -X utf8 -m src train --condition $condition --runs runs/new --epochs 60 --device auto
    if ($LASTEXITCODE -ne 0) { throw "Trening $condition zakonczony bledem" }
}
.venv/Scripts/python.exe -X utf8 -m src tune-baseline --output runs/new/bilateral.json
.venv/Scripts/python.exe -X utf8 -m src evaluate --runs runs/new --output runs/new-evaluation --bilateral-config runs/new/bilateral.json
```

Pobieranie jest jawne i nie uruchamia się przy `train` ani `evaluate`.
Przerwany trening pojedynczego wariantu można wznowić, dodając `--resume`
do jego komendy `train` z tymi samymi parametrami i zachowanym `last.pt`.
Zmiana parametrów wymaga nowego katalogu `--runs` i `--output`.

Domyślny plan eksperymentu to **60 epok na wariant**, Adam, MSE, learning rate
0.001 z harmonogramem kosinusowym, batch 8, 32 kanały, 6 bloków residualnych.
Każdy obraz treningowy jest używany 4 razy na epokę, z rotacjami, odbiciami i
nową realizacją szumu. To punkt startowy do eksperymentów, nie gwarancja poprawy
względem baseline. Czas zależy od urządzenia; pierwsze epoki podają rzeczywisty czas.

Seria wykonana w tym projekcie używa mniejszej sieci. Odtworzenie treningu
w osobnym katalogu:

```powershell
foreach ($condition in @("sr4", "sr8", "dn001", "dn003")) {
    $precision = if ($condition.StartsWith("dn")) { "bf16" } else { "fp32" }
    .venv/Scripts/python.exe -X utf8 -m src train --condition $condition --runs runs/reproduction --epochs 30 --width 24 --blocks 4 --repeat 2 --device cuda --precision $precision
    if ($LASTEXITCODE -ne 0) { throw "Trening $condition zakonczony bledem" }
}
.venv/Scripts/python.exe -X utf8 -m src tune-baseline --output runs/reproduction/bilateral.json
.venv/Scripts/python.exe -X utf8 -m src evaluate --runs runs/reproduction --output runs/reproduction-evaluation --bilateral-config runs/reproduction/bilateral.json
```

Dokładne wyniki: [sprawozdanie](docs/sprawozdanie/sprawozdanie.pdf). Do wznowienia
przerwanego wariantu należy użyć jego komendy `train` z `--resume` i identycznymi
parametrami. Odtworzenie od zera wymaga pustych katalogów `--runs` i `--output`,
aby zachować istniejące wyniki.
Plik `requirements-lock.txt` zawiera wersje zainstalowanego środowiska Windows/CUDA.
W tej serii SR używa FP32, a trening odszumiania BF16 (autocast, parametry i
optymalizator FP32). Walidacja, końcowe predykcje i metryki używają FP32.
BF16 wymaga zgodnej karty CUDA; dla CPU należy użyć `--device cpu --precision fp32`.
W treningu odszumiania `--training-noise auto` generuje szum bezpośrednio
w PyTorch na GPU (osobny generator z seed zależnym od epoki). Na CPU domyślnie
używa NumPy. `--training-noise numpy` pozwala wymusić generator CPU.
Walidacja i test zawsze korzystają z deterministycznego generatora NumPy
zależnego od nazwy obrazu i warunku, wspólnego dla wszystkich porównywanych metod.

### Ważne: podział danych i brak oficjalnych test HR

[Oficjalny DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) udostępnia 800 obrazów
train HR i 100 validation HR. Oficjalnych test HR 0901-1000 nie udostępniono
publicznie, więc nie da się policzyć dla nich pełnoreferencyjnych PSNR/SSIM/LPIPS
bez uzyskania referencji od prowadzącego.

Projekt stosuje jawny, rozłączny podział:

| Rola | Obrazy | Liczba | Zastosowanie |
| --- | --- | --- | --- |
| train | 0001-0720 z train HR | 720 | uczenie wag |
| val | 0721-0800 z train HR | 80 | wybór epoki i parametrów bilateral |
| test zastępczy | 0801-0900 z validation HR | 100 | końcowe porównanie wszystkich obrazów |

Raport oznacza test zastępczy jako `public-validation-proxy`. Nie nazywa go
oficjalnym testem DIV2K. Aby spełnić dosłownie wymóg testu z PDF, trzeba otrzymać
testowe referencje HR i przygotować nowy zbiór:

```powershell
.venv/Scripts/python.exe -m src prepare --test-dir C:/sciezka/DIV2K_test_HR --data data/div2k-official
```

Parametr `--test-dir` przyjmuje wszystkie obrazy PNG z katalogu. Przygotowanie
odrzuca powtórzone pliki źródłowe między podziałami. Manifest zapisuje SHA-256
źródeł i gotowych obrazów; trening i ewaluacja sprawdzają ich zgodność.

Każdy pełny obraz jest przeskalowany do 256 x 256 przez OpenCV `INTER_AREA`
(bez wycinania; proporcje oryginału mogą się zmienić). Redukcja do 64/32 px
również używa `INTER_AREA`. Baseline SR powiększa przez `INTER_CUBIC`.
Szum to niezależne RGB `N(0, sigma^2)` dodane do danych float32 w [0,1],
z końcowym clippingiem do [0,1]. W walidacji i teście szum jest deterministyczny
dla nazwy obrazu, wariantu i ziarna; każda metoda dostaje identyczne wejście.

## Osobne komendy

```powershell
# Jeden model; pozostałe warianty: sr8, dn001, dn003
.venv/Scripts/python.exe -m src train --condition sr4 --epochs 60 --device auto

# Wznowienie przerwanego treningu (identyczna konfiguracja)
.venv/Scripts/python.exe -m src train --condition sr4 --epochs 60 --resume

# Porównawczy eksperyment L1 zamiast MSE, w osobnym katalogu
.venv/Scripts/python.exe -m src train --condition sr4 --loss l1 --runs runs/ablation-l1

# Dobór baseline odszumiania na wewnętrznym val
.venv/Scripts/python.exe -m src tune-baseline

# Wszystkie obrazy testowe, wszystkie 4 warianty i komplet metryk z SNE=SSE
.venv/Scripts/python.exe -m src evaluate --bilateral-config runs/main/bilateral.json

# Raport HTML z zapisanych wyników (bez ponownego liczenia metryk)
.venv/Scripts/python.exe -m src report

# Opcjonalna, dodatkowa wersja Markdown
.venv/Scripts/python.exe -m src report --markdown

# Własne już zdegradowane wejścia: 64x64 dla sr4 albo 256x256 dla denoising
.venv/Scripts/python.exe -m src predict --checkpoint runs/main/sr4/best.pt --input examples/input_sr4.png --output restored256.png
```

`evaluate --conditions sr4` pozwala ocenić pojedynczy wariant, a `--limit 5`
sprawdza krótki przebieg. **Do końcowych wyników nie używać `--limit` ani
`--skip-lpips`**. Wyniki trafiają do nowego, pustego katalogu `--output`.

Archiwum `output/SIGK_projekt.zip` zawiera źródła, checkpointy, przykłady i wyniki,
bez surowego DIV2K i środowiska `.venv`. Aby przygotować nowe archiwum ręcznie,
spakuj katalogi `sigk`, `tests`, `docs`, `examples`, `runs/main`, `runs/evaluation`
oraz pliki z katalogu głównego. Pomiń `__pycache__`, `.git`, `.venv`, `.cache`,
`data`, `tmp`, `.idea` i sam katalog `output`.
Raport i `predict` można uruchamiać po rozpakowaniu i zainstalowaniu zależności;
ponowny trening wymaga osobnego przygotowania danych. Gotowe wejścia i wyniki
z jednego obrazu walidacyjnego są w `examples/`. Nowe wyniki dla tych wejść
można uzyskać komendą `predict`, opisaną również w `examples/README.md`.

## Metryki i wyniki

Metryki są liczone na RGB float [0,1], na całym obrazie 256 x 256, bez usuwania
brzegów. Wynikiem zbiorczym jest średnia arytmetyczna wyników każdego obrazu,
a nie PSNR obliczone z uśrednionego MSE.

- MSE = suma kwadratów błędu / (H * W * 3).
- SNE = SSE = suma kwadratów błędu po wszystkich pikselach i kanałach RGB.
- PSNR = -10 log10(MSE); dla identycznych obrazów wynosi +inf.
- SSIM: skimage, RGB, okno Gaussa, sigma=1.5, `data_range=1`,
  `use_sample_covariance=False`.
- LPIPS: pretrained SqueezeNet, LPIPS v0.1, wejście RGB [-1,1], tryb `eval()`.
  `--lpips-net alex` / `vgg` zmienia backbone i pobiera większe wagi. Nie należy
  porównywać bezpośrednio wartości LPIPS uzyskanych z różnymi backbone'ami.
- **SNE (Squared Norm Error)** zweryfikowano na podstawie slajdu 41 wykładu:
  `sum((reference - prediction)**2)`, bez dzielenia przez liczbę wartości
  ani przez energię obrazu referencyjnego. Dla RGB 256 x 256 zachodzi
  `SNE = 256 * 256 * 3 * MSE`. Niższe SNE jest lepsze. Obrazy są w [0,1];
  dla czarnej referencji nie występuje dzielenie przez zero.
  Wariant `sse` jest domyślny i jedyny dopuszczony przez `--sne-definition`.
  W CSV zachowano kolumnę `SSE` jako alias SNE. Istniejące wyniki uzupełniono
  bez zmiany PSNR, SSIM, LPIPS, wag modeli i podziału danych.

Starsze wyniki z tego eksperymentu zostały już jednorazowo uzupełnione z SSE.
Kopia wcześniejszych plików znajduje się lokalnie w `tmp/sne_before_update/`.
Nowa komenda `evaluate` od razu zapisuje poprawne SNE, a `report` odtwarza HTML
z zapisanych wyników. Do dalszej pracy nie jest potrzebne osobne narzędzie migracji.

LPIPS nigdy nie przechodzi po cichu na losowe wagi. Błąd pobierania wag przerywa
ewaluację; `--skip-lpips` jest wyłącznie opcją do niepełnej próby technicznej.

Po eksperymencie powstają:

```text
runs/main/<wariant>/
  config.json          parametry i wersje środowiska
  history.csv          strata, PSNR walidacji, czas każdej epoki
  best.pt              model wybrany na val
  last.pt              stan modelu/optymalizatora/harmonogramu do wznowienia
runs/main/bilateral.json  parametry baseline wybrane na val
runs/evaluation/
  raport.html          raport do przeglądarki i wydruku
  raport.md            opcjonalna wersja tekstowa (report --markdown)
  summary.csv/json     tabela dla wariantów i metod
  per_image.csv        metryki każdego obrazu
  metadata.json        protokół, seed, hashe modeli i danych, wersje bibliotek
  charts/              krzywe treningu i walidacji
  images/              porównania RGB i mapy błędu
```

Wnioski w raporcie powstają z pomiarów i nie zakładają, że sieć wygrała.
Więcej o architekturze i przeprowadzonych badaniach: [sprawozdanie](docs/sprawozdanie/sprawozdanie.pdf).

## Źródła

- [Specyfikacja zadania](SIGK___Projekt_1.pdf).
- [DIV2K: źródło danych, podział i licencja akademicka](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Agustsson i Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, CVPR Workshops, 2017.
- [PyTorch: oficjalne pary wersji i warianty CUDA](https://pytorch.org/get-started/previous-versions/).
- [LPIPS: implementacja autorów i opis normalizacji](https://github.com/richzhang/PerceptualSimilarity).
- [scikit-image: denoise_bilateral](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral).
