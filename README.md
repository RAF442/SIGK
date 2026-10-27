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
nie zastępują eksperymentów na DIV2K**. Stan wykonania opisuje [STATUS.md](STATUS.md).

Wykonano już pełną serię 4 modeli po 30 epok i ewaluację na 100 publicznych
obrazach validation HR. Rzeczywiste tabele i interpretacja są w
[docs/WYNIKI.md](docs/WYNIKI.md), a ilustracje w `runs/evaluation/raport.html`.
Wzór SNE pozostaje do potwierdzenia; wartości tej metryki nie zostały zmyślone.

## Instalacja (Windows / PowerShell)

Z katalogu projektu, z Pythonem 3.10 lub nowszym:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -X utf8 -m sigk doctor
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
`.venv/bin/python`. Skrypty `.ps1` są przeznaczone dla Windows.

## Szybka próba bez DIV2K

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
./scripts/smoke.ps1
```

Skrypt tworzy 13 obrazów proceduralnych, trenuje cztery małe modele po dwie epoki
i zapisuje raport w `runs/smoke-evaluation/raport.html`. Pierwsze użycie LPIPS
pobiera wagi SqueezeNet (ok. 5 MB). `./scripts/smoke.ps1 -SkipLpips` pomija ten
download i wyraźnie oznacza niekompletne metryki. Skrypt wymaga pustych katalogów
`data/demo`, `runs/smoke` i `runs/smoke-evaluation`; nie nadpisuje wcześniejszych prób.

## Pełny eksperyment DIV2K

Pobranie archiwów HR to około 3.9 GB. Rozpakowane źródła, środowisko CUDA,
archiwa i wyniki wymagają orientacyjnie 15-20 GB miejsca.

```powershell
.venv/Scripts/python.exe -X utf8 -m sigk download
.venv/Scripts/python.exe -X utf8 -m sigk prepare
.venv/Scripts/python.exe -X utf8 -m sigk verify
./scripts/run_all.ps1 -Epochs 60 -Device auto
```

Można też wykonać wszystko jednym skryptem: `./scripts/run_all.ps1 -Download`.
Pobieranie jest jawne i nie uruchamia się przy `train` ani `evaluate`.
Przerwanie treningu można wznowić przez `./scripts/run_all.ps1 -Resume` z tymi
samymi parametrami. Zmiana parametrów wymaga nowego katalogu `-Runs` i `-Output`.

Domyślny plan eksperymentu to **60 epok na wariant**, Adam, MSE, learning rate
0.001 z harmonogramem kosinusowym, batch 8, 32 kanały, 6 bloków residualnych.
Każdy obraz treningowy jest używany 4 razy na epokę, z rotacjami, odbiciami i
nową realizacją szumu. To punkt startowy do eksperymentów, nie gwarancja poprawy
względem baseline. Czas zależy od urządzenia; pierwsze epoki podają rzeczywisty czas.

Seria wykonana w tym projekcie używa mniejszej sieci i następującej komendy:

```powershell
./scripts/run_all.ps1 -Epochs 30 -Width 24 -Blocks 4 -Repeat 2 -Device cuda -DenoisePrecision bf16
```

Dokładny stan zakończenia i wyniki: [STATUS.md](STATUS.md). Do wznowienia tej
konkretnej serii należy dodać `-Resume` do powyższej komendy. Odtworzenie od zera
wymaga nowego `-Runs` i `-Output`, aby zachować istniejące wyniki.
Plik `requirements-lock.txt` zawiera wersje zainstalowanego środowiska Windows/CUDA.
W tej serii SR używa FP32, a trening odszumiania BF16 (autocast, parametry i
optymalizator FP32). Walidacja, końcowe predykcje i metryki używają FP32.
BF16 wymaga zgodnej karty CUDA; dla CPU należy pominąć `-DenoisePrecision bf16`.
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
.venv/Scripts/python.exe -m sigk prepare --test-dir C:/sciezka/DIV2K_test_HR --data data/div2k-official
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
.venv/Scripts/python.exe -m sigk train --condition sr4 --epochs 60 --device auto

# Wznowienie przerwanego treningu (identyczna konfiguracja)
.venv/Scripts/python.exe -m sigk train --condition sr4 --epochs 60 --resume

# Porównawczy eksperyment L1 zamiast MSE, w osobnym katalogu
.venv/Scripts/python.exe -m sigk train --condition sr4 --loss l1 --runs runs/ablation-l1

# Dobór baseline odszumiania na wewnętrznym val
.venv/Scripts/python.exe -m sigk tune-baseline

# Wszystkie obrazy testowe, wszystkie 4 warianty i komplet metryk poza niezdefiniowanym SNE
.venv/Scripts/python.exe -m sigk evaluate --bilateral-config runs/main/bilateral.json

# Raport z zapisanych wyników (bez ponownego liczenia metryk)
.venv/Scripts/python.exe -m sigk report

# Własne już zdegradowane wejścia: 64x64 dla sr4 albo 256x256 dla denoising
.venv/Scripts/python.exe -m sigk predict --checkpoint runs/main/sr4/best.pt --input examples/input_sr4.png --output restored256.png
```

`evaluate --conditions sr4` pozwala ocenić pojedynczy wariant, a `--limit 5`
sprawdza krótki przebieg. **Do końcowych wyników nie używać `--limit` ani
`--skip-lpips`**. Wyniki trafiają do nowego, pustego katalogu `--output`.

Po zakończeniu głównej ewaluacji można utworzyć archiwum do przekazania:
`.venv/Scripts/python.exe scripts/package_project.py`. Powstanie
`output/SIGK_projekt.zip` ze źródłami, checkpointami i wynikami, bez surowego
DIV2K i środowiska `.venv`. Raport i `predict` można uruchamiać z archiwum
po zainstalowaniu zależności; ponowny trening wymaga osobnego przygotowania danych.
Przykłady w `examples/` zawierają gotowe wejścia i wyniki z jednego obrazu
walidacyjnego. Odtworzenie przykładów po treningu:
`.venv/Scripts/python.exe scripts/make_examples.py`.

## Metryki i wyniki

Metryki są liczone na RGB float [0,1], na całym obrazie 256 x 256, bez usuwania
brzegów. Wynikiem zbiorczym jest średnia arytmetyczna wyników każdego obrazu,
a nie PSNR obliczone z uśrednionego MSE.

- MSE = suma kwadratów błędu / (H * W * 3).
- SSE = suma kwadratów błędu; dodatkowa jawnie nazwana metryka.
- PSNR = -10 log10(MSE); dla identycznych obrazów wynosi +inf.
- SSIM: skimage, RGB, okno Gaussa, sigma=1.5, `data_range=1`,
  `use_sample_covariance=False`.
- LPIPS: pretrained SqueezeNet, LPIPS v0.1, wejście RGB [-1,1], tryb `eval()`.
  `--lpips-net alex` / `vgg` zmienia backbone i pobiera większe wagi. Nie należy
  porównywać bezpośrednio wartości LPIPS uzyskanych z różnymi backbone'ami.
- **SNE: w PDF nie ma wzoru.** Domyślnie kolumna ma `brak`; nie podstawiamy
  niepotwierdzonej definicji. Po sprawdzeniu notatek można wybrać
  `--sne-definition sse`, `mse` lub `relative-squared-error` (SSE / suma
  kwadratów referencji). Jeżeli definicja wykładowa jest inna, należy dodać ją
  w `sigk/metrics.py`. Wybrany wzór zapisuje się w metadanych i raporcie.

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
  raport.md            wersja tekstowa
  summary.csv/json     tabela dla wariantów i metod
  per_image.csv        metryki każdego obrazu
  metadata.json        protokół, seed, hashe modeli i danych, wersje bibliotek
  charts/              krzywe treningu i walidacji
  images/              porównania RGB i mapy błędu
```

Wnioski w raporcie powstają z pomiarów i nie zakładają, że sieć wygrała.
Więcej o architekturze, planie badań i prezentacji: [docs/PROJEKT.md](docs/PROJEKT.md).

## Źródła

- [Specyfikacja zadania](SIGK___Projekt_1.pdf).
- [DIV2K: źródło danych, podział i licencja akademicka](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Agustsson i Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, CVPR Workshops, 2017.
- [PyTorch: oficjalne pary wersji i warianty CUDA](https://pytorch.org/get-started/previous-versions/).
- [LPIPS: implementacja autorów i opis normalizacji](https://github.com/richzhang/PerceptualSimilarity).
- [scikit-image: denoise_bilateral](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral).
