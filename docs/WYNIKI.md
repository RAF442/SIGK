# Wyniki wykonanych eksperymentów

Wyniki pochodzą z rzeczywistego treningu i ewaluacji w lokalnym środowisku
PyTorch 2.10.0+cu128 / CUDA 12.8, na RTX 5080 Laptop GPU. Zbiór treningowy
obejmował 720 obrazów, wewnętrzna walidacja 80, a końcowa ewaluacja wszystkie
100 publicznych obrazów DIV2K validation HR (0801-0900).
**Jest to test zastępczy, nie oficjalny, niepubliczny DIV2K test HR.**

Każda referencja ma 256 x 256 px i 3 kanały RGB. Metryki obliczono przed zapisem
PNG, na float [0,1], bez dodatkowego obcinania brzegów; tabele pokazują średnią
arytmetyczną metryk każdego obrazu. LPIPS używa pretrained SqueezeNet v0.1
i normalizacji RGB do [-1,1]. Seed degradacji: 42. Manifest źródeł i gotowych
obrazów znajduje się w `DIV2K_manifest.json`; metadane ewaluacji zawierają jego SHA-256.

## Zwiększanie rozdzielczości

Redukcja do 64/32 px: OpenCV INTER_AREA. Baseline: OpenCV INTER_CUBIC.
N=100 dla każdego wiersza. Większe PSNR i SSIM oraz mniejsze LPIPS są lepsze.

| Metoda i degradacja | PSNR [dB] | SSIM | LPIPS | SNE ↓ |
| --- | --- | --- | --- | --- |
| Bicubic x4, 64 x 64 -> 256 x 256 | 22.91310 | 0.64445 | 0.39782 | 1216.18878 |
| ResidualNet x4, 64 x 64 -> 256 x 256 | 23.62919 | 0.68984 | 0.28860 | 1055.00826 |
| Bicubic x8, 32 x 32 -> 256 x 256 | 20.29669 | 0.46727 | 0.73548 | 2148.49404 |
| ResidualNet x8, 32 x 32 -> 256 x 256 | 20.74784 | 0.50014 | 0.59899 | 1962.03603 |

Sieć poprawiła średni PSNR o 0.7161 dB dla x4 i 0.4511 dB dla x8.
Poprawie towarzyszył wzrost SSIM i spadek LPIPS. Przy x8 wyniki pozostają
wyraźnie słabsze niż przy x4: z 32 x 32 px trzeba odtworzyć więcej utraconej
informacji. Obrazy pokazują lepsze krawędzie, ale nadal brak części drobnych
tekstur; nie należy utożsamiać poprawy średnich metryk z pełnym odzyskaniem detali.

## Odszumianie

Szum to N(0,sigma^2), z clippingiem do [0,1]. Dla obu metod wykorzystano
identyczne realizacje szumu. Bilateral wybrano na wewnętrznym val spośród
sigma_color / sigma w {1,2,3} i sigma_spatial w {1,3}; w obu warunkach wygrały
sigma_color=sigma, sigma_spatial=3. Użyto skimage, RGB, channel_axis=-1,
mode=reflect. N=100 dla każdego wiersza.

| Metoda i degradacja | PSNR [dB] | SSIM | LPIPS | SNE ↓ |
| --- | --- | --- | --- | --- |
| Wejście zaszumione, sigma=0.01 | 40.06963 | 0.97085 | 0.01785 | 19.36078 |
| Bilateral, sigma=0.01 | 38.78925 | 0.98155 | 0.01427 | 27.52392 |
| ResidualNet, sigma=0.01 | 42.55638 | 0.98815 | 0.01148 | 11.13978 |
| Wejście zaszumione, sigma=0.03 | 30.59695 | 0.82854 | 0.10088 | 171.57069 |
| Bilateral, sigma=0.03 | 31.34846 | 0.91921 | 0.06084 | 152.79940 |
| ResidualNet, sigma=0.03 | 36.01371 | 0.96636 | 0.02206 | 52.21454 |

Względem wybranego bilateral sieć poprawiła PSNR o 3.7671 dB i 4.6653 dB.
Względem samego wejścia wzrost wyniósł odpowiednio 2.4868 dB i 5.4168 dB.
Sieć poprawiła też średnie SSIM i LPIPS względem obu odniesień.

Dla sigma=0.01 bilateral pogorszył PSNR względem wejścia, jednocześnie poprawiając
SSIM i LPIPS. W tym przypadku filtr zmienia też detale obrazu: nie każda poprawa
strukturalna lub percepcyjna oznacza mniejszy błąd pikselowy. Porównanie z wejściem
jest więc potrzebne obok porównania z wymaganą metodą bazową. Wyniki dotyczą
podanej siatki parametrów bilateral, a nie wszystkich możliwych konfiguracji filtra.

## Parametry i wybór modeli

Wszystkie cztery modele: 30 epok, 24 kanały, 4 bloki residualne, Adam,
początkowe lr=0.001, harmonogram kosinusowy, MSE, batch=8, repeat=2 i seed=42.
SR trenowano w FP32; odszumianie z autocast BF16 i szumem generowanym na GPU.
Walidacja i końcowe predykcje korzystają z FP32. Checkpoint wybrano na podstawie
średniego PSNR wewnętrznej walidacji, bez strojenia na wynikach końcowych.

| Wariant | Wybrana epoka | PSNR wewnętrznej walidacji [dB] |
| --- | --- | --- |
| sr4 | 29 | 24.46774 |
| sr8 | 28 | 21.46231 |
| dn001 | 30 | 42.65204 |
| dn003 | 29 | 36.17687 |

Wstępne, niepełne próby wydajności w `runs/preliminary` nie wchodzą do tabel
końcowych. Nie wykonano dodatkowych serii L1, zmiany architektury ani wielu seedów;
są one propozycjami dalszych eksperymentów w dokumentacji, nie wykonanymi badaniami.

## Pliki i ograniczenia

- Pełny raport z obrazami i krzywymi: `../runs/evaluation/raport.html`.
- Tabele maszynowe: `summary.csv`, `summary.json`; wyniki jednostkowe: `per_image.csv`.
- Łącznie 1000 rekordów pomiarowych, 20 porównań wizualnych i 4 wykresy treningu.
- Wszystkie 15 testów przeszło. Zweryfikowano hashe danych i modeli, kompletność
  ewaluacji oraz predykcję dla wszystkich 4 wariantów na plikach PNG.
- SNE zweryfikowano ze slajdem 41 wykładu: `sum((GT-wynik)^2)` po wszystkich
  pikselach i kanałach RGB [0,1]. Jest identyczne z zapisanym SSE, więc uzupełniono
  wyniki jednostkowe i średnie bez ponownego treningu i bez zmiany innych metryk.
  Dla czarnej referencji nie ma dzielenia przez zero. Niższe SNE jest lepsze.
- Publiczny validation HR jest testem zastępczym; oficjalne testowe HR muszą zostać
  dostarczone osobno. [Opis DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Porównano średnie na ustalonych 100 obrazach, bez testu istotności i badania
  zmienności między seedami. Wnioski nie oznaczają poprawy na każdym możliwym obrazie.
