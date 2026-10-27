# Stan wykonania

Ten plik oddziela implementację od wyników rzeczywiście wykonanych eksperymentów.

- Zaimplementowano dwa zadania, cztery warianty degradacji, trening i ewaluację.
- Zainstalowano PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 i pozostałe zależności.
- Potwierdzono działanie CUDA 12.8 na RTX 5080 Laptop GPU.
- Wszystkie 12 testów jednostkowych przeszło; `pip check` nie wykazał konfliktów.
- Wykonano syntetyczny test całego procesu (4 warianty po 2 epoki, bez LPIPS).
- Niezależnie sprawdzono pretrained LPIPS SqueezeNet: wynik identycznych obrazów = 0.
- Pobrano pełne publiczne archiwa DIV2K train/validation HR (3.98 GB).
- Przygotowano i sprawdzono podział 720/80/100 obrazów RGB 256 x 256.
- Ukończono trening DIV2K wszystkich 4 modeli: po 30 epok, 24 kanały, 4 bloki, repeat=2, batch=8, MSE, seed=42.
- SR: trening FP32; odszumianie: BF16 z parametrami FP32; walidacja i ewaluacja FP32.
- Wstępne 4 epoki DN001 w FP32 zachowano w `runs/preliminary/dn001-fp32`; nie są końcowym eksperymentem.
- Wstępne 9 epok BF16 z szumem CPU zachowano w `runs/preliminary/dn001-bf16-numpy`.
- Końcowy trening odszumiania generuje szum na GPU, a walidacja i test stosują stały szum NumPy.
- Ukończono dobór bilateral na wszystkich 80 obrazach wewnętrznej walidacji.
- Ukończono ewaluację wszystkich 100 publicznych DIV2K validation HR jako testu zastępczego.
- Sprawdzono 1000 rekordów pomiarów: 10 par metoda/wariant, każda po 100 obrazów.
- Obliczono PSNR, SSIM, pretrained LPIPS SqueezeNet v0.1 oraz MSE i SSE.
- Modele poprawiły średnie PSNR, SSIM i LPIPS względem porównywanych baseline we wszystkich 4 wariantach.
- Powstały raport HTML/Markdown, 20 porównań obrazów i 4 wykresy treningu.
- Sprawdzono predict dla 4 wariantów; gotowe wejścia i wyjścia są w `examples/`.
- Sprawdzono hashe modeli i danych w metadanych oraz wymiary obrazów przykładowych.
- Definicja SNE wymaga potwierdzenia na podstawie wykładu.
- Publiczny DIV2K validation HR może być testem zastępczym; nie jest oficjalnym test HR.

Logi treningu znajdują się w `runs/main/<wariant>/history.csv`.
Nie ma tu przykładowych liczb udających wyniki DIV2K.

## Wyniki PSNR (100 obrazów na metodę)

| Wariant | Baseline [dB] | ResidualNet [dB] | Różnica [dB] |
| --- | --- | --- | --- |
| SR x4 | 22.9131 | 23.6292 | +0.7161 |
| SR x8 | 20.2967 | 20.7478 | +0.4511 |
| Denoise sigma 0.01 | 38.7893 | 42.5564 | +3.7671 |
| Denoise sigma 0.03 | 31.3485 | 36.0137 | +4.6653 |

Wybrane epoki: sr4=29, sr8=28, dn001=30, dn003=29.
Pełne tabele PSNR/SSIM/LPIPS i interpretacja: `docs/WYNIKI.md` oraz
`runs/evaluation/raport.html`. Dla SNE pozostawiono jawny brak wartości;
nie jest to deklaracja wykonania tej nieokreślonej metryki.
