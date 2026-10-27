# SIGK - wyniki eksperymentów

> Użyto publicznych DIV2K validation HR jako testu zastępczego. Oficjalne DIV2K test HR nie są publicznie udostępnione.

> SNE: brak definicji w PDF. Nie zgadywano wzoru; policzono dodatkowo MSE i SSE. Przed oddaniem należy potwierdzić definicję z wykładu.

Zbiór: DIV2K; podział: test; N=100; seed=42.

| Wariant | Metoda | N | PSNR ↑ [dB] | SSIM ↑ | LPIPS ↓ | SNE* | MSE ↓ | SSE ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dn001 | ResidualNet denoise sigma=0.01 | 100 | 42.55638 | 0.98815 | 0.01148 | brak | 0.00006 | 11.13978 |
| dn001 | bilateral sigma=0.01 | 100 | 38.78925 | 0.98155 | 0.01427 | brak | 0.00014 | 27.52392 |
| dn001 | noisy sigma=0.01 | 100 | 40.06963 | 0.97085 | 0.01785 | brak | 0.00010 | 19.36078 |
| dn003 | ResidualNet denoise sigma=0.03 | 100 | 36.01371 | 0.96636 | 0.02206 | brak | 0.00027 | 52.21454 |
| dn003 | bilateral sigma=0.03 | 100 | 31.34846 | 0.91921 | 0.06084 | brak | 0.00078 | 152.79940 |
| dn003 | noisy sigma=0.03 | 100 | 30.59695 | 0.82854 | 0.10088 | brak | 0.00087 | 171.57069 |
| sr4 | ResidualNet SR 64x64->256x256 (x4) | 100 | 23.62919 | 0.68984 | 0.28860 | brak | 0.00537 | 1055.00826 |
| sr4 | bicubic OpenCV x4 | 100 | 22.91310 | 0.64445 | 0.39782 | brak | 0.00619 | 1216.18878 |
| sr8 | ResidualNet SR 32x32->256x256 (x8) | 100 | 20.74784 | 0.50014 | 0.59899 | brak | 0.00998 | 1962.03603 |
| sr8 | bicubic OpenCV x8 | 100 | 20.29669 | 0.46727 | 0.73548 | brak | 0.01093 | 2148.49404 |

## Protokół

RGB [0,1]; full 256x256; mean per-image metrics; no border crop. LPIPS: squeeze. SNE*: niezdefiniowane. SSIM: okno Gaussa, sigma=1.5, data_range=1, covariance population. Tabele zawierają średnie arytmetyczne metryk liczonych osobno dla każdego obrazu. Pełna konfiguracja i wersje bibliotek: metadata.json; wyniki jednostkowe: per_image.csv.

## Architektura

ResidualNet: Conv 3x3, bloki residualne Conv-ReLU-Conv ze skalą 0.1, globalny skip cech i rekonstrukcja RGB. SR wykorzystuje PixelShuffle x2 powtarzany 2/3 razy oraz dodaje poprawkę do OpenCV bicubic. Odszumianie dodaje uczoną poprawkę do wejścia. Osobne wagi dla każdego wariantu. Brak BatchNorm; zerowa inicjalizacja ostatniej warstwy. Najlepszą epokę wybrano według PSNR wewnętrznej walidacji; test nie służy do doboru wag.

- sr4: 24 kanałów, 4 bloków; wybrana epoka 29. Plan: 30 epok, repeat=2, batch=8, loss=mse, lr=0.001, precyzja treningu=fp32, generator szumu=numpy; walidacja FP32.
- sr8: 24 kanałów, 4 bloków; wybrana epoka 28. Plan: 30 epok, repeat=2, batch=8, loss=mse, lr=0.001, precyzja treningu=fp32, generator szumu=numpy; walidacja FP32.
- dn001: 24 kanałów, 4 bloków; wybrana epoka 30. Plan: 30 epok, repeat=2, batch=8, loss=mse, lr=0.001, precyzja treningu=bf16, generator szumu=torch; walidacja FP32.
- dn003: 24 kanałów, 4 bloków; wybrana epoka 29. Plan: 30 epok, repeat=2, batch=8, loss=mse, lr=0.001, precyzja treningu=bf16, generator szumu=torch; walidacja FP32.

Parametry bilateral: {'dn001': {'color_factor': 1.0, 'spatial': 3.0}, 'dn003': {'color_factor': 1.0, 'spatial': 3.0}}.

## Wnioski

- dn001: zmiana PSNR sieci względem metody bazowej wynosi +3.767 dB, a SSIM +0.00660. Zmiana LPIPS: -0.00279 (mniej = lepiej). Uzyskano poprawę średniego PSNR.
- dn003: zmiana PSNR sieci względem metody bazowej wynosi +4.665 dB, a SSIM +0.04715. Zmiana LPIPS: -0.03879 (mniej = lepiej). Uzyskano poprawę średniego PSNR.
- sr4: zmiana PSNR sieci względem metody bazowej wynosi +0.716 dB, a SSIM +0.04539. Zmiana LPIPS: -0.10922 (mniej = lepiej). Uzyskano poprawę średniego PSNR.
- sr8: zmiana PSNR sieci względem metody bazowej wynosi +0.451 dB, a SSIM +0.03287. Zmiana LPIPS: -0.13649 (mniej = lepiej). Uzyskano poprawę średniego PSNR.

## Wizualizacje

![sr4_training](charts/sr4_training.png)

![sr8_training](charts/sr8_training.png)

![dn001_training](charts/dn001_training.png)

![dn003_training](charts/dn003_training.png)

![dn001_0801](images/dn001_0801.png)

![dn001_0802](images/dn001_0802.png)

![dn001_0803](images/dn001_0803.png)

![dn001_0804](images/dn001_0804.png)

![dn001_0805](images/dn001_0805.png)

![dn003_0801](images/dn003_0801.png)

![dn003_0802](images/dn003_0802.png)

![dn003_0803](images/dn003_0803.png)

![dn003_0804](images/dn003_0804.png)

![dn003_0805](images/dn003_0805.png)

![sr4_0801](images/sr4_0801.png)

![sr4_0802](images/sr4_0802.png)

![sr4_0803](images/sr4_0803.png)

![sr4_0804](images/sr4_0804.png)

![sr4_0805](images/sr4_0805.png)

![sr8_0801](images/sr8_0801.png)

![sr8_0802](images/sr8_0802.png)

![sr8_0803](images/sr8_0803.png)

![sr8_0804](images/sr8_0804.png)

![sr8_0805](images/sr8_0805.png)
