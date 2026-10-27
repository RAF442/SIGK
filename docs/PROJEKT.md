# Dokumentacja projektu

## Cel i zakres

Celem jest porównanie uczonej rekonstrukcji obrazów z klasycznymi metodami
dla dwóch zadań: super-resolution (x4, x8) i odszumiania (sigma 0.01, 0.03).
W każdym wariancie obraz referencyjny jest RGB 256 x 256. Deblurowanie
i inpainting nie należą do wybranych dwóch zadań.

## Architektura

Rodzina `Restorer` dzieli konstrukcję bloku residualnego między zadaniami.
Uczymy osobne wagi dla czterech warunków degradacji.

```text
wejście RGB
  -> Conv3x3 (3 -> C)
  -> B bloków: z <- z + 0.1 * Conv3x3(ReLU(Conv3x3(z)))
  -> Conv3x3 + globalny skip cech
  -> [SR: 2 lub 3 razy Conv3x3(C -> 4C), PixelShuffle(2), ReLU]
  -> Conv3x3 (C -> 3), start z zerowymi wagami i biasem
  -> dodanie poprawki do bicubic (SR) lub zaszumionego obrazu (DN)
  -> na potrzeby ewaluacji: clip [0,1]
```

Domyślnie C=32, B=6. Brak BatchNorm pozwala zachować informację o jasności
i zakresie wartości. Bloki ze skalą 0.1 i globalne połączenie skrótowe
ułatwiają uczenie poprawki. Początkowa predykcja SR jest dokładnie zgodna
z baseline OpenCV; dla odszumiania jest identyczna z wejściem.
Większość obliczeń SR odbywa się w niskiej rozdzielczości.
Końcowe powiększenie cech odbywa się przez PixelShuffle.

W treningu brak clippingu wyjścia, żeby zachować gradient przy granicach [0,1].
Walidacja, metryki i zapis PNG używają clippingu. Obliczenia metryk korzystają
z predykcji float, przed kwantyzacją do PNG.

## Protokół badań

1. Przygotować dane i zachować manifest. Nie zmieniać podziału po obejrzeniu testu.
2. Uruchomić wszystkie cztery warianty z tym samym seed=42 i budżetem epok.
3. Wybierać najlepszy checkpoint według średniego PSNR wewnętrznego val.
4. Dobrać bilateral na tym samym wewnętrznym val: sigma_color / sigma
   w {1,2,3}, sigma_spatial w {1,3}; użyć najlepszego średniego PSNR.
5. Ustalić definicję SNE na podstawie wykładu i zapisać ją w konfiguracji ewaluacji.
6. Zamrozić parametry i ocenić wszystkie obrazy testu dla każdej metody.
7. Sprawdzić PSNR, SSIM, LPIPS, mapy błędu i konkretne przykłady.

Bez doboru baseline program używa sigma_color=2*sigma, sigma_spatial=3,
channel_axis=-1 i mode=reflect. W tabelach odszumiania pokazuje również
wyniki wejścia zaszumionego. SR porównuje OpenCV bicubic i ResidualNet.
Raport zawiera osobny wiersz dla każdej metody i degradacji.

Seed walidacji i testu zależy od nazwy pliku oraz warunku, a nie od kolejności
iteracji. Wznowienie od granicy epoki odtwarza tasowanie i augmentację dzięki
seed zależnemu od epoki i indeksu próbki. Szum treningowy na GPU korzysta
z osobnego generatora PyTorch z seed=seed_bazowy+epoka; rozmiar batcha i kolejność
obrazów są zapisane w konfiguracji. Wariant CPU korzysta z NumPy dla każdej próbki.
Checkpoint przechowuje optymalizator,
harmonogram, historię, architekturę oraz hash manifestu. Obliczenia na różnych
wersjach CUDA lub różnych GPU mogą nadal różnić się numerycznie.

W wykonanej serii zastosowano 30 epok, C=24, B=4 i repeat=2. Modele SR uczono
w FP32. Odszumianie uczono z autocast BF16 na GPU, zachowując parametry modelu
i stan optymalizatora FP32. BF16 ma zakres wykładnika jak FP32, więc nie wymaga
skalowania gradientów używanego zwykle przy FP16. Walidacja i ewaluacja obu
zadań odbywają się w FP32. Wstępne próby DN001 (4 epoki FP32 i 9 epok BF16
z generowaniem szumu na CPU) zachowano osobno w `runs/preliminary`;
nie wchodzą do końcowej tabeli jakości. Końcowy trening DN używa generowania
szumu na GPU, aby ograniczyć koszt transferów i obliczeń CPU.

## Eksperymenty rozszerzające

Minimalna seria obejmuje cztery warunki wejścia i odpowiednie baseline.
Aby zbadać wpływ architektury, można porównać po 60 epok:

| Eksperyment | Kanały | Bloki | Strata | Cel |
| --- | --- | --- | --- | --- |
| główny | 32 | 6 | MSE | optymalizacja PSNR |
| mniejsza sieć | 16 | 3 | MSE | wpływ pojemności i czasu |
| alternatywna strata | 32 | 6 | L1 | porównanie PSNR i LPIPS |

Każdy eksperyment zapisuje się w osobnym `--runs` i `--output`.
Do wyboru konfiguracji używać `evaluate --split val`; po jej wyborze wykonać
jedno końcowe porównanie testowe. Nie stroić parametrów na podstawie testu.
Przy większym budżecie powtórzyć trening z kilkoma seedami i podać zmienność.
Powyższe rozszerzenia są planem badania, a nie deklaracją wykonanych pomiarów.

## Interpretacja

PSNR i SSIM powinny rosnąć, LPIPS powinien maleć. Poprawa PSNR przy pogorszeniu
LPIPS oznacza kompromis między błędem pikselowym a podobieństwem percepcyjnym.
Duże x8 traci więcej informacji niż x4. Przy małym szumie sigma=0.01 zbyt silne
wygładzanie może usuwać detale, dlatego należy porównać wynik również z wejściem.
To hipotezy do sprawdzenia; wnioski o konkretnym modelu muszą wynikać z tabel.

Nie ma podstaw, by przed treningiem twierdzić, że sieć przewyższa baseline.
Raport oblicza rzeczywiste różnice i jawnie pokazuje brak poprawy.

## Zgodność z zadaniem i ograniczenia

| Wymóg | Realizacja |
| --- | --- |
| Python i PyTorch | `sigk/models.py`, `sigk/engine.py` |
| dwa zadania, po dwa parametry | sr4, sr8, dn001, dn003 |
| DIV2K i 256 x 256 | `download`, `prepare`, manifest SHA-256 |
| interpolacja udokumentowana | INTER_AREA w redukcji, INTER_CUBIC w baseline |
| odpowiednie metody bazowe | OpenCV bicubic i skimage bilateral |
| PSNR, SSIM, LPIPS | `sigk/metrics.py`, pomiary każdego obrazu |
| SNE | oczekuje definicji wykładowej; jawny wybór wzoru w CLI |
| cały zbiór testowy | domyślnie wszystkie obrazy podziału test; referencje oficjalnego testu trzeba uzyskać |
| tabele z nazwą wariantu | summary.csv/json i raport HTML/Markdown |
| wizualizacje i dokumentacja | obrazy, mapy błędu, krzywe treningu, wnioski z pomiarów |

Publiczny zbiór validation HR służy domyślnie za zastępczy test obejmujący
100 obrazów. Raport nie utożsamia tego z oficjalnym testem HR.
Zmiana proporcji przy resize całego obrazu do kwadratu jest częścią przyjętego
protokołu i wpływa na porównywalność z innymi implementacjami.
LPIPS SqueezeNet musi być porównywany z wynikami tej samej wersji LPIPS i sieci.

## Materiały do prezentacji

Pokazać schemat modelu, podział danych, degradacje i ich parametry.
Następnie otworzyć `raport.html`: tabelę każdego warunku, krzywe walidacji,
porównania obrazów i mapy błędu. Omówić przypadki poprawy oraz pogorszenia
względem baseline, protokół testu zastępczego i przyjętą definicję SNE.
Na koniec uruchomić `predict` na przygotowanym wejściu, żeby pokazać działający model.

## Źródła

- Dokument zadania dostarczony w katalogu projektu.
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Agustsson, E. i Timofte, R. (2017), *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*.
- Zhang i in. (2018), *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR; [implementacja LPIPS](https://github.com/richzhang/PerceptualSimilarity).
- [scikit-image restoration](https://scikit-image.org/docs/stable/api/skimage.restoration.html).
- [Oficjalne wersje PyTorch](https://pytorch.org/get-started/previous-versions/).
