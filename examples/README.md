# Przykłady działania predict

Referencja: DIV2K train HR 0721, wewnętrzny podział val, resize RGB 256x256.
Dane służą do prezentacji działania; nie są końcowym zbiorem testowym.

Wejścia input_sr4/input_sr8 mają rozmiary 64/32 px, a input_dn001/input_dn003 256 px i odpowiednio szum sigma 0.01/0.03 (seed=42). Wszystkie zapisano jako PNG 8-bit. Wyjścia output_* pochodzą z komendy predict na zapisanych wejściach. Metryki końcowego raportu obliczono osobno na danych float przed zapisem PNG.

Przykładowa komenda z katalogu projektu:

```powershell
.venv/Scripts/python.exe -m sigk predict --checkpoint runs/main/sr4/best.pt --input examples/input_sr4.png --output examples/my_result.png
```

Źródło i licencja akademicka: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
