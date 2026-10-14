# Segmentacja zębów i kanałów z CBCT

Pipeline przyjmuje **CBCT oraz gotową maskę zębów**, rozdziela zęby na instancje,
segmentuje kanały i zapisuje trzy pliki NIfTI. Można też podać gotową maskę
instancji i pominąć rozdzielanie zębów.

W repozytorium nie ma potwierdzonego modelu do kroku „surowe CBCT → maska
zębów”, więc maskę wejściową trzeba przygotować osobno. Do pozostałych etapów
wagi są już w repozytorium; nie trzeba ich pobierać z innego miejsca ani trenować.

## 1. Instalacja

Uruchamiaj polecenia z katalogu projektu. Zalecany Python 3.12; sprawdzono
PyTorch 2.10.0 z CUDA 12.8. NVIDIA GPU przyspiesza obliczenia, dostępny jest też CPU.

```powershell
git clone https://github.com/tycjantyc/zeby_tooth.git
cd zeby_tooth
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
```

Na Linuxie aktywuj środowisko przez `source .venv/bin/activate`. Jeśli PowerShell
blokuje aktywację, używaj `.venv\Scripts\python.exe` zamiast `python` w kolejnych
poleceniach; aktywacja nie jest konieczna.

Dla instalacji bez GPU zamień adres indeksu w poleceniu instalacji PyTorch na
`https://download.pytorch.org/whl/cpu`. `requirements-inference.txt` zawiera
zależności do inferencji; do pracy ze starymi skryptami treningowymi służy
`requirements.txt`.

## 2. Przygotowanie wejścia

- `scan.nii.gz`: jedno badanie CBCT, skalarna objętość 3D. Akceptowane są też `.nii`.
- `teeth_mask.nii.gz`: maska obejmująca zęby razem z wnętrzem/miazgą; `0` oznacza
  tło, wszystkie dodatnie etykiety są traktowane jako zęby. Nie podawaj całej
  wieloklasowej maski anatomii zawierającej np. kości szczęki.
- CBCT i maska muszą mieć ten sam rozmiar, spacing, origin i direction.
  Pipeline sprawdza zgodność i zachowuje tę geometrię w wynikach.
- Model centrów jest przeznaczony do maski całego łuku zębowego. Dla wycinka
  pojedynczego zęba użyj `--instances` (np. etykiety `0` i `1`), aby pominąć
  wykrywanie centrów i rozdzielanie.

## 3. Uruchomienie

Poniższe polecenie wykonuje rozdzielenie zębów i segmentację kanałów:

```powershell
python run_pipeline.py --cbct scan.nii.gz --teeth-mask teeth_mask.nii.gz --output-dir results/patient_001
```

Jeżeli masz już maskę, w której każdy ząb ma osobną dodatnią etykietę:

```powershell
python run_pipeline.py --cbct scan.nii.gz --instances teeth_instances.nii.gz --output-dir results/patient_002
```

Katalog wyników musi być pusty albo jeszcze nie istnieć; wybierz nowy katalog
dla każdego badania/uruchomienia. Na końcu program wypisuje JSON ze ścieżkami:

| Plik | Zawartość |
| --- | --- |
| `teeth_mask.nii.gz` | Binarna maska wszystkich zębów, `uint8` |
| `teeth_instances.nii.gz` | Osobne identyfikatory zębów, `uint16`; `0` = tło |
| `canals_mask.nii.gz` | Binarna maska kanałów wewnątrz zębów, `uint8` |

Identyfikatory wykrytych instancji nie są numeracją anatomiczną FDI. Przy wejściu
`--instances` zachowywane są jego identyfikatory. Pusta maska wejściowa daje
puste wyniki. Jeśli niepusta maska nie daje żadnych wykrytych zębów, program
zgłasza błąd.

Domyślnie `--device auto` wybiera CUDA, jeśli jest dostępna, w przeciwnym razie CPU.
Możesz podać `--device cpu` lub `--device cuda:0`. CPU może działać znacznie wolniej.
Opcjonalnie: `--weights-dir models/weights`, `--threshold 0.5`, `--stride 48`.
Mniejszy stride zwiększa nakładanie okien i czas obliczeń. Rozdzielanie używa
okien 128³; kanały — 96³. Duże zęby są przetwarzane w wielu oknach, także przy
brzegach objętości. Nie jest wykonywane automatyczne przeskalowanie spacingu.
Wyniki wymagają oceny jakości na danych, dla których mają być używane.

## Użycie jako funkcji / backend aplikacji webowej

```python
from inference.pipeline import SegmentationPipeline

# Utwórz raz przy starcie procesu. Modele są ładowane przy pierwszym użyciu
# i przechowywane do kolejnych badań.
pipeline = SegmentationPipeline(device="auto")

results = pipeline.run(
    cbct_path="scan.nii.gz",
    teeth_mask_path="teeth_mask.nii.gz",
    output_dir="results/patient_003",
)
print(results["canals_mask"])
```

Możesz zamiast `teeth_mask_path` podać `instances_path`. `run()` zwraca słownik
z absolutnymi ścieżkami `teeth_mask`, `teeth_instances`, `canals_mask`.
Wywołania na tej samej instancji pipeline są wykonywane kolejno. W aplikacji
webowej uruchamiaj `run()` w zadaniu w tle, przydzielaj osobny katalog każdemu
zadaniu i udostępniaj zwrócone pliki do pobrania. Nie ma tu jeszcze serwera HTTP
ani interfejsu webowego.

## Wagi

Wagi są zwykłymi plikami Git i przychodzą z `git clone` — Git LFS nie jest
wymagany. Repozytorium jest przez to duże. Domyślny pipeline korzysta z:

| Etap | Plik w `models/weights/` |
| --- | --- |
| Wykrywanie centrów | `agh_centroid_best_s2.pt` |
| Rozdzielanie zębów | `dl_separator_best_v2.pt` |
| Kanały | `tooth_fairy_3d_best_agh.pt` |

Brakujące lub niezgodne wagi powodują błąd, zamiast uruchomienia modelu
z losowymi parametrami. Ładowany jest `state_dict` z `weights_only=True`.

## Pojedynczy model i testy

`main_evaluate.py` pozostaje punktem wejścia do inferencji jednego modelu.
Wykrywa architekturę oraz liczbę kanałów z wag, obsługuje kanały współrzędnych
i zapisuje maskę z geometrią wejściowego NIfTI. Przykładowo:

```powershell
python main_evaluate.py --file scan.nii.gz --model_weights path/to/compatible_weights.pt --output results/single_mask.nii.gz
python -m unittest discover -s tests -v
```

Dobierz wagi i normalizację do zadania modelu. Wagi kanałów z domyślnego
pipeline są przeznaczone do wycinków pojedynczych zębów po normalizacji, a nie
do uzyskania maski zębów z całego surowego CBCT. Dla `.npy` nie ma informacji
o geometrii badania — wynik pojedynczego modelu otrzymuje domyślną geometrię.

