# GSOD Continual Learning Data Preparation

This standalone data preparation project converts NOAA Global Summary of the
Day (GSOD) station records into chronological seasonal tasks for continual
forecasting experiments. It prepares data for a separate model repository; it
does not train FedMixer or another model. NOAA's archive contains one
`YEAR.tar.gz` per year, with station CSV files inside. The archive reader can
extract only the selected stations.

Raw and generated files belong under `data/global_surface/` and are ignored by
Git. The NOAA archive directory can stay outside the repository, for example
`C:\Users\aarav\Downloads\GSOD` on Windows. GSOD and its successor SSOD v2
are different data products; do not mix them within one experiment.

GSOD observations end on 2025-08-28; NOAA stopped updating the product on
2025-08-29 and identifies SSOD v2 as its successor. See the
[NOAA archive index](https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/)
and [NOAA GSOD end-of-life notice](https://www.ncei.noaa.gov/operating-system-upgrade-outage).

The annual `.tar.gz` archives from 1929–2025 sum to about 4.72 GB compressed;
2010–2019 is about 1.04 GB. NOAA's bulk order interface also reports a roughly
40.55 GB output total for its separate bulk delivery. Extracted station CSVs
need additional storage, so download only needed years and extract only the
stations used in the experiment. A GPU is not needed for data preparation.

## Setup

Run commands from this repository's root. Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass`
in that PowerShell window, or call `.\.venv\Scripts\python.exe` instead of
`python`.

### Linux / SSH

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Windows quick start

The folder you mentioned currently has the 1929 archive. That archive is only
about 49 KB and does not contain a full seasonal year for most stations. For a
useful first run, download a recent decade into that same folder. In
PowerShell, from the repository root:

```powershell
python -m pipeline.global_surface.download_archives --start-year 2010 --end-year 2019 --archive-dir "C:\Users\aarav\Downloads\GSOD"
python -m pipeline.global_surface.archive inventory --archive-dir "C:\Users\aarav\Downloads\GSOD" --start-year 2010 --end-year 2019 --output .\data\global_surface\station_inventory.csv
```

Choose station IDs from `data\global_surface\station_inventory.csv` and put
the IDs you want in `pipeline\global_surface\stations.txt`, one ID per line.
Then run:

```powershell
python -m pipeline.global_surface.archive extract --archive-dir "C:\Users\aarav\Downloads\GSOD" --start-year 2010 --end-year 2019 --stations-file .\pipeline\global_surface\stations.txt --raw-dir .\data\global_surface\raw
python -m pipeline.global_surface.season_tasks --raw-dir .\data\global_surface\raw --output-dir .\data\global_surface\processed --features TEMP --input-days 31 --forecast-days 7
```

For a quick low-resource check, use one or two complete years and a few
stations. The pipeline creates only seasons with enough complete station data.

## 1. Get the yearly archives

If you already downloaded archives, point `--archive-dir` at that folder. To
download chosen years directly on this computer or an SSH machine:

```bash
python -m pipeline.global_surface.download_archives \
  --start-year 2010 \
  --end-year 2019 \
  --archive-dir ./data/global_surface/archives
```

Downloads show bytes, rate, and ETA; completed archives are skipped on a rerun
and interrupted `.part` downloads resume when NOAA accepts HTTP Range requests.
Download only the years needed. The NOAA archive index currently lists about
4.72 GB for all 1929–2025 yearly archives combined; the 2010–2019 subset is
about 1.0 GB compressed. Extracted CSVs require additional disk space.

### Download and build every station for a year range

Set the start and end years to the range you want. These commands omit
`--stations-file`, so every station in each yearly archive is extracted. For
all currently archived GSOD years, use 1929 and 2025. This full run needs much
more space than the compressed archives: NOAA's bulk output estimate is about
40.55 GB, and the seasonal outputs need additional room. Plan for at least
50 GB free and check available storage before starting; a full build may need
more depending on generated windows.

PowerShell:

```powershell
$StartYear = 1929
$EndYear = 2025
$ArchiveDir = "C:\Users\aarav\Downloads\GSOD"
$RawDir = ".\data\global_surface\raw"
$ProcessedDir = ".\data\global_surface\processed"
python -m pipeline.global_surface.download_archives --start-year $StartYear --end-year $EndYear --archive-dir $ArchiveDir
python -m pipeline.global_surface.archive inventory --archive-dir $ArchiveDir --start-year $StartYear --end-year $EndYear --output .\data\global_surface\station_inventory.csv
python -m pipeline.global_surface.archive extract --archive-dir $ArchiveDir --start-year $StartYear --end-year $EndYear --raw-dir $RawDir
python -m pipeline.global_surface.season_tasks --raw-dir $RawDir --output-dir $ProcessedDir --features TEMP --input-days 31 --forecast-days 7
```

Linux / SSH:

```bash
START_YEAR=1929
END_YEAR=2025
ARCHIVE_DIR=./data/global_surface/archives
RAW_DIR=./data/global_surface/raw
PROCESSED_DIR=./data/global_surface/processed
python -m pipeline.global_surface.download_archives --start-year "$START_YEAR" --end-year "$END_YEAR" --archive-dir "$ARCHIVE_DIR"
python -m pipeline.global_surface.archive inventory --archive-dir "$ARCHIVE_DIR" --start-year "$START_YEAR" --end-year "$END_YEAR" --output ./data/global_surface/station_inventory.csv
python -m pipeline.global_surface.archive extract --archive-dir "$ARCHIVE_DIR" --start-year "$START_YEAR" --end-year "$END_YEAR" --raw-dir "$RAW_DIR"
python -m pipeline.global_surface.season_tasks --raw-dir "$RAW_DIR" --output-dir "$PROCESSED_DIR" --features TEMP --input-days 31 --forecast-days 7
```

Change only `START_YEAR` and `END_YEAR` (or `$StartYear` and `$EndYear`) to
build a different range. For selected stations, add
`--stations-file ./pipeline/global_surface/stations.txt` to the extraction
command. That reduces extracted CSV storage but the annual archives still
contain every station.

## 2. Inspect station IDs available in your archive folder

This writes the same inventory format as the direct NOAA index inspector:

```bash
python -m pipeline.global_surface.archive inventory \
  --archive-dir "C:\Users\aarav\Downloads\GSOD" \
  --start-year 2010 \
  --end-year 2019 \
  --output ./data/global_surface/station_inventory.csv
```

On an SSH machine, substitute the archive directory, for example
`./data/global_surface/archives`. Choose station IDs from the inventory and
put one 11-digit ID per line in `pipeline/global_surface/stations.txt`.
Keeping the same station list across years gives clients consistent IDs;
stations absent in some archive years simply have no file for those years.
For transfer efficiency, download the archives on the SSH machine rather than
copying the extracted tree from Windows. If transferring manually, copy only
the selected `YEAR.tar.gz` files and the station list.

## 3. Extract selected station files

```bash
python -m pipeline.global_surface.archive extract \
  --archive-dir "C:\Users\aarav\Downloads\GSOD" \
  --start-year 2010 \
  --end-year 2019 \
  --stations-file ./pipeline/global_surface/stations.txt \
  --raw-dir ./data/global_surface/raw
```

If `--stations-file` is omitted, all station CSVs in the chosen archives are
extracted. Existing files are skipped. Extraction reports progress and ETA by
year and station files. Missing yearly archives are reported and skipped.
The archive files themselves stay intact.

## 4. Build seasonal tasks

```bash
python -m pipeline.global_surface.season_tasks \
  --raw-dir ./data/global_surface/raw \
  --output-dir ./data/global_surface/processed \
  --features TEMP \
  --input-days 31 \
  --forecast-days 7
```

Seasonal preprocessing reads one station at a time to keep memory use bounded.
It inserts missing calendar dates, converts GSOD missing-value sentinels to
nulls, and keeps zero readings as valid values. It skips a station/season when
temperature coverage is below 90%, or when the first/last observation falls
more than 7 days inside the season boundary. Short interior gaps of up to two
days are interpolated within a season only. Windows that still contain missing
feature values are excluded.

Each included station/season task produces a cleaned daily CSV and a compressed
NPZ with chronological train/test windows (80/20 by time). `manifest.json`
lists included and skipped station/season pairs. Scaling is not applied here;
fit it on training windows in the training stage. NPZ input arrays have shape
`(number_of_windows, input_days, features)` and target arrays have shape
`(number_of_windows, forecast_days, features)`. Outputs are organized as
`processed/stations/<station_id>/` and `processed/windows/<station_id>/`.

### Example output

For example, a northern station might produce:

```text
data/global_surface/processed/
├── manifest.json
├── stations/
│   └── 03005099999/
│       └── 03005099999_2010_spring.csv
└── windows/
    └── 03005099999/
        └── 03005099999_2010_spring.npz
```

The cleaned CSV contains one row per date plus station ID, season ID, season,
hemisphere, station name, latitude, longitude, elevation, and the selected
weather features. The NPZ contains arrays such as:

```text
train_x: (N_train, 31, 1)   # 31 daily TEMP inputs, Fahrenheit
train_y: (N_train,  7, 1)   # next 7 daily TEMP targets
test_x:  (N_test,  31, 1)
test_y:  (N_test,   7, 1)
features: ['TEMP']
station_id: '03005099999'
season: 'spring'
season_start: '2010-03-01'
season_end: '2010-05-31'
```

`N_train` and `N_test` depend on valid, non-missing windows in that station's
season; the example shapes and station are illustrative.

## Using these tasks for federated continual learning

The intended mapping is:

| Prepared data | Federated continual learning role |
|---|---|
| One station ID | One federated client |
| One season NPZ for a station | That client's task for that seasonal time step |
| Seasons ordered by `season_start` | Continual sequence / communication rounds |
| `train_x`, `train_y` | Local client training data for that task |
| `test_x`, `test_y` | Held-out evaluation for that task |
| A station/season in the manifest marked skipped | Client has no training task for that time step |

At training time, group NPZ files by their season start, sort those starts
chronologically, and pass each season's training windows to the matching
station client. Convert each array batch to tensors; for the default univariate
configuration the model input length is 31 and forecast length is 7. Keep each
client's model and replay memory between seasons. Evaluate on that season's
test arrays after training, before moving to the next season. Do not use test
arrays for optimization or replay.

In the continual trainer's expected structure, one round is a list of client
batch lists:

```python
round_batches[round_index][client_index] = [
    (input_tensor_batch, target_tensor_batch),
    ...,
]
```

For example, a training-side adapter can read and batch one task's training
windows like this:

```python
import numpy as np
import torch


def load_training_batches(npz_path, batch_size=32):
    with np.load(npz_path, allow_pickle=False) as task:
        inputs = task['train_x']
        targets = task['train_y']
    return [
        (torch.from_numpy(inputs[start:start + batch_size]),
         torch.from_numpy(targets[start:start + batch_size]))
        for start in range(0, len(inputs), batch_size)
    ]
```

Use `test_x` and `test_y` only in a separate evaluation pass. In a full
training implementation, build the season rounds by grouping station NPZs with
the same `season_start` and sort the groups by date.

The NPZ files do not directly plug into the Electricity CSV runner. A training
adapter must load each NPZ, batch its training arrays, and assemble this
structure. Also, stations may be skipped for individual seasons by design.
The current `ContinualFedMixer` implementation expects every client in every
round and does not safely handle an inactive client during aggregation. Before
training a set with skipped station/season tasks, the trainer must support an
active-client mask and exclude inactive clients from that round's aggregation;
do not replace missing tasks with zero-valued data. If every selected station
has a task in every round, ordinary all-client rounds can be assembled without
that change.

## Existing direct station download

For a small station/year selection, the original per-station CSV downloader is
still available and avoids annual archives:

```bash
python -m pipeline.global_surface.inspect \
  --start-year 2010 \
  --end-year 2019 \
  --output ./data/global_surface/station_inventory.csv

python -m pipeline.global_surface.download \
  --stations-file ./pipeline/global_surface/stations.txt \
  --start-year 2010 \
  --end-year 2019 \
  --raw-dir ./data/global_surface/raw
```

## Notes

- A station is one client; station records are never merged together.
- Seasons recur in timestamp order across years rather than pooling all winters
  or summers into out-of-order buckets.
- Season names follow hemisphere: Northern Hemisphere winter is DJF;
  Southern Hemisphere winter is JJA. Stations within 23.5 degrees of the
  equator are flagged because four temperature seasons may fit poorly there.
- GSOD temperatures remain in Fahrenheit. Other source units are also retained.
- For transfer to another machine, transfer only the selected year archives
  (or download them on that machine) and the station list. Avoid copying the
  entire extracted station tree if only a subset of stations is needed.
- Raw archives, extracted CSVs, station inventories, and processed outputs are
  local experiment data and should not be committed to Git.
- NOAA's data description is at
  [GSOD dataset description and downloads](https://catalog.data.gov/dataset/global-surface-summary-of-the-day-gsod).

