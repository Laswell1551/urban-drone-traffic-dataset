# Urban drone traffic trajectories under varying wind and operating conditions

Analysis-ready, self-describing per-second state trajectories for thousands of
concurrently operating delivery drones over the metropolitan area of Bogotá,
Colombia. The traffic is generated with a **fast-time, point-mass multirotor
simulator** (released here) using a performance model representative of the DJI
Matrice 600. A single fixed traffic demand is propagated through a
one-factor-at-a-time (OFAT) sweep of cruise speed, cruise altitude, wind speed
and wind direction around a common baseline.

This repository contains the full generation and processing code, the analysis
code, documentation, and a small **~4 MB sample**. The complete dataset
(30 runs, 207,330 flights, 30,583,133 state records, ≈1.19 GB) is archived in a
formal repository (see [The full dataset](#the-full-dataset)).

| | |
|---|---|
| Runs | 30 (6 condition sets × 5 wind cases) |
| Flights | 207,330 |
| State records | 30,583,133 (one row per drone per second) |
| Concurrent drones per run | 6,911 |
| Origin hubs | 821 |
| Vehicle model | DJI Matrice 600 multirotor (`M600`) |
| Area | Bogotá, Colombia (≈4.47–4.78°N, −74.22–−74.02°W, WGS84) |
| Atmosphere | ISA referenced to the ≈2,640 m plateau (density ≈0.94 kg/m³) |
| Format | Apache Parquet (states) + CSV (summaries) |
| Full size | ≈1.19 GB |

## How the data are generated

Everything is reproducible from the released code, in three stages:

```
generation/01_make_demand.py   # fresh Bogotá delivery demand: 821 hubs + 6,911 O–D pairs (seeded)
        ↓ demand/
generation/02_simulate.py      # M600 fast-time simulator → per-second raw state logs
        ↓ logs/
build_dataset.py               # raw logs → analysis-ready Parquet + flights/hubs/scenarios tables
        ↓ processed/
analysis/                      # reproduces the paper's figures and tables
```

The simulator advances each drone on a 1 s step: it launches at 10 ft, accelerates
to the commanded cruise airspeed, transits to its destination under pure-pursuit
guidance, and decelerates to land. The ground velocity is the air velocity plus a
spatially uniform mean wind (the swept OFAT parameter) plus a small, zero-mean,
temporally-correlated turbulence term, so trajectories are realistically variable
while the marginal effect of the mean wind remains isolated. Atmosphere follows the
International Standard Atmosphere referenced to Bogotá's field elevation; thrust and
drag are computed from a documented M600 point-mass model. See `generation/` for
the exact constants and `DATA_DICTIONARY.md` for the output schema.

## Repository contents

```
urban-drone-traffic-dataset/
├── README.md
├── LICENSE                   # data: CC-BY-4.0 | code: MIT
├── CITATION.cff
├── DATA_DICTIONARY.md
├── requirements.txt
├── generation/
│   ├── 01_make_demand.py     # fresh demand (hubs + O–D pairs)
│   └── 02_simulate.py        # M600 fast-time simulator
├── build_dataset.py          # raw logs → analysis-ready release
├── analysis/                 # reproduces the figures and tables
│   ├── run_conflicts.py  run_prediction_benchmark.py  make_figures.py  make_tables.py
│   └── README.md
├── examples/quickstart.py    # loads the sample, prints a summary, plots a trajectory
├── sample/                   # ~4 MB self-consistent subset (committed to git)
│   ├── states/               # 4 scenarios × 200 drones
│   ├── flights_sample.csv  hubs.csv  scenarios.csv  DATA_DICTIONARY.md
└── processed/                # full release goes here (git-ignored; download separately)
```

## The full dataset

The full release is hosted at:

> **https://doi.org/10.5281/zenodo.20730170**

with a flat structure: `states/` (30 per-run Parquet files), `flights.csv`
(207,330 rows), `hubs.csv` (821 hubs), `scenarios.csv` (30-run manifest),
`DATA_DICTIONARY.md`, and the generation + processing code. To run the `analysis/`
scripts, download the full release into `processed/` at the repository root.

## Condition sets (OFAT design)

A single fixed demand of 6,911 flights is reused in every run; only the swept
parameter changes, so any difference between runs is attributable to it.

| Condition set | Cruise speed (m/s) | Cruise alt (ft) | Wind speed (m/s) | Wind dir (°) | Runs |
|---|---|---|---|---|---|
| `baseline` | 20 | 100 | 0 / 10 | 0/90/180/270 | 5 |
| `alt180` | 20 | 180 | 0 / 10 | 0/90/180/270 | 5 |
| `alt200` | 20 | 200 | 0 / 10 | 0/90/180/270 | 5 |
| `speed15` | 15 | 100 | 0 / 10 | 0/90/180/270 | 5 |
| `wind5` | 20 | 100 | 0 / 5 | 0/90/180/270 | 5 |
| `speed10` | 10 | 100 | 0 / 5 | 0/90/180/270 | 5 |

Scenario file names encode the parameters: `<set>__<wind>`, where `wind` is `calm`
or a compass letter plus speed (n/e/s/w = wind from N/E/S/W) — e.g. `alt200__e10`
is the altitude-200 set with wind from the east at 10 m/s.

## State schema (22 columns)

| Group | Columns |
|---|---|
| Identity / index | `scenario`, `drone_id`, `sim_time_s` |
| Run parameters | `cruise_speed_ms`, `wind_dir_deg`, `wind_speed_ms`, `cruise_alt_ft` |
| Position | `lat_deg`, `lon_deg` (float64, WGS84), `alt_m` |
| Kinematics | `tas_ms`, `cas_ms`, `gs_ms`, `vs_ms`, `distflown_m`, `trk_deg`, `hdg_deg` |
| Atmosphere | `temp_k`, `pressure_pa`, `rho_kgm3` |
| Forces | `thrust_n`, `drag_n` |

The four run parameters are attached to **every** record, so each row is
self-describing. See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for full definitions.

## Quickstart

```bash
pip install -r requirements.txt
python examples/quickstart.py          # runs against sample/ out of the box
```

```python
import pandas as pd
df = pd.read_parquet("sample/states/baseline__calm.parquet")
flights = pd.read_csv("sample/flights_sample.csv")   # joins on (scenario, drone_id)
```

## Reproducing the dataset, figures and tables

```bash
python generation/01_make_demand.py                  # → generation/demand/
python generation/02_simulate.py                     # → generation/logs/  (all 30 runs)
python build_dataset.py --raw generation/logs --out processed
python analysis/run_conflicts.py               # after placing the full data in processed/
python analysis/run_prediction_benchmark.py
python analysis/make_figures.py
python analysis/make_tables.py
```

## How to cite

Please cite the Data Descriptor and this dataset:

```
Lin, J., Yan, S., Zhang, S. & Peng, M. A dataset of simulated urban drone traffic
trajectories under varying wind and operating conditions. Scientific Data (in review).

Lin, J., Yan, S., Zhang, S. & Peng, M. Urban drone traffic trajectories under varying
wind and operating conditions. Zenodo https://doi.org/10.5281/zenodo.20730170 (2026).
```

See [`CITATION.cff`](CITATION.cff).

## License

Data are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
and code under the MIT License; see [`LICENSE`](LICENSE).

## Acknowledgements

Processing and analysis use pandas, pyarrow, NumPy, SciPy, scikit-learn and matplotlib.
