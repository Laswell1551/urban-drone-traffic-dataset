# Analysis

Code that reproduces the figures and tables of the Data Descriptor from the
released data. These scripts operate on the **full** dataset.

## Data path

Each script computes `ROOT` as the repository root and reads from
`ROOT/processed/`. Download the full release and place its contents there:

```
processed/
├── states/                # 30 *.parquet files
├── flights.csv
├── hubs.csv
└── scenarios.csv
```

(The small `sample/` is for `examples/quickstart.py`, not for these scripts.)

## Scripts

| Script | Produces |
|---|---|
| `run_conflicts.py` | per-run loss-of-separation metrics, time series, spatial hotspot (`out/`) |
| `run_prediction_benchmark.py` | trajectory-prediction ADE/FDE for the four benchmark methods (`out/`) |
| `make_figures.py` | the descriptor's figures (`out/`) |
| `make_tables.py` | the descriptor's LaTeX tables (`out/tables/`) |

```bash
pip install -r ../requirements.txt
python run_conflicts.py
python run_prediction_benchmark.py
python make_figures.py
python make_tables.py
```

Outputs are written to `analysis/out/` (git-ignored, regenerable).

**Definitions.** Conflicts use an illustrative loss-of-separation criterion of
< 50 m horizontal and < 15 m vertical between en-route drones (those having
flown > 100 m from their hub). Prediction observes 5 s and predicts 10 s, scored
by average and final displacement error (ADE/FDE) in a local metric frame. Both
are recomputable from the released per-second positions with any parameters.
