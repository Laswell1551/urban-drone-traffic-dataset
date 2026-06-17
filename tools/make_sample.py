# -*- coding: utf-8 -*-
"""Regenerate the committed sample/ from the full release in processed/.

Selects a fixed, seeded subset of drones across four scenarios spanning the
three OFAT axes. Requires the full dataset to be present in processed/.

    python tools/make_sample.py
"""
import os
import shutil
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = os.path.join(REPO, 'processed', 'states')
SAMP = os.path.join(REPO, 'sample')
os.makedirs(os.path.join(SAMP, 'states'), exist_ok=True)

N_DRONES = 200
SEED = 42
SAMPLE_SCENARIOS = [
    'baseline__calm',    # baseline, calm
    'baseline__n10',   # baseline + 10 m/s wind from north -> wind axis
    'speed15__calm',       # cruise speed 15 m/s, calm          -> speed axis
    'alt200__calm',           # cruise altitude 200 ft, calm       -> altitude axis
]

base = pd.read_parquet(os.path.join(ST, 'baseline__calm.parquet'), columns=['drone_id'])
ids = np.sort(base['drone_id'].unique())
keep = set(np.sort(np.random.default_rng(SEED).choice(ids, size=N_DRONES, replace=False)).tolist())
print('sampled %d of %d drones (seed=%d)' % (len(keep), len(ids), SEED))

for sc in SAMPLE_SCENARIOS:
    df = pd.read_parquet(os.path.join(ST, sc + '.parquet'))
    sub = df[df['drone_id'].isin(keep)].reset_index(drop=True)
    sub.to_parquet(os.path.join(SAMP, 'states', sc + '.parquet'), index=False, compression='snappy')
    print('  %-28s rows=%d' % (sc, len(sub)))

fl = pd.read_csv(os.path.join(REPO, 'processed', 'flights.csv'))
fl[fl['scenario'].isin(SAMPLE_SCENARIOS) & fl['drone_id'].isin(keep)].to_csv(
    os.path.join(SAMP, 'flights_sample.csv'), index=False)
shutil.copy(os.path.join(REPO, 'processed', 'hubs.csv'), os.path.join(SAMP, 'hubs.csv'))
shutil.copy(os.path.join(REPO, 'processed', 'scenarios.csv'), os.path.join(SAMP, 'scenarios.csv'))
print('sample rebuilt at', SAMP)
