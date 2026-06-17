# -*- coding: utf-8 -*-
"""Quickstart: load the bundled sample, print a summary, and make two plots
(ground tracks, and the calm-vs-wind ground-speed split). Runs out of the box
against sample/ -- no full download needed.

    python examples/quickstart.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(os.path.dirname(HERE), 'sample')
ST = os.path.join(SAMPLE, 'states')

# ---- the experimental design -------------------------------------------------
scen = pd.read_csv(os.path.join(SAMPLE, 'scenarios.csv'))
print('Full design: %d runs across %d condition sets' % (len(scen), scen['folder'].nunique()))
print(scen[['scenario', 'cruise_speed_ms', 'cruise_alt_ft', 'wind_dir_deg', 'wind_speed_ms', 'n_records']].head(8).to_string(index=False))

# ---- load two sample scenarios -----------------------------------------------
calm = pd.read_parquet(os.path.join(ST, 'baseline__calm.parquet'))
wind = pd.read_parquet(os.path.join(ST, 'baseline__n10.parquet'))  # 10 m/s from north
print('\nbaseline calm : %d drones, %d state rows, t in [%d, %d] s'
      % (calm.drone_id.nunique(), len(calm), calm.sim_time_s.min(), calm.sim_time_s.max()))
print('baseline wind : %d drones, %d state rows' % (wind.drone_id.nunique(), len(wind)))

# ---- figure ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# (a) ground tracks of up to 60 sampled drones, calm baseline
for did, g in calm.groupby('drone_id'):
    ax1.plot(g['lon_deg'], g['lat_deg'], lw=0.6, alpha=0.6)
    if did >= sorted(calm.drone_id.unique())[min(59, calm.drone_id.nunique() - 1)]:
        break
ax1.set_title('Ground tracks (sample, baseline calm)')
ax1.set_xlabel('longitude (deg)'); ax1.set_ylabel('latitude (deg)')
ax1.set_aspect(1 / np.cos(np.radians(4.65)))

# (b) ground-speed distribution: calm (unimodal at cruise) vs north wind (head/tail split)
bins = np.linspace(0, 30, 61)
ax2.hist(calm['gs_ms'], bins=bins, alpha=0.6, label='calm', density=True)
ax2.hist(wind['gs_ms'], bins=bins, alpha=0.6, label='10 m/s wind from N', density=True)
ax2.axvline(20, color='k', ls='--', lw=1, label='cruise 20 m/s')
ax2.set_title('Ground speed: wind splits it into head/tail-wind modes')
ax2.set_xlabel('ground speed (m/s)'); ax2.set_ylabel('density'); ax2.legend()

fig.tight_layout()
out = os.path.join(HERE, 'quickstart_output.png')
fig.savefig(out, dpi=130)
print('\nsaved figure ->', out)
