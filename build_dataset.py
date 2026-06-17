# -*- coding: utf-8 -*-
"""Process the raw multirotor-simulator logs into a clean, documented,
analysis-ready dataset.

Inputs : a directory of raw logs with one sub-folder per condition set
         (baseline, alt180, alt200, speed15, wind5, speed10), each containing
         the simulator's plain-text per-run logs (e.g. ``calm.log``, ``n10.log``).
Outputs: <out>/states/*.parquet, flights.csv, hubs.csv, scenarios.csv,
         DATA_DICTIONARY.md and a copy of this script.

Usage  : python build_dataset.py --raw generation/logs --out processed

Depends only on pandas and pyarrow.
"""
import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

COLS = ['simt', 'id', 'type', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs',
        'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 'drag',
        'phase', 'fuelflow']
DROP = ['type', 'phase', 'fuelflow']          # constant over the whole dataset -> zero information
RENAME = {'simt': 'sim_time_s', 'id': 'drone_id', 'lat': 'lat_deg', 'lon': 'lon_deg', 'alt': 'alt_m',
          'tas': 'tas_ms', 'cas': 'cas_ms', 'vs': 'vs_ms', 'gs': 'gs_ms', 'distflown': 'distflown_m',
          'Temp': 'temp_k', 'trk': 'trk_deg', 'hdg': 'hdg_deg', 'p': 'pressure_pa', 'rho': 'rho_kgm3',
          'thrust': 'thrust_n', 'drag': 'drag_n'}
RD = {'id': 'int32', 'type': 'category', 'lat': 'float64', 'lon': 'float64'}
for c in COLS:
    if c not in RD:
        RD[c] = 'float32'

# condition set -> (cruise speed m/s, cruise altitude ft); the wind is taken from the file name
SETS = {'baseline': (20, 100), 'alt180': (20, 180), 'alt200': (20, 200),
        'speed15': (15, 100), 'wind5': (20, 100), 'speed10': (10, 100)}
# wind label -> (direction deg, speed m/s); 'calm' = no wind, n/e/s/w = wind from N/E/S/W
WIND = {'calm': (0, 0), 'n10': (0, 10), 'e10': (90, 10), 's10': (180, 10), 'w10': (270, 10),
        'n5': (0, 5), 'e5': (90, 5), 's5': (180, 5), 'w5': (270, 5)}
FOLDERS = list(SETS)


def gc(lat1, lon1, lat2, lon2):
    """Great-circle distance (m) on a spherical Earth."""
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse(fol, fn):
    """Scenario name and run parameters from the condition set and the log file name."""
    wl = os.path.splitext(os.path.basename(fn))[0]   # e.g. 'n10', 'calm'
    sp, altft = SETS[fol]
    wd, ws = WIND[wl]
    return f'{fol}__{wl}', sp, wd, ws, altft


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--raw', default=os.path.join(here, 'Datasets'),
                    help='directory of raw logs, one sub-folder per condition set')
    ap.add_argument('--out', default=os.path.join(here, 'processed'),
                    help='output directory for the analysis-ready release')
    args = ap.parse_args()
    ROOT, OUT = args.raw, args.out
    os.makedirs(os.path.join(OUT, 'states'), exist_ok=True)

    flights = []; scen = []
    for fol in FOLDERS:
        for fn in sorted(glob.glob(os.path.join(ROOT, fol, '*.log'))):
            scenario, sp, wd, ws, altft = parse(fol, fn)
            parts = []
            for ch in pd.read_csv(fn, comment='#', header=None, names=COLS, dtype=RD, chunksize=2_000_000):
                ch = ch.drop(columns=DROP).rename(columns=RENAME)
                ch['sim_time_s'] = ch['sim_time_s'].round().astype('int32')
                parts.append(ch)
            df = pd.concat(parts, ignore_index=True)
            df.insert(0, 'scenario', scenario)
            df['cruise_speed_ms'] = np.int16(sp); df['wind_dir_deg'] = np.int16(wd)
            df['wind_speed_ms'] = np.int16(ws); df['cruise_alt_ft'] = np.int16(altft)
            meta = ['scenario', 'cruise_speed_ms', 'wind_dir_deg', 'wind_speed_ms', 'cruise_alt_ft']
            df = df[meta + [c for c in df.columns if c not in meta]]
            pq = os.path.join(OUT, 'states', scenario + '.parquet')
            df.to_parquet(pq, index=False, compression='snappy')
            # per-flight summary
            g = df.groupby('drone_id', sort=True)
            first = g.first(); last = g.last()
            fl = pd.DataFrame({'scenario': scenario, 'cruise_speed_ms': sp, 'wind_dir_deg': wd,
                               'wind_speed_ms': ws, 'cruise_alt_ft': altft,
                               'drone_id': first.index.values,
                               'orig_lat': first['lat_deg'].values, 'orig_lon': first['lon_deg'].values,
                               'dest_lat': last['lat_deg'].values, 'dest_lon': last['lon_deg'].values,
                               't_start_s': g['sim_time_s'].min().values, 't_end_s': g['sim_time_s'].max().values,
                               'duration_s': (g['sim_time_s'].max() - g['sim_time_s'].min()).values,
                               'od_distance_m': gc(first['lat_deg'].values, first['lon_deg'].values,
                                                   last['lat_deg'].values, last['lon_deg'].values),
                               'path_length_m': last['distflown_m'].values,
                               'gs_mean_ms': g['gs_ms'].mean().values, 'gs_max_ms': g['gs_ms'].max().values,
                               'alt_cruise_m': g['alt_m'].median().values})
            flights.append(fl)
            scen.append({'scenario': scenario, 'folder': fol, 'cruise_speed_ms': sp, 'wind_dir_deg': wd,
                         'wind_speed_ms': ws, 'cruise_alt_ft': altft, 'n_drones': int(df['drone_id'].nunique()),
                         'duration_s': int(df['sim_time_s'].max()), 'n_records': len(df),
                         'states_file': 'states/' + scenario + '.parquet',
                         'states_mb': round(os.path.getsize(pq) / 1e6, 2)})
            print('processed %-34s rows=%-8d parquet=%.1fMB' % (scenario, len(df), os.path.getsize(pq) / 1e6))

    flights = pd.concat(flights, ignore_index=True)
    flights.to_csv(os.path.join(OUT, 'flights.csv'), index=False)
    scendf = pd.DataFrame(scen); scendf.to_csv(os.path.join(OUT, 'scenarios.csv'), index=False)
    # hubs (origin set is identical across scenarios; take baseline calm)
    base = flights[flights['scenario'] == 'baseline__calm']
    hubs = base.groupby([base['orig_lat'].round(6), base['orig_lon'].round(6)]).size().reset_index()
    hubs.columns = ['lat', 'lon', 'n_drones']; hubs.insert(0, 'hub_id', range(1, len(hubs) + 1))
    hubs.to_csv(os.path.join(OUT, 'hubs.csv'), index=False)

    total_pq = sum(scendf['states_mb'])
    shutil.copy(os.path.abspath(__file__), os.path.join(OUT, 'build_dataset.py'))

    DD = f"""# Data dictionary — drone-traffic dataset

This is an analysis-ready release of high-density M600 drone-traffic state logs
over Bogota, generated with a fast-time point-mass multirotor simulator (see
01_make_demand.py and 02_simulate.py). Processing (see build_dataset.py): three
constant columns (type, phase, fuelflow) were removed; the per-run operating
parameters that were encoded in the log file names were added as explicit
columns; logs were converted from plain text to columnar Parquet; and two
summary tables were derived.

## states/<scenario>.parquet  ({len(scendf)} files, {total_pq:.0f} MB total)
One row per drone per second.
- scenario (str): <set>__<wind>, e.g. baseline__calm, speed15__n10 (n/e/s/w = wind from N/E/S/W)
- cruise_speed_ms, wind_dir_deg, wind_speed_ms, cruise_alt_ft (int): run parameters
- sim_time_s (int): seconds since scenario start
- drone_id (int): stable within a run
- lat_deg, lon_deg (float64, WGS84); alt_m (float)
- tas_ms, cas_ms, gs_ms, vs_ms (float): airspeeds, ground speed, vertical speed
- distflown_m (float): cumulative distance flown
- temp_k, pressure_pa, rho_kgm3 (float): atmosphere at the drone
- trk_deg, hdg_deg (float): track and heading, clockwise from north
- thrust_n, drag_n (float): modelled thrust and drag

## flights.csv  (one row per scenario x drone, {len(flights)} rows)
scenario, run parameters, drone_id, orig_lat/lon, dest_lat/lon,
t_start_s, t_end_s, duration_s, od_distance_m (great-circle origin->dest),
path_length_m (flown), gs_mean_ms, gs_max_ms, alt_cruise_m.

## hubs.csv  ({len(hubs)} rows)
hub_id, lat, lon, n_drones — the fixed set of origin hubs (identical across runs).

## scenarios.csv  ({len(scendf)} rows)
Run manifest: scenario, folder, parameters, n_drones, duration_s, n_records, states_file, states_mb.

Note: each condition set includes its own calm reference run; all 30 runs are
generated independently (distinct turbulence realisations).
"""
    open(os.path.join(OUT, 'DATA_DICTIONARY.md'), 'w', encoding='utf-8').write(DD)
    print('\nFLIGHTS rows:', len(flights), '| HUBS:', len(hubs), '| total parquet MB: %.0f' % total_pq)
    print('DONE')


if __name__ == '__main__':
    main()
