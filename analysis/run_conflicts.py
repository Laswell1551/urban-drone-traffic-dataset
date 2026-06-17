# -*- coding: utf-8 -*-
"""Conflict / loss-of-separation (LoS) analysis -- utility demonstration #1.

For every run and every second, the pairwise horizontal separation between all
EN-ROUTE drones (those having flown > 100 m from their hub, which excludes
co-located drones still on shared launch pads) is computed with a KD-tree.
A loss of separation is recorded under an illustrative criterion of
< 50 m horizontal AND < 15 m vertical separation.

Per-run metrics:
    peak           max simultaneous LoS pair count over time
    total_los_s    sum over time of LoS pair counts (separation-loss-seconds)
    unique_pairs   distinct drone pairs that lose separation at any time

Outputs (analysis/out/):
    conflicts_per_run.csv        one row per run + the three metrics
    conflicts_timeseries.csv     LoS count per second (for the time figure)
    concurrency_timeseries.csv   drones airborne per second
    conflict_hotspot_baseline.npz  2-D histogram of LoS locations (calm run)
"""
import os, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST   = os.path.join(ROOT, 'processed', 'states')
OUT  = os.path.join(ROOT, 'analysis', 'out'); os.makedirs(OUT, exist_ok=True)

H_SEP, V_SEP, ENROUTE = 50.0, 15.0, 100.0
LAT0, LON0 = 4.62, -74.10
MX = np.cos(np.radians(LAT0)) * 111320.0; MY = 110540.0
# hotspot grid (degrees) over the urban footprint
LAT_EDGES = np.linspace(4.47, 4.78, 301)
LON_EDGES = np.linspace(-74.22, -74.02, 301)

SET_OF = {'baseline':'baseline','alt180':'alt180','alt200':'alt200',
          'speed15':'speed15','wind5':'wind5','speed10':'speed10'}

def analyse(path, want_hotspot=False, want_series=False):
    df = pd.read_parquet(path, columns=['sim_time_s','drone_id','lat_deg','lon_deg','alt_m','distflown_m'])
    scn = os.path.basename(path).replace('.parquet','')
    x = (df['lon_deg'].to_numpy() - LON0) * MX
    y = (df['lat_deg'].to_numpy() - LAT0) * MY
    alt = df['alt_m'].to_numpy(); enr = df['distflown_m'].to_numpy() > ENROUTE
    tt = df['sim_time_s'].to_numpy(); did = df['drone_id'].to_numpy()
    order = np.argsort(tt, kind='stable')
    tt, x, y, alt, enr, did = tt[order], x[order], y[order], alt[order], enr[order], did[order]
    bounds = np.searchsorted(tt, np.arange(tt.min(), tt.max()+1))
    bounds = np.append(bounds, len(tt))
    peak = 0; total = 0; unique = set()
    series = []; airborne = []
    hot_lat = []; hot_lon = []
    for s in range(len(bounds)-1):
        a, b = bounds[s], bounds[s+1]
        airborne.append((s, b - a))
        m = enr[a:b]
        if m.sum() < 2:
            series.append((s, 0)); continue
        xi, yi, ai, di = x[a:b][m], y[a:b][m], alt[a:b][m], did[a:b][m]
        tree = cKDTree(np.column_stack([xi, yi]))
        pairs = tree.query_pairs(H_SEP, output_type='ndarray')
        if len(pairs) == 0:
            series.append((s, 0)); continue
        dz = np.abs(ai[pairs[:,0]] - ai[pairs[:,1]])
        los = pairs[dz < V_SEP]
        c = len(los)
        series.append((s, c)); total += c; peak = max(peak, c)
        p0 = di[los[:,0]]; p1 = di[los[:,1]]
        lo = np.minimum(p0, p1); hi = np.maximum(p0, p1)
        unique.update(zip(lo.tolist(), hi.tolist()))
        if want_hotspot and c:
            mx = (xi[los[:,0]] + xi[los[:,1]]) / 2; my = (yi[los[:,0]] + yi[los[:,1]]) / 2
            hot_lat.append(my / MY + LAT0); hot_lon.append(mx / MX + LON0)
    res = dict(scenario=scn, peak=int(peak), total_los_s=int(total), unique_pairs=int(len(unique)))
    extra = {}
    if want_series:
        extra['series'] = pd.DataFrame(series, columns=['sim_time_s','los_count']).assign(scenario=scn)
        extra['air'] = pd.DataFrame(airborne, columns=['sim_time_s','n_airborne']).assign(scenario=scn)
    if want_hotspot and hot_lat:
        H, _, _ = np.histogram2d(np.concatenate(hot_lat), np.concatenate(hot_lon),
                                 bins=[LAT_EDGES, LON_EDGES])
        extra['hot'] = H
    return res, extra

rows = []; series_all = []; air_all = []
files = sorted(glob.glob(os.path.join(ST, '*.parquet')))
for path in files:
    fol = os.path.basename(path).split('__')[0]
    cset = SET_OF[fol]
    is_base = (fol == 'baseline')
    is_calm = path.endswith('baseline__calm.parquet')
    res, extra = analyse(path, want_hotspot=is_calm, want_series=is_base)
    res['condition'] = cset
    rows.append(res)
    if 'series' in extra: series_all.append(extra['series']); air_all.append(extra['air'])
    if 'hot' in extra:
        np.savez_compressed(os.path.join(OUT,'conflict_hotspot_baseline.npz'),
                            H=extra['hot'], lat_edges=LAT_EDGES, lon_edges=LON_EDGES)
    print('%-32s peak=%-6d total=%-8d unique=%-6d' % (res['scenario'],res['peak'],res['total_los_s'],res['unique_pairs']))

per_run = pd.DataFrame(rows)
# attach run parameters from scenarios.csv
sc = pd.read_csv(os.path.join(ROOT,'processed','scenarios.csv'))
per_run = per_run.merge(sc[['scenario','cruise_speed_ms','wind_dir_deg','wind_speed_ms','cruise_alt_ft']], on='scenario')
per_run.to_csv(os.path.join(OUT,'conflicts_per_run.csv'), index=False)
pd.concat(series_all).to_csv(os.path.join(OUT,'conflicts_timeseries.csv'), index=False)
pd.concat(air_all).to_csv(os.path.join(OUT,'concurrency_timeseries.csv'), index=False)

# per-condition-set means (Table 3)
agg = (per_run.groupby('condition')
       .agg(cruise_speed_ms=('cruise_speed_ms','first'), cruise_alt_ft=('cruise_alt_ft','first'),
            mean_peak=('peak','mean'), mean_total=('total_los_s','mean'),
            mean_unique=('unique_pairs','mean'))
       .reset_index())
agg.to_csv(os.path.join(OUT,'conflicts_per_condition.csv'), index=False)
print('\n=== per-condition-set means (Table 3) ===')
print(agg.to_string(index=False))
print('\nDONE')
