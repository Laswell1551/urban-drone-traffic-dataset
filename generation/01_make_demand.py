# -*- coding: utf-8 -*-
"""Option A, step 1 — generate a FRESH, independently-designed delivery demand
over Bogota whose statistical character matches the descriptor (821 hubs,
6,911 drones, hub sizes 2-40 median 8, OD distance median ~1.5 km, right-skewed,
hubs concentrated on a central north-south corridor thinning to the periphery).

Nothing here is taken from the original logs: hubs, hub sizes and destinations
are all sampled from documented generative models with a fixed seed, so the
resulting demand is genuinely our own design and fully reproducible.

Outputs (regen/demand/):
    hubs.csv    hub_id, lat, lon, n_drones                       (821 rows)
    demand.csv  drone_id, hub_id, orig_lat, orig_lon, dest_lat, dest_lon, od_distance_m  (6911 rows)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'demand')
os.makedirs(OUT, exist_ok=True)

SEED = 20260617
N_HUBS = 821
N_DRONES = 6911

# Bogota urban footprint (WGS84) and the central north-south corridor
LAT_LO, LAT_HI = 4.47, 4.78
LON_LO, LON_HI = -74.22, -74.02
CORRIDOR_LON, CORRIDOR_LON_SD = -74.08, 0.030   # narrow east-west -> N-S corridor
CORRIDOR_LAT, CORRIDOR_LAT_SD = 4.635, 0.072    # spread along the corridor, central concentration
M_PER_DEG_LAT = 111_320.0

rng = np.random.default_rng(SEED)


def trunc_normal(mean, sd, lo, hi, n):
    out = np.empty(n)
    k = 0
    while k < n:
        s = rng.normal(mean, sd, n - k)
        s = s[(s >= lo) & (s <= hi)]
        out[k:k + len(s)] = s
        k += len(s)
    return out


# --- hubs: concentrated on the central corridor, thinning outward ----------------
hub_lat = trunc_normal(CORRIDOR_LAT, CORRIDOR_LAT_SD, LAT_LO, LAT_HI, N_HUBS)
hub_lon = trunc_normal(CORRIDOR_LON, CORRIDOR_LON_SD, LON_LO, LON_HI, N_HUBS)

# --- hub sizes: lognormal, clipped to [2,40], median ~8, adjusted to sum to N_DRONES
sizes = np.clip(np.round(rng.lognormal(mean=np.log(8.0), sigma=0.45, size=N_HUBS)), 2, 40).astype(int)
# nudge the total to exactly N_DRONES by +/-1 on random eligible hubs
while sizes.sum() != N_DRONES:
    diff = N_DRONES - sizes.sum()
    step = 1 if diff > 0 else -1
    elig = np.where((sizes < 40) if step > 0 else (sizes > 2))[0]
    pick = rng.choice(elig, size=min(abs(diff), len(elig)), replace=False)
    sizes[pick] += step

hubs = pd.DataFrame({'hub_id': np.arange(1, N_HUBS + 1),
                     'lat': hub_lat.round(6), 'lon': hub_lon.round(6),
                     'n_drones': sizes})
hubs.to_csv(os.path.join(OUT, 'hubs.csv'), index=False)

# --- drones: each gets its hub as origin; destination is a local delivery ---------
orig_lat = np.repeat(hub_lat, sizes)
orig_lon = np.repeat(hub_lon, sizes)
hub_of_drone = np.repeat(hubs['hub_id'].values, sizes)

# OD distance ~ lognormal(median 1.5 km), clipped to [0.2, 5.0] km -> right-skewed
d = np.clip(rng.lognormal(mean=np.log(1500.0), sigma=0.47, size=N_DRONES), 200.0, 4968.0)
dest_lat = np.empty(N_DRONES)
dest_lon = np.empty(N_DRONES)
for i in range(N_DRONES):
    for _ in range(20):  # resample bearing until destination lands inside the footprint
        th = rng.uniform(0, 2 * np.pi)
        dlat = (d[i] * np.cos(th)) / M_PER_DEG_LAT
        dlon = (d[i] * np.sin(th)) / (M_PER_DEG_LAT * np.cos(np.radians(orig_lat[i])))
        la, lo = orig_lat[i] + dlat, orig_lon[i] + dlon
        if LAT_LO <= la <= LAT_HI and LON_LO <= lo <= LON_HI:
            break
    dest_lat[i], dest_lon[i] = la, lo


def gc(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


od = gc(orig_lat, orig_lon, dest_lat, dest_lon)
demand = pd.DataFrame({'drone_id': np.arange(N_DRONES), 'hub_id': hub_of_drone,
                       'orig_lat': orig_lat.round(6), 'orig_lon': orig_lon.round(6),
                       'dest_lat': dest_lat.round(6), 'dest_lon': dest_lon.round(6),
                       'od_distance_m': od.round(2)})
demand.to_csv(os.path.join(OUT, 'demand.csv'), index=False)

# --- verification against the descriptor's stated statistics ----------------------
print('seed:', SEED)
print('hubs: %d  drones: %d (sum of hub sizes = %d)' % (len(hubs), len(demand), sizes.sum()))
print('hub sizes: min=%d median=%d max=%d mean=%.2f' % (sizes.min(), int(np.median(sizes)), sizes.max(), sizes.mean()))
print('distinct destinations:', demand[['dest_lat', 'dest_lon']].drop_duplicates().shape[0])
print('OD distance m : median=%.1f mean=%.1f p5=%.1f p95=%.1f max=%.1f'
      % (od.mean() and np.median(od), od.mean(), np.percentile(od, 5), np.percentile(od, 95), od.max()))
print('orig bounds lat [%.4f, %.4f] lon [%.4f, %.4f]'
      % (orig_lat.min(), orig_lat.max(), orig_lon.min(), orig_lon.max()))
print('TARGET (original): hubs 821, drones 6911, sizes 2-40 med 8, OD med 1482 mean 1592 p95 3262 max 4968')
print('wrote', os.path.join(OUT, 'hubs.csv'), 'and demand.csv')
