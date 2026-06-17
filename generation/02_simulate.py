# -*- coding: utf-8 -*-
"""Option A, step 2 -- fast-time point-mass multirotor (DJI Matrice 600) simulator.

Flies the fresh demand (regen/demand/demand.csv) through the 30-run OFAT sweep and
writes per-second state logs in the SAME plain-text column format as the original
raw logs, so processed/build_dataset.py runs unchanged on the output.

Time-stepped flight model (1 s step, fully reproducible via --seed):
  * Each drone launches at t=0 from its hub at 10 ft, at rest, and flies to its
    destination, accelerating at A_ACCEL to the commanded cruise airspeed and
    decelerating to land.
  * Guidance is pure pursuit: the commanded air-velocity points straight at the
    destination; ground velocity = air velocity + mean wind + turbulence. Because
    the drone does not perfectly crab, a steady cross/along wind bends the ground
    track, and the turbulence keeps the velocity wandering, so the trajectories
    are realistically non-trivial to predict (unlike a perfect straight line).
  * Mean wind is spatially uniform (the swept OFAT parameter); turbulence is a
    small, zero-mean, temporally-correlated (Ornstein-Uhlenbeck) gust added per
    drone, so the marginal effect of the mean wind is still isolated.
  * Vertical: climb at +5 m/s to the cruise altitude, level cruise, descend at
    -5 m/s timed to land on arrival.
  * Atmosphere: ISA referenced to Bogota field elevation (FIELD_ELEV_M=2640 m).
  * Forces: drag = 0.5*rho*Va^2*CdA; thrust = sqrt((m g)^2 + drag^2).

Usage:
  python 02_simulate.py                       # all 30 runs -> regen/logs/<set>/*.log
  python 02_simulate.py --only baseline__calm
  python 02_simulate.py --maxdrones 300
"""
import os
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEMAND = os.path.join(HERE, 'demand', 'demand.csv')
LOGDIR = os.path.join(HERE, 'logs')

# --- M600 / environment constants ------------------------------------------------
LAUNCH_ALT_M = 3.048          # 10 ft launch height
VS = 5.0                      # climb / descent rate (m/s)
A_ACCEL = 2.0                 # horizontal accel/decel (m/s^2)
V_FLOOR = 3.0                 # min commanded airspeed on final approach (keeps wind authority)
FIELD_ELEV_M = 2640.0         # Bogota plateau; set 0.0 for a sea-level reference
MASS_KG = 15.0                # M600 with payload (~MTOW)
G = 9.80665
CDA = 0.28                    # drag area Cd*A (m^2)
GUST_SIGMA = 0.8              # turbulence std (m/s) per horizontal component (light)
GUST_RHO = 0.85               # OU correlation per second (~6 s timescale)
ARRIVE_M = 8.0                # landed when within this distance of destination
MAXT = 2600                   # safety cap on steps per run
LAT0, LON0 = 4.65, -74.08
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))

T0, P0, L, R = 288.15, 101325.0, 0.0065, 287.058
RHO_SL = P0 / (R * T0)

WINDS10 = [(0, 0), (0, 10), (90, 10), (180, 10), (270, 10)]
WINDS5 = [(0, 0), (0, 5), (90, 5), (180, 5), (270, 5)]
DESIGN = [
    ('baseline', 20, 100, WINDS10),
    ('alt180',   20, 180, WINDS10),
    ('alt200',   20, 200, WINDS10),
    ('speed15',  15, 100, WINDS10),
    ('wind5',    20, 100, WINDS5),
    ('speed10',  10, 100, WINDS5),
]
_DIR = {0: 'n', 90: 'e', 180: 's', 270: 'w'}   # wind FROM north/east/south/west


def wlabel(wd, ws):
    return 'calm' if ws == 0 else f'{_DIR[wd]}{ws}'
COLS = ['simt', 'id', 'type', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs',
        'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 'drag', 'phase', 'fuelflow']


def isa(alt_agl):
    h = FIELD_ELEV_M + alt_agl
    T = T0 - L * h
    P = P0 * (T / T0) ** (G / (R * L))
    return T, P, P / (R * T)


def simulate(demand, speed, alt_ft, wd, ws, seed):
    n = len(demand)
    rng = np.random.default_rng(seed)
    cruise_alt = alt_ft * 0.3048
    Va = float(speed)
    v_floor = max(V_FLOOR, ws + 2.0)   # keep airspeed above the wind so drones always reach their goal

    x = (demand['orig_lon'].values - LON0) * M_PER_DEG_LON
    y = (demand['orig_lat'].values - LAT0) * M_PER_DEG_LAT
    gx = (demand['dest_lon'].values - LON0) * M_PER_DEG_LON
    gy = (demand['dest_lat'].values - LAT0) * M_PER_DEG_LAT
    did = demand['drone_id'].values

    alt = np.full(n, LAUNCH_ALT_M)
    vcmd = np.zeros(n)
    distflown = np.zeros(n)
    descending = np.zeros(n, bool)
    done = np.zeros(n, bool)
    gust = np.zeros((n, 2))
    wE, wN = -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))

    cols = {c: [] for c in COLS}
    t = 0
    while (not done.all()) and t < MAXT:
        act = ~done
        rx, ry = gx - x, gy - y
        dist = np.hypot(rx, ry)
        ux, uy = rx / np.maximum(dist, 1e-6), ry / np.maximum(dist, 1e-6)

        # speed schedule: decelerate inside braking distance, else accelerate to cruise
        brake = vcmd * vcmd / (2 * A_ACCEL)
        dec = dist <= brake
        vcmd = np.where(dec, np.maximum(vcmd - A_ACCEL, v_floor), np.minimum(vcmd + A_ACCEL, Va))

        # pure-pursuit air velocity + mean wind + OU turbulence
        gust = GUST_RHO * gust + GUST_SIGMA * np.sqrt(1 - GUST_RHO ** 2) * rng.standard_normal((n, 2))
        avx, avy = vcmd * ux, vcmd * uy
        vgx = avx + wE + gust[:, 0]
        vgy = avy + wN + gust[:, 1]
        gs = np.hypot(vgx, vgy)

        # vertical profile
        descending |= (dist <= np.maximum(gs, 1.0) * (cruise_alt / VS))
        alt_new = np.where(descending, np.maximum(alt - VS, 0.0),
                           np.minimum(alt + VS, cruise_alt))
        vs = np.clip(alt_new - alt, -VS, VS)

        T, P, rho = isa(alt_new)
        cas = vcmd * np.sqrt(rho / RHO_SL)
        drag = 0.5 * rho * vcmd * vcmd * CDA
        thrust = np.sqrt((MASS_KG * G) ** 2 + drag * drag)
        trk = np.degrees(np.arctan2(vgx, vgy)) % 360.0
        hdg = np.degrees(np.arctan2(avx, avy)) % 360.0

        a = act  # log every currently-active drone this second
        cols['simt'].append(np.full(a.sum(), t, dtype=float))
        cols['id'].append(did[a])
        cols['type'].append(np.full(a.sum(), 'M600'))
        cols['lat'].append(LAT0 + y[a] / M_PER_DEG_LAT)
        cols['lon'].append(LON0 + x[a] / M_PER_DEG_LON)
        cols['alt'].append(alt_new[a])
        cols['tas'].append(vcmd[a]); cols['cas'].append(cas[a])
        cols['vs'].append(vs[a]); cols['gs'].append(gs[a])
        cols['distflown'].append((distflown + gs)[a])
        cols['Temp'].append(T[a]); cols['trk'].append(trk[a]); cols['hdg'].append(hdg[a])
        cols['p'].append(P[a]); cols['rho'].append(rho[a])
        cols['thrust'].append(thrust[a]); cols['drag'].append(drag[a])
        cols['phase'].append(np.zeros(a.sum())); cols['fuelflow'].append(np.zeros(a.sum()))

        # advance state for active drones
        x = np.where(act, x + vgx, x)
        y = np.where(act, y + vgy, y)
        alt = np.where(act, alt_new, alt)
        distflown = np.where(act, distflown + gs, distflown)
        done |= act & (dist <= np.maximum(ARRIVE_M, gs))
        t += 1

    out = pd.DataFrame({c: np.concatenate(cols[c]) for c in COLS})
    out['id'] = out['id'].astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--maxdrones', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260617)
    args = ap.parse_args()

    demand = pd.read_csv(DEMAND)
    if args.maxdrones:
        demand = demand.iloc[:args.maxdrones].copy()

    total = 0
    for si, (folder, speed, alt_ft, winds) in enumerate(DESIGN):
        for wi, (wd, ws) in enumerate(winds):
            wl = wlabel(wd, ws)
            scenario = f'{folder}__{wl}'
            if args.only and args.only not in scenario:
                continue
            df = simulate(demand, speed, alt_ft, wd, ws, args.seed + 100 * si + wi)
            d = os.path.join(LOGDIR, folder); os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f'{wl}.log')
            with open(path, 'w', newline='') as fh:
                fh.write('# M600 fast-time simulation -- ' + scenario + '\n')
                fh.write('# ' + ', '.join(COLS) + '\n')
                df.to_csv(fh, header=False, index=False, float_format='%.8f')
            total += len(df)
            print('%-30s rows=%-8d -> %s' % (scenario, len(df), os.path.relpath(path, HERE)))
    print('TOTAL rows:', f'{total:,}')


if __name__ == '__main__':
    main()
