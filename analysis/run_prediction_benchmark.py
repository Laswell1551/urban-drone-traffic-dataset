# -*- coding: utf-8 -*-
"""Trajectory-prediction benchmark for the drone-traffic dataset.

Task: observe T_OBS seconds of horizontal position, predict the next T_PRED
seconds. Four methods spanning physics-based -> learned:
    CV     constant-velocity extrapolation (no training)
    KF     constant-velocity Kalman filter  (no training)
    Ridge  linear seq2seq regression        (trained on baseline)
    MLP    multilayer perceptron            (trained on baseline)
Metric: ADE / FDE in metres, averaged over test windows.
Generalisation design: the learned models are trained ONLY on the baseline
(baseline) condition and evaluated on every condition set, so the table
shows how prediction error transfers across the OFAT sweep.

Outputs (analysis/out/):
    prediction_benchmark.csv     method x condition  ADE/FDE
    prediction_examples.npz      a few (obs,true,preds) windows for figures
"""
import os, glob, numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST   = os.path.join(ROOT, 'processed', 'states')
OUT  = os.path.join(ROOT, 'analysis', 'out'); os.makedirs(OUT, exist_ok=True)

T_OBS, T_PRED = 5, 10            # observe 5 s, predict 10 s (1 Hz)
RNG = np.random.default_rng(42)

# condition sets -> their five run files (folder prefix)
SETS = {
    'baseline'      : 'baseline',
    'alt180'       : 'alt180',
    'alt200'       : 'alt200',
    'speed15'   : 'speed15',
    'wind5'        : 'wind5',
    'speed10' : 'speed10',
}
LAT0, LON0 = 4.62, -74.10        # projection reference near Bogota
MX = np.cos(np.radians(LAT0)) * 111320.0
MY = 110540.0

def to_xy(lat, lon):
    return np.column_stack([(lon - LON0) * MX, (lat - LAT0) * MY]).astype(np.float32)

def windows_from_run(path, max_drones):
    """Return canonical-frame (obs, fut) arrays for sampled drones of one run.
    Frame: translate so last observed point is origin, rotate so the observed
    mean heading points to +x (rotation-invariant for ADE/FDE)."""
    df = pd.read_parquet(path, columns=['drone_id', 'sim_time_s', 'lat_deg', 'lon_deg'])
    ids = df['drone_id'].unique()
    if len(ids) > max_drones:
        ids = RNG.choice(ids, max_drones, replace=False)
    df = df[df['drone_id'].isin(ids)].sort_values(['drone_id', 'sim_time_s'])
    obs_l, fut_l = [], []
    L = T_OBS + T_PRED
    for _, g in df.groupby('drone_id', sort=False):
        t = g['sim_time_s'].to_numpy()
        P = to_xy(g['lat_deg'].to_numpy(), g['lon_deg'].to_numpy())
        # split into contiguous (1 s step) segments
        brk = np.where(np.diff(t) != 1)[0] + 1
        for seg in np.split(np.arange(len(t)), brk):
            if len(seg) < L:
                continue
            Q = P[seg]
            sw = sliding_window_view(Q, (L, 2)).reshape(-1, L, 2)  # (nwin,L,2)
            step = T_PRED                                          # non-overlapping
            sw = sw[::step]
            for w in sw:
                o, f = w[:T_OBS], w[T_OBS:]
                p0 = o[-1]
                head = o[-1] - o[0]
                ang = np.arctan2(head[1], head[0])
                c, s = np.cos(-ang), np.sin(-ang)
                R = np.array([[c, -s], [s, c]], np.float32)
                obs_l.append(((o - p0) @ R.T))
                fut_l.append(((f - p0) @ R.T))
    if not obs_l:
        return np.empty((0, T_OBS, 2), np.float32), np.empty((0, T_PRED, 2), np.float32)
    return np.asarray(obs_l, np.float32), np.asarray(fut_l, np.float32)

def collect(prefix, max_drones):
    o, f = [], []
    for p in sorted(glob.glob(os.path.join(ST, prefix + '__*.parquet'))):
        oo, ff = windows_from_run(p, max_drones)
        if len(oo):
            o.append(oo); f.append(ff)
    return np.concatenate(o), np.concatenate(f)

# ---- methods -----------------------------------------------------------------
def pred_cv(obs):
    """constant velocity from the observed window."""
    v = (obs[:, -1] - obs[:, 0]) / (T_OBS - 1)             # (n,2)
    k = np.arange(1, T_PRED + 1, dtype=np.float32)[:, None]
    return obs[:, -1][:, None, :] + v[:, None, :] * k[None]  # (n,T_PRED,2)

def pred_kf(obs):
    """constant-velocity Kalman filter per window, then forecast. Measurement
    noise is small because the simulator output is near-noiseless, and the
    velocity is initialised from the first finite difference."""
    n = len(obs); out = np.empty((n, T_PRED, 2), np.float32)
    dt = 1.0
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    H = np.array([[1,0,0,0],[0,1,0,0]], float)
    Q = np.diag([1e-3, 1e-3, 0.2, 0.2]); R = np.eye(2) * 0.05
    for i in range(n):
        v0 = obs[i,1] - obs[i,0]
        x = np.array([obs[i,0,0], obs[i,0,1], v0[0], v0[1]], float)
        P = np.diag([0.05, 0.05, 1.0, 1.0])
        for z in obs[i][1:]:
            x = F @ x; P = F @ P @ F.T + Q
            y = z - H @ x; S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(4) - K @ H) @ P
        for kk in range(T_PRED):
            x = F @ x; out[i, kk] = x[:2]
    return out

def fit_learned(model, Xtr, Ytr):
    model.fit(Xtr.reshape(len(Xtr), -1), Ytr.reshape(len(Ytr), -1))
    return model

def pred_learned(model, obs):
    p = model.predict(obs.reshape(len(obs), -1))
    return p.reshape(len(obs), T_PRED, 2).astype(np.float32)

def ade_fde(pred, fut):
    d = np.linalg.norm(pred - fut, axis=2)   # (n,T_PRED)
    return float(d.mean()), float(d[:, -1].mean())

# ---- run ---------------------------------------------------------------------
print('collecting training windows (baseline)...')
Xtr_all, Ytr_all = collect('baseline', max_drones=2500)
# split baseline drones into train/test by window index (held-out)
idx = RNG.permutation(len(Xtr_all)); ntr = int(0.7 * len(idx))
tr, te = idx[:ntr], idx[ntr:]
Xtr, Ytr = Xtr_all[tr], Ytr_all[tr]
print(f'  train windows: {len(Xtr)}')

print('fitting learned models...')
ridge = fit_learned(Ridge(alpha=1.0), Xtr, Ytr)
mlp   = fit_learned(MLPRegressor(hidden_layer_sizes=(128,128), max_iter=80,
                                 early_stopping=True, random_state=0), Xtr, Ytr)

rows = []
for name, prefix in SETS.items():
    if name == 'baseline':
        Xte, Yte = Xtr_all[te], Ytr_all[te]          # held-out baseline windows
    else:
        Xte, Yte = collect(prefix, max_drones=900)
        if len(Xte) > 6000:
            s = RNG.choice(len(Xte), 6000, replace=False); Xte, Yte = Xte[s], Yte[s]
    preds = {'CV': pred_cv(Xte), 'KF': pred_kf(Xte),
             'Ridge': pred_learned(ridge, Xte), 'MLP': pred_learned(mlp, Xte)}
    for m, pr in preds.items():
        ade, fde = ade_fde(pr, Yte)
        rows.append({'condition': name, 'method': m, 'n_windows': len(Xte),
                     'ADE_m': round(ade, 2), 'FDE_m': round(fde, 2)})
    print(f'{name:14s} n={len(Xte):5d}  ' +
          '  '.join(f'{m}:{ade_fde(pr,Yte)[0]:.1f}/{ade_fde(pr,Yte)[1]:.1f}'
                    for m, pr in preds.items()))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'prediction_benchmark.csv'), index=False)

# per-horizon error curve on a sample of the baseline held-out test set
sub = te[:4000]
Xb, Yb = Xtr_all[sub], Ytr_all[sub]
curve = {f'step_{m}': np.linalg.norm(pr - Yb, axis=2).mean(axis=0)
         for m, pr in [('CV', pred_cv(Xb)), ('KF', pred_kf(Xb)),
                       ('Ridge', pred_learned(ridge, Xb)), ('MLP', pred_learned(mlp, Xb))]}

# save a few example windows for the qualitative figure
exX, exY = collect('baseline', max_drones=60)
sel = RNG.choice(len(exX), min(12, len(exX)), replace=False)
np.savez(os.path.join(OUT, 'prediction_examples.npz'),
         obs=exX[sel], fut=exY[sel],
         cv=pred_cv(exX[sel]), kf=pred_kf(exX[sel]),
         ridge=pred_learned(ridge, exX[sel]), mlp=pred_learned(mlp, exX[sel]),
         horizon=np.arange(1, T_PRED + 1), **curve)
print('\nwrote', os.path.join(OUT, 'prediction_benchmark.csv'))
print(res.pivot(index='method', columns='condition', values='ADE_m'))
