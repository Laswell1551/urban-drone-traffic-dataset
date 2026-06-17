# -*- coding: utf-8 -*-
"""Regenerate all 12 manuscript figures in one consistent Scientific Data style.
Layout mirrors s41597-025-06318-5 (12 figures). 8 are computed from the data,
4 are schematics. Output: figures/fig01_*.png ... fig12_*.png (300 dpi)."""
import os, glob, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST   = os.path.join(ROOT, 'processed', 'states')
OUT  = os.path.join(ROOT, 'analysis', 'out')
FIG  = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)

# ---- consistent house style --------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'mathtext.default': 'regular',
})
C = {'blue':'#3b6ea5','orange':'#e08214','green':'#4a9b5e','red':'#c0392b',
     'grey':'#6f6f6f','purple':'#8856a7','teal':'#2a9d8f','sand':'#f4ede4'}
LATR = (4.47, 4.78); LONR = (-74.22, -74.02)
ASPECT = 1.0 / np.cos(np.radians(4.62))

def geo(ax):
    ax.set_xlim(*LONR); ax.set_ylim(*LATR); ax.set_aspect(ASPECT)
    ax.set_xlabel('Longitude ($^\\circ$)'); ax.set_ylabel('Latitude ($^\\circ$)')

def panel(ax, lab):
    ax.text(-0.02, 1.04, f'({lab})', transform=ax.transAxes, fontweight='bold',
            fontsize=10, ha='right', va='bottom')

def load(scn, cols):
    return pd.read_parquet(os.path.join(ST, scn + '.parquet'), columns=cols)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print('wrote', name)

def box(ax, x, y, w, h, text, fc, ec='#33373b', fs=8, tc='#1a1a1a', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                linewidth=1.0, edgecolor=ec, facecolor=fc, mutation_aspect=0.6))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold' if bold else 'normal', zorder=5)

def arrow(ax, x1, y1, x2, y2, c='#33373b'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 lw=1.2, color=c, shrinkA=2, shrinkB=2))

# =============================================================================
# Fig 1  construction pipeline (schematic)
# =============================================================================
def fig01():
    fig, ax = plt.subplots(figsize=(7.4, 2.4)); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [('Scenario\ndesign', 'OFAT sweep of speed,\naltitude and wind\naround a baseline', C['sand']),
             ('Fast-time\nsimulation', 'M600 point-mass model;\n6,911 drones; full\nstate logged at 1 Hz', '#dce6f0'),
             ('Post-\nprocessing', 'clean constant fields,\nenrich parameters,\nParquet, derive tables', '#e6f0e6'),
             ('Analysis-ready\nrelease', 'states + flights, hubs,\nscenarios, dictionary,\nbuild script', '#f0e6ef')]
    w, h, y = 0.20, 0.58, 0.22
    xs = np.linspace(0.03, 0.77, 4)
    for (t, d, fc), x in zip(steps, xs):
        box(ax, x, y, w, h, '', fc)
        ax.text(x + w/2, y + h - 0.105, t, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1a1a1a')
        ax.text(x + w/2, y + h/2 - 0.105, d, ha='center', va='center', fontsize=6.7, color='#333')
    for i in range(3):
        arrow(ax, xs[i] + w, y + h/2, xs[i+1], y + h/2)
    save(fig, 'fig01_pipeline.png')

# =============================================================================
# Fig 2  example trajectories across conditions (small multiples)
# =============================================================================
def fig02():
    runs = [('baseline__calm', 'Calm, 20 m/s'),
            ('baseline__n10', 'Head/tail wind N, 10 m/s'),
            ('baseline__e10', 'Cross wind E, 10 m/s'),
            ('speed15__calm', 'Calm, 15 m/s'),
            ('speed10__calm', 'Calm, 10 m/s'),
            ('alt200__calm', 'Calm, 200 ft')]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    rng = np.random.default_rng(3)
    for ax, (scn, title) in zip(axes.ravel(), runs):
        d = load(scn, ['drone_id', 'lat_deg', 'lon_deg', 'gs_ms'])
        ids = rng.choice(d['drone_id'].unique(), 120, replace=False)
        for _, g in d[d['drone_id'].isin(ids)].groupby('drone_id'):
            ax.plot(g['lon_deg'], g['lat_deg'], lw=0.4, alpha=0.5, color=C['blue'])
        ax.set_xlim(*LONR); ax.set_ylim(*LATR); ax.set_aspect(ASPECT)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True)
    fig.text(0.5, 0.04, 'Longitude', ha='center'); fig.text(0.06, 0.5, 'Latitude', va='center', rotation=90)
    fig.tight_layout(rect=[0.06, 0.05, 1, 1])
    save(fig, 'fig02_samples.png')

# =============================================================================
# Fig 3  demand geography: hubs + destinations
# =============================================================================
def fig03():
    hubs = pd.read_csv(os.path.join(ROOT, 'processed', 'hubs.csv'))
    fl = pd.read_csv(os.path.join(ROOT, 'processed', 'flights.csv'),
                     usecols=['scenario', 'dest_lat', 'dest_lon'])
    fl = fl[fl['scenario'] == 'baseline__calm']
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(fl['dest_lon'], fl['dest_lat'], s=2, marker='+', linewidths=0.3,
               color=C['grey'], alpha=0.5, label=f'destinations ({len(fl):,})')
    ax.scatter(hubs['lon'], hubs['lat'], s=hubs['n_drones']*1.4, marker='o',
               facecolor=C['orange'], edgecolor='#7a3e00', linewidth=0.3, alpha=0.85,
               label=f'origin hubs ({len(hubs)})')
    geo(ax); ax.legend(loc='upper right', framealpha=0.9, markerscale=1)
    save(fig, 'fig03_od_map.png')

# =============================================================================
# Fig 4  OFAT experimental design (schematic)
# =============================================================================
def fig04():
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.05, 0.40, 0.26, 0.22, '', '#dce6f0')
    ax.text(0.18, 0.555, 'Baseline', ha='center', fontsize=9, fontweight='bold')
    ax.text(0.18, 0.475, '20 m/s  $\\cdot$  100 ft\ncalm + 10 m/s wind', ha='center', fontsize=7.4, color='#333')
    branches = [(0.66, 0.71, 'Cruise-speed sweep', '15 m/s,  10 m/s', C['teal']),
                (0.66, 0.425, 'Cruise-altitude sweep', '180 ft,  200 ft', C['green']),
                (0.66, 0.14, 'Wind-speed sweep', '5 m/s', C['orange'])]
    for x, y, t, v, col in branches:
        box(ax, x, y, 0.30, 0.15, '', '#f7f7f7', ec=col)
        ax.text(x + 0.15, y + 0.103, t, ha='center', fontsize=7.8, fontweight='bold', color='#1a1a1a')
        ax.text(x + 0.15, y + 0.042, v, ha='center', fontsize=7.6, color='#333')
        arrow(ax, 0.31, 0.51, x, y + 0.075, c=col)
    ax.text(0.5, 0.95, 'One-factor-at-a-time sweep over a fixed 6,911-flight demand',
            ha='center', fontsize=8.6, fontweight='bold')
    ax.text(0.5, 0.035, 'Every windy run is repeated for wind from N, E, S and W (0 / 90 / 180 / 270$^\\circ$).',
            ha='center', fontsize=7.4, style='italic', color='#444')
    save(fig, 'fig04_ofat_design.png')

# =============================================================================
# Fig 5  columnar schema + a sample record (schematic)
# =============================================================================
def fig05():
    groups = [('Identity & index', ['scenario', 'drone_id', 'sim_time_s'], '#dce6f0'),
              ('Run parameters', ['cruise_speed_ms', 'wind_dir_deg', 'wind_speed_ms', 'cruise_alt_ft'], '#e6f0e6'),
              ('Position', ['lat_deg', 'lon_deg', 'alt_m'], '#fbe6d6'),
              ('Kinematics', ['tas_ms', 'cas_ms', 'gs_ms', 'vs_ms', 'distflown_m', 'trk_deg', 'hdg_deg'], '#f0e6ef'),
              ('Atmosphere', ['temp_k', 'pressure_pa', 'rho_kgm3'], '#e6eff0'),
              ('Forces', ['thrust_n', 'drag_n'], '#f4ede4')]
    rec = load('baseline__calm', None).iloc[100]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1.6, 1]})
    for a in (axl, axr): a.axis('off'); a.set_xlim(0, 1); a.set_ylim(0, 1)
    axl.text(0.5, 0.965, 'states/<scenario>.parquet  -  22 columns, one row per drone-second',
             ha='center', fontsize=8.0, fontweight='bold')
    n = len(groups); top, bot, gap = 0.90, 0.03, 0.016
    hh = (top - bot - gap * (n - 1)) / n
    y = top
    for name, cols, fc in groups:
        box(axl, 0.02, y - hh, 0.96, hh, '', fc)
        axl.text(0.045, y - 0.028, name, fontsize=7.5, fontweight='bold', va='top')
        axl.text(0.045, y - hh + 0.016, ',  '.join(cols), fontsize=6.5, va='bottom',
                 color='#333', family='monospace')
        y -= hh + gap
    axr.text(0.5, 0.97, 'Example record', ha='center', fontsize=8.2, fontweight='bold')
    show = ['scenario', 'drone_id', 'sim_time_s', 'cruise_speed_ms', 'wind_speed_ms',
            'lat_deg', 'lon_deg', 'alt_m', 'gs_ms', 'vs_ms', 'rho_kgm3', 'thrust_n']
    yy = 0.9
    for k in show:
        v = rec[k]
        vs = v if isinstance(v, str) else (f'{v:.5f}' if abs(v) < 1000 and v != int(v) else f'{v:g}')
        axr.text(0.04, yy, k, fontsize=6.8, va='center', family='monospace', color=C['blue'])
        axr.text(0.62, yy, str(vs), fontsize=6.8, va='center', family='monospace', color='#222')
        yy -= 0.072
    save(fig, 'fig05_schema.png')

# =============================================================================
# Fig 6  spatial traffic density heatmap
# =============================================================================
def fig06():
    d = load('baseline__calm', ['lat_deg', 'lon_deg'])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    H, xe, ye = np.histogram2d(d['lon_deg'], d['lat_deg'], bins=240,
                               range=[list(LONR), list(LATR)])
    pcm = ax.pcolormesh(xe, ye, H.T, norm=LogNorm(vmin=1, vmax=H.max()), cmap='magma')
    geo(ax); cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('drone-seconds per cell (log scale)')
    save(fig, 'fig06_density.png')

# =============================================================================
# Fig 7  deposit file structure (schematic)
# =============================================================================
def fig07():
    fig, ax = plt.subplots(figsize=(6.0, 3.3)); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.02, 0.04, 0.96, 0.92, '', '#fcfbf9', ec='#b9b2a6')
    tree = [
        ('drone_traffic_dataset/', '', 0, True),
        ('states/', '30 Parquet files, 937 MB', 1, True),
        ('baseline__calm.parquet', '631,424 rows', 2, False),
        ('...  (29 further run files)', '', 2, False),
        ('flights.csv', '207,330 per-flight summaries', 1, False),
        ('hubs.csv', '821 origin hubs', 1, False),
        ('scenarios.csv', '30-run manifest', 1, False),
        ('DATA_DICTIONARY.md', 'every table and column', 1, False),
        ('build_dataset.py', 'reproducible processing script', 1, False)]
    y = 0.87
    for name, desc, depth, folder in tree:
        x = 0.06 + depth * 0.06
        ax.text(x, y, ('|- ' if depth else '') + name, family='monospace',
                fontsize=8.2 if depth == 0 else 7.6,
                fontweight='bold' if folder else 'normal',
                color=C['blue'] if folder else '#1a1a1a', va='center')
        if desc:
            ax.text(0.66, y, desc, fontsize=7.0, color='#555', va='center', style='italic')
        y -= 0.095
    save(fig, 'fig07_filetree.png')

# =============================================================================
# Fig 8  technical-validation workflow (schematic)
# =============================================================================
def fig08():
    fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [('Integrity &\ncompleteness', 'record counts,\n0 NaN, unique keys', '#dce6f0'),
             ('Physical\nplausibility', 'speed/vertical limits,\natmosphere, profiles', '#e6f0e6'),
             ('Sensitivity to\ninputs', 'OFAT response of\nspeed, duration, density', '#fbeing' if False else '#fbe6d6'),
             ('Utility\ndemonstration', 'conflict detection +\ntrajectory prediction', '#f0e6ef')]
    w, h, y = 0.205, 0.52, 0.26
    xs = np.linspace(0.04, 0.79, 4)
    for (t, d, fc), x in zip(steps, xs):
        box(ax, x, y, w, h, '', fc)
        ax.text(x + w/2, y + h - 0.1, t, ha='center', va='center', fontsize=8.2, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.09, d, ha='center', va='center', fontsize=6.8, color='#333')
    for i in range(3):
        arrow(ax, xs[i] + w, y + h/2, xs[i+1], y + h/2)
    save(fig, 'fig08_validation.png')

# =============================================================================
# Fig 9  flight-level distributions (4 panels)
# =============================================================================
def fig09():
    fl = pd.read_csv(os.path.join(ROOT, 'processed', 'flights.csv'))
    b = fl[fl['scenario'] == 'baseline__calm']
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    specs = [('od_distance_m', 'OD distance (m)', C['blue']),
             ('path_length_m', 'Path length (m)', C['teal']),
             ('duration_s', 'Flight duration (s)', C['green']),
             ('gs_mean_ms', 'Mean ground speed (m/s)', C['orange'])]
    for ax, (col, xl, c), lab in zip(axes, specs, 'abcd'):
        ax.hist(b[col], bins=40, color=c, alpha=0.85, edgecolor='white', linewidth=0.2)
        ax.set_xlabel(xl); panel(ax, lab)
        if lab == 'a': ax.set_ylabel('Flights')
    fig.tight_layout()
    save(fig, 'fig09_flight_dist.png')

# =============================================================================
# Fig 10  kinematic plausibility (3 panels)
# =============================================================================
def fig10():
    d = load('baseline__calm', ['drone_id', 'sim_time_s', 'alt_m', 'vs_ms', 'gs_ms'])
    rng = np.random.default_rng(1)
    ids = rng.choice(d['drone_id'].unique(), 40, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for _, g in d[d['drone_id'].isin(ids)].groupby('drone_id'):
        axes[0].plot(g['sim_time_s'], g['alt_m'], lw=0.6, alpha=0.6, color=C['blue'])
    axes[0].axhline(30.48, ls='--', lw=0.8, color=C['red']); axes[0].set_xlabel('Simulation time (s)')
    axes[0].set_ylabel('Altitude (m)'); panel(axes[0], 'a')
    axes[1].hist(d['vs_ms'], bins=60, color=C['teal'], alpha=0.85)
    axes[1].set_xlabel('Vertical speed (m/s)'); axes[1].set_ylabel('Records'); axes[1].set_yscale('log'); panel(axes[1], 'b')
    axes[2].hist(d['gs_ms'], bins=60, color=C['orange'], alpha=0.85)
    axes[2].axvline(20, ls='--', lw=0.8, color=C['red']); axes[2].set_xlabel('Ground speed (m/s)')
    axes[2].set_ylabel('Records'); panel(axes[2], 'c')
    fig.tight_layout()
    save(fig, 'fig10_kinematics.png')

# =============================================================================
# Fig 11  OFAT sensitivity (4 panels)
# =============================================================================
def fig11():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    # (a) ground-speed distribution under wind, baseline 20 m/s
    winds = [('baseline__calm', 'calm', C['grey']),
             ('baseline__n10', 'N wind', C['blue']),
             ('baseline__e10', 'E wind', C['green'])]
    for scn, lab, c in winds:
        g = load(scn, ['gs_ms'])['gs_ms']
        axes[0,0].hist(g, bins=80, range=(0, 30), histtype='step', lw=1.3, color=c, label=lab)
    axes[0,0].set_yscale('log'); axes[0,0].axvline(20, ls='--', lw=0.8, color=C['red'])
    axes[0,0].set_xlabel('Ground speed (m/s)'); axes[0,0].set_ylabel('Records (log)')
    axes[0,0].legend(); panel(axes[0,0], 'a')
    # (b) drones airborne over time
    air = pd.read_csv(os.path.join(OUT, 'concurrency_timeseries.csv'))
    lab = {'baseline__calm':'calm','baseline__n10':'N wind',
           'baseline__e10':'E wind','baseline__s10':'S wind'}
    for scn, l in lab.items():
        a = air[air['scenario'] == scn]
        axes[0,1].plot(a['sim_time_s'], a['n_airborne'], lw=1.0, label=l)
    axes[0,1].set_xlabel('Simulation time (s)'); axes[0,1].set_ylabel('Drones airborne')
    axes[0,1].legend(); panel(axes[0,1], 'b')
    # (c) mean flight duration vs cruise speed
    fl = pd.read_csv(os.path.join(ROOT, 'processed', 'flights.csv'))
    md = fl.groupby('cruise_speed_ms')['duration_s'].mean()
    axes[1,0].bar(md.index.astype(str), md.values, color=C['teal'], width=0.5)
    axes[1,0].set_xlabel('Cruise speed (m/s)'); axes[1,0].set_ylabel('Mean flight duration (s)'); panel(axes[1,0], 'c')
    # (d) prediction ADE per condition
    pred = pd.read_csv(os.path.join(OUT, 'prediction_benchmark.csv'))
    piv = pred.pivot(index='condition', columns='method', values='ADE_m')
    order = ['baseline','alt180','alt200','speed15','wind5','speed10']
    piv = piv.reindex(order)
    x = np.arange(len(order)); wbar = 0.2
    for i, m in enumerate(['CV','KF','Ridge','MLP']):
        axes[1,1].bar(x + (i-1.5)*wbar, piv[m], wbar, label=m)
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(['base','a180','a200','sp15','w5','sp10'], fontsize=6.5)
    axes[1,1].set_ylabel('Prediction ADE (m)'); axes[1,1].legend(ncol=2); panel(axes[1,1], 'd')
    fig.tight_layout()
    save(fig, 'fig11_sensitivity.png')

# =============================================================================
# Fig 12  utility: conflicts + prediction overlay (3 panels)
# =============================================================================
def fig12():
    fig = plt.figure(figsize=(7.4, 2.5))
    ax0 = fig.add_subplot(1, 3, 1); ax1 = fig.add_subplot(1, 3, 2); ax2 = fig.add_subplot(1, 3, 3)
    # (a) conflicts over time
    ts = pd.read_csv(os.path.join(OUT, 'conflicts_timeseries.csv'))
    lab = {'baseline__calm':'calm','baseline__n10':'N wind',
           'baseline__s10':'S wind','baseline__e10':'E wind'}
    for scn, l in lab.items():
        s = ts[ts['scenario'] == scn]
        ax0.plot(s['sim_time_s'], s['los_count'], lw=1.0, label=l)
    ax0.set_xlabel('Simulation time (s)'); ax0.set_ylabel('Losses of separation'); ax0.legend(fontsize=6.5); panel(ax0, 'a')
    # (b) conflict hotspots
    z = np.load(os.path.join(OUT, 'conflict_hotspot_baseline.npz'))
    H, le, lo = z['H'], z['lat_edges'], z['lon_edges']
    pcm = ax1.pcolormesh(lo, le, H, norm=LogNorm(vmin=1, vmax=H.max()), cmap='inferno')
    ax1.set_xlim(*LONR); ax1.set_ylim(*LATR); ax1.set_aspect(ASPECT)
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude'); panel(ax1, 'b')
    cb = fig.colorbar(pcm, ax=ax1, shrink=0.82, pad=0.03); cb.ax.tick_params(labelsize=6.5)
    # (c) prediction error vs horizon
    ex = np.load(os.path.join(OUT, 'prediction_examples.npz'))
    h = ex['horizon']
    for m, c in [('CV', C['grey']), ('KF', C['orange']), ('Ridge', C['blue']), ('MLP', C['green'])]:
        ax2.plot(h, ex[f'step_{m}'], 'o-', ms=3, lw=1.3, color=c, label=m)
    ax2.set_xlabel('Prediction horizon (s)'); ax2.set_ylabel('Mean error (m)')
    ax2.legend(fontsize=6.5); panel(ax2, 'c')
    fig.tight_layout()
    save(fig, 'fig12_utility.png')

if __name__ == '__main__':
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12]:
        fn()
    print('\nall figures written to', FIG)
