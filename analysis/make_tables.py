# -*- coding: utf-8 -*-
"""Compose the four manuscript tables (strict mirror of s41597-025-06318-5):
   Table 1  dataset composition           (<- target: defect distribution matrix)
   Table 2  comparison with resources     (<- target: dataset comparison)
   Table 3  per-condition conflict summary(<- target: cross-dataset result)
   Table 4  trajectory-prediction benchmark(<- target: multi-model benchmark)
Writes LaTeX to analysis/out/tables/*.tex and prints a digest."""
import os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'analysis', 'out')
TBL  = os.path.join(OUT, 'tables'); os.makedirs(TBL, exist_ok=True)

sc  = pd.read_csv(os.path.join(ROOT, 'processed', 'scenarios.csv'))
fl  = pd.read_csv(os.path.join(ROOT, 'processed', 'flights.csv'),
                  usecols=['scenario', 'duration_s'])
pred = pd.read_csv(os.path.join(OUT, 'prediction_benchmark.csv'))
conf = pd.read_csv(os.path.join(OUT, 'conflicts_per_condition.csv'))

FOLDERS = ['baseline','alt180','alt200','speed15','wind5','speed10']
NAME = {'baseline':'\\texttt{baseline}','alt180':'\\texttt{alt180}',
        'alt200':'\\texttt{alt200}','speed15':'\\texttt{speed15}',
        'wind5':'\\texttt{wind5}','speed10':'\\texttt{speed10}'}
CSET = {'baseline':'baseline','alt180':'alt180','alt200':'alt200',
        'speed15':'speed15','wind5':'wind5','speed10':'speed10'}
fol_of = lambda s: s.split('__')[0]
sc['folder'] = sc['scenario'].map(fol_of)
flights_per_run = 6911

def grp(s): return s.replace('_', '\\_')

# ============================ Table 1: composition ============================
rows1 = []
for fo in FOLDERS:
    g = sc[sc['folder'] == fo]
    sp = sorted(g['cruise_speed_ms'].unique())
    al = sorted(g['cruise_alt_ft'].unique())
    ws = sorted(g['wind_speed_ms'].unique())
    wd = sorted(g[g['wind_speed_ms'] > 0]['wind_dir_deg'].unique())
    rows1.append(dict(name=NAME[fo],
                      speed='/'.join(map(str, sp)), alt='/'.join(map(str, al)),
                      wind='/'.join(map(str, ws)),
                      dirs='/'.join(map(str, wd)) if wd else '--',
                      runs=len(g), flights=len(g) * flights_per_run,
                      records=int(g['n_records'].sum())))
tot = dict(name='\\textbf{Total}', speed='10--20', alt='100--200', wind='0--10', dirs='0/90/180/270',
           runs=int(sc.shape[0]), flights=int(sc.shape[0] * flights_per_run),
           records=int(sc['n_records'].sum()))

def f(n): return f'{n:,}'
t1 = [r"\begin{table}[h]\centering",
      r"\caption*{\textbf{Table 1.} Composition of the dataset. The fixed demand of 6{,}911 flights is propagated through six condition sets that sweep cruise speed, cruise altitude and wind one factor at a time around the baseline; every run logs the full per-second state of all 6{,}911 drones.}",
      r"\begin{adjustbox}{max width=\textwidth}",
      r"\begin{tabular}{|l|c|c|c|c|c|c|c|}", r"\hline",
      r"\textbf{Condition set} & \textbf{Cruise speed (m/s)} & \textbf{Cruise alt (ft)} & \textbf{Wind speed (m/s)} & \textbf{Wind dir ($^\circ$)} & \textbf{Runs} & \textbf{Flights} & \textbf{State records}\\",
      r"\hline"]
for r in rows1:
    t1.append(f"{r['name']} & {r['speed']} & {r['alt']} & {r['wind']} & {r['dirs']} & {r['runs']} & {f(r['flights'])} & {f(r['records'])}\\\\")
t1.append(r"\hline")
t1.append(f"{tot['name']} & {tot['speed']} & {tot['alt']} & {tot['wind']} & {tot['dirs']} & {tot['runs']} & {f(tot['flights'])} & {f(tot['records'])}\\\\")
t1 += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(TBL, 'table1_composition.tex'), 'w', encoding='utf-8').write('\n'.join(t1))

# ============================ Table 3: conflicts ==============================
conf['key'] = conf['condition'].map(lambda c: FOLDERS.index(CSET[c]))
conf = conf.sort_values('key')
t3 = [r"\begin{table}[h]\centering",
      r"\caption*{\textbf{Table 3.} Per-condition conflict summary: en-route losses of separation (horizontal $<$50~m, vertical $<$15~m), averaged over the five runs of each condition set. Peak is the maximum simultaneous count; total is summed over time (separation-loss-seconds); unique pairs is the number of distinct drone pairs that lose separation at any time.}",
      r"\begin{adjustbox}{max width=\textwidth}",
      r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
      r"\textbf{Condition set} & \textbf{Cruise speed (m/s)} & \textbf{Cruise alt (ft)} & \textbf{Mean peak} & \textbf{Mean total (LoS-s)} & \textbf{Mean unique pairs}\\",
      r"\hline"]
for _, r in conf.iterrows():
    t3.append(f"{NAME[CSET[r['condition']]]} & {int(r['cruise_speed_ms'])} & {int(r['cruise_alt_ft'])} & {int(round(r['mean_peak'])):,} & {int(round(r['mean_total'])):,} & {int(round(r['mean_unique'])):,}\\\\")
t3 += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(TBL, 'table3_conflicts.tex'), 'w', encoding='utf-8').write('\n'.join(t3))

# ============================ Table 4: prediction =============================
ORDER_M = ['CV', 'KF', 'Ridge', 'MLP']
LABEL_M = {'CV':'Constant velocity', 'KF':'Kalman filter', 'Ridge':'Linear seq2seq (Ridge)', 'MLP':'MLP'}
ORDER_C = ['baseline','alt180','alt200','speed15','wind5','speed10']
CHEAD = {'baseline':'baseline','alt180':'alt180','alt200':'alt200',
         'speed15':'speed15','wind5':'wind5','speed10':'speed10'}
ade = pred.pivot(index='method', columns='condition', values='ADE_m')
fde = pred.pivot(index='method', columns='condition', values='FDE_m')
ade['Overall'] = ade[ORDER_C].mean(axis=1); fde['Overall'] = fde[ORDER_C].mean(axis=1)
hdr = ' & '.join([r'\textbf{Method}', r'\textbf{Overall}'] + [f'\\textbf{{{CHEAD[c]}}}' for c in ORDER_C])
t4 = [r"\begin{table}[h]\centering",
      r"\caption*{\textbf{Table 4.} Trajectory-prediction benchmark. Average displacement error / final displacement error (ADE\,/\,FDE, in metres) for a 5\,s-observe, 10\,s-predict task. The two learned models are trained only on the baseline condition; all methods are then evaluated on held-out windows of every condition set, so the table reports cross-condition generalisation. Lower is better.}",
      r"\begin{adjustbox}{max width=\textwidth}",
      r"\begin{tabular}{|l|c|c|c|c|c|c|c|}", r"\hline",
      r"\multicolumn{8}{|c|}{\textbf{ADE (m)}}\\", r"\hline", hdr + r"\\", r"\hline"]
for m in ORDER_M:
    t4.append(f"{LABEL_M[m]} & " + ' & '.join(f'{ade.loc[m, c]:.2f}' for c in (['Overall'] + ORDER_C)) + r"\\")
t4 += [r"\hline", r"\multicolumn{8}{|c|}{\textbf{FDE (m)}}\\", r"\hline", hdr + r"\\", r"\hline"]
for m in ORDER_M:
    t4.append(f"{LABEL_M[m]} & " + ' & '.join(f'{fde.loc[m, c]:.2f}' for c in (['Overall'] + ORDER_C)) + r"\\")
t4 += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(TBL, 'table4_prediction.tex'), 'w', encoding='utf-8').write('\n'.join(t4))

# ============================ Table 2: comparison (static) =====================
t2 = r"""\begin{table}[h]\centering
\caption*{\textbf{Table 2.} Position of this dataset relative to representative open and simulated air-traffic resources.}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{|l|l|l|l|l|}
\hline
\textbf{Resource} & \textbf{Type} & \textbf{Vehicles} & \textbf{Per-vehicle state} & \textbf{Conditions varied}\\
\hline
OpenSky$^{15}$ & real (ADS-B) & mostly crewed & position + limited state & observational\\
Terminal-area release$^{16}$ & real (ADS-B+) & mostly crewed & multimodal, terminal area & observational\\
Matrice-100 set$^{17}$ & real (flight test) & 1 multirotor & full, energy + position & speed, altitude, payload, wind\\
Metropolis$^{14,18}$ & simulation & mixed & full (study, not released) & airspace structure\\
This dataset & simulation & M600 drones, 6{,}911 concurrent & full, 22 fields at 1\,Hz & speed, altitude, wind\\
\hline
\end{tabular}
\end{adjustbox}
\end{table}"""
open(os.path.join(TBL, 'table2_comparison.tex'), 'w', encoding='utf-8').write(t2)

# ============================ digest ==========================================
print('Total runs   :', sc.shape[0])
print('Total flights:', f(sc.shape[0] * flights_per_run))
print('Total records:', f(int(sc['n_records'].sum())))
print('\n--- Table 1 composition ---')
print(pd.DataFrame(rows1)[['name','speed','alt','wind','runs','flights','records']].to_string(index=False))
print('\n--- Table 4 ADE (m) ---')
print(ade[['Overall'] + ORDER_C].round(2).to_string())
print('\nwrote 4 .tex files to', TBL)
