# Data dictionary — drone-traffic dataset

This is an analysis-ready release of high-density M600 drone-traffic state logs
over Bogota, generated with a fast-time point-mass multirotor simulator (see
01_make_demand.py and 02_simulate.py). Processing (see build_dataset.py): three
constant columns (type, phase, fuelflow) were removed; the per-run operating
parameters that were encoded in the log file names were added as explicit
columns; logs were converted from plain text to columnar Parquet; and two
summary tables were derived.

## states/<scenario>.parquet  (30 files, 1190 MB total)
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

## flights.csv  (one row per scenario x drone, 207330 rows)
scenario, run parameters, drone_id, orig_lat/lon, dest_lat/lon,
t_start_s, t_end_s, duration_s, od_distance_m (great-circle origin->dest),
path_length_m (flown), gs_mean_ms, gs_max_ms, alt_cruise_m.

## hubs.csv  (821 rows)
hub_id, lat, lon, n_drones — the fixed set of origin hubs (identical across runs).

## scenarios.csv  (30 rows)
Run manifest: scenario, folder, parameters, n_drones, duration_s, n_records, states_file, states_mb.

Note: each condition set includes its own calm reference run; all 30 runs are
generated independently (distinct turbulence realisations).
