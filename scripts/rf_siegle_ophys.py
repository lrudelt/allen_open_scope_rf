"""Siegle-style receptive field mapping for two-photon traces.

Mirrors `compute_rf_siegle_gabors.py` (ecephys) but replaces the per-trial
spike rate with the mean fluorescence over the response window, so mesoscope
and SLAP2 sessions produce result files with exactly the same layout and can
be fed to the same `_optimize_probe_files` / analysis code.

Three differences from the ecephys version, all forced by the signal:

* **Test statistic.** Siegle uses chi-squared, Σ(O − E)²/E. Under a permutation
  of the trial responses E is invariant, so Σ(O − E)² ranks the shuffles
  identically and gives the same p-value — but it stays well defined when the
  responses are ΔF/F and can be negative. That is what is used here.
* **Missing windows.** Imaging is not continuous: SLAP2 records in ~30 s bouts
  with a blanking period in between, and drops short stretches of samples to
  NaN. A response window that is not sampled densely enough (see
  `min_samples_for`) is marked NaN and excluded from both the observed and the
  shuffled statistic.
* **Baseline.** Optionally the mean over a pre-stimulus window is subtracted
  from each trial response, which removes slow drift that the fluorescence
  carries and the spike rate does not.
"""

import numpy as np
import pandas as pd
import h5py
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm

N_SHUFFLE = 1000
MIN_COVERAGE = 0.5   # fraction of a window that must be sampled for it to count


# ---------------------------------------------------------------------------
# Trial responses
# ---------------------------------------------------------------------------

def drop_missing_samples(traces, times):
    """Remove timepoints where any ROI is NaN.

    SLAP2 blanks short stretches of samples across all ROIs at once. Dropping
    them keeps the cumulative sums in `window_means` finite; the resulting holes
    are then caught by the per-window coverage check.
    """
    keep = np.isfinite(traces).all(axis=0)
    return traces[:, keep], times[keep]


def min_samples_for(times, duration, min_coverage=MIN_COVERAGE):
    """Fewest samples a `duration`-second window must hold to be counted."""
    rate = 1.0 / np.median(np.diff(times))
    return max(1, int(np.ceil(min_coverage * duration * rate)))


def window_means(traces, times, starts, stops, min_samples=1):
    """Mean of every trace over every [start, stop) window.

    traces: (n_rois, n_t), times: (n_t,), starts/stops: (n_windows,).
    Returns (n_rois, n_windows), NaN for windows holding fewer than
    `min_samples` samples — i.e. those falling in a blanking gap.
    """
    i0 = np.searchsorted(times, starts, side='left')
    i1 = np.searchsorted(times, stops, side='left')
    n = i1 - i0
    valid = n >= max(min_samples, 1)

    csum = np.concatenate(
        [np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1, dtype=np.float64)],
        axis=1,
    )
    out = np.full((traces.shape[0], len(starts)), np.nan)
    out[:, valid] = (csum[:, i1[valid]] - csum[:, i0[valid]]) / n[valid]
    return out


def grid_indices(df_rf, x_pos, y_pos, orientations):
    """Per-presentation (orientation, repeat, x, y) indices into the response grid.

    Repeats are numbered in presentation order within each (orientation, x, y)
    cell, matching the ecephys version's `R[i, x_i, y_i]`.
    """
    keys = pd.DataFrame({
        'o': np.searchsorted(orientations, df_rf['Orientation'].to_numpy(float)),
        'x': np.searchsorted(x_pos, df_rf['X'].to_numpy(float)),
        'y': np.searchsorted(y_pos, df_rf['Y'].to_numpy(float)),
    })
    keys['rep'] = keys.groupby(['o', 'x', 'y']).cumcount()
    return keys['o'].values, keys['rep'].values, keys['x'].values, keys['y'].values


def response_grids(traces, times, df_rf, x_pos, y_pos, orientations,
                   delay, duration, baseline=0.0):
    """Trial response grid and its per-cell count of usable trials.

    R:       (n_rois, n_orientations, n_repeats, nx, ny), NaN for unsampled trials
    n_valid: (n_orientations, nx, ny)
    """
    onsets = df_rf['start_time'].to_numpy(float)

    resp = window_means(traces, times, onsets + delay, onsets + delay + duration,
                        min_samples_for(times, duration))
    if baseline > 0:
        resp = resp - window_means(traces, times, onsets - baseline, onsets,
                                   min_samples_for(times, baseline))

    o_i, r_i, x_i, y_i = grid_indices(df_rf, x_pos, y_pos, orientations)
    shape = (len(orientations), r_i.max() + 1, len(x_pos), len(y_pos))

    R = np.full((traces.shape[0],) + shape, np.nan)
    R[:, o_i, r_i, x_i, y_i] = resp

    # window coverage depends only on the timestamps, so it is the same for every ROI
    n_valid = np.isfinite(R[0]).sum(axis=1)
    return R, n_valid


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _unit_stats(args, n_shuffle=N_SHUFFLE):
    """Siegle statistics for one ROI. `args` is (seed, R_unit).

    R_unit: (n_orientations, n_repeats, nx, ny). Returns (E, O, Z, P) with
    E/P shaped (n_orientations,) and O/Z shaped (n_orientations, nx, ny).
    """
    seed, R_unit = args
    rng = np.random.default_rng(seed)
    n_or, _, nx, ny = R_unit.shape

    E = np.full(n_or, np.nan)
    O = np.full((n_or, nx, ny), np.nan)
    Z = np.full((n_or, nx, ny), np.nan)
    P = np.ones(n_or)

    for o in range(n_or):
        R = R_unit[o]
        finite = np.isfinite(R)
        vals = R[finite]
        if vals.size == 0:
            continue

        e = vals.mean()
        with np.errstate(invalid='ignore'):
            o_map = np.nanmean(R, axis=0)
        stat = np.nansum((o_map - e) ** 2)

        # all shuffles at once: permute the observed values, keep the NaNs in place
        shuffled = np.full((n_shuffle,) + R.shape, np.nan)
        shuffled[:, finite] = rng.permuted(np.tile(vals, (n_shuffle, 1)), axis=1)
        with np.errstate(invalid='ignore'):
            o_shuf = np.nanmean(shuffled, axis=1)
        stat_shuf = np.nansum((o_shuf - e) ** 2, axis=(1, 2))

        E[o] = e
        O[o] = o_map
        Z[o] = (o_map - e) / (vals.std() + 1e-10)
        P[o] = (np.sum(stat_shuf > stat) + 1) / (n_shuffle + 1)

    return E, O, Z, P


def compute_siegle_ophys(traces, times, unit_names, df_rf, x_pos, y_pos, orientations,
                         delay, duration, results_path, attributes,
                         baseline=0.0, n_shuffle=N_SHUFFLE, seed=0, n_procs=None):
    """Run the Siegle RF test on every ROI and write one HDF5 result file.

    traces: (n_rois, n_t) fluorescence, times: (n_t,) sample times in seconds.
    Dataset names match `compute_rf_siegle_gabors.py`, so its
    `_optimize_probe_files` consumes these files unchanged.
    """
    if len(unit_names) != traces.shape[0]:
        raise ValueError(f'{len(unit_names)} unit names for {traces.shape[0]} traces')

    R, n_valid = response_grids(traces, times, df_rf, x_pos, y_pos, orientations,
                                delay, duration, baseline)
    n_units = R.shape[0]

    worker = partial(_unit_stats, n_shuffle=n_shuffle)
    tasks = ((seed + i, R[i]) for i in range(n_units))
    with multiprocessing.Pool(processes=n_procs) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=n_units, desc='rois'))

    mean_rates       = np.stack([r[0] for r in results])
    mean_responses   = np.stack([r[1] for r in results])
    z_score_response = np.stack([r[2] for r in results])
    p_values         = np.stack([r[3] for r in results])

    results_path = Path(results_path).with_suffix('.h5')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving results to {results_path}...')
    with h5py.File(results_path, 'w') as hf:
        for key, value in attributes.items():
            hf.attrs[key] = value
        hf.create_dataset('unit_names', data=np.asarray(unit_names).astype('S'))
        hf.create_dataset('orientations', data=np.asarray(orientations, dtype=float))
        hf.create_dataset('x_positions', data=x_pos)
        hf.create_dataset('y_positions', data=y_pos)
        hf.create_dataset('n_valid_trials', data=n_valid)
        hf.create_dataset('mean_rate', data=mean_rates)
        hf.create_dataset('mean_response', data=mean_responses, compression='gzip', compression_opts=4)
        hf.create_dataset('z_score_response', data=z_score_response, compression='gzip', compression_opts=4)
        hf.create_dataset('p_value', data=p_values)

    return results_path


# ---------------------------------------------------------------------------
# Stimulus table helpers
# ---------------------------------------------------------------------------

def split_gabor_block(df_rf):
    """First and second half of one Gabor RF block, in presentation order.

    Trial 0 is the earlier half and trial 1 the later half, so the two maps
    mimic Zebra trial 0 and trial 1. An odd presentation count gives the extra
    presentation to trial 1.
    """
    ordered = df_rf.sort_values("start_time")
    mid = len(ordered) // 2
    if mid == 0 or mid == len(ordered):
        raise ValueError(
            f"RF block has {len(ordered)} presentations; cannot split it in half"
        )
    return {0: ordered.iloc[:mid].copy(), 1: ordered.iloc[mid:].copy()}


def rf_grid(df_rf):
    """Sorted x positions, y positions and orientations of an RF mapping table."""
    x_pos = np.sort(np.unique(df_rf['X'].to_numpy(float)))
    y_pos = np.sort(np.unique(df_rf['Y'].to_numpy(float)))
    orientations = np.sort(np.unique(df_rf['Orientation'].to_numpy(float)))
    return x_pos, y_pos, orientations


def slice_to_stimulus(traces, times, df_rf, margin):
    """Restrict traces to the RF mapping block, plus `margin` seconds either side."""
    keep = ((times >= df_rf['start_time'].min() - margin) &
            (times <= df_rf['stop_time'].max() + margin))
    return traces[:, keep], times[keep]


def _window_key(delay, duration):
    return (round(float(delay), 5), round(float(duration), 5))


def assemble_fixed_windows(full_optimized, sweep_files, output_path):
    """Build one half-block file using the full run's delay and duration.

    ``best_delay`` and ``best_duration`` are copied from the full-block
    optimized file. Each unit and orientation keeps that window; the half
    block is not allowed to pick a new one.
    """
    full_optimized = Path(full_optimized)
    with h5py.File(full_optimized, "r") as hf:
        unit_names = hf["unit_names"][:]
        names = unit_names.astype(str)
        orientations = hf["orientations"][:]
        x_positions = hf["x_positions"][:]
        y_positions = hf["y_positions"][:]
        best_delay = np.asarray(hf["best_delay"][:], dtype=float)
        best_duration = np.asarray(hf["best_duration"][:], dtype=float)

    needed = {
        _window_key(delay, duration)
        for delay, duration in zip(best_delay.ravel(), best_duration.ravel())
        if np.isfinite(delay) and np.isfinite(duration)
    }
    found = {}
    for path in sweep_files:
        with h5py.File(path, "r") as hf:
            if "delay_s" not in hf.attrs:
                continue
            key = _window_key(hf.attrs["delay_s"], hf.attrs["duration_s"])
            got = hf["unit_names"][:].astype(str)
            if len(got) != len(names) or not np.array_equal(got, names):
                raise ValueError(
                    f"Unit list in {Path(path).name} does not match {full_optimized.name}"
                )
            found[key] = dict(
                p=hf["p_value"][:],
                rate=hf["mean_rate"][:],
                mean=hf["mean_response"][:],
                z=hf["z_score_response"][:],
            )
    missing = sorted(needed - set(found))
    if missing:
        raise FileNotFoundError(
            "Half-block files are missing the full run's windows: "
            + ", ".join(f"delay={d:g} duration={u:g}" for d, u in missing)
        )

    n_units, n_ori = best_delay.shape
    nx, ny = len(x_positions), len(y_positions)
    p_values = np.full((n_units, n_ori), np.nan)
    mean_rate = np.full((n_units, n_ori), np.nan)
    mean_response = np.full((n_units, n_ori, nx, ny), np.nan)
    z_score = np.full((n_units, n_ori, nx, ny), np.nan)
    for unit in range(n_units):
        for ori in range(n_ori):
            if not np.isfinite(best_delay[unit, ori]):
                continue
            src = found[_window_key(best_delay[unit, ori], best_duration[unit, ori])]
            p_values[unit, ori] = src["p"][unit, ori]
            mean_rate[unit, ori] = src["rate"][unit, ori]
            mean_response[unit, ori] = src["mean"][unit, ori]
            z_score[unit, ori] = src["z"][unit, ori]

    output_path = Path(output_path).with_suffix(".h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving fixed-window results to {output_path}...")
    with h5py.File(output_path, "w") as hf:
        hf.attrs["parameter_source"] = str(full_optimized)
        hf.attrs["parameters"] = "full_run_delay_duration"
        hf.create_dataset("unit_names", data=unit_names)
        hf.create_dataset("orientations", data=orientations)
        hf.create_dataset("x_positions", data=x_positions)
        hf.create_dataset("y_positions", data=y_positions)
        hf.create_dataset("best_delay", data=best_delay)
        hf.create_dataset("best_duration", data=best_duration)
        hf.create_dataset("p_value", data=p_values)
        hf.create_dataset("mean_rate", data=mean_rate)
        hf.create_dataset("mean_response", data=mean_response, compression="gzip", compression_opts=4)
        hf.create_dataset("z_score_response", data=z_score, compression="gzip", compression_opts=4)
    return output_path


def optimize_over_delay_duration(session_dir, pattern, output_path):
    """Pick, per (ROI, orientation), the (delay, duration) with the lowest p-value.

    Thin wrapper around the ecephys implementation: the result files share a
    layout, so the same reducer applies. `pattern` is a glob selecting the files
    of a single plane / DMD, whose delay × duration grid is reduced to one file.
    """
    from compute_rf_siegle_gabors import _optimize_probe_files

    files = sorted(
        f for f in Path(session_dir).glob(pattern)
        if 'optimized' not in f.stem and '__trial_' not in f.stem
    )
    if not files:
        raise FileNotFoundError(f'No result files matching {pattern!r} in {session_dir}')
    return _optimize_probe_files(files, Path(output_path))
