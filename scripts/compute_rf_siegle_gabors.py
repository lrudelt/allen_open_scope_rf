import pandas as pd
import numpy as np
import os
import sys
import multiprocessing
from functools import partial

import neo.core as neo
from quantities import s
import quantities as pq

sys.path.append('..')
import utils
import argparse
from datetime import datetime
from tqdm import tqdm

from pathlib import Path
import h5py

RESULTS_DIR = os.path.abspath('../../../results/allen_open_scope/rf/gabors/')


def get_stim_onsets_offsets(df_rf, combination_xy, orientation='0'):
    x, y = combination_xy
    df_sub = df_rf[
        (df_rf['X'] == str(float(x))) &
        (df_rf['Y'] == str(float(y))) &
        (df_rf['Orientation'] == str(float(orientation)))
    ]
    return (df_sub['start_time'].values * pq.s, df_sub['stop_time'].values * pq.s)


def time_slice_spike_train(spike_train, start, stop):
    if (stop < spike_train.t_start) or (start > spike_train.t_stop):
        return []
    start = max(start, spike_train.t_start)
    stop = min(stop, spike_train.t_stop)
    return spike_train.time_slice(start, stop)


def _compute_unit(idx, x_pos, y_pos, unique_orientations, df_units, df_rf, combinations_xy, delay, duration):
    """Compute receptive field stats for a single unit; returns a list of dicts (one per orientation)."""
    rows = []
    spike_times = df_units.loc[idx, 'spike_times']
    spike_train = neo.SpikeTrain(spike_times, t_start=spike_times[0], t_stop=spike_times[-1], units=pq.s)

    for orientation in unique_orientations.astype(float):
        R = np.zeros((5, len(x_pos), len(y_pos)))
        for (x, y) in combinations_xy:
            x_i = np.digitize(float(x), x_pos) - 1
            y_i = np.digitize(float(y), y_pos) - 1
            onsets, offsets = get_stim_onsets_offsets(df_rf, combination_xy=(x, y), orientation=orientation)
            for i, (onset, offset) in enumerate(zip(onsets, offsets)):
                n_spikes = len(time_slice_spike_train(spike_train, onset+delay, onset+delay+duration))
                R[i, x_i, y_i] = n_spikes / ((offset - onset).rescale(s).magnitude)

        E = np.mean(R) + 1e-10
        O = np.mean(R, axis=0)
        chi_squared = np.sum((O - E) ** 2 / E)

        n_shuffle = 1000
        chi_squared_shuffled = np.zeros(n_shuffle)
        for i in range(n_shuffle):
            R_shuffled = np.random.permutation(R.flatten()).reshape(R.shape)
            O_shuffled = np.mean(R_shuffled, axis=0)
            chi_squared_shuffled[i] = np.sum((O_shuffled - E) ** 2 / E)

        p_value = (np.sum(chi_squared_shuffled > chi_squared) + 1) / (n_shuffle + 1)
        std = np.std(R)

        rows.append({
            'orientation': orientation,
            'mean_rate': E,
            'mean_response': O,
            'z_score_response': (O - E) / (std + 1e-10),
            'p_value': p_value,
        })

    return rows


def compute_siegle(x_pos, y_pos, unique_orientations, df_units, unit_idx, df_rf, dandi_filepath, combinations_xy, results_path, attributes, delay, duration):
    # Materialize h5py-backed columns into plain numpy so they can be pickled by multiprocessing
    df_units = df_units.copy()
    df_units['spike_times'] = [np.asarray(st) for st in df_units['spike_times']]
    df_rf = df_rf[['X', 'Y', 'Orientation', 'start_time', 'stop_time']].copy()

    worker = partial(_compute_unit, x_pos=x_pos, y_pos=y_pos, unique_orientations=unique_orientations,
                     df_units=df_units, df_rf=df_rf, combinations_xy=combinations_xy, delay=delay, duration=duration)

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(worker, unit_idx), total=len(unit_idx), desc='units'))

    # results[u] is a list of n_orientations dicts; pool.imap preserves order
    n_units = len(unit_idx)
    n_orientations = len(unique_orientations)
    nx, ny = len(x_pos), len(y_pos)

    mean_rates = np.zeros((n_units, n_orientations))
    mean_responses = np.zeros((n_units, n_orientations, nx, ny))
    z_score_responses = np.zeros((n_units, n_orientations, nx, ny))
    p_values = np.zeros((n_units, n_orientations))

    for u_i, unit_rows in enumerate(results):
        for o_i, row in enumerate(unit_rows):
            mean_rates[u_i, o_i] = row['mean_rate']
            mean_responses[u_i, o_i] = row['mean_response']
            z_score_responses[u_i, o_i] = row['z_score_response']
            p_values[u_i, o_i] = row['p_value']

    results_path = Path(results_path).with_suffix('.h5')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving results to {results_path}...')
    with h5py.File(results_path, 'w') as hf:
        for key, value in attributes.items():
            hf.attrs[key] = value
        hf.create_dataset('unit_names', data=df_units.loc[unit_idx, 'unit_name'].values.astype('S'))
        hf.create_dataset('orientations', data=unique_orientations.astype(float))
        hf.create_dataset('x_positions', data=x_pos)
        hf.create_dataset('y_positions', data=y_pos)
        hf.create_dataset('mean_rate', data=mean_rates)
        hf.create_dataset('mean_response', data=mean_responses, compression='gzip', compression_opts=4)
        hf.create_dataset('z_score_response', data=z_score_responses, compression='gzip', compression_opts=4)
        hf.create_dataset('p_value', data=p_values)


def _optimize_probe_files(h5_files, output_path):
    """Optimize delay/duration for a single probe's files (all must share the same unit count)."""
    with h5py.File(h5_files[0], 'r', locking=False) as hf:
        unit_names = hf['unit_names'][:]
        orientations = hf['orientations'][:]
        x_positions = hf['x_positions'][:]
        y_positions = hf['y_positions'][:]
        session_attrs = dict(hf.attrs)

    n_units = len(unit_names)
    n_orientations = len(orientations)
    nx, ny = len(x_positions), len(y_positions)

    best_p_values = np.full((n_units, n_orientations), np.inf)
    best_mean_rate = np.zeros((n_units, n_orientations))
    best_mean_response = np.zeros((n_units, n_orientations, nx, ny))
    best_z_score_response = np.zeros((n_units, n_orientations, nx, ny))
    best_delay = np.full((n_units, n_orientations), np.nan)
    best_duration = np.full((n_units, n_orientations), np.nan)

    for fpath in tqdm(h5_files, desc=str(output_path.name)):
        with h5py.File(fpath, 'r', locking=False) as hf:
            p_values = hf['p_value'][:]
            mean_rate = hf['mean_rate'][:]
            mean_response = hf['mean_response'][:]
            z_score_response = hf['z_score_response'][:]
            delay = hf.attrs['delay_s']
            duration = hf.attrs['duration_s']

        better = p_values < best_p_values                           # (n_units, n_orientations)
        best_p_values = np.where(better, p_values, best_p_values)
        best_mean_rate = np.where(better, mean_rate, best_mean_rate)
        best_delay = np.where(better, delay, best_delay)
        best_duration = np.where(better, duration, best_duration)

        better_4d = better[:, :, np.newaxis, np.newaxis]           # broadcast to spatial dims
        best_mean_response = np.where(better_4d, mean_response, best_mean_response)
        best_z_score_response = np.where(better_4d, z_score_response, best_z_score_response)

    output_path = Path(output_path).with_suffix('.h5')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving optimized results to {output_path}...')
    with h5py.File(output_path, 'w', locking=False) as hf:
        for key, value in session_attrs.items():
            if key not in ('delay_s', 'duration_s'):
                hf.attrs[key] = value
        hf.attrs['date_optimized'] = datetime.today().strftime('%Y-%m-%d')
        hf.create_dataset('unit_names', data=unit_names)
        hf.create_dataset('orientations', data=orientations)
        hf.create_dataset('x_positions', data=x_positions)
        hf.create_dataset('y_positions', data=y_positions)
        hf.create_dataset('best_delay', data=best_delay)
        hf.create_dataset('best_duration', data=best_duration)
        hf.create_dataset('p_value', data=best_p_values)
        hf.create_dataset('mean_rate', data=best_mean_rate)
        hf.create_dataset('mean_response', data=best_mean_response, compression='gzip', compression_opts=4)
        hf.create_dataset('z_score_response', data=best_z_score_response, compression='gzip', compression_opts=4)

    return output_path


def optimize_over_delay_duration(session_dir, output_path=None):
    """For each (unit, orientation), select the (delay, duration) with the lowest p-value
    across all HDF5 files in session_dir and save the corresponding spatial maps.

    Files are grouped by probe before comparison because different probes have different
    unit counts. One output file is written per probe."""
    from collections import defaultdict

    session_dir = Path(session_dir)
    # Exclude any previously written optimized files to avoid mixing them in
    # Split-trial files live beside these and must not enter the full-block reduction.
    h5_files = sorted(
        f for f in session_dir.glob('*.h5')
        if 'optimized' not in f.stem and '__trial_' not in f.stem
    )
    if not h5_files:
        raise FileNotFoundError(f'No .h5 files found in {session_dir}')

    # Filename pattern: {session}__rf-spike-count__{probe}__delay_{d}__duration_{dur}.h5
    probe_files = defaultdict(list)
    for fpath in h5_files:
        probe = fpath.stem.split('__')[2]
        probe_files[probe].append(fpath)

    saved = []
    for probe, files in sorted(probe_files.items()):
        if output_path is not None:
            op = Path(output_path)
            probe_output = op.parent / f'{op.stem}__{probe}{op.suffix}'
        else:
            probe_output = session_dir / f'{session_dir.name}__rf-spike-count__{probe}__optimized.h5'
        saved.append(_optimize_probe_files(files, probe_output))

    return saved[0] if len(saved) == 1 else saved


def main(nwb_path, probe_idx, results_dir=RESULTS_DIR):
    delays = np.array([0., .025, 0.5, .1]) * pq.s
    durations = np.array([0.1, .2, .3]) * pq.s

    stream = utils.open_local(nwb_path)
    units_df = stream.units_df()
    probes = units_df['probe'].unique()
    probe = probes[probe_idx]

    df_rf = stream.gabor_rf_df()
    unit_idx = units_df.loc[units_df['probe'] == probe].index.values

    unique_x = np.unique(df_rf['X'])
    unique_y = np.unique(df_rf['Y'])
    unique_orientations = np.unique(df_rf['Orientation'])

    x_pos = np.sort(unique_x.astype(float))
    y_pos = np.sort(unique_y.astype(float))
    combinations_xy = [(x, y) for x in unique_x for y in unique_y]

    results_dir = Path(results_dir) / Path(nwb_path).stem

    for delay in delays:
        for duration in durations:
            filename = f'{Path(nwb_path).stem}__rf-spike-count__{probe}__delay_{delay.magnitude}__duration_{duration.magnitude}.h5'
            results_path = results_dir / filename

            attributes = {
                'session': Path(nwb_path).stem,
                'probe': probe,
                'date_computed': datetime.today().strftime('%Y-%m-%d'),
                'delay_s': delay.magnitude,
                'duration_s': duration.magnitude,
            }

            compute_siegle(
                x_pos,
                y_pos,
                unique_orientations,
                units_df, #units_df.loc[unit_idx[:10]],
                unit_idx,#[:10],
                df_rf,
                Path(nwb_path).stem,
                combinations_xy,
                results_path,
                attributes,
                delay, 
                duration,
            )


def recompute_current_session(dandiset, asset_path, results_dir, probes=None, n_shuffle=1000):
    """Full five-repeat Siegle Gabors for the current NWB, one file per probe.

    Delay and duration are the same grid as ``main``. Each (x, y, orientation)
    is required to have five presentations; the block is not split in half.
    Files land in ``results/gabors/ephys/<session>/`` and replace the copies
    whose unit ids came from the older NWB. Those older files are moved to
    ``legacy_misaligned_units/`` so the reduction does not mix the two id lists.
    """
    from compute_rf_siegle_gabors_split_trials import compute_half, open_stream, _text
    import rf_siegle_ophys as siegle

    delays = (0.0, 0.025, 0.5, 0.1)
    durations = (0.1, 0.2, 0.3)
    stream, session_name = open_stream(None, dandiset, asset_path)
    session_dir = Path(results_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    legacy = session_dir / "legacy_misaligned_units"
    legacy.mkdir(exist_ok=True)
    for path in session_dir.glob("*.h5"):
        if "__trial_" in path.name:
            continue
        destination = legacy / path.name
        if not destination.exists():
            path.replace(destination)
            print(f"moved aside {path.name}")

    with stream:
        units = stream.units_df(include_spikes=True)
        df_rf = stream.gabor_rf_df()
    units = units.copy()
    units["probe"] = units["probe"].map(_text)
    units["unit_name"] = units["unit_name"].map(_text)
    available = list(pd.unique(units["probe"]))
    selected = available if not probes else list(probes)
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Probes {missing} not in {available}")

    counts = df_rf.groupby(["X", "Y", "Orientation"], observed=True).size()
    print(
        f"{len(df_rf)} presentations, repeats per cell {int(counts.min())}–{int(counts.max())} "
        f"({(counts == 5).sum()} cells with 5, {(counts < 5).sum()} short)"
    )
    if int(counts.max()) != 5 or int(counts.min()) < 1:
        raise RuntimeError(
            "The full RF block is not a five-repeat grid: "
            f"{int(counts.min())}–{int(counts.max())} presentations per cell"
        )
    if int(counts.min()) < 5:
        print("Short cells keep their presented trials. Missing trials are left out, not filled with zeros.")
    x_pos, y_pos, orientations = siegle.rf_grid(df_rf)
    print(f"grid {len(x_pos)} x {len(y_pos)}, orientations {len(orientations)}")

    # VISp is on Probe C, so that probe is finished before the others.
    ordered = [name for name in selected if name == "ProbeC"] + [name for name in selected if name != "ProbeC"]
    for probe_name in ordered:
        on_probe = units.loc[units["probe"] == probe_name].drop_duplicates("unit_name")
        print(f"\n{probe_name}: {len(on_probe)} units, full block")
        sweep = []
        for delay in delays:
            for duration in durations:
                filename = (
                    f"{session_name}__rf-spike-count__{probe_name}"
                    f"__delay_{float(delay)}__duration_{float(duration)}.h5"
                )
                path = session_dir / filename
                sweep.append(path)
                attributes = {
                    "session": session_name,
                    "probe": probe_name,
                    "n_presentations": int(len(df_rf)),
                    "n_repeats": 5,
                    "split": "full_block",
                    "n_shuffle": int(n_shuffle),
                    "date_computed": datetime.today().strftime("%Y-%m-%d"),
                    "delay_s": float(delay),
                    "duration_s": float(duration),
                    "source_asset": asset_path,
                }
                print(f"\n{probe_name}  delay={float(delay)}s  duration={float(duration)}s")
                compute_half(
                    on_probe, df_rf, x_pos, y_pos, orientations,
                    float(delay), float(duration), path, attributes, n_shuffle,
                )
        optimized = session_dir / f"{session_name}__rf-spike-count__{probe_name}__optimized.h5"
        _optimize_probe_files(sweep, optimized)
        print(f"optimized {probe_name} -> {optimized.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-idx", type=int, default=None)
    parser.add_argument("--nwb-path", default=None)
    parser.add_argument("--results-path", default=RESULTS_DIR)
    parser.add_argument("--dandiset", default=None)
    parser.add_argument("--asset-path", default=None)
    parser.add_argument("--probe", action="append", default=None,
                        help="Probe name to recompute from the current NWB. Repeat for several. Default: all probes.")
    parser.add_argument("--n-shuffle", type=int, default=1000)
    args = parser.parse_args()
    if args.dandiset or args.asset_path:
        if not args.dandiset or not args.asset_path:
            parser.error("Recomputing from DANDI needs both --dandiset and --asset-path")
        repo_results = Path(__file__).resolve().parents[2] / "results" / "gabors" / "ephys"
        results = args.results_path if args.results_path != RESULTS_DIR else repo_results
        recompute_current_session(
            args.dandiset, args.asset_path, results, args.probe, args.n_shuffle,
        )
    else:
        if args.nwb_path is None or args.probe_idx is None:
            parser.error("Pass --nwb-path and --probe-idx, or --dandiset and --asset-path")
        main(args.nwb_path, args.probe_idx, args.results_path)


