"""Siegle Gabor RFs on each half of the RF block, ecephys.

Same spike-count analysis as ``compute_rf_siegle_gabors.py``. Presentations are
sorted by onset and split in half: the earlier half is trial 0 and the later
half is trial 1, matching the two Zebra repeats. Delay and duration are copied
from the full-block ``*__optimized.h5``; each half is scored at those windows
and does not pick a new minimum p. Files are written next to the
full-block results::

    results/gabors/ephys/<session>/<session>__rf-spike-count__<probe>__trial_<t>__delay_<d>__duration_<u>.h5
    results/gabors/ephys/<session>/<session>__rf-spike-count__<probe>__trial_<t>__optimized.h5

    cd code/scripts
    python compute_rf_siegle_gabors_split_trials.py --probe ProbeC \\
        --dandiset 001637 \\
        --asset-path sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb
"""
from __future__ import annotations

import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import h5py
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parent))

import rf_siegle_ophys as siegle
import utils

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results" / "gabors" / "ephys"

# Same sweep as compute_rf_siegle_gabors.main.
DELAYS = (0.0, 0.025, 0.5, 0.1)
DURATIONS = (0.1, 0.2, 0.3)


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value)
    if len(text) >= 3 and text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    return text


def open_stream(nwb_path, dandiset, asset_path):
    if nwb_path:
        print(f"Opening {nwb_path}")
        return utils.open_local(nwb_path), Path(nwb_path).stem
    if not dandiset or not asset_path:
        raise ValueError("Pass --nwb-path, or both --dandiset and --asset-path")
    session = utils.DandiSession(dandiset)
    assets = session.assets()
    hits = [a for a in assets if a.path == asset_path]
    if not hits:
        raise ValueError(f"Asset path {asset_path!r} not found in dandiset {dandiset}")
    asset = hits[0]
    print(f"Streaming {asset.path}")
    return session.open(asset.identifier), Path(asset_path).stem


def presentation_rates(spike_times, onsets, offsets, delay, duration):
    """Spike rate in [onset + delay, onset + delay + duration).

    The divisor is the stimulus duration, as in ``compute_rf_siegle_gabors``.
    """
    onsets = np.asarray(onsets, dtype=float)
    stim = np.maximum(np.asarray(offsets, dtype=float) - onsets, 1e-9)
    spikes = np.asarray(spike_times, dtype=float)
    if spikes.size == 0:
        return np.zeros(len(onsets))
    starts = onsets + delay
    i0 = np.searchsorted(spikes, starts, side="left")
    i1 = np.searchsorted(spikes, starts + duration, side="left")
    return (i1 - i0) / stim


def rate_grids(spike_list, df_half, x_pos, y_pos, orientations, delay, duration):
    """(n_units, n_orientations, n_repeats, nx, ny), NaN where that half has no trial."""
    onsets = df_half["start_time"].to_numpy(float)
    offsets = df_half["stop_time"].to_numpy(float)
    o_i, r_i, x_i, y_i = siegle.grid_indices(df_half, x_pos, y_pos, orientations)
    shape = (len(orientations), int(r_i.max()) + 1, len(x_pos), len(y_pos))
    grids = np.full((len(spike_list),) + shape, np.nan)
    for unit, spikes in enumerate(spike_list):
        rates = presentation_rates(spikes, onsets, offsets, delay, duration)
        grids[unit, o_i, r_i, x_i, y_i] = rates
    return grids


def _write_h5(path, attributes, unit_names, orientations, x_pos, y_pos, results, n_valid):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_rates = np.stack([row[0] for row in results])
    mean_responses = np.stack([row[1] for row in results])
    z_scores = np.stack([row[2] for row in results])
    p_values = np.stack([row[3] for row in results])
    print(f"Saving results to {path}...")
    with h5py.File(path, "w") as hf:
        for key, value in attributes.items():
            hf.attrs[key] = value
        hf.create_dataset("unit_names", data=np.asarray(unit_names).astype("S"))
        hf.create_dataset("orientations", data=np.asarray(orientations, dtype=float))
        hf.create_dataset("x_positions", data=np.asarray(x_pos, dtype=float))
        hf.create_dataset("y_positions", data=np.asarray(y_pos, dtype=float))
        hf.create_dataset("n_valid_trials", data=n_valid)
        hf.create_dataset("mean_rate", data=mean_rates)
        hf.create_dataset("mean_response", data=mean_responses, compression="gzip", compression_opts=4)
        hf.create_dataset("z_score_response", data=z_scores, compression="gzip", compression_opts=4)
        hf.create_dataset("p_value", data=p_values)


def compute_half(units, df_half, x_pos, y_pos, orientations, delay, duration,
                 results_path, attributes, n_shuffle):
    spikes = [np.asarray(st, dtype=float) for st in units["spike_times"]]
    grids = rate_grids(spikes, df_half, x_pos, y_pos, orientations, delay, duration)
    n_valid = np.isfinite(grids[0]).sum(axis=1) if len(grids) else np.zeros((len(orientations), len(x_pos), len(y_pos)))
    worker = partial(siegle._unit_stats, n_shuffle=n_shuffle)
    tasks = ((i, grids[i]) for i in range(len(grids)))
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(grids), desc="units"))
    _write_h5(
        results_path, attributes, units["unit_name"].map(_text).tolist(),
        orientations, x_pos, y_pos, results, n_valid,
    )


def main(nwb_path=None, probe_idx=None, probe_name=None, results_dir=RESULTS_DIR,
         dandiset=None, asset_path=None, n_shuffle=siegle.N_SHUFFLE, recompute=False):
    stream, session_name = open_stream(nwb_path, dandiset, asset_path)
    with stream:
        units = stream.units_df(include_spikes=False)
        df_rf = stream.gabor_rf_df()
        units = units.copy()
        units["probe"] = units["probe"].map(_text)
        units["unit_name"] = units["unit_name"].map(_text)
        probes = list(pd.unique(units["probe"]))
        if probe_name is None:
            if probe_idx is None:
                raise ValueError(f"Pass --probe or --probe-idx. Probes: {probes}")
            probe_name = probes[probe_idx]
        elif probe_name not in probes:
            raise ValueError(f"{probe_name!r} not in {probes}")
        on_probe = units.loc[units["probe"] == probe_name]
        session_dir = Path(results_dir) / session_name
        full_run = session_dir / f"{session_name}__rf-spike-count__{probe_name}__optimized.h5"
        if not full_run.is_file():
            raise FileNotFoundError(
                f"Full-run optimized file not found: {full_run}. "
                "Delay and duration are taken from that file and are not fit again."
            )
        with h5py.File(full_run, "r") as hf:
            saved_order = [_text(name) for name in hf["unit_names"][:].astype(str)]
        known = set(on_probe["unit_name"])
        missing = [unit_id for unit_id in saved_order if unit_id not in known]
        if missing:
            raise RuntimeError(
                f"{full_run.name} lists {len(saved_order)} units, and {len(missing)} of "
                "them are not in this NWB probe. The saved delay and duration cannot "
                "be applied unit by unit."
            )
        print(f"Using delay/duration from {full_run.name} for {len(saved_order)} units")
        spiked = stream.units_df(include_spikes=True)
    spiked = spiked.copy()
    spiked["probe"] = spiked["probe"].map(_text)
    spiked["unit_name"] = spiked["unit_name"].map(_text)
    units = (
        spiked.loc[spiked["probe"] == probe_name]
        .drop_duplicates("unit_name")
        .set_index("unit_name")
        .loc[saved_order]
        .reset_index()
    )

    x_pos, y_pos, orientations = siegle.rf_grid(df_rf)
    halves = siegle.split_gabor_block(df_rf)
    for trial, df_half in halves.items():
        print(f"trial {trial}: {len(df_half)} presentations "
              f"({df_half['start_time'].min():.1f}–{df_half['start_time'].max():.1f} s)")
        sweep = []
        for delay in DELAYS:
            for duration in DURATIONS:
                filename = (
                    f"{session_name}__rf-spike-count__{probe_name}"
                    f"__trial_{trial}__delay_{delay:g}__duration_{duration:g}.h5"
                )
                path = session_dir / filename
                sweep.append(path)
                if path.exists() and not recompute:
                    print(f"exists, skipping {path.name}")
                    continue
                attributes = {
                    "session": session_name,
                    "probe": probe_name,
                    "trial": int(trial),
                    "split": "first_half" if trial == 0 else "second_half",
                    "n_presentations": int(len(df_half)),
                    "n_shuffle": int(n_shuffle),
                    "date_computed": datetime.today().strftime("%Y-%m-%d"),
                    "delay_s": float(delay),
                    "duration_s": float(duration),
                }
                print(f"\n{probe_name} trial {trial}  delay={delay:g}s  duration={duration:g}s")
                compute_half(
                    units, df_half, x_pos, y_pos, orientations, delay, duration,
                    path, attributes, n_shuffle,
                )
        optimized = session_dir / (
            f"{session_name}__rf-spike-count__{probe_name}__trial_{trial}__optimized.h5"
        )
        siegle.assemble_fixed_windows(full_run, sweep, optimized)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", default=None)
    parser.add_argument("--dandiset", default=None)
    parser.add_argument("--asset-path", default=None)
    parser.add_argument("--probe-idx", type=int, default=None)
    parser.add_argument("--probe", default=None, help="Probe name, e.g. ProbeC")
    parser.add_argument("--results-path", default=str(RESULTS_DIR))
    parser.add_argument("--n-shuffle", type=int, default=siegle.N_SHUFFLE)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    main(
        args.nwb_path, args.probe_idx, args.probe, args.results_path,
        args.dandiset, args.asset_path, args.n_shuffle, args.recompute,
    )
