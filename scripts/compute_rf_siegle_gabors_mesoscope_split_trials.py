"""Siegle Gabor RFs on each half of the RF block, mesoscope.

Same trace analysis as ``compute_rf_siegle_gabors_mesoscope.py``. The earlier
half of the RF-mapping presentations is trial 0 and the later half is trial 1.
Delay and duration are copied from the full-block ``*__optimized.h5``; each
half is scored at those windows and does not pick a new minimum p.
Files land next to the full-block results::

    results/gabors/meso/<session>/<signal>/<session>__rf-<signal>__<plane>__trial_<t>__....h5

    cd code/scripts
    python compute_rf_siegle_gabors_mesoscope_split_trials.py --plane VISp_0 \\
        --nwb-path /path/to/session.nwb
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parent))

import rf_siegle_ophys as siegle
import utils
from compute_rf_siegle_gabors_mesoscope import DELAYS, DURATIONS, STIM_TABLE, load_traces

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results" / "gabors" / "meso"


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


def main(nwb_path=None, plane_idx=None, plane_name=None, results_dir=RESULTS_DIR,
         signal="events", baseline=0.0, n_shuffle=siegle.N_SHUFFLE, optimize=True,
         dandiset=None, asset_path=None, recompute=False):
    stream, session_name = open_stream(nwb_path, dandiset, asset_path)
    results_dir = Path(results_dir) / session_name / signal
    with stream:
        planes = stream.imaging_planes()
        if plane_name is None:
            if plane_idx is None:
                raise ValueError(f"Pass --plane or --plane-idx. Planes: {planes}")
            plane_name = planes[plane_idx]
        elif plane_name not in planes:
            raise ValueError(f"{plane_name!r} not in {planes}")

        df_rf = stream.stim_df(STIM_TABLE)
        x_pos, y_pos, orientations = siegle.rf_grid(df_rf)
        traces, times, unit_names = load_traces(stream, plane_name, signal)
        traces, times = siegle.drop_missing_samples(traces, times)
        traces, times = siegle.slice_to_stimulus(
            traces, times, df_rf, margin=baseline + float(DELAYS.max()) + float(DURATIONS.max()) + 1.0
        )

    halves = siegle.split_gabor_block(df_rf)
    tag = f"rf-{signal}"
    print(f"{plane_name}: {traces.shape[0]} ROIs, {len(df_rf)} presentations")
    for trial, df_half in halves.items():
        print(f"trial {trial}: {len(df_half)} presentations")
        sweep = []
        for delay in DELAYS:
            for duration in DURATIONS:
                filename = (
                    f"{session_name}__{tag}__{plane_name}"
                    f"__trial_{trial}__delay_{float(delay):g}__duration_{float(duration):g}.h5"
                )
                path = results_dir / filename
                sweep.append(path)
                if path.exists() and not recompute:
                    print(f"exists, skipping {path.name}")
                    continue
                attributes = {
                    "session": session_name,
                    "modality": "mesoscope",
                    "plane": plane_name,
                    "probe": plane_name,
                    "signal": signal,
                    "trial": int(trial),
                    "split": "first_half" if trial == 0 else "second_half",
                    "n_presentations": int(len(df_half)),
                    "baseline_s": baseline,
                    "n_shuffle": n_shuffle,
                    "date_computed": datetime.today().strftime("%Y-%m-%d"),
                    "delay_s": float(delay),
                    "duration_s": float(duration),
                }
                print(f"\n{plane_name} trial {trial}  delay={float(delay):g}s  duration={float(duration):g}s")
                siegle.compute_siegle_ophys(
                    traces, times, unit_names, df_half, x_pos, y_pos, orientations,
                    delay=float(delay), duration=float(duration),
                    results_path=path, attributes=attributes,
                    baseline=baseline, n_shuffle=n_shuffle,
                )
        if not optimize:
            continue
        full_run = results_dir / f"{session_name}__{tag}__{plane_name}__optimized.h5"
        optimized = results_dir / f"{session_name}__{tag}__{plane_name}__trial_{trial}__optimized.h5"
        siegle.assemble_fixed_windows(full_run, sweep, optimized)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", default=None)
    parser.add_argument("--dandiset", default=None)
    parser.add_argument("--asset-path", default=None)
    parser.add_argument("--plane-idx", type=int, default=None)
    parser.add_argument("--plane", default=None, help="Imaging plane, e.g. VISp_0")
    parser.add_argument("--results-path", default=str(RESULTS_DIR))
    parser.add_argument("--signal", default="events", choices=["dff", "events"])
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--n-shuffle", type=int, default=siegle.N_SHUFFLE)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    main(
        args.nwb_path, args.plane_idx, args.plane, args.results_path, args.signal,
        args.baseline, args.n_shuffle, optimize=not args.no_optimize,
        dandiset=args.dandiset, asset_path=args.asset_path, recompute=args.recompute,
    )
