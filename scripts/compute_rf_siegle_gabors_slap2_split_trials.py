"""Siegle Gabor RFs on each half of the RF block, SLAP2.

Same trace analysis as ``compute_rf_siegle_gabors_slap2.py``. The earlier half
of the RF-mapping presentations is trial 0 and the later half is trial 1.
Delay and duration are copied from the full-block ``*__optimized.h5``; each
half is scored at those windows and does not pick a new minimum p.
Files land next to the full-block results::

    results/gabors/slap2/<session>/<signal>/<session>__rf-<signal>-<channel>__<dmd>__trial_<t>__....h5

SLAP2 has no events in the NWB. This script uses ``utils.deconvolution`` when
that module is importable, and otherwise the OASIS event train from
``oasis_deconvolve_slap2.py`` (same tau per channel).

    cd code/scripts
    python compute_rf_siegle_gabors_slap2_split_trials.py --dmd DMD1 --channel green \\
        --nwb-path ../../data/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parent))

import rf_siegle_ophys as siegle
import utils
from oasis_deconvolve_slap2 import TAU_D, deconvolve_bouts

# Same sweep as compute_rf_siegle_gabors_slap2.py. Imported here rather than
# from that module, which requires utils.deconvolution at import time.
STIM_TABLE = "rf_mapping"
DELAYS = np.array([0.0, 0.025, 0.05, 0.1, 0.2])
DURATIONS = np.array([0.1, 0.25, 0.5])

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results" / "gabors" / "slap2"


def _bouts(times: np.ndarray) -> pd.DataFrame:
    """Contiguous stretches of the windowed timestamps. Indices match ``times``."""
    if len(times) == 0:
        return pd.DataFrame(columns=["i_start", "i_stop"])
    dt = np.diff(times)
    gaps = np.where(dt > 5 * np.median(dt))[0] if len(dt) else np.array([], dtype=int)
    starts = np.concatenate([[0], gaps + 1])
    stops = np.concatenate([gaps, [len(times) - 1]])
    return pd.DataFrame({"i_start": starts, "i_stop": stops})


def oasis_events(stream, dmd, channel, t_start, t_stop):
    """OASIS event amplitudes over the requested window. Shape (n_rois, n_t)."""
    dff = stream.slap2_dff(dmd, channel, t_start=t_start, t_stop=t_stop)
    traces = dff.to_numpy(dtype=np.float64)
    times = dff.columns.to_numpy(dtype=float)
    segments = _bouts(times)
    events = np.full_like(traces, np.nan)
    tau = TAU_D[channel]
    for roi in range(traces.shape[0]):
        _, event_train = deconvolve_bouts(traces[roi], times, segments, tau)
        events[roi] = event_train
    roi_ids = stream.slap2_rois(dmd)["roi_id"].to_numpy()
    unit_names = [f"{dmd}_{channel}_roi{roi_id}" for roi_id in roi_ids]
    return events, times, unit_names


def load_traces(stream, dmd, channel, t_start, t_stop, signal, nwb_path):
    if signal == "dff":
        dff = stream.slap2_dff(dmd, channel, t_start=t_start, t_stop=t_stop)
        roi_ids = stream.slap2_rois(dmd)["roi_id"].to_numpy()
        names = [f"{dmd}_{channel}_roi{roi_id}" for roi_id in roi_ids]
        return dff.to_numpy(dtype=np.float64), dff.columns.to_numpy(dtype=float), names
    if signal != "events":
        raise ValueError(f"signal must be 'dff' or 'events', got {signal!r}")
    try:
        from utils import deconvolution
    except ImportError:
        deconvolution = None
    if deconvolution is not None and hasattr(deconvolution, "slap2_events_df"):
        frame = deconvolution.slap2_events_df(
            nwb_path, dmd, channel, t_start=t_start, t_stop=t_stop, stream=stream,
        )
        roi_ids = stream.slap2_rois(dmd)["roi_id"].to_numpy()
        names = [f"{dmd}_{channel}_roi{roi_id}" for roi_id in roi_ids]
        return frame.to_numpy(dtype=np.float64), frame.columns.to_numpy(dtype=float), names
    print(f"utils.deconvolution is not available; using OASIS events (tau={TAU_D[channel]} s)")
    return oasis_events(stream, dmd, channel, t_start, t_stop)


def main(nwb_path, dmd="DMD1", results_dir=RESULTS_DIR, channel="green", signal="events",
         baseline=0.0, n_shuffle=siegle.N_SHUFFLE, optimize=True, recompute=False):
    session_name = Path(nwb_path).stem
    results_dir = Path(results_dir) / session_name / signal
    print(f"Opening {nwb_path}")
    stream = utils.open_local(nwb_path)
    with stream:
        if dmd not in stream.dmds():
            raise ValueError(f"{dmd!r} not in this session; available: {stream.dmds()}")
        df_rf = stream.stim_df(STIM_TABLE)
        x_pos, y_pos, orientations = siegle.rf_grid(df_rf)
        margin = baseline + float(np.max(DELAYS)) + float(np.max(DURATIONS)) + 1.0
        traces, times, unit_names = load_traces(
            stream, dmd, channel,
            t_start=float(df_rf["start_time"].min()) - margin,
            t_stop=float(df_rf["stop_time"].max()) + margin,
            signal=signal, nwb_path=nwb_path,
        )
    n_raw = traces.shape[1]
    traces, times = siegle.drop_missing_samples(traces, times)
    print(f"{dmd}/{channel} ({signal}): {traces.shape[0]} ROIs, {traces.shape[1]} samples "
          f"({n_raw - traces.shape[1]} blanked), {len(df_rf)} presentations")

    halves = siegle.split_gabor_block(df_rf)
    tag = f"rf-{signal}-{channel}"
    for trial, df_half in halves.items():
        print(f"trial {trial}: {len(df_half)} presentations")
        sweep = []
        for delay in DELAYS:
            for duration in DURATIONS:
                filename = (
                    f"{session_name}__{tag}__{dmd}"
                    f"__trial_{trial}__delay_{float(delay):g}__duration_{float(duration):g}.h5"
                )
                path = results_dir / filename
                sweep.append(path)
                if path.exists() and not recompute:
                    print(f"exists, skipping {path.name}")
                    continue
                attributes = {
                    "session": session_name,
                    "modality": "slap2",
                    "dmd": dmd,
                    "probe": dmd,
                    "channel": channel,
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
                print(f"\n{dmd}/{channel} trial {trial}  delay={float(delay):g}s  duration={float(duration):g}s")
                siegle.compute_siegle_ophys(
                    traces, times, unit_names, df_half, x_pos, y_pos, orientations,
                    delay=float(delay), duration=float(duration),
                    results_path=path, attributes=attributes,
                    baseline=baseline, n_shuffle=n_shuffle,
                )
        if not optimize:
            continue
        full_run = results_dir / f"{session_name}__{tag}__{dmd}__optimized.h5"
        optimized = results_dir / f"{session_name}__{tag}__{dmd}__trial_{trial}__optimized.h5"
        siegle.assemble_fixed_windows(full_run, sweep, optimized)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", required=True)
    parser.add_argument("--dmd", default="DMD1")
    parser.add_argument("--results-path", default=str(RESULTS_DIR))
    parser.add_argument("--channel", default="green", choices=["green", "red"])
    parser.add_argument("--signal", default="events", choices=["dff", "events"])
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--n-shuffle", type=int, default=siegle.N_SHUFFLE)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    main(
        args.nwb_path, args.dmd, args.results_path, args.channel, args.signal,
        args.baseline, args.n_shuffle, optimize=not args.no_optimize, recompute=args.recompute,
    )
