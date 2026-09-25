"""Siegle receptive field mapping of the local Gabor patches, SLAP2 sessions.

The SLAP2 counterpart of `compute_rf_siegle_gabors.py`: instead of a spike rate
per presentation it uses the mean ΔF/F (or deconvolved event amplitude) of each
ROI over the response window. One HDF5 file is written per
(DMD, channel, delay, duration) into <results-path>/<session>/<signal>/, then
the grid is reduced to one optimized file per DMD, exactly as for the probes.

SLAP2 does not image continuously — it records in ~30 s bouts and blanks short
stretches of samples to NaN — so presentations that are not sampled densely
enough are dropped from the test rather than averaged over a hole.

The test runs on the deconvolved events by default; `--signal dff` restores the
raw ΔF/F. Unlike the mesoscope release, the SLAP2 NWB files carry no events, so
`utils.deconvolution` computes and caches them per session, DMD and channel on
first use.

    python compute_rf_siegle_gabors_slap2.py \
        --nwb-path ../../../rawdata/allen_open_scope/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb \
        --dmd DMD1
"""

import numpy as np
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append('..')
import utils
from utils import deconvolution
import rf_siegle_ophys as siegle

RESULTS_DIR = os.path.abspath('../../../results/allen_open_scope/rf/gabors/')
STIM_TABLE = 'rf_mapping'

# SLAP2 samples at ~200 Hz and the green channel carries iGluSnFR4f, which is
# far faster than GCaMP, so short windows are worth testing here.
DELAYS = np.array([0.0, 0.025, 0.05, 0.1, 0.2])
DURATIONS = np.array([0.1, 0.25, 0.5])


def load_traces(stream, dmd, channel, t_start, t_stop, signal='dff', nwb_path=None):
    """(traces, times, unit_names) for one DMD, restricted to [t_start, t_stop].

    signal: 'dff' for ΔF/F, 'events' for the deconvolved series — the latter is
    non-negative and so the closer analogue of a spike rate.
    """
    if signal == 'dff':
        df = stream.slap2_dff(dmd, channel, t_start=t_start, t_stop=t_stop)
    elif signal == 'events':
        df = deconvolution.slap2_events_df(nwb_path, dmd, channel, t_start=t_start,
                                           t_stop=t_stop, stream=stream)
    else:
        raise ValueError(f"signal must be 'dff' or 'events', got {signal!r}")

    traces = df.to_numpy(dtype=np.float64)
    times = df.columns.to_numpy(dtype=float)
    roi_ids = stream.slap2_rois(dmd)['roi_id'].to_numpy()
    unit_names = [f'{dmd}_{channel}_roi{roi_id}' for roi_id in roi_ids]
    return traces, times, unit_names


def main(nwb_path, dmd, results_dir=RESULTS_DIR, channel='green', signal='events',
         baseline=0.0, n_shuffle=siegle.N_SHUFFLE, optimize=True):
    session_name = Path(nwb_path).stem
    # One tree per signal: readers key the optimized files on the DMD alone,
    # so ΔF/F and events must not share a directory.
    results_dir = Path(results_dir) / session_name / signal

    print(f'Loading {nwb_path}...')
    stream = utils.open_local(nwb_path)
    if dmd not in stream.dmds():
        raise ValueError(f'{dmd!r} not in this session; available: {stream.dmds()}')

    df_rf = stream.stim_df(STIM_TABLE)
    x_pos, y_pos, orientations = siegle.rf_grid(df_rf)

    # only the RF block is read off disk — the full traces are several hundred MB
    margin = baseline + DELAYS.max() + DURATIONS.max() + 1.0
    traces, times, unit_names = load_traces(
        stream, dmd, channel,
        t_start=df_rf['start_time'].min() - margin,
        t_stop=df_rf['stop_time'].max() + margin,
        signal=signal, nwb_path=nwb_path,
    )
    n_raw = traces.shape[1]
    traces, times = siegle.drop_missing_samples(traces, times)
    print(f'{dmd}/{channel} ({signal}): {traces.shape[0]} ROIs, {traces.shape[1]} samples over the '
          f'RF block ({n_raw - traces.shape[1]} blanked), {len(df_rf)} presentations, '
          f'{len(x_pos)}x{len(y_pos)} grid, {len(orientations)} orientations')

    tag = f'rf-{signal}-{channel}'
    for delay in DELAYS:
        for duration in DURATIONS:
            filename = (f'{session_name}__{tag}__{dmd}'
                        f'__delay_{delay}__duration_{duration}.h5')

            attributes = {
                'session': session_name,
                'modality': 'slap2',
                'dmd': dmd,
                'probe': dmd,            # keeps the ecephys reducer's naming happy
                'channel': channel,
                'signal': signal,
                'baseline_s': baseline,
                'n_shuffle': n_shuffle,
                'date_computed': datetime.today().strftime('%Y-%m-%d'),
                'delay_s': float(delay),
                'duration_s': float(duration),
            }

            print(f'\n{dmd}/{channel}  delay={delay}s  duration={duration}s')
            siegle.compute_siegle_ophys(
                traces, times, unit_names, df_rf, x_pos, y_pos, orientations,
                delay=float(delay), duration=float(duration),
                results_path=results_dir / filename,
                attributes=attributes,
                baseline=baseline,
                n_shuffle=n_shuffle,
            )

    if optimize:
        print('\nOptimizing over the delay x duration grid...')
        siegle.optimize_over_delay_duration(
            results_dir,
            pattern=f'*__{tag}__{dmd}__delay_*.h5',
            output_path=results_dir / f'{session_name}__{tag}__{dmd}__optimized.h5',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb-path', required=True, help='Path to the SLAP2 NWB file')
    parser.add_argument('--dmd', default='DMD1', help='DMD imaging plane, e.g. DMD1 or DMD2')
    parser.add_argument('--results-path', default=RESULTS_DIR, help='Root directory for results')
    parser.add_argument('--channel', default='green', choices=['green', 'red'],
                        help="'green' (iGluSnFR4f, default) or 'red' (RCaMP3)")
    parser.add_argument('--signal', default='events', choices=['dff', 'events'],
                        help="Trace to analyse: the deconvolved 'events' (default), "
                             'computed and cached by utils.deconvolution on first use, '
                             "or raw 'dff'. Results go to <results-path>/<session>/<signal>/")
    parser.add_argument('--baseline', type=float, default=0.0,
                        help='Seconds of pre-stimulus signal to subtract per trial. '
                             'Default 0 (no subtraction), matching the spike-rate version; '
                             'note the 0.267 s ISI leaves no truly blank baseline.')
    parser.add_argument('--n-shuffle', type=int, default=siegle.N_SHUFFLE,
                        help='Permutations used for the p-value')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip the delay x duration reduction step')

    args = parser.parse_args()
    main(args.nwb_path, args.dmd, args.results_path, args.channel, args.signal,
         args.baseline, args.n_shuffle, optimize=not args.no_optimize)
