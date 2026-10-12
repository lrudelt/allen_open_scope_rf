"""Siegle receptive field mapping of the local Gabor patches, mesoscope sessions.

The mesoscope counterpart of `compute_rf_siegle_gabors.py`: instead of a spike
rate per presentation it uses the mean ΔF/F (or deconvolved event amplitude) of
each ROI over the response window. One HDF5 file is written per
(plane, delay, duration) into <results-path>/<session>/<signal>/, then the grid
is reduced to one optimized file per plane, exactly as for the probes.

    python compute_rf_siegle_gabors_mesoscope.py \
        --nwb-path ../../../rawdata/allen_open_scope/meso/sub-832700/sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys.nwb \
        --plane-idx 0
"""

import numpy as np
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append('..')
import utils
import rf_siegle_ophys as siegle

RESULTS_DIR = os.path.abspath('../../../results/allen_open_scope/rf/gabors/')
STIM_TABLE = 'RF mapping_presentations'

# Imaging runs at ~9.5 Hz and GCaMP is slow, so the windows are longer than the
# ones used for spikes; durations beyond the 0.267 s ISI reach into neighbouring
# presentations, which are randomised and therefore add noise but no bias.
DELAYS = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
DURATIONS = np.array([0.25, 0.5, 1.0])


def load_traces(stream, plane, signal='dff'):
    """(traces, times, unit_names) for one imaging plane.

    signal: 'dff' for ΔF/F, 'events' for the deconvolved event series — the
    latter is non-negative and so is the closer analogue of a spike rate.
    """
    if signal == 'dff':
        df = stream.dff_df(plane)
        traces, times = df.to_numpy(dtype=np.float64), df.columns.to_numpy(dtype=float)
        roi_ids = stream.nwb.processing[plane]['dff_timeseries']['dff_timeseries'].rois.data[:]
    elif signal == 'events':
        ts = stream.nwb.processing[plane]['event_timeseries']
        traces = np.asarray(ts.data[:], dtype=np.float64).T   # stored (n_t, n_rois)
        times = ts.timestamps[:]
        roi_ids = ts.rois.data[:]
    else:
        raise ValueError(f"signal must be 'dff' or 'events', got {signal!r}")

    unit_names = [f'{plane}_roi{roi_id}' for roi_id in roi_ids]
    return traces, times, unit_names


def main(nwb_path, plane_idx, results_dir=RESULTS_DIR, signal='events', baseline=0.0,
         n_shuffle=siegle.N_SHUFFLE, optimize=True):
    session_name = Path(nwb_path).stem
    # One tree per signal: readers key the optimized files on the plane alone,
    # so dff and events maps sharing a directory would silently shadow each other
    results_dir = Path(results_dir) / session_name / signal

    print(f'Loading {nwb_path}...')
    stream = utils.open_local(nwb_path)
    plane = stream.imaging_planes()[plane_idx]

    df_rf = stream.stim_df(STIM_TABLE)
    x_pos, y_pos, orientations = siegle.rf_grid(df_rf)

    traces, times, unit_names = load_traces(stream, plane, signal)
    traces, times = siegle.drop_missing_samples(traces, times)
    traces, times = siegle.slice_to_stimulus(
        traces, times, df_rf, margin=baseline + DELAYS.max() + DURATIONS.max() + 1.0
    )
    print(f'{plane}: {traces.shape[0]} ROIs, {traces.shape[1]} samples over the RF block, '
          f'{len(df_rf)} presentations, {len(x_pos)}x{len(y_pos)} grid, '
          f'{len(orientations)} orientations')

    tag = f'rf-{signal}'
    for delay in DELAYS:
        for duration in DURATIONS:
            filename = (f'{session_name}__{tag}__{plane}'
                        f'__delay_{delay}__duration_{duration}.h5')

            attributes = {
                'session': session_name,
                'modality': 'mesoscope',
                'plane': plane,
                'probe': plane,          # keeps the ecephys reducer's naming happy
                'signal': signal,
                'baseline_s': baseline,
                'n_shuffle': n_shuffle,
                'date_computed': datetime.today().strftime('%Y-%m-%d'),
                'delay_s': float(delay),
                'duration_s': float(duration),
            }

            print(f'\n{plane}  delay={delay}s  duration={duration}s')
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
            pattern=f'*__{tag}__{plane}__delay_*.h5',
            output_path=results_dir / f'{session_name}__{tag}__{plane}__optimized.h5',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb-path', required=True, help='Path to the mesoscope NWB file')
    parser.add_argument('--plane-idx', type=int, required=True,
                        help='Index into stream.imaging_planes(), 0-based')
    parser.add_argument('--results-path', default=RESULTS_DIR, help='Root directory for results')
    parser.add_argument('--signal', default='events', choices=['dff', 'events'],
                        help="Trace to analyse: the OASIS-deconvolved 'events' (default) "
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
    main(args.nwb_path, args.plane_idx, args.results_path, args.signal,
         args.baseline, args.n_shuffle, optimize=not args.no_optimize)
