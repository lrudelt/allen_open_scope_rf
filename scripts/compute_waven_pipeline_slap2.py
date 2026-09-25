"""Waven RF mapping of the Zebra noise movie, SLAP2 sessions.

The SLAP2 counterpart of `compute_waven_pipeline_mesoscope.py`. Three things
differ from the mesoscope version, all forced by how SLAP2 records:

* **Resampling direction.** SLAP2 samples at ~200 Hz against a 30 fps movie, so
  there are ~6 imaging samples per frame instead of the mesoscope's one sample
  per three frames. The pipeline's `slap2` input type therefore averages dF/F
  per movie frame rather than looking up a frame per imaging sample, which
  keeps the design matrix at one row per frame instead of six near-duplicates.

* **Gaps.** Imaging runs in bouts with blanking in between, and short stretches
  of samples are NaN. Frames whose response window is not sampled densely enough
  are dropped from stimulus and response alike rather than filled in, and each
  bout is centred on its own mean so that baseline steps across the gaps cannot
  pose as slow stimulus structure. How much this removes depends on the session:
  on sub-829704/DMD1 the bouts happen to tile the Zebra block almost completely
  and only ~2.5 % of frames are dropped, but nothing guarantees that elsewhere,
  and the fraction retained is written to every result file
  (`frac_frames_covered`).

* **Repeats are pooled.** A single Zebra repeat may retain too few imaged frames
  to correlate against, so by default all repeats go into one design matrix,
  with `frame_index` mapping each onset back to its row of the decomposition.
  `--per-trial` restores the mesoscope behaviour of one result set per repeat,
  which is what the repeatability checks need. (sub-829704 has only one repeat,
  so the two are equivalent there.)

Two DMDs x two colour channels means four runs per session. Green is
iGluSnFR4f, fast enough to resolve single movie frames; red is RCaMP3, whose
calcium kinetics blur the response across frames no matter which window is
chosen - so each channel gets its own delay/duration grid.

The correlation runs against the deconvolved events by default; `--signal dff`
restores the raw ΔF/F. The SLAP2 release carries no events (unlike the mesoscope
one), so `utils.deconvolution` computes and caches them per session, DMD and
channel on first use. Note that bout centring exists to remove ΔF/F baseline
steps across the blanking gaps; the events have no such drift, so
`--no-centre-bouts` is the more natural choice for them.

The two outputs go to different places: the per-(delay, duration) files are
bulky and land in the HDD workspace, while the reduced one-row-per-unit results
go to the project results tree. Each signal gets its own subtree in both.

    python compute_waven_pipeline_slap2.py \
        --nwb-path ../../../rawdata/allen_open_scope/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb \
        --results-dir /mnt/ceph-hdd/workspaces/ws/cidbn_wibral_neuro_nonhuman/u19361-allen-hdd \
        --dmd DMD1 --channel green
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append('..')

import utils
from utils import deconvolution
from waven_settings import *
from waven_pipeline import *
import optimize_waven_parameters as owp
import rf_siegle_ophys as siegle   # MIN_COVERAGE, shared with the Siegle maps

OPTIMIZED_DIR = Path('/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/'
                     'SPP2205_mraabe/results/allen_open_scope/rf/waven/zebra/optimized')

# One Zebra frame is 1/30 s. Green (iGluSnFR4f) decays in tens of milliseconds,
# so it is the one signal in this dataset that can be integrated over a single
# frame; durations beyond ~0.1 s only smear it across neighbours. Red (RCaMP3)
# is a calcium sensor - its rise alone outlasts a frame, so the grid is shifted
# later and wider and the maps it yields are a lower-resolution cross-check
# rather than a second independent estimate.
GRIDS = {
    'green': {
        'delays':    np.arange(0.0, 0.15, 0.025),
        'durations': np.array([1 / 30, 1 / 15, 0.1]),
    },
    'red': {
        'delays':    np.arange(0.0, 0.4, 0.05),
        'durations': np.array([0.1, 0.25, 0.5]),
    },
}


def load_signal(stream, dmd, channel, t_start, t_stop, signal='dff', nwb_path=None):
    """(traces_df, unit_names) for one DMD/channel over [t_start, t_stop].

    signal: 'dff' for ΔF/F, 'events' for the deconvolved series.

    Only the Zebra block is read: the full-session traces are several hundred MB
    per channel, and everything outside the block is dead weight in the cumulative
    sums downstream.
    """
    if signal == 'dff':
        traces_df = stream.slap2_dff(dmd, channel, t_start=t_start, t_stop=t_stop)
    elif signal == 'events':
        traces_df = deconvolution.slap2_events_df(nwb_path, dmd, channel, t_start=t_start,
                                                  t_stop=t_stop, stream=stream)
    else:
        raise ValueError(f"signal must be 'dff' or 'events', got {signal!r}")

    roi_ids = stream.slap2_rois(dmd)['roi_id'].to_numpy()
    unit_names = [f'{dmd}_{channel}_roi{roi_id}' for roi_id in roi_ids]
    return traces_df, unit_names


def main(nwb_path, results_dir, dmd='DMD1', channel='green', phases=('0', '1'),
         signal='events', per_trial=False, fps=30.0, min_coverage=siegle.MIN_COVERAGE,
         centre_bouts=True, recompute=False, optimize=True,
         optimized_dir=OPTIMIZED_DIR):

    session_name = Path(nwb_path).stem
    # full_pipeline skips grid cells whose file exists and the optimized stems
    # carry no signal, so each signal needs its own subtree in both places
    results_dir = Path(results_dir) / session_name / signal
    optimized_dir = Path(optimized_dir)

    delays = GRIDS[channel]['delays']
    durations = GRIDS[channel]['durations']

    print('Loading NWB file...')
    stream = utils.open_local(nwb_path)
    if dmd not in stream.dmds():
        raise ValueError(f'{dmd!r} not in this session; available: {stream.dmds()}')

    # one row per movie frame per repeat; 'frame' is the row of the decomposition
    zebra_frames = stream.zebra_frame_times(fps=fps)

    # read only the Zebra block, with room for the longest response window
    margin = float(delays.max() + np.abs(durations).max() + 1.0)
    t_start = zebra_frames['start_time'].min() - margin
    t_stop = zebra_frames['stop_time'].max() + margin
    traces_df, unit_names = load_signal(stream, dmd, channel, t_start, t_stop,
                                        signal=signal, nwb_path=nwb_path)

    bout_bounds = None
    if centre_bouts:
        segments = stream.slap2_segments(dmd, channel)
        segments = segments.query('stop_time > @t_start and start_time < @t_stop')
        bout_bounds = segments[['start_time', 'stop_time']].to_numpy()
        # bouts may run past either end of the block, so clip before reporting
        imaged = float(np.clip(bout_bounds[:, 1], t_start, t_stop).sum()
                       - np.clip(bout_bounds[:, 0], t_start, t_stop).sum())
        print(f'{len(bout_bounds)} imaging bouts overlap the Zebra block, '
              f'{imaged:.0f} s imaged out of {t_stop - t_start:.0f} s')

    trials = ([(i, t) for i, t in enumerate(zebra_frames['TrialNumber'].unique())]
              if per_trial else [(None, None)])

    for i_trial, trialnumber in trials:
        if per_trial:
            block = zebra_frames[zebra_frames['TrialNumber'] == trialnumber]
            trial_part = [f'trial_{i_trial}']
        else:
            block = zebra_frames
            trial_part = []

        frame_onset_times = block['start_time'].to_numpy(dtype=float)
        frame_index = block['frame'].to_numpy(dtype=int)

        for phase in phases:
            attributes = {
                'session': session_name,
                'modality': 'slap2',
                'dmd': dmd,
                'plane': dmd,          # keeps the shared analysis code's naming happy
                'channel': channel,
                'signal': signal,
                'phase': phase,
                'fps_assumed': fps,
                'pooled_repeats': not per_trial,
                'date_computed': datetime.today().strftime('%Y-%m-%d'),
            }
            if per_trial:
                attributes['nwb_trialnumber'] = trialnumber
                attributes['trial'] = i_trial

            results_path = results_dir.joinpath(dmd, channel, *trial_part, f'phase_{phase}')
            results_path = full_pipeline(
                frame_onset_times,
                traces_df,
                delays,
                durations,
                unit_names,
                results_path=results_path,
                results_filename='',
                attributes=attributes,
                recompute=recompute,
                phase=phase,
                input_type='slap2',
                frame_index=frame_index,
                bout_bounds=bout_bounds,
                min_coverage=min_coverage,
            )

            if optimize:
                print(f'Optimizing parameters over {results_path}...')
                stem = '__'.join(['optimized', session_name, dmd, channel,
                                  *trial_part, f'phase_{phase}'])
                outdir = optimized_dir / session_name / signal
                df = owp.optimize_parameters(results_dir=results_path,
                                             output_path=outdir / (stem + '.csv'))
                df = owp.load_rf_maps(df)
                owp.save_rf_results(df, outdir / (stem + '.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb-path', required=True, help='Path to the SLAP2 NWB file')
    parser.add_argument('--results-dir', required=True,
                        help='Root directory for the per-(delay, duration) result files. '
                             'These are the bulky ones - the correlation matrix dominates - '
                             'so this normally points at the HDD workspace.')
    parser.add_argument('--optimized-dir', default=str(OPTIMIZED_DIR),
                        help='Root directory for the reduced one-row-per-unit results. '
                             f'Default: {OPTIMIZED_DIR}')
    parser.add_argument('--dmd', default='DMD1', help='DMD imaging plane, e.g. DMD1 or DMD2')
    parser.add_argument('--channel', default='green', choices=['green', 'red'],
                        help="'green' (iGluSnFR4f, default) or 'red' (RCaMP3)")
    parser.add_argument('--phases', nargs='+', default=['0', '1'],
                        choices=['0', '1', 'complex'], help='Wavelet phases to run')
    parser.add_argument('--signal', default='events', choices=['dff', 'events'],
                        help="Trace to correlate: the deconvolved 'events' (default), "
                             'computed and cached by utils.deconvolution on first use, '
                             "or raw 'dff'. Each signal gets its own results subtree.")
    parser.add_argument('--per-trial', action='store_true',
                        help='One result set per Zebra repeat instead of pooling them. '
                             'Needed for repeatability checks; note a single repeat may '
                             'retain few imaged frames.')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frame rate assumed when reconstructing frame onsets from '
                             'the per-repeat Zebra intervals')
    parser.add_argument('--min-coverage', type=float, default=siegle.MIN_COVERAGE,
                        help='Fraction of a response window that must be sampled for the '
                             'frame to be kept')
    parser.add_argument('--no-centre-bouts', action='store_true',
                        help='Skip per-bout mean subtraction')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute grid cells whose result file already exists')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip the delay x duration reduction step')

    args = parser.parse_args()
    main(args.nwb_path, args.results_dir, args.dmd, args.channel, tuple(args.phases),
         signal=args.signal,
         per_trial=args.per_trial, fps=args.fps, min_coverage=args.min_coverage,
         centre_bouts=not args.no_centre_bouts, recompute=args.recompute,
         optimize=not args.no_optimize, optimized_dir=args.optimized_dir)
