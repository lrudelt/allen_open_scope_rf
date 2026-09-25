import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
# from Waven.LoadPinkNoise import *
from Waven import WaveletGenerator as wg

import sys
sys.path.append('..')

import utils
from waven_settings import *
from waven_pipeline import *
import argparse

from datetime import datetime
from pathlib import Path

import optimize_waven_parameters as owp


SIGNALS = ('dff', 'events')


def main(nwb_path=None, results_dir=None, plane_idx=0, signal='events'):

    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
    session_name = Path(nwb_path).stem
    # full_pipeline skips existing files, so each signal needs its own tree or
    # an events run would silently reuse the dF/F results; the session sits
    # above it so that one session's results stay in one place
    results_dir = Path(results_dir) / session_name / signal

    # delays = np.arange(0.0, 0.35, 0.05)
    durations = np.arange(0., 0.3, 0.1)
    # durations *= -1
    # delays = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09])
    # delays = -1 * np.array([0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.125, 0.15])
    delays = np.arange(0., 0.18, .03)


    print('Loading Dandi NWB file...')
    stream = utils.open_local(nwb_path)
    # nwb = stream.nwb

    planes = stream.imaging_planes()
    plane = planes[plane_idx]
    traces_df = stream.dff_df(plane) if signal == 'dff' else stream.events_df(plane)
    unit_names = traces_df.index.to_numpy()


    zebra_df = stream.zebra_df()
    for i_trial, trialnumber in enumerate(zebra_df['TrialNumber'].unique()):

        frame_onset_times = zebra_df.loc[zebra_df['TrialNumber'] == trialnumber, 'start_time'].values

        # for plane in planes:
            # dff_df = stream.dff_df(plane)
            # unit_names = dff_df.index.to_numpy()
        for phase in ['0', '1']:
            attributes = {'session' : session_name, 
                            'phase' : phase,
                            'plane' : plane,
                            'signal' : signal,
                            'date_computed' : datetime.today().strftime('%Y-%m-%d'),
                            'nwb_trialnumber' : trialnumber,
                            'trial' : i_trial,
                            }

            results_path = results_dir/plane/f'trial_{i_trial}'/f'phase_{phase}'
            results_path = full_pipeline(frame_onset_times,
                    traces_df,
                    delays,
                    durations,
                    unit_names,
                #   xis=xis,
                #   yis=yis,
                    results_path=results_path,
                    results_filename='',
                    attributes=attributes,
                    recompute=False,
                    phase=phase,
                    input_type='mesoscope'
                    )
                
            # Optimize parameters 
            print('Optimizing parameters...')
            print(results_path)
            outpath = Path(f"/mnt/ceph-hdd/projects/cidbn_wibral_neuro_nonhuman/SPP2205_mraabe/results/allen_open_scope/rf/waven/zebra/optimized/{session_name}/{signal}")
            stem = f"optimized__{session_name}__{plane}__trial_{i_trial}__phase_{phase}"
            csv_out = outpath / (stem + ".csv")
            h5_out = outpath / (stem + ".h5")
            df = owp.optimize_parameters(results_dir=results_path, output_path=csv_out)
            df = owp.load_rf_maps(df)
            owp.save_rf_results(df, h5_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run waven pipeline for a given probe")
    # parser.add_argument('--probe', nargs='?', default='ProbeB', help='Probe name (e.g., ProbeB)')
    parser.add_argument('--nwb-path', required=True, help='Path to the NWB file (e.g., /data/sub-820454.nwb)')
    parser.add_argument('--results-dir', required=True, help='Root directory for saving results')
    parser.add_argument('--plane-idx', required=True, type=int, help='Index of the plane to process (0-based)')
    parser.add_argument('--signal', default='events', choices=SIGNALS,
                        help="Trace to correlate: the OASIS-deconvolved 'events' (default) "
                             "or raw 'dff'. Results go to <results-dir>/<session>/<signal>/")
    args = parser.parse_args()
    main(nwb_path=args.nwb_path, results_dir=args.results_dir, plane_idx=args.plane_idx, signal=args.signal)