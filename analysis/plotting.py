import pandas as pd
import numpy as np
import re
import sys
sys.path.append('..')

from pathlib import Path
import matplotlib.pyplot as plt

from scripts.optimize_waven_parameters import load_rf_maps, load_rf_results
import utils



import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from scripts.waven_settings import *
from matplotlib.ticker import MultipleLocator
import Waven.WaveletGenerator as wg




def has_rf_map(df):
    """
    Rows of df whose 'rf_map' is an actual array.

    An optimized .h5 written without the 'rf_maps' dataset (an interrupted or
    older run) contributes rows with no rf_map at all; concatenating those with
    complete files turns the column into NaN for them. Every map plot has to
    skip those rows instead of calling .T on a float.
    """
    if 'rf_map' not in df.columns:
        return df.iloc[0:0]
    return df[df['rf_map'].apply(lambda m: isinstance(m, np.ndarray))]


def get_statistics_per_probe(probe:str, nwb_path, results_path,  rate_threshold=2):

    # nwb_path = '../../../rawdata/allen_open_scope/sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb'
    stream = utils.open_local(nwb_path)
    units_df = stream.units_df()
    units_df['unit_name'] = units_df['unit_name'].astype(str)

    zebra_df = stream.zebra_df()

    # RESULTS_PATH = Path('../../../results/allen_open_scope/rf/waven/zebra/optimized/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys')
    dfs = []
    for trial in [0,1]:
        for phase in [0, 1]:
            ov_file = list(results_path.glob(f'*{probe}__trial_{trial}__phase_{phase}.h5'))[0]
            tmp_df = load_rf_results(ov_file)
            tmp_df['trial'] = trial
            tmp_df['phase'] = phase
            dfs.append(tmp_df)
            
    ov_df = pd.concat(dfs, ignore_index=True)
    ov_df = ov_df.sort_values('abs_max_value', ascending=False)
    ov_df.reset_index(inplace=True, drop=True)


    first_trial = zebra_df['TrialNumber'].unique()[0]
    t_start = zebra_df.loc[zebra_df['TrialNumber'] == first_trial, 'start_time'].values[0]
    t_stop  = zebra_df.loc[zebra_df['TrialNumber'] == first_trial, 'stop_time'].values[-1]
    dt = t_stop - t_start

    def rate(unit):
        matches = units_df.loc[units_df['unit_name'] == unit, 'spike_times'].values
        if len(matches) == 0:
            return 0.0
        spk = matches[0]
        return (np.searchsorted(spk, t_stop) - np.searchsorted(spk, t_start)) / dt
    ov_df['rate'] = ov_df['unit_id'].apply(rate)

    n_total = len(ov_df['unit_id'].unique())
    n_matched = (ov_df['rate'] > 0).sum()
    if n_matched == 0:
        print(f"WARNING: 0/{n_total} units matched in units_df — likely session mismatch.")
        print(f"  nwb_path   : {nwb_path}")
        print(f"  results    : {results_path}")
        print(f"  units_df unit_name sample : {units_df['unit_name'].values[:3].tolist()}")
        print(f"  ov_df unit_id sample      : {ov_df['unit_id'].values[:3].tolist()}")

    ov_df = ov_df[ov_df['rate'] > rate_threshold]
    ov_df.reset_index(inplace=True, drop=True)



    dfs = []
    df_all = []
    for trial in [0,1]:
        for phase in [0, 1]:
            file = list(results_path.glob(f'*{probe}__trial_{trial}__phase_{phase}_all_values.csv'))[0]
            df = pd.read_csv(file)
            df['trial'] = trial
            df['phase'] = phase
            df_all.append(df)
            
            df_best = pd.DataFrame(df.groupby('unit_id')['abs_max_value'].idxmax()).reset_index()
            df_best['duration'] = [df.loc[idx, 'duration'] for idx in df_best['abs_max_value']]
            df_best['delay'] = [df.loc[idx, 'delay'] for idx in df_best['abs_max_value']]
            df_best['trial'] = trial
            df_best['phase'] = phase
            dfs.append(df_best)
    df_best = pd.concat(dfs, ignore_index=True)
    df_all = pd.concat(df_all, ignore_index=True)


    trial_phase_00 = ov_df.query('trial==0 and phase==0').sort_values('abs_max_value', ascending=False).reset_index(drop=True)

    return ov_df, df_best, df_all, trial_phase_00


def discover_trials_and_phases(results_path, key:str):
    """
    Return the sorted (trial, phase) pairs for which an optimized .h5 exists.

    key is the probe or imaging plane name embedded in the filename, i.e.
    optimized__{session}__{key}__trial_{t}__phase_{p}.h5
    """
    pattern = re.compile(rf'__{re.escape(key)}__trial_(\d+)__phase_(\d+)\.h5$')
    pairs = set()
    for f in Path(results_path).glob(f'*{key}__trial_*__phase_*.h5'):
        m = pattern.search(f.name)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


def get_statistics_per_plane(plane:str, nwb_path, results_path, activity_threshold=0.0):
    """
    Mesoscope counterpart of get_statistics_per_probe.

    Two things differ from the ecephys case. First, dF/F traces have no firing
    rate to threshold on, so ROIs are screened on the standard deviation of
    their trace inside the first zebra trial — a flat (dead or unsegmented)
    ROI has std ~0. Second, the number of zebra trials varies per session, so
    trials and phases are discovered from the filenames instead of assumed to
    be [0, 1].

    plane: imaging plane name as returned by stream.imaging_planes(), e.g. 'VISp_0'.

    Returns (ov_df, df_best, df_all, trial_phase_00), the same tuple as
    get_statistics_per_probe, with an 'activity' column in place of 'rate'.
    """
    results_path = Path(results_path)

    stream = utils.open_local(nwb_path)
    zebra_df = stream.zebra_df()
    dff_df = stream.dff_df(plane)     # (n_rois, n_timepoints), columns = timestamps

    pairs = discover_trials_and_phases(results_path, plane)
    if not pairs:
        raise FileNotFoundError(
            f"No optimized .h5 files for plane {plane} under {results_path}. "
            f"Planes present: {sorted({m.group(1) for m in (re.search(r'__(VIS[^_]*_\d+)__trial_', f.name) for f in results_path.glob('*.h5')) if m})}"
        )

    trials = sorted({t for t, _ in pairs})
    phases = sorted({p for _, p in pairs})
    incomplete = [(t, p) for t in trials for p in phases if (t, p) not in pairs]
    if incomplete:
        print(f"WARNING: plane {plane} is missing optimized results for (trial, phase) {incomplete} — "
              f"plots laid out as a trial x phase grid will have empty cells.")

    dfs = []
    for trial, phase in pairs:
        ov_file = list(results_path.glob(f'*{plane}__trial_{trial}__phase_{phase}.h5'))[0]
        tmp_df = load_rf_results(ov_file)
        if 'rf_map' not in tmp_df.columns:
            print(f"WARNING: {ov_file.name} holds no rf_maps dataset — scalar statistics for "
                  f"trial {trial}, phase {phase} are usable, but its RF maps are missing from every map plot.")
        tmp_df['trial'] = trial
        tmp_df['phase'] = phase
        dfs.append(tmp_df)

    ov_df = pd.concat(dfs, ignore_index=True)
    ov_df['unit_id'] = ov_df['unit_id'].astype(str)
    ov_df = ov_df.sort_values('abs_max_value', ascending=False)
    ov_df.reset_index(inplace=True, drop=True)


    first_trial = zebra_df['TrialNumber'].unique()[0]
    t_start = zebra_df.loc[zebra_df['TrialNumber'] == first_trial, 'start_time'].values[0]
    t_stop  = zebra_df.loc[zebra_df['TrialNumber'] == first_trial, 'stop_time'].values[-1]

    times = dff_df.columns.to_numpy(dtype=float)
    in_stim = (times >= t_start) & (times <= t_stop)
    # ROI ids are the dff_df row labels; the pipeline stores them as strings
    activity = pd.Series(dff_df.to_numpy()[:, in_stim].std(axis=1),
                         index=dff_df.index.astype(str))
    ov_df['activity'] = ov_df['unit_id'].map(activity)

    n_total = len(ov_df['unit_id'].unique())
    n_matched = int(ov_df['activity'].notna().sum())
    if n_matched == 0:
        print(f"WARNING: 0/{n_total} ROIs matched in dff_df — likely session/plane mismatch. "
              f"Every ROI will be filtered out below.")
        print(f"  nwb_path   : {nwb_path}")
        print(f"  results    : {results_path}")
        print(f"  dff_df index sample  : {dff_df.index[:3].astype(str).tolist()}")
        print(f"  ov_df unit_id sample : {ov_df['unit_id'].values[:3].tolist()}")

    ov_df = ov_df[ov_df['activity'] > activity_threshold]
    ov_df.reset_index(inplace=True, drop=True)



    dfs = []
    df_all = []
    for trial, phase in pairs:
        matches = list(results_path.glob(f'*{plane}__trial_{trial}__phase_{phase}_all_values.csv'))
        if not matches:
            print(f"WARNING: no _all_values.csv for plane {plane}, trial {trial}, phase {phase} — "
                  f"delay/duration curves will be missing for this combination.")
            continue
        df = pd.read_csv(matches[0])
        # read_csv turns purely numeric ROI ids into ints; ov_df keeps them as strings
        df['unit_id'] = df['unit_id'].astype(str)
        df['trial'] = trial
        df['phase'] = phase
        df_all.append(df)

        df_best = pd.DataFrame(df.groupby('unit_id')['abs_max_value'].idxmax()).reset_index()
        best_rows = df_best['abs_max_value'].values     # row indices of the per-unit maxima
        df_best['duration'] = df.loc[best_rows, 'duration'].values
        df_best['delay'] = df.loc[best_rows, 'delay'].values
        df_best['abs_max_value'] = df.loc[best_rows, 'abs_max_value'].values
        df_best['trial'] = trial
        df_best['phase'] = phase
        dfs.append(df_best)
    df_best = pd.concat(dfs, ignore_index=True)
    df_all = pd.concat(df_all, ignore_index=True)


    first_trial, first_phase = pairs[0]
    trial_phase_00 = (ov_df.query(f'trial=={first_trial} and phase=={first_phase}')
                           .sort_values('abs_max_value', ascending=False)
                           .reset_index(drop=True))

    return ov_df, df_best, df_all, trial_phase_00


PHASE_VALUES = {0: 0.0, 1: np.pi / 2}
PHASE_LABELS = {0: "Phase 0", 1: r"Phase $\pi$/2"}
PHASE_COLORS = {0: 'b', 1: 'r'}
TRIAL_LINESTYLES = ['-', '--', ':', '-.']


def get_statistics(key:str, nwb_path, results_path, threshold=2, modality='ecephys'):
    """Dispatch to the ecephys (probe) or mesoscope (imaging plane) statistics getter."""
    if modality == 'ecephys':
        return get_statistics_per_probe(key, nwb_path, results_path, threshold)
    elif modality == 'mesoscope':
        return get_statistics_per_plane(key, nwb_path, results_path, threshold)
    raise ValueError(f"modality must be 'ecephys' or 'mesoscope', got {modality!r}")


def plot_optimized_results(probe:str, NY, nwb_path, results_path, rate_threshold=2, path_printing=None, modality='ecephys'):

    ov_df, df_best, df_all, trial_phase_00 = get_statistics(probe, nwb_path, results_path, rate_threshold, modality)

    # trials and phases present in the results — mesoscope sessions are not
    # limited to the two trials / two phases the ecephys sessions always have
    trials = sorted(ov_df['trial'].unique())
    phases = sorted(ov_df['phase'].unique())
    n_tr, n_ph = len(trials), len(phases)

    f, axs = plt.subplots(NY, 4, figsize=(NY*1.2, NY*2.5))

    # delays = np.arange(0.0, 0.35, 0.05)
    # durations = np.arange(0.03, 0.28, 0.05)

    trial_handles = [
        Line2D([0], [0], color='k', linestyle=TRIAL_LINESTYLES[i % len(TRIAL_LINESTYLES)],
               lw=2, label=f'Trial {trial+1}')
        for i, trial in enumerate(trials)
    ]
    highest_abs_max = np.round(ov_df['abs_max_value'].max(),1)


    # PLOTTING 
    for idx in range(NY):
        ax = axs[idx]
        unit = trial_phase_00.loc[idx, 'unit_id']

        # Split axes[0] into a trials x phases grid
        inner = gridspec.GridSpecFromSubplotSpec(n_tr, n_ph, subplot_spec=ax[0].get_subplotspec(), wspace=0.05, hspace=0.05)
        ax_inner = np.array([[f.add_subplot(inner[i, j]) for j in range(n_ph)] for i in range(n_tr)])
        for i, trial in enumerate(trials):
            for j, phase in enumerate(phases):
                a = ax_inner[i, j]
                sel = has_rf_map(ov_df.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'"))
                if len(sel) == 0:
                    a.set_axis_off()
                    continue
                map = sel['rf_map'].values[0].T
                abs_max = sel['abs_max_value'].values[0]
                a.imshow(map, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)

                a.set_xticks([])
                a.set_yticks([])
                # rate = ov_df.loc[idx, 'rate']
                # ax[0].set_title(f"Abs. max correlation: {ov_df.loc[idx, 'abs_max_value']:.2f}")#\nRate: {rate:.1f} Hz")
        for j, phase in enumerate(phases):
            ax_inner[0, j].set_title(PHASE_LABELS[phase], fontsize=12, color=PHASE_COLORS[phase])
        for i, trial in enumerate(trials):
            ax_inner[i, 0].set_ylabel(f"Trial {trial+1}", fontsize=12)
        ax[0].set_axis_off()


        inner = gridspec.GridSpecFromSubplotSpec(n_tr, n_ph, subplot_spec=ax[1].get_subplotspec(), wspace=0.05, hspace=0.05)
        ax_inner = np.array([[f.add_subplot(inner[i, j]) for j in range(n_ph)] for i in range(n_tr)])
        lx = 67
        ly = 53

        for i, trial in enumerate(trials):
            for j, phase in enumerate(phases):
                a = ax_inner[i, j]
                sel = ov_df.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'")
                if len(sel) == 0:
                    a.set_axis_off()
                    continue

                theta = thetas[sel['theta_idx'].values[0]]
                sigma = sigmas[sel['sigma_idx'].values[0]]
                frequency = frequencies[sel['frequency_idx'].values[0]]
                best_filter = wg.makeGaborFilter(lx//2, #ov_df.loc[idx, 'xi'],
                                            ly//2, #ov_df.loc[idx, 'yi'],
                                            theta,
                                            sigma,
                                            PHASE_VALUES[phase],
                                            frequency,
                                            lx, ly)
                a.imshow(best_filter, cmap='coolwarm', vmin=-best_filter.max(), vmax=best_filter.max())
                a.set_xticks([])
                a.set_yticks([])
                # rate = ov_df.loc[idx, 'rate']
                # ax[0].set_title(f"Abs. max correlation: {ov_df.loc[idx, 'abs_max_value']:.2f}")#\nRate: {rate:.1f} Hz")
        for j, phase in enumerate(phases):
            ax_inner[0, j].set_title(PHASE_LABELS[phase], fontsize=12, color=PHASE_COLORS[phase])
        for i, trial in enumerate(trials):
            ax_inner[i, 0].set_ylabel(f"Trial {trial+1}", fontsize=12)
        ax[1].set_axis_off()




        for i, trial in enumerate(trials):
            linestyle = TRIAL_LINESTYLES[i % len(TRIAL_LINESTYLES)]
            for phase in phases:
                color = PHASE_COLORS[phase]
                best = df_best.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'")
                if len(best) == 0:
                    continue
                delay = best['delay'].values[0]
                durations = df_all.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}' and delay=={delay}")['duration'].values
                abs_max_values = df_all.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}' and delay=={delay}")['abs_max_value'].values

                ax[2].plot(durations, abs_max_values, alpha=1, marker='o', linestyle=linestyle, color=color, label=f'Trial {trial+1}, phase {PHASE_VALUES[phase]:.2f}')
                ax[2].set_xlabel('Duration (s)')
                ax[2].set_ylabel('Absolute max\ncorrelation')
                ax[2].set_title(f"Durations for best delay")
                ax[2].set_xticks(durations)
                ax[2].legend(handles=trial_handles, title='Trial',frameon=False)
                ax[2].set_yticks(np.arange(0, highest_abs_max, 0.1))
                ax[2].set_yticklabels(np.round(np.arange(0, highest_abs_max, 0.1),1))



        for i, trial in enumerate(trials):
            linestyle = TRIAL_LINESTYLES[i % len(TRIAL_LINESTYLES)]
            for phase in phases:
                color = PHASE_COLORS[phase]
                best = df_best.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}'")
                if len(best) == 0:
                    continue
                duration = best['duration'].values[0]
                delays = df_all.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}' and duration=={duration}")['delay'].values
                abs_max_values = df_all.query(f"trial=={trial} and phase=={phase} and unit_id=='{unit}' and duration=={duration}")['abs_max_value'].values

                ax[3].plot(delays, abs_max_values, alpha=1, marker='o', markersize=3, linestyle=linestyle, color=color, label=f'Trial {trial+1}, phase {PHASE_VALUES[phase]:.2f}')
        ax[3].set_xlabel('Delay (s)')
        ax[3].set_ylabel('Absolute max\ncorrelation')
        ax[3].set_title(f"Delays for best duration")
        # ax[3].set_xticks(np.unique(delays), np.unique(delays), rotation=90)
        # ax[3].set_xticks(np.round(np.arange(0,.4,.05),2), np.round(np.arange(0,.4,.05),2), rotation=90)
        major_ticks = np.round(np.arange(0, .35, .05), 2)
        ax[3].set_xticks(major_ticks)
        ax[3].set_xticklabels(major_ticks, rotation=90)

        step = major_ticks[1] - major_ticks[0] if len(major_ticks) > 1 else 0.05
        ax[3].xaxis.set_minor_locator(MultipleLocator(step / 2))
        ax[3].tick_params(axis='x', which='minor', length=4)

        # print(ax[3].get_xticks())
        ax[3].legend(handles=trial_handles, title='Trial', frameon=False)
        ax[3].set_yticks(np.arange(0, highest_abs_max, 0.1))
        ax[3].set_yticklabels(np.round(np.arange(0, highest_abs_max, 0.1),1))

        ax[3].grid()

    #     duration = df_best[df_best['unit_id'] == unit]['duration'].values[0]
    #     delays = df.loc[(df['unit_id'] == unit) & (df['duration']==duration), 'delay'].values
    #     abs_max_values = df.loc[(df['unit_id'] == unit) & (df['duration']==duration), 'abs_max_value'].values
    #     ax[3].plot(delays, abs_max_values, alpha=1, color='b', marker='o')
    #     ax[3].set_xlabel('Delay (s)')
    #     ax[3].set_ylabel('Absolute max\ncorrelation')
    #     ax[3].set_title(f"Delays for best duration")
    #     ax[3].set_xticks(delays)

    axs[0,0].set_title("Receptive field\n\n", fontsize=12)
    axs[0,1].set_title("Best filter\n\n", fontsize=12)

    f.suptitle(probe, fontsize=30, y=1.02)
    f.tight_layout()


    if path_printing is not None:
        path_printing = Path(path_printing)
        path_printing.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_printing/f'{probe}_optimized_results.png', dpi=300, bbox_inches='tight')

    return f, axs


def plot_avg_rf(probe, nwb_path, results_path, rate_threshold, cor_threshold, f, ax, modality='ecephys'):
    try:
        ov_df, df_best, df_all, trial_phase_00 = get_statistics(probe, nwb_path, results_path, rate_threshold, modality)
        ov_df = ov_df[ov_df['abs_max_value'] > cor_threshold]

    except Exception as e:
        print(f"Error processing probe {probe}: {e}")
        ax.text(0.5, 0.5, f'{probe} failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return None

    # ax.set_axis_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    trials = sorted(ov_df['trial'].unique())
    phases = sorted(ov_df['phase'].unique())
    n_tr, n_ph = len(trials), len(phases)

    inner = gridspec.GridSpecFromSubplotSpec(n_tr, n_ph, subplot_spec=ax.get_subplotspec(), wspace=0.05, hspace=0.05)
    ax_inner = np.array([[f.add_subplot(inner[i, j]) for j in range(n_ph)] for i in range(n_tr)])

    for j, phase in enumerate(phases):
        for i, trial in enumerate(trials):
            axi = ax_inner[i, j]
            subset = has_rf_map(ov_df.query(f'trial=={trial} and phase=={phase}'))
            n = len(subset)
            if n == 0:
                axi.text(0.5, 0.5, f'No units', ha='center', va='center', transform=axi.transAxes)
                axi.set_axis_off()
                continue
        
            # avg_rf = np.sum(subset['abs_max_value'].values * subset['rf_map'].values) / np.sum(subset['abs_max_value'].values)
            avg_rf = subset['rf_map'].sum()/len(subset['rf_map'])

            # print(trial, phase, avg_rf.shape)
            axi.imshow(avg_rf.T, cmap='coolwarm', vmin=-avg_rf.max(), vmax=avg_rf.max())
            axi.text(0.3, .7, f'N={n}', ha='center', va='bottom', transform=axi.transAxes, fontsize=9)
            axi.set_xticks([])
            axi.set_yticks([])

    # ax.set_title(probe, fontsize=12)
    for j, phase in enumerate(phases):
        ax_inner[0, j].set_title(PHASE_LABELS[phase], fontsize=12)
    for i, trial in enumerate(trials):
        ax_inner[i, 0].set_ylabel(f"Trial {trial+1}", fontsize=12)



    return ax_inner