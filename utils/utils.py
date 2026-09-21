
from pathlib import Path
# from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
from dandi import dandiapi, download
import os   
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from pynwb import NWBHDF5IO

from tqdm import tqdm

import pickle
import cv2

from scipy.ndimage import label
import h5py
from compute_rf_siegle_gabors import get_stim_onsets_offsets, time_slice_spike_train

FILE_ZEBRA = '../../rawdata/allen_open_scope/stimulus/zebra/zebra_allen_screen_tscale_30_scale_10.mp4'

dandiset_id = "001637"
dandi_dirpath = "../../rawdata/allen_open_scope/sub-820454/"
dandi_filepath = "sub-820454_ses-ecephys-820454-2025-11-04-14-59-22_ecephys.nwb"
derivitive_dir = '../../rawdata/allen_open_scope/derivitives/'
filenames = os.listdir(dandi_dirpath)

class VideoLoader:

    def __init__(self, file_path=FILE_ZEBRA):
        self.file_path = file_path
        self.frames = []
        self._load_frames()
        self.shape = self.frames[0].shape

    def _load_frames(self):
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.file_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            frame = (frame > 127).astype(np.uint8)
            self.frames.append(frame)
        cap.release()
        print(len(self.frames))

    def get_frame(self, frame_number):
        if frame_number >= len(self.frames):
            raise ValueError(f"Frame number {frame_number} is out of range.")
        return self.frames[frame_number]



def load_dandi(dandi_filepath, dandiset_id, dandi_dirpath):
    print(f"Processing file: {dandi_filepath}")

    client = dandiapi.DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    dandi_filepath = f"sub-820454/{dandi_filepath}"
    file = dandiset.get_asset_by_path(dandi_filepath)
    file_url = file.download_url
    filename = dandi_filepath.split("/")[-1]
    filepath = f"{dandi_dirpath}/{filename}"    
    if os.path.exists(filepath):
        print("File already exists")
    else:
        # This can sometimes take a while depending on the size of the file
        download.download(file_url, output_dir=dandi_dirpath)
        print(f"Downloaded file to {filepath}")

    io = NWBHDF5IO(filepath, mode="r", load_namespaces=True)
    nwb = io.read()
    return nwb



def load_probe_dicts(filenames=[], derivitive_dir =derivitive_dir):

    probe_a = {}
    probe_b = {}
    probe_c = {}
    probe_d = {}


    if not os.path.exists(derivitive_dir):
        os.makedirs(derivitive_dir)


    if os.path.exists(derivitive_dir+'probe_a.pkl'):
        with open(derivitive_dir+'probe_a.pkl', 'rb') as f:
            probe_a = pickle.load(f)
    if os.path.exists(derivitive_dir+'probe_b.pkl'):
        with open(derivitive_dir+'probe_b.pkl', 'rb') as f:
            probe_b = pickle.load(f)
    if os.path.exists(derivitive_dir+'probe_c.pkl'):
        with open(derivitive_dir+'probe_c.pkl', 'rb') as f:
            probe_c = pickle.load(f)
    if os.path.exists(derivitive_dir+'probe_d.pkl'):
        with open(derivitive_dir+'probe_d.pkl', 'rb') as f:
            probe_d = pickle.load(f)
    

    if len(probe_a.keys()) == 0 :

        for dandi_filepath in filenames:
            nwb = load_dandi(dandi_filepath)
        
            probe_c_idxs = []
            probe_d_idxs = []
            probe_a_idxs = []
            probe_b_idxs = []
            for idx , df_electrode in tqdm(enumerate(nwb.units['electrodes'])):
                if 'ProbeC' in df_electrode.group_name.values:
                    probe_c_idxs.append(idx)
                if 'ProbeD' in df_electrode.group_name.values:
                    probe_d_idxs.append(idx)
                if 'ProbeA' in df_electrode.group_name.values:
                    probe_a_idxs.append(idx)
                if 'ProbeB' in df_electrode.group_name.values:
                    probe_b_idxs.append(idx)

            probe_a[dandi_filepath] = probe_a_idxs
            probe_b[dandi_filepath] = probe_b_idxs
            probe_c[dandi_filepath] = probe_c_idxs
            probe_d[dandi_filepath] = probe_d_idxs



        with open(derivitive_dir+'probe_a.pkl', 'wb') as f:
            pickle.dump(probe_a, f)
        with open(derivitive_dir+'probe_b.pkl', 'wb') as f:
            pickle.dump(probe_b, f)
        with open(derivitive_dir+'probe_c.pkl', 'wb') as f:
            pickle.dump(probe_c, f)
        with open(derivitive_dir+'probe_d.pkl', 'wb') as f:
            pickle.dump(probe_d, f)

    return probe_a, probe_b, probe_c, probe_d


import utils 
import neo.core as neo
import quantities as pq
import scipy


def get_siegle_steps(dandi_filepath, unit_name, orientation):

    dandi_filepath = dandi_filepath
    dandiset_id = utils.dandiset_id
    dandi_dirpath = utils.dandi_dirpath
    nwb = utils.load_dandi(dandi_filepath, dandiset_id, dandi_dirpath)
    df_units = nwb.units.to_dataframe()

    df_rf = nwb.intervals['RF mapping_presentations'].to_dataframe()
    unique_x = np.unique(df_rf['X'])
    unique_y = np.unique(df_rf['Y'])
    x_pos = np.sort(unique_x.astype(float))
    y_pos = np.sort(unique_y.astype(float))
    combinations_xy = [(x, y) for x in unique_x for y in unique_y]

    spike_times = df_units.loc[df_units['unit_name']==unit_name, 'spike_times'].values[0]
    spike_times = neo.SpikeTrain(spike_times, t_start=spike_times[0], t_stop=spike_times[-1], units=pq.s)

    # Compute response grid
    R = np.zeros((5, len(x_pos), len(y_pos)))
    for (x,y) in combinations_xy:
        x_i = np.digitize(float(x), x_pos)-1
        y_i = np.digitize(float(y), y_pos)-1

        onsets, offsets = get_stim_onsets_offsets(df_rf, combination_xy=(x,y), orientation=orientation)
        for i, (onset, offset) in enumerate(zip(onsets, offsets)):
            n_spikes = len(time_slice_spike_train(spike_times, onset, offset))
            R[i, x_i, y_i] = n_spikes / ((offset - onset).rescale(pq.s).magnitude)
    E = np.mean(R) + 1e-10  # add small value to avoid division by zero
    O = np.mean(R, axis=0)

    response_grid_filtered = scipy.ndimage.gaussian_filter(O, sigma=1)

    std = np.std(response_grid_filtered)
    max = np.max(response_grid_filtered)
    threshold = max - std
    response_grid_filtered[response_grid_filtered < threshold] = np.nan

    # Label connected components in the filtered grid
    labeled_array, num_features = label(~np.isnan(response_grid_filtered))
    # Get the size of each connected component
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (label 0)
    # Get the size of the largest connected component
    largest_cluster_size = component_sizes.max() if component_sizes.size > 0 else 0
    largest_cluster_label = np.argmax(component_sizes) + 1  # Labels start from 1

    positions = np.where(labeled_array==largest_cluster_label)
    center_of_mass = [np.mean(positions[0]), np.mean(positions[1])]

    largest_cluster = np.full_like(response_grid_filtered, np.nan)
    for pos_x, pos_y in zip(positions[0], positions[1]):
        largest_cluster[pos_x, pos_y] = response_grid_filtered[pos_x, pos_y]

    return R, O, response_grid_filtered, largest_cluster, center_of_mass



def PlotTuningCurve_Marcel(rfs, idx, visual_coverage, sigmas, screen_ratio, show=True, canvas=None):
    """
    Quick and drity fix of Waven.Analysis_Utils.PlotTuningCurve
    """

    xM, xm, yM, ym = visual_coverage
    cc_f_1_xy=rfs[0][idx, :, :, np.array(rfs[1])[2, idx], np.array(rfs[1])[3, idx]]
    cc_f_1_o=rfs[0][idx,np.array(rfs[1])[0, idx], np.array(rfs[1])[1, idx], :, :]
    s=np.array(rfs[1])[3, idx]
    o = np.array(rfs[1])[2, idx]

    u, s__, v = svds(cc_f_1_xy, 2)
    ori_tun = np.append(cc_f_1_o[:, s], cc_f_1_o[0, s])
    i = 1
    if v[1][np.argmax(abs(v[1]))] < 0:
        i = -1
    if show:
        if canvas is None:
            fig, ax = plt.subplots(1, 5, figsize=(15, 1.5))
        else:
            fig = canvas[0]
            ax = canvas[1]

        m=ax[0].imshow(cc_f_1_xy.T, cmap='coolwarm', vmin=-abs(cc_f_1_xy).max(), vmax=abs(cc_f_1_xy).max())
        fig.colorbar(m)
        ax[0].set_xticks([0 , cc_f_1_xy.shape[0]], [xM, xm])
        ax[0].set_yticks([0, cc_f_1_xy.shape[1]], [yM, ym])
        ax[0].set_title('2D correlation')
        
        ax[1].plot(i*v[1][::-1], c='k')
        ax[1].set_xticks([0, cc_f_1_xy.shape[1]], [ym, yM])
        ax[1].set_ylabel('Corr.')
        ax[1].set_xlabel('Elevation (deg)')


        ax[2].plot(i*u[:, 1], c='k')
        ax[2].set_xticks([0, cc_f_1_xy.shape[0]], [xM, xm])
        ax[2].set_xlabel('Azimuth (deg)')
        # ax[1].plot(np.max(cc_f_1_xy, axis=0))
        # ax[2].plot(np.max(cc_f_1_xy, axis=1))
        # ax[1].plot(cc_f_1_xy[x, :])
        mm = max(cc_f_1_o.min(), cc_f_1_o.max(), key=abs)

        ax[3].plot(ori_tun, 'o-', c='k')
        ax[3].set_xticks([0, 4, 8], [0,90, 180])
        ax[3].set_xlabel('Orientation (deg)')

        # ax[3].set_ylim(bottom=0)
        # if mm<=0:
        #     ax[1].set_ylim(mm, 0)
        # else:
        #     ax[1].set_ylim(0, mm)
        # ax[2].plot(abs(cc_f_1[:, s, f]))
        ax[4].plot(cc_f_1_o[o, :], 'o-', c='k')
        ax[4].set_xticks(np.arange(len(sigmas)), sigmas)
        ax[4].set_xlabel('Size (deg)')

        # fig.tight_layout()
        # plt.show()
    return [cc_f_1_xy.T, i*v[1][::-1], i*u[:, 1], ori_tun, cc_f_1_o[o, :]]


def load_all_gabor_rf_results(results_dir=None):
    """One row per (unit, orientation) from the per-unit gabor spike-count files.

    results_dir: directory of the .h5 files. The default below is where they used
    to live; the tree has since moved under .../gabors/backup/, so callers that
    still rely on the default will not find it. Pass the directory explicitly.
    """
    # RESULTS_DIR = '../../results/allen_open_scope/rf/siegle/gabors/z-score_responses/new'
    RESULTS_DIR = results_dir or '../../results/allen_open_scope/rf/siegle/gabors/z-score_responses/sub-820454_ses-ecephys-820454-2025-11-04-14-59-22_ecephys/100.0'

    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(
            f'{RESULTS_DIR} does not exist. Pass results_dir= pointing at the '
            'directory holding the gabors_spikecount_*.h5 files.'
        )

    results_files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.endswith('.h5')]
    results = []
    for file in results_files:
        with h5py.File(file, 'r') as hf:
            for unit in hf.keys():
                grp = hf[unit]
                results.append({
                    'unit_name': unit,
                    'file': grp['file'][()].decode('utf-8'),
                    'orientation': grp['orientation'][()],
                    'mean_response': grp['mean_response'][()],
                    'z_score_response': grp['z_score_response'][()],
                    'p_value': grp['p_value'][()],
                    'probe': grp['probe'][()].decode('utf-8'),
                    'mean_rate': grp['mean_rate'][()]
                })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# CCF area / layer (same decoding as alexmaier_code/ai_oscp_neuro/openscope_ccf)
# Prefer `from utils import load_unit_areas, decode_ccf` via the package __init__.
# ---------------------------------------------------------------------------

def decode_ccf(acronym):
    """Decode an electrodes.location CCF acronym into area / layer / group / tissue."""
    from .ccf import decode_ccf as _decode
    return _decode(acronym)


def enrich_unit_locations(df, location_col="location"):
    """Add area / layer / group / tissue columns by decoding ``location``."""
    from .ccf import enrich_unit_locations as _enrich
    return _enrich(df, location_col=location_col)


def load_unit_areas(stream, probe=None):
    """Per-unit CCF area and layer from a streamed ecephys NWB (no spike times).

    ``stream`` is an :class:`~utils.streaming.NWBStream` from
    :func:`utils.open_nwb` / :meth:`DandiSession.open` / :func:`open_local`.
    """
    from .ccf import enrich_unit_locations, structure_label
    df = stream.units_df(probe=probe, include_spikes=False)
    if "area" not in df.columns:
        if "location" not in df.columns:
            raise ValueError(
                "No electrodes.location in this NWB — CCF acronyms are absent."
            )
        df = enrich_unit_locations(df)
    df = df.copy()
    df["structure"] = [
        structure_label(a, L) for a, L in zip(df["area"], df["layer"])
    ]
    return df


def parse_mesoscope_location(location, plane_fallback=None):
    """Parse mesoscope ``Structure: … Depth: …`` optophysiology location strings."""
    from .ccf import parse_mesoscope_location as _parse
    return _parse(location, plane_fallback=plane_fallback)
