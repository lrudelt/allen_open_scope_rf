# allen_open_scope — Wavelet RF Mapping

Receptive field mapping pipeline applied to [Allen Open Scope](https://dandiarchive.org/dandiset/001637) electrophysiology data. Zebra noise (pink noise video) stimuli are decomposed by a Gabor wavelet filter bank, and the resulting wavelet responses are cross-correlated with single-unit spike rates to estimate receptive field maps.

## Setup

### 1. Install the environment

```bash
conda env create -f envs/waven_dandi.yml
conda activate waven_dandi
```

This installs the `waven` package directly from the `develop` branch of [github.com/RaabeM/waven](https://github.com/RaabeM/waven).

### 2. Configure paths

Open `waven_settings.py` and update the two path entries to point to your data:

```python
"Movie Path": "/path/to/zebra_allen_screen_tscale_30_scale_10.mp4",
"Library Path": "/path/to/filter_libraries/",
```
- `Movie Path`: Path to Zebra stimulus, can be downloaded from Open-Scope-Repo [Link](https://github.com/AllenNeuralDynamics/openscope-community-predictive-processing/blob/main/code/stimulus-control/src/Mindscope/movies/zebra_allen_screen_tscale_30_scale_10.mp4)
- `Library Path`: Path to where the wavelet library will be saved


## Running the pipeline
**Caution:** Under the current settings loading the filter libraries takes **256 GB of RAM!** I am working on making it more handy.

### Wavelet RF mapping (main pipeline)

```bash
python compute_waven_pipeline.py ProbeB \***
    --nwb-path /data/sub-820454.nwb \
    --results-dir /results/waven/
```

- `probe` (positional, default `ProbeB`): probe to process
- `--nwb-path`: path to the NWB file
- `--results-dir`: root directory for output HDF5 files

Results are saved to `<results-dir>/<probe>/lib_<hash>/` as one `.h5` file per (delay, duration) combination.

#### Output format

Each result file is an HDF5 with the following structure:

```
/attrs:  sigmas, thetas, frequencies, delay, duration, session, probe
/unit_ids                — (n_units,) unit identifiers
/correlation_matrix      — (n_units, nx, ny, n_thetas, ns, n_freqs)  gzip-4
/best_gabor_params_idx   — (5, n_units)
/best_gabor_params_degree — (4, n_units)
/abs_max_value           — (n_units,)
```

### Optimize parameters

After running the pipeline across the full delay × duration grid:

```bash
python optimize_waven_parameters.py /results/waven/ProbeB/lib_<hash>/ \
    --output /results/waven/ProbeB/lib_<hash>/best_params.h5
```

#### Output format

```
/unit_ids      — (n_units,) unit identifier strings
/rf_maps       — (n_units, nx, ny)  float32  gzip-4  spatial RF map at best params
/delay         — (n_units,)  optimal delay in seconds
/duration      — (n_units,)  optimal duration in seconds
/abs_max_value — (n_units,)  peak |Pearson r| across all (delay, duration, θ, σ, f)
/x_deg         — (n_units,)  RF centre azimuth in visual degrees
/y_deg         — (n_units,)  RF centre elevation in visual degrees
/theta_deg     — (n_units,)  orientation in Waven's degree convention
/theta_rad     — (n_units,)  orientation in radians
/sigma_deg     — (n_units,)  Gabor σ in visual degrees
/sigma         — (n_units,)  Gabor σ in pixels
/frequency     — (n_units,)  spatial frequency in cycles/pixel
/xi, /yi       — (n_units,)  grid indices of RF centre
/theta_idx, /sigma_idx, /frequency_idx  — (n_units,)  grid indices of best params
/lib_id        — (n_units,)  filter library hash string
/source_file   — (n_units,)  path to the per-(delay, duration) source .h5
```

### Analysis and plots for data release

Example sessions: ecephys 830794, mesoscope 832700, SLAP2 829704. Results go under `results/`. Then:

```bash
cd code/scripts
python plot_data_release_panels.py
```

Generating the plots requires the following steps:
1. **Zebra maps.** `compute_waven_pipeline.py` (ephys spikes), `compute_waven_pipeline_mesoscope.py --signal events`, `compute_waven_pipeline_slap2.py --signal dff`. Each calls `waven_pipeline.full_pipeline`, then `optimize_waven_parameters.py`.
2. **Zebra nulls.** `compute_null_distribution.py`, `compute_null_distribution_mesoscope.py`, and `compute_null_distribution_slap2.py` write `results/zebra/nullresults/`. The plotter fits a Gumbel to those surrogates and applies BH at q = 0.05. Those three scripts are not in this checkout.
3. **Local Gabor maps.** `compute_rf_gabors_ephys.py` (self-contained). `compute_rf_gabors_mesoscope.py` and `compute_rf_gabors_slap2.py` (default `--signal events`) call `rf_siegle_ophys.py`, which is a library, not a script you run. All three write `*__optimized.h5`.
4. **Gabor unit significance.** `compute_gabor_unit_significance.py` adds `p_value_gumbel` and `p_value_bh` to those files. Needed for the overlap, pie, and Gaussian-summary panels.
5. **Ephys areas.** `export_ephys_unit_areas.py` writes `unit_areas.csv`, used to keep visual cortex layer 2/3.

