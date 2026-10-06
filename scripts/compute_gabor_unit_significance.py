"""Unit-level Gabor p-values after the full search, then BH across the session.

Each unit gets one Gumbel p-value: 1000 presentation shuffles, the map
statistic studentized in every delay x duration x orientation, and the
maximum of those studentized values. Benjamini–Hochberg q-values are then
computed once per family and written beside the existing per-orientation
counting ``p_value``:

* ``p_value_gumbel`` — uncorrected unit-level p-value
* ``p_value_bh`` — BH q-value, the stored p-value-BH

Families match the session used for the Zebra null, with SLAP2 green and red
kept separate. Full-run optimized files only; trial splits and the legacy
misaligned ephys units are left unchanged.

    cd code/scripts
    python compute_gabor_unit_significance.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parent))

import rf_siegle_ophys as siegle
import test_significance_Gabor as gabor
import utils
from oasis_deconvolve_slap2 import TAU_D, deconvolve_bouts

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "gabors"
N_SHUFFLE = gabor.N_SHUFFLE

EPHYS_SESSION = "sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys"
MESO_SESSION = "sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys"
SLAP2_SESSION = "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys"


def _names(path: Path) -> list[str]:
    with h5py.File(path, "r") as hf:
        return [gabor._text(value) for value in hf["unit_names"][:]]


def _read_vector(path: Path, name: str, n_units: int):
    with h5py.File(path, "r") as hf:
        if name not in hf or hf[name].shape != (n_units,):
            return None
        return np.asarray(hf[name][:], dtype=float)


def _write_vector(path: Path, name: str, values: np.ndarray, **attrs) -> None:
    values = np.asarray(values, dtype=np.float64)
    with h5py.File(path, "a") as hf:
        if name in hf:
            del hf[name]
        hf.create_dataset(name, data=values)
        for key, value in attrs.items():
            hf.attrs[key] = value


def gumbel_p_values(responses, df_rf, batch: int = 64) -> np.ndarray:
    """Gumbel p-value per unit. Units are scored in batches so the shuffle fits."""
    x_pos, y_pos, orientations = siegle.rf_grid(df_rf)
    parts = []
    for start in range(0, responses.shape[0], batch):
        result = gabor.joint_null(
            responses[start:start + batch], df_rf, x_pos, y_pos, orientations,
            N_SHUFFLE, seed=0,
        )
        parts.append(np.asarray(result["gumbel_p"], dtype=float))
    p_values = np.concatenate(parts)
    return np.where(np.isfinite(p_values), p_values, 1.0)


def _full_run_files(directory: Path) -> list[Path]:
    files = []
    for path in sorted(directory.glob("*__optimized.h5")):
        if "__trial_" in path.name or "legacy_misaligned_units" in path.parts:
            continue
        files.append(path)
    return files


def _align(saved_names, available: dict[str, int]) -> np.ndarray:
    missing = [name for name in saved_names if name not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)} saved units are absent from the recording, "
            f"for example {missing[:3]}"
        )
    return np.asarray([available[name] for name in saved_names], dtype=int)


def compute_ephys(recompute: bool) -> None:
    directory = RESULTS / "ephys" / EPHYS_SESSION
    files = _full_run_files(directory)
    pending = []
    for path in files:
        names = _names(path)
        cached = None if recompute else _read_vector(path, "p_value_gumbel", len(names))
        pending.append((path, names, cached))
    if all(cached is not None for _, _, cached in pending):
        print("ephys: using saved Gumbel p-values", flush=True)
        _write_bh("ephys", [(path, names, cached) for path, names, cached in pending])
        return

    print("ephys: streaming spikes", flush=True)
    stream, _ = gabor.open_stream(
        "001637",
        f"sub-830794/{EPHYS_SESSION}.nwb",
    )
    with stream:
        # Anatomy once; spike times once. Each probe is then scored alone.
        units = stream.units_df(include_spikes=False)
        units["unit_name"] = units["unit_name"].map(gabor._text)
        row_of = {name: int(row) for row, name in enumerate(units["unit_name"])}
        bounds = np.concatenate([[0], np.asarray(
            stream._h5["units"]["spike_times_index"][:], dtype=np.int64
        )])
        flat = np.asarray(stream._h5["units"]["spike_times"][:], dtype=float)
        df_rf = stream.gabor_rf_df()
    scored = []
    for path, names, cached in pending:
        if cached is not None:
            print(f"  {path.name}: saved Gumbel p-values", flush=True)
            scored.append((path, names, cached))
            continue
        rows = _align(names, row_of)
        spikes = [flat[int(bounds[row]):int(bounds[row + 1])] for row in rows]
        started = time.perf_counter()
        responses, _ = gabor.spike_responses(
            spikes, df_rf, gabor.EPHYS_DELAYS, gabor.EPHYS_DURATIONS
        )
        p_values = gumbel_p_values(responses, df_rf)
        elapsed = time.perf_counter() - started
        _write_vector(
            path, "p_value_gumbel", p_values,
            p_value_gumbel_n_shuffle=N_SHUFFLE,
        )
        print(
            f"  {path.stem.split('__')[-2]}: {len(names)} units in {elapsed:.1f} s "
            f"({elapsed / len(names):.3f} s/unit)",
            flush=True,
        )
        scored.append((path, names, p_values))
        del spikes, responses
    _write_bh("ephys", scored)


def _plane_events(stream, plane, df_rf, margin):
    series = stream.nwb.processing[plane]["event_timeseries"]
    times = np.asarray(series.timestamps[:], dtype=float)
    i0 = int(np.searchsorted(times, float(df_rf["start_time"].min()) - margin, side="left"))
    i1 = int(np.searchsorted(times, float(df_rf["stop_time"].max()) + margin, side="right"))
    block = np.asarray(series.data[i0:i1], dtype=np.float64)
    if block.shape[0] != i1 - i0:
        block = block.T
    roi_ids = np.asarray(series.rois.data[:])
    names = [f"{plane}_roi{roi_id}" for roi_id in roi_ids]
    traces, kept_times = siegle.drop_missing_samples(block.T, times[i0:i1])
    return names, traces, kept_times


def compute_mesoscope(recompute: bool) -> None:
    directory = RESULTS / "meso" / MESO_SESSION / "events"
    files = [path for path in _full_run_files(directory) if "__trial_" not in path.name]
    # Trial splits are already excluded. Plane files remain.
    pending = []
    for path in files:
        names = _names(path)
        cached = None if recompute else _read_vector(path, "p_value_gumbel", len(names))
        pending.append((path, names, cached))
    if all(cached is not None for _, _, cached in pending):
        print("mesoscope: using saved Gumbel p-values", flush=True)
        _write_bh("mesoscope", [(path, names, cached) for path, names, cached in pending])
        return

    print("mesoscope: streaming events", flush=True)
    stream, _ = gabor.open_stream(
        "001768",
        f"sub-832700/{MESO_SESSION}.nwb",
    )
    margin = float(max(gabor.MESO_DELAYS) + max(gabor.MESO_DURATIONS) + 1.0)
    with stream:
        df_rf = stream.stim_df("RF mapping_presentations")
        loaded = {}
        for path, names, cached in pending:
            if cached is not None:
                continue
            plane = path.stem.split("__")[-2]
            print(f"  loading {plane}", flush=True)
            available, traces, times = _plane_events(stream, plane, df_rf, margin)
            loaded[plane] = (available, traces, times)
    scored = []
    for path, names, cached in pending:
        if cached is not None:
            print(f"  {path.stem.split('__')[-2]}: saved Gumbel p-values", flush=True)
            scored.append((path, names, cached))
            continue
        plane = path.stem.split("__")[-2]
        available, traces, times = loaded[plane]
        index = {name: i for i, name in enumerate(available)}
        rows = _align(names, index)
        started = time.perf_counter()
        responses, _ = gabor.trace_responses(
            traces[rows], times, df_rf, gabor.MESO_DELAYS, gabor.MESO_DURATIONS
        )
        p_values = gumbel_p_values(responses, df_rf)
        elapsed = time.perf_counter() - started
        _write_vector(
            path, "p_value_gumbel", p_values,
            p_value_gumbel_n_shuffle=N_SHUFFLE,
        )
        print(
            f"  {plane}: {len(names)} units in {elapsed:.1f} s "
            f"({elapsed / len(names):.3f} s/unit)",
            flush=True,
        )
        scored.append((path, names, p_values))
    _write_bh("mesoscope", scored)


def _slap2_events(stream, dmd, channel, df_rf):
    margin = float(max(gabor.SLAP2_DELAYS) + max(gabor.SLAP2_DURATIONS) + 1.0)
    t0 = float(df_rf["start_time"].min() - margin)
    t1 = float(df_rf["stop_time"].max() + margin)
    dff = stream.slap2_dff(dmd, channel, t_start=t0, t_stop=t1)
    traces = dff.to_numpy(dtype=np.float64)
    times = dff.columns.to_numpy(dtype=float)
    roi_ids = stream.slap2_rois(dmd)["roi_id"].to_numpy()
    if len(roi_ids) != traces.shape[0]:
        raise RuntimeError(f"{dmd} {channel}: {len(roi_ids)} ROI ids for {traces.shape[0]} traces")
    segments = gabor._bouts(times)
    events = np.empty_like(traces)
    tau = TAU_D[channel]
    for roi in range(traces.shape[0]):
        _, event_train = deconvolve_bouts(traces[roi], times, segments, tau)
        events[roi] = event_train
    events, times = siegle.drop_missing_samples(events, times)
    names = [f"{dmd}_{channel}_roi{roi_id}" for roi_id in roi_ids]
    return names, events, times


def compute_slap2(recompute: bool) -> None:
    directory = RESULTS / "slap2" / SLAP2_SESSION / "events"
    files = _full_run_files(directory)
    by_channel = {"green": [], "red": []}
    for path in files:
        if "rf-events-green" in path.name:
            channel = "green"
        elif "rf-events-red" in path.name:
            channel = "red"
        else:
            continue
        names = _names(path)
        cached = None if recompute else _read_vector(path, "p_value_gumbel", len(names))
        by_channel[channel].append((path, names, cached))

    needs_stream = any(
        cached is None for group in by_channel.values() for _, _, cached in group
    )
    loaded = {}
    df_rf = None
    if needs_stream:
        print(f"slap2: opening {gabor.SLAP2_NWB.name}", flush=True)
        stream = utils.open_local(gabor.SLAP2_NWB)
        with stream:
            df_rf = stream.stim_df("rf_mapping")
            for channel, group in by_channel.items():
                for path, names, cached in group:
                    if cached is not None:
                        continue
                    dmd = path.stem.split("__")[-2]
                    print(f"  deconvolving {dmd} {channel}", flush=True)
                    loaded[(dmd, channel)] = _slap2_events(stream, dmd, channel, df_rf)
    else:
        print("slap2: using saved Gumbel p-values", flush=True)

    for channel, group in by_channel.items():
        scored = []
        for path, names, cached in group:
            if cached is not None:
                scored.append((path, names, cached))
                continue
            dmd = path.stem.split("__")[-2]
            available, events, times = loaded[(dmd, channel)]
            index = {name: i for i, name in enumerate(available)}
            rows = _align(names, index)
            started = time.perf_counter()
            responses, _ = gabor.trace_responses(
                events[rows], times, df_rf, gabor.SLAP2_DELAYS, gabor.SLAP2_DURATIONS
            )
            p_values = gumbel_p_values(responses, df_rf)
            elapsed = time.perf_counter() - started
            _write_vector(
                path, "p_value_gumbel", p_values,
                p_value_gumbel_n_shuffle=N_SHUFFLE,
            )
            print(
                f"  {dmd} {channel}: {len(names)} units in {elapsed:.1f} s "
                f"({elapsed / len(names):.3f} s/unit)",
                flush=True,
            )
            scored.append((path, names, p_values))
        _write_bh(f"slap2 {channel}", scored)


def _write_bh(family: str, scored) -> None:
    p_values = np.concatenate([p for _, _, p in scored])
    q_values = gabor.bh_fdr(p_values)
    print(
        f"{family}: BH q ≤ 0.05 for {(q_values <= 0.05).sum()} / {len(q_values)} units",
        flush=True,
    )
    cursor = 0
    for path, names, _ in scored:
        q_group = q_values[cursor:cursor + len(names)]
        cursor += len(names)
        _write_vector(
            path, "p_value_bh", q_group,
            p_value_bh_family=family,
            p_value_bh_n_units=int(len(q_values)),
            p_value_bh_q_level=0.05,
        )


def main(recompute: bool = False) -> None:
    compute_ephys(recompute)
    compute_mesoscope(recompute)
    compute_slap2(recompute)


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
