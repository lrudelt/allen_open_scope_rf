"""Time the Gabor unit-level null on three units of each modality.

Each of 1000 shuffles reassigns presentations to positions once per
orientation and applies that same reassignment to every delay x duration
window. Two tests are computed from those shuffles:

* the previous counting p-value, one per window and orientation,
  ``(1 + #{surrogate > observed}) / (n_shuffle + 1)``
* the studentized maximum over windows and orientations, with a Gumbel
  fit by method of moments and a survival-function p-value

Dividing the map statistic by the grand mean, as the spike-rate chi-squared
does, does not change either p-value: the mean is invariant under a
permutation, and studentizing each window removes a window-specific scale.

    cd code/scripts
    python test_significance_Gabor.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append(str(Path(__file__).resolve().parent))

import rf_siegle_ophys as siegle
import utils
from oasis_deconvolve_slap2 import TAU_D, deconvolve_bouts

N_SHUFFLE = 1000
N_UNITS = 3
EULER = 0.5772156649015329

REPO = Path(__file__).resolve().parents[2]
SLAP2_NWB = (
    REPO / "data" / "slap2" / "sub-829704"
    / "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb"
)

EPHYS_DELAYS = (0.0, 0.025, 0.5, 0.1)
EPHYS_DURATIONS = (0.1, 0.2, 0.3)
MESO_DELAYS = (0.0, 0.1, 0.2, 0.3, 0.5)
MESO_DURATIONS = (0.25, 0.5, 1.0)
SLAP2_DELAYS = (0.0, 0.025, 0.05, 0.1, 0.2)
SLAP2_DURATIONS = (0.1, 0.25, 0.5)


def counting_p(observed, surrogate):
    """Previous test. ``surrogate`` is (n_shuffle, ...) and ``observed`` is (...)."""
    greater = np.sum(surrogate > observed, axis=0)
    return (greater + 1) / (surrogate.shape[0] + 1)


def gumbel_mom(draws):
    """Method-of-moments Gumbel fit. ``draws`` is (n_units, n_shuffle)."""
    beta = np.std(draws, axis=1, ddof=1) * np.sqrt(6.0) / np.pi
    mu = np.mean(draws, axis=1) - EULER * beta
    return mu, beta


def gumbel_sf(x, mu, beta):
    """Gumbel survival function. A non-positive scale returns 1."""
    x, mu, beta = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(mu, dtype=float), np.asarray(beta, dtype=float)
    )
    out = np.ones(x.shape, dtype=float)
    ok = np.isfinite(beta) & (beta > 0) & np.isfinite(mu) & np.isfinite(x)
    z = (x[ok] - mu[ok]) / beta[ok]
    out[ok] = -np.expm1(-np.exp(-z))
    return out


def bh_fdr(p):
    """Benjamini-Hochberg q-values, same accumulation as the Zebra notebooks."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((n / np.arange(n, 0, -1)) * ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _cell_slots(x, y):
    cells = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        cells.setdefault((int(xi), int(yi)), []).append(i)
    return [np.asarray(slots, dtype=int) for slots in cells.values()]


def _cell_means(values, perms, cell_slots):
    """Mean response in each grid cell after each presentation permutation.

    values: (n_units, n_windows, n_presentations)
    perms:  (n_shuffle, n_presentations), or (1, n_presentations) for the observed
    returns (n_cells, n_shuffle, n_units, n_windows)
    """
    means = []
    for slots in cell_slots:
        gathered = np.moveaxis(values[:, :, perms[:, slots]], 2, 0)
        with np.errstate(invalid="ignore"):
            means.append(np.nanmean(gathered, axis=-1))
    return np.stack(means, axis=0)


def _sum_of_squares(values, perms, cell_slots, grand):
    """Map statistic for each permutation, accumulated one grid cell at a time.

    ``perms`` is (n_shuffle, n_presentations). The result is
    (n_shuffle, n_units, n_windows). Cells are not stacked, so a full probe
    stays in memory.
    """
    acc = np.zeros((perms.shape[0],) + grand.shape, dtype=np.float64)
    for slots in cell_slots:
        gathered = np.moveaxis(values[:, :, perms[:, slots]], 2, 0)
        with np.errstate(invalid="ignore"):
            cell_mean = np.nanmean(gathered, axis=-1)
        diff = cell_mean - grand[None]
        acc += np.where(np.isfinite(diff), diff * diff, 0.0)
    return acc


def orientation_statistics(values, x, y, n_shuffle, rng):
    """Observed and surrogate map statistics for one orientation.

    ``values`` is (n_units, n_windows, n_presentations). One permutation of
    those presentations is drawn per shuffle and applied to every window.
    The statistic is the sum of squared deviations of the repeat-mean map
    from the grand mean. Returns ``observed`` (n_units, n_windows) and
    ``surrogate`` (n_shuffle, n_units, n_windows).
    """
    n_pres = values.shape[-1]
    slots = _cell_slots(x, y)
    with np.errstate(invalid="ignore"):
        grand = np.nanmean(values, axis=-1)

    identity = np.arange(n_pres, dtype=int)[None, :]
    observed = _sum_of_squares(values, identity, slots, grand)[0]

    order = np.tile(np.arange(n_pres), (n_shuffle, 1))
    perms = rng.permuted(order, axis=1)
    surrogate = _sum_of_squares(values, perms, slots, grand)
    return observed, surrogate


def studentized_max_gumbel(observed, surrogate):
    """Studentize each window, take the maximum, and fit a Gumbel to that maximum.

    ``observed`` is (n_units, n_tests) and ``surrogate`` is
    (n_shuffle, n_units, n_tests). Windows with no surrogate spread are left
    out of the maximum. Returns the Gumbel p-value per unit, the per-test
    counting p-value, and the location of the largest studentized statistic.
    """
    null_mean = surrogate.mean(axis=0)
    with np.errstate(invalid="ignore"):
        null_spread = surrogate.std(axis=0, ddof=1)
    usable = np.isfinite(null_spread) & (null_spread > 0) & np.isfinite(observed)

    student_obs = np.full(observed.shape, -np.inf)
    student_sur = np.full(surrogate.shape, -np.inf)
    student_obs[usable] = (observed[usable] - null_mean[usable]) / null_spread[usable]
    student_sur[:, usable] = (
        (surrogate[:, usable] - null_mean[usable]) / null_spread[usable]
    )
    none = ~usable.any(axis=1)
    student_obs[none] = 0.0
    student_sur[:, none] = 0.0

    maximum_obs = student_obs.max(axis=1)
    maximum_sur = student_sur.max(axis=-1)
    mu, beta = gumbel_mom(maximum_sur.T)
    return {
        "gumbel_p": gumbel_sf(maximum_obs, mu, beta),
        "counting_p": counting_p(observed, surrogate),
        "maximum_index": student_obs.argmax(axis=1),
        "maximum_T": maximum_obs,
        "n_usable": usable.sum(axis=1),
        "null_mean_T": mu,
        "null_spread_T": beta,
    }


def previous_counting_test(responses, df_rf, x_pos, y_pos, orientations, n_shuffle, seed):
    """The existing per-window test, via ``rf_siegle_ophys._unit_stats``.

    Each window is shuffled on its own, with missing responses left in place.
    Returns p-values of shape (n_units, n_windows, n_orientations).
    """
    o_i, r_i, x_i, y_i = siegle.grid_indices(df_rf, x_pos, y_pos, orientations)
    n_rep = int(r_i.max()) + 1
    n_units, n_windows, _ = responses.shape
    n_ori = len(orientations)
    p_values = np.ones((n_units, n_windows, n_ori))
    for window in range(n_windows):
        grid = np.full((n_units, n_ori, n_rep, len(x_pos), len(y_pos)), np.nan)
        grid[:, o_i, r_i, x_i, y_i] = responses[:, window]
        for unit in range(n_units):
            _, _, _, p_orient = siegle._unit_stats(
                (seed + unit + window * n_units, grid[unit]), n_shuffle=n_shuffle
            )
            p_values[unit, window] = p_orient
    return p_values


def presentation_rates(spike_times, onsets, offsets, delay, duration):
    """Spike rate in [onset + delay, onset + delay + duration), per stimulus duration."""
    onsets = np.asarray(onsets, dtype=float)
    stimulus = np.maximum(np.asarray(offsets, dtype=float) - onsets, 1e-9)
    spikes = np.asarray(spike_times, dtype=float)
    if spikes.size == 0:
        return np.zeros(len(onsets))
    starts = onsets + delay
    i0 = np.searchsorted(spikes, starts, side="left")
    i1 = np.searchsorted(spikes, starts + duration, side="left")
    return (i1 - i0) / stimulus


def window_list(delays, durations):
    return [(float(delay), float(duration)) for delay in delays for duration in durations]


def spike_responses(spikes, df_rf, delays, durations):
    onsets = df_rf["start_time"].to_numpy(float)
    offsets = df_rf["stop_time"].to_numpy(float)
    windows = window_list(delays, durations)
    out = np.empty((len(spikes), len(windows), len(onsets)))
    for i, (delay, duration) in enumerate(windows):
        for unit, times in enumerate(spikes):
            out[unit, i] = presentation_rates(times, onsets, offsets, delay, duration)
    return out, windows


def trace_responses(traces, times, df_rf, delays, durations):
    onsets = df_rf["start_time"].to_numpy(float)
    windows = window_list(delays, durations)
    out = np.empty((traces.shape[0], len(windows), len(onsets)))
    for i, (delay, duration) in enumerate(windows):
        out[:, i] = siegle.window_means(
            traces, times,
            onsets + delay, onsets + delay + duration,
            siegle.min_samples_for(times, duration),
        )
    return out, windows


def by_orientation(responses, df_rf, x_pos, y_pos, orientations):
    o_i, _, x_i, y_i = siegle.grid_indices(df_rf, x_pos, y_pos, orientations)
    groups = []
    for o, orientation in enumerate(orientations):
        chosen = o_i == o
        groups.append({
            "values": responses[:, :, chosen],
            "x": x_i[chosen],
            "y": y_i[chosen],
            "orientation": float(orientation),
        })
    return groups


def joint_null(responses, df_rf, x_pos, y_pos, orientations, n_shuffle, seed):
    """1000 joint shuffles, then the counting p-values and the Gumbel p-value."""
    groups = by_orientation(responses, df_rf, x_pos, y_pos, orientations)
    observed = []
    surrogate = []
    rng = np.random.default_rng(seed)
    windows = responses.shape[1]
    for group in groups:
        obs, sur = orientation_statistics(
            group["values"], group["x"], group["y"], n_shuffle, rng
        )
        observed.append(obs)
        surrogate.append(sur)
    result = studentized_max_gumbel(
        np.concatenate(observed, axis=1), np.concatenate(surrogate, axis=2)
    )
    result["orientation_of_test"] = np.concatenate([
        np.full(windows, group["orientation"]) for group in groups
    ])
    return result


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def open_stream(dandiset, asset_path):
    session = utils.DandiSession(dandiset)
    hits = [asset for asset in session.assets() if asset.path == asset_path]
    if not hits:
        raise ValueError(f"{asset_path!r} not found in dandiset {dandiset}")
    asset = hits[0]
    print(f"  streaming {asset.path}", flush=True)
    return session.open(asset.identifier), Path(asset_path).stem


def load_ephys():
    stream, session = open_stream(
        "001637",
        "sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb",
    )
    with stream:
        units = stream.units_df(include_spikes=False)
        units["probe"] = units["probe"].map(_text)
        units["unit_name"] = units["unit_name"].map(_text)
        picked = units.loc[units["probe"] == "ProbeC"].head(N_UNITS)
        if len(picked) < N_UNITS:
            raise RuntimeError(f"ProbeC has {len(picked)} units, need {N_UNITS}")
        bounds = np.concatenate([[0], np.asarray(
            stream._h5["units"]["spike_times_index"][:], dtype=np.int64
        )])
        spikes = []
        for row in picked.index.to_numpy():
            spikes.append(np.asarray(
                stream._h5["units"]["spike_times"][int(bounds[row]):int(bounds[row + 1])],
                dtype=float,
            ))
        df_rf = stream.gabor_rf_df()
    return {
        "session": session,
        "unit_names": picked["unit_name"].tolist(),
        "kind": "spikes",
        "spikes": spikes,
        "df_rf": df_rf,
        "delays": EPHYS_DELAYS,
        "durations": EPHYS_DURATIONS,
    }


def _slice_events(stream, plane, df_rf, margin):
    series = stream.nwb.processing[plane]["event_timeseries"]
    times = np.asarray(series.timestamps[:], dtype=float)
    i0 = int(np.searchsorted(times, df_rf["start_time"].min() - margin, side="left"))
    i1 = int(np.searchsorted(times, df_rf["stop_time"].max() + margin, side="right"))
    block = np.asarray(series.data[i0:i1], dtype=np.float64)
    if block.shape[0] != i1 - i0:
        block = block.T
    roi_ids = np.asarray(series.rois.data[:])
    return block[:, :N_UNITS].T, times[i0:i1], roi_ids[:N_UNITS]


def load_mesoscope():
    stream, session = open_stream(
        "001768",
        "sub-832700/sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys.nwb",
    )
    with stream:
        df_rf = stream.stim_df("RF mapping_presentations")
        plane = "VISp_0"
        margin = float(max(MESO_DELAYS) + max(MESO_DURATIONS) + 1.0)
        traces, times, roi_ids = _slice_events(stream, plane, df_rf, margin)
    traces, times = siegle.drop_missing_samples(traces, times)
    return {
        "session": session,
        "unit_names": [f"{plane}_roi{roi_id}" for roi_id in roi_ids],
        "kind": "traces",
        "traces": traces,
        "times": times,
        "df_rf": df_rf,
        "delays": MESO_DELAYS,
        "durations": MESO_DURATIONS,
    }


def _bouts(times):
    if len(times) == 0:
        return pd.DataFrame(columns=["i_start", "i_stop"])
    dt = np.diff(times)
    gaps = np.where(dt > 5 * np.median(dt))[0] if len(dt) else np.array([], dtype=int)
    return pd.DataFrame({
        "i_start": np.concatenate([[0], gaps + 1]),
        "i_stop": np.concatenate([gaps, [len(times) - 1]]),
    })


def load_slap2():
    print(f"  opening {SLAP2_NWB.name}", flush=True)
    stream = utils.open_local(SLAP2_NWB)
    with stream:
        df_rf = stream.stim_df("rf_mapping")
        margin = float(max(SLAP2_DELAYS) + max(SLAP2_DURATIONS) + 1.0)
        t0 = float(df_rf["start_time"].min() - margin)
        t1 = float(df_rf["stop_time"].max() + margin)
        dff = stream.slap2_dff("DMD1", "green", t_start=t0, t_stop=t1)
        traces = dff.to_numpy(dtype=np.float64)[:N_UNITS]
        times = dff.columns.to_numpy(dtype=float)
        roi_ids = stream.slap2_rois("DMD1")["roi_id"].to_numpy()[:N_UNITS]
    segments = _bouts(times)
    events = np.full_like(traces, np.nan)
    for roi in range(traces.shape[0]):
        _, event_train = deconvolve_bouts(traces[roi], times, segments, TAU_D["green"])
        events[roi] = event_train
    events, times = siegle.drop_missing_samples(events, times)
    return {
        "session": SLAP2_NWB.stem,
        "unit_names": [f"DMD1_green_roi{roi_id}" for roi_id in roi_ids],
        "kind": "traces",
        "traces": events,
        "times": times,
        "df_rf": df_rf,
        "delays": SLAP2_DELAYS,
        "durations": SLAP2_DURATIONS,
    }


def responses_for(loaded):
    if loaded["kind"] == "spikes":
        return spike_responses(
            loaded["spikes"], loaded["df_rf"], loaded["delays"], loaded["durations"]
        )
    return trace_responses(
        loaded["traces"], loaded["times"], loaded["df_rf"],
        loaded["delays"], loaded["durations"],
    )


def _fmt_p(value):
    if not np.isfinite(value):
        return "nan"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def report(name, loaded, responses, windows, joint, previous, times):
    x_pos, y_pos, orientations = siegle.rf_grid(loaded["df_rf"])
    n_tests = joint["counting_p"].shape[1]
    print(
        f"{name}  session={loaded['session']}  units={len(loaded['unit_names'])}  "
        f"windows={len(windows)}  orientations={len(orientations)}  "
        f"presentations={responses.shape[-1]}  grid={len(x_pos)}x{len(y_pos)}  "
        f"shuffles={N_SHUFFLE}",
        flush=True,
    )
    print(f"  load                  {times['load']:.2f} s", flush=True)
    print(f"  response windows      {times['responses']:.2f} s", flush=True)
    print(f"  previous counting     {times['previous']:.2f} s", flush=True)
    print(f"  joint null + Gumbel   {times['joint']:.2f} s", flush=True)
    print(
        f"  joint null per unit   {times['joint'] / len(loaded['unit_names']):.2f} s",
        flush=True,
    )
    q_values = bh_fdr(joint["gumbel_p"])
    for unit, unit_name in enumerate(loaded["unit_names"]):
        test = int(joint["maximum_index"][unit])
        window = test % len(windows)
        delay, duration = windows[window]
        orientation = joint["orientation_of_test"][test]
        if joint["n_usable"][unit] == 0:
            chosen = "no window had surrogate spread"
        else:
            chosen = (
                f"best T={joint['maximum_T'][unit]:.2f} at "
                f"delay={delay:g}s duration={duration:g}s orientation={orientation:g}"
            )
        print(
            f"  {_text(unit_name)}  "
            f"Gumbel p={_fmt_p(joint['gumbel_p'][unit])}  "
            f"BH q={_fmt_p(q_values[unit])}  "
            f"min counting p={_fmt_p(joint['counting_p'][unit].min())}  "
            f"min previous p={_fmt_p(previous[unit].min())}  "
            f"{chosen}",
            flush=True,
        )
    print(
        f"  ({n_tests} window x orientation tests; "
        "BH q is only over these three units)",
        flush=True,
    )
    return {
        "modality": name,
        "load_s": times["load"],
        "previous_s": times["previous"],
        "joint_s": times["joint"],
    }


def run_modality(name, loader):
    print(f"\n=== {name} ===", flush=True)
    started = time.perf_counter()
    loaded = loader()
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    responses, windows = responses_for(loaded)
    response_s = time.perf_counter() - started

    x_pos, y_pos, orientations = siegle.rf_grid(loaded["df_rf"])
    started = time.perf_counter()
    previous = previous_counting_test(
        responses, loaded["df_rf"], x_pos, y_pos, orientations, N_SHUFFLE, seed=0
    )
    previous_s = time.perf_counter() - started

    started = time.perf_counter()
    joint = joint_null(
        responses, loaded["df_rf"], x_pos, y_pos, orientations, N_SHUFFLE, seed=0
    )
    joint_s = time.perf_counter() - started

    return report(
        name, loaded, responses, windows, joint, previous,
        {"load": load_s, "responses": response_s, "previous": previous_s, "joint": joint_s},
    )


def _self_check():
    """A single bright cell must be detected, and a flat map must sit at the floor."""
    n_rep, n_shuffle = 4, 200
    x, y = [], []
    values = np.zeros((1, 2, n_rep * 4))
    cursor = 0
    for yi in range(2):
        for xi in range(2):
            x.extend([xi] * n_rep)
            y.extend([yi] * n_rep)
            if xi == 0 and yi == 0:
                values[0, 0, cursor:cursor + n_rep] = 5.0
            cursor += n_rep
    observed, surrogate = orientation_statistics(
        values, np.asarray(x), np.asarray(y), n_shuffle, np.random.default_rng(0)
    )
    expected = (5 - 1.25) ** 2 + 3 * (0 - 1.25) ** 2
    if abs(observed[0, 0] - expected) > 1e-6:
        raise AssertionError(f"map statistic {observed[0, 0]} != {expected}")
    if observed[0, 1] != 0:
        raise AssertionError("a flat window produced a non-zero statistic")
    result = studentized_max_gumbel(observed, surrogate)
    floor = 1 / (n_shuffle + 1)
    if result["counting_p"][0, 1] != floor:
        raise AssertionError(f"flat-map counting p {result['counting_p'][0, 1]} != {floor}")
    if result["counting_p"][0, 0] > 0.05 or result["gumbel_p"][0] > 0.05:
        raise AssertionError(
            f"bright cell was not detected: counting {result['counting_p'][0, 0]}, "
            f"Gumbel {result['gumbel_p'][0]}"
        )
    q = bh_fdr(np.array([result["gumbel_p"][0], 0.5]))
    if not (q[0] <= q[1] and q[0] <= 1):
        raise AssertionError(f"BH q-values out of order: {q}")


def main():
    _self_check()
    summaries = []
    for name, loader in (
        ("ephys", load_ephys),
        ("mesoscope", load_mesoscope),
        ("slap2", load_slap2),
    ):
        try:
            summaries.append(run_modality(name, loader))
        except Exception:
            print(f"\n{name} failed:", flush=True)
            traceback.print_exc()
    if summaries:
        print("\nruntime summary (seconds)", flush=True)
        print(f"  {'modality':<12} {'load':>8} {'previous':>10} {'joint null':>12}", flush=True)
        for row in summaries:
            print(
                f"  {row['modality']:<12} {row['load_s']:8.2f} "
                f"{row['previous_s']:10.2f} {row['joint_s']:12.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
