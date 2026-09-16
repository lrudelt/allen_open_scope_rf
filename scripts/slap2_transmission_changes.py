# %% Imports
"""Pre vs post oddball transmission in the two standard-control blocks.

For each colour channel, OASIS deconvolution and iterative event detection
(see `oasis_deconvolve_slap2.py`) are run independently on Control block 1.1
(before the oddball) and Control block 1.2 (after it). Each unit contributes:

* median event amplitude — median over events of the time-integral of
  ``s`` from the threshold crossing until ``s`` decays to zero or a new
  peak begins (the trace rises again). Sub-threshold tails are included.
* event rate — events per second of imaged time in the block
* baseline mean — mean of non-zero, event-excluded OASIS coefficients
* activity — total event area (sum of those integrals, tails included)

Significance is an unpaired Mood's median test (``scipy.stats.median_test``)
on the per-event amplitudes, and on the per-bout rates, baselines, and
activities. Units with p < 0.01 are drawn bold; the rest light.

CLI (analysis + figure)::

    python slap2_transmission_changes.py \\
        --nwb-path ../../data/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb

Plot only, from a results CSV already on disk::

    python slap2_transmission_changes.py --plot-only --dmd both

Or run the ``# %% Plot`` cell at the bottom in the Interactive Window.
The ``# %% Event waveforms`` cell writes ``event_waveforms_{green|red}_<stem>.png``:
both DMDs, pre vs post side by side, for the units with the largest relative
amplitude increase and decrease in that channel.

Requires `oasis-deconv`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_test

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

from oasis_deconvolve_slap2 import (  # noqa: E402
    DEFAULT_NWB,
    TAU_D,
    deconvolve_bouts,
    event_onsets,
    iterative_threshold,
)
import utils  # noqa: E402

DEFAULT_OUT = Path('../../results/slap2/transmission')
ALPHA = 0.01
MIN_SAMPLES = 3
ATOL = 1e-12
WAVE_T_PRE = 0.005
WAVE_T_POST = 0.050
WAVE_N_EXAMPLES = 20

trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def control_blocks(stream):
    """Control block 1.1 (pre-oddball) and 1.2 (post-oddball)."""
    rows = []
    for name in stream.stim_tables():
        df = stream.stim_df(name)
        if 'BlockNumber' not in df.columns or 'BlockLabel' not in df.columns:
            continue
        for (bn, label), g in df.groupby(['BlockNumber', 'BlockLabel']):
            rows.append({
                'block': int(bn), 'label': str(label), 'table': name,
                'n_trials': len(g),
                'start': float(g.start_time.min()),
                'stop': float(g.stop_time.max()),
            })
    blocks = pd.DataFrame(rows).sort_values(['start', 'block']).reset_index(drop=True)
    std = blocks[blocks.table == 'standard_control']
    if len(std) < 2:
        raise ValueError(
            'Need two standard_control blocks (1.1 before and 1.2 after the '
            f'oddball); found {len(std)} in this session.'
        )
    return std.iloc[0], std.iloc[-1], blocks


def segments_in_slice(segments: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Reindex imaging bouts onto a time-sliced trace (i_start/i_stop local)."""
    if times.size == 0:
        return segments.iloc[0:0].copy()
    t0, t1 = float(times[0]), float(times[-1])
    rows = []
    for bout in segments.itertuples():
        if bout.stop_time < t0 or bout.start_time > t1:
            continue
        i0 = int(np.searchsorted(times, bout.start_time))
        i1 = int(np.searchsorted(times, bout.stop_time, side='right')) - 1
        if i1 < i0:
            continue
        rows.append({
            'start_time': float(times[i0]),
            'stop_time': float(times[i1]),
            'duration': float(times[i1] - times[i0]),
            'n_samples': i1 - i0 + 1,
            'i_start': i0,
            'i_stop': i1,
        })
    return pd.DataFrame(rows)


def event_runs(events: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) index ranges of contiguous True runs."""
    idx = np.flatnonzero(events)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    stops = np.concatenate([idx[cuts], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def extend_event(s: np.ndarray, i0: int, atol: float = ATOL) -> int:
    """First index after the event that starts at ``i0``.

    Walks forward from the threshold crossing until ``s`` hits ~0 or, after
    having descended from the peak, starts rising again (a new peak).
    """
    n = len(s)
    i = i0
    seen_descent = False
    while i + 1 < n:
        cur, nxt = s[i], s[i + 1]
        if not np.isfinite(nxt) or nxt <= atol:
            break
        if np.isfinite(cur) and nxt < cur:
            seen_descent = True
        elif seen_descent and nxt > cur:
            break
        i += 1
    return i + 1


def event_spans(s: np.ndarray, events: np.ndarray, atol: float = ATOL) -> list[tuple[int, int]]:
    """[start, stop) for each threshold-crossing, extended through the tail."""
    onsets = event_onsets(events)
    spans = []
    occupied = -1
    for i0 in onsets:
        if i0 < occupied or not np.isfinite(s[i0]):
            continue
        i1 = extend_event(s, int(i0), atol=atol)
        if i1 <= i0:
            continue
        spans.append((int(i0), int(i1)))
        occupied = i1
    return spans


def span_mask(n: int, spans: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for i0, i1 in spans:
        m[i0:i1] = True
    return m


def event_areas(s: np.ndarray, times: np.ndarray,
                spans: list[tuple[int, int]]) -> np.ndarray:
    """Integral of s on each event span (including sub-threshold tail), in s·a.u."""
    dt = sample_dt(times)
    areas = []
    for i0, i1 in spans:
        height = np.clip(s[i0:i1], 0, None)
        t = times[i0:i1]
        ok = np.isfinite(height) & np.isfinite(t)
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            areas.append(float(height[ok][0] * dt) if np.isfinite(dt) else np.nan)
        else:
            areas.append(float(trapz(height[ok], t[ok])))
    return np.asarray(areas, dtype=float)


def sample_dt(times: np.ndarray) -> float:
    t = times[np.isfinite(times)]
    if t.size < 2:
        return np.nan
    return float(np.median(np.diff(t)))


def imaged_duration(times: np.ndarray, mask: np.ndarray) -> float:
    dt = sample_dt(times)
    n = int(np.isfinite(times[mask]).sum())
    return float(n * dt) if np.isfinite(dt) else 0.0


def bout_observables(s, times, spans, bouts) -> dict:
    """Per-imaging-bout amplitude / rate / baseline / activity, plus pooled event areas."""
    occupied = span_mask(len(s), spans)
    amp, rate, base, act = [], [], [], []
    all_areas = []
    for bout in bouts.itertuples():
        sl = slice(int(bout.i_start), int(bout.i_stop) + 1)
        m = np.zeros(len(s), dtype=bool)
        m[sl] = np.isfinite(s[sl])
        if m.sum() < 20:
            continue
        local = [(i0, i1) for i0, i1 in spans if m[i0]]
        areas = event_areas(s, times, local)
        dur = imaged_duration(times, m)
        nz = m & (np.abs(s) > ATOL) & ~occupied
        all_areas.append(areas)
        amp.append(float(np.median(areas)) if areas.size else np.nan)
        rate.append((len(areas) / dur) if dur > 0 else np.nan)
        base.append(float(np.mean(s[nz])) if nz.any() else np.nan)
        act.append(float(np.sum(areas)) if areas.size else 0.0)
    areas = (np.concatenate(all_areas) if all_areas
             else np.array([], dtype=float))
    return dict(
        areas=areas,
        bout_amplitudes=np.asarray(amp, dtype=float),
        bout_rates=np.asarray(rate, dtype=float),
        bout_baselines=np.asarray(base, dtype=float),
        bout_activities=np.asarray(act, dtype=float),
    )


def median_p(a, b) -> float:
    """Unpaired Mood's median test; nan if either sample is too small."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < MIN_SAMPLES or b.size < MIN_SAMPLES:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, p, _, _ = median_test(a, b)
        return float(p)
    except ValueError:
        return np.nan


def analyse_roi_block(y, times, bouts, tau_d, n_sd, min_baseline_frac) -> dict:
    """OASIS + event detection for one ROI on one control-block slice."""
    _, s = deconvolve_bouts(y, times, bouts, tau_d)
    det = iterative_threshold(s, n_sd=n_sd,
                              min_baseline_frac=min_baseline_frac, atol=ATOL)
    events, thr = det['events'], det['threshold']
    spans = event_spans(s, events) if np.isfinite(thr) else []
    obs = bout_observables(s, times, spans, bouts)
    areas = obs['areas']
    in_block = np.isfinite(s)
    dur = imaged_duration(times, in_block)
    occupied = span_mask(len(s), spans)
    nz_base = in_block & (np.abs(s) > ATOL) & ~occupied
    return dict(
        threshold=thr, n_iter=det['n_iter'],
        areas=areas,
        amplitude=float(np.median(areas)) if areas.size else np.nan,
        rate=(len(areas) / dur) if dur > 0 else np.nan,
        baseline=float(np.mean(s[nz_base])) if nz_base.any() else np.nan,
        activity=float(np.sum(areas)) if areas.size else 0.0,
        n_events=int(areas.size),
        duration=dur,
        n_bouts=int(len(bouts)),
        bout_amplitudes=obs['bout_amplitudes'],
        bout_rates=obs['bout_rates'],
        bout_baselines=obs['bout_baselines'],
        bout_activities=obs['bout_activities'],
    )


def analyse_session(nwb_path, dmd='DMD1', channels=('green', 'red'),
                    n_sd=1.0, min_baseline_frac=0.5,
                    tau_green=TAU_D['green'], tau_red=TAU_D['red']):
    nwb_path = Path(nwb_path)
    stream = utils.open_local(nwb_path)
    if stream.modality != 'slap2':
        raise ValueError(f'expected SLAP2 NWB, got {stream.modality!r}')

    pre, post, _blocks = control_blocks(stream)
    print(f'session {stream.nwb.session_id}')
    print(f'  pre  {pre.label}: {pre.start:.1f}–{pre.stop:.1f} s')
    print(f'  post {post.label}: {post.start:.1f}–{post.stop:.1f} s')

    rois = stream.slap2_rois(dmd)
    segments = stream.slap2_segments(dmd)
    tau = {'green': tau_green, 'red': tau_red}
    periods = {'pre': pre, 'post': post}

    # channel -> period -> roi -> result
    results = {ch: {p: {} for p in periods} for ch in channels}

    for ch in channels:
        for period, block in periods.items():
            print(f'{dmd}/{ch}  {period}  {block.label}')
            dff = stream.slap2_dff(dmd, ch, t_start=block.start, t_stop=block.stop)
            times = dff.columns.to_numpy(dtype=float)
            bouts = segments_in_slice(segments, times)
            print(f'  {len(bouts)} bouts, {len(rois)} ROIs')
            for i, roi in enumerate(rois.index):
                y = dff.loc[int(roi)].to_numpy(dtype=float)
                results[ch][period][int(roi)] = analyse_roi_block(
                    y, times, bouts, tau[ch], n_sd, min_baseline_frac)
                if (i + 1) % 10 == 0 or i + 1 == len(rois):
                    print(f'    ROI {i + 1}/{len(rois)}')

    rows = []
    for ch in channels:
        for roi in rois.index:
            pre_r, post_r = results[ch]['pre'][int(roi)], results[ch]['post'][int(roi)]
            p_amp = median_p(pre_r['areas'], post_r['areas'])
            p_rate = median_p(pre_r['bout_rates'], post_r['bout_rates'])
            p_base = median_p(pre_r['bout_baselines'], post_r['bout_baselines'])
            p_act = median_p(pre_r['bout_activities'], post_r['bout_activities'])
            rows.append({
                'session': stream.nwb.session_id,
                'dmd': dmd, 'channel': ch, 'roi': int(roi),
                'roi_id': int(rois.loc[roi, 'roi_id']),
                'pre_label': pre.label, 'post_label': post.label,
                'pre_amplitude': pre_r['amplitude'],
                'post_amplitude': post_r['amplitude'],
                'p_amplitude': p_amp,
                'sig_amplitude': int(np.isfinite(p_amp) and p_amp < ALPHA),
                'pre_rate': pre_r['rate'],
                'post_rate': post_r['rate'],
                'p_rate': p_rate,
                'sig_rate': int(np.isfinite(p_rate) and p_rate < ALPHA),
                'pre_baseline': pre_r['baseline'],
                'post_baseline': post_r['baseline'],
                'p_baseline': p_base,
                'sig_baseline': int(np.isfinite(p_base) and p_base < ALPHA),
                'pre_activity': pre_r['activity'],
                'post_activity': post_r['activity'],
                'p_activity': p_act,
                'sig_activity': int(np.isfinite(p_act) and p_act < ALPHA),
                'n_events_pre': pre_r['n_events'],
                'n_events_post': post_r['n_events'],
                'duration_pre': pre_r['duration'],
                'duration_post': post_r['duration'],
                'n_bouts_pre': pre_r['n_bouts'],
                'n_bouts_post': post_r['n_bouts'],
                'threshold_pre': pre_r['threshold'],
                'threshold_post': post_r['threshold'],
                'n_iter_pre': pre_r['n_iter'],
                'n_iter_post': post_r['n_iter'],
                'n_sd': n_sd,
            })
    return pd.DataFrame(rows)


def _sig_mask(series) -> np.ndarray:
    if series.dtype == bool or np.issubdtype(series.dtype, np.number):
        return np.asarray(series, dtype=bool)
    return series.astype(str).str.lower().isin(['true', '1', 'yes', 't']).to_numpy()


def _scatter(ax, pre, post, sig, xlabel, ylabel, title, color):
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    sig = np.asarray(sig, bool)
    ok = np.isfinite(pre) & np.isfinite(post)
    light, bold = ok & ~sig, ok & sig
    ax.scatter(pre[light], post[light], s=28, c=color, alpha=0.28,
               linewidths=0, zorder=2, label=f'n.s. (n={int(light.sum())})')
    ax.scatter(pre[bold], post[bold], s=48, c=color, alpha=1.0,
               linewidths=0.4, edgecolors='k', zorder=3,
               label=f'p < {ALPHA} (n={int(bold.sum())})')
    finite = np.concatenate([pre[ok], post[ok]]) if ok.any() else np.array([])
    if finite.size:
        lo, hi = np.nanmin(finite), np.nanmax(finite)
        pad = 0.05 * (hi - lo if hi > lo else abs(hi) + 1e-6)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, '--', color='.5', lw=0.8, zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper left', frameon=False)


def plot_pre_post(df: pd.DataFrame, outfile=None, show=True):
    """Four pre×post scatters per channel. Light = n.s., bold = median-test p<0.01."""
    df = df.copy()
    for name in ('amplitude', 'rate', 'baseline', 'activity'):
        pcol, scol = f'p_{name}', f'sig_{name}'
        if pcol in df.columns:
            p = pd.to_numeric(df[pcol], errors='coerce')
            df[scol] = np.isfinite(p) & (p < ALPHA)
        elif scol in df.columns:
            df[scol] = _sig_mask(df[scol])

    channels = list(df.channel.unique())
    colours = {'green': 'tab:green', 'red': 'tab:red'}
    observables = [
        ('amplitude', 'pre_amplitude', 'post_amplitude', 'sig_amplitude',
         'Median event amplitude (s·a.u.)'),
        ('rate', 'pre_rate', 'post_rate', 'sig_rate',
         'Event rate (Hz)'),
        ('baseline', 'pre_baseline', 'post_baseline', 'sig_baseline',
         'Baseline mean (OASIS s)'),
        ('activity', 'pre_activity', 'post_activity', 'sig_activity',
         'Total activity (s·a.u.)'),
    ]
    observables = [obs for obs in observables if obs[1] in df.columns]
    fig, axs = plt.subplots(len(channels), len(observables),
                            figsize=(4.2 * len(observables), 4.2 * len(channels)),
                            squeeze=False)
    session = df.session.iloc[0] if len(df) and 'session' in df else ''
    dmd = df.dmd.iloc[0] if len(df) and 'dmd' in df else ''
    fig.suptitle(f'{session}  {dmd}  — control 1.1 (pre) vs 1.2 (post), '
                 f'Mood median test p < {ALPHA}', fontsize=11, y=1.02)

    for row, ch in enumerate(channels):
        sub = df[df.channel == ch]
        color = colours.get(ch, 'k')
        for col, (name, pre_c, post_c, sig_c, title) in enumerate(observables):
            _scatter(axs[row, col], sub[pre_c], sub[post_c], sub[sig_c],
                     f'pre  ({name})', f'post  ({name})',
                     f'{ch}  {title}', color)

    fig.tight_layout()
    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'saved {outfile}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _relative_change(pre, post) -> np.ndarray:
    pre, post = np.asarray(pre, dtype=float), np.asarray(post, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (post - pre) / pre


def event_waveform(s: np.ndarray, i0: int, i1: int, dt: float,
                   t_pre: float = WAVE_T_PRE, t_post: float = WAVE_T_POST) -> np.ndarray:
    """Align one event to [-t_pre, t_post]; pad the post-stop tail with zeros."""
    n_pre = int(round(t_pre / dt))
    n_post = int(round(t_post / dt))
    wave = np.zeros(n_pre + 1 + n_post, dtype=float)
    for k, j in enumerate(range(i0 - n_pre, i0)):
        if 0 <= j < len(s) and np.isfinite(s[j]):
            wave[k] = s[j]
        else:
            wave[k] = np.nan
    last = min(i1, i0 + n_post + 1, len(s))
    for j in range(i0, last):
        k = n_pre + (j - i0)
        if 0 <= k < len(wave) and np.isfinite(s[j]):
            wave[k] = s[j]
    return wave


def waveform_time(dt: float, t_pre: float = WAVE_T_PRE, t_post: float = WAVE_T_POST) -> np.ndarray:
    n_pre = int(round(t_pre / dt))
    n_post = int(round(t_post / dt))
    return (np.arange(-n_pre, n_post + 1) * dt) * 1e3  # ms


def collect_unit_waveforms(nwb_path, dmd, channel, roi, n_sd=1.0,
                           min_baseline_frac=0.5, tau_d=None,
                           t_pre=WAVE_T_PRE, t_post=WAVE_T_POST):
    """Deconvolve one ROI in both control blocks; return pre/post aligned waves."""
    tau_d = TAU_D[channel] if tau_d is None else tau_d
    stream = utils.open_local(nwb_path)
    pre, post, _ = control_blocks(stream)
    segments = stream.slap2_segments(dmd)
    by_period = {'pre': [], 'post': []}
    dt = np.nan
    for period, block in ('pre', pre), ('post', post):
        dff = stream.slap2_dff(dmd, channel, t_start=block.start, t_stop=block.stop)
        times = dff.columns.to_numpy(dtype=float)
        dt = sample_dt(times)
        bouts = segments_in_slice(segments, times)
        y = dff.loc[int(roi)].to_numpy(dtype=float)
        _, s = deconvolve_bouts(y, times, bouts, tau_d)
        det = iterative_threshold(s, n_sd=n_sd,
                                  min_baseline_frac=min_baseline_frac, atol=ATOL)
        spans = event_spans(s, det['events']) if np.isfinite(det['threshold']) else []
        for i0, i1 in spans:
            by_period[period].append(event_waveform(s, i0, i1, dt, t_pre, t_post))
    n_pre = int(round(t_pre / dt)) if np.isfinite(dt) else 1
    n_post = int(round(t_post / dt)) if np.isfinite(dt) else 1
    empty = n_pre + 1 + n_post
    stack = lambda xs: np.vstack(xs) if xs else np.zeros((0, empty))
    return dict(
        t_ms=waveform_time(dt, t_pre, t_post) if np.isfinite(dt) else np.array([]),
        pre_waves=stack(by_period['pre']),
        post_waves=stack(by_period['post']),
        n_pre=len(by_period['pre']),
        n_post=len(by_period['post']),
    )


def pick_extreme_amplitude_units(df: pd.DataFrame, channel=None, min_events: int = 20):
    """Rows with the largest relative amplitude increase and decrease."""
    df = df.copy()
    if channel is not None:
        df = df[df.channel == channel]
    df['rel'] = _relative_change(df['pre_amplitude'], df['post_amplitude'])
    enough = ((df['n_events_pre'] >= min_events)
              & (df['n_events_post'] >= min_events)
              & np.isfinite(df['rel']))
    sub = df[enough] if int(enough.sum()) >= 2 else df[np.isfinite(df['rel'])]
    if len(sub) == 0:
        raise ValueError('no units with a finite relative amplitude change')
    inc = sub.loc[sub['rel'].idxmax()]
    dec = sub.loc[sub['rel'].idxmin()]
    if inc.name == dec.name and len(sub) > 1:
        rest = sub.drop(inc.name)
        dec = rest.loc[rest['rel'].idxmin()]
    return inc, dec


def event_waveforms_path(nwb_path, channel, out_dir=DEFAULT_OUT) -> Path:
    return Path(out_dir) / f'event_waveforms_{channel}_{Path(nwb_path).stem}.png'


def _plot_wave_panel(ax, t_ms, waves, color, rng, n_examples, period_label):
    if len(waves):
        n_draw = min(n_examples, len(waves))
        idx = rng.choice(len(waves), size=n_draw, replace=False)
        for w in waves[idx]:
            ax.plot(t_ms, w, color=color, lw=0.55, alpha=0.28, zorder=1)
        ax.plot(t_ms, np.nanmean(waves, axis=0), color=color, lw=2.4,
                zorder=3, label=f'mean (n={len(waves)})')
        ax.legend(fontsize=7, loc='upper right', frameon=False)
    ax.axvline(0, color='.5', lw=0.7, ls=':', zorder=0)
    ax.axhline(0, color='.5', lw=0.5, zorder=0)
    ax.set_xlim(-WAVE_T_PRE * 1e3, WAVE_T_POST * 1e3)
    ax.set_title(period_label, fontsize=9)


def plot_extreme_event_waveforms(csv_paths, channel, nwb_path=DEFAULT_NWB,
                                 outfile=None, out_dir=DEFAULT_OUT,
                                 n_sd=None, min_baseline_frac=0.5,
                                 tau_green=TAU_D['green'], tau_red=TAU_D['red'],
                                 n_examples=WAVE_N_EXAMPLES, seed=0, show=True):
    """Pre vs post mean OASIS event for extreme units; both DMDs on one figure."""
    if isinstance(csv_paths, (str, Path)):
        p = Path(csv_paths)
        dmd = pd.read_csv(p, usecols=['dmd']).dmd.iloc[0]
        csv_paths = {dmd: p}
    csv_paths = {dmd: Path(p) for dmd, p in csv_paths.items()}
    dfs = {dmd: pd.read_csv(p) for dmd, p in csv_paths.items()}
    if n_sd is None:
        any_df = next(iter(dfs.values()))
        n_sd = float(any_df['n_sd'].iloc[0]) if 'n_sd' in any_df.columns else 1.0

    colour = {'green': 'tab:green', 'red': 'tab:red'}[channel]
    tau = tau_green if channel == 'green' else tau_red
    rng = np.random.default_rng(seed)
    session = next(iter(dfs.values())).session.iloc[0]

    rows = []
    for dmd in sorted(csv_paths):
        inc, dec = pick_extreme_amplitude_units(dfs[dmd], channel=channel)
        rows.append((dmd, 'largest increase', inc))
        rows.append((dmd, 'largest decrease', dec))

    fig, axs = plt.subplots(len(rows), 2, figsize=(8.5, 3.2 * len(rows)),
                            squeeze=False)
    fig.suptitle(
        f'{session}  {channel}  — mean OASIS event  '
        f'[{-WAVE_T_PRE * 1e3:.0f}, +{WAVE_T_POST * 1e3:.0f}] ms',
        fontsize=11, y=1.01)

    for r, (dmd, kind, row) in enumerate(rows):
        roi = int(row['roi'])
        rel = float(_relative_change(row['pre_amplitude'], row['post_amplitude']))
        rec = collect_unit_waveforms(
            nwb_path, dmd, channel, roi, n_sd=n_sd,
            min_baseline_frac=min_baseline_frac, tau_d=tau)
        t_ms = rec['t_ms']
        row_rng = np.random.default_rng(seed + r)
        for c, (waves, plabel) in enumerate((
            (rec['pre_waves'], f'pre  (n={rec["n_pre"]})'),
            (rec['post_waves'], f'post  (n={rec["n_post"]})'),
        )):
            ax = axs[r, c]
            header = f'{dmd}  {kind}  ROI {roi}  Δrel={rel:+.2f}'
            _plot_wave_panel(ax, t_ms, waves, colour, row_rng, n_examples,
                             f'{header}\n{plabel}' if c == 0 else plabel)
            if c == 0:
                ax.set_ylabel('OASIS $s$', fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel('Time from threshold crossing (ms)')
        # Same y-range for pre and post of this ROI: 3× the larger mean peak.
        means = []
        for waves in (rec['pre_waves'], rec['post_waves']):
            if len(waves):
                m = np.nanmean(waves, axis=0)
                if np.isfinite(m).any():
                    means.append(m[np.isfinite(m)])
        if means:
            peak = float(np.max(np.abs(np.concatenate(means))))
            ylim = (min(0.0, -0.05 * peak), 3.0 * peak)
            axs[r, 0].set_ylim(ylim)
            axs[r, 1].set_ylim(ylim)

    fig.tight_layout()
    if outfile is None:
        outfile = event_waveforms_path(nwb_path, channel, out_dir)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'saved {outfile}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_from_csv(csv_path, outfile=None, show=True):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if outfile is None:
        outfile = csv_path.with_suffix('.png')
    outfile = Path(outfile)
    return plot_pre_post(df, outfile=outfile, show=show)


def default_csv(nwb_path, dmd, out_dir=DEFAULT_OUT) -> Path:
    return Path(out_dir) / f'transmission_changes_{Path(nwb_path).stem}_{dmd}.csv'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nwb-path', default=DEFAULT_NWB)
    p.add_argument('--dmd', default='DMD1', choices=['DMD1', 'DMD2', 'both'])
    p.add_argument('--channel', default='both', choices=['green', 'red', 'both'])
    p.add_argument('--n-sd', type=float, default=1.0)
    p.add_argument('--min-baseline-frac', type=float, default=0.5)
    p.add_argument('--out-dir', default=str(DEFAULT_OUT))
    p.add_argument('--csv-path', default=None,
                   help='Results CSV (default: <out-dir>/transmission_changes_<stem>_<dmd>.csv)')
    p.add_argument('--plot-only', action='store_true',
                   help='Skip deconvolution; plot from --csv-path')
    p.add_argument('--tau-green', type=float, default=TAU_D['green'])
    p.add_argument('--tau-red', type=float, default=TAU_D['red'])
    p.add_argument('--no-show', action='store_true')
    return p.parse_args(argv)


def run_analysis(args=None):
    args = args or parse_args([])
    channels = ('green', 'red') if args.channel == 'both' else (args.channel,)
    csv_path = Path(args.csv_path) if args.csv_path else default_csv(
        args.nwb_path, args.dmd, args.out_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = analyse_session(args.nwb_path, args.dmd, channels,
                         n_sd=args.n_sd, min_baseline_frac=args.min_baseline_frac,
                         tau_green=args.tau_green, tau_red=args.tau_red)
    df.to_csv(csv_path, index=False)
    print(f'saved {csv_path}')
    return df, csv_path


def _dmds(args):
    return ('DMD1', 'DMD2') if args.dmd == 'both' else (args.dmd,)


def _interactive() -> bool:
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


def _cli(args=None):
    args = args or parse_args()
    csv_by_dmd = {}
    for dmd in _dmds(args):
        dmd_args = argparse.Namespace(**{
            **vars(args),
            'dmd': dmd,
            'csv_path': None if args.dmd == 'both' else args.csv_path,
        })
        csv_path = Path(args.csv_path) if args.csv_path and args.dmd != 'both' else default_csv(
            args.nwb_path, dmd, args.out_dir)
        if args.plot_only:
            if not csv_path.exists():
                raise FileNotFoundError(
                    f'no results at {csv_path}; run without --plot-only first')
        else:
            _, csv_path = run_analysis(dmd_args)
        csv_by_dmd[dmd] = csv_path
        plot_from_csv(csv_path, outfile=csv_path.with_suffix('.png'), show=not args.no_show)

    for ch in ('green', 'red'):
        plot_extreme_event_waveforms(
            csv_by_dmd, ch, nwb_path=args.nwb_path, out_dir=args.out_dir,
            n_sd=args.n_sd, min_baseline_frac=args.min_baseline_frac,
            tau_green=args.tau_green, tau_red=args.tau_red,
            show=not args.no_show,
        )


if __name__ == '__main__' and not _interactive():
    _cli()
    raise SystemExit(0)


# %% Analysis
# Interactive Window: run this cell to deconvolve both control blocks.
# Skip it (run the Plot cell instead) if the CSV is already on disk.
NWB_PATH = DEFAULT_NWB
DMD = 'both'  # 'DMD1', 'DMD2', or 'both'
if _interactive():
    DFS = {}
    CSV_PATHS = {}
    for _dmd in (('DMD1', 'DMD2') if DMD == 'both' else (DMD,)):
        DFS[_dmd], CSV_PATHS[_dmd] = run_analysis(parse_args([
            '--nwb-path', str(NWB_PATH),
            '--dmd', _dmd,
        ]))
    CSV_PATH = CSV_PATHS.get('DMD1', next(iter(CSV_PATHS.values())))


# %% Plot
# Interactive Window: run this cell alone to redraw from the saved CSV.
# Does not re-run OASIS.
if _interactive():
    _paths = CSV_PATHS if 'CSV_PATHS' in dir() else {
        dmd: default_csv(DEFAULT_NWB, dmd) for dmd in ('DMD1', 'DMD2')
    }
    for _csv in _paths.values():
        if Path(_csv).exists():
            plot_from_csv(_csv)


# %% Event waveforms
# event_waveforms_{green|red}_<stem>.png — both DMDs, pre vs post side by side.
# Traces that end early are zero-filled to +50 ms. Does not rebuild the ROI table.
if _interactive():
    _nwb = NWB_PATH if 'NWB_PATH' in dir() else DEFAULT_NWB
    _paths = CSV_PATHS if 'CSV_PATHS' in dir() else {
        dmd: default_csv(DEFAULT_NWB, dmd) for dmd in ('DMD1', 'DMD2')
    }
    _paths = {dmd: p for dmd, p in _paths.items() if Path(p).exists()}
    if _paths:
        for _ch in ('green', 'red'):
            plot_extreme_event_waveforms(_paths, _ch, nwb_path=_nwb, show=True)
