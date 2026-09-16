"""OASIS deconvolution of SLAP2 green (iGluSnFR4f) and red (RCaMP3) traces.

Deconvolves each imaging bout separately (SLAP2 is not continuous), detects
strong events by iterative 1-SD clipping of the *non-zero* OASIS coefficients,
and plots five example ROIs with the deconvolved trace under the raw ΔF/F for
both channels.

    python oasis_deconvolve_slap2.py \
        --nwb-path ../../data/slap2/sub-829704/sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb \
        --dmd DMD1

Requires `oasis-deconv` (`pip install oasis-deconv`).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.append('..')
import utils

try:
    from oasis.functions import deconvolve
except ImportError as exc:
    raise ImportError(
        "OASIS is not installed. In the waven_dandi env run: pip install oasis-deconv"
    ) from exc

DEFAULT_NWB = (
    '../../data/slap2/sub-829704/'
    'sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys.nwb'
)

# iGluSnFR4f decays in tens of ms; RCaMP3 is a cytosolic calcium indicator.
TAU_D = {'green': 0.020, 'red': 0.30}


def fill_nans(y: np.ndarray) -> np.ndarray:
    """Linear interpolate isolated NaNs; OASIS needs a finite vector."""
    out = np.asarray(y, dtype=float).copy()
    nans = ~np.isfinite(out)
    if nans.all():
        return out
    if nans.any():
        out[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), out[~nans])
    return out


def deconvolve_bouts(y: np.ndarray, times: np.ndarray, segments: pd.DataFrame,
                     tau_d: float) -> tuple[np.ndarray, np.ndarray]:
    """OASIS per imaging bout. Returns (denoised c, event train s), NaN in gaps."""
    c = np.full_like(y, np.nan, dtype=float)
    s = np.full_like(y, np.nan, dtype=float)
    y = np.asarray(y, dtype=float)
    times = np.asarray(times, dtype=float)

    for bout in segments.itertuples():
        sl = slice(int(bout.i_start), int(bout.i_stop) + 1)
        yy = y[sl]
        tt = times[sl]
        valid = np.isfinite(yy)
        if valid.sum() < 20:
            continue
        dt = np.median(np.diff(tt)) if len(tt) > 1 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            continue
        filled = fill_nans(yy)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = deconvolve(filled, tau_d=float(tau_d), tau_r=0.0,
                             framerate=float(1.0 / dt), penalty=1)
        c[sl] = res.c
        s[sl] = res.s
        # do not invent events in the original blanked samples
        c[sl][~valid] = np.nan
        s[sl][~valid] = np.nan
    return c, s


def iterative_threshold(s: np.ndarray, n_sd: float = 1.0, max_iter: int = 20,
                        min_baseline_frac: float = 0.5, atol: float = 1e-12) -> dict:
    """Threshold = mean + n_sd * SD of *non-zero* OASIS coefficients.

    Zeros are idle samples, not baseline. Mean and SD are computed only on
    coefficients with |s| > atol, starting from all of them and then dropping
    samples already called as events, until the event set stops changing or
    the next step would leave too few non-zero coefficients in the baseline.
    """
    finite = np.isfinite(s)
    nonzero = finite & (np.abs(s) > atol)
    n_nz = int(nonzero.sum())
    empty = dict(mu=np.nan, sd=np.nan, threshold=np.nan,
                 events=np.zeros(len(s), dtype=bool), n_iter=0,
                 baseline_frac=np.nan)
    if n_nz < 20:
        return empty

    baseline = nonzero.copy()
    accepted = None

    for n_iter in range(1, max_iter + 1):
        x = s[baseline]
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        thr = mu + n_sd * sd
        new_events = nonzero & (s > thr)
        new_baseline = nonzero & ~new_events
        frac = float(new_baseline.sum() / n_nz)

        if new_baseline.sum() < 20 or frac < min_baseline_frac or sd == 0.0:
            break

        unchanged = (accepted is not None
                     and np.array_equal(new_events, accepted['events']))
        accepted = dict(mu=mu, sd=sd, threshold=thr, events=new_events,
                        n_iter=n_iter, baseline_frac=frac)
        if unchanged:
            break
        baseline = new_baseline

    if accepted is None:
        accepted = dict(mu=mu, sd=sd, threshold=thr, events=new_events,
                        n_iter=n_iter, baseline_frac=frac)
    return accepted


def event_onsets(events: np.ndarray) -> np.ndarray:
    """Index of the first sample of each contiguous above-threshold run."""
    if not events.any():
        return np.array([], dtype=int)
    return np.flatnonzero(events & np.concatenate([[True], ~events[:-1]]))


def pick_rois(rois: pd.DataFrame, n: int) -> np.ndarray:
    """Even spread along the field of view, always including ROI 0."""
    along = np.argsort(rois.y.to_numpy())
    idx = along[np.linspace(0, len(along) - 1, n).astype(int)]
    if 0 not in idx:
        idx = np.append(idx, 0)
    return np.sort(np.unique(idx))[:n]


def plot_window(times, traces, roi_idx, t0, t1, outfile):
    """One column per channel, raw above deconvolved, one row per ROI."""
    n = len(roi_idx)
    fig = plt.figure(figsize=(14, max(8.0, 2.15 * n)))
    outer = GridSpec(n, 2, figure=fig, hspace=0.38, wspace=0.22,
                     left=0.07, right=0.99, top=0.96, bottom=0.04)
    colours = {'green': 'tab:green', 'red': 'tab:red'}
    win = (times >= t0) & (times <= t1)

    for i, roi in enumerate(roi_idx):
        for j, ch in enumerate(('green', 'red')):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.06)
            ax_raw = fig.add_subplot(inner[0])
            ax_dec = fig.add_subplot(inner[1], sharex=ax_raw)
            rec = traces[ch][int(roi)]
            t = times[win]
            y = rec['y'][win]
            s = rec['s'][win]
            ev = rec['events'][win]
            thr = rec['threshold']
            colour = colours[ch]

            ax_raw.plot(t, y, lw=0.6, color=colour)
            ax_raw.set_ylabel('$\\Delta F/F$', fontsize=8)
            ax_raw.tick_params(labelbottom=False, labelsize=8)
            ax_raw.set_title(f'ROI {roi}  {ch}', fontsize=9, loc='left', color=colour)

            ax_dec.plot(t, s, lw=0.55, color=colour, zorder=1)
            if np.isfinite(thr):
                s_lo = np.nanmin(s) if np.isfinite(s).any() else 0.0
                s_hi = np.nanmax(s) if np.isfinite(s).any() else thr
                ax_dec.axhspan(min(0.0, s_lo), thr, color=colour, alpha=0.12, zorder=0)
                ax_dec.axhline(thr, color='k', ls='--', lw=1.0, zorder=2)
                ax_dec.fill_between(t, thr, s, where=ev, color='k', alpha=0.25,
                                    linewidth=0, zorder=1)
                onsets = event_onsets(ev)
                if len(onsets):
                    ax_dec.scatter(t[onsets], s[onsets], s=14, marker='v',
                                   color='k', zorder=4, linewidths=0)
                    ax_raw.plot(t[onsets], np.full(len(onsets), np.nanmax(y) if np.isfinite(y).any() else 0),
                                ls='none', marker='|', ms=6, mew=0.7, color='k', alpha=0.7)
                ax_dec.set_ylim(bottom=min(-0.05 * max(s_hi, 1e-6), s_lo))

            ax_dec.set_ylabel('OASIS $s$', fontsize=8)
            ax_dec.tick_params(labelsize=8)
            if i == n - 1:
                ax_dec.set_xlabel('Time (s)', fontsize=8)
            ax_dec.set_xlim(t0, t1)

            n_ev_win = len(event_onsets(ev))
            ax_dec.text(0.99, 0.92,
                        f'thr = {thr:.3g}  ({n_ev_win} events in window, {rec["n_iter"]} iter)',
                        transform=ax_dec.transAxes, ha='right', va='top', fontsize=7,
                        color='.25')

    fig.suptitle(f'OASIS events, {t0:.0f}–{t1:.0f} s.  '
                 'Dashed line / shading: iterative mean + 1 SD of non-zero coefficients.  '
                 'Triangles: event onsets.',
                 fontsize=10)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f'saved {outfile}')


def main(nwb_path, dmd, n_rois, n_sd, min_baseline_frac, plot_start, plot_duration,
         out_dir, tau_green, tau_red):
    nwb_path = Path(nwb_path)
    stream = utils.open_local(nwb_path)
    if stream.modality != 'slap2':
        raise ValueError(f'expected a SLAP2 NWB, got modality={stream.modality!r}')

    rois = stream.slap2_rois(dmd)
    roi_idx = pick_rois(rois, n_rois)
    segments = stream.slap2_segments(dmd)
    print(f'session {stream.nwb.session_id}  {dmd}  {len(rois)} ROIs, '
          f'plotting {list(roi_idx)}')

    tau = {'green': tau_green, 'red': tau_red}
    traces = {ch: {} for ch in tau}
    times = None

    for ch in tau:
        print(f'deconvolving {ch} (tau_d = {tau[ch]*1e3:.0f} ms)...')
        dff = stream.slap2_dff(dmd, ch)
        if times is None:
            times = dff.columns.to_numpy(dtype=float)
        for roi in roi_idx:
            y = dff.loc[int(roi)].to_numpy(dtype=float)
            c, s = deconvolve_bouts(y, times, segments, tau[ch])
            det = iterative_threshold(s, n_sd=n_sd,
                                      min_baseline_frac=min_baseline_frac)
            traces[ch][int(roi)] = dict(y=y, c=c, s=s, **det)
            n_ev = len(event_onsets(det['events']))
            dur = np.isfinite(s).sum() * np.median(np.diff(times))
            print(f'  ROI {roi:3d}  mu={det["mu"]:.4f}  sd={det["sd"]:.4f}  '
                  f'thr={det["threshold"]:.4f}  events={n_ev:4d}  '
                  f'({n_ev / dur:.2f} Hz)  iter={det["n_iter"]}  '
                  f'nonzero baseline={100*det["baseline_frac"]:.0f}%')

    if plot_start is None:
        try:
            sc = stream.stim_df('standard_control')
            plot_start = float(sc.start_time.min())
        except Exception:
            plot_start = float(times[np.isfinite(times)][0])
    t0 = plot_start
    t1 = plot_start + plot_duration

    out_dir = Path(out_dir)
    outfile = out_dir / f'oasis_events_{nwb_path.stem}_{dmd}.png'
    plot_window(times, traces, roi_idx, t0, t1, outfile)

    rows = []
    for ch in tau:
        for roi in roi_idx:
            det = traces[ch][int(roi)]
            rows.append({
                'channel': ch, 'roi': int(roi),
                'tau_d': tau[ch], 'mu': det['mu'], 'sd': det['sd'],
                'threshold': det['threshold'],
                'n_events': len(event_onsets(det['events'])),
                'n_iter': det['n_iter'],
                'baseline_frac': det['baseline_frac'],
            })
    csv_path = out_dir / f'oasis_events_{nwb_path.stem}_{dmd}.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'saved {csv_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nwb-path', default=DEFAULT_NWB, help='SLAP2 NWB file')
    p.add_argument('--dmd', default='DMD1', help='Imaging plane, e.g. DMD1')
    p.add_argument('--n-rois', type=int, default=5, help='How many example ROIs to plot')
    p.add_argument('--n-sd', type=float, default=1.0,
                   help='Event threshold in SD above the mean of non-zero, '
                        'event-excluded OASIS coefficients (default 1)')
    p.add_argument('--min-baseline-frac', type=float, default=0.5,
                   help='Stop iterating once events would occupy more than '
                        '1 minus this fraction of the *non-zero* coefficients')
    p.add_argument('--plot-start', type=float, default=None,
                   help='Window start in seconds (default: start of standard_control)')
    p.add_argument('--plot-duration', type=float, default=5.0,
                   help='Window length in seconds (native rate)')
    p.add_argument('--out-dir', default='../../results/slap2/oasis',
                   help='Directory for the figure and per-ROI CSV')
    p.add_argument('--tau-green', type=float, default=TAU_D['green'],
                   help='iGluSnFR4f decay time constant, seconds')
    p.add_argument('--tau-red', type=float, default=TAU_D['red'],
                   help='RCaMP3 decay time constant, seconds')
    args = p.parse_args()
    main(args.nwb_path, args.dmd, args.n_rois, args.n_sd, args.min_baseline_frac,
         args.plot_start, args.plot_duration, args.out_dir,
         args.tau_green, args.tau_red)
