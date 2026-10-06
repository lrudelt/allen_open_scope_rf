"""Data-release RF panels for one example session of each modality.

Sessions match ``compare_zebra_siegle_rfs.ipynb``:

* ecephys 830794
* mesoscope 832700
* SLAP2 829704, green and red kept separate (different indicators)

For each session, the top 3 units are plotted twice: once ranked by Zebra
peak correlation (``abs_max_value``, the notebook's Selection A), and once
ranked by local-Gabor significance (lowest ``p_value``, ties broken by peak
|z|, the notebook's Selection B). Each ranking writes two PNGs with the same
unit order: the three Zebra maps, then the three Gabor maps.

Ephys units are restricted to visual cortex layer 2/3 using the saved
extremum-channel table from ``export_ephys_unit_areas.py``
(``results/ephys/<session>/unit_areas.csv``). Mesoscope units are restricted to
plane ``VISp_0``. SLAP2 ROIs are all kept: that recording is already V1 layer 2/3.

Zebra units are then limited to those that pass the null test after
Benjamini–Hochberg correction at q = 0.05, pooled over the session. The null
files store the surrogate maxima; the pass/fail used here is the Gumbel
tail probability and BH q-value from ``explore_null_results_*.ipynb``. SLAP2
nulls are dF/F, so the SLAP2 Zebra maps are the dF/F results.

A paired figure (Gabor above, Zebra trial 0 below, one shared colorbar per
row) is written for ephys VISp layer 2/3, mesoscope VISp_0, and SLAP2 green.

    cd code/scripts
    python plot_data_release_panels.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(str(Path(__file__).resolve().parents[1]))  # code/
sys.path.append(str(Path(__file__).resolve().parent))

from analysis.gaussian_envelope import fit_gaussian_envelope
from optimize_waven_parameters import load_rf_results
from waven_settings import analysis_coverage

# Repo-level results/ (not code/results). Layout is documented in results/README.md.
RESULTS = Path(__file__).resolve().parents[2] / "results"
RF_ROOT = RESULTS
PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"
FILE_PREFIX = "data_release"

N_TOP = 3
TRIAL, PHASE = 0, 0
GABOR_SIGNAL = "events"
# Same rule as explore_null_results_*.ipynb: BH over the whole session.
Q_LEVEL = 0.05
FONTSIZE = 14
TITLE_FONTSIZE = FONTSIZE + 1
PAIRED_TITLE_SIZE = 22
PAIRED_LABEL_SIZE = 27
PAIRED_TICK_SIZE = 22.5
_EULER = 0.5772156649015329

xM, xm, yM, ym = analysis_coverage
WAVEN_EXTENT = (xM, xm, ym, yM)

# One example session per modality, as in the notebook's SESSIONS / UNIT_LABEL.
SESSIONS = {
    "ecephys 830794": (
        "sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys",
        None,
    ),
    "mesoscope 832700": (
        "sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys",
        None,
    ),
    "slap2 829704 green": (
        "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys",
        "green",
    ),
    "slap2 829704 red": (
        "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys",
        "red",
    ),
}

# Saved by export_ephys_unit_areas.py. Join on unit_id; do not stream the NWB.
EPHYS_SESSION = "sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys"
UNIT_AREAS_CSV = RESULTS / "ephys" / EPHYS_SESSION / "unit_areas.csv"

WAVEN_COLS = [
    "unit_id", "abs_max_value", "rf_map", "delay", "duration",
    "x_deg", "y_deg", "theta_rad", "theta_deg", "sigma_deg", "frequency",
]
GABOR_COLS = [
    "unit_id", "p_value", "p_value_gumbel", "p_value_bh", "z_max", "ori_idx", "ori_rad", "delay",
    "duration", "rf_map", "rf_maps_all", "p_all",
]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def in_filter(group: str, keep: str | None) -> bool:
    return keep is None or group.split("__")[-1] == keep


def canonical_unit_id(uid) -> str:
    m = re.match(r"^.*_roi(\d+)$", str(uid))
    return m.group(1) if m else str(uid)


def _modality_session_dir(kind: str, session: str, signal: str | None = None) -> Path | None:
    """Directory holding one session's reduced results.

    Ophys has ``dff`` and ``events`` side by side. ``signal=None`` keeps
    ``GABOR_SIGNAL``. SLAP2 dF/F sits in the session directory itself when
    there is no ``dff`` subdirectory.
    """
    chosen = GABOR_SIGNAL if signal is None else signal
    base = RF_ROOT / kind
    for modality in ("ephys", "meso", "mesoscope", "slap2"):
        direct = base / modality / session
        if not direct.is_dir():
            continue
        if chosen == "dff":
            nested = direct / "dff"
            return nested if nested.is_dir() else direct
        signaled = direct / chosen
        if signaled.is_dir():
            return signaled
        return direct
    return None


def discover_waven(session: str, trial: int = TRIAL, signal: str | None = None) -> dict[str, Path]:
    out = {}
    d = _modality_session_dir("zebra/optimized", session, signal=signal)
    if d is None:
        return out
    for f in sorted(d.glob(f"optimized__{session}__*.h5")):
        if "all_values" in f.stem:
            continue
        tail = f.stem[len(f"optimized__{session}__"):]
        m = re.match(
            r"^(?P<group>.+?)(?:__trial_(?P<trial>\d+))?__phase_(?P<phase>\d+)$",
            tail,
        )
        if not m:
            continue
        if int(m.group("phase")) != PHASE:
            continue
        # SLAP2 filenames omit the trial. That pooled repeat is trial 0.
        named = m.group("trial")
        if named is None:
            if trial != 0:
                continue
        elif int(named) != trial:
            continue
        out[m.group("group")] = f
    return out


def discover_gabor(session: str) -> dict[str, Path]:
    out = {}
    d = _modality_session_dir("gabors", session)
    if d is None:
        return out
    for f in sorted(d.glob(f"{session}__rf-spike-count__*__optimized.h5")):
        if "__trial_" in f.stem:
            continue
        out[f.stem.split("__")[-2]] = f

    channel_prefix = f"rf-{GABOR_SIGNAL}-"
    for f in sorted(d.glob(f"{session}__*__optimized.h5")):
        if "__trial_" in f.stem:
            continue
        parts = f.stem.split("__")
        if len(parts) < 4:
            continue
        tag, group = parts[-3], parts[-2]
        if tag.startswith(channel_prefix):
            group = f"{group}__{tag[len(channel_prefix):]}"
        out[group] = f
    return out


def load_waven(path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=WAVEN_COLS)
    df = load_rf_results(path)
    if "rf_map" not in df.columns:
        print(f"  {Path(path).name}: no rf_maps stored — skipping")
        return pd.DataFrame(columns=WAVEN_COLS)
    df = df[[c for c in WAVEN_COLS if c in df.columns]].copy()
    df["unit_id"] = df["unit_id"].map(canonical_unit_id)
    return df


def _extent_from_centres(x_pos, y_pos):
    dx = np.median(np.diff(x_pos)) if len(x_pos) > 1 else 1.0
    dy = np.median(np.diff(y_pos)) if len(y_pos) > 1 else 1.0
    return (
        x_pos[0] - dx / 2, x_pos[-1] + dx / 2,
        y_pos[0] - dy / 2, y_pos[-1] + dy / 2,
    )


def load_gabor(path) -> pd.DataFrame:
    """Best orientation per unit: lowest p, then largest peak |z| on a tie."""
    if path is None or not Path(path).exists():
        empty = pd.DataFrame(columns=GABOR_COLS)
        empty.attrs["extent"] = (-45, 45, -45, 45)
        return empty

    with h5py.File(path, "r") as hf:
        unit_names = hf["unit_names"][:].astype(str)
        p_values = hf["p_value"][:]
        z_scores = hf["z_score_response"][:]
        orientations = hf["orientations"][:]
        x_pos, y_pos = hf["x_positions"][:], hf["y_positions"][:]
        if "p_value_bh" in hf and hf["p_value_bh"].shape == (len(unit_names),):
            p_value_bh = np.asarray(hf["p_value_bh"][:], dtype=float)
        else:
            p_value_bh = np.full(len(unit_names), np.nan)
        if "p_value_gumbel" in hf and hf["p_value_gumbel"].shape == (len(unit_names),):
            p_value_gumbel = np.asarray(hf["p_value_gumbel"][:], dtype=float)
        else:
            p_value_gumbel = np.full(len(unit_names), np.nan)

    z_peak = np.nanmax(np.abs(z_scores), axis=(2, 3))
    at_min = p_values <= p_values.min(axis=1, keepdims=True) + 1e-12
    best_ori = np.argmax(np.where(at_min, z_peak, -np.inf), axis=1)
    u = np.arange(len(unit_names))

    df = pd.DataFrame({
        "unit_id": [canonical_unit_id(x) for x in unit_names],
        "p_value": p_values[u, best_ori],
        "p_value_gumbel": p_value_gumbel,
        "p_value_bh": p_value_bh,
        "ori_idx": best_ori,
        "ori_rad": orientations[best_ori],
    })
    df["rf_map"] = [z_scores[i, best_ori[i]] for i in u]
    df["z_max"] = z_peak[u, best_ori]
    with np.errstate(invalid="ignore"):
        df["z_max_all"] = np.nanmax(z_peak, axis=1)
    df.attrs["extent"] = _extent_from_centres(x_pos, y_pos)
    df.attrs["x_positions"] = np.asarray(x_pos, dtype=float)
    df.attrs["y_positions"] = np.asarray(y_pos, dtype=float)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    return df


def load_session(session: str, keep: str | None, signal: str | None = None) -> pd.DataFrame:
    waven_files, gabor_files = discover_waven(session, signal=signal), discover_gabor(session)
    frames, extent, orientations = [], None, ()
    x_positions = y_positions = None
    for group in sorted(
        g for g in set(waven_files) | set(gabor_files) if in_filter(g, keep)
    ):
        w = load_waven(waven_files.get(group))
        s = load_gabor(gabor_files.get(group))
        if extent is None and len(s):
            extent = s.attrs["extent"]
            orientations = s.attrs.get("orientations", ())
            x_positions = s.attrs.get("x_positions")
            y_positions = s.attrs.get("y_positions")
        merged = w.merge(s, on="unit_id", how="outer", suffixes=("_zebra", "_gabor"))
        merged.insert(0, "group", group)
        if len(merged):
            frames.append(merged)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["group", "unit_id", "abs_max_value", "rf_map_zebra", "p_value", "rf_map_gabor"]
    )
    df.attrs["extent"] = extent or (-45, 45, -45, 45)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    if x_positions is not None:
        df.attrs["x_positions"] = np.asarray(x_positions, dtype=float)
        df.attrs["y_positions"] = np.asarray(y_positions, dtype=float)
    return df


def load_ephys_unit_areas() -> pd.DataFrame:
    """Unit id, probe, area, and layer from the saved table."""
    if not UNIT_AREAS_CSV.is_file():
        raise FileNotFoundError(
            f"No unit-area table at {UNIT_AREAS_CSV}. "
            "Run export_ephys_unit_areas.py once to build it."
        )
    anatomy = pd.read_csv(UNIT_AREAS_CSV, dtype=str, keep_default_na=False)
    anatomy["unit_id"] = anatomy["unit_id"].map(_text)
    anatomy["probe"] = anatomy["probe"].map(_text)
    return anatomy


def ephys_vis_layer23(df: pd.DataFrame) -> pd.DataFrame:
    """Keep visual-cortex layer 2/3 units, labeled from the saved area table."""
    anatomy = load_ephys_unit_areas()
    print(f"Unit areas from {UNIT_AREAS_CSV.name}")
    n_match = int(df["unit_id"].isin(set(anatomy["unit_id"])).sum())
    print(f"  unit-id overlap with saved areas: {n_match} / {len(df)}")
    keep = (
        anatomy["area"].astype(str).str.startswith("VIS")
        & (anatomy["layer"].astype(str) == "2/3")
    )
    anatomy = anatomy.loc[keep, ["probe", "unit_id", "area", "layer", "structure"]]
    print(f"  visual cortex layer 2/3: {len(anatomy)} units")
    out = df.merge(anatomy.drop(columns=["probe"]), on="unit_id", how="inner")
    print(f"  visual cortex layer 2/3 with a zebra map: {out['rf_map_zebra'].map(lambda m: isinstance(m, np.ndarray)).sum() if len(out) else 0}")
    return out


def restrict_population(label: str, df: pd.DataFrame) -> pd.DataFrame:
    if label.startswith("ecephys"):
        return ephys_vis_layer23(df)
    if label.startswith("mesoscope"):
        out = df[df["group"] == "VISp_0"].copy()
        print(f"  VISp_0: {len(out)} paired units")
        return out
    print(f"  all SLAP2 ROIs (V1 layer 2/3): {len(df)} paired units")
    return df


def _null_h5_files(session: str) -> list[tuple[str, Path, str | None]]:
    """Trial 0, phase 0 null files as ``(group, path, signal)``.

    ``signal`` is ``dff`` or ``events`` for ophys and ``None`` for ephys.
    The ``validate`` gate runs are not nulls.
    """
    found = []
    root = RESULTS / "zebra" / "nullresults"
    for base in root.glob(f"**/{session}"):
        if not base.is_dir() or "validate" in base.parts:
            continue
        for path in sorted(base.rglob("null__*.h5")):
            rel = path.relative_to(base).parts
            if "phase_0" not in rel:
                continue
            if any(part.startswith("trial_") and part != "trial_0" for part in rel):
                continue
            signal = rel[0] if rel and rel[0] in ("dff", "events") else None
            if signal is None:
                group = rel[0]
            elif len(rel) >= 3 and rel[1].startswith("DMD"):
                group = f"{rel[1]}__{rel[2]}"
            else:
                group = rel[1]
            found.append((group, path, signal))
    return found


def tested_zebra_signal(session: str) -> str | None:
    """Signal the saved null was computed on. Ephys has no signal level."""
    signals = {signal for _, _, signal in _null_h5_files(session) if signal}
    if not signals:
        return None
    if "events" in signals and GABOR_SIGNAL == "events":
        return "events" if "events" in signals else next(iter(signals))
    return next(iter(signals))


def _gumbel_sf(x, mu, beta):
    beta = np.maximum(np.asarray(beta, dtype=float), 1e-12)
    return -np.expm1(-np.exp(-(np.asarray(x, dtype=float) - mu) / beta))


def _bh_fdr(p) -> np.ndarray:
    """Benjamini–Hochberg q-values, as in the null-result notebooks."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order][::-1]
    q = np.minimum.accumulate((n / np.arange(n, 0, -1)) * ranked)[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _unit_null_rows(group: str, path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as hf:
        if int(hf.attrs.get("n_surrogates", 1)) < 1:
            raise ValueError(f"{path} has no surrogates")
        family = hf["family_max"]
        observed = np.max(family[:, :, 0], axis=1)
        surrogates = np.max(family[:, :, 1:], axis=1)
        unit_ids = [_text(value) for value in hf["unit_ids"][:]]
    beta = surrogates.std(axis=1, ddof=1) * np.sqrt(6) / np.pi
    mu = surrogates.mean(axis=1) - _EULER * beta
    p_values = _gumbel_sf(observed, mu, beta)
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    return pd.DataFrame({
        "group": group,
        "unit_id": [canonical_unit_id(unit_id) for unit_id in unit_ids],
        "p_null": p_values,
    })


_Q_CACHE: dict[str, pd.DataFrame] = {}


def zebra_q_table(session: str) -> pd.DataFrame:
    """Every tested unit, with its Gumbel p-value and session-wide BH q-value."""
    if session in _Q_CACHE:
        return _Q_CACHE[session]
    files = _null_h5_files(session)
    if not files:
        raise FileNotFoundError(f"No trial-0 phase-0 null files for {session}")
    signal = tested_zebra_signal(session)
    parts = [_unit_null_rows(group, path) for group, path, _ in files]
    table = pd.concat(parts, ignore_index=True)
    table["q"] = _bh_fdr(table["p_null"].to_numpy())
    table["significant"] = table["q"] <= Q_LEVEL
    print(
        f"  null test ({signal or 'spikes'}), BH q ≤ {Q_LEVEL}: "
        f"{int(table['significant'].sum())} / {len(table)} tested units"
    )
    _Q_CACHE[session] = table
    return table


def zebra_significant_units(session: str) -> pd.DataFrame:
    """Units with BH q ≤ ``Q_LEVEL``. The family is every tested unit in the session."""
    table = zebra_q_table(session)
    return table.loc[table["significant"], ["group", "unit_id", "p_null", "q"]].copy()


def keep_significant_zebra(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """Keep rows whose (group, unit) passes the session-wide Zebra null test."""
    significant = zebra_significant_units(session)
    out = df.merge(significant, on=["group", "unit_id"], how="inner")
    out.attrs = dict(df.attrs)
    print(f"  plotted units passing the null test: {len(out)} / {len(df)}")
    return out


def keep_zebra_q(df: pd.DataFrame, session: str, channel: str | None = None) -> pd.DataFrame:
    """Keep rows with corrected Zebra q ≤ ``Q_LEVEL``.

    ``channel`` recomputes BH on that channel alone. SLAP2 green does not
    include the red channel in the family.
    """
    table = zebra_q_table(session)
    if channel is not None:
        table = table[table["group"].map(lambda group: str(group).split("__")[-1] == channel)].copy()
        table["q"] = _bh_fdr(table["p_null"].to_numpy())
        print(
            f"  Zebra BH within {channel}: "
            f"{int((table['q'] <= Q_LEVEL).sum())} / {len(table)}"
        )
    passed = table.loc[table["q"] <= Q_LEVEL, ["group", "unit_id"]]
    out = df.merge(passed, on=["group", "unit_id"], how="inner")
    out.attrs = dict(df.attrs)
    print(f"  Zebra q ≤ {Q_LEVEL}: {len(out)} / {len(df)} in the plotted groups")
    return out


def rank_units(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Top units. ``zebra`` uses |r|; ``gabor`` uses p-value then |z|."""
    if by == "zebra":
        both = df.dropna(subset=["abs_max_value"])
        both = both[both["rf_map_zebra"].map(lambda m: isinstance(m, np.ndarray))]
        ordered = both.sort_values("abs_max_value", ascending=False)
    elif by == "gabor":
        if "p_value" not in df.columns or "z_max" not in df.columns:
            return df.iloc[0:0]
        both = df.dropna(subset=["p_value", "z_max"])
        if not len(both) or "rf_map_gabor" not in both.columns:
            return both.iloc[0:0]
        both = both[both["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray))]
        if not len(both):
            return both
        ordered = both.sort_values(["p_value", "z_max"], ascending=[True, False])
    else:
        raise ValueError(by)
    return ordered.head(N_TOP)


def _show(fig, ax, image, extent, origin, cbar_label, xlabel):
    v = np.nanmax(np.abs(image))
    v = v if np.isfinite(v) and v > 0 else 1.0
    im = ax.imshow(
        image.T, cmap="coolwarm", vmin=-v, vmax=v, extent=extent,
        origin=origin, aspect="equal", interpolation="nearest",
    )
    ax.axhline(0, color=".4", lw=.4)
    ax.axvline(0, color=".4", lw=.4)
    ax.set_ylabel("Elevation (deg)", fontsize=FONTSIZE)
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_xticks(np.arange(-60, 80, 20))
    ax.set_yticks(np.arange(-40, 60, 20))
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    cbar.set_label(cbar_label, fontsize=FONTSIZE)


def _panel_id(unit_id) -> str:
    """Break a long unit id so neighbouring panel titles do not run together."""
    text = str(unit_id)
    if len(text) <= 22:
        return text
    parts = text.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:2]) + "-\n" + "-".join(parts[2:])
    return text[: len(text) // 2] + "\n" + text[len(text) // 2 :]


def plot_maps(rows: pd.DataFrame, kind: str, extent, orientations, suptitle: str) -> plt.Figure:
    """One row of maps, unit order left to right matching ``rows``."""
    n = len(rows)
    fig, axs = plt.subplots(1, n, figsize=(5.0 * n, 5.0), squeeze=False, sharex=True, sharey=True)
    zebra = kind == "zebra"
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axs[0, i]
        where = row["structure"] if "structure" in row and pd.notna(row["structure"]) else row["group"]
        uid = _panel_id(row["unit_id"])
        if zebra:
            _show(fig, ax, row["rf_map_zebra"], WAVEN_EXTENT, "upper", "Correlation", True)
            for edge in (-40, 40):
                ax.axvline(edge, color=".75", lw=.6, zorder=1)
                ax.axhline(edge, color=".75", lw=.6, zorder=1)
            ax.set_title(
                f"{where}\n{uid}\n|r|max = {row['abs_max_value']:.3f}",
                fontsize=TITLE_FONTSIZE,
            )
        else:
            _show(fig, ax, row["rf_map_gabor"], extent, "lower", "response (z-scored)", True)
            ori = ""
            if pd.notna(row.get("ori_idx")) and len(orientations):
                ori = f", θ = {np.rad2deg(orientations[int(row['ori_idx'])]):.0f}°"
            ax.set_title(
                f"{where}\n{uid}{ori}\n"
                f"p = {row['p_value']:.4f}, |z|max = {row['z_max']:.2f}",
                fontsize=TITLE_FONTSIZE,
            )
    # Shared axes would otherwise crop to whichever panel was drawn last.
    if zebra:
        x0, x1, y0, y1 = WAVEN_EXTENT
    else:
        x0, x1, y0, y1 = extent
    axs[0, 0].set_xlim(x0, x1)
    axs[0, 0].set_ylim(y0, y1)
    # Leave the top strip for the figure title; tight_layout otherwise
    # places panel titles in the same band.
    fig.suptitle(suptitle, fontsize=TITLE_FONTSIZE + 2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"{FILE_PREFIX}__{stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def main() -> None:
    print(f"RESULTS = {RESULTS}")
    if not (RESULTS / "zebra" / "optimized").is_dir():
        raise FileNotFoundError(f"Zebra results not found at {RESULTS / 'zebra' / 'optimized'}")

    for label, (session, keep) in SESSIONS.items():
        print(f"\n=== {label} ===")
        signal = tested_zebra_signal(session)
        loaded = load_session(session, keep, signal=signal)
        extent = loaded.attrs.get("extent", (-45, 45, -45, 45))
        orientations = loaded.attrs.get("orientations", ())
        try:
            df = restrict_population(label, loaded)
            df = keep_significant_zebra(df, session)
        except Exception as exc:
            print(f"  could not apply the area or null-test filter ({exc}); skipping {label}")
            continue
        for by, note in (("zebra", "ranked-by-zebra"), ("gabor", "ranked-by-local-gabors")):
            top = rank_units(df, by)
            if not len(top):
                print(f"  no paired units to rank by {by}")
                continue
            print(f"  top {len(top)} by {by}:")
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                print(
                    f"    {rank}. {row.group} / {row.unit_id}  "
                    f"|r|={row.abs_max_value:.3f}  p={row.p_value:.4f}  |z|={row.z_max:.2f}"
                )
            stem = f"{slug(label)}__{note}"
            rank_title = (
                "significant Zebra, ranked by |r|"
                if by == "zebra"
                else "significant Zebra, ranked by local Gabors"
            )
            has_zebra = top["rf_map_zebra"].map(lambda m: isinstance(m, np.ndarray)).any()
            has_gabor = (
                "rf_map_gabor" in top.columns
                and top["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray)).any()
            )
            if has_zebra:
                fig_z = plot_maps(top, "zebra", extent, orientations, f"{label} — {rank_title}")
                save_figure(fig_z, f"{stem}__zebra-rfs")
            if has_gabor:
                fig_g = plot_maps(top, "gabor", extent, orientations, f"{label} — {rank_title}")
                save_figure(fig_g, f"{stem}__gabor-rfs")
            elif by == "gabor":
                print("  no Gabor maps for these units (unit ids do not match the zebra/NWB set)")


# %% Zebra summary tables (code/results/zebra), not the missing RF maps
ZEBRA_ROOT = RESULTS / "zebra" / "optimized"
CORR_MIN = 0.15
LOCATION_CORR_MIN = 0.15

SUMMARY_SESSIONS = {
    "ecephys": EPHYS_SESSION,
    "mesoscope": "sub-832700_ses-multiplane-ophys-832700-2026-01-24-12-06-12_ophys",
    "slap2": "sub-829704_ses-829704-2025-12-18-10-57-36_image+ophys",
}

GROUP_COLORS = {
    "VISp": "#2166ac",
    "VISl": "#b2182b",
    "V1": "#2166ac",
    "LM": "#b2182b",
    "basal (DMD1)": "#2166ac",
    "apical (DMD2)": "#b2182b",
    "basal": "#2166ac",
    "apical": "#b2182b",
    "DMD1": "#2166ac",
    "DMD2": "#b2182b",
}


def load_zebra_summaries(session: str, signal: str | None = None) -> pd.DataFrame:
    """Best-filter summary per unit from trial 0, phase 0 CSVs."""
    d = _modality_session_dir("zebra/optimized", session, signal=signal)
    frames = []
    if d is None:
        raise FileNotFoundError(f"no zebra summary directory for {session}")
    for path in sorted(d.glob(f"optimized__{session}__*.csv")):
        if "all_values" in path.name:
            continue
        tail = path.stem[len(f"optimized__{session}__"):]
        # Trial 0 and phase 0 only. SLAP2 filenames omit the trial field.
        match = re.match(
            r"^(?P<group>.+?)(?:__trial_(?P<trial>\d+))?__phase_(?P<phase>\d+)$",
            tail,
        )
        if not match or match.group("phase") != "0":
            continue
        if match.group("trial") not in (None, "0"):
            continue
        frame = pd.read_csv(path)
        frame["group"] = match.group("group")
        frame["unit_id"] = frame["unit_id"].map(canonical_unit_id)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no zebra summary CSVs in {d}")
    return pd.concat(frames, ignore_index=True)


def label_summary_groups(modality: str, df: pd.DataFrame) -> pd.DataFrame:
    """VISp vs VISl, or SLAP2 basal (DMD1 green) vs apical (DMD2 green)."""
    if modality == "ecephys":
        anatomy = load_ephys_unit_areas()
        print(f"Unit areas from {UNIT_AREAS_CSV.name}")
        n_match = int(df["unit_id"].isin(set(anatomy["unit_id"])).sum())
        print(f"  unit-id overlap with saved areas: {n_match} / {len(df)}")
        if n_match == 0:
            print(f"  summary id sample: {df['unit_id'].head(3).tolist()}")
            print(f"  NWB id sample: {anatomy['unit_id'].head(3).tolist()}")
            return df.iloc[0:0].assign(structure=pd.Series(dtype=str))
        # Summaries have no area. Walk every probe, match that probe's unit
        # ids, and keep VISp / VISl from the updated extremum-channel labels.
        pieces = []
        probes = sorted(set(anatomy["probe"]) | set(df["group"]))
        for probe in probes:
            summary = df[df["group"] == probe]
            labeled = anatomy[anatomy["probe"] == probe]
            vis = labeled[labeled["area"].isin(["VISp", "VISl"])]
            matched = summary.merge(
                vis[["unit_id", "area"]], on="unit_id", how="inner"
            )
            print(
                f"  {probe}: {len(summary)} summary units, "
                f"{len(vis)} VISp/VISl in NWB, {len(matched)} matched"
            )
            if len(matched):
                pieces.append(matched)
        if not pieces:
            print(f"  summary id sample: {df['unit_id'].head(3).tolist()}")
            print(f"  NWB id sample: {anatomy['unit_id'].head(3).tolist()}")
            return df.iloc[0:0].assign(structure=pd.Series(dtype=str))
        out = pd.concat(pieces, ignore_index=True)
        out["structure"] = out["area"]
        return out

    if modality == "mesoscope":
        out = df[df["group"].str.match(r"VIS[pl]")].copy()
        out["structure"] = np.where(out["group"].str.startswith("VISp"), "VISp", "VISl")
        return out

    out = df[df["group"].isin(["DMD1__green", "DMD2__green"])].copy()
    out["structure"] = np.where(
        out["group"] == "DMD1__green", "basal (DMD1)", "apical (DMD2)"
    )
    return out


def prepare_summaries() -> dict[str, pd.DataFrame]:
    tables = {}
    for modality, session in SUMMARY_SESSIONS.items():
        signal = tested_zebra_signal(session)
        labeled = label_summary_groups(modality, load_zebra_summaries(session, signal=signal))
        channel = "green" if modality == "slap2" else None
        labeled = keep_zebra_q(labeled, session, channel=channel)
        print(f"{modality}: {len(labeled)} units with Zebra q ≤ {Q_LEVEL}")
        tables[modality] = labeled
    return tables


def prepare_gabor_peak_z() -> dict[str, pd.DataFrame]:
    """Peak |z| for units with Gabor q ≤ 0.05, labeled V1/LM or DMD1/DMD2.

    SLAP2 uses the green channel. The peak is the largest |z| over positions
    and orientations.
    """
    rename = {
        "VISp": "V1",
        "VISl": "LM",
        "basal (DMD1)": "DMD1",
        "apical (DMD2)": "DMD2",
    }
    tables = {}
    for modality, session in SUMMARY_SESSIONS.items():
        keep = "green" if modality == "slap2" else None
        frames = []
        for group, path in discover_gabor(session).items():
            if not in_filter(group, keep):
                continue
            with h5py.File(path, "r") as hf:
                z_scores = hf["z_score_response"][:]
                with np.errstate(invalid="ignore"):
                    peak = np.nanmax(np.abs(z_scores), axis=(1, 2, 3))
                names = [_text(value) for value in hf["unit_names"][:]]
                if "p_value_bh" in hf and hf["p_value_bh"].shape == (len(names),):
                    q_gabor = np.asarray(hf["p_value_bh"][:], dtype=float)
                else:
                    q_gabor = np.full(len(names), np.nan)
            frames.append(pd.DataFrame({
                "group": group,
                "unit_id": [canonical_unit_id(name) for name in names],
                "z_max": peak,
                "p_value_bh": q_gabor,
            }))
        if not frames:
            print(f"{modality}: no Gabor maps")
            continue
        labeled = label_summary_groups(modality, pd.concat(frames, ignore_index=True))
        if not len(labeled):
            print(f"{modality}: no Gabor units in the plotted areas")
            continue
        labeled["structure"] = labeled["structure"].replace(rename)
        n_area = len(labeled)
        labeled = labeled.loc[labeled["p_value_bh"] <= Q_LEVEL].dropna(subset=["z_max"])
        print(f"{modality}: {len(labeled)} / {n_area} Gabor units with q ≤ {Q_LEVEL}")
        tables[modality] = labeled
    return tables


STRUCTURE_ORDER = (
    "VISp", "VISl", "V1", "LM",
    "basal", "apical", "basal (DMD1)", "apical (DMD2)", "DMD1", "DMD2",
)
SUMMARY_FONT = 13
# Summary-figure text is twice the original size. Tick labels are 1.5×.
SUMMARY_LABEL = 26
SUMMARY_TICK = 20
PERCENTILES = (25, 50, 75)


def _structure_order(structures) -> list[str]:
    """VISp left of VISl. Other groups keep their listed order."""
    present = set(structures)
    return [name for name in STRUCTURE_ORDER if name in present]


def _summary_ci(values: np.ndarray, stat: str = "median", n_boot: int = 2000) -> tuple[float, float, float]:
    """Point estimate and bootstrap 95% CI. ``stat`` is ``median`` or ``mean``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    reduce = np.median if stat == "median" else np.mean
    center = float(reduce(values))
    if len(values) == 1:
        return center, center, center
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    lo, hi = np.percentile(reduce(draws, axis=1), [2.5, 97.5])
    return center, float(lo), float(hi)


def _location_limits(tables: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """One shared range for azimuth and elevation, and for every modality."""
    coords = []
    for df in tables.values():
        coords.append(df["x_deg"].to_numpy(dtype=float))
        coords.append(df["y_deg"].to_numpy(dtype=float))
    values = np.concatenate(coords)
    pad = 2.0
    return float(np.nanmin(values) - pad), float(np.nanmax(values) + pad)


def _star_label(p_value: float) -> str:
    if p_value > 0.05:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _median_permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided permutation p-value for a difference of medians."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    observed = abs(float(np.median(a) - np.median(b)))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(float(np.median(pooled[:n_a]) - np.median(pooled[n_a:])))
        count += diff >= observed - 1e-15
    return (count + 1) / (n_perm + 1)


def plot_rf_locations(tables: dict[str, pd.DataFrame]) -> None:
    """Scatter of Zebra RF centres for units that pass the null test."""
    lo, hi = _location_limits(tables)
    for modality, df in tables.items():
        kept = df
        structures = _structure_order(kept["structure"])
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        for structure in structures:
            sub = kept[kept["structure"] == structure]
            ax.scatter(
                sub["x_deg"], sub["y_deg"],
                s=18, alpha=0.75, linewidths=0,
                color=GROUP_COLORS[structure], label=f"{structure} (n={len(sub)})",
            )
        ax.axhline(0, color=".75", lw=.6)
        ax.axvline(0, color=".75", lw=.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ticks = np.arange(np.ceil(lo / 10) * 10, hi + 1e-6, 10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Azimuth (deg)", fontsize=SUMMARY_FONT)
        ax.set_ylabel("Elevation (deg)", fontsize=SUMMARY_FONT)
        ax.set_title(f"{modality} — Zebra RF location, q ≤ {Q_LEVEL}", fontsize=SUMMARY_FONT + 1)
        ax.tick_params(labelsize=SUMMARY_FONT)
        ax.legend(frameon=False, fontsize=SUMMARY_FONT - 1)
        fig.tight_layout()
        save_figure(fig, f"{modality}__zebra-rf-locations__significant")


def plot_grouped_values(
    tables: dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    stem: str,
    stat: str = "median",
    show_percentiles: bool = False,
    test_medians: bool = False,
    title: str = f"q ≤ {Q_LEVEL}",
) -> None:
    """Group summary with a bootstrap 95% CI, and one dot per unit."""
    for modality, df in tables.items():
        structures = _structure_order(df["structure"])
        fig, ax = plt.subplots(figsize=(3.6, 4.6))
        rng = np.random.default_rng(1)
        grouped = []
        for i, structure in enumerate(structures):
            values = df.loc[df["structure"] == structure, column].to_numpy(dtype=float)
            grouped.append(values)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), i) + jitter, values,
                s=12, alpha=0.35, linewidths=0, color=GROUP_COLORS[structure], zorder=2,
            )
            if show_percentiles:
                for percentile in PERCENTILES:
                    level = float(np.nanpercentile(values, percentile))
                    ax.plot(
                        [i - 0.18, i + 0.18], [level, level],
                        color="k", lw=1.6 if percentile == 50 else 0.8, zorder=4,
                    )
            if stat == "median" or not show_percentiles:
                center, lo, hi = _summary_ci(values, stat=stat)
                ax.errorbar(
                    i, center, yerr=[[center - lo], [hi - center]],
                    fmt="o", color="k", ms=6, capsize=4, zorder=5,
                )
        if test_medians and len(grouped) == 2:
            p_value = _median_permutation_p(grouped[0], grouped[1])
            print(f"  {modality} median permutation p = {p_value:.4g}")
            y_top = np.nanmax(np.concatenate(grouped))
            y_span = y_top - np.nanmin(np.concatenate(grouped))
            bracket = y_top + 0.06 * y_span
            ax.plot([0, 0, 1, 1], [bracket, bracket + 0.03 * y_span, bracket + 0.03 * y_span, bracket],
                    color="k", lw=0.8, clip_on=False)
            ax.text(0.5, bracket + 0.05 * y_span, _star_label(p_value),
                    ha="center", va="bottom", fontsize=SUMMARY_FONT - 1)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks(range(len(structures)))
        ax.set_xticklabels(structures, fontsize=SUMMARY_FONT)
        ax.set_ylabel(ylabel, fontsize=SUMMARY_FONT)
        ax.set_title(f"{modality} — {title}", fontsize=SUMMARY_FONT + 1)
        ax.tick_params(axis="y", labelsize=SUMMARY_FONT)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()
        save_figure(fig, f"{modality}__{stem}")


# Summary figures name the same comparison on both rows.
_SUMMARY_RENAME = {
    "VISp": "V1",
    "VISl": "LM",
    "basal (DMD1)": "basal",
    "apical (DMD2)": "apical",
}


def _fit_records(df: pd.DataFrame, map_col: str, mode: str, strength_col: str,
                 centres) -> tuple[pd.DataFrame, int]:
    """Gaussian centre and one-sigma ellipse area for each significant map."""
    rows = []
    failed = 0
    for row in df.itertuples(index=False):
        image = getattr(row, map_col)
        if not isinstance(image, np.ndarray):
            failed += 1
            continue
        x_pos, y_pos = centres(image.shape) if callable(centres) else centres
        fit = _fit_map(image, x_pos, y_pos, mode)
        sigmas = (fit["sigma_x"], fit["sigma_y"], fit["x0"], fit["y0"])
        if not fit["success"] or not np.isfinite(sigmas).all() or min(fit["sigma_x"], fit["sigma_y"]) <= 0:
            failed += 1
            continue
        rows.append({
            "structure": row.structure,
            "x0": float(fit["x0"]),
            "y0": float(fit["y0"]),
            "area": float(np.pi * fit["sigma_x"] * fit["sigma_y"]),
            "strength": float(getattr(row, strength_col)),
            "fit_r2": float(fit["r2"]),
        })
    fitted = pd.DataFrame(rows, columns=["structure", "x0", "y0", "area", "strength", "fit_r2"])
    return fitted, failed


def prepare_fit_summaries() -> dict[str, dict]:
    """Gaussian summaries for units with q ≤ 0.05 on that stimulus.

    Position is the fitted centre. Area is the one-sigma ellipse. Gabor is the
    signed z-map of the best orientation; Zebra is the absolute correlation.
    """
    out = {}
    for modality, session in SUMMARY_SESSIONS.items():
        keep = "green" if modality == "slap2" else None
        signal = tested_zebra_signal(session)
        print(f"\n=== fit summary {modality} ===")
        loaded = load_session(session, keep, signal=signal)
        extent = loaded.attrs.get("extent", (-45.0, 45.0, -45.0, 45.0))
        gabor_centres = (
            np.asarray(loaded.attrs["x_positions"], dtype=float),
            np.asarray(loaded.attrs["y_positions"], dtype=float),
        ) if "x_positions" in loaded.attrs else None
        labeled = label_summary_groups(modality, loaded)
        if not len(labeled):
            print(f"  {modality}: no units in the plotted areas")
            continue
        labeled["structure"] = labeled["structure"].replace(_SUMMARY_RENAME)
        zebra = keep_zebra_q(labeled, session, channel=keep)
        gabor = labeled.loc[labeled["p_value_bh"] <= Q_LEVEL].copy()
        print(f"  Gabor q ≤ {Q_LEVEL}: {len(gabor)}")
        if gabor_centres is None and len(gabor):
            sample = next(m for m in gabor["rf_map_gabor"] if isinstance(m, np.ndarray))
            from analysis.gaussian_envelope import patch_centres
            gabor_centres = (
                patch_centres(extent[:2], sample.shape[0]),
                patch_centres(extent[2:], sample.shape[1]),
            )
        gabor_fit, gabor_failed = _fit_records(
            gabor, "rf_map_gabor", "signed", "z_max_all", gabor_centres,
        )
        zebra_fit, zebra_failed = _fit_records(
            zebra, "rf_map_zebra", "abs", "abs_max_value", _zebra_centres,
        )
        print(
            f"  fits: Gabor {len(gabor_fit)} ({gabor_failed} failed), "
            f"Zebra {len(zebra_fit)} ({zebra_failed} failed)"
        )
        for name, fitted in (("Gabor", gabor_fit), ("Zebra", zebra_fit)):
            if len(fitted):
                print(f"  {name} median fit r2 {fitted['fit_r2'].median():.2f}")
        out[modality] = {"gabor": gabor_fit, "zebra": zebra_fit, "gabor_extent": extent}
    return out


def _scatter_legend_handles(df: pd.DataFrame):
    from matplotlib.lines import Line2D

    structures = _structure_order(df["structure"]) if len(df) else []
    return [
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=16,
            markerfacecolor=GROUP_COLORS[structure], markeredgecolor="none",
            label=f"{structure} (n={(df['structure'] == structure).sum()})",
        )
        for structure in structures
    ]


def _draw_position(ax, df: pd.DataFrame, extent, xlabel: bool) -> None:
    structures = _structure_order(df["structure"]) if len(df) else []
    for structure in structures:
        sub = df[df["structure"] == structure]
        ax.scatter(
            sub["x0"], sub["y0"],
            s=14 * 1.5 ** 2, alpha=0.55, linewidths=0,
            color=GROUP_COLORS[structure], zorder=2,
        )
    x0, x1, y0, y1 = (float(v) for v in extent)
    ax.axhline(0, color=".75", lw=.7, zorder=1)
    ax.axvline(0, color=".75", lw=.7, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(np.arange(-60, 80, 20))
    ax.set_yticks(np.arange(-40, 60, 20))
    ax.tick_params(labelsize=SUMMARY_TICK, length=6, width=1.1)
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=SUMMARY_LABEL)
    ax.set_ylabel("Elevation (deg)", fontsize=SUMMARY_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _place_scatter_legends(fig, axes, frames, fig_w: float, fig_h: float, center_in: float) -> None:
    """Put one legend above each scatter, sharing the same left edge.

    The pair is centered on the position panels, in the band the row titles
    used to occupy. A shared left edge keeps the group dots in one column.
    """
    legends = []
    for ax, df in zip(axes, frames):
        handles = _scatter_legend_handles(df)
        if not handles:
            legends.append(None)
            continue
        legends.append(ax.legend(
            handles=handles, frameon=False, fontsize=SUMMARY_LABEL - 4,
            loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0,
            handlelength=0.7, handletextpad=0.35, labelspacing=0.3, borderpad=0.15,
        ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        0.0 if leg is None else leg.get_window_extent(renderer).width / fig.dpi
        for leg in legends
    ]
    left = center_in - max(widths, default=0.0) / 2
    pad = 0.12
    for leg, ax in zip(legends, axes):
        if leg is None:
            continue
        top = ax.get_position().y1 * fig_h
        leg.set_bbox_to_anchor(
            (left / fig_w, (top + pad) / fig_h), transform=fig.transFigure,
        )


def _draw_comparison(ax, df: pd.DataFrame, column: str, ylabel: str, ymax: float | None = None) -> None:
    """Box plot of the two groups, with a bootstrap CI on the median.

    ``ymax`` fixes the top of the axis, and the top tick, at that value.
    Points above it are left outside the frame.
    """
    structures = _structure_order(df["structure"]) if len(df) else []
    grouped = []
    for structure in structures:
        values = df.loc[df["structure"] == structure, column].to_numpy(dtype=float)
        grouped.append(values[np.isfinite(values)])
    boxed = [(i, values) for i, values in enumerate(grouped) if len(values) >= 2]
    if boxed:
        bp = ax.boxplot(
            [values for _, values in boxed],
            positions=[i for i, _ in boxed],
            widths=0.55,
            patch_artist=True,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "k", "linewidth": 1.6},
            whiskerprops={"color": "k", "linewidth": 1.0},
            capprops={"color": "k", "linewidth": 1.0},
            flierprops={
                "marker": "o", "markersize": 3.5, "markerfacecolor": "0.35",
                "markeredgecolor": "none", "alpha": 0.55,
            },
        )
        for patch, (i, _) in zip(bp["boxes"], boxed):
            patch.set_facecolor(GROUP_COLORS[structures[i]])
            patch.set_edgecolor("k")
            patch.set_alpha(0.45)
            patch.set_linewidth(1.0)
    for i, values in enumerate(grouped):
        if len(values) == 1:
            ax.scatter(
                i, values[0], s=28, color=GROUP_COLORS[structures[i]],
                edgecolors="k", linewidths=0.6, zorder=4,
            )
        if len(values):
            center, lo, hi = _summary_ci(values, stat="median")
            ax.errorbar(
                i, center, yerr=[[max(center - lo, 0.0)], [max(hi - center, 0.0)]],
                fmt="none", ecolor="k", elinewidth=1.3, capsize=5, capthick=1.3, zorder=6,
            )
    if len(grouped) == 2 and all(len(values) for values in grouped):
        p_value = _median_permutation_p(grouped[0], grouped[1])
        print(f"    {ylabel}: median permutation p = {p_value:.4g}")
        pooled = np.concatenate(grouped)
        if ymax is None:
            y_top = float(np.nanmax(pooled))
            y_span = y_top - float(np.nanmin(pooled))
            if y_span <= 0:
                y_span = abs(y_top) if y_top else 1.0
            bracket = y_top + 0.08 * y_span
            rise = 0.035 * y_span
            y_lo = float(np.nanmin(pooled)) - 0.05 * y_span
            y_hi = bracket + 0.28 * y_span
        else:
            y_lo, y_hi = 0.0, float(ymax)
            y_span = y_hi - y_lo
            bracket = y_hi - 0.16 * y_span
            rise = 0.03 * y_span
        ax.plot(
            [0, 0, 1, 1], [bracket, bracket + rise, bracket + rise, bracket],
            color="k", lw=0.8, clip_on=ymax is not None,
        )
        ax.text(
            0.5, bracket + 0.045 * y_span, _star_label(p_value),
            ha="center", va="bottom", fontsize=SUMMARY_LABEL - 2,
            clip_on=ymax is not None,
        )
        ax.set_ylim(y_lo, y_hi)
        if ymax is not None:
            ax.set_yticks(np.arange(0, ymax + 1, 1000))
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(range(len(structures)))
    ax.set_xticklabels(structures, fontsize=SUMMARY_TICK)
    ax.set_ylabel(ylabel, fontsize=SUMMARY_LABEL)
    ax.tick_params(axis="y", labelsize=SUMMARY_TICK, length=6, width=1.1)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _ylabel_overhang(ax) -> float:
    """Inches from the left edge of the y-axis label to the axes spine."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spine = ax.get_window_extent(renderer).x0
    label = ax.yaxis.label.get_window_extent(renderer).x0
    return max((spine - label) / fig.dpi, 0.0)


def plot_fit_summaries(tables: dict[str, dict]) -> None:
    """One figure per modality: Gabor on top, Zebra below.

    Each row is the Gaussian centre, the one-sigma ellipse area, and the
    response strength. A degree has the same length in both position panels,
    and the Zebra window is the wider one. On the Zebra row the clear gap
    from the scatter to the area label equals the gap from the area panel to
    the correlation label. Area and correlation panels share that spacing in
    both rows, and share the vertical span of the position panel in their row.
    """
    inch = 0.034
    # Clear space between a panel edge and the next panel's y-axis label.
    clear = 0.42
    zebra_span_x, zebra_span_y = _span(WAVEN_EXTENT)
    zebra_w, zebra_h = zebra_span_x * inch, zebra_span_y * inch
    group_w, gap_row = 3.2, 1.65
    margin_l, margin_r = 1.7, 0.45
    margin_b, margin_t = 1.15, 1.95
    # Draft gaps are wide enough that the labels can be measured without overlap.
    draft_gap = 2.3
    draft_w = margin_l + zebra_w + draft_gap + group_w + draft_gap + group_w + margin_r
    fig_h = margin_b + zebra_h + gap_row + zebra_h + margin_t

    for modality, packed in tables.items():
        gabor, zebra = packed["gabor"], packed["zebra"]
        gabor_extent = packed["gabor_extent"]
        gabor_w, gabor_h = (span * inch for span in _span(gabor_extent))
        fig = plt.figure(figsize=(draft_w, fig_h))

        def place(x, y, w, h, fig_w):
            return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

        gabor_y = margin_b + zebra_h + gap_row + (zebra_h - gabor_h) / 2
        zebra_y = margin_b
        gabor_ax = place(
            margin_l + (zebra_w - gabor_w) / 2, gabor_y, gabor_w, gabor_h, draft_w,
        )
        zebra_ax = place(margin_l, zebra_y, zebra_w, zebra_h, draft_w)
        print(f"\n{modality}")
        _draw_position(gabor_ax, gabor, gabor_extent, xlabel=False)
        _draw_position(zebra_ax, zebra, WAVEN_EXTENT, xlabel=True)

        draft_area_x = margin_l + zebra_w + draft_gap
        draft_strength_x = draft_area_x + group_w + draft_gap
        rows = (
            (gabor, gabor_y, gabor_h, "Peak |z score|"),
            (zebra, zebra_y, zebra_h, "Peak |correlation|"),
        )
        area_axes, strength_axes = [], []
        for frame, y, height, strength_label in rows:
            area_ax = place(draft_area_x, y, group_w, height, draft_w)
            strength_ax = place(draft_strength_x, y, group_w, height, draft_w)
            _draw_comparison(area_ax, frame, "area", "RF area (deg²)", ymax=2000)
            _draw_comparison(strength_ax, frame, "strength", strength_label)
            area_axes.append(area_ax)
            strength_axes.append(strength_ax)

        # Zebra is the wider scatter, so its labels set the column positions.
        area_x = margin_l + zebra_w + clear + _ylabel_overhang(area_axes[1])
        strength_x = area_x + group_w + clear + _ylabel_overhang(strength_axes[1])
        fig_w = strength_x + group_w + margin_r
        fig.set_size_inches(fig_w, fig_h, forward=True)
        for ax, x, y, w, h in (
            (gabor_ax, margin_l + (zebra_w - gabor_w) / 2, gabor_y, gabor_w, gabor_h),
            (zebra_ax, margin_l, zebra_y, zebra_w, zebra_h),
        ):
            ax.set_position([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        for area_ax, strength_ax, (_, y, height, _) in zip(area_axes, strength_axes, rows):
            area_ax.set_position([area_x / fig_w, y / fig_h, group_w / fig_w, height / fig_h])
            strength_ax.set_position([
                strength_x / fig_w, y / fig_h, group_w / fig_w, height / fig_h,
            ])
        fig.suptitle(f"{modality} — q ≤ {Q_LEVEL}", fontsize=SUMMARY_LABEL + 4, y=0.985)
        _place_scatter_legends(
            fig, (gabor_ax, zebra_ax), (gabor, zebra),
            fig_w, fig_h, margin_l + zebra_w / 2,
        )
        save_figure(fig, f"{modality}__rf-summary")


PAIRED_PANELS = ("ecephys 830794", "mesoscope 832700", "slap2 829704 green")


def _lookup(df: pd.DataFrame, group: str, unit_id: str):
    if df is None or not len(df) or "unit_id" not in df.columns:
        return None
    hit = df[(df["group"] == group) & (df["unit_id"].astype(str) == str(unit_id))]
    if not len(hit):
        return None
    return hit.iloc[0]


def _map_image(row, column):
    if row is None or column not in row.index:
        return None
    image = row[column]
    return image if isinstance(image, np.ndarray) else None


def _span(extent) -> tuple[float, float]:
    x0, x1, y0, y1 = (float(v) for v in extent)
    return abs(x1 - x0), abs(y1 - y0)


def _peak(image) -> float:
    if image is None:
        return 1.0
    peak = np.nanmax(np.abs(image))
    if not np.isfinite(peak) or peak <= 0:
        return 1.0
    return float(peak)


def _row_scale(maps) -> float:
    peaks = [_peak(image) for image in maps if image is not None]
    return max(peaks) if peaks else 1.0


def _paint_selection(ax, color: str) -> None:
    """Frame one map in the pie-chart color of its selection column."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(3.6)


def _draw_paired(ax, image, extent, origin, vmax, xlabel, ylabel, faint_zero=False):
    im = ax.imshow(
        image.T, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=extent,
        origin=origin, aspect="equal", interpolation="nearest",
    )
    zero = ".75" if faint_zero else ".4"
    ax.axhline(0, color=zero, lw=.7, zorder=1)
    ax.axvline(0, color=zero, lw=.7, zorder=1)
    ax.tick_params(labelsize=PAIRED_TICK_SIZE)
    ax.set_xticks(np.arange(-60, 80, 20))
    ax.set_yticks(np.arange(-40, 60, 20))
    if xlabel:
        ax.set_xlabel("Azimuth (deg)", fontsize=PAIRED_LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PAIRED_LABEL_SIZE)
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return im


def _zebra_centres(shape):
    """Pixel centres of a Zebra map, in the same degrees as the panel.

    The map is drawn with ``origin='upper'`` and ``WAVEN_EXTENT`` as the
    pixel edges, so index 0 of the elevation axis is the top edge. The stored
    ``x_deg`` / ``y_deg`` are the grid index times the step, half a pixel off
    these centres.
    """
    nx, ny = shape
    left, right, bottom, top = (float(v) for v in WAVEN_EXTENT)
    x = left + (np.arange(nx) + 0.5) * (right - left) / nx
    y = top - (np.arange(ny) + 0.5) * (top - bottom) / ny
    return x, y


def _draw_gaussian(ax, fit):
    """Mark the Gaussian centre and the one-sigma ellipse of the main axes."""
    from matplotlib.patches import Ellipse

    if not fit.get("success"):
        return False
    sx, sy = float(fit["sigma_x"]), float(fit["sigma_y"])
    x0, y0 = float(fit["x0"]), float(fit["y0"])
    if not np.isfinite([sx, sy, x0, y0, fit["theta"]]).all() or min(sx, sy) <= 0:
        return False
    # sigma_x lies along angle theta: see the rotation in gaussian2d.
    kw = dict(
        xy=(x0, y0), width=2 * sx, height=2 * sy,
        angle=np.degrees(float(fit["theta"])), fill=False, zorder=3,
    )
    ax.add_patch(Ellipse(edgecolor="black", linewidth=3.2, **kw))
    ax.add_patch(Ellipse(edgecolor="white", linewidth=1.5, **kw))
    ax.plot(
        x0, y0, marker="o", ms=7, mfc="white", mec="black", mew=1.3,
        linestyle="none", zorder=4,
    )
    return True


def _fit_map(image, x_pos, y_pos, mode):
    """``mode`` is ``signed`` for a Gabor z-map, or ``abs`` for a Zebra envelope."""
    values = np.abs(image) if mode == "abs" else image
    return fit_gaussian_envelope(values, x_pos, y_pos, nonnegative=mode == "abs")


def _gabor_centres(frame, extent, shape):
    x_pos = frame.attrs.get("x_positions")
    y_pos = frame.attrs.get("y_positions")
    if x_pos is None or len(x_pos) != shape[0] or len(y_pos) != shape[1]:
        from analysis.gaussian_envelope import patch_centres
        x_pos = patch_centres(extent[:2], shape[0])
        y_pos = patch_centres(extent[2:], shape[1])
    return np.asarray(x_pos, dtype=float), np.asarray(y_pos, dtype=float)


def _overlay_gaussian(ax, image, frame, extent, mode, unit=""):
    if image is None:
        return
    if mode == "signed":
        x_pos, y_pos = _gabor_centres(frame, extent, image.shape)
        kind = "Gabor z"
    else:
        x_pos, y_pos = _zebra_centres(image.shape)
        kind = "Zebra |r|"
    fit = _fit_map(image, x_pos, y_pos, mode)
    prefix = f"    {unit}: " if unit else "    "
    if _draw_gaussian(ax, fit):
        print(
            f"{prefix}{kind} centre ({fit['x0']:.1f}, {fit['y0']:.1f}) deg, "
            f"sigma {fit['sigma_x']:.1f} x {fit['sigma_y']:.1f}, fit r2 {fit['r2']:.2f}"
        )
    else:
        print(f"{prefix}{kind} Gaussian fit did not converge")


def _paired_colorbar(fig, im, cax, label):
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=PAIRED_TICK_SIZE)
    cbar.set_label(label, fontsize=PAIRED_LABEL_SIZE)
    return cbar


def _both_maps(df: pd.DataFrame) -> pd.DataFrame:
    if "rf_map_gabor" not in df.columns or "rf_map_zebra" not in df.columns:
        return df.iloc[0:0]
    keep = df["rf_map_gabor"].map(lambda m: isinstance(m, np.ndarray)) & df["rf_map_zebra"].map(
        lambda m: isinstance(m, np.ndarray)
    )
    out = df.loc[keep].copy()
    out.attrs = dict(df.attrs)
    return out


def primary_visual(label: str, df: pd.DataFrame) -> pd.DataFrame:
    """Primary visual cortex for the paired figure: VISp layer 2/3, or VISp_0."""
    if label.startswith("ecephys"):
        anatomy = load_ephys_unit_areas()
        visp = anatomy[(anatomy["area"] == "VISp") & (anatomy["layer"] == "2/3")]
        out = df.merge(
            visp[["unit_id", "area", "layer", "structure"]], on="unit_id", how="inner",
        )
        print(f"  VISp layer 2/3: {len(out)} units")
    elif label.startswith("mesoscope"):
        out = df[df["group"] == "VISp_0"].copy()
        print(f"  VISp_0: {len(out)} units")
    else:
        out = df
        print(f"  SLAP2 V1 layer 2/3: {len(out)} units")
    for key in ("extent", "orientations", "x_positions", "y_positions"):
        if key in df.attrs:
            out.attrs[key] = df.attrs[key]
    if "extent" not in out.attrs:
        out.attrs["extent"] = (-45, 45, -45, 45)
    return out


def plot_paired_panel(
    label, selected, frame, extent, criterion, overlay_gaussian=False, column_colors=None,
):
    """One column per unit: Gabor on top, Zebra trial 0 below.

    A degree has the same length on both stimuli. Zebra covers more azimuth,
    so it is wider and the Gabor map is centered on the same vertical meridian.
    Each row has one shared color scale and one colorbar. ``column_colors``
    paints both maps in a column with that selection color.
    """
    n = len(selected)
    az_g, el_g = _span(extent)
    az_z, el_z = _span(WAVEN_EXTENT)
    inch_per_deg = 0.038
    gabor_w, gabor_h = az_g * inch_per_deg, el_g * inch_per_deg
    zebra_w, zebra_h = az_z * inch_per_deg, el_z * inch_per_deg
    cbar_w, cbar_pad = 0.32, 0.28
    panel_gap, row_gap = 1.05, 0.95
    margin_l, margin_r = 2.55, 1.35
    margin_b, margin_t = 1.55, 0.95

    fig_w = margin_l + n * zebra_w + (n - 1) * panel_gap + cbar_pad + cbar_w + margin_r
    fig_h = margin_b + zebra_h + row_gap + gabor_h + margin_t
    fig = plt.figure(figsize=(fig_w, fig_h))

    def column_x(index: int) -> float:
        return margin_l + index * (zebra_w + panel_gap)

    rows = [_lookup(frame, picked.group, picked.unit_id) for picked in selected.itertuples(index=False)]
    gabor_maps = [_map_image(row, "rf_map_gabor") for row in rows]
    zebra_maps = [_map_image(row, "rf_map_zebra") for row in rows]
    gabor_vmax = _row_scale(gabor_maps)
    zebra_vmax = _row_scale(zebra_maps)
    gabor_im = zebra_im = None

    for col, (gabor_map, zebra_map) in enumerate(zip(gabor_maps, zebra_maps)):
        zx = column_x(col)
        zy = margin_b
        gx = zx + (zebra_w - gabor_w) / 2
        gy = margin_b + zebra_h + row_gap
        gabor_ax = fig.add_axes([gx / fig_w, gy / fig_h, gabor_w / fig_w, gabor_h / fig_h])
        zebra_ax = fig.add_axes([zx / fig_w, zy / fig_h, zebra_w / fig_w, zebra_h / fig_h])
        color = None if column_colors is None else column_colors[col]
        if gabor_map is not None:
            gabor_im = _draw_paired(
                gabor_ax, gabor_map, extent, "lower", gabor_vmax,
                xlabel=False, ylabel="Gabor\nElevation (deg)" if col == 0 else None,
            )
        elif color is None:
            gabor_ax.set_axis_off()
        else:
            _draw_paired(
                gabor_ax, np.full((2, 2), np.nan), extent, "lower", gabor_vmax,
                xlabel=False, ylabel="Gabor\nElevation (deg)" if col == 0 else None,
            )
        if zebra_map is not None:
            zebra_im = _draw_paired(
                zebra_ax, zebra_map, WAVEN_EXTENT, "upper", zebra_vmax,
                xlabel=True, ylabel="Zebra trial 0\nElevation (deg)" if col == 0 else None,
                faint_zero=True,
            )
        elif color is None:
            zebra_ax.set_axis_off()
        else:
            _draw_paired(
                zebra_ax, np.full((2, 2), np.nan), WAVEN_EXTENT, "upper", zebra_vmax,
                xlabel=True, ylabel="Zebra trial 0\nElevation (deg)" if col == 0 else None,
                faint_zero=True,
            )
        if color is not None:
            _paint_selection(gabor_ax, color)
            _paint_selection(zebra_ax, color)
        if overlay_gaussian and rows[col] is not None:
            unit = f"{rows[col].group} / {rows[col].unit_id}"
            # A fit is a position and a size only when that stimulus's RF passed.
            if bool(rows[col].get("pass_gabor", False)):
                _overlay_gaussian(gabor_ax, gabor_map, frame, extent, "signed", unit)
            if bool(rows[col].get("pass_zebra", False)):
                _overlay_gaussian(zebra_ax, zebra_map, frame, extent, "abs", unit)

    cbar_x = (column_x(n - 1) + zebra_w + cbar_pad) / fig_w
    if gabor_im is not None:
        gabor_cax = fig.add_axes([
            cbar_x, (margin_b + zebra_h + row_gap) / fig_h, cbar_w / fig_w, gabor_h / fig_h,
        ])
        _paired_colorbar(fig, gabor_im, gabor_cax, "response (z)")
    if zebra_im is not None:
        zebra_cax = fig.add_axes([
            cbar_x, margin_b / fig_h, cbar_w / fig_w, zebra_h / fig_h,
        ])
        _paired_colorbar(fig, zebra_im, zebra_cax, "Correlation")
    fig.suptitle(f"{label} — {criterion}", fontsize=PAIRED_TITLE_SIZE, y=0.98)
    return fig


def plot_paired_release() -> None:
    """Gabor above Zebra for primary visual cortex, among null-test passers."""
    populations = {
        "ecephys 830794": "VISp layer 2/3",
        "mesoscope 832700": "VISp_0",
        "slap2 829704 green": "V1 layer 2/3",
    }
    for label in PAIRED_PANELS:
        session, keep = SESSIONS[label]
        print(f"\n=== paired {label} ===")
        signal = tested_zebra_signal(session)
        loaded = load_session(session, keep, signal=signal)
        frame = keep_significant_zebra(primary_visual(label, loaded), session)
        paired = _both_maps(frame)
        extent = frame.attrs.get("extent") or (-45, 45, -45, 45)
        if not len(paired):
            print("  no significant units have both a Gabor map and a Zebra map")
            continue
        population = populations[label]
        for by, note in (("zebra", "ranked-by-zebra"), ("gabor", "ranked-by-local-gabors")):
            criterion = "ranked by Zebra" if by == "zebra" else "ranked by Gabor"
            pool = paired
            if by == "gabor" and "z_max" in pool.columns:
                pool = pool[pool["z_max"] > 0]
            top = rank_units(pool, by)
            if not len(top):
                print(f"  no units to rank by {by}")
                continue
            print(f"  top {len(top)} by {by}:")
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                print(f"    {rank}. {row.group} / {row.unit_id}")
            fig = plot_paired_panel(
                f"{label} — {population}", top, paired, extent, criterion,
            )
            save_figure(fig, f"{slug(label)}__{note}__full-gabor")


def attach_tests(df: pd.DataFrame, session: str, channel: str | None = None) -> pd.DataFrame:
    """Add Zebra q and the Gabor BH q. Keep units tested by both.

    ``channel`` restricts the Zebra family before BH. SLAP2 green is corrected
    on its own; the red channel is not part of that family.
    """
    q_table = zebra_q_table(session)
    if channel is not None:
        q_table = q_table[q_table["group"].map(lambda group: str(group).split("__")[-1] == channel)].copy()
        q_table["q"] = _bh_fdr(q_table["p_null"].to_numpy())
        q_table["significant"] = q_table["q"] <= Q_LEVEL
        print(
            f"  Zebra BH within {channel}: "
            f"{int(q_table['significant'].sum())} / {len(q_table)}"
        )
    q_table = q_table[["group", "unit_id", "p_null", "q"]].rename(
        columns={"q": "q_zebra", "p_null": "p_zebra"}
    )
    out = df.merge(q_table, on=["group", "unit_id"], how="inner")
    out = out.dropna(subset=["p_value_bh", "q_zebra"]).copy().reset_index(drop=True)
    out["pass_zebra"] = out["q_zebra"] <= Q_LEVEL
    out["pass_gabor"] = out["p_value_bh"] <= Q_LEVEL
    out.attrs = dict(df.attrs)
    return out


def _top_zebra(pool: pd.DataFrame) -> pd.DataFrame:
    """Lowest Zebra p-value first, then largest peak |r|."""
    ordered = pool.dropna(subset=["p_zebra"]).sort_values(
        ["p_zebra", "abs_max_value"], ascending=[True, False],
    )
    return ordered.head(N_TOP)


def _top_gabor(pool: pd.DataFrame) -> pd.DataFrame:
    """Lowest unit-level Gabor p-value first, then largest peak |z|."""
    ordered = pool.dropna(subset=["p_value_gumbel"]).sort_values(
        ["p_value_gumbel", "z_max"], ascending=[True, False],
    )
    return ordered.head(N_TOP)


def _top_both(pool: pd.DataFrame) -> pd.DataFrame:
    """Smallest sum of the Zebra rank and the Gabor rank.

    Rank 1 is the best unit on that stimulus. Ties share the better rank.
    """
    work = pool.dropna(subset=["p_zebra", "p_value_gumbel"]).copy()
    if not len(work):
        return work
    work["zebra_rank"] = work["p_zebra"].rank(ascending=True, method="min")
    work["gabor_rank"] = work["p_value_gumbel"].rank(ascending=True, method="min")
    work["combined_rank"] = work["zebra_rank"] + work["gabor_rank"]
    ordered = work.sort_values(
        ["combined_rank", "zebra_rank", "p_zebra"],
        ascending=[True, True, True],
    )
    return ordered.head(N_TOP)


def plot_overlap_examples() -> None:
    """Top units for Zebra only, Gabor only, and both.

    Zebra only and Gabor only each keep the three lowest unit-level p-values.
    Both adds those two ranks and keeps the three smallest sums. Ephys is
    VISp layer 2/3, mesoscope is VISp_0, and SLAP2 is the green channel.
    """
    jobs = (
        ("ecephys 830794", "VISp layer 2/3", None),
        ("mesoscope 832700", "VISp_0", None),
        ("slap2 829704 green", "V1 layer 2/3", "green"),
    )
    categories = (
        ("zebra-only", "Zebra only, lowest p", lambda df: df["pass_zebra"] & ~df["pass_gabor"], _top_zebra),
        ("gabor-only", "Gabor only, lowest p", lambda df: df["pass_gabor"] & ~df["pass_zebra"], _top_gabor),
        ("both", "both, lowest rank sum", lambda df: df["pass_zebra"] & df["pass_gabor"], _top_both),
    )
    for label, population, channel in jobs:
        session, keep = SESSIONS[label]
        print(f"\n=== overlap examples {label} ===")
        signal = tested_zebra_signal(session)
        loaded = load_session(session, keep, signal=signal)
        if "p_value_bh" not in loaded.columns or loaded["p_value_bh"].isna().all():
            print("  p_value_bh is missing. Run compute_gabor_unit_significance.py first.")
            continue
        tested = attach_tests(primary_visual(label, loaded), session, channel=channel)
        paired = _both_maps(tested)
        extent = paired.attrs.get("extent") or (-45, 45, -45, 45)
        print(
            f"  tested by both: {len(tested)}; with both maps: {len(paired)}; "
            f"Zebra {(paired['pass_zebra']).sum() if len(paired) else 0}, "
            f"Gabor {(paired['pass_gabor']).sum() if len(paired) else 0}"
        )
        for stem, criterion, choose, ranker in categories:
            pool = paired.loc[choose(paired)] if len(paired) else paired
            selected = ranker(pool) if len(pool) else pool
            print(f"  {stem}: {len(pool)} eligible, drawing {len(selected)}")
            for row in selected.itertuples(index=False):
                extra = ""
                if stem == "both" and hasattr(row, "combined_rank"):
                    extra = (
                        f"  p {row.p_zebra:.3g}+{row.p_value_gumbel:.3g}"
                        f"  ranks {row.zebra_rank:.0f}+{row.gabor_rank:.0f}={row.combined_rank:.0f}"
                    )
                elif stem == "zebra-only":
                    extra = f"  p {row.p_zebra:.3g}"
                elif stem == "gabor-only":
                    extra = f"  p {row.p_value_gumbel:.3g}"
                print(f"    {row.group} / {row.unit_id}{extra}")
            if not len(selected):
                continue
            fig = plot_paired_panel(
                f"{label} — {population}", selected, paired, extent, criterion,
                overlay_gaussian=True,
            )
            save_figure(fig, f"{slug(label)}__{stem}__full-gabor")


_PIE_ORDER = ("neither", "Zebra only", "Gabor only", "both")
_PIE_COLORS = {
    "neither": "#c8c8c8",
    "Zebra only": "#4C78A8",
    "Gabor only": "#E45756",
    "both": "#54A24B",
}


def plot_overlap_example_summaries() -> None:
    """One example per selection, three columns by two rows.

    Columns are both, Gabor only, then Zebra only. Ephys takes the second
    ranked example in each column. Mesoscope and SLAP2 take the first. Both
    maps in a column share that selection's pie-chart color, and each row
    uses one color scale.
    """
    jobs = (
        ("ecephys 830794", "VISp layer 2/3", None, 1),
        ("mesoscope 832700", "VISp_0", None, 0),
        ("slap2 829704 green", "V1 layer 2/3", "green", 0),
    )
    columns = (
        ("both", "both", lambda df: df["pass_zebra"] & df["pass_gabor"], _top_both),
        ("gabor-only", "Gabor only", lambda df: df["pass_gabor"] & ~df["pass_zebra"], _top_gabor),
        ("zebra-only", "Zebra only", lambda df: df["pass_zebra"] & ~df["pass_gabor"], _top_zebra),
    )
    for label, population, channel, index in jobs:
        session, keep = SESSIONS[label]
        print(f"\n=== example summary {label} ===")
        signal = tested_zebra_signal(session)
        loaded = load_session(session, keep, signal=signal)
        if "p_value_bh" not in loaded.columns or loaded["p_value_bh"].isna().all():
            print("  p_value_bh is missing. Run compute_gabor_unit_significance.py first.")
            continue
        tested = attach_tests(primary_visual(label, loaded), session, channel=channel)
        paired = _both_maps(tested)
        extent = paired.attrs.get("extent") or (-45, 45, -45, 45)
        picked = []
        colors = []
        for stem, pie_name, choose, ranker in columns:
            pool = paired.loc[choose(paired)] if len(paired) else paired
            selected = ranker(pool) if len(pool) else pool
            colors.append(_PIE_COLORS[pie_name])
            if len(selected) > index:
                row = selected.iloc[index]
                print(f"  {stem}: example {index + 1}  {row.group} / {row.unit_id}")
                picked.append({"group": row.group, "unit_id": row.unit_id})
            else:
                print(f"  {stem}: no example at position {index + 1}")
                picked.append({"group": None, "unit_id": None})
        summary = pd.DataFrame(picked)
        fig = plot_paired_panel(
            f"{label} — {population}", summary, paired, extent, "one example",
            overlay_gaussian=True, column_colors=colors,
        )
        save_figure(fig, f"{slug(label)}__example-summary")


def _overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    if not len(df):
        return {name: 0 for name in _PIE_ORDER}
    return {
        "neither": int((~df["pass_zebra"] & ~df["pass_gabor"]).sum()),
        "Zebra only": int((df["pass_zebra"] & ~df["pass_gabor"]).sum()),
        "Gabor only": int((df["pass_gabor"] & ~df["pass_zebra"]).sum()),
        "both": int((df["pass_zebra"] & df["pass_gabor"]).sum()),
    }


_PIE_AREA = {
    "VISp": "V1",
    "VISl": "LM",
    "VISp_0": "V1",
    "DMD1": "basal",
    "DMD2": "apical",
}


def _draw_pie(ax, title: str, counts: dict[str, int]) -> None:
    """Percentages sit in the slices. The legend is the upper right of the pie."""
    from matplotlib.patches import Patch

    present = [name for name in _PIE_ORDER if counts[name] > 0]
    sizes = [counts[name] for name in present]
    colors = [_PIE_COLORS[name] for name in present]
    total = sum(counts.values())
    if total == 0:
        ax.set_axis_off()
        ax.set_title(title, fontsize=SUMMARY_LABEL * 1.3, pad=12)
        return
    _, _, autotexts = ax.pie(
        sizes, colors=colors, startangle=90,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 6 else "",
        pctdistance=0.62,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(SUMMARY_LABEL)
    handles = [
        Patch(
            facecolor=_PIE_COLORS[name], edgecolor="0.15", linewidth=0.6,
            label=f"{name} ({counts[name]})",
        )
        for name in _PIE_ORDER
    ]
    ax.legend(
        handles=handles, frameon=False, fontsize=SUMMARY_LABEL * 1.3,
        loc="upper left", bbox_to_anchor=(0.98, 1.05), borderaxespad=0,
    )
    ax.set_title(title, fontsize=SUMMARY_LABEL * 1.3, pad=14)


def _tested_frame(session: str, keep: str | None, channel: str | None = None) -> pd.DataFrame:
    signal = tested_zebra_signal(session)
    loaded = load_session(session, keep, signal=signal)
    return attach_tests(loaded, session, channel=channel)


def _save_overlap_pies(modality: str, scope: str, sides, stem: str) -> None:
    """One pie per group. ``sides`` is ``(name, frame)`` pairs."""
    n = len(sides)
    fig, axes = plt.subplots(1, n, figsize=(10.4 * n, 8.8), squeeze=False)
    fig.subplots_adjust(wspace=0.42, top=0.84, bottom=0.04, left=0.04, right=0.97)
    for ax, (name, subset) in zip(axes[0], sides):
        counts = _overlap_counts(subset)
        zebra = counts["Zebra only"] + counts["both"]
        print(f"  {modality} {name} ({scope}): {counts} (n={len(subset)}, zebra {zebra})")
        _draw_pie(ax, f"{_PIE_AREA.get(name, name)} ({len(subset)})", counts)
    note = "" if scope in ("all layers", "all planes", "green") else f", {scope}"
    fig.suptitle(
        f"{modality}{note} — significant at q ≤ {Q_LEVEL}",
        fontsize=SUMMARY_LABEL, y=1.02,
    )
    save_figure(fig, stem)


def plot_overlap_pies() -> None:
    """Two pies per modality: VISp and VISl, or DMD1 and DMD2.

    Ephys pools every layer. Mesoscope pools every plane of that area.
    SLAP2 is the green channel only. A second pair keeps ephys layer 2/3
    and mesoscope plane VISp_0, the populations matched to SLAP2.
    """
    ephys_session = SESSIONS["ecephys 830794"][0]
    meso_session = SESSIONS["mesoscope 832700"][0]
    slap_session, slap_keep = SESSIONS["slap2 829704 green"]
    print("\n=== overlap pies ===")
    ephys = _tested_frame(ephys_session, None)
    anatomy = load_ephys_unit_areas()[["unit_id", "area", "layer"]]
    ephys = ephys.merge(anatomy, on="unit_id", how="left")
    meso = _tested_frame(meso_session, None)
    slap = _tested_frame(slap_session, slap_keep, channel="green")
    panels = (
        ("ecephys", "all layers", (
            ("VISp", ephys[ephys["area"] == "VISp"]),
            ("VISl", ephys[ephys["area"] == "VISl"]),
        ), "ecephys__significance-overlap"),
        ("mesoscope", "all planes", (
            ("VISp", meso[meso["group"].astype(str).str.startswith("VISp_")]),
            ("VISl", meso[meso["group"].astype(str).str.startswith("VISl_")]),
        ), "mesoscope__significance-overlap"),
        ("slap2", "green", (
            ("DMD1", slap[slap["group"].astype(str).str.startswith("DMD1")]),
            ("DMD2", slap[slap["group"].astype(str).str.startswith("DMD2")]),
        ), "slap2__significance-overlap"),
    )
    for modality, scope, sides, stem in panels:
        _save_overlap_pies(modality, scope, sides, stem)
    # SLAP2 is already V1 layer 2/3. These pies hold depth to that population.
    layer23 = ephys[ephys["layer"].astype(str) == "2/3"]
    _save_overlap_pies("ecephys", "layer 2/3", (
        ("VISp", layer23[layer23["area"] == "VISp"]),
        ("VISl", layer23[layer23["area"] == "VISl"]),
    ), "ecephys__significance-overlap__layer23")
    _save_overlap_pies("mesoscope", "VISp_0", (
        ("VISp_0", meso[meso["group"].astype(str) == "VISp_0"]),
    ), "mesoscope__significance-overlap__VISp_0")


if __name__ == "__main__":
    plot_fit_summaries(prepare_fit_summaries())

    try:
        main()
        plot_paired_release()
        plot_overlap_examples()
        plot_overlap_example_summaries()
        plot_overlap_pies()
    except FileNotFoundError as exc:
        print(f"Skipping correlation-map panels: {exc}")

