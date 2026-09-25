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

Ephys units are restricted to visual cortex layer 2/3 using
``load_unit_areas`` (extremum channel -> ``electrodes.location``), the same
assignment as ``test_ephys_unit_areas.py``. Mesoscope units are restricted to
plane ``VISp_0``. SLAP2 ROIs are all kept: that recording is already V1 layer 2/3.

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

from optimize_waven_parameters import load_rf_results
from utils import DandiSession, load_unit_areas
from waven_settings import analysis_coverage

# Same relative root the notebook uses from code/notebooks.
RF_ROOT = Path(__file__).resolve().parents[3] / "results" / "allen_open_scope" / "rf"
PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"
FILE_PREFIX = "data_release"

N_TOP = 3
TRIAL, PHASE = 0, 0
GABOR_SIGNAL = "events"
FONTSIZE = 14
TITLE_FONTSIZE = FONTSIZE + 1

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

# Same asset the area test streams.
EPHYS_DANDISET = "001637"
EPHYS_ASSET = (
    "sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb"
)

WAVEN_COLS = [
    "unit_id", "abs_max_value", "rf_map", "delay", "duration",
    "x_deg", "y_deg", "theta_rad", "theta_deg", "sigma_deg", "frequency",
]
GABOR_COLS = [
    "unit_id", "p_value", "z_max", "ori_idx", "ori_rad", "delay",
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


def discover_waven(session: str) -> dict[str, Path]:
    out = {}
    d = RF_ROOT / "waven" / "zebra" / "optimized" / session
    if not d.is_dir():
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
        if m.group("trial") is not None and int(m.group("trial")) != TRIAL:
            continue
        out[m.group("group")] = f
    return out


def discover_gabor(session: str) -> dict[str, Path]:
    out = {}
    for f in sorted((RF_ROOT / "gabors" / session).glob(
        f"{session}__rf-spike-count__*__optimized.h5"
    )):
        out[f.stem.split("__")[-2]] = f

    channel_prefix = f"rf-{GABOR_SIGNAL}-"
    for f in sorted((RF_ROOT / "gabors" / session / GABOR_SIGNAL).glob(
        f"{session}__*__optimized.h5"
    )):
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

    z_peak = np.nanmax(np.abs(z_scores), axis=(2, 3))
    at_min = p_values <= p_values.min(axis=1, keepdims=True) + 1e-12
    best_ori = np.argmax(np.where(at_min, z_peak, -np.inf), axis=1)
    u = np.arange(len(unit_names))

    df = pd.DataFrame({
        "unit_id": [canonical_unit_id(x) for x in unit_names],
        "p_value": p_values[u, best_ori],
        "ori_idx": best_ori,
        "ori_rad": orientations[best_ori],
    })
    df["rf_map"] = [z_scores[i, best_ori[i]] for i in u]
    df["z_max"] = z_peak[u, best_ori]
    df.attrs["extent"] = _extent_from_centres(x_pos, y_pos)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    return df


def load_session(session: str, keep: str | None) -> pd.DataFrame:
    waven_files, gabor_files = discover_waven(session), discover_gabor(session)
    frames, extent, orientations = [], None, ()
    for group in sorted(
        g for g in set(waven_files) | set(gabor_files) if in_filter(g, keep)
    ):
        w = load_waven(waven_files.get(group))
        s = load_gabor(gabor_files.get(group))
        if extent is None and len(s):
            extent = s.attrs["extent"]
            orientations = s.attrs.get("orientations", ())
        merged = w.merge(s, on="unit_id", how="inner", suffixes=("_zebra", "_gabor"))
        merged.insert(0, "group", group)
        if len(merged):
            frames.append(merged)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["group", "unit_id", "abs_max_value", "rf_map_zebra", "p_value", "rf_map_gabor"]
    )
    df.attrs["extent"] = extent or (-45, 45, -45, 45)
    df.attrs["orientations"] = tuple(float(o) for o in orientations)
    return df


def ephys_vis_layer23(df: pd.DataFrame) -> pd.DataFrame:
    """Keep visual-cortex layer 2/3 units, labeled via ``load_unit_areas``."""
    session = DandiSession(EPHYS_DANDISET)
    assets = session.assets()
    try:
        index = np.where([a.path == EPHYS_ASSET for a in assets])[0][0]
    except IndexError as exc:
        raise ValueError(f"Asset path {EPHYS_ASSET!r} not found") from exc
    asset = assets[index]
    print(f"Streaming {asset.path} for area / layer")
    with session.open(asset.identifier) as stream:
        anatomy = load_unit_areas(stream)

    anatomy = anatomy.copy()
    anatomy["unit_id"] = anatomy["unit_name"].map(_text)
    anatomy["probe"] = anatomy["probe"].map(_text)
    keep = (
        anatomy["area"].astype(str).str.startswith("VIS")
        & (anatomy["layer"].astype(str) == "2/3")
    )
    anatomy = anatomy.loc[keep, ["probe", "unit_id", "area", "layer", "structure"]]
    print(f"  visual cortex layer 2/3: {len(anatomy)} units")
    out = df.merge(
        anatomy, left_on=["group", "unit_id"], right_on=["probe", "unit_id"], how="inner"
    )
    return out.drop(columns=["probe"])


def restrict_population(label: str, df: pd.DataFrame) -> pd.DataFrame:
    if label.startswith("ecephys"):
        return ephys_vis_layer23(df)
    if label.startswith("mesoscope"):
        out = df[df["group"] == "VISp_0"].copy()
        print(f"  VISp_0: {len(out)} paired units")
        return out
    print(f"  all SLAP2 ROIs (V1 layer 2/3): {len(df)} paired units")
    return df


def rank_units(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Top units. ``zebra`` uses |r|; ``gabor`` uses p-value then |z|."""
    both = df.dropna(subset=["abs_max_value", "p_value", "rf_map_zebra", "rf_map_gabor"])
    if by == "zebra":
        ordered = both.sort_values("abs_max_value", ascending=False)
    elif by == "gabor":
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


def plot_maps(rows: pd.DataFrame, kind: str, extent, orientations) -> plt.Figure:
    """One row of maps, unit order left to right matching ``rows``."""
    n = len(rows)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False, sharex=True, sharey=True)
    zebra = kind == "zebra"
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axs[0, i]
        if zebra:
            _show(fig, ax, row["rf_map_zebra"], WAVEN_EXTENT, "upper", "Correlation", True)
            for edge in (-40, 40):
                ax.axvline(edge, color=".75", lw=.6, zorder=1)
                ax.axhline(edge, color=".75", lw=.6, zorder=1)
            where = row["structure"] if "structure" in row and pd.notna(row["structure"]) else row["group"]
            ax.set_title(
                f'{where}\n{row["unit_id"]}\n|r|max = {row["abs_max_value"]:.3f}',
                fontsize=TITLE_FONTSIZE,
            )
        else:
            _show(fig, ax, row["rf_map_gabor"], extent, "lower", "response (z-scored)", True)
            ori = ""
            if pd.notna(row.get("ori_idx")) and len(orientations):
                ori = f', θ = {np.rad2deg(orientations[int(row["ori_idx"])]):.0f}°'
            where = row["structure"] if "structure" in row and pd.notna(row["structure"]) else row["group"]
            ax.set_title(
                f'{where}\n{row["unit_id"]}{ori}\n'
                f'p = {row["p_value"]:.4f}, |z|max = {row["z_max"]:.2f}',
                fontsize=TITLE_FONTSIZE,
            )
    # Shared axes would otherwise crop to whichever panel was drawn last.
    if zebra:
        x0, x1, y0, y1 = WAVEN_EXTENT
    else:
        x0, x1, y0, y1 = extent
    axs[0, 0].set_xlim(x0, x1)
    axs[0, 0].set_ylim(y0, y1)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"{FILE_PREFIX}__{stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def main() -> None:
    print(f"RF_ROOT = {RF_ROOT}")
    if not RF_ROOT.is_dir():
        raise FileNotFoundError(
            f"RF results not found at {RF_ROOT}. "
            "This is the same '../../../results/allen_open_scope/rf' root the notebook uses."
        )

    for label, (session, keep) in SESSIONS.items():
        print(f"\n=== {label} ===")
        loaded = load_session(session, keep)
        extent = loaded.attrs.get("extent", (-45, 45, -45, 45))
        orientations = loaded.attrs.get("orientations", ())
        df = restrict_population(label, loaded)
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
            fig_z = plot_maps(top, "zebra", extent, orientations)
            fig_z.suptitle(f"{label} — ranked by Zebra", fontsize=TITLE_FONTSIZE + 2, y=1.02)
            save_figure(fig_z, f"{stem}__zebra-rfs")
            fig_g = plot_maps(top, "gabor", extent, orientations)
            fig_g.suptitle(f"{label} — ranked by local Gabors", fontsize=TITLE_FONTSIZE + 2, y=1.02)
            save_figure(fig_g, f"{stem}__gabor-rfs")


if __name__ == "__main__":
    main()
