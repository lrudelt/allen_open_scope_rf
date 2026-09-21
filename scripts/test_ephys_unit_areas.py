"""Stream the example ecephys session and count units per CCF structure.

Uses the same session as the ephys example in ``download_session.py``
(dandiset 001637, sub-830794). CCF acronyms come from NWB
``electrodes.location`` at each unit's extremum channel
(``device_name`` + ``extremum_channel_index``); area and layer are decoded
locally (no atlas lookup). To verify raw channel labels only, count
``electrodes.location`` on the electrodes table (see ``utils.ecephys_io``).

    cd code/scripts
    python test_ephys_unit_areas.py
"""

from __future__ import annotations

import sys

sys.path.append("..")

import numpy as np
import pandas as pd

from utils import DandiSession, load_unit_areas, structure_label

# Same ephys example as download_session.py
SESSION_ID = "001637"
ASSET_PATH = (
    "sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb"
)


def main(session_id: str = SESSION_ID, asset_path: str = ASSET_PATH) -> pd.DataFrame:
    session = DandiSession(session_id)
    assets = session.assets()
    try:
        index = np.where([a.path == asset_path for a in assets])[0][0]
    except IndexError as exc:
        raise ValueError(
            f"Asset path {asset_path!r} not found in dandiset {session_id}"
        ) from exc

    asset = assets[index]
    print(f"Streaming {asset.path}")
    print(f"  asset id: {asset.identifier}")

    with session.open(asset.identifier) as stream:
        if stream.modality != "ecephys":
            raise ValueError(f"expected ecephys NWB, got {stream.modality!r}")
        units = load_unit_areas(stream)

    # Prefer decoded area+layer; fall back to raw location if decode left gaps.
    if "structure" not in units.columns:
        units = units.copy()
        units["structure"] = [
            structure_label(a, L) for a, L in zip(units["area"], units["layer"])
        ]

    counts = (
        units.groupby(["area", "layer"], dropna=False)
        .size()
        .reset_index(name="n_units")
        .sort_values(["area", "layer"], kind="mergesort")
    )
    counts["structure"] = [
        structure_label(a, L) for a, L in zip(counts["area"], counts["layer"])
    ]

    print(f"\nTotal units: {len(units)}")
    print(f"Unique structures (area × layer): {len(counts)}\n")
    print(f"{'structure':<16} {'area':<12} {'layer':<8} {'n_units':>8}")
    print("-" * 48)
    for row in counts.itertuples(index=False):
        layer = "" if pd.isna(row.layer) else str(row.layer)
        area = "" if pd.isna(row.area) else str(row.area)
        print(f"{row.structure:<16} {area:<12} {layer:<8} {row.n_units:>8}")

    print("\nBy structure (n_units):")
    by_struct = units["structure"].value_counts(dropna=False)
    for name, n in by_struct.items():
        print(f"  {name}: {n}")

    return counts


if __name__ == "__main__":
    main()
