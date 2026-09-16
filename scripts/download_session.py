import sys 
sys.path.append('..')

from utils import DandiSession
import numpy as np
from pathlib import Path

# Example session ID
# # Ephys session
# session_id = '001637'
# asset_path = 'sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb'
# # SLAP2 session
# session_id = '001424'
# asset_path = 'sub-829704/sub-829704_ses-slap2-829704-2025-12-18-10-57-36_image+ophys.nwb'
# output_path = '../../data'
# # Mesoscope session
# session_id = '001768'
# asset_path = 'sub-853137_ses-multiplane-ophys-853137-2026-07-14-15-24-43_ophys.nwb'

def main(session_id:str, asset_path:str, output_path:str):
    
    try:
        session = DandiSession(session_id)
        assets = session.assets()
    except KeyError:
        raise ValueError(f"Session ID '{session_id}' not found in DANDI archive")

    try:
        index = np.where([a.path == asset_path for a in assets])[0][0]
    except IndexError:
        raise ValueError(f"Asset path '{asset_path}' not found in session '{session_id}'")
    else:
        asset = assets[index]
        
        output_path = Path(output_path)
        output_path = output_path / asset.path
        destination = session.download(asset.identifier, output_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a specific NWB asset from a DANDI session.")
    parser.add_argument("session_id", type=str, help="DANDI session ID (e.g. '001637')")
    parser.add_argument("asset_path", type=str, help="Path of the asset to download (e.g. 'sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb')")
    parser.add_argument("--output_path", type=str, default="../../../rawdata/allen_open_scope", help="Directory to save the downloaded asset")

    args = parser.parse_args()
    main(args.session_id, args.asset_path, args.output_path)