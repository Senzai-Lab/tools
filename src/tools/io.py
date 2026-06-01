from pathlib import Path

import numpy as np
import pandas as pd

def nearest_indices(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    dist2 = ((points[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    return dist2.argmin(axis=1)

def load_probe_ks(path: Path) -> pd.DataFrame:
    pos = np.load(path / "channel_positions.npy")
    shank = np.load(path / "channel_shanks.npy").astype(int)

    df = pd.DataFrame(pos, columns=["x", "depth"])
    df["shank_ids"] = shank
    return df

def load_probe(path: str | Path) -> pd.DataFrame:
    """
    path can be:
    - Kilosort4 output directory
    - ProbeInterface .JSON file
    """

    path = Path(path)

    if path.is_dir():
        df = load_probe_ks(path)
    elif path.is_file() and path.suffix == '.json':
        from probeinterface import read_probeinterface
        df = read_probeinterface(path).to_dataframe()
    else:
        raise ValueError(f"Unknown probe path: {path}")
    # Sort shank-wise ascending then depth-wise descending
    df.sort_values(by=['shank_ids', 'depth'],
                   ascending=[True, True],
                   inplace=True)
    
    df['depth_rank'] = df.groupby('shank_ids').cumcount()

    return df

def load_units_ks(path: Path) -> pd.DataFrame:
    # compute units summary from kilosort output
    metadata = pd.read_csv(path / 'cluster_group.tsv', sep='\t')
    metadata = metadata[metadata['KSLabel'] == 'good']
    good_units = metadata['cluster_id']
    
    # Load spikes
    clusters = np.load(path / 'spike_clusters.npy', mmap_mode='r')
    positions = np.load(path / 'spike_positions.npy', mmap_mode='r')

    mask = np.isin(clusters, good_units)
    spikes_df = pd.DataFrame(positions[mask], columns=["x", "depth"])
    spikes_df['cluster_id'] = clusters[mask]
    
    # Units summary
    units_df = spikes_df.groupby('cluster_id').agg(
        x = ('x', 'mean'),
        depth = ('depth', 'mean'),
    )

    # Compute nearest channel contact and shank
    channel_positions = np.load(path / 'channel_positions.npy')
    channel_shanks = np.load(path / 'channel_shanks.npy')
    channel_map = np.load(path /'channel_map.npy')
    
    units_xy = units_df[['x', 'depth']].to_numpy()
    nearest_idx = nearest_indices(units_xy, channel_positions)

    units_df['nearest_channel'] = channel_map[nearest_idx]
    units_df['shank_ids'] = channel_shanks[nearest_idx]

    return units_df

def load_units_metadata(path: str | Path, mode: str = 'auto') -> pd.DataFrame:
    """
    path is Kilosort4 output directory.
    """
    path = Path(path)
    assert path.is_dir()
    
    mode = mode.lower().strip()
    if mode == 'auto':
        if (path / 'cluster_info.tsv').is_file():
            mode = 'phy'
        else:
            mode = 'kilosort'
    
    # Load cluster metadata
    if mode == 'phy':
        metadata = pd.read_csv(path / 'cluster_info.tsv', sep='\t')
        metadata = metadata[metadata['group'] == 'good']
        metadata.rename(columns={'sh': 'shank_ids'}, inplace=True)
    elif mode == 'kilosort' or mode == 'ks':
        metadata = load_units_ks(path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'auto', 'phy', 'kilosort | 'ks'")

    # Sort and rank units shank-wise then depth-wise
    metadata.sort_values(
        by=['shank_ids', 'depth', 'cluster_id'],
        ascending=[True, True, True],
        inplace=True)
    
    metadata["depth_rank"] = metadata.groupby("shank_ids").cumcount()
    
    return metadata