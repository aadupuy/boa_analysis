"""
number_basins.py
-----------------

Analyze how the number of basins of attraction evolves with smoothing scale
and redshift, based on BoA segmentation maps.

Functions
---------
compute_n_basins(snapshot, smooth_list, base_path, prefix)
    Count number of basins for a given snapshot over different smoothing scales.

plot_n_basins_vs_smoothing(tab_snapshots, tab_redshifts, smooth_list, base_path, output_path)
    Plot number of basins as a function of R_smooth for multiple redshifts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path


def compute_n_basins(snapshot: str, smooth_list, base_path: str, prefix: str) -> list:
    """
    Compute the number of basins for a given snapshot across smoothing scales.

    Parameters
    ----------
    snapshot : str
        Snapshot ID (e.g. "015").
    smooth_list : list of str
        List of smoothing radii as strings (e.g. ["1.50", "3.00", "5.00"]).
    base_path : str
        Path to directory containing BoA .npy files.
    prefix : str
        File prefix before smooth and snapshot identifiers.

    Returns
    -------
    n_basins : list of int
        Number of basins (non-zero unique IDs) for each smoothing scale.
    """
    n_basins = []
    for smooth in smooth_list:
        fname = Path(base_path) / f"{prefix}{smooth}_{snapshot}_C256_forward_8_BoA.npy"
        boa = np.load(fname)
        boa = np.swapaxes(boa, 0, 2)
        n_basins.append(np.unique(boa)[1:].size)  # exclude ID=0
    return n_basins


def plot_n_basins_vs_smoothing(
    tab_snapshots=None,
    tab_redshifts=None,
    smooth_list=None,
    base_path: str = "segmentation_data/npy",
    prefix: str = "vel_s",
    output_path: str = "fig/Nbasins_rs_snapshots_BOA.png",
):
    """
    Plot the number of basins as a function of R_smooth for multiple redshifts.

    Parameters
    ----------
    tab_snapshots : list of str
        List of snapshot identifiers.
    tab_redshifts : list of float
        Corresponding redshift values.
    smooth_list : list of str
        Smoothing radii to analyze.
    base_path : str
        Path to directory containing BoA .npy files.
    prefix : str
        File prefix before smooth and snapshot identifiers.
    output_path : str
        Path to save the resulting figure.
    output_path : str
        Path to save the resulting figure.
    """

    if tab_snapshots is None:
        tab_snapshots = ['015', '016', '017', '018', '019', '027', '042', '050', '057', '067', '088']
    if tab_redshifts is None:
        tab_redshifts = [2.89, 2.48, 2.14, 1.44, 1.00, 0.74, 0.51, 0.40, 0.29, 0.20, 0.00]
    if smooth_list is None:
        smooth_list = ['1.50', '3.00', '5.00', '7.50', '10.0', '12.5', '15.0']

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=len(tab_snapshots))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, snapshot in enumerate(tab_snapshots):
        color_val = scalar_map.to_rgba(i)
        n_basins = compute_n_basins(snapshot, smooth_list, base_path, prefix)
        smooth_vals = [float(s) for s in smooth_list]

        ax.scatter(smooth_vals, n_basins, color=color_val, marker='s', s=25)
        # Add legend entry with corresponding redshift
        ax.scatter([], [], color=color_val, marker='s', s=25,
                   label=fr"$z = {tab_redshifts[i]:.2f}$")

    ax.set_xlim([0, 16])
    ax.set_ylim([0, 800])
    ax.legend(loc='best')
    ax.set_xlabel(r"$R_\mathrm{smooth}$ (Mpc/$h$)")
    ax.set_ylabel(r"$N_\mathrm{basin}$")

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Number of basins plot saved to {output_path}")