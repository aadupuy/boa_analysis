"""
visualizations_resolution.py
----------------------------

Visualize BoA (Basin of Attraction) segmentation at different grid resolutions
for a fixed smoothing radius (r_s) and redshift snapshot.

Functions
---------
plot_boa_vs_resolution(rs, snapshot, gridsizes, base_path, output_path)
    Plot mid-plane BoA slices for different grid resolutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def create_colormap(n_colors: int, hues: np.ndarray = None) -> mcolors.Colormap:
    """Generate an HSV-based colormap."""
    if hues is None:
        hues = np.linspace(0, 0.9, n_colors)
    hsv = np.ones((n_colors, 3))
    hsv[:, 0] = hues
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.LinearSegmentedColormap.from_list("boa_cmap", rgb, N=n_colors)


def plot_boa_vs_resolution(
    rs: str = "5.00",
    snapshot: str = "088",
    gridsizes=None,
    L: float = 400.0,
    base_path: str = "segmentation_data/npy",
    output_path: str = "fig/visualizations_resolution.png",
):
    """
    Plot BoA mid-plane slices for different grid resolutions.

    Parameters
    ----------
    rs : str
        Smoothing radius (e.g. '5.00').
    snapshot : str
        Snapshot identifier (e.g. '088').
    gridsizes : list of str
        Grid resolutions to compare (e.g. ['64', '256']).
    L : float
        Box size in Mpc/h.
    base_path : str
        Directory containing BoA .npy files.
    output_path : str
        Path to save the resulting figure.
    """

    if gridsizes is None:
        gridsizes = ['64', '256']

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows=1, ncols=len(gridsizes), figsize=(5 * len(gridsizes), 6))
    if len(gridsizes) == 1:
        ax = [ax]

    base_path = Path(base_path)

    for i, size in enumerate(gridsizes):
        boa_file = base_path / f"vel_s{rs}_{snapshot}_C{size}_forward_8_BoA.npy"
        boa = np.swapaxes(np.load(boa_file), 0, 2)

        N = boa.shape[0]
        axis = np.linspace(-L / 2, L / 2, N)
        boa_slice = boa[N // 2, :, :]

        labels = np.unique(boa_slice)
        NBOA = len(labels)
        cmap = create_colormap(NBOA)

        pcm = ax[i].pcolormesh(axis, axis, boa_slice, cmap=cmap)
        ax[i].set_aspect("equal")
        ax[i].set_xlim([-L / 2, L / 2])
        ax[i].set_ylim([-L / 2, L / 2])
        ax[i].set_title(fr"$N_\mathrm{{grid}} = {size}$")

        ax[i].set_xticks([-200, -100, 0, 100, 200])
        ax[i].set_yticks([-200, -100, 0, 100, 200])
        ax[i].set_xlabel(r"SGX (Mpc/$h$)")
        if i == 0:
            ax[i].set_ylabel(r"SGY (Mpc/$h$)")
        else:
            ax[i].set_yticklabels([])

#     fig.savefig(output_path, bbox_inches="tight", dpi=300)
#     print(f"✅ Resolution comparison figure saved to {output_path}")