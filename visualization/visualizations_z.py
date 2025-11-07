"""
visualizations_z.py
-------------------

Visualize BoA (Basin of Attraction) mid-plane slices across redshifts
for a fixed smoothing scale r_s.

Functions
---------
plot_boa_vs_redshift(rs, snapshots, redshifts, base_path, output_path)
    Plot BoA slices at different redshifts for a fixed smoothing radius.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def create_colormap(n_colors: int, hues: np.ndarray = None) -> mcolors.Colormap:
    """Generate a simple HSV colormap."""
    if hues is None:
        hues = np.linspace(0, 0.9, n_colors)
    hsv = np.ones((n_colors, 3))
    hsv[:, 0] = hues
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.LinearSegmentedColormap.from_list("boa_cmap", rgb, N=n_colors)


def plot_boa_vs_redshift(
    rs: str = "5.00",
    snapshots=None,
    redshifts=None,
    L: float = 400.0,
    base_path: str = "segmentation_data/npy",
    output_path: str = "fig/visualizations_z.png",
):
    """
    Plot BoA mid-plane slices at multiple redshifts for a fixed smoothing radius r_s.

    Parameters
    ----------
    rs : str
        Smoothing radius (e.g. '5.00').
    snapshots : list of str
        Snapshot identifiers (e.g. ['015', '088']).
    redshifts : list of str
        Redshift labels corresponding to each snapshot.
    L : float
        Box size in Mpc/h.
    base_path : str
        Directory containing BoA .npy files.
    output_path : str
        Path to save the resulting figure.
    """

    if snapshots is None:
        snapshots = ['015', '088']
    if redshifts is None:
        redshifts = ['2.89', '0']

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows=1, ncols=len(snapshots), figsize=(5 * len(snapshots), 6))
    if len(snapshots) == 1:
        ax = [ax]  # ensure iterable

    base_path = Path(base_path)

    for i, (snap, zlabel) in enumerate(zip(snapshots, redshifts)):
        # --- load BoA ---
        boa_file = base_path / f"vel_s{rs}_{snap}_C256_forward_8_BoA.npy"
        boa = np.swapaxes(np.load(boa_file), 0, 2)
        N = boa.shape[0]
        axis = np.linspace(-L / 2, L / 2, N)
        boa_slice = boa[N // 2, :, :]

        # Remap IDs to contiguous values
        for j, m in enumerate(np.unique(boa_slice)):
            boa_slice[boa_slice == m] = j + 1

        labels = np.unique(boa_slice)
        NBOA = len(labels)
        cmap = create_colormap(NBOA)

        pcm = ax[i].pcolormesh(axis, axis, boa_slice, cmap=cmap)
        ax[i].set_aspect("equal")
        ax[i].set_xlim([-L / 2, L / 2])
        ax[i].set_ylim([-L / 2, L / 2])
        ax[i].set_title(fr"$z = {zlabel}$")

        ax[i].set_xticks([-200, -100, 0, 100, 200])
        ax[i].set_yticks([-200, -100, 0, 100, 200])
        ax[i].set_xlabel(r"X (Mpc/$h$)")
        if i == 0:
            ax[i].set_ylabel(r"Y (Mpc/$h$)")
        else:
            ax[i].set_yticklabels([])

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"✅ Visualization across redshifts saved to {output_path}")