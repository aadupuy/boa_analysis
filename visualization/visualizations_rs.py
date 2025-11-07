"""
visualizations_rs.py
--------------------

Visualize BoA (Basin of Attraction) maps across different smoothing scales (r_s)
for a fixed redshift snapshot. Each subplot shows a mid-plane slice of the 3D BoA field.

Functions
---------
plot_boa_vs_smoothing(snapshot, tab_rs, base_path, attractor_path, output_path)
    Plot BoA slices at different smoothing radii for a given snapshot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path


def create_colormap(n_colors: int, hues: np.ndarray = None) -> mcolors.Colormap:
    """Generate an HSV-based colormap."""
    if hues is None:
        hues = np.linspace(0, 0.9, n_colors)
    hsv = np.ones((n_colors, 3))
    hsv[:, 0] = hues
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.LinearSegmentedColormap.from_list("boa_cmap", rgb, N=n_colors)


def plot_boa_vs_smoothing(
    snapshot: str = "088",
    tab_rs=None,
    L: float = 400.0,
    base_path: str = "segmentation_data/npy",
    output_path: str = "fig/visualizations_rs.png",
):
    """
    Plot BoA slices across different smoothing radii (r_s) for a fixed snapshot.

    Parameters
    ----------
    snapshot : str
        Snapshot identifier (default: '088').
    tab_rs : list of str
        List of smoothing radii (e.g. ['1.50', '3.00', ...]).
    L : float
        Box size in Mpc/h.
    base_path : str
        Directory containing BoA .npy files.
    output_path : str
        Path to save the final multi-panel figure.
    """

    if tab_rs is None:
        tab_rs = ['1.50', '3.00', '5.00', '7.50', '10.0', '12.5', '15.0']

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

    base_path = Path(base_path)

    # --- reference BoA (largest r_s) ---
    rs_ref = tab_rs[-1]
    boa_ref_file = base_path / f"vel_s{rs_ref}_{snapshot}_C256_forward_8_BoA.npy"
    boa_ref = np.swapaxes(np.load(boa_ref_file), 0, 2)
    N = boa_ref.shape[0]
    axis = np.linspace(-L / 2, L / 2, N)
    boa_slice_ref = boa_ref[N // 2, :, :]

    labels_ref = np.unique(boa_slice_ref)
    NBOA = len(labels_ref)
    colors_ref = np.linspace(0, 0.9, NBOA)
    cmap_ref = create_colormap(NBOA, hues=colors_ref)

    row, col = divmod(len(tab_rs) - 1, 4)
    ax[row, col].pcolormesh(axis, axis, boa_slice_ref, cmap=cmap_ref)
    ax[row, col].set_aspect("equal")
    ax[row, col].set_xlim([-L / 2, L / 2])
    ax[row, col].set_ylim([-L / 2, L / 2])
    ax[row, col].set_title(fr"$r_s = {rs_ref}$")

    # --- plot other r_s values ---
    for idx, rs in enumerate(tab_rs[:-1]):
        row, col = divmod(idx, 4)
        boa_file = base_path / f"vel_s{rs}_{snapshot}_C256_forward_8_BoA.npy"
        boa = np.swapaxes(np.load(boa_file), 0, 2)
        boa_slice = boa[N // 2, :, :]

        labels = np.unique(boa_slice)
        NBOA = len(labels)
        NCOLORS = int(labels[-1] - labels[0] + 1)
        hue = np.zeros(NCOLORS)

        for k, newi in enumerate(range(int(labels[0]), int(labels[-1]) + 1)):
            if newi in labels:
                old_boa = boa_slice_ref[boa_slice == newi].astype(int)
                counts = np.bincount(old_boa)
                oldi = np.argmax(counts)
                hue[k] = colors_ref[labels_ref == oldi]
            else:
                hue[k] = 0.95  # mark missing regions

        halfs = (NCOLORS + 1) // 2
        halfv = NCOLORS // 2
        saturation = np.linspace(0.2, 1, halfs)
        value = np.linspace(0.4, 1, halfv)

        hsv = np.ones((NCOLORS, 3))
        hsv[:, 0] = hue
        hsv[:halfs, 1] = saturation
        hsv[halfs:, 2] = value
        hsv[hue == 0.95, 1] = 0.0  # grey

        cmap = mcolors.LinearSegmentedColormap.from_list("boa_map", mcolors.hsv_to_rgb(hsv), N=NCOLORS)
        ax[row, col].pcolormesh(axis, axis, boa_slice, cmap=cmap)
        ax[row, col].set_aspect("equal")
        ax[row, col].set_xlim([-L / 2, L / 2])
        ax[row, col].set_ylim([-L / 2, L / 2])
        ax[row, col].set_title(fr"$r_s = {rs}$")

    # --- shared formatting ---
    for i in range(len(tab_rs)):
        r, c = divmod(i, 4)
        ax[r, c].set_xticks([-200, -100, 0, 100, 200])
        ax[r, c].set_yticks([-200, -100, 0, 100, 200])

    ax[0, 0].set_ylabel(r"Y (Mpc/$h$)")
    ax[1, 0].set_ylabel(r"Y (Mpc/$h$)")
    for a in ax[-1, :3]:
        a.set_xlabel(r"X (Mpc/$h$)")

    # remove unused subplot (bottom-right)
    ax[1, 3].set_axis_off()

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"✅ Visualization across R_s saved to {output_path}")