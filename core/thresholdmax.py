"""
thresholdmax.py
----------------

Visualize the effect of the streamline count or threshold parameter (`n_sl`)
on Basin-of-Attraction (BoA) maps. Overlays missing regions (boa == 0) in red.

Functions
---------
plot_threshold_comparison(tab_n, L, base_path, output_path)
    Load BoA maps for different threshold parameters and plot them with missing regions highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from pathlib import Path


def load_boa_file(path: str) -> np.ndarray:
    """Load and reorient a BoA (Basin of Attraction) .npy file."""
    boa = np.load(path)
    return np.swapaxes(boa, 0, 2)


def create_colormap(n_colors: int, hues: np.ndarray = None) -> mcolors.Colormap:
    """Generate a linear colormap in HSV space."""
    if hues is None:
        hues = np.linspace(0, 0.9, n_colors)
    hsv = np.ones((n_colors, 3))
    hsv[:, 0] = hues
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.LinearSegmentedColormap.from_list("boa_cmap", rgb, N=n_colors)


def plot_threshold_comparison(
    tab_n=(5, 5000, 50000),
    L: float = 400.0,
    base_path: str = "segmentation_data/npy",
    prefix: str = "vel_s1.50_088_C256_forward_",
    suffix: str = "_BoA.npy",
    output_path: str = "fig/thresholdmax_150_088.png",
):
    """
    Compare BoA maps across different streamline counts or threshold parameters.

    Parameters
    ----------
    tab_n : tuple of int
        List of parameter values (e.g. (5, 5000, 50000)).
    L : float
        Physical box size in Mpc/h.
    base_path : str
        Folder containing BoA .npy files.
    prefix, suffix : str
        Naming convention for BoA files.
    output_path : str
        File path to save the resulting plot.
    """

    base_path = Path(base_path)
    fig, axes = plt.subplots(1, len(tab_n), figsize=(15, 6))
    if len(tab_n) == 1:
        axes = [axes]

    # Reference BoA (first value)
    n_ref = tab_n[0]
    ref_file = base_path / f"{prefix}{n_ref}{suffix}"
    boa_ref = load_boa_file(ref_file)
    N = boa_ref.shape[0]
    axis = np.linspace(-L / 2, L / 2, N)
    boa_slice_ref = boa_ref[N // 2, :, :]

    # Relabel for consistency
    for i, m in enumerate(np.unique(boa_slice_ref)):
        boa_slice_ref[boa_slice_ref == m] = i + 1

    labels_ref = np.unique(boa_slice_ref)
    colors_ref = np.linspace(0, 0.9, len(labels_ref))
    cmap_ref = create_colormap(len(labels_ref), hues=colors_ref)

    # Plot reference
    axes[0].pcolormesh(axis, axis, boa_slice_ref, cmap=cmap_ref)
    axes[0].set_aspect("equal")
    axes[0].set_xlim([-L / 2, L / 2])
    axes[0].set_ylim([-L / 2, L / 2])
    axes[0].set_title(f"$n_\\mathrm{{sl}}$ = {n_ref}")

    # Overlay other BoA maps
    for i, n in enumerate(tab_n[1:], start=1):
        file = base_path / f"{prefix}{n}{suffix}"
        boa = load_boa_file(file)
        boa_slice = boa[N // 2, :, :]

        # Percentage of missing voxels (boa == 0)
        missing_fraction = np.sum(boa == 0) / boa.size * 100
        print(f"{n}: {missing_fraction:.2f}% missing regions")

        # Create overlay mask (red for missing)
        boa_mask = boa_slice.copy()
        boa_mask[boa_mask != 0] = np.nan
        cmap_missing = matplotlib.cm.binary_r
        cmap_missing.set_bad("red", 0.0)

        axes[i].pcolormesh(axis, axis, boa_slice_ref, cmap=cmap_ref)
        axes[i].pcolormesh(axis, axis, boa_mask, cmap=cmap_missing)
        axes[i].set_aspect("equal")
        axes[i].set_xlim([-L / 2, L / 2])
        axes[i].set_ylim([-L / 2, L / 2])
        axes[i].set_title(f"$n_\\mathrm{{sl}}$ = {n}")

    # Formatting
    for ax in axes:
        ax.set_xlabel("X (Mpc/$h$)")
        ax.set_xticks([-200, -100, 0, 100, 200])
        ax.set_yticks([-200, -100, 0, 100, 200])
    axes[0].set_ylabel("Y (Mpc/$h$)")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✅ Figure saved to {output_path}")