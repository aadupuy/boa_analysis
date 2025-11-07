"""
step.py
--------

Visualization of Basin-of-Attraction (BoA) maps for different integration step sizes Δτ.

Functions
---------
plot_step_comparison(tab_n, L, base_path, output_path)
    Load BoA maps at different Δτ values and plot them side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def plot_step_comparison(
    tab_n=(10, 2, 0.25),
    L: float = 400.0,
    base_path: str = "segmentation_data/npy",
    prefix: str = "vel_s1.50_088_C256_forward_forward_step",
    suffix: str = "_8_BoA.npy",
    output_path: str = "fig/step_150_088.png",
):
    """
    Compare BoA maps at different integration step sizes Δτ.

    Parameters
    ----------
    tab_n : tuple of float
        List of step sizes to visualize (e.g. (10, 2, 0.25)).
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

    # Reference BoA (smallest step)
    n_ref = tab_n[-1]
    ref_file = base_path / f"{prefix}{n_ref:8.6f}{suffix}"
    boa_ref = load_boa_file(ref_file)
    N = boa_ref.shape[0]
    axis = np.linspace(-L / 2, L / 2, N)
    boa_slice_ref = boa_ref[N // 2, :, :]

    # Relabel reference BoA
    for i, m in enumerate(np.unique(boa_slice_ref)):
        boa_slice_ref[boa_slice_ref == m] = i + 1

    labels_ref = np.unique(boa_slice_ref)
    colors_ref = np.linspace(0, 0.9, len(labels_ref))
    cmap_ref = create_colormap(len(labels_ref), hues=colors_ref)

    # Plot reference
    axes[-1].pcolormesh(axis, axis, boa_slice_ref, cmap=cmap_ref)
    axes[-1].set_aspect("equal")
    axes[-1].set_xlim([-L / 2, L / 2])
    axes[-1].set_ylim([-L / 2, L / 2])
    axes[-1].set_title(f"$\\Delta\\tau$ = {n_ref * L / N:.1f} Mpc/$h$")

    # Plot earlier Δτ steps
    for i, n in enumerate(tab_n[:-1]):
        file = base_path / f"{prefix}{n:8.6f}{suffix}"
        boa = load_boa_file(file)
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
                hue[k] = 0.95

        halfs = (NCOLORS + 1) // 2
        halfv = NCOLORS // 2
        saturation = np.linspace(0.2, 1, halfs)
        value = np.linspace(0.4, 1, halfv)

        hsv = np.ones((NCOLORS, 3))
        hsv[:, 0] = hue
        hsv[:halfs, 1] = saturation
        hsv[halfs:, 2] = value
        hsv[hue == 0.95, 1] = 0.0

        cmap = mcolors.LinearSegmentedColormap.from_list("cmap", mcolors.hsv_to_rgb(hsv), N=NCOLORS)
        axes[i].pcolormesh(axis, axis, boa_slice, cmap=cmap)
        axes[i].set_aspect("equal")
        axes[i].set_xlim([-L / 2, L / 2])
        axes[i].set_ylim([-L / 2, L / 2])
        axes[i].set_title(f"$\\Delta\\tau$ = {n * L / N:.1f} Mpc/$h$")

    # Common formatting
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