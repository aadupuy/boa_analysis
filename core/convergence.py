"""
convergence.py
---------------

Plot the convergence of the number of segmented basins as a function of
smoothing length (l_s), integration step (Δτ), and number of streamlines (n_sl).

Functions
---------
plot_convergence(length, nb_length, step, nb_step, threshold, nb_threshold, L, N, output_path)
    Plot the evolution of basin counts across different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
    length=None,
    nb_length=None,
    step=None,
    nb_step=None,
    threshold=None,
    nb_threshold=None,
    L: float = 400.0,
    N: int = 256,
    output_path: str = "fig/convergence_params.png",
):
    """
    Plot the number of segmented basins as a function of key parameters.

    Parameters
    ----------
    length, nb_length : array-like
        Smoothing lengths and corresponding number of basins.
    step, nb_step : array-like
        Integration step sizes and corresponding number of basins.
    threshold, nb_threshold : array-like
        Streamline counts or threshold parameters and corresponding number of basins.
    L : float
        Physical box size in Mpc/h.
    N : int
        Grid size (for axis scaling).
    output_path : str
        Path to save the output figure.
    """

    plt.rcParams.update({'font.size': 14})

    # Default data (if not provided)
    if length is None:
        length = np.array([5, 25, 50, 100, 200, 300]) * L / N
        nb_length = np.array([4931, 5236, 2201, 755, 647, 647])

    if step is None:
        step = np.array([10, 5, 2, 1, 0.5, 0.25]) * L / N
        nb_step = np.array([595, 472, 542, 610, 647, 647])

    if threshold is None:
        threshold = np.array([1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000])
        nb_threshold = np.array([647, 647, 646, 644, 643, 638, 608, 455, 329, 92])

    # Initialize figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # --- Panel 1: smoothing length ---
    ax1.plot(length, nb_length, color='blue', lw=1)
    ax1.set_xlim([0, 500])
    ax1.set_ylim([0, 5500])
    ax1.set_aspect(500 / 5500)
    ax1.axhline(647, ls='--', color='black', lw=0.5)
    ax1.set_xlabel(r"$l_s$ (Mpc/$h$)")
    ax1.set_ylabel("Number of segmented basins")

    # --- Panel 2: integration step ---
    logstep = np.log10(step)
    ax2.plot(logstep, nb_step, color='blue', lw=1)
    ax2.set_xlim([-0.5, 1.5])
    ax2.set_ylim([450, 700])
    ax2.set_aspect((1.5 + 0.5) / (700 - 450))
    ax2.axhline(647, ls='--', color='black', lw=0.5)
    ax2.set_xlabel(r"$\Delta\tau$ (Mpc/$h$)")
    ax2.set_xticks([np.log10(1), np.log10(10)])
    minorticks = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax2.set_xticks(np.log10(minorticks), minor=True)
    ax2.set_xticklabels([1, 10])

    # --- Panel 3: number of streamlines ---
    logthres = np.log10(threshold)
    ax3.plot(logthres, nb_threshold, color='blue', lw=1)
    ax3.set_xlim([-0.1, 4.8])
    ax3.set_ylim([50, 700])
    ax3.set_aspect((4.8 + 0.1) / (700 - 50))
    ax3.axhline(647, ls='--', color='black', lw=0.5)
    ax3.set_xlabel(r"$n_\mathrm{sl}$")

    # ✅ fixed ticks and labels
    xticks_major = np.log10([1, 100, 1000, 10000])
    ax3.set_xticks(xticks_major)
    ax3.set_xticklabels(["1", "10²", "10³", "10⁴"])

    # Minor ticks (unchanged)
    minorticks = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000
    ])
    ax3.set_xticks(np.log10(minorticks), minor=True)

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✅ Convergence figure saved to {output_path}")