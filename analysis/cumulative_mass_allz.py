"""
cumulative_mass_allz.py
-----------------------

Compute and plot the cumulative mass distribution of basins of attraction
for multiple redshifts and smoothing scales.

Functions
---------
compute_cumulative_mass(boa, mass)
    Compute the normalized cumulative mass of basins from a BoA segmentation.

plot_cumulative_mass_allz(snapshots, tab_rs, base_path, sav_path, output_path)
    Plot cumulative mass distribution for all redshifts and smoothing scales.
"""

import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
from pathlib import Path


def compute_cumulative_mass(boa: np.ndarray, mass: np.ndarray):
    """
    Compute normalized cumulative mass of basins.

    Parameters
    ----------
    boa : ndarray
        3D array of BoA basin IDs.
    mass : ndarray
        3D mass field corresponding to the same volume.

    Returns
    -------
    masses : ndarray
        Sorted basin masses.
    sum_mass : ndarray
        Normalized cumulative mass (0–1).
    """
    indices = np.unique(boa)[1:]  # skip ID=0 (background)
    basin_masses = np.array([np.sum(mass[boa == i]) for i in indices])
    basin_masses.sort()
    cumulative = np.cumsum(basin_masses)
    cumulative = (cumulative - cumulative.min()) / (cumulative.max() - cumulative.min())
    return basin_masses, cumulative


def plot_cumulative_mass_allz(
    snapshots=None,
    tab_rs=None,
    sav_path: str = "/path/to/density/field/sav/",
    base_path: str = "segmentation_data/output_redshift/npy",
    output_path: str = "fig/cumulative_mass_allz.png",
):
    """
    Plot cumulative mass distribution of basins for all redshifts and smoothing scales.

    Parameters
    ----------
    snapshots : list of str
        Snapshot identifiers.
    tab_rs : list of str
        List of smoothing radii (as strings, e.g. ['1.50', '3.00', ...]).
    sav_path : str
        Path to directory containing .sav density fields.
    base_path : str
        Path to directory containing BoA .npy files.
    output_path : str
        File path to save the resulting plot.
    """

    # --- constants ---
    L = 400.0  # box size (Mpc/h)
    Npart = 3840**3
    V = L**3
    mass_part = 9.4e7
    mean_rho = Npart / V * mass_part

    if snapshots is None:
        snapshots = ['015', '016', '017', '018', '019', '027', '042', '050', '057', '067', '088']
    if tab_rs is None:
        tab_rs = ['1.50', '3.00', '5.00', '7.50', '10.0', '12.5', '15.0']

    colors = ['red', 'orange', 'yellow', 'limegreen', 'cyan', 'blue', 'indigo']
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(6, 6))

    for rs, color in zip(tab_rs, colors):
        for snapshot in snapshots:
            # --- load density grid from .sav ---
            sav_file = Path(sav_path) / f"vel_s{rs}_{snapshot}_C256.sav"
            sav_data = readsav(sav_file, verbose=False)
            # handle field name automatically (IDL structure)
            if hasattr(sav_data, "vel_s1_field"):
                d_grid = sav_data.vel_s1_field.d[0]
            elif hasattr(sav_data, "d"):
                d_grid = sav_data.d[0]
            else:
                raise KeyError(f"Could not find density field in {sav_file}")

            mass = d_grid * mean_rho * (L / d_grid.shape[0])**3

            # --- load BoA file ---
            boa_file = Path(base_path) / f"vel_s{rs}_{snapshot}_C256_forward_8_BoA.npy"
            boa = np.swapaxes(np.load(boa_file), 0, 2)

            # --- compute cumulative mass ---
            masses, cum_mass = compute_cumulative_mass(boa, mass)

            # --- plot (only add legend at z=0) ---
            lw = 1
            if snapshot == '088':
                ax.plot(masses, cum_mass, color=color, lw=lw, label=rs)
            else:
                ax.plot(masses, cum_mass, color=color, lw=lw)

    ax.legend(title=r"$R_\mathrm{smooth}$", loc='best')
    ax.set_xlim([1e14, 1e19])
    ax.set_ylim([0, 1])
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_\mathrm{basin}$ (M$_\odot/h$)")
    ax.set_ylabel("Cumulative sum mass of basins")

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Cumulative mass figure saved to {output_path}")