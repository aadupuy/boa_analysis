"""
mass_function.py
----------------

Compute and plot the cumulative mass function of basins of attraction
(number of basins with mass < M) across redshifts and smoothing scales.

Functions
---------
compute_mass_function(boa, mass)
    Compute cumulative mass function for a given BoA segmentation.

plot_mass_function_allz(snapshots, tab_rs, sav_path, base_path, output_path)
    Plot cumulative mass function across all redshifts and smoothing scales.
"""

import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
from pathlib import Path


def compute_mass_function(boa: np.ndarray, mass: np.ndarray):
    """
    Compute the cumulative mass function N(<M) for basins of attraction.

    Parameters
    ----------
    boa : ndarray
        3D BoA segmentation map.
    mass : ndarray
        3D mass density field.

    Returns
    -------
    masses : ndarray
        Sorted basin masses.
    number : ndarray
        Cumulative number of basins with mass < M.
    """
    indices = np.unique(boa)[1:]  # exclude ID=0 (background)
    basin_masses = np.array([np.sum(mass[boa == i]) for i in indices])
    basin_masses.sort()
    number = np.arange(1, basin_masses.size + 1)
    return basin_masses, number


def plot_mass_function_allz(
    snapshots=None,
    tab_rs=None,
    sav_path: str = "/path/to/density/field/sav/",
    base_path: str = "segmentation_data/npy",
    output_path: str = "mass_function_allz.png",
):
    """
    Plot the cumulative mass function of basins for multiple redshifts and smoothing scales.

    Parameters
    ----------
    snapshots : list of str
        Snapshot identifiers.
    tab_rs : list of str
        List of smoothing radii.
    sav_path : str
        Directory containing .sav density grids.
    base_path : str
        Directory containing BoA .npy segmentation maps.
    output_path : str
        Path to save the resulting plot.
    """

    # --- constants ---
    L = 400.0  # Mpc/h
    Npart = 3840**3
    V = L**3
    mass_part = 9.4e7
    mean_rho = Npart / V * mass_part

    if snapshots is None:
        snapshots = ['015','016','017','018','019','027','042','050','057','067','088']
    if tab_rs is None:
        tab_rs = ['1.50', '3.00', '5.00', '7.50', '10.0', '12.5', '15.0']

    colors = ['red','orange','yellow','limegreen','cyan','blue','indigo']
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(6, 6))

    for rs, color in zip(tab_rs, colors):
        print(f"▶ Processing smoothing radius {rs}...")
        for snapshot in snapshots:
            print(f"   ↳ Snapshot {snapshot}")

            # --- load density grid ---
            sav_file = Path(sav_path) / f"vel_s{rs}_{snapshot}_C256.sav"
            sav_data = readsav(sav_file, verbose=False)
            if hasattr(sav_data, "vel_s1_field"):
                d_grid = sav_data.vel_s1_field.d[0]
            elif hasattr(sav_data, "d"):
                d_grid = sav_data.d[0]
            else:
                raise KeyError(f"Could not find density field in {sav_file}")
            mass = d_grid * mean_rho * (L / d_grid.shape[0]) ** 3

            # --- load BoA segmentation ---
            boa_file = Path(base_path) / f"vel_s{rs}_{snapshot}_C256_forward_8_BoA.npy"
            boa = np.swapaxes(np.load(boa_file), 0, 2)

            # --- compute cumulative mass function ---
            masses, number = compute_mass_function(boa, mass)

            # --- plot only z=0 in legend for clarity ---
            if snapshot == '088':
                ax.plot(masses, number, color=color, lw=1, label=rs)
            else:
                ax.plot(masses, number, color=color, lw=1)

    # --- formatting ---
    ax.legend(title=r"$R_\mathrm{smooth}$", loc='best')
    ax.set_xlim([1e14, 1e19])
    ax.set_ylim([0, 800])
    ax.set_xscale('log')
    ax.set_xlabel(r"$M_\mathrm{basin}$ (M$_\odot/h$)")
    ax.set_ylabel(r"Number of basins with $M < M_\mathrm{basin}$")

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Mass function figure saved to {output_path}")