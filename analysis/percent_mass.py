"""
percent_mass.py
----------------

Compute and plot 5th, 50th (median), and 95th percentile basin masses as a
function of smoothing radius across redshifts.

Functions
---------
compute_percentile_masses(snapshot, tab_rs, sav_path, base_path)
    Compute percentile masses (5, 50, 95%) for one snapshot.

plot_percent_mass(tab_rs, snapshots, base_path, sav_path, data_dir, output_path)
    Plot percentile mass vs. smoothing radius for all snapshots.
"""

import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
from pathlib import Path


def compute_percentile_masses(snapshot: str, tab_rs, sav_path: str, base_path: str, mean_rho: float, L: float):
    """
    Compute 5th, 50th, and 95th percentile basin masses for one snapshot.

    Parameters
    ----------
    snapshot : str
        Snapshot identifier (e.g. '088').
    tab_rs : list of str
        List of smoothing radii (e.g. ['1.50', '3.00', ...]).
    sav_path : str
        Directory containing .sav density grids.
    base_path : str
        Directory containing BoA .npy files.
    mean_rho : float
        Mean mass density.
    L : float
        Box length in Mpc/h.

    Returns
    -------
    min_p, med_p, max_p : np.ndarray
        5th, 50th, and 95th percentile basin masses per smoothing radius.
    """
    min_p, med_p, max_p = np.zeros(len(tab_rs)), np.zeros(len(tab_rs)), np.zeros(len(tab_rs))

    for i, rs in enumerate(tab_rs):
        # Load density grid
        sav_file = Path(sav_path) / f"vel_s{rs}_{snapshot}_C256.sav"
        sav_data = readsav(sav_file, verbose=False)
        if hasattr(sav_data, "vel_s1_field"):
            d_grid = sav_data.vel_s1_field.d[0]
        elif hasattr(sav_data, "d"):
            d_grid = sav_data.d[0]
        else:
            raise KeyError(f"Could not find density field in {sav_file}")
        mass = d_grid * mean_rho * (L / d_grid.shape[0]) ** 3

        # Load BoA segmentation
        boa_file = Path(base_path) / f"vel_s{rs}_{snapshot}_C256_forward_8_BoA.npy"
        boa = np.swapaxes(np.load(boa_file), 0, 2)

        # Compute basin masses
        indices = np.unique(boa)[1:]  # skip ID=0
        masses = np.array([np.sum(mass[boa == idx]) for idx in indices])
        masses.sort()

        # Compute normalized cumulative distribution
        cum = np.cumsum(masses)
        cum = (cum - cum.min()) / (cum.max() - cum.min())

        # Extract percentiles
        med_p[i] = np.interp(0.5, cum, masses)
        min_p[i] = np.interp(0.05, cum, masses)
        max_p[i] = np.interp(0.95, cum, masses)

    return min_p, med_p, max_p


def plot_percent_mass(
    tab_rs=None,
    snapshots=None,
    sav_path: str = "/path/to/density/field/sav/",
    base_path: str = "/z/adupuy/Software/lss_finder_mpi/output_redshift/npy",
    data_dir: str = "data",
    output_path: str = "fig/percent_mass.png",
):
    """
    Plot 5th, 50th, and 95th percentile basin masses vs. smoothing radius.

    Parameters
    ----------
    tab_rs : list of float
        List of smoothing radii (default: [1.5, 3, 5, 7.5, 10, 12.5, 15]).
    snapshots : list of str
        Snapshot identifiers.
    sav_path : str
        Directory containing .sav density fields.
    base_path : str
        Directory containing BoA .npy files.
    data_dir : str
        Directory where precomputed percentile arrays are stored or saved.
    output_path : str
        Path to save the resulting plot.
    """

    # --- constants ---
    L = 400.0
    Npart = 3840**3
    V = L**3
    mass_part = 9.4e7
    mean_rho = Npart / V * mass_part

    if tab_rs is None:
        tab_rs = np.array([1.5, 3, 5, 7.5, 10, 12.5, 15])
    if snapshots is None:
        snapshots = ['015','016','017','018','019','027','042','050','057','067','088']

    Path(data_dir).mkdir(exist_ok=True)

    save_min_file = Path(data_dir) / "mass_min_5.npy"
    save_med_file = Path(data_dir) / "mass_median_50.npy"
    save_max_file = Path(data_dir) / "mass_max_95.npy"

    # --- compute or load ---
    if not (save_min_file.exists() and save_med_file.exists() and save_max_file.exists()):
        print("🔄 Computing percentile masses (this may take a while)...")
        save_min, save_med, save_max = np.zeros((len(snapshots), len(tab_rs))), np.zeros((len(snapshots), len(tab_rs))), np.zeros((len(snapshots), len(tab_rs)))

        for s, snapshot in enumerate(snapshots):
            print(f"Processing snapshot {snapshot}...")
            min_p, med_p, max_p = compute_percentile_masses(snapshot, [f"{r:.2f}" for r in tab_rs], sav_path, base_path, mean_rho, L)
            save_min[s, :] = min_p
            save_med[s, :] = med_p
            save_max[s, :] = max_p

        np.save(save_min_file, save_min)
        np.save(save_med_file, save_med)
        np.save(save_max_file, save_max)
        print("✅ Saved computed percentile arrays.")
    else:
        save_min = np.load(save_min_file)
        save_med = np.load(save_med_file)
        save_max = np.load(save_max_file)

    # --- plot ---
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(save_med.shape[0]):
        if i == 0:
            ax.plot(tab_rs, save_max[i, :], color='green', lw=0.5, label='95%')
            ax.plot(tab_rs, save_med[i, :], color='black', lw=0.5, label='50%')
            ax.plot(tab_rs, save_min[i, :], color='blue', lw=0.5, label='5%')
        else:
            ax.plot(tab_rs, save_med[i, :], color='black', lw=0.5)
            ax.plot(tab_rs, save_min[i, :], color='blue', lw=0.5)
            ax.plot(tab_rs, save_max[i, :], color='green', lw=0.5)

    ax.set_yscale('log')
    ax.set_ylim([1e15, 3e18])
    ax.set_xlim([1.5, 15])
    ax.legend(loc='best')
    ax.set_ylabel(r"$M_\mathrm{basin}$ (M$_\odot/h$)")
    ax.set_xlabel(r"$r_s$")

    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ Percentile mass figure saved to {output_path}")