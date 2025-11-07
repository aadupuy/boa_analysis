"""
meandensity_rs_snapshots.py
---------------------------

Compute and plot the weighted median basin density (ρ/ρ̄) as a function of
smoothing radius R_s across multiple redshifts.

Functions
---------
weighted_quantile(values, quantiles, sample_weight, values_sorted=False)
    Compute weighted quantiles with sample weights.

compute_mean_density(snapshot, tab_rs, sav_path, base_path, mean_rho, L)
    Compute median, lower, and upper density quantiles for one snapshot.

plot_mean_density_vs_rs(...)
    Plot weighted mean density vs. smoothing radius for multiple redshifts.
"""

import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from pathlib import Path


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False):
    """
    Compute weighted quantiles (similar to np.percentile, but with weights).

    Parameters
    ----------
    values : ndarray
        Data values.
    quantiles : float or array-like
        Desired quantile(s) in [0, 1].
    sample_weight : ndarray
        Weights for each data point.
    values_sorted : bool
        If True, assumes 'values' are already sorted.

    Returns
    -------
    quantile_values : ndarray
        Weighted quantile(s).
    """
    values = np.asarray(values)
    quantiles = np.atleast_1d(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.asarray(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles must be in [0,1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_cdf, values)


def compute_mean_density(snapshot: str, tab_rs, sav_path, base_path, mean_rho: float, L: float):
    """
    Compute weighted median basin density and its 16–84% range for one snapshot.

    Parameters
    ----------
    snapshot : str
        Snapshot identifier (e.g. '088').
    tab_rs : list of str
        List of smoothing radii (as strings).
    sav_path : str
        Directory containing .sav density grids.
    base_path : str
        Directory containing BoA .npy segmentation files.
    mean_rho : float
        Mean density (M_sun / (Mpc/h)^3).
    L : float
        Box size (Mpc/h).

    Returns
    -------
    median, lower_err, upper_err : np.ndarray
        Arrays of shape (len(tab_rs),).
    """
    median = np.zeros(len(tab_rs))
    lower_err = np.zeros(len(tab_rs))
    upper_err = np.zeros(len(tab_rs))

    for i, rs in enumerate(tab_rs):
        sav_file = Path(sav_path) / f"vel_s{rs}_{snapshot}_C256.sav"
        sav_data = readsav(sav_file, verbose=False)
        if hasattr(sav_data, "vel_s1_field"):
            d_grid = sav_data.vel_s1_field.d[0]
        elif hasattr(sav_data, "d"):
            d_grid = sav_data.d[0]
        else:
            raise KeyError(f"Density field not found in {sav_file}")

        mass = d_grid * mean_rho * (L / d_grid.shape[0]) ** 3
        boa_file = Path(base_path) / f"vel_s{rs}_{snapshot}_C256_forward_8_BoA.npy"
        boa = np.swapaxes(np.load(boa_file), 0, 2)

        indices = np.unique(boa)[1:]  # exclude background (0)
        dens, vol = np.zeros(len(indices)), np.zeros(len(indices))

        for j, idx in enumerate(indices):
            volume = np.count_nonzero(boa == idx) * (L / boa.shape[0]) ** 3
            dens[j] = np.sum(mass[boa == idx]) / volume
            vol[j] = volume

        dens /= mean_rho  # normalize by mean density
        med = np.median(dens)
        q16 = weighted_quantile(dens, 0.16, sample_weight=vol)
        q84 = weighted_quantile(dens, 0.84, sample_weight=vol)

        median[i] = med
        lower_err[i] = med - q16
        upper_err[i] = q84 - med

    return median, lower_err, upper_err


def plot_mean_density_vs_rs(
    snapshots=None,
    redshifts=None,
    tab_rs=None,
    sav_path: str = "/path/to/density/field/sav/",
    base_path: str = "segmentation_data/npy",
    data_dir: str = "data",
    output_path: str = "fig/meandens_rs_snapshots_BOA.png",
):
    """
    Plot weighted median basin density (ρ/ρ̄) vs. smoothing radius across redshifts.

    Parameters
    ----------
    snapshots : list of str
        Snapshot identifiers.
    redshifts : list of float
        Redshift values corresponding to snapshots.
    tab_rs : list of str
        Smoothing radii (as strings).
    sav_path : str
        Path to .sav density fields.
    base_path : str
        Path to BoA .npy segmentation fields.
    data_dir : str
        Directory for saving/loading computed data.
    output_path : str
        Output path for figure.
    """

    # --- constants ---
    L = 400.0
    Npart = 3840**3
    V = L**3
    mass_part = 9.4e7
    mean_rho = Npart * mass_part / V

    if snapshots is None:
        snapshots = ['015', '016', '017', '018', '019', '027', '042', '050', '057', '067', '088']
    if redshifts is None:
        redshifts = [2.89, 2.48, 2.14, 1.44, 1.00, 0.74, 0.51, 0.40, 0.29, 0.20, 0]
    if tab_rs is None:
        tab_rs = ['1.50', '3.00', '5.00', '7.50', '10.0', '12.5', '15.0']

    Path(data_dir).mkdir(exist_ok=True)
    save_median = Path(data_dir) / "median_meandens.npy"
    save_lower = Path(data_dir) / "lower_error_meandens.npy"
    save_upper = Path(data_dir) / "upper_error_meandens.npy"

    # --- load or compute ---
    if not (save_median.exists() and save_lower.exists() and save_upper.exists()):
        print("🔄 Computing mean densities...")
        med_arr = np.zeros((len(snapshots), len(tab_rs)))
        low_arr = np.zeros_like(med_arr)
        up_arr = np.zeros_like(med_arr)

        for s, snapshot in enumerate(snapshots):
            print(f"Processing snapshot {snapshot}...")
            med, low, up = compute_mean_density(snapshot, tab_rs, sav_path, base_path, mean_rho, L)
            med_arr[s, :] = med
            low_arr[s, :] = low
            up_arr[s, :] = up

        np.save(save_median, med_arr)
        np.save(save_lower, low_arr)
        np.save(save_upper, up_arr)
        print("✅ Mean density arrays saved.")
    else:
        med_arr = np.load(save_median)
        low_arr = np.load(save_lower)
        up_arr = np.load(save_upper)

    # --- plot ---
    jet = plt.get_cmap("jet")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(snapshots))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(6, 6))
    rs_values = np.array([float(r) for r in tab_rs])

    for i, z in enumerate(redshifts):
        color = scalarMap.to_rgba(i)
        med, low, up = med_arr[i, :], low_arr[i, :], up_arr[i, :]
        ax.errorbar(
            rs_values, med, yerr=[low, up],
            color=color, fmt="s", markersize=5, capsize=3,
            label=fr"$z = {z:.2f}$"
        )

    ax.axhline(1, lw=0.75, ls="--", color="black")
    ax.legend(loc="best", ncol=2)
    ax.set_xlim([0, 16])
    ax.set_ylim([0.98, 1.04])
    ax.set_xlabel(r"$R_\mathrm{s}$ (Mpc/$h$)")
    ax.set_ylabel(r"$\rho / \bar{\rho}$")

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"✅ Mean density figure saved to {output_path}")