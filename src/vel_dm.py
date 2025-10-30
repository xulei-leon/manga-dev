from operator import le
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit


from matplotlib import colors
from scipy.special import gamma, gammainc

# my imports
from util.fits_util import FitsUtil
from util.drpall_util import DrpallUtil
from util.firefly_util import FireflyUtil
from util.maps_util import MapsUtil

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


# PLATE_IFU = "8723-12705"
PLATE_IFU = "8723-12703"

# Gravitational constant in kpc (km/s)^2 / Msun
G = 4.30091727003628e-6 # kpc (km/s)^2 / Msun

# Sersic b_n approximation
def b_n(n):
    return 2.0*n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n)


# Orbital Velocity Formula
# V(r) ^ 2 = G * M(r) / r


# M_cumulative(r) = SUM( M_cell,i ) for all Voronoi cells 'i' where Radius_i <= r
def calc_mass_r(mass_cell: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the cumulative mass within radius r, ignoring negative or NaN entries."""
    mass_cell = np.asarray(mass_cell)
    radius = np.asarray(radius)

    if mass_cell.shape != radius.shape:
        raise ValueError("mass_cell and radius must have the same shape")

    # mask out negative values and NaNs so they don't participate in the computation
    valid_mask = (~np.isnan(mass_cell)) & (~np.isnan(radius)) & (mass_cell >= 0) & (radius >= 0)

    if not np.any(valid_mask):
        # No valid data points, return empty arrays
        return np.array([], dtype=float), np.array([], dtype=float)

    filtered_mass = mass_cell[valid_mask]
    filtered_radius = radius[valid_mask]

    sorted_idx = np.argsort(filtered_radius)

    sorted_radius = filtered_radius[sorted_idx]
    sorted_masses = filtered_mass[sorted_idx]

    mass_r = np.cumsum(sorted_masses)
    r_bins = sorted_radius

    return mass_r, r_bins


def main():
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)

    binid = firefly_util.get_binid(PLATE_IFU)
    print(f"Bin ID shape: {binid.shape}, bin ID valid size: {np.sum(binid>=0)}")

    total_stellar_mass_1, total_stellar_mass_2 = drpall_util.get_stellar_mass(PLATE_IFU)
    print(f"Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")

    mass_stellar_cell, mass_stellar_cell_err = firefly_util.get_stellar_mass_cell(PLATE_IFU)
    print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Stellar Mass (FIREFLY) total: {np.nansum(mass_stellar_cell):,.1f} M solar")

    density_stellar, density_stellar_err = firefly_util.get_stellar_density(PLATE_IFU)
    print(f"Stellar Surface Mass Density shape: {density_stellar.shape}, min: {np.nanmin(density_stellar[density_stellar>=0]):.3f}, max: {np.nanmax(density_stellar):,.1f} M solar/kpc^2")

    radius, azimuth = firefly_util.get_radius_eff(PLATE_IFU)
    print(f"Radius shape: {radius.shape}, Radius (eff) min: {np.nanmin(radius[azimuth>=0]):.3f}, max: {np.nanmax(radius):.3f}")
    print(f"Azimuth shape: {azimuth.shape}, Azimuth (degrees) min: {np.nanmin(azimuth[azimuth>=0]):.3f}, max: {np.nanmax(azimuth):.3f}")

    # radius, r_h_kpc, azimuth = maps_util.get_radius_map()
    # print(f"Radius shape: {radius.shape}, r_h_kpc shape: {r_h_kpc.shape}, azimuth shape: {azimuth.shape}")

    mass_r, r_bins = calc_mass_r(mass_stellar_cell, radius)
    print(f"Cumulative Mass shape: {mass_r.shape}, min: {np.nanmin(mass_r):.3f}, max: {np.nanmax(mass_r):,.1f} M solar")
    print(f"Radius bins shape: {r_bins.shape}, min: {np.nanmin(r_bins):.3f}, max: {np.nanmax(r_bins):.3f} eff")

    return

if __name__ == "__main__":
    main()

