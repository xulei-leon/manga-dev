from bdb import effective
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

# calculate ratio of conversion between arcsec and kpc/h
def calc_ratio_r(r_arcsec: np.ndarray, r_h_kpc: np.ndarray) -> float:
    r_arcsec = np.asarray(r_arcsec)
    r_h_kpc = np.asarray(r_h_kpc)
    if r_arcsec.shape != r_h_kpc.shape:
        raise ValueError("r_arcsec and r_h_kpc must have the same shape")
    # avoid division by zero
    r_h_kpc = np.where(r_h_kpc == 0, np.nan, r_h_kpc)
    # skip negative or NaN entries
    r_arcsec = np.where((r_arcsec < 0) | (np.isnan(r_arcsec)), np.nan, r_arcsec)
    r_h_kpc = np.where((r_h_kpc < 0) | (np.isnan(r_h_kpc)), np.nan, r_h_kpc)
    ratio_r =  r_h_kpc / r_arcsec

    ratio = np.median(ratio_r[~np.isnan(ratio_r)]) # return mean of valid entries
    return ratio

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

# Orbital Velocity Formula
# V(r) ^ 2 = G * M(r) / r
def calc_velocity_r(mass_r: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Calculate the orbital velocity at radius r given cumulative mass M(r)."""
    mass_r = np.asarray(mass_r)
    radius = np.asarray(radius)
    if mass_r.shape != radius.shape:
        raise ValueError("mass_r and radius must have the same shape")
    # avoid division by zero
    radius = np.where(radius == 0, np.nan, radius)
    # skip negative mass entries
    mass_r = np.where(mass_r < 0, np.nan, mass_r)

    # mass unit: M_sun
    # radius unit: kpc/h
    G = 4.302e-6  # kpc km^2 / (M_sun s^2)

    # unit of velocity: km/s
    velocity = np.sqrt(G * mass_r / radius)
    return velocity

def main():
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)

    # binid = firefly_util.get_binid(PLATE_IFU)
    # print(f"Bin ID shape: {binid.shape}, bin ID valid size: {np.sum(binid>=0)}")

    ########################################################
    ## 1. calculate stellar M(r)
    ########################################################
    print("")
    print("#######################################################")
    print("# 1. calculate stellar M(r)")
    print("#######################################################")

    mass_stellar_cell, mass_stellar_cell_err = firefly_util.get_stellar_mass_cell(PLATE_IFU)
    print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Stellar Mass (FIREFLY) total: {np.nansum(mass_stellar_cell):,.1f} M solar")

    # density_stellar, density_stellar_err = firefly_util.get_stellar_density(PLATE_IFU)
    # print(f"Stellar Surface Mass Density shape: {density_stellar.shape}, min: {np.nanmin(density_stellar[density_stellar>=0]):.3f}, max: {np.nanmax(density_stellar):,.1f} M solar/kpc^2")

    _radius_eff, azimuth = firefly_util.get_radius_eff(PLATE_IFU)
    print(f"Radius Eff shape: {_radius_eff.shape}, Radius (eff) min: {np.nanmin(_radius_eff[azimuth>=0]):.3f}, max: {np.nanmax(_radius_eff):.3f}")
    print(f"Azimuth shape: {azimuth.shape}, Azimuth (degrees) min: {np.nanmin(azimuth[azimuth>=0]):.3f}, max: {np.nanmax(azimuth):.3f}")

    mass_map, radius_eff_map = calc_mass_r(mass_stellar_cell, _radius_eff)
    print(f"Cumulative Mass shape: {mass_map.shape}, min: {np.nanmin(mass_map):.3f}, max: {np.nanmax(mass_map):,.1f} M solar")
    print(f"Radius bins shape: {radius_eff_map.shape}, min: {np.nanmin(radius_eff_map):.3f}, max: {np.nanmax(radius_eff_map):.3f} eff")

    total_stellar_mass_1, total_stellar_mass_2 = drpall_util.get_stellar_mass(PLATE_IFU)
    print("Verification with DRPALL stellar mass:")
    print(f"Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")

    ########################################################
    ## 2. calculate stellar r
    ########################################################

    print("")
    print("#######################################################")
    print("# 2. calculate stellar r")
    print("#######################################################")

    # get data from DRPALL and maps file
    effective_radius = drpall_util.get_effective_radius(PLATE_IFU)
    print(f"Effective Radius (DRPALL): {effective_radius:.3f} arcsec")

    # convert radius map from effective radius to arcsec
    radius_arcsec = radius_eff_map * effective_radius
    print(f"Radius (arcsec) shape: {radius_arcsec.shape}, min: {np.nanmin(radius_arcsec[radius_arcsec>=0]):.3f}, max: {np.nanmax(radius_arcsec):.3f}")

    # calculate ratio of conversion between arcsec and kpc/h
    _r_arcsec_map, _r_h_kpc_map, _ = maps_util.get_radius_map()
    print(f"Radius (MAPS) shape: {_r_arcsec_map.shape}, r_h_kpc shape: {_r_h_kpc_map.shape}")
    print(f"Radius (MAPS) min: {np.nanmin(_r_arcsec_map[_r_arcsec_map>=0]):.3f}, max: {np.nanmax(_r_arcsec_map):.3f} eff")

    ratio_r = calc_ratio_r(_r_arcsec_map, _r_h_kpc_map)
    print(f"Radius ratio (arcsec/h_kpc): {ratio_r:.3f} arcsec/h_kpc")

    # covert arcsec to kpc/h
    radius_h_kpc = radius_arcsec * ratio_r
    print(f"Radius (kpc) shape: {radius_h_kpc.shape}, min: {np.nanmin(radius_h_kpc[radius_h_kpc>=0]):.3f}, max: {np.nanmax(radius_h_kpc):.3f}")

    ########################################################
    ## 3. calculate stellar rotation velocity V(r)
    ########################################################
    print("")
    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    vel_r = calc_velocity_r(mass_map, radius_h_kpc)
    print(f"Velocity shape: {vel_r.shape}, min: {np.nanmin(vel_r):.3f}, max: {np.nanmax(vel_r):,.1f} km/s")

    return

if __name__ == "__main__":
    main()

