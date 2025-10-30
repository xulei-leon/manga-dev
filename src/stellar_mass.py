from bdb import effective
from operator import le
from pathlib import Path
from tkinter import N

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


class StellarMass:
    drpall_util = None
    firefly_util = None
    maps_util = None

    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util

    @staticmethod
    def calc_ratio_r(r_arcsec: np.ndarray, r_h_kpc: np.ndarray) -> float:
        r_arcsec = np.asarray(r_arcsec)
        r_h_kpc = np.asarray(r_h_kpc)
        if r_arcsec.shape != r_h_kpc.shape:
            raise ValueError("r_arcsec and r_h_kpc must have the same shape")
        r_h_kpc = np.where(r_h_kpc == 0, np.nan, r_h_kpc)
        r_arcsec = np.where((r_arcsec < 0) | (np.isnan(r_arcsec)), np.nan, r_arcsec)
        r_h_kpc = np.where((r_h_kpc < 0) | (np.isnan(r_h_kpc)), np.nan, r_h_kpc)
        ratio_r = r_h_kpc / r_arcsec
        ratio = np.median(ratio_r[~np.isnan(ratio_r)])
        return ratio

    @staticmethod
    def calc_mass_r(mass_cell: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mass_cell = np.asarray(mass_cell)
        radius = np.asarray(radius)

        if mass_cell.shape != radius.shape:
            raise ValueError("mass_cell and radius must have the same shape")

        valid_mask = (~np.isnan(mass_cell)) & (~np.isnan(radius)) & (mass_cell >= 0) & (radius >= 0)

        if not np.any(valid_mask):
            return np.array([], dtype=float), np.array([], dtype=float)

        filtered_mass = mass_cell[valid_mask]
        filtered_radius = radius[valid_mask]

        sorted_idx = np.argsort(filtered_radius)

        sorted_radius = filtered_radius[sorted_idx]
        sorted_masses = filtered_mass[sorted_idx]

        mass_r = np.cumsum(sorted_masses)
        r_bins = sorted_radius

        return mass_r, r_bins

    @staticmethod
    def calc_velocity_r(mass_r: np.ndarray, radius: np.ndarray) -> np.ndarray:
        mass_r = np.asarray(mass_r)
        radius = np.asarray(radius)
        if mass_r.shape != radius.shape:
            raise ValueError("mass_r and radius must have the same shape")
        radius = np.where(radius == 0, np.nan, radius)
        mass_r = np.where(mass_r < 0, np.nan, mass_r)

        G = 4.302e-6

        velocity = np.sqrt(G * mass_r / radius)
        return velocity

    def calc_stellar_vel(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        print("")
        print("#######################################################")
        print("# 1. calculate stellar M(r)")
        print("#######################################################")

        mass_stellar_cell, mass_stellar_cell_err = self.firefly_util.get_stellar_mass_cell(PLATE_IFU)
        print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Stellar Mass (FIREFLY) total: {np.nansum(mass_stellar_cell):,.1f} M solar")

        _radius_eff, azimuth = self.firefly_util.get_radius_eff(PLATE_IFU)
        print(f"Radius Eff shape: {_radius_eff.shape}, Radius (eff) min: {np.nanmin(_radius_eff[azimuth>=0]):.3f}, max: {np.nanmax(_radius_eff):.3f}")
        print(f"Azimuth shape: {azimuth.shape}, Azimuth (degrees) min: {np.nanmin(azimuth[azimuth>=0]):.3f}, max: {np.nanmax(azimuth):.3f}")

        mass_map, radius_eff_map = self.calc_mass_r(mass_stellar_cell, _radius_eff)
        print(f"Cumulative Mass shape: {mass_map.shape}, min: {np.nanmin(mass_map):.3f}, max: {np.nanmax(mass_map):,.1f} M solar")
        print(f"Radius bins shape: {radius_eff_map.shape}, min: {np.nanmin(radius_eff_map):.3f}, max: {np.nanmax(radius_eff_map):.3f} eff")

        total_stellar_mass_1, total_stellar_mass_2 = self.drpall_util.get_stellar_mass(PLATE_IFU)
        print("Verification with DRPALL stellar mass:")
        print(f"Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")

        print("")
        print("#######################################################")
        print("# 2. calculate stellar r")
        print("#######################################################")

        effective_radius = self.drpall_util.get_effective_radius(PLATE_IFU)
        print(f"Effective Radius (DRPALL): {effective_radius:.3f} arcsec")

        radius_arcsec = radius_eff_map * effective_radius
        print(f"Radius (arcsec) shape: {radius_arcsec.shape}, min: {np.nanmin(radius_arcsec[radius_arcsec>=0]):.3f}, max: {np.nanmax(radius_arcsec):.3f}")

        _r_arcsec_map, _r_h_kpc_map, _ = self.maps_util.get_radius_map()
        print(f"Radius (MAPS) shape: {_r_arcsec_map.shape}, r_h_kpc shape: {_r_h_kpc_map.shape}")
        print(f"Radius (MAPS) min: {np.nanmin(_r_arcsec_map[_r_arcsec_map>=0]):.3f}, max: {np.nanmax(_r_arcsec_map):.3f} eff")

        ratio_r = self.calc_ratio_r(_r_arcsec_map, _r_h_kpc_map)
        print(f"Radius ratio (arcsec/h_kpc): {ratio_r:.3f} arcsec/h_kpc")

        radius_h_kpc = radius_arcsec * ratio_r
        print(f"Radius (kpc) shape: {radius_h_kpc.shape}, min: {np.nanmin(radius_h_kpc[radius_h_kpc>=0]):.3f}, max: {np.nanmax(radius_h_kpc):.3f}")

        print("")
        print("#######################################################")
        print("# 3. calculate stellar rotation velocity V(r)")
        print("#######################################################")
        vel_r = self.calc_velocity_r(mass_map, radius_h_kpc)
        print(f"Velocity shape: {vel_r.shape}, min: {np.nanmin(vel_r):.3f}, max: {np.nanmax(vel_r):,.1f} km/s")

        return vel_r, radius_h_kpc


def main() -> None:
    PLATE_IFU = "8723-12703"

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)

    _, _ = StellarMass(drpall_util, firefly_util, maps_util).calc_stellar_vel(PLATE_IFU)

if __name__ == "__main__":
    main()
