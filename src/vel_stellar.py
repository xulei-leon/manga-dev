from bdb import effective
from operator import le
from pathlib import Path
from tkinter import N

from deprecated import deprecated
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
import scipy.special as special
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


from matplotlib import colors

# my imports
from util.fits_util import FitsUtil
from util.drpall_util import DrpallUtil
from util.firefly_util import FireflyUtil
from util.maps_util import MapsUtil
from util.plot_util import PlotUtil

RADIUS_MIN_KPC = 0.1  # kpc
G = const.G.to('kpc km2 / (Msun s2)').value  # gravitational constant in kpc km^2 / (M_sun s^2)

class Stellar:
    drpall_util = None
    firefly_util = None
    maps_util = None

    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util

    @staticmethod
    def calc_r_ratio_to_h_kpc(r_arcsec: np.ndarray, r_h_kpc: np.ndarray) -> float:
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

    def _calc_radius_to_h_kpc(self, PLATE_IFU, radius_eff_map):
        _r_arcsec_map, _r_h_kpc_map, _ = self.maps_util.get_radius_map()
        print(f"Radius (MAPS) shape: {_r_arcsec_map.shape}, Unit: arcsec, range [{np.nanmin(_r_arcsec_map[_r_arcsec_map>=0]):.3f}, {np.nanmax(_r_arcsec_map):.3f}]")
        print(f"Radius (MAPS) shape: {_r_h_kpc_map.shape}, Unit: kpc/h, range [{np.nanmin(_r_h_kpc_map[_r_h_kpc_map>=0]):.3f}, {np.nanmax(_r_h_kpc_map):.3f}]")

        effective_radius = self.drpall_util.get_effective_radius(PLATE_IFU)
        print(f"Effective Radius (DRPALL): {effective_radius:.3f} arcsec")

        radius_arcsec_map = radius_eff_map * effective_radius
        print(f"Radius (arcsec) shape: {radius_arcsec_map.shape}, Unit: arcsec, range [{np.nanmin(radius_arcsec_map[radius_arcsec_map>=0]):.3f}, {np.nanmax(radius_arcsec_map):.3f}]")

        _r_err_percent = np.abs(np.nanmax(radius_arcsec_map) - np.nanmax(_r_arcsec_map)) / np.nanmax(_r_arcsec_map)
        print(f"  Verification with MAPS radius:")
        if np.nanmax(_r_err_percent) < 0.01:
            print(f"  Calculated radius matches MAPS radius within {np.nanmax(_r_err_percent):.1%}")
        else:
            print("  WARNING: Calculated radius does not match MAPS radius within 1%")

        ratio_r = self.calc_r_ratio_to_h_kpc(_r_arcsec_map, _r_h_kpc_map)
        print(f"Radius ratio (arcsec to kpc/h): {ratio_r:.3f}")

        radius_h_kpc_map = radius_arcsec_map * ratio_r
        print(f"Radius (kpc/h) shape: {radius_h_kpc_map.shape}, Unit: kpc/h, range [min: {np.nanmin(radius_h_kpc_map[radius_h_kpc_map>=0]):.3f}, max: {np.nanmax(radius_h_kpc_map):.3f}]")
        _r_err_percent = np.abs(np.nanmax(radius_h_kpc_map) - np.nanmax(_r_h_kpc_map)) / np.nanmax(_r_h_kpc_map)
        print(f"  Verification with MAPS radius:")
        if np.nanmax(_r_err_percent) < 0.01:
            print(f"  Calculated radius matches MAPS radius within {np.nanmax(_r_err_percent):.1%}")
        else:
            print("  WARNING: Calculated radius does not match MAPS radius within 1%")
        return radius_h_kpc_map

    ################################################################################
    # Stellar Mass M(r)
    # Unused for now
    ################################################################################

    @staticmethod
    def _calc_mass_of_radius(mass_cell: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    @deprecated(reason="To use Mass Density method instead", version="0.1.0")
    def _get_stellar_mass(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        print("")
        print("#######################################################")
        print("# 1. calculate stellar M(r)")
        print("#######################################################")

        mass_stellar_cell, mass_stellar_cell_err = self.firefly_util.get_stellar_mass_cell(PLATE_IFU)
        print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Unit: M solar, total: {np.nansum(mass_stellar_cell):,.1f} M solar")

        # This mass use h = 1
        total_stellar_mass_1, total_stellar_mass_2 = self.drpall_util.get_stellar_mass(PLATE_IFU)
        print("Verification with DRPALL stellar mass:")
        _mass_err2_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_2) / total_stellar_mass_2
        _mass_err1_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_1) / total_stellar_mass_1
        if abs(_mass_err2_percent) < 0.03:
            print(f"  FIREFLY total stellar mass matches DRPALL Elpetro stellar mass within {abs(_mass_err2_percent):.1%}")
        elif abs(_mass_err1_percent) < 0.03:
            print(f"  FIREFLY total stellar mass matches DRPALL Sersic stellar mass within {abs(_mass_err1_percent):.1%}")
        else:
            print("  WARNING: FIREFLY total stellar mass does not match DRPALL stellar masses within 3%")
            print(f"  Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")

        _radius_eff, azimuth = self.firefly_util.get_radius_eff(PLATE_IFU)
        print(f"Radius Eff shape: {_radius_eff.shape}, Unit: effective radius, range [{np.nanmin(_radius_eff[azimuth>=0]):.3f}, {np.nanmax(_radius_eff):.3f}]")
        print(f"Azimuth shape: {azimuth.shape}, Unit: degrees, range [{np.nanmin(azimuth[azimuth>=0]):.3f}, {np.nanmax(azimuth):.3f}]")

        mass_map, radius_eff_map = self._calc_mass_of_radius(mass_stellar_cell, _radius_eff)
        print(f"Cumulative Mass shape: {mass_map.shape}, Unit: M solar, range [{np.nanmin(mass_map):.3f}, {np.nanmax(mass_map):,.1f}]")
        print(f"Radius bins shape: {radius_eff_map.shape}, Unit: effective radius, range [{np.nanmin(radius_eff_map):.3f}, {np.nanmax(radius_eff_map):.3f}]")

        radius_h_kpc_map = self._calc_radius_to_h_kpc(PLATE_IFU, radius_eff_map)

        return radius_h_kpc_map, mass_map

    # Formula: M(r) = MB * r^2 / (r + a)^2 + MD * (1 - (1 + r / rd) * exp(-r / rd))
    def _stellar_mass_model(self, r: np.ndarray, MB: float, a: float, MD: float, rd: float) -> np.ndarray:
        bulge_mass = MB * np.square(r) / np.square(r + a)
        disk_mass = MD * (1 - (1 + r / rd) * np.exp(-r / rd))
        total_mass = bulge_mass + disk_mass
        return total_mass

    # used the minimum χ 2 method for fitting
    @deprecated(reason="To use Mass Density method instead", version="0.1.0")
    def _stellar_mass_fit(self, radius: np.ndarray, mass: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> np.ndarray:
        radius_filter = radius[radius>r_min]
        mass_filter = mass[radius>r_min]

        initial_guess = [1e10, 1.0, 1e10, 3.0]  # Initial guess for MB, a, MD, rd
        popt, pcov = curve_fit(self._stellar_mass_model, radius_filter, mass_filter, p0=initial_guess, maxfev=10000)
        fitted_mass = self._stellar_mass_model(radius_fitted, *popt)
        print(f"Fitted parameters: MB={popt[0]:.3e}, a={popt[1]:.3f}, MD={popt[2]:.3e}, rd={popt[3]:.3f}")
        return radius_fitted, fitted_mass

    ################################################################################
    # Mass Density Method
    ################################################################################

    # Exponential Disk rotation velocity formula
    # V^2(R) = 4 * pi * G * Sigma_0 * R_d * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    # where y = R / (2 * R_d)
    def __stellar_vel_sq(self, R: np.ndarray, Sigma_0: float, R_d: float) -> np.ndarray:
        y = R / (2.0 * R_d)
        I_0 = special.i0(y)
        I_1 = special.i1(y)
        K_0 = special.k0(y)
        K_1 = special.k1(y)

        V2 = 4.0 * np.pi * G * Sigma_0 * R_d * (np.square(y)) * (I_0 * K_0 - I_1 * K_1)
        return V2

    # Exponential Disk Model fitting function
    # Sigma(R) = Sigma_0 * exp(-R / R_d)
    def _stellar_density_model_ff(self, R: np.ndarray, Sigma_0: float, R_d: float) -> np.ndarray:
        return Sigma_0 * np.exp(-R / R_d)
    
    # Central Surface Mass Density Fitting
    def _stellar_central_density_fit(self, radius: np.ndarray, density: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius_filter = radius[radius>r_min]
        density_filter = density[radius>r_min]

        initial_guess = [1e8, 3.0]  # Initial guess for Sigma_0, R_d
        popt, pcov = curve_fit(self._stellar_density_model_ff, radius_filter, density_filter, p0=initial_guess, maxfev=10000)

        sigma_0_fitted, r_d_fitted = popt
        print(f"Fitted parameters: Sigma_0={popt[0]:.3e}, R_d={popt[1]:.3f}")
        return sigma_0_fitted, r_d_fitted

    def _calc_stellar_vel_sq(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        _radius_eff, _ = self.firefly_util.get_radius_eff(PLATE_IFU)
        _radius_h_kpc = self._calc_radius_to_h_kpc(PLATE_IFU, _radius_eff)

        _density, _ = self.firefly_util.get_stellar_density_cell(PLATE_IFU)
        print(f"Stellar Mass Density shape: {_density.shape}, Unit: M solar / kpc^2, range [{np.nanmin(_density[_density>=0]):.3f}, {np.nanmax(_density):,.1f}] M solar / kpc^2")

        sigma_0_fitted, r_d_fitted = self._stellar_central_density_fit(_radius_h_kpc, _density, r_min=RADIUS_MIN_KPC, radius_fitted=_radius_h_kpc)

        _vel_sq = self.__stellar_vel_sq(_radius_h_kpc, sigma_0_fitted, r_d_fitted)
        return _radius_h_kpc, _vel_sq, sigma_0_fitted, r_d_fitted
    
    ################################################################################
    # public methods
    ################################################################################
    def get_stellar_vel(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        r_map, vel_sq, _, _ = self._calc_stellar_vel_sq(PLATE_IFU)
        vel_map = np.sqrt(vel_sq)
        return r_map, vel_map
    
    def get_stellar_vel_sq(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        r_map, vel_sq, sigma_0, r_d = self._calc_stellar_vel_sq(PLATE_IFU)
        return r_map, vel_sq, sigma_0, r_d

    def get_stellar_density(self, radius: np.ndarray, sigma_0: float, r_d: float) -> np.ndarray:
        return self._stellar_density_model_ff(radius, sigma_0, r_d)



######################################################
# main function for test
######################################################
def main() -> None:
    PLATE_IFU = "8723-12705"

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)
    plot_util = PlotUtil(fits_util)

    stellar = Stellar(drpall_util, firefly_util, maps_util)

    r_map, vel_map = stellar.get_stellar_vel(PLATE_IFU)

    print("#######################################################")
    print("# calculate stellar rotation velocity V(r)")
    print("#######################################################")
    print(f"Calc Velocity shape: {vel_map.shape}, range: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}]")

    plot_util.plot_rv_curve(r_map, vel_map, title="Stellar")
    return

if __name__ == "__main__":
    main()
