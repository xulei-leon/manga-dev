from ast import mod
from bdb import effective
from math import log
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
from scipy.optimize import curve_fit, minimize
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
    # profile
    ################################################################################

    #
    # Stellar Mass M(r)
    #

    # The RC of the baryonic disk model (V_baryon) was derived by a method in Noordermeer (2008).

    # Hernquist bulge + Freeman thin exponential disk
    # V_bulge^2(r) = (G * MB * r) / (r + a)^2
    def _vel_sq_bulge_hernquist(self, r: np.ndarray, MB: float, a: float) -> np.ndarray:
        r = np.where(r == 0, 1e-6, r)  # avoid division by zero
        v_sq = G * MB * r / (r + a)**2
        return v_sq

    # V_disk^2(r) = (2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    def _vel_sq_disk_freeman(self, r: np.ndarray, M_d: float, Rd: float) -> np.ndarray:
        r = np.where(r == 0, 1e-6, r)  # avoid division by zero
        y = r / (2.0 * Rd)
        # Use exponentially scaled Bessel functions to avoid overflow/underflow
        # i0e(x) = exp(-|x|) * I0(x)
        # k0e(x) = exp(x) * K0(x)
        # The product I0(x)*K0(x) becomes i0e(x)*k0e(x) because exp(-x)*exp(x) = 1
        I_0 = special.i0e(y)
        I_1 = special.i1e(y)
        K_0 = special.k0e(y)
        K_1 = special.k1e(y)
        v_sq = (2.0 * G * M_d / Rd) * (np.square(y)) * (I_0 * K_0 - I_1 * K_1)
        return v_sq

    # M_star: total mass of star
    # Re: Half-mass radius
    # f_bulge: bulge mass fraction
    # a: Hernquist scale radius
    # V_star^2 = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    def _stellar_vel_sq_mass_profile(self, r: np.ndarray, M_star: float, Re: float, f_bulge: float, a: float) -> np.ndarray:
        Rd = Re / 1.678
        MB = f_bulge * M_star
        MD = (1 - f_bulge) * M_star

        v_bulge_sq = self._vel_sq_bulge_hernquist(r, MB, a)
        v_disk_sq = self._vel_sq_disk_freeman(r, MD, Rd)
        v_baryon_sq = v_bulge_sq + v_disk_sq
        return v_baryon_sq

    # M_star: total mass of star
    # Re: Half-mass radius
    def _stellar_vel_sq_disk_profile(self, r: np.ndarray, M_star: float, Re: float):
        Rd = Re / 1.678
        return self._vel_sq_disk_freeman(r, M_star, Rd)


    # Formula: M(r) = MB * r^2 / (r + a)^2 + MD * (1 - (1 + r / rd) * exp(-r / rd))
    def _stellar_mass_bulge_profile(self, r: np.ndarray, MB: float, a: float,) -> np.ndarray:
        bulge_mass = MB * np.square(r) / np.square(r + a)
        return bulge_mass

    def _stellar_mass_disk_profile(self, r: np.ndarray, MD: float, rd: float) -> np.ndarray:
        disk_mass = MD * (1.0 - (1.0 + r / rd) * np.exp(-r / rd))
        return disk_mass

    def _stellar_mass_profile(self, r: np.ndarray, M_star: float, Re: float, f_bulge: float=None, a: float=None) -> np.ndarray:
        Rd = Re / 1.678

        if f_bulge:
            MB = f_bulge * M_star
            MD = (1 - f_bulge) * M_star
            bulge_mass = self._stellar_mass_bulge_profile(r, MB, a)
        else:
            MD = M_star
            bulge_mass = 0

        disk_mass = self._stellar_mass_disk_profile(r, MD, Rd)
        total_mass = bulge_mass + disk_mass
        return total_mass


    ################################################################################
    # functions
    ################################################################################
    @staticmethod
    def _calc_mass_of_radius(mass_cell: np.ndarray, mass_cell_err: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mass_cell = np.asarray(mass_cell)
        mass_cell_err = np.asarray(mass_cell_err)
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
        sorted_mass_err = mass_cell_err[valid_mask][sorted_idx]
        mass_r = np.cumsum(sorted_masses)
        mass_err_r = np.sqrt(np.cumsum(np.square(sorted_mass_err)))
        r_bins = sorted_radius

        return r_bins, mass_r, mass_err_r


    def _get_stellar_mass(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        mass_stellar_cell, mass_stellar_cell_err = self.firefly_util.get_stellar_mass_cell(PLATE_IFU)
        print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Unit: M solar, total: {np.nansum(mass_stellar_cell):,.1f} M solar")

        # This mass use h = 1
        total_stellar_mass_1, total_stellar_mass_2 = self.drpall_util.get_stellar_mass(PLATE_IFU)
        # print("Verification with DRPALL stellar mass:")
        _mass_err2_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_2) / total_stellar_mass_2
        _mass_err1_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_1) / total_stellar_mass_1
        print(f"  Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")
        # if abs(_mass_err2_percent) < 0.03:
        #     print(f"  FIREFLY total stellar mass matches DRPALL Elpetro stellar mass within {abs(_mass_err2_percent):.1%}")
        # elif abs(_mass_err1_percent) < 0.03:
        #     print(f"  FIREFLY total stellar mass matches DRPALL Sersic stellar mass within {abs(_mass_err1_percent):.1%}")
        # else:
        #     print("  WARNING: FIREFLY total stellar mass does not match DRPALL stellar masses within 3%")

        _radius_eff, azimuth = self.firefly_util.get_radius_eff(PLATE_IFU)
        # print(f"Radius Eff shape: {_radius_eff.shape}, Unit: effective radius, range [{np.nanmin(_radius_eff[azimuth>=0]):.3f}, {np.nanmax(_radius_eff):.3f}]")
        # print(f"Azimuth shape: {azimuth.shape}, Unit: degrees, range [{np.nanmin(azimuth[azimuth>=0]):.3f}, {np.nanmax(azimuth):.3f}]")

        radius_eff_map, mass_map, mass_err_map = self._calc_mass_of_radius(mass_stellar_cell, mass_stellar_cell_err, _radius_eff)
        # print(f"Cumulative Mass shape: {mass_map.shape}, Unit: M solar, range [{np.nanmin(mass_map):.3f}, {np.nanmax(mass_map):,.1f}]")
        # print(f"Radius bins shape: {radius_eff_map.shape}, Unit: effective radius, range [{np.nanmin(radius_eff_map):.3f}, {np.nanmax(radius_eff_map):.3f}]")

        radius_h_kpc_map = self._calc_radius_to_h_kpc(PLATE_IFU, radius_eff_map)

        return radius_h_kpc_map, mass_map, mass_err_map


    def _fit_stellar_mass(self, radius: np.ndarray, mass_map: np.ndarray, std_err: np.ndarray) -> tuple[float, float, float, float]:
        # Filter valid data
        valid_mask = (np.isfinite(radius))
        valid_mask &= (np.isfinite(mass_map))
        valid_mask &= (np.isfinite(std_err))

        radius_valid = radius[valid_mask]
        mass_valid = mass_map[valid_mask]
        std_err_valid = std_err[valid_mask]
        mass_star_total = np.nanmax(mass_valid)

        if radius_valid.size < 4:
            print("Not enough valid data points for fitting stellar mass profile.")
            return np.nan, np.nan, np.nan, np.nan

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'Re': (0.1, np.nanmax(radius_valid)),  # kpc
        }

        def _denormalize_params(params_norm):
            _Re_n = params_norm
            _Re = _Re_n * (params_range['Re'][1] - params_range['Re'][0]) + params_range['Re'][0]
            return _Re

        ######################################
        # Fitting process using curve_fit
        ######################################
        def model_func(r, Re_n):
            Re = _denormalize_params(Re_n)
            return self._stellar_mass_profile(r, mass_star_total, Re)

        initial_guess = [0.5]  # normalized initial guess
        bounds = ([0.0], [1.0])  # normalized bounds
        try:
            popt, pcov = curve_fit(
                model_func,
                radius_valid,
                mass_valid,
                p0=initial_guess,
                sigma=std_err_valid,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=10000
            )
        except RuntimeError as e:
            print(f"Fitting failed: {e}")
            return np.nan, np.nan, np.nan, np.nan

        Re_fit = _denormalize_params(popt)[0]

        ######################################
        # Error estimation
        ######################################
        residuals = mass_valid - model_func(radius_valid, *popt)
        dof = len(mass_valid) - len(popt)
        chi_sq = np.sum(np.square(residuals / std_err_valid))
        CHI_SQ_V = chi_sq / dof
        F_factor = np.sqrt(CHI_SQ_V) if CHI_SQ_V > 1 else 1.0

        COR_MATRIX = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))

        perr = np.sqrt(np.diag(pcov)) * F_factor
        Re_err_n = perr
        Re_err = Re_err_n * (params_range['Re'][1] - params_range['Re'][0])
        Re_err = Re_err[0]
        Re_err_pct = (Re_err / Re_fit) * 100.0 if Re_fit != 0 else np.nan

        mass_star_map_fit = self._stellar_mass_profile(radius_valid, mass_star_total, Re_fit)
        M_star_fit = np.nanmax(mass_star_map_fit)

        print(f"\n------------ Fitted Stellar Mass Profile Parameters ------------")
        print(f" Fit Re                 : {Re_fit:.3f} ± {Re_err:.3f} kpc ({Re_err_pct:.2f} %)")
        print(f" Fit M_star             : {M_star_fit:.3e} M solar")
        print(f" Calc M_star            : {mass_star_total:.3e} M solar")
        print("--------------------------")
        print(f" Reduced Chi-Squared    : {CHI_SQ_V:.3f}")
        print(f" Correlation Matrix     : \n{COR_MATRIX}")
        print("--------------------------------------------------------------------\n")

        return M_star_fit, Re_fit

    ################################################################################
    # Mass Density Method
    # bulge: Hernquist profile
    # disk: Exponential profile
    ################################################################################

    # Hernquist Bulge Model fitting function
    # according to Hernquist (1990)
    # formula: Sigma(R) = (Sigma_0 * a^4) / (R + a)^3
    def _stellar_density_profile_bulge(self, R: np.ndarray, Sigma_0: float, a: float) -> np.ndarray:
        return (Sigma_0 * a**4) / (R + a)**3

    # Exponential Disk Model fitting function
    # according to Freeman (1970)
    # Sigma(R) = Sigma_0 * exp(-R / Rd)
    def _stellar_density_profile_disk(self, R: np.ndarray, Sigma_0: float, Re: float) -> np.ndarray:
        Rd = Re / 1.678
        return Sigma_0 * np.exp(-R / Rd)


    def _stellar_density_profile(self, R: np.ndarray, Sigma_0: float, Re: float) -> np.ndarray:
        disk_density = self._stellar_density_profile_disk(R, Sigma_0, Re)
        total_density = disk_density
        return total_density

    # Central Surface Mass Density Fitting
    def _fit_stellar_central_density(self, radius: np.ndarray, density: np.ndarray, std_err: np.ndarray) -> tuple[float, float]:
        # Filter valid data
        valid_mask = (np.isfinite(radius)) & (radius >= RADIUS_MIN_KPC) & (radius < 0.99*np.nanmax(radius))
        valid_mask &= (np.isfinite(density)) & (density >= 0)
        valid_mask &= (np.isfinite(std_err)) & (std_err > 0)

        radius_valid = radius[valid_mask]
        density_valid = density[valid_mask]
        std_err_valid = std_err[valid_mask]

        if radius_valid.size < 2:
            print("Not enough valid data points for fitting stellar central density.")
            return np.nan, np.nan

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'Sigma_0': (1e6, 1e11),  # M solar / kpc^2
            'Re': (0.1, np.nanmax(radius_valid))  # kpc
        }

        def _denormalize_params(params_norm):
            _Sigma_0_n, _Re_n = params_norm
            _Sigma_0 = _Sigma_0_n * (params_range['Sigma_0'][1] - params_range['Sigma_0'][0]) + params_range['Sigma_0'][0]
            _Re = _Re_n * (params_range['Re'][1] - params_range['Re'][0]) + params_range['Re'][0]
            return _Sigma_0, _Re

        ######################################
        # Fitting process using curve_fit
        ######################################
        def model_func(R, Sigma_0_n, Re_n):
            Sigma_0, Re = _denormalize_params([Sigma_0_n, Re_n])
            return self._stellar_density_profile(R, Sigma_0, Re)

        initial_guess = [0.5, 0.2]  # normalized initial guess
        bounds = ([0.0, 0.0], [1.0, 1.0])  # normalized bounds

        # Perform curve fitting
        popt, pcov = curve_fit(
            model_func,
            radius_valid,
            density_valid,
            p0=initial_guess,
            sigma=std_err_valid, #Standard Deviation of the Errors
            absolute_sigma=True,
            bounds=bounds,
            maxfev=10000
        )

        Sigma_0_fit, Re_fit = _denormalize_params(popt)

        ######################################
        # Error estimation
        ######################################

        # Reduced Chi-Squared
        residuals = density_valid - model_func(radius_valid, *popt)
        dof = len(density_valid) - len(popt)  # degrees of freedom
        chi_sq = np.sum(np.square(residuals / std_err_valid))
        CHI_SQ_V = chi_sq / dof
        F_factor = np.sqrt(CHI_SQ_V) if CHI_SQ_V > 1 else 1.0

        # Correlation Matrix
        COR_MATRIX = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))

        # Standard Errors of the Parameters
        perr = np.sqrt(np.diag(pcov)) * F_factor
        Sigma_0_err_n, Re_err_n = perr
        Sigma_0_err = Sigma_0_err_n * (params_range['Sigma_0'][1] - params_range['Sigma_0'][0])
        Re_err = Re_err_n * (params_range['Re'][1] - params_range['Re'][0])
        Sigma_0_err_pct = (Sigma_0_err / Sigma_0_fit) * 100.0 if Sigma_0_fit != 0 else np.nan
        Re_err_pct = (Re_err / Re_fit) * 100.0 if Re_fit != 0 else np.nan


        print(f"\n------------ Fitted Stellar Central Density Parameters ------------")
        print(f" Fit Sigma_0            : {Sigma_0_fit:.3e} ± {Sigma_0_err:.3e} M solar / kpc^2 ({Sigma_0_err_pct:.2f} %)")
        print(f" Fit Re                 : {Re_fit:.3f} ± {Re_err:.3f} kpc ({Re_err_pct:.2f} %)")
        print("--------------------------")
        print(f" Reduced Chi-Squared    : {CHI_SQ_V:.3f}")
        print(f" Correlation Matrix     : \n{COR_MATRIX}")
        print("--------------------------------------------------------------------\n")

        return Sigma_0_fit, Re_fit


    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def stellar_vel_sq_profile(self, r: np.ndarray, M_star: float, Re: float) -> np.ndarray:
        return self._stellar_vel_sq_disk_profile(r, M_star, Re)

    def fit_stellar_mass(self):
        radius_map, mass_map, std_err_map = self._get_stellar_mass(self.PLATE_IFU)
        M_star_fit, Re_fit = self._fit_stellar_mass(radius_map, mass_map, std_err_map)
        return M_star_fit, Re_fit

    def fit_stellar_density_sigma0(self):
        _radius_eff, _ = self.firefly_util.get_radius_eff(self.PLATE_IFU)
        _radius_h_kpc = self._calc_radius_to_h_kpc(self.PLATE_IFU, _radius_eff)

        _density, _std_err = self.firefly_util.get_stellar_density_cell(self.PLATE_IFU)
        print(f"Stellar Mass Density shape: {_density.shape}, Unit: M solar / kpc^2, range [{np.nanmin(_density[_density>=0]):.3f}, {np.nanmax(_density):,.1f}] M solar / kpc^2")

        sigma0_fit, Re_fit = self._fit_stellar_central_density(_radius_h_kpc, _density, _std_err)
        return sigma0_fit



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

    _, radius_h_kpc_map, _ = maps_util.get_radius_map()
    radius_max = np.nanmax(radius_h_kpc_map)

    stellar.set_PLATE_IFU(PLATE_IFU)

    stellar_mass, stellar_Re = stellar.fit_stellar_mass()
    print(f"Stellar fit mass: {stellar_mass:.3e} M solar, Re: {stellar_Re:.2f} kpc/h")
    print("")

    stellar_sigma0 = stellar.fit_stellar_density_sigma0()
    print(f"Stellar fit Central Density sigma_0: {stellar_sigma0:.3e} M solar / kpc^2")
    print("")

    V_star_sq = stellar.stellar_vel_sq_profile(radius_h_kpc_map, stellar_mass, stellar_Re)
    V_star = np.sqrt(V_star_sq)
    print(f"Stellar Velocity shape: {V_star.shape}, Unit: km/s, range: [{np.nanmin(V_star[V_star>=0]):.1f}, {np.nanmax(V_star):.1f}] km/s")

    return
if __name__ == "__main__":
    main()
