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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


from matplotlib import colors
from scipy.special import gamma, gammainc

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

    # Orbital Velocity Formula
    # V(r) = sqrt( G * M(r) / r )
    # mass_r unit: M_sun
    # radius unit: kpc
    # velocity unit: km/s
    @staticmethod
    def calc_mass_to_V2(radius: np.ndarray, mass_r: np.ndarray, r_min: float=0.1) -> np.ndarray:
        mass_r = np.asarray(mass_r)
        radius = np.asarray(radius)
        if mass_r.shape != radius.shape:
            raise ValueError("mass_r and radius must have the same shape")
        radius = np.where(radius < r_min, np.nan, radius)
        mass_r = np.where(mass_r < 0, np.nan, mass_r)

        print(f"G constant value: {G} kpc km2 / (Msun s2)")
        V2 = G * mass_r / radius  # in (km/s)^2
        return V2

    # Savitzky-Golay fitter for V^2(r) = G * M(r) / r
    # return smoothed/interpolated values
    @staticmethod
    def _get_vel_sg_func(radius: np.ndarray, vel_sq: np.ndarray) -> interp1d:
        polyorder = 3
        # Filter out invalid data points
        valid_mask = np.isfinite(vel_sq) & np.isfinite(radius)
        radius_valid = radius[valid_mask]
        V2_valid = vel_sq[valid_mask]

        # Ensure data is sorted by radius for interpolation
        sort_indices = np.argsort(radius_valid)
        radius_valid = radius_valid[sort_indices]
        V2_valid = V2_valid[sort_indices]

        n = len(V2_valid)
        
        # If too few points, return a simple linear interpolation
        if n < polyorder + 2:
            print(f"Warning: Too few data points ({n}) for smoothing. Returning unsmoothed interpolation.")
            return interp1d(radius_valid, V2_valid, bounds_error=False, fill_value=np.nan)

        # Adjust window_length for Savitzky-Golay filter
        # It must be an odd integer and greater than polyorder.
        window_length = min(51, n)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= polyorder:
            # This ensures window_length is odd and > polyorder
            window_length = polyorder + 1 if polyorder % 2 == 0 else polyorder + 2
        if window_length > n:
            window_length = n
            if window_length % 2 == 0:
                window_length -= 1


        # Apply Savitzky–Golay smoothing
        smoothed_V2_valid = savgol_filter(V2_valid, window_length, polyorder, mode='nearest')

        # Interpolate smoothed results
        f_smooth = interp1d(radius_valid, smoothed_V2_valid,
                            kind='cubic', bounds_error=False, fill_value=np.nan)

        return f_smooth

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

        mass_map, radius_eff_map = self.calc_mass_r(mass_stellar_cell, _radius_eff)
        print(f"Cumulative Mass shape: {mass_map.shape}, Unit: M solar, range [{np.nanmin(mass_map):.3f}, {np.nanmax(mass_map):,.1f}]")
        print(f"Radius bins shape: {radius_eff_map.shape}, Unit: effective radius, range [{np.nanmin(radius_eff_map):.3f}, {np.nanmax(radius_eff_map):.3f}]")

        print("")
        print("#######################################################")
        print("# 2. calculate stellar r")
        print("#######################################################")

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

        return radius_h_kpc_map, mass_map

    ################################################################################
    # fitting methods
    ################################################################################
    def _stellar_mass_model(self, r: np.ndarray, MB: float, a: float, MD: float, rd: float) -> np.ndarray:
        bulge_mass = MB * (r**2) / (r + a)**2
        disk_mass = MD * (1 - (1 + r / rd) * np.exp(-r / rd))
        total_mass = bulge_mass + disk_mass
        return total_mass

    # used the minimum χ 2 method for fitting
    def _stellar_mass_fit(self, radius: np.ndarray, mass: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> np.ndarray:
        radius_filter = radius[radius>r_min]
        mass_filter = mass[radius>r_min]

        initial_guess = [1e10, 1.0, 1e10, 3.0]  # Initial guess for MB, a, MD, rd
        popt, pcov = curve_fit(self._stellar_mass_model, radius_filter, mass_filter, p0=initial_guess, maxfev=10000)
        fitted_mass = self._stellar_mass_model(radius_fitted, *popt)
        print(f"Fitted parameters: MB={popt[0]:.3e}, a={popt[1]:.3f}, MD={popt[2]:.3e}, rd={popt[3]:.3f}")
        return radius_fitted, fitted_mass


    def _get_stellar_V2(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        r_map, mass_map = self._get_stellar_mass(PLATE_IFU)
        vel_sq = self.calc_mass_to_V2(r_map, mass_map, r_min=RADIUS_MIN_KPC)
        return r_map, vel_sq


    ################################################################################
    # asymmetric drift correction
    # V_drift^2(R) = - Sigma_R^2 * [ (d ln(Sigma_star * Sigma_R^2) / d ln R) + (1 - Sigma_phi^2 / Sigma_R^2) ]
    ################################################################################

    def _safe_dln_dlnR(self, y: np.ndarray, R: np.ndarray, smooth: bool=True,
                    window: int=11, polyorder: int=3) -> np.ndarray:
        """
        Compute d ln y / d ln R robustly.
        """
        # avoid zero or negative values for log-derivative computation
        y = np.asarray(y, dtype=float)
        R = np.asarray(R, dtype=float)
        eps = 1e-12
        y = np.maximum(y, eps)

        # optional smoothing to reduce noise amplification in derivative
        if smooth and len(y) >= window and window % 2 == 1:
            y_s = savgol_filter(y, window_length=window, polyorder=polyorder)
        else:
            y_s = y

        dy_dR = np.gradient(y_s, R)
        # d ln y / d ln R = (R / y) * (dy / dR)
        with np.errstate(divide='ignore', invalid='ignore'):
            dln = (R / y_s) * dy_dR
        # replace NaN/inf with zeros (happens at R=0 or y nearly const)
        dln = np.nan_to_num(dln, nan=0.0, posinf=0.0, neginf=0.0)
        return dln

    def _estimate_sigma_phi2_over_sigmaR2(self, R: np.ndarray, Vc_guess: np.ndarray=None,
                                        Vphi_guess: np.ndarray=None) -> np.ndarray:
        """
        Estimate sigma_phi^2 / sigma_R^2 using epicycle approx:
        sigma_phi^2 / sigma_R^2 ≈ 0.5 * (1 + d ln Vc / d ln R)
        Try to derive d ln Vc / d ln R from:
        1) provided Vc_guess
        2) provided Vphi_guess
        3) class attributes self.vc or self.vphi if present
        4) fallback to 0 (flat rotation) -> ratio = 0.5
        """
        # choose available velocity array
        if Vc_guess is None:
            if Vphi_guess is None:
                Vc_arr = getattr(self, "vc", None)
                if Vc_arr is None:
                    Vc_arr = getattr(self, "vphi", None)
            else:
                Vc_arr = Vphi_guess
        else:
            Vc_arr = Vc_guess

        if Vc_arr is None:
            # no velocity info -> assume flat rotation curve
            dlnVc = np.zeros_like(R, dtype=float)
        else:
            Vc_arr = np.asarray(Vc_arr, dtype=float)
            # ensure non-negative
            Vc_arr = np.maximum(Vc_arr, 1e-8)
            dlnVc = _safe_dln_dlnR(self, Vc_arr, R, smooth=True)

        ratio = 0.5 * (1.0 + dlnVc)
        # clamp to reasonable physical bounds
        ratio = np.clip(ratio, 0.0, 2.0)
        return ratio

    def _calc_vel_drift_sq(self, radius: np.ndarray, sigma_stellar: np.ndarray, sigma_R: np.ndarray) -> np.ndarray:
        """
        Compute V_drift^2(R) using the formula:
        V_drift^2(R) = - sigma_R^2 * [ d ln (Sigma_* * sigma_R^2) / d ln R + (1 - sigma_phi^2 / sigma_R^2) ]
        Inputs:
        radius: 1D array of R (same unit, should be > 0)
        sigma_stellar: Sigma_*(R) (surface density, arbitrary units)
        sigma_R: radial velocity dispersion sigma_R(R) (same velocity units)
        Returns:
        V_drift_sq: 1D array of V_drift^2 (same units as velocity^2)
        Notes:
        - The function will try to estimate sigma_phi^2 / sigma_R^2 using class attributes
            or a flat-rotation fallback if necessary.
        - Numerical derivatives are smoothed to reduce noise amplification.
        """
        R = np.asarray(radius, dtype=float)
        Sigma = np.asarray(sigma_stellar, dtype=float)
        sR = np.asarray(sigma_R, dtype=float)

        # basic checks and shapes
        if R.ndim != 1 or Sigma.ndim != 1 or sR.ndim != 1:
            raise ValueError("radius, sigma_stellar, sigma_R must be 1D arrays of same length")
        if not (len(R) == len(Sigma) == len(sR)):
            raise ValueError("radius, sigma_stellar, sigma_R must have same length")

        # avoid zero/negative inputs for logs
        eps = 1e-12
        Sigma_safe = np.maximum(Sigma, eps)
        sR_safe = np.maximum(sR, eps)

        # compute d ln (Sigma * sigma_R^2) / d ln R
        product = Sigma_safe * (sR_safe**2)
        dln_product = _safe_dln_dlnR(self, product, R, smooth=True)

        # compute sigma_phi^2 / sigma_R^2 estimate
        # Prefer explicit class attributes if user provided them:
        #   self.sigma_phi (array) or self.vc / self.vphi for epicycle approx
        sigma_phi_arr = getattr(self, "sigma_phi", None)
        if sigma_phi_arr is not None:
            sigma_phi2_over_sigmaR2 = (np.asarray(sigma_phi_arr, dtype=float)**2) / (sR_safe**2)
            sigma_phi2_over_sigmaR2 = np.nan_to_num(sigma_phi2_over_sigmaR2, nan=0.5, posinf=2.0, neginf=0.0)
        else:
            # try to use vc or vphi if available on self
            Vc_guess = getattr(self, "vc", None)
            Vphi_guess = getattr(self, "vphi", None)
            sigma_phi2_over_sigmaR2 = _estimate_sigma_phi2_over_sigmaR2(self, R, Vc_guess=Vc_guess, Vphi_guess=Vphi_guess)

        # build the bracket term
        bracket = dln_product + (1.0 - sigma_phi2_over_sigmaR2)

        # V_drift^2 = - sigma_R^2 * bracket
        Vdrift2 = - (sR_safe**2) * bracket

        # numerical safety: negative values can appear due to noisy derivatives or approximations.
        # physically Vdrift^2 should be >= 0. Clamp small negative values to zero.
        Vdrift2 = np.where(Vdrift2 < 0.0, 0.0, Vdrift2)

        return Vdrift2
    
    ################################################################################
    # public methods
    ################################################################################
    def get_stellar_vel(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        r_map, vel_sq = self._get_stellar_V2(PLATE_IFU)
        vel_map = np.sqrt(vel_sq)
        print(f"Velocity shape: {vel_map.shape}, min: {np.nanmin(vel_map):.3f}, max: {np.nanmax(vel_map):,.1f} km/s")
        return r_map, vel_map
    
    def fit_vel_stellar(self, PLATE_IFU: str, radius_fitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_map, mass_map = self._get_stellar_mass(PLATE_IFU)
        _, mass_fitted = self._stellar_mass_fit(r_map, mass_map, r_min=RADIUS_MIN_KPC, radius_fitted=radius_fitted)
        vel_sq_fitted = self.calc_mass_to_V2(radius_fitted, mass_fitted, r_min=RADIUS_MIN_KPC)
        vel_fitted = np.sqrt(vel_sq_fitted)

        return radius_fitted, vel_fitted

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

    r_map, vel_map = Stellar(drpall_util, firefly_util, maps_util).get_stellar_vel(PLATE_IFU)
    r_fitted, vel_fitted = Stellar(drpall_util, firefly_util, maps_util).fit_vel_stellar(PLATE_IFU, radius_fitted=r_map)

    print("#######################################################")
    print("# calculate stellar rotation velocity V(r)")
    print("#######################################################")
    print(f"Calc Velocity shape: {vel_map.shape}, range: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}]")
    print(f"Fitted Velocity shape: {vel_fitted.shape}, range: [{np.nanmin(vel_fitted):.3f}, {np.nanmax(vel_fitted):.3f}]")

    plot_util.plot_rv_curve(r_map, vel_map, title="Stellar", r_rot2_map=r_fitted, v_rot2_map=vel_fitted, title2="Fitted Stellar")
    return

if __name__ == "__main__":
    main()
