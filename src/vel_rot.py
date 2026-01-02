
from pathlib import Path
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit, minimize
from lmfit import Model, Parameters
from matplotlib import colors

# my imports
from util import plot_util
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil
from vel_stellar import RADIUS_MIN_KPC


######################################################################
# constants definitions
######################################################################
# Thresholds for filtering velocity data points
SNR_THRESHOLD = 10.0
PHI_DEG_THRESHOLD = 45.0
# GSIGMA_THRESHOLD = 100.0  # km/s
IVAR_RATIO_THRESHOLD = 0.10  # drop the worst 10% of ivar values

# Thresholds for filtering fitting results
NRMSE_THRESHOLD1 = 0.07  # threshold for first fitting
NRMSE_THRESHOLD2 = 0.05  # tighter threshold for second fitting
CHI_SQ_V_THRESHOLD1 = 5.0  # looser threshold for first fitting
CHI_SQ_V_THRESHOLD2 = 3.0  # threshold for reduced chi-squared to filter weak fitting
VEL_OBS_COUNT_THRESHOLD1 = 150  # minimum number of valid velocity data points
VEL_OBS_COUNT_THRESHOLD2 = 100  # minimum number of valid velocity data points

RADIUS_MIN_KPC = 0.1  # kpc/h
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation
VEL_SYSTEM_ERROR = 5.0  # km/s, floor error as systematic uncertainty in velocity measurements


root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"

######################################################################
# class
######################################################################

class VelRot:
    drpall_util = None
    firefly_util = None
    maps_util = None
    plot_util = None
    fit_debug = False


    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil, plot_util: PlotUtil=None) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util
        self.plot_util = plot_util

    ################################################################################
    # calculate functions
    ################################################################################

    # Calculate the galaxy inclination i (in radians)
    # Formula for inclination i
    # The inclination is the angle between the galaxy disk normal and the observer's line of sight.
    # ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
    @staticmethod
    def _calc_inc(ba, ba_0=0.2):
        ba_sq = ba**2
        BA_0_sq = ba_0**2

        # Compute the numerator part of cos^2(i)
        numerator = ba_sq - BA_0_sq
        denominator = 1.0 - BA_0_sq

        cos_i_sq = numerator / denominator
        cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)

        inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
        return inc_rad


    # Filter the velocity map with SNR above the threshold and within ±phi_limit of the major axis.
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, phi_map: np.ndarray, ivar_map: np.ndarray, gsigma_map: np.ndarray) -> np.ndarray:
        phi_delta = (phi_map + np.pi/2) % np.pi - np.pi/2
        phi_limit_rad = np.radians(PHI_DEG_THRESHOLD)

        valid_mask = (np.isfinite(vel_map) &
                      (snr_map >= SNR_THRESHOLD) &
                      (np.abs(phi_delta) <= phi_limit_rad) &
                    #   (gsigma_map <= GSIGMA_THRESHOLD) &
                        np.isfinite(gsigma_map))

        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]

        # ivar filtering: drop the worst 5% of ivar values (within valid_mask)
        ivar_valid_mask = valid_mask & np.isfinite(ivar_map)
        ivar_valid = ivar_map[ivar_valid_mask]
        if ivar_valid.size > 0:
            ivar_limit = float(np.nanpercentile(ivar_valid, 100 * IVAR_RATIO_THRESHOLD))
            ivar_keep_mask = ivar_valid_mask & (ivar_map >= ivar_limit)
        else:
            ivar_keep_mask = ivar_valid_mask

        vel_map_filtered[~ivar_keep_mask] = np.nan

        return vel_map_filtered

    # PA: The position angle of the major axis of the galaxy, measured from north to east.
    # b/a: The axis ratio (b/a) of the galaxy
    def _calc_pa_inc(self) -> float:
        phi = self.maps_util.get_pa()
        ba = self.maps_util.get_ba()
        # print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")

        inc = self._calc_inc(ba, ba_0=BA_0)
        # print(f"Calculated Inclination i: {np.degrees(inc):.2f} deg")
        # Convert PA from degrees to radians and rotate so North is at +90°
        pa = np.mod(np.radians(phi), 2 * np.pi)
        return pa, inc

    def _get_vel_obs_raw(self, type: str='gas', is_filter: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offset_x, offset_y = self.maps_util.get_sky_offsets()
        # print(f"Sky offsets shape: {offset_x.shape}, X offset: [{np.nanmin(offset_x):.3f}, {np.nanmax(offset_x):.3f}] arcsec")

        # R: radial distance map
        radius_map, radius_h_kpc_map, azimuth_map = self.maps_util.get_radius_map()
        # print(f"r_map: [{np.nanmin(radius_map):.3f}, {np.nanmax(radius_map):.3f}] spaxel,", f"shape: {radius_map.shape}")
        # print(f"r_h_kpc_map: [{np.nanmin(radius_h_kpc_map):.3f}, {np.nanmax(radius_h_kpc_map):.3f}] kpc,", f"shape: {radius_h_kpc_map.shape}")
        # print(f"azimuth_map: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] deg,", f"shape: {azimuth_map.shape}")

        # SNR: signal-to-noise ratio map
        snr_map = self.maps_util.get_snr_map()
        # print(f"SNR map shape: {snr_map.shape}, SNR range: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

        ra_map, dec_map = self.maps_util.get_skycoo_map()
        # print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

        ## Get the gas velocity map (H-alpha)
        v_obs_gas_map, _gv_unit, _gv_ivar = self.maps_util.get_eml_vel_map()
        # print(f"Gas velocity map shape: {v_obs_gas_map.shape}, Unit: {_gv_unit}, Velocity range: [{np.nanmin(v_obs_gas_map):.3f}, {np.nanmax(v_obs_gas_map):.3f}] {_gv_unit}, size: {np.sum(np.isfinite(v_obs_gas_map))}")
        eml_binid = self.maps_util.get_emli_binid()
        # print(f"Gas Unique indices shape: {eml_binid.shape}, range: [{np.nanmin(eml_binid):.0f}, {np.nanmax(eml_binid):.0f}], size: {len(np.unique(eml_binid))}")

        gsigma_map, gsigma_inst_map = self.maps_util.get_eml_gsigma_map()
        print(f"Gas sigma map shape: {gsigma_map.shape}, range: [{np.nanmin(gsigma_map):.3f}, {np.nanmax(gsigma_map):.3f}] km/s, mean Gas sigma: {np.nanmean(gsigma_map):.3f} km/s")


        ## Get the stellar velocity map
        v_obs_stellar_map, _sv_unit, _sv_ivar = self.maps_util.get_stellar_vel_map()
        stellar_binid = self.maps_util.get_stellar_binid()

        # Velocity correction
        if type == 'gas':
            v_obs_map = v_obs_gas_map
            v_unit = _gv_unit
            v_ivar = _gv_ivar
        else:
            v_obs_map = v_obs_stellar_map
            v_unit = _sv_unit
            v_ivar = _sv_ivar

        azimuth_rad_map = np.radians(azimuth_map)

        # Filter velocity map
        if not is_filter:
            filtered_vel_map = v_obs_map
        else:
            filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, v_ivar, gsigma_map)

        # print(f"Filtered Velocity map shape: {filtered_vel_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(filtered_vel_map):.3f}, {np.nanmax(filtered_vel_map):.3f}]")
        print(f"vel_obs data count: {np.sum(np.isfinite(v_obs_map))}, after filter: {np.sum(np.isfinite(filtered_vel_map))}  ({100.0 * np.sum(np.isfinite(filtered_vel_map)) / np.sum(np.isfinite(v_obs_map)):.2f}%)")

        mask = np.isfinite(filtered_vel_map)
        r_obs_map = np.where(mask, radius_h_kpc_map, np.nan)
        v_obs_map = np.where(mask, v_obs_map, np.nan)
        ivar_obs_map = np.where(mask, v_ivar, np.nan)
        phi_obs_map = np.where(mask, azimuth_rad_map, np.nan)

        return r_obs_map, v_obs_map, ivar_obs_map, phi_obs_map

    def _get_radius(self) -> np.ndarray:
        _, radius_h_kpc_map, _ = self.maps_util.get_radius_map()
        return radius_h_kpc_map


    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).

    # Formula: V_obs = V_rot * (sin(i) * cos(phi - phi_0))
    # Warning: The sign of the calculated velocity may be different from the observed velocity.
    def _vel_obs_project_profile(self, vel_rot: np.ndarray, inc: float, phi_map: np.ndarray) -> np.ndarray:
        phi_delta = (phi_map + np.pi) % (2 * np.pi)  # phi_map is (phi - phi_0)

        correction = np.sin(inc) * np.cos(phi_delta)
        vel_obs = vel_rot * correction
        return vel_obs

    # formula: V_rot = V_obs / (sin(i) * cos(phi - phi_0))
    def _vel_rot_disproject_profile(self, vel_obs: np.ndarray, inc: float, phi_map: np.ndarray) -> np.ndarray:
        phi_delta = (phi_map + np.pi) % (2 * np.pi)  # phi_map is (phi - phi_0)

        correction = np.sin(inc) * np.cos(phi_delta)
        correction = np.where(np.abs(correction) < 1e-3, np.nan, correction)
        vel_rot = vel_obs / correction

        # set the sign of vel_rot to be the same as vel_obs
        vel_rot = np.copysign(np.abs(vel_rot), vel_obs)

        return vel_rot


    ################################################################################
    # profile
    ################################################################################

    # Formula: V(r) = Vc * tanh(r / Rt) + s_out * r
    # Vc: Vc is the asymptotic circular velocity at large radii
    # Rt: Rt is the turnover radius where the hyperbolic tangent term begins to be flat
    # s_out: sout is the slope of the RC at large radii r >> Rt
    # Negativity: The s_out parameter may have bad standard errors.
    def _vel_rot_tan_sout_profile(self, r: np.ndarray, Vc: float, Rt: float, s_out: float) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + s_out * r

    # Formula: V(r) = Vc * tanh(r / Rt) * (1 + beta * r / Rmax)
    # def _vel_rot_tan_beta_profile(self, r: np.ndarray, Vc: float, Rt: float, beta: float, Rmax: float) -> np.ndarray:
    #     return Vc * np.tanh(r / Rt) * (1 + beta * r / Rmax)

    # Universal Rotation Curve (URC)
    # Positivity: stable model with good Standard Errors of the parameters
    # Negativity: the reduced Chi-Squared is not good enough
    # Formula: V(r) = V0 + (2/pi) * Vc * arctan(r / Rt)
    def _vel_rot_arctan_profile(self, r: np.ndarray, V0: float, Vc: float, Rt: float) -> np.ndarray:
        return V0 + (2 / np.pi) * Vc * np.arctan(r / Rt)

    # Formula: V(r) = V0 * (1 - e^(-r / Rt)) (1 + alpha * r / Rt)
    # Negativity: The alpha parameter parameter may have bad standard errors.
    def _vel_rot_polyex_profile(self, r: np.ndarray, V0: float, Rt: float, alpha: float) -> np.ndarray:
        return V0 * (1 - np.exp(-r / Rt)) * (1 + alpha * r / Rt)


    ################################################################################
    # Error functions
    ################################################################################
    def _calc_loss(self, y_obs: np.ndarray, y_model: np.ndarray, ivar: np.ndarray) -> float:
        # sigma = np.sqrt(1.0 / ivar)
        sigma_sq = 1.0 / ivar + (VEL_SYSTEM_ERROR)**2  # adding floor error
        residuals = np.abs(y_obs) - np.abs(y_model)
        loss = np.sum(residuals**2 / sigma_sq)
        return loss

    def _calc_chi_sq_v(self, vel_obs: np.ndarray, vel_model: np.ndarray, ivar_map: np.ndarray, num_params: int) -> float:
        chi_sq = self._calc_loss(vel_obs, vel_model, ivar_map)
        N = np.sum(np.isfinite(vel_obs) & np.isfinite(ivar_map))
        k = num_params
        dof = max(N - k, 1)
        chi_sq_v = chi_sq / dof
        return chi_sq_v

    def _calc_R_sq_adj(self, vel_valid, vel_obs_model, ivar_map_valid, k):
        ss_res = np.nansum(((vel_valid - vel_obs_model) ** 2) * ivar_map_valid)
        vel_mean = np.nansum(vel_valid * ivar_map_valid) / np.nansum(ivar_map_valid)
        ss_tot = np.nansum(((vel_valid - vel_mean) ** 2) * ivar_map_valid)
        r_squared = 1 - (ss_res / ss_tot)
        n = np.sum(np.isfinite(vel_valid))
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        return r_squared_adj

    ################################################################################
    # Fitting methods
    ################################################################################
    # use tanh profile to fit vel_rot
    def _fit_vel_rot(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, ivar_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[bool, dict, dict]:
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > RADIUS_MIN_KPC)
        radius_valid = radius_map[valid_mask]
        vel_obs_valid = vel_obs_map[valid_mask]
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()
        R_max = np.nanmax(radius_valid)
        SIGMA_OBS_BAR = np.sqrt(1.0 / np.nanmean(ivar_map_valid))

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'Vc': (20.0, 500.0),  # km/s
            'Rt': (R_max * 0.01, R_max * 1.0),  # kpc
            's_out': (-10.0, 10.0),  # km/s/kpc
            'Vsys': (0.0, 100.0),  # km/s
            'inc': (np.deg2rad(np.rad2deg(inc_act)-10), np.deg2rad(np.rad2deg(inc_act)+10)),  # rad
            'phi_delta': (np.deg2rad(-10), np.deg2rad(10)),  # rad
        }

        def _denormalize_params(params_n):
            Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n = params_n
            Vc = Vc_n * (params_range['Vc'][1] - params_range['Vc'][0]) + params_range['Vc'][0]
            Rt = Rt_n * (params_range['Rt'][1] - params_range['Rt'][0]) + params_range['Rt'][0]
            s_out = s_out_n * (params_range['s_out'][1] - params_range['s_out'][0]) + params_range['s_out'][0]
            Vsys = Vsys_n * (params_range['Vsys'][1] - params_range['Vsys'][0]) + params_range['Vsys'][0]
            inc = inc_n * (params_range['inc'][1] - params_range['inc'][0]) + params_range['inc'][0]
            phi_delta = phi_delta_n * (params_range['phi_delta'][1] - params_range['phi_delta'][0]) + params_range['phi_delta'][0]
            return [Vc, Rt, s_out, Vsys, inc, phi_delta]

        ######################################
        # Fitting process using lmfit (replace curve_fit)
        ######################################
        def model_func(r, Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n):
            Vc, Rt, s_out, Vsys, inc, phi_delta = _denormalize_params([Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n])
            vel_rot_model = self._vel_rot_tan_sout_profile(r, Vc, Rt, s_out)
            vel_obs_model = Vsys + self._vel_obs_project_profile(vel_rot_model, inc, phi_map_valid - phi_delta)
            # fixed the sign of vel_obs_model to be the same as vel_valid
            vel_obs_model = np.copysign(np.abs(vel_obs_model), vel_obs_valid)
            return vel_obs_model

        # sigma: Standard Deviation of the Errors
        sigma = np.sqrt(1.0 / ivar_map_valid + (VEL_SYSTEM_ERROR)**2)  # adding floor error
        weights = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma, 0.0)

        lm_model = Model(model_func, independent_vars=["r"])

        # normalized initial guesses
        params = lm_model.make_params(Vc_n=0.5, Rt_n=0.2, s_out_n=0.5, Vsys_n=0.1, inc_n=0.5, phi_delta_n=0.5)

        # Fix some parameters during fitting
        params['inc_n'].set(value=(inc_act - params_range['inc'][0]) / (params_range['inc'][1] - params_range['inc'][0]))
        params['inc_n'].vary = False  # fix inclination during fitting
        params['phi_delta_n'].set(value=0.5)  # start from zero offset
        params['phi_delta_n'].vary = False  # fix phi_delta during fitting

        # normalized bounds [0, 1]
        for name in ("Vc_n", "Rt_n", "s_out_n", "Vsys_n", "inc_n", "phi_delta_n"):
            params[name].set(min=0.0, max=1.0)

        lm_result = lm_model.fit(
            vel_obs_valid,
            params=params,
            r=radius_valid,
            weights=weights,
            method="least_squares",
            max_nfev=10000,
            nan_policy='omit',
            fit_kws={'ftol': 1e-8, 'xtol': 1e-8},
        )

        popt = np.array([
            lm_result.params["Vc_n"].value,
            lm_result.params["Rt_n"].value,
            lm_result.params["s_out_n"].value,
            lm_result.params["Vsys_n"].value,
            lm_result.params["inc_n"].value,
            lm_result.params["phi_delta_n"].value,
        ], dtype=float)

        # Covariance matrix (normalized space); provide a fallback if not available
        if lm_result.covar is not None and np.shape(lm_result.covar) == (6, 6):
            pcov = np.array(lm_result.covar, dtype=float)
        else:
            perr_n = np.array([
            lm_result.params["Vc_n"].stderr,
            lm_result.params["Rt_n"].stderr,
            lm_result.params["s_out_n"].stderr,
            lm_result.params["Vsys_n"].stderr,
            lm_result.params["inc_n"].stderr,
            lm_result.params["phi_delta_n"].stderr,
            ], dtype=float)
            perr_n = np.where(np.isfinite(perr_n), perr_n, np.nan)
            pcov = np.diag(perr_n**2)

        Vc_fit, Rt_fit, s_out_fit, Vsys_fit, inc_fit, phi_delta_fit = _denormalize_params(popt)

        ######################################
        # Error estimation (use lmfit built-ins)
        ######################################
        # Best-fit model from lmfit
        vel_obs_model = lm_result.best_fit
        residuals = np.abs(vel_obs_valid) - np.abs(vel_obs_model)

        # Basic fit metrics
        CHI_SQ_V = float(lm_result.redchi)  # reduced chi-squared from lmfit
        # Inflate uncertainties if reduced chi-squared > 1
        F_factor = float(np.maximum(np.sqrt(CHI_SQ_V), 1.0))

        RMSE = float(np.sqrt(np.nanmean(residuals**2)))
        NRMSE = float(RMSE / np.nanmean(np.abs(vel_obs_valid)))

        # Covariance / correlation from lmfit
        if lm_result.covar is not None and np.shape(lm_result.covar) == (6, 6):
            pcov = np.array(lm_result.covar, dtype=float)
            COR_MATRIX = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))
        else:
            pcov = None
            COR_MATRIX = None

        # Parameter standard errors (from lmfit), then scale to physical units + optional inflation
        def _stderr(name: str) -> float:
            v = lm_result.params[name].stderr
            return float(v) if v is not None and np.isfinite(v) else np.nan

        Vc_norm_err = _stderr("Vc_n")
        Rt_norm_err = _stderr("Rt_n")
        s_out_norm_err = _stderr("s_out_n")
        Vsys_norm_err = _stderr("Vsys_n")
        inc_norm_err = _stderr("inc_n")
        phi_delta_norm_err = _stderr("phi_delta_n")

        Vc_err = Vc_norm_err * (params_range["Vc"][1] - params_range["Vc"][0]) * F_factor
        Rt_err = Rt_norm_err * (params_range["Rt"][1] - params_range["Rt"][0]) * F_factor
        s_out_err = s_out_norm_err * (params_range["s_out"][1] - params_range["s_out"][0]) * F_factor
        Vsys_err = Vsys_norm_err * (params_range["Vsys"][1] - params_range["Vsys"][0]) * F_factor
        inc_err = inc_norm_err * (params_range["inc"][1] - params_range["inc"][0]) * F_factor
        phi_delta_err = phi_delta_norm_err * (params_range["phi_delta"][1] - params_range["phi_delta"][0]) * F_factor

        Vc_err_pct = (Vc_err / Vc_fit) * 100 if Vc_fit != 0 else np.nan
        Rt_err_pct = (Rt_err / Rt_fit) * 100 if Rt_fit != 0 else np.nan
        s_out_err_pct = (s_out_err / s_out_fit) * 100 if s_out_fit != 0 else np.nan
        Vsys_err_pct = (Vsys_err / Vsys_fit) * 100 if Vsys_fit != 0 else np.nan
        inc_err_pct = (inc_err / inc_fit) * 100 if inc_fit != 0 else np.nan
        phi_delta_err_pct = (phi_delta_err / phi_delta_fit) * 100 if phi_delta_fit != 0 else np.nan

        if self.fit_debug:
            print(f"\n------------ Fitted Rotational Velocity (tanh + sout lmfit) ------------")
            print(f" IFU                    : {self.PLATE_IFU}")
            print(f" Fit  Vc                : {Vc_fit:.1e} km/s, ± {Vc_err:.0e} km/s", f"({Vc_err_pct:.2f} %)")
            print(f" Fit  Rt                : {Rt_fit:.1e} kpc/h, ± {Rt_err:.0e} kpc/h", f"({Rt_err_pct:.2f} %)")
            print(f" Fit  s_out             : {s_out_fit:.1e} km/s/kpc, ± {s_out_err:.0e} km/s/kpc", f"({s_out_err_pct:.2f} %)")
            print(f" Fit  Vsys              : {Vsys_fit:.1e} km/s, ± {Vsys_err:.0e} km/s", f"({Vsys_err_pct:.2f} %)")
            print(f" Fit  inc               : {inc_fit:.1e} rad, ± {inc_err:.0e} rad", f"({inc_err_pct:.2f} %)")
            print(f" Fit  phi_delta         : {phi_delta_fit:.1e} rad, ± {phi_delta_err:.0e} rad", f"({phi_delta_err_pct:.2f} %)")
            print("--------------")
            print(f" Calc inc from b/a      : {inc_act:.1e} rad, {np.degrees(inc_act):.2f} deg")
            print("--------------")
            print(f" Reduced Chi-Squared    : {CHI_SQ_V:.2f}")
            print(f" RMSE                   : {RMSE:.3f} km/s")
            print(f" NRMSE                  : {NRMSE:.3f}")
            print(f" Correlation Matrix     : \n{COR_MATRIX if COR_MATRIX is not None else 'N/A'}")
            print("--------------------------------------------------------------------\n")


        ######################################
        # Return fitted velocity profile
        ######################################
        if radius_fit is None:
            radius_fit = radius_map

        # Evaluate vel_rot(r) using lmfit, and get uncertainties via eval_uncertainty
        def vel_rot_func(r, Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n):
            Vc, Rt, s_out, _Vsys, inc, phi_delta = _denormalize_params([Vc_n, Rt_n, s_out_n, Vsys_n, inc_n, phi_delta_n])
            return self._vel_rot_tan_sout_profile(r, Vc, Rt, s_out)

        vel_rot_model = Model(vel_rot_func, independent_vars=["r"])
        vel_rot_fitted = vel_rot_model.eval(params=lm_result.params, r=radius_fit)

        # lmfit uncertainty propagation (uses covariance internally); fallback to NaN if unavailable
        try:
            vel_fit_stderr = vel_rot_model.eval_uncertainty(params=lm_result.params, r=radius_fit, sigma=1) * F_factor
        except Exception:
            vel_fit_stderr = np.full_like(vel_rot_fitted, np.nan, dtype=float)

        # Apply filter to output maps
        residuals = np.abs(vel_obs_valid) - np.abs(vel_obs_model)
        stderr = np.sqrt(1.0 / ivar_map_valid + (VEL_SYSTEM_ERROR)**2)
        STD_ERROR_RATIO = 3.0
        STD_ERROR_RATIO = RMSE / SIGMA_OBS_BAR if SIGMA_OBS_BAR > 0 else np.nan
        clip_mask_1d = np.abs(residuals) <= STD_ERROR_RATIO * stderr

        # Map the 1D mask (for valid points) back to the original map shape
        clip_mask = np.zeros_like(vel_obs_map, dtype=bool)
        clip_mask[valid_mask] = clip_mask_1d
        print(f"length of valid vel_obs: {np.sum(valid_mask)}, after clipping: {np.sum(clip_mask)}")


        radius_mask = np.full_like(radius_map, np.nan, dtype=float)
        radius_mask[clip_mask] = radius_map[clip_mask]
        vel_obs_mask = np.full_like(vel_obs_map, np.nan, dtype=float)
        vel_obs_mask[clip_mask] = vel_obs_map[clip_mask]
        ivar_mask = np.full_like(ivar_map, np.nan, dtype=float)
        ivar_mask[clip_mask] = ivar_map[clip_mask]

        fit_result = {
            'radius_obs': radius_mask,
            'vel_obs': vel_obs_mask,
            'ivar_obs': ivar_mask,
            'radius_rot': radius_fit,
            'vel_rot': vel_rot_fitted,
            'stderr_rot': vel_fit_stderr,
        }

        fit_parameters = {
            'result': 'success',
            'Vc': f"{Vc_fit:.2f}",
            'Rt': f"{Rt_fit:.2f}",
            's_out': f"{s_out_fit:.2f}",
            'Vsys': f"{Vsys_fit:.2f}",
            'inc': f"{inc_fit:.2f}",
            'phi_delta': f"{phi_delta_fit:.3f}",
            'R_max': f"{R_max:.3f}",
            'RMSE': f"{RMSE:.3f}",
            'NRMSE': f"{NRMSE:.3f}",
            'CHI_SQ_V': f"{CHI_SQ_V:.2f}",
        }
        return True, fit_result, fit_parameters


    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, plate_ifu: str) -> None:
        self.PLATE_IFU = plate_ifu
        return

    def set_fit_debug(self, debug: bool=True) -> None:
        self.fit_debug = debug
        return

    def get_inc_rad(self):
        _, inc_rad = self._calc_pa_inc()
        return inc_rad

    def get_radius_fit(self, radius_max, count: int=100) -> np.ndarray:
        radius_fit = np.linspace(0.0, radius_max, num=count)
        return radius_fit

    # observed velocity
    def get_vel_obs(self, is_filter: bool=True):
        return self._get_vel_obs_raw(type='gas', is_filter=is_filter)

    # disprojected velocity
    def get_vel_obs_disp(self, inc_rad:float, vel_sys: float, phi_delta: float=0.0):
        r_map, v_obs_map, ivar_map, phi_map = self.get_vel_obs()
        v_rot_map = self._vel_rot_disproject_profile(v_obs_map-vel_sys, inc_rad, phi_map - phi_delta)
        return r_map, v_rot_map, ivar_map

    def fit_vel_rot(self, radius_map, vel_obs_map, ivar_map, phi_map, radius_fit=None):
        return self._fit_vel_rot(radius_map, vel_obs_map, ivar_map, phi_map, radius_fit=radius_fit)


######################################################
# main function for test
######################################################
def test_process(PLATE_IFU: str):
    fits_util = FitsUtil(data_dir)
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    # print(f"DRPALL file: {drpall_file}")
    # print(f"FIREFLY file: {firefly_file}")
    # print(f"MAPS file: {maps_file}")

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)
    plot_util = PlotUtil(fits_util)

    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=None)
    vel_rot.set_PLATE_IFU(PLATE_IFU)
    vel_rot.set_fit_debug(debug=True)

    r_obs_raw, V_obs_raw, ivar_obs_raw, phi_map = vel_rot.get_vel_obs(is_filter=False)

    r_obs_map, V_obs_map, ivar_obs_map, phi_map = vel_rot.get_vel_obs()
    if np.sum(np.isfinite(V_obs_map)) < min(VEL_OBS_COUNT_THRESHOLD1, VEL_OBS_COUNT_THRESHOLD2):
        print(f"Valid data {np.sum(np.isfinite(V_obs_map))} for {PLATE_IFU}, skipping...")
        return

    pa, inc_rad = vel_rot._calc_pa_inc()
    print(f"Calculated PA: {np.degrees(pa):.3f} deg, Inc: {np.degrees(inc_rad):.3f} deg")

    r_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

    #----------------------------------------------------------------------
    # First fitting
    #----------------------------------------------------------------------
    print(f"## First fitting {PLATE_IFU} ##")
    success, fit_result, fit_params = vel_rot.fit_vel_rot(r_obs_map, V_obs_map, ivar_obs_map, phi_map, radius_fit=r_fit)
    if not success:
        print(f"Fitting rotational velocity failed for {PLATE_IFU}")
        return

    r_obs_new = fit_result['radius_obs']
    V_obs_new = fit_result['vel_obs']
    ivar_obs_new = fit_result['ivar_obs']
    r_rot_fit = fit_result['radius_rot']
    V_rot_fit = fit_result['vel_rot']
    stderr_rot_fit = fit_result['stderr_rot']
    inc_rad_fit = float(fit_params['inc'])
    V_sys_fit = float(fit_params['Vsys'])
    phi_delta_fit = float(fit_params['phi_delta'])

    # Filter fitting parameters
    data_count = np.sum(np.isfinite(V_obs_new))
    NRMSE = float(fit_params['NRMSE'])
    CHI_SQ_V = float(fit_params['CHI_SQ_V'])
    if (data_count < VEL_OBS_COUNT_THRESHOLD1) or (NRMSE > NRMSE_THRESHOLD1) or (CHI_SQ_V > CHI_SQ_V_THRESHOLD1):
        print(f"First fitting results failure for {PLATE_IFU}, COUNT: {data_count}, NRMSE: {NRMSE:.3f}, CHI_SQ_V: {CHI_SQ_V:.3f}, skipping...")
        return

    #----------------------------------------------------------------------
    # Second fitting
    #----------------------------------------------------------------------
    # print(f"## Second fitting {PLATE_IFU} ##")
    # success, fit_result, fit_params = vel_rot.fit_vel_rot(r_obs_new, V_obs_new, ivar_obs_new, phi_map, radius_fit=r_fit)
    # if not success:
    #     print(f"Fitting rotational velocity failed for {PLATE_IFU}")
    #     return

    # r_rot_fit = fit_result['radius_rot']
    # V_rot_fit = fit_result['vel_rot']
    # stderr_rot_fit = fit_result['stderr_rot']
    # inc_rad_fit = float(fit_params['inc'])
    # V_sys_fit = float(fit_params['Vsys'])
    # phi_delta_fit = float(fit_params['phi_delta'])

    # # Filter fitting parameters
    # data_count = np.sum(np.isfinite(V_obs_new))
    # NRMSE = float(fit_params['NRMSE'])
    # CHI_SQ_V = float(fit_params['CHI_SQ_V'])
    # if data_count < VEL_OBS_COUNT_THRESHOLD2 or (NRMSE > NRMSE_THRESHOLD2) or (CHI_SQ_V > CHI_SQ_V_THRESHOLD2):
    #     print(f"Second fitting results failure for {PLATE_IFU}, COUNT: {data_count}, NRMSE: {NRMSE:.3f}, CHI_SQ_V: {CHI_SQ_V:.3f}, skipping...")
    #     return

    #----------------------------------------------------------------------
    # End of second fitting
    #----------------------------------------------------------------------

    r_disp_map, V_disp_map, ivar_obs_map = vel_rot.get_vel_obs_disp(inc_rad_fit, V_sys_fit, phi_delta_fit)

    # plot_util.plot_rv_curve(r_obs_raw, V_obs_raw, title=f"[{PLATE_IFU}] Obs Raw", r_rot2_map=r_disp_map, v_rot2_map=V_disp_map, title2=f"[{PLATE_IFU}] Obs Deprojected")

    # V_obs Vs. V_rot fitted
    # plot_util.plot_rv_curve(r_obs_map, V_obs_map, title=f"[{PLATE_IFU}] Obs Raw", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2=f"[{PLATE_IFU}] Obs Fit")

    # V_disp Vs. V_rot fitted
    plot_util.plot_rv_curve(r_disp_map, V_disp_map, title=f" [{PLATE_IFU}] Obs Deproject", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2=f" [{PLATE_IFU}] Obs Fit")

    return

PLATES_FILENAME = "plateifus.txt"
def get_plate_ifu_list():
    plate_ifu_file = data_dir / PLATES_FILENAME

    with open(plate_ifu_file, 'r') as f:
        plate_ifu_list = [line.strip() for line in f if line.strip()]

    # sort the list
    plate_ifu_list.sort()
    return plate_ifu_list

TEST_PLATE_IFUS = [
    "7957-3701",
    "8078-1902",
    "10218-6102",
    "8329-6103",
    "8723-12703",
    "8723-12705",
    "7495-12704",
    "10220-12705"
]

def main():
    plate_ifu_list = []
    plate_ifu_list = get_plate_ifu_list()
    if not plate_ifu_list or len(plate_ifu_list) == 0:
        plate_ifu_list = TEST_PLATE_IFUS
    else:
        print(f"Total {len(plate_ifu_list)} plate-IFUs to process.")


    for plate_ifu in plate_ifu_list:
        print(f"\n\n================ Processing [{plate_ifu}] ================")
        try:
            test_process(plate_ifu)
        except Exception as e:
            print(f"Error processing {plate_ifu}: {e}")
            continue


# main entry
if __name__ == "__main__":
    main()

