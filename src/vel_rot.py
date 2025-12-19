
from ctypes.wintypes import PINT
from inspect import Parameter
from pathlib import Path
from tkinter import N
from turtle import shape

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit, minimize
from matplotlib import colors

# my imports
from util import plot_util
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil


######################################################################
# constants definitions
######################################################################
SNR_THRESHOLD = 10.0
PHI_LIMIT_DEG = 60.0
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation
VEL_SYSTEM_ERROR = 20.0  # km/s, floor error as systematic uncertainty in velocity measurements
NRMSE_THRESHOLD = 0.20  # threshold for normalized root mean square error to filter weak fitting
RMAX_RT_RATIO_THRESHOLD = 3.0  # threshold for Rmax / Rt to filter bad Rt fitting

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
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, phi_map: np.ndarray, snr_threshold: float = 10.0, phi_limit_deg: float = 60.0) -> np.ndarray:
        phi_delta = (phi_map + np.pi/2) % np.pi - np.pi/2
        phi_limit_rad = np.radians(phi_limit_deg)
        valid_mask = ((snr_map >= snr_threshold) & (np.abs(phi_delta) <= phi_limit_rad) & np.isfinite(vel_map))

        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]
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

    def _get_vel_obs_raw(self, type: str='gas') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG)
        # print(f"Filtered Velocity map shape: {filtered_vel_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(filtered_vel_map):.3f}, {np.nanmax(filtered_vel_map):.3f}]")
        # print(f"Velocity data before filtering: {np.sum(np.isfinite(v_obs_map))}, after filtering: {np.sum(np.isfinite(filtered_vel_map))}")

        r_obs_map = np.where(np.isfinite(filtered_vel_map), radius_h_kpc_map, np.nan)
        v_obs_map = filtered_vel_map
        ivar_obs_map = v_ivar
        phi_obs_map = azimuth_rad_map

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
        sigma = np.sqrt(1.0 / ivar + (VEL_SYSTEM_ERROR)**2)  # adding floor error
        residuals = np.abs(y_obs) - np.abs(y_model)
        loss = np.sum(residuals**2 / sigma**2)
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
    def _fit_vel_rot(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, ivar_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.01)
        radius_valid = radius_map[valid_mask]
        vel_obs_valid = vel_obs_map[valid_mask]
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'Vc': (20.0, 500.0),  # km/s
            'Rt': (np.nanmax(radius_valid)*0.01, np.nanmax(radius_valid)*1.0),  # kpc
        }

        def _denormalize_params(params_n):
            Vc_n, Rt_n = params_n
            Vc = Vc_n * (params_range['Vc'][1] - params_range['Vc'][0]) + params_range['Vc'][0]
            Rt = Rt_n * (params_range['Rt'][1] - params_range['Rt'][0]) + params_range['Rt'][0]
            return [Vc, Rt]

        ######################################
        # Fitting process using curve_fit
        ######################################
        def model_func(r, Vc_n, Rt_n):
            Vc, Rt = _denormalize_params([Vc_n, Rt_n])
            vel_rot_model = self._vel_rot_arctan_profile(r, 0.0, Vc, Rt)
            vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
            # fixed the sign of vel_obs_model to be the same as vel_valid
            vel_obs_model = np.copysign(np.abs(vel_obs_model), vel_obs_valid)
            return vel_obs_model

        # sigma: Standard Deviation of the Errors
        sigma = np.sqrt(1.0 / ivar_map_valid + (VEL_SYSTEM_ERROR)**2)  # adding floor error

        initial_guess = [0.5, 0.3] # normalized initial guesses
        bounds = ([0.0, 0.0], [1.0, 1.0])  # normalized bounds

        # Perform curve fitting
        popt, pcov = curve_fit(model_func, radius_valid, vel_obs_valid, p0=initial_guess, sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=10000)
        Vc_fit, Rt_fit = _denormalize_params(popt)
        #--------------------------------------
        # filter Rt
        #--------------------------------------
        if np.nanmax(radius_valid) / Rt_fit < RMAX_RT_RATIO_THRESHOLD:
            print(f"Error: Fitting Rotational Velocity: Rmax / Rt = {np.nanmax(radius_valid) / Rt_fit:.3f} < {RMAX_RT_RATIO_THRESHOLD}")
            return False, None, None, None


        ######################################
        # Error estimation
        ######################################
        # Reduced Chi-Squared
        vel_rot_model = self._vel_rot_arctan_profile(radius_valid, 0.0, Vc_fit, Rt_fit)
        vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
        # set the sign of vel_obs_model to be the same as vel_valid
        vel_obs_model = np.copysign(np.abs(vel_obs_model), vel_obs_valid)

        # RMSE: Root Mean Square Error
        RMSE = np.sqrt(np.nansum((vel_obs_valid - vel_obs_model)**2) / np.sum(np.isfinite(vel_obs_valid)))
        # NRMSE: Normalized Root Mean Square Error
        NRMSE = RMSE / np.mean(np.abs(vel_obs_valid))

        #--------------------------------------
        # filter weak fitting
        #--------------------------------------
        if NRMSE > NRMSE_THRESHOLD:
            print(f"Error: Fitting Rotational Velocity: NRMSE = {NRMSE:.3f} > {NRMSE_THRESHOLD}")
            return False, None, None, None

        if self.fit_debug:
            CHI_SQ_V = self._calc_chi_sq_v(vel_obs_valid, vel_obs_model, ivar_map_valid, num_params=len(popt))
            F_factor = np.maximum(np.sqrt(CHI_SQ_V), 1.0)

            # Adjusted Coefficient of Determination (R²)
            R_SQ_ADJ = self._calc_R_sq_adj(vel_obs_valid, vel_obs_model, ivar_map_valid, k=len(popt))

            # MAE: Mean Absolute Error
            MAE = np.sum(np.abs(vel_obs_valid - vel_obs_model)) / np.sum(np.isfinite(vel_obs_valid))
            mask_pos = (vel_obs_valid > 1.0)
            # MAPE: Mean Absolute Percentage Error
            MAPE = np.sum(np.abs((vel_obs_valid[mask_pos] - vel_obs_model[mask_pos]) / vel_obs_valid[mask_pos])) / np.sum(mask_pos) * 100.0
            # SMAPE: Symmetric Mean Absolute Percentage Error
            SMAPE = (100.0 / np.sum(np.isfinite(vel_obs_valid))) * np.sum(2.0 * np.abs(vel_obs_valid - vel_obs_model) / (np.abs(vel_obs_valid) + np.abs(vel_obs_model)))

            # Correlation Matrix
            COR_MATRIX = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))

            # Standard Errors of the Parameters
            perr = np.sqrt(np.diag(pcov))
            Vc_norm_err, Rt_norm_err = perr
            Vc_err, Rt_err = (
                Vc_norm_err * (params_range['Vc'][1] - params_range['Vc'][0]),
                Rt_norm_err * (params_range['Rt'][1] - params_range['Rt'][0]),
            )
            Vc_err_pct = (Vc_err / Vc_fit) * 100 if Vc_fit != 0 else np.nan
            Rt_err_pct = (Rt_err / Rt_fit) * 100 if Rt_fit != 0 else np.nan

            print(f"\n------------ Fitted Rotational Velocity (arctan curve-fit) ------------")
            print(f" IFU        : {self.PLATE_IFU}")
            print(f" Fit  Vc    : {Vc_fit:.3f} km/s, ± {Vc_err:.3f} km/s", f"({Vc_err_pct:.2f} %)")
            print(f" Fit  Rt    : {Rt_fit:.3f} kpc/h, ± {Rt_err:.3f} kpc/h", f"({Rt_err_pct:.2f} %)")
            print("--------------")
            print(f" Adjusted R²            : {R_SQ_ADJ:.4f}")
            print(f" RMSE                   : {RMSE:.3f} km/s")
            print(f" NRMSE                  : {NRMSE:.3f}")
            print("--------------")
            print(f" Reduced Chi-Squared    : {CHI_SQ_V:.3f}")
            print(f" MAE                    : {MAE:.3f} km/s")
            print(f" MAPE                   : {MAPE:.3f} %")
            print(f" SMAPE                  : {SMAPE:.2f} %")
            print(f" Correlation Matrix     : \n{COR_MATRIX}")
            print("--------------------------------------------------------------------\n")

        ######################################
        # Return fitted velocity profile
        ######################################
        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_arctan_profile(radius_fit, 0.0, Vc_fit, Rt_fit)

        ######################################
        # Standard Errors of the output fitted velocity
        ######################################
        # Using propagation of uncertainty
        #  V(r) = V0 + (2/pi) * Vc * arctan(r / Rt)
        # Partial derivatives
        dV_dVc = (2 / np.pi) * np.arctan(radius_fit / Rt_fit)
        dV_dRt = (-2 / np.pi) * Vc_fit * (radius_fit / (Rt_fit**2 + radius_fit**2))

        # Error propagation including covariance
        # sigma_f^2 = (df/dp1)^2 * sigma_p1^2 + (df/dp2)^2 * sigma_p2^2 + 2 * (df/dp1)*(df/dp2) * cov(p1, p2)
        # Note: pcov is normalized, need to scale it back to physical units

        # Scale covariance matrix to physical units
        scale_matrix = np.array([
            params_range['Vc'][1] - params_range['Vc'][0],
            params_range['Rt'][1] - params_range['Rt'][0]
        ])
        pcov_physical = pcov * np.outer(scale_matrix, scale_matrix)

        var_Vc = pcov_physical[0, 0]
        var_Rt = pcov_physical[1, 1]
        cov_Vc_Rt = pcov_physical[0, 1]

        vel_fit_var = (dV_dVc**2 * var_Vc) + (dV_dRt**2 * var_Rt) + (2 * dV_dVc * dV_dRt * cov_Vc_Rt) + (VEL_SYSTEM_ERROR)**2  # adding floor error
        vel_fit_err = np.sqrt(vel_fit_var)
        print(f"Velocity Fit Standard Errors: range: [{np.nanmin(vel_fit_err):.3f}, {np.nanmax(vel_fit_err):.3f}] km/s")


        return True, radius_fit, vel_rot_fitted, vel_fit_err

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
    def get_vel_obs(self):
        r_map, vel_obs_map, ivar_map, phi_map = self._get_vel_obs_raw(type='gas')
        return r_map, vel_obs_map, ivar_map, phi_map

    # disprojected velocity
    def get_vel_obs_disp(self):
        r_map, v_obs_map, ivar_map, phi_map = self._get_vel_obs_raw(type='gas')
        inc_rad = self.get_inc_rad()
        v_rot_map = self._vel_rot_disproject_profile(v_obs_map, inc_rad, phi_map)
        return r_map, v_rot_map, ivar_map

    def fit_vel_rot(self, radius_map, vel_obs_map, ivar_map, phi_map, radius_fit=None):
        return self._fit_vel_rot(radius_map, vel_obs_map, ivar_map, phi_map, radius_fit=radius_fit)


######################################################
# main function for test
######################################################
def test_process(PLATE_IFU: str):

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    print(f"DRPALL file: {drpall_file}")
    print(f"FIREFLY file: {firefly_file}")
    print(f"MAPS file: {maps_file}")

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)
    plot_util = PlotUtil(fits_util)

    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=None)
    vel_rot.set_PLATE_IFU(PLATE_IFU)
    vel_rot.set_fit_debug(debug=True)

    r_obs_map, V_obs_map, ivar_obs_map, phi_map = vel_rot.get_vel_obs()
    print(f"V_obs_map range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}] km/s")
    pa, inc_rad = vel_rot._calc_pa_inc()
    print(f"Calculated PA: {np.degrees(pa):.3f} deg, Inc: {np.degrees(inc_rad):.3f} deg")


    r_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

    r_disp_map, V_disp_map, ivar_obs_map = vel_rot.get_vel_obs_disp()

    # Fitting rotational velocity
    success, r_rot_fit, V_rot_fit , V_rot_fit_err = vel_rot._fit_vel_rot(r_obs_map, V_obs_map, ivar_obs_map, phi_map, radius_fit=r_fit)
    if not success:
        print(f"Fitting rotational velocity failed for {PLATE_IFU}")
        return

    print("#######################################################")
    print(f"# {PLATE_IFU} calculate results")
    print("#######################################################")
    print(f"Obs Radius shape: {r_obs_map.shape}, range: [{np.nanmin(r_obs_map):.3f}, {np.nanmax(r_obs_map):.3f}] kpc/h")
    print(f"Obs Velocity shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}]")
    print(f"Obs Deprojected Velocity shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):.3f}, {np.nanmax(V_disp_map):.3f}]")
    print(f"Fitted Rot Velocity (arctan curve-fit) shape: {V_rot_fit.shape}, range: [{np.nanmin(V_rot_fit):.3f}, {np.nanmax(V_rot_fit):.3f}]")
    print(f"Fitted Rot Velocity Error (arctan curve-fit) shape: {V_rot_fit_err.shape}, range: [{np.nanmin(V_rot_fit_err):.3f}, {np.nanmax(V_rot_fit_err):.3f}]")

    # plot_util.plot_rv_curve(r_obs_map, V_obs_map, title=f"[{PLATE_IFU}] Obs Raw", r_rot2_map=r_disp_map, v_rot2_map=V_disp_map, title2=f"[{PLATE_IFU}] Obs Deprojected")
    # plot_util.plot_rv_curve(r_obs_map, V_obs_map, title=f"[{PLATE_IFU}] Obs Raw", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2=f"[{PLATE_IFU}] Obs Fit")
    plot_util.plot_rv_curve(r_disp_map, V_disp_map, title=f" [{PLATE_IFU}] Obs Deproject", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2=f" [{PLATE_IFU}] Obs Fit")
    return


def main():
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

    for plate_ifu in TEST_PLATE_IFUS:
        print(f"\n\n================ Processing {plate_ifu} ================")
        test_process(plate_ifu)


# main entry
if __name__ == "__main__":
    main()

