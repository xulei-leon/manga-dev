
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
DEFAULT_BETA = 0.5   # sigma_phi^2 / sigma_R^2
DEFAULT_GAMMA = 0.6  # sigma_z / sigma_R



######################################################################
# class
######################################################################

class VelRot:
    drpall_util = None
    firefly_util = None
    maps_util = None
    plot_util = None
    
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
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, phi_delta: np.ndarray, snr_threshold: float = 10.0, phi_limit_deg: float = 60.0) -> np.ndarray:
        phi_delta = (phi_delta + np.pi/2) % np.pi - np.pi/2
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
    def _vel_obs_project_profile(self, vel_rot: np.ndarray, inc: float, phi_delta: np.ndarray) -> np.ndarray:
        phi_delta = (phi_delta + np.pi) % (2 * np.pi) # WHY?? rotate pi to make sure cos() works correctly 
        correction = np.sin(inc) * np.cos(phi_delta)
        vel_obs = vel_rot * correction
        return vel_obs
    
    # formula: V_rot = V_obs / (sin(i) * cos(phi - phi_0))
    def _vel_rot_disproject_profile(self, vel_obs: np.ndarray, inc: float, phi_delta: np.ndarray) -> np.ndarray:
        phi_delta = (phi_delta + np.pi) % (2 * np.pi) # WHY?? rotate pi to make sure cos() works correctly 
        correction = np.sin(inc) * np.cos(phi_delta)
        with np.errstate(divide='ignore', invalid='ignore'):
            vel_rot = np.where(correction != 0, vel_obs / correction, np.nan)
        vel_rot = np.where(vel_obs < 0, -vel_rot, vel_rot)
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
    # Negativity: the reduced chi-squared is not good enough
    # Formula: V(r) = V0 + (2/pi) * Vc * arctan(r / Rt)
    def _vel_rot_arctan_profile(self, r: np.ndarray, V0: float, Vc: float, Rt: float) -> np.ndarray:
        return V0 + (2 / np.pi) * Vc * np.arctan(r / Rt)
    
    # Formula: V(r) = V0 * (1 - e^(-r / Rt)) (1 + alpha * r / Rt)
    # Negativity: The alpha parameter parameter may have bad standard errors.
    def _vel_rot_polyex_profile(self, r: np.ndarray, V0: float, Rt: float, alpha: float) -> np.ndarray:
        return V0 * (1 - np.exp(-r / Rt)) * (1 + alpha * r / Rt)
    

    ################################################################################
    # Fitting methods
    ################################################################################
    def _calc_loss(self, vel, vel_fit, ivar):
        # sigma = np.sqrt(1.0 / ivar)
        sigma = np.sqrt(1.0 / ivar + (10.0)**2)  # adding 10 km/s in quadrature
        residuals = np.abs(vel) - np.abs(vel_fit)
        loss = np.sum(residuals**2 / sigma**2)
        return loss

    def _calc_chi_sq_v(self, vel_obs: np.ndarray, vel_model: np.ndarray, ivar_map: np.ndarray, num_params: int) -> float:
        chi_sq = self._calc_loss(vel_obs, vel_model, ivar_map)
        N = np.sum(np.isfinite(vel_obs) & np.isfinite(ivar_map))
        k = num_params
        dof = max(N - k, 1)
        chi_sq_v = chi_sq / dof
        return chi_sq_v
   
    #  used the minimize method for fitting
    def _fit_vel_rot_tan(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, ivar_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & np.isfinite(ivar_map) & (radius_map > 0.01)
        radius_valid = radius_map[valid_mask]
        vel_obs_valid = vel_obs_map[valid_mask]
        vel_obs_valid = np.abs(vel_obs_valid)
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()

        # normal all fit parameters
        param_ranges = {
            'Vc': (20.0, 500.0),  # km/s
            'Rt': (np.nanmax(radius_valid)*0.1, np.nanmax(radius_valid)*1.0),  # kpc/h
            'Sout': (-50.0, 50.0)  # km/s
        }

        def _denormal_params(params_n):
            Vc_n, Rt_n, Sout_n = params_n
            Vc = Vc_n * (param_ranges['Vc'][1] - param_ranges['Vc'][0]) + param_ranges['Vc'][0]
            Rt = Rt_n * (param_ranges['Rt'][1] - param_ranges['Rt'][0]) + param_ranges['Rt'][0]
            Sout = Sout_n * (param_ranges['Sout'][1] - param_ranges['Sout'][0]) + param_ranges['Sout'][0]
            return [Vc, Rt, Sout]

        # ivar: Inverse Variance
        # ivar = 1 / sigma^2
        # Loss = sum((vel_obs - vel_model)^2 / sigma^2)
        def _loss_function(params):
            Vc, Rt, Sout = _denormal_params(params)
            vel_rot_model = self._vel_rot_tan_sout_profile(radius_valid, Vc, Rt, Sout)
            vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
            loss = self._calc_loss(vel_obs_valid, vel_obs_model, ivar_map_valid)
            return loss
        
        # Initial guesses for Vc, Rt, Sout
        initial_guess = [0.5, 0.5, 0.1]  # normalized initial guesses
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # normalized bounds
        result = minimize(_loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        Vc_fit, Rt_fit, Sout_fit = _denormal_params(result.x)

        # Reduced chi-squared
        vel_rot_model = self._vel_rot_tan_sout_profile(radius_valid, Vc_fit, Rt_fit, Sout_fit)
        vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
        chi_sq_v = self._calc_chi_sq_v(vel_obs_valid, vel_obs_model, ivar_map_valid, num_params=len(result.x))
        F_factor = np.sqrt(chi_sq_v)

        # Covariance Matrix
        C = result.hess_inv.todense()

        # Correlation Matrix
        correlation_matrix = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))


        # Standard Errors of the Parameters
        param_errors = F_factor * np.sqrt(np.diag(C))
        Vc_norm_err, Rt_norm_err, Sout_norm_err = param_errors
        Vc_err, Rt_err, Sout_err = (
            Vc_norm_err * (param_ranges['Vc'][1] - param_ranges['Vc'][0]),
            Rt_norm_err * (param_ranges['Rt'][1] - param_ranges['Rt'][0]),
            Sout_norm_err * (param_ranges['Sout'][1] - param_ranges['Sout'][0]),
        )

        Vc_err_pct = (Vc_err / Vc_fit) * 100 if Vc_fit != 0 else np.nan
        Rt_err_pct = (Rt_err / Rt_fit) * 100 if Rt_fit != 0 else np.nan
        Sout_err_pct = (Sout_err / Sout_fit) * 100 if Sout_fit != 0 else np.nan

        print(f"\n------------ Fitted Total Rotational Velocity (tan) ------------")
        print(f" IFU        : {self.PLATE_IFU}")
        print(f" Fit  Vc    : {Vc_fit:.3f} km/s, ± {Vc_err:.3f} km/s", f"({Vc_err_pct:.2f} %)")
        print(f" Fit  Rt    : {Rt_fit:.3f} kpc/h, ± {Rt_err:.3f} kpc/h", f"({Rt_err_pct:.2f} %)")
        print(f" Fit  s_out : {Sout_fit:.3f} km/s, ± {Sout_err:.3f} km/s", f"({Sout_err_pct:.2f} %)")
        print(f" Reduced chi-squared    : {chi_sq_v:.3f}")
        print(f"Correlation Matrix      :")
        print(correlation_matrix)
        print("------------------------------------------------------------------------------\n")

        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_tan_sout_profile(radius_fit, Vc_fit, Rt_fit, Sout_fit)
        return radius_fit, vel_rot_fitted
    

    def _fit_vel_rot_arctan(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, ivar_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.01)
        radius_valid = radius_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]
        # vel_valid = np.abs(vel_valid)
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()

        # normal all fit parameters
        param_ranges = {
            'Vc': (20.0, 500.0),  # km/s
            'Rt': (np.nanmax(radius_valid)*0.01, np.nanmax(radius_valid)*1.0),  # kpc/h
        }

        def _denormal_params(params_n):
            Vc_n, Rt_n = params_n
            Vc = Vc_n * (param_ranges['Vc'][1] - param_ranges['Vc'][0]) + param_ranges['Vc'][0]
            Rt = Rt_n * (param_ranges['Rt'][1] - param_ranges['Rt'][0]) + param_ranges['Rt'][0]
            return [Vc, Rt]


        # ivar: Inverse Variance
        # ivar = 1 / sigma^2
        # Loss = sum((vel_obs - vel_model)^2 / sigma^2)
        def _loss_function(params):
            Vc, Rt = _denormal_params(params)
            vel_rot_model = self._vel_rot_arctan_profile(radius_valid, 0.0, Vc, Rt)
            vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
            loss = self._calc_loss(vel_valid, vel_obs_model, ivar_map_valid)
            return loss

        # Initial guesses for V0, Vc, Rt, inc
        # Using normalized parameters for better convergence
        initial_guess = [0.5, 0.3]
        bounds = [(0.0, 1.0), (0.0, 1.0)]  # normalized bounds

        result = minimize(_loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        Vc_fit, Rt_fit = _denormal_params(result.x)

        ######################################
        # Error estimation
        ######################################

        # Reduced chi-squared
        vel_rot_model = self._vel_rot_arctan_profile(radius_valid, 0, Vc_fit, Rt_fit)
        vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
        chi_sq_v = self._calc_chi_sq_v(vel_valid, vel_obs_model, ivar_map_valid, num_params=len(result.x))
        F_factor = np.sqrt(chi_sq_v)

        # Covariance Matrix
        C = result.hess_inv.todense()
        # Correlation Matrix
        correlation_matrix = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))

        # Standard Errors of the Parameters
        param_errors = F_factor * np.sqrt(np.diag(C))
        Vc_norm_err, Rt_norm_err = param_errors
        Vc_err, Rt_err = (
            Vc_norm_err * (param_ranges['Vc'][1] - param_ranges['Vc'][0]),
            Rt_norm_err * (param_ranges['Rt'][1] - param_ranges['Rt'][0]),
        )
        Vc_err_pct = (Vc_err / Vc_fit) * 100 if Vc_fit != 0 else np.nan
        Rt_err_pct = (Rt_err / Rt_fit) * 100 if Rt_fit != 0 else np.nan


        print(f"\n------------ Fitted Total Rotational Velocity (arctan) ------------")
        print(f" IFU        : {self.PLATE_IFU}")
        print(f" Fit  Vc    : {Vc_fit:.3f} km/s, ± {Vc_err:.3f} km/s", f"({Vc_err_pct:.2f} %)")
        print(f" Fit  Rt    : {Rt_fit:.3f} kpc/h, ± {Rt_err:.3f} kpc/h", f"({Rt_err_pct:.2f} %)")
        print(f" Reduced chi-squared    : {chi_sq_v:.3f}")
        print(f"Correlation Matrix      :")
        print(correlation_matrix)
        print("------------------------------------------------------------------------------\n")

        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_arctan_profile(radius_fit, 0, Vc_fit, Rt_fit)
        return radius_fit, vel_rot_fitted
    

    def _fit_vel_rot_polyex(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, ivar_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.01)
        radius_valid = radius_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]
        vel_valid = np.abs(vel_valid)
        ivar_map_valid = ivar_map[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        inc_act = self.get_inc_rad()

        param_ranges = {
            'V0': (20.0, 500.0),  # km/s
            'Rt': (np.nanmax(radius_valid)*0.01, np.nanmax(radius_valid)*1.0),  # kpc/h
            'alpha': (-1.0, 1.0),
        }

        def _denormal_params(params_n):
            V0_n, Rt_n, alpha_n = params_n
            V0 = V0_n * (param_ranges['V0'][1] - param_ranges['V0'][0]) + param_ranges['V0'][0]
            Rt = Rt_n * (param_ranges['Rt'][1] - param_ranges['Rt'][0]) + param_ranges['Rt'][0]
            alpha = alpha_n * (param_ranges['alpha'][1] - param_ranges['alpha'][0]) + param_ranges['alpha'][0]
            return [V0, Rt, alpha]

        # ivar: Inverse Variance
        # ivar = 1 / sigma^2
        # Loss = sum((vel_obs - vel_model)^2 / sigma^2)
        def _loss_function(params):
            V0, Rt, alpha = _denormal_params(params)
            vel_rot_model = self._vel_rot_polyex_profile(radius_valid, V0, Rt, alpha)
            vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
            loss = self._calc_loss(vel_valid, vel_obs_model, ivar_map_valid)
            return loss

        # Initial guesses for V0, Rt, alpha
        initial_guess = [0.5, 0.3, 0.0] # normalized initial guesses
        # normalized bounds
        bounds = [
                    (0.0, 1.0), 
                    (0.0, 1.0), 
                    (0.0, 1.0)
                ]

        result = minimize(_loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        V0_fit, Rpe_fit, alpha_fit = _denormal_params(result.x)

        # Reduced chi-squared
        vel_rot_model = self._vel_rot_polyex_profile(radius_valid, V0_fit, Rpe_fit, alpha_fit)
        vel_obs_model = self._vel_obs_project_profile(vel_rot_model, inc_act, phi_map_valid)
        chi_sq_v = self._calc_chi_sq_v(vel_valid, vel_obs_model, ivar_map_valid, num_params=len(result.x))
        F_factor = np.sqrt(chi_sq_v)

        # Covariance Matrix
        C = result.hess_inv.todense()

        # Correlation Matrix
        correlation_matrix = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))

        # Standard Errors of the Parameters
        param_errors = F_factor * np.sqrt(np.diag(C))
        V0_norm_err, Rpe_norm_err, alpha_norm_err = param_errors
        V0_err, Rpe_err, alpha_err = (
            V0_norm_err * (param_ranges['V0'][1] - param_ranges['V0'][0]),
            Rpe_norm_err * (param_ranges['Rt'][1] - param_ranges['Rt'][0]),
            alpha_norm_err * (param_ranges['alpha'][1] - param_ranges['alpha'][0]),
        )

        V0_err_pct = (V0_err / V0_fit) * 100 if V0_fit != 0 else np.nan
        Rpe_err_pct = (Rpe_err / Rpe_fit) * 100 if Rpe_fit != 0 else np.nan
        alpha_err_pct = (alpha_err / alpha_fit) * 100 if alpha_fit != 0 else np.nan

        print(f"\n------------ Fitted Total Rotational Velocity (polyex) ------------")
        print(f" IFU        : {self.PLATE_IFU}")
        print(f" Fit  V0    : {V0_fit:.3f} km/s ± {V0_err:.3f} km/s ({V0_err_pct:.2f}%)")
        print(f" Fit  Rt   : {Rpe_fit:.3f} kpc/h ± {Rpe_err:.3f} kpc/h ({Rpe_err_pct:.2f}%)")
        print(f" Fit  alpha : {alpha_fit:.3f} ± {alpha_err:.3f} ({alpha_err_pct:.2f}%)")
        print(f" Reduced chi-squared    : {chi_sq_v:.3f}")
        print(f"Correlation Matrix      :")
        print(correlation_matrix)
        print("------------------------------------------------------------------------------\n")
 
        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_polyex_profile(radius_fit, V0_fit, Rpe_fit, alpha_fit)
        return radius_fit, vel_rot_fitted

    # ivar = 1 / sigma^2
    def _calc_residuals(self, vel_obs, vel_fit, ivar):
        residuals = np.abs(vel_obs) - vel_fit
        # standardized residuals = residuals / sigma
        # sigma = np.sqrt(1 / ivar)
        standardized_residuals = residuals * np.sqrt(ivar)
        return residuals, standardized_residuals

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, plate_ifu: str) -> None:
        self.PLATE_IFU = plate_ifu
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
        radius_fitted, vel_rot_fitted =  self._fit_vel_rot_arctan(radius_map, vel_obs_map, ivar_map, phi_map, radius_fit=radius_fit)
        return radius_fitted, vel_rot_fitted



######################################################
# main function for test
######################################################
def main():
    PLATE_IFU = "8723-12703"

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
    r_obs_map, V_obs_map, ivar_obs_map, phi_map = vel_rot.get_vel_obs()
    r_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

    r_disp_map, V_disp_map, ivar_obs_map = vel_rot.get_vel_obs_disp()
    r_rot_fit, V_rot_fit = vel_rot._fit_vel_rot_tan(r_obs_map, V_obs_map, ivar_obs_map, phi_map, radius_fit=r_fit)
    r_rot_fit2, V_rot_fit2 = vel_rot._fit_vel_rot_arctan(r_obs_map, V_obs_map, ivar_obs_map, phi_map, radius_fit=r_fit)
    r_rot_fit3, V_rot_fit3 = vel_rot._fit_vel_rot_polyex(r_obs_map, V_obs_map, ivar_obs_map, phi_map, radius_fit=r_fit)

    print("#######################################################")
    print("# calculate results")
    print("#######################################################")
    print(f"Obs Radius shape: {r_obs_map.shape}, range: [{np.nanmin(r_obs_map):.3f}, {np.nanmax(r_obs_map):.3f}] kpc/h")
    print(f"Obs Velocity shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}]")
    print(f"Obs Deprojected Velocity shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):.3f}, {np.nanmax(V_disp_map):.3f}]")
    print(f"Obs Inverse Variance shape: {ivar_obs_map.shape}, range: [{np.nanmin(ivar_obs_map):.3f}, {np.nanmax(ivar_obs_map):.3f}]")
    print(f"Fitted Rot Velocity (Minimize) shape: {V_rot_fit.shape}, range: [{np.nanmin(V_rot_fit):.3f}, {np.nanmax(V_rot_fit):.3f}]")

    # plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs Raw", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2="Obs Fit")
    # plot_util.plot_rv_curve(r_disp_map, V_disp_map, title="Obs Deproject", r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2="Obs Fit tan")
    # plot_util.plot_rv_curve(r_disp_map, V_disp_map, title="Obs Deproject", r_rot2_map=r_rot_fit2, v_rot2_map=V_rot_fit2, title2="Obs Fit arctan")
    # plot_util.plot_rv_curve(r_disp_map, V_disp_map, title="Obs Deproject", r_rot2_map=r_rot_fit3, v_rot2_map=V_rot_fit3, title2="Obs Fit polyex")
    return


# main entry
if __name__ == "__main__":
    main()

