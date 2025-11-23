
from ctypes.wintypes import PINT
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




PLATE_IFU = "8723-12705"
# PLATE_IFU = "8723-12703"

# constants definitions
SNR_THRESHOLD = 10.0
PHI_LIMIT_DEG = 60.0
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation
DEFAULT_BETA = 0.5   # sigma_phi^2 / sigma_R^2
DEFAULT_GAMMA = 0.6  # sigma_z / sigma_R

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
        print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")

        inc = self._calc_inc(ba, ba_0=BA_0)
        print(f"Calculated Inclination i: {np.degrees(inc):.2f} deg")
        # Convert PA from degrees to radians and rotate so North is at +90°
        pa = np.mod(np.radians(phi), 2 * np.pi)
        return pa, inc

    def _get_vel_obs(self, type: str='gas') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset_x, offset_y = self.maps_util.get_sky_offsets()
        print(f"Sky offsets shape: {offset_x.shape}, X offset: [{np.nanmin(offset_x):.3f}, {np.nanmax(offset_x):.3f}] arcsec")

        # R: radial distance map
        radius_map, radius_h_kpc_map, azimuth_map = self.maps_util.get_radius_map()
        print(f"r_map: [{np.nanmin(radius_map):.3f}, {np.nanmax(radius_map):.3f}] spaxel,", f"shape: {radius_map.shape}")
        print(f"r_h_kpc_map: [{np.nanmin(radius_h_kpc_map):.3f}, {np.nanmax(radius_h_kpc_map):.3f}] kpc,", f"shape: {radius_h_kpc_map.shape}")
        print(f"azimuth_map: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] deg,", f"shape: {azimuth_map.shape}")

        # SNR: signal-to-noise ratio map
        snr_map = self.maps_util.get_snr_map()
        print(f"SNR map shape: {snr_map.shape}, SNR range: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

        ra_map, dec_map = self.maps_util.get_skycoo_map()
        print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

        ## Get the gas velocity map (H-alpha)
        v_obs_gas_map, _gv_unit, _gv_ivar = self.maps_util.get_eml_vel_map()
        print(f"Gas velocity map shape: {v_obs_gas_map.shape}, Unit: {_gv_unit}, Velocity: [{np.nanmin(v_obs_gas_map):.3f}, {np.nanmax(v_obs_gas_map):.3f}] {_gv_unit}")
        eml_uindx = self.maps_util.get_emli_uindx()
        print(f"Gas Unique indices shape: {eml_uindx.shape}")

        ## Get the stellar velocity map
        v_obs_stellar_map, _sv_unit, _ = self.maps_util.get_stellar_vel_map()
        print(f"Stellar velocity map shape: {v_obs_stellar_map.shape}, Unit: {_sv_unit}, Velocity: [{np.nanmin(v_obs_stellar_map):.3f}, {np.nanmax(v_obs_stellar_map):.3f}] {_sv_unit}")
        stellar_uindx = self.maps_util.get_stellar_uindx()
        print(f"Stellar Unique indices shape: {stellar_uindx.shape}")

        # Velocity correction
        if type == 'gas':
            v_obs_map = v_obs_gas_map
            v_unit = _gv_unit
            v_uindx = eml_uindx
        else:
            v_obs_map = v_obs_stellar_map
            v_unit = _sv_unit
            v_uindx = stellar_uindx
        
        azimuth_rad_map = np.radians(azimuth_map)

        filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG)
        print(f"Filtered Velocity map shape: {filtered_vel_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(filtered_vel_map):.3f}, {np.nanmax(filtered_vel_map):.3f}]")
        print(f"Velocity data before filtering: {np.sum(np.isfinite(v_obs_map))}, after filtering: {np.sum(np.isfinite(filtered_vel_map))}")

        v_obs_map = filtered_vel_map
        r_obs_map = radius_h_kpc_map
        phi_obs_map = azimuth_rad_map

        return r_obs_map, v_obs_map, phi_obs_map
    
    def _get_radius(self) -> np.ndarray:
        _, radius_h_kpc_map, _ = self.maps_util.get_radius_map()
        return radius_h_kpc_map


    ################################################################################
    # rotation curve fitting procedures
    ################################################################################

    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).
    # Formula: V_obs = V_rot * (sin(i) * cos(phi - phi_0))
    def _calc_vel_obs_from_rot(self, vel_rot: np.ndarray, inc: float, phi_delta: np.ndarray) -> np.ndarray:
        phi_delta = (phi_delta + np.pi) % (2 * np.pi) # WHY?? rotate pi to make sure cos() works correctly 
        correction = np.sin(inc) * np.cos(phi_delta)
        vel_obs = vel_rot * correction
        return vel_obs
    
    def _calc_vel_rot_from_obs(self, vel_obs: np.ndarray, inc: float, phi_delta: np.ndarray) -> np.ndarray:
        phi_delta = (phi_delta + np.pi) % (2 * np.pi) # WHY?? rotate pi to make sure cos() works correctly 
        correction = np.sin(inc) * np.cos(phi_delta)
        with np.errstate(divide='ignore', invalid='ignore'):
            vel_rot = np.where(correction != 0, vel_obs / correction, np.nan)
        vel_rot = np.where(vel_obs < 0, -vel_rot, vel_rot)
        return vel_rot


    # Formula: V(r) = Vc * tanh(r / Rt) + s_out * r
    def _vel_rot_profile_tanh(self, r: np.ndarray, Vc: float, Rt: float, s_out: float) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + s_out * r
    
    # Formula: V(r) = Vc * tanh(r / Rt) * (1 + beta * r / Rmax)
    def _vel_rot_profile_tanh2(self, r: np.ndarray, Vc: float, Rt: float, beta: float, Rmax: float) -> np.ndarray:
        return Vc * np.tanh(r / Rt) * (1 + beta * r / Rmax)
        
    # Formula: V(r) = V0 + (2/pi) * Vc * arctan(r / Rt) 
    # def _vel_rot_profile_arctan(self, r: np.ndarray, V0: float, Vc: float, Rt: float) -> np.ndarray:
    #     return V0 + (2 / np.pi) * Vc * np.arctan(r / Rt)
    
    # Formula: V(r) = V0 * (1 - e^(-r / Rpe)) (1 + alpha * r / Rpe)
    def _vel_rot_profile_polyex(self, r: np.ndarray, V0: float, Rpe: float, alpha: float) -> np.ndarray:
        return V0 * (1 - np.exp(-r / Rpe)) * (1 + alpha * r / Rpe)
    
    #  V_obs = (Vc * tanh(r / Rt) + s_out * r) * (sin(i) * cos(phi_delta))
    def _vel_obs_fit_profile(self, r: np.ndarray, Vc: float, Rt: float, s_out: float, inc: float, phi_delta: np.ndarray) -> np.ndarray:
        vel_rot = self._vel_rot_profile_tanh(r, Vc, Rt, s_out)
        vel_obs = self._calc_vel_obs_from_rot(vel_rot, inc, phi_delta)
        return vel_obs
    
    # used the minimum χ 2 method for fitting
    def _fit_vel_rot(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, phi_delta_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Fit the rotation curve using the experience curve function."""
        # Flatten the maps and remove NaN values
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.1) & np.isfinite(phi_delta_map)
        radius_valid = radius_map[valid_mask]
        phi_delta_valid = phi_delta_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]


        phi_0, inc_0 = self._calc_pa_inc()
        
        # Fix inclination to the photometric value to avoid Vc-sin(i) degeneracy
        def fit_func_partial(r, Vc, Rt, s_out):
            return self._vel_obs_fit_profile(r, Vc, Rt, s_out, inc_0, phi_delta_valid)       

        Xdata = radius_valid
        Ydata = vel_valid

        r_max = np.nanmax(radius_valid)
        
        # Adjusted initial guesses and bounds based on expected values (Rt~2.4, s_out~2.9)
        p0 = [100.0, r_max*0.5, 0.0]  # Initial guesses for Vc, Rt, s_out
        lb = [1e-6, r_max*0.1, -50.0]  # Lower bounds
        ub = [500.0, r_max, 50.0]  # Upper bounds

        popt, pcov = curve_fit(fit_func_partial, Xdata, Ydata, p0=p0, bounds=(lb, ub), method='trf')
        Vc_fit, Rt_fit, s_out_fit = popt
        inc_fit = inc_0

        print(f"\n------------  IFU [{self.PLATE_IFU}] Fitted parameters  ------------")
        print(f"Fit  Vc: {Vc_fit:.3f} km/s")
        print(f"Fit  Rt: {Rt_fit:.3f} kpc/h")
        print(f"Fit  s_out: {s_out_fit:.3f} km/s")
        print(f"fixed  i: {np.degrees(inc_fit):.3f} deg")
        print(f"Fixed  phi_0: {np.degrees(phi_0):.3f} deg")

        print(f" error estimates:")
        perr = np.sqrt(np.diag(pcov))
        print(f"  Vc error: {perr[0]:.3f} km/s")
        print(f"  Rt error: {perr[1]:.3f} kpc/h")
        print(f"  s_out error: {perr[2]:.3f} km/s")
        print("------------------------------------------------------------------------------\n")


        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_profile_tanh(radius_fit, Vc_fit, Rt_fit, s_out_fit)
        return radius_fit, vel_rot_fitted
    

    #  used the minimize method for fitting
    def _fit_vel_rot_tanh_minimize(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Fit the rotation curve using the minimize method."""
        # Flatten the maps and remove NaN values
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.1)
        radius_valid = radius_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]
        vel_valid = np.abs(vel_valid)

        # Fix inclination to the photometric value to avoid Vc-sin(i) degeneracy
        def residuals(params):
            Vc, Rt, Sout = params
            vel_model = self._vel_rot_profile_tanh(radius_valid, Vc, Rt, Sout)
            return np.sum((vel_valid - vel_model) ** 2)
        
        phi_0, inc_0 = self._calc_pa_inc()

        r_max = np.nanmax(radius_valid)
        v_obs_max = np.nanmax(vel_valid)
        Vc_0 = v_obs_max / np.sin(inc_0)
        
        # Initial guesses for Vc, Rt, Sout
        initial_guess = [Vc_0, r_max*0.2, 0.0]
        bounds = [(50, 500.0), (r_max*0.1, r_max), (-50.0, 50.0)]

        result = minimize(residuals, initial_guess, bounds=bounds, method='L-BFGS-B')
        Vc_fit, Rt_fit, Sout_fit = result.x

        print(f"\n------------  IFU [{self.PLATE_IFU}] Fitted parameters (minimize)  ------------")
        print(f"Fit  Vc: {Vc_fit:.3f} km/s")
        print(f"Fit  Rt: {Rt_fit:.3f} kpc/h")
        print(f"Fit  s_out: {Sout_fit:.3f} km/s")
        print("------------------------------------------------------------------------------\n")

        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_profile_tanh(radius_fit, Vc_fit, Rt_fit, Sout_fit)
        # vel_rot_fitted = np.where(vel_obs_map < 0, -vel_rot_fitted, vel_rot_fitted)
        return radius_fit, vel_rot_fitted


    def _fit_vel_rot_polyex_minimize(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Fit the rotation curve using the minimize method."""
        # Flatten the maps and remove NaN values
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.1)
        radius_valid = radius_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]
        vel_valid = np.abs(vel_valid)

        # Fix inclination to the photometric value to avoid Vc-sin(i) degeneracy
        def residuals(params):
            V0, Rpe, alpha = params
            vel_model = self._vel_rot_profile_polyex(radius_valid, V0, Rpe, alpha)
            return np.sum((vel_valid - vel_model) ** 2)
        
        r_max = np.nanmax(radius_valid)
        
        # Initial guesses for Vc, Rt, Sout
        initial_guess = [100, r_max*0.5, 0.0]
        bounds = [(50, 500.0), (0.3, r_max), (-0.3, 0.3)]

        result = minimize(residuals, initial_guess, bounds=bounds, method='L-BFGS-B')
        V0_fit, Rpe_fit, alpha_fit = result.x

        print(f"\n------------  IFU [{self.PLATE_IFU}] Fitted parameters (minimize)  ------------")
        print(f"Fit  V0: {V0_fit:.3f} km/s")
        print(f"Fit  Rpe: {Rpe_fit:.3f} kpc/h")
        print(f"Fit  alpha: {alpha_fit:.3f} km/s")
        print("------------------------------------------------------------------------------\n")

        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self._vel_rot_profile_polyex(radius_fit, V0_fit, Rpe_fit, alpha_fit)
        # vel_rot_fitted = np.where(vel_obs_map < 0, -vel_rot_fitted, vel_rot_fitted)
        return radius_fit, vel_rot_fitted

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, plate_ifu: str) -> None:
        self.PLATE_IFU = plate_ifu
        return

    def get_gas_vel_obs(self):
        radius_map, vel_obs_map, phi_map = self._get_vel_obs(type='gas')
        return radius_map, vel_obs_map, phi_map

    def get_stellar_vel_obs(self):
        radius_map, vel_obs_map, phi_map = self._get_vel_obs(type='stellar')
        return radius_map, vel_obs_map, phi_map

    def fit_rot_vel(self, radius_map, vel_obs_map, phi_map, radius_fit=None):
        radius_fitted, vel_rot_fitted =  self._fit_vel_rot(radius_map, vel_obs_map, phi_map, radius_fit=radius_fit)
        return radius_fitted, vel_rot_fitted
    
    def fit_rot_vel_minimize(self, radius_map, vel_obs_map, radius_fit=None):
        radius_fitted, vel_rot_fitted =  self._fit_vel_rot_tanh_minimize(radius_map, vel_obs_map, radius_fit=radius_fit)
        return radius_fitted, vel_rot_fitted

    def get_inc_rad(self):
        _, inc_rad = self._calc_pa_inc()
        return inc_rad
    
    def get_vel_obs_deprojected(self):
        r_map, v_obs_map, phi_map = self._get_vel_obs(type='gas')
        inc_rad = self.get_inc_rad()
        v_rot_map = self._calc_vel_rot_from_obs(v_obs_map, inc_rad, phi_map)
        return r_map, v_rot_map
    
    def get_radius_fit(self, count: int=100) -> np.ndarray:
        radius_map = self._get_radius()
        radius_max = np.nanmax(radius_map)

        radius_fit = np.linspace(0.0, radius_max, num=count)
        return radius_fit


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
    r_fit = vel_rot.get_radius_fit(count=1000)
    r_obs_map, V_obs_map, phi_map = vel_rot.get_gas_vel_obs()
    _, V_obs_map_deprojected = vel_rot.get_vel_obs_deprojected()
    r_rot_fitted, V_rot_fitted = vel_rot.fit_rot_vel(r_obs_map, V_obs_map, phi_map, radius_fit=r_fit)
    r_rot_fitted_mini, V_rot_fitted_mini = vel_rot.fit_rot_vel_minimize(r_obs_map, V_obs_map_deprojected, radius_fit=r_obs_map)

    print("#######################################################")
    print("# calculate results")
    print("#######################################################")
    print(f"Obs Radius shape: {r_obs_map.shape}, range: [{np.nanmin(r_obs_map):.3f}, {np.nanmax(r_obs_map):.3f}] kpc/h")
    print(f"Obs Velocity shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}]")
    print(f"Obs Phi shape: {phi_map.shape}, range: [{np.nanmin(phi_map):.3f}, {np.nanmax(phi_map):.3f}] rad")
    print(f"Fitted Rot Velocity shape: {V_rot_fitted.shape}, range: [{np.nanmin(V_rot_fitted):.3f}, {np.nanmax(V_rot_fitted):.3f}]")
    print(f"Fitted Rot Velocity (Minimize) shape: {V_rot_fitted_mini.shape}, range: [{np.nanmin(V_rot_fitted_mini):.3f}, {np.nanmax(V_rot_fitted_mini):.3f}]")

    plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs Fitted", r_rot2_map=r_rot_fitted, v_rot2_map=V_rot_fitted, title2="Rot Fitted")
    plot_util.plot_rv_curve(r_obs_map, V_obs_map_deprojected, title="Obs Deprojected", r_rot2_map=r_rot_fitted_mini, v_rot2_map=V_rot_fitted_mini, title2="Rot Fitted Minimize")
    plot_util.plot_rv_curve(r_rot_fitted, V_rot_fitted, title="Rot Fitted", r_rot2_map=r_rot_fitted_mini, v_rot2_map=V_rot_fitted_mini, title2="Rot Minimize Fitted")
    return


# main entry
if __name__ == "__main__":
    main()

