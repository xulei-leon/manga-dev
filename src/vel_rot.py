from hashlib import sha1
from operator import le
from pathlib import Path
from tkinter import N
from turtle import shape

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import colors

# my imports
from util import plot_util
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil
from vel_stellar import Stellar
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


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

    @staticmethod
    def _calc_phi_delta(phi, phi_0=0.0):
        """Normalize azimuth angles to [-pi/2, +pi/2] relative to the major axis."""
        return ((phi - phi_0) + np.pi/2) % np.pi - np.pi/2

    # Filter the velocity map with SNR above the threshold and within ±phi_limit of the major axis.
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, azimuth_map: np.ndarray, snr_threshold: float = 10.0, phi_limit_deg: float = 60.0) -> np.ndarray:
        phi_delta = self._calc_phi_delta(azimuth_map)
        phi_limit_rad = np.radians(phi_limit_deg)
        valid_mask = ((snr_map >= snr_threshold) & (np.abs(phi_delta) <= phi_limit_rad) & np.isfinite(vel_map))

        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]
        return vel_map_filtered
    
    # PA: The position angle of the major axis of the galaxy, measured from north to east.
    # b/a: The axis ratio (b/a) of the galaxy
    def _calc_pa_inc(self) -> float:
        phi, ba_1 = self.maps_util.get_pa_inc()
        ba = 1 - ba_1
        print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")
        # convert PA to radians
        pa = np.radians(phi) + np.pi/2  # North is in 90 deg position
        inc = self._calc_inc(ba, ba_0=BA_0)
        print(f"Calculated Inclination i: {np.degrees(inc):.2f} deg from b/a={ba:.3f}")

        return pa, inc

    def _get_vel_obs(self, PLATE_IFU: str, type: str='gas') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"Plate-IFU {PLATE_IFU} MAPS")

        print("")
        print("#######################################################")
        print("# Galaxy Parameters")
        print("#######################################################")

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


        print("")
        print("#######################################################")
        print("# Galaxy Velocity")
        print("#######################################################")

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


        print("")
        print("#######################################################")
        print("# Filter Velocity Processing")
        print("#######################################################")
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

    ################################################################################
    # rotation curve fitting procedures
    ################################################################################

    # Formula: V(r) = Vc * tanh(r / Rt) + V_out * r
    def __vel_rot_profile(self, r, Vc, Rt, V_out) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + V_out * r

    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).
    # Formula: V_obs = V_rot * (sin(i) * cos(phi_delta))
    def _calc_vel_obs_from_rot(self, vel_rot: np.ndarray, inc: float, phi: np.ndarray) -> np.ndarray:
        phi_delta = self._calc_phi_delta(phi, phi_0=0.0)
        correction = np.sin(inc) * np.cos(phi_delta)
        return vel_rot * correction

    #  V_obs = (Vc * tanh(r / Rt) + V_out * r) * (sin(i) * cos(phi_delta))
    def _vel_obs_fit_profile(self, r, Vc, Rt, V_out, inc, phi) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        vel_rot = self.__vel_rot_profile(r, Vc, Rt, V_out)
        vel_obs = self._calc_vel_obs_from_rot(vel_rot, inc, phi)
        return vel_obs

    # used the minimum χ 2 method for fitting
    def _rot_curve_fit(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, phi_map: np.ndarray):
        """Fit the rotation curve using the experience curve function."""
        # Flatten the maps and remove NaN values
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map)
        radius_valid = radius_map[valid_mask]
        phi_valid = phi_map[valid_mask]
        # vel_valid = np.abs(vel_map[valid_mask])
        vel_valid = vel_obs_map[valid_mask]
        radius_valid = np.where(vel_valid < 0, -np.abs(radius_valid), np.abs(radius_valid))

        # Initial guess for parameters: Vc, Rt, V_out
        sign_guess = np.sign(np.nanmedian(vel_valid)) or 1.0
        Vc0 = sign_guess * np.nanmax(np.abs(vel_valid))
        Rt0 = np.nanmedian(radius_valid) / 2.0
        Vout0 = np.nanmedian(vel_valid)
        _, inc0 = self._calc_pa_inc()
        initial_guess = [Vc0, Rt0, Vout0, inc0]

        fit_func_partial = lambda r, Vc, Rt, V_out, inc: self._vel_obs_fit_profile(r, Vc, Rt, V_out, inc, phi=phi_valid)
        popt, _ = curve_fit(fit_func_partial, radius_valid, vel_valid, p0=initial_guess,
                            bounds=([-np.inf, 1e-6, -np.inf, 0], [np.inf, np.inf, np.inf, np.pi]))
        Vc_fit, Rt_fit, V_out_fit, inc_fit = popt

        print("Fitted parameters:")
        print(f"  Vc: {Vc_fit:.3f} km/s")
        print(f"  Rt: {Rt_fit:.3f} kpc/h")
        print(f"  V_out: {V_out_fit:.3f} km/s")
        print(f"  inc: {np.degrees(inc_fit):.3f} deg, inc0: {np.degrees(inc0):.3f} deg")

        vel_rot_fitted = self.__vel_rot_profile(radius_map, Vc_fit, Rt_fit, V_out_fit)
        return vel_rot_fitted, (Vc_fit, Rt_fit, V_out_fit)


    ################################################################################
    # Gas Pressure Support Correction: Vc^2 = V_rot^2 + V_drift^2
    ################################################################################

    # formula: sigma_gas^2 = sigma_gas_obs^2 - sigma_gas_inst^2
    def _get_gas_sigma_sq(self) -> tuple[np.ndarray]:
        sigma_obs, sigma_inst = self.maps_util.get_eml_sigma_map()
        sigma_sq = np.maximum(np.square(sigma_obs) - np.square(sigma_inst), 0.0)
        return sigma_sq
    

    ################################################################################
    # Stellar Pressure Support Correction: 
    ################################################################################

    # formula: sigma_stellar^2 = sigma_stellar_obs^2 - sigma_stellar_inst^2
    def _get_stellar_sigma_sq(self) -> np.ndarray:
        sigma_obs, sigma_inst = self.maps_util.get_stellar_sigma_map()
        print(f"Stellar observed sigma shape: {sigma_obs.shape}, range: [{np.nanmin(sigma_obs):.3f}, {np.nanmax(sigma_obs):.3f}] km/s")
        print(f"Stellar instrumental sigma shape: {sigma_inst.shape}, range: [{np.nanmin(sigma_inst):.3f}, {np.nanmax(sigma_inst):.3f}] km/s")
        # If observed dispersion is smaller than instrumental, do not subtract instrumental term.
        sigma_sq = np.where(sigma_obs >= sigma_inst,
                    np.square(sigma_obs) - np.square(sigma_inst),
                    np.square(sigma_obs))

        return sigma_sq

    def _calc_vel_drift_sq(self, radius_map: np.ndarray, stellar_density_map: np.ndarray, sigma_sq_map: np.ndarray, incl,
                           beta: float = 0.5, gamma: float = 0.6, smooth_sigma: float = 1.0):
        """
        Compute asymmetric drift v_drift^2 map using Jeans approximation.

        Parameters
        ----------
        radius_map : 2D array (kpc)
            Physical radius per pixel.
        stellar_density_map : 2D array (Msun/kpc^2)
            Stellar surface density map Σ(R).
        sigma_sq_map : 2D array (km^2/s^2)
            LOS velocity dispersion map σ_LOS^2 (already instrument-corrected).
        incl : float
            Galaxy inclination in degrees.
        beta : float
            σ_phi^2 / σ_R^2  (default 0.5)
        gamma : float
            σ_z / σ_R (default 0.6)
        smooth_sigma : float
            Gaussian smoothing width for LOS sigma vs radius.

        Returns
        -------
        v_drift_sq_map : 2D array (km^2/s^2)
            Full 2D map of asymmetric drift term.
        """

        # ======================
        # 0. Prepare arrays
        # ======================
        mask = (
            np.isfinite(radius_map) &
            np.isfinite(stellar_density_map) &
            np.isfinite(sigma_sq_map)
        )

        r = radius_map[mask]
        Sigma = stellar_density_map[mask]
        sigma_los_sq = sigma_sq_map[mask]
        sigma_los = np.sqrt(np.maximum(sigma_los_sq, 0.0))

        # avoid r=0
        eps = 1e-6
        r = np.where(r > 0, r, eps)

        # ======================
        # 1. Radial averaging
        # ======================
        r_unique = np.unique(r)
        sigma_los_R = np.zeros_like(r_unique)
        Sigma_R = np.zeros_like(r_unique)

        for i, rv in enumerate(r_unique):
            sel = (r == rv)
            sigma_los_R[i] = np.nanmedian(sigma_los[sel])
            Sigma_R[i] = np.nanmedian(Sigma[sel])

        # smooth LOS sigma (avoid noisy derivative)
        sigma_los_R_s = gaussian_filter1d(sigma_los_R, smooth_sigma)

        # ======================
        # 2. LOS → σ_R
        # ======================
        i_rad = np.deg2rad(incl)
        sin2i = np.sin(i_rad)**2
        cos2i = np.cos(i_rad)**2

        denom = 0.5 * (1 + beta) * sin2i + (gamma**2) * cos2i
        denom = max(denom, 1e-6)

        sigma_R = sigma_los_R_s / np.sqrt(denom)
        sigma_R_sq = sigma_R**2

        # ======================
        # 3. d ln(Σ σ_R^2) / d ln r
        # ======================
        f = Sigma_R * sigma_R_sq
        f = np.where(f > 0, f, eps)

        ln_f = np.log(f)
        ln_r = np.log(r_unique)

        dlnf_dlnr = np.gradient(ln_f, ln_r)

        # ======================
        # 4. Jeans formula
        # ======================
        v_drift_sq_R = sigma_R_sq * (dlnf_dlnr + 1 - beta)
        v_drift_sq_R = np.where(v_drift_sq_R > 0, v_drift_sq_R, 0.0)

        # ======================
        # 5. Interpolate back to 2D map
        # ======================
        v_drift_sq_map = np.zeros_like(radius_map)
        v_interp = np.interp(r, r_unique, v_drift_sq_R)
        v_drift_sq_map[mask] = v_interp

        return v_drift_sq_map

    ################################################################################
    # public methods
    ################################################################################
    def get_gas_vel_obs(self, PLATE_IFU: str):
        radius_map, vel_obs_map, phi_map = self._get_vel_obs(PLATE_IFU, type='gas')
        return radius_map, vel_obs_map, phi_map
    
    def get_stellar_vel_obs(self, PLATE_IFU: str):
        radius_map, vel_obs_map, phi_map = self._get_vel_obs(PLATE_IFU, type='stellar')
        return radius_map, vel_obs_map, phi_map

    def fit_rot_vel(self, radius_map, vel_obs_map, phi_map):
        vel_rot_fitted, _ =  self._rot_curve_fit(radius_map, vel_obs_map, phi_map)
        return radius_map, vel_rot_fitted

    def get_vel_drift_sq(self, radius_map: np.ndarray, stellar_density_map: np.ndarray) -> np.ndarray:
        sigma_sq_map = self._get_stellar_sigma_sq()
        _, inc = self._calc_pa_inc()
        vel_drift_sq_map = self._calc_vel_drift_sq(radius_map, stellar_density_map, sigma_sq_map, inc)
        return radius_map, vel_drift_sq_map


######################################################
# main function for test
######################################################
def main():
    PLATE_IFU = "8723-12705"

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
    r_obs_map, V_obs_map, phi_map = vel_rot.get_gas_vel_obs(PLATE_IFU)
    r_rot_fitted, V_rot_fitted = vel_rot.fit_rot_vel(r_obs_map, V_obs_map, phi_map)

    print("#######################################################")
    print("# calculate results")
    print("#######################################################")
    print(f"Obs Velocity shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}]")
    print(f"Fitted Rot Velocity shape: {V_rot_fitted.shape}, range: [{np.nanmin(V_rot_fitted):.3f}, {np.nanmax(V_rot_fitted):.3f}]")

    plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs")
    plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs Fitted", r_rot2_map=r_rot_fitted, v_rot2_map=V_rot_fitted, title2="Rot Fitted")
    return


# main entry
if __name__ == "__main__":
    main()

