
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
from scipy.optimize import curve_fit
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
        phi = self.maps_util.get_pa()
        ba = self.maps_util.get_ba()
        print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")

        inc = self._calc_inc(ba, ba_0=BA_0)
        print(f"Calculated Inclination i: {np.degrees(inc):.2f} deg")
        # Convert PA from degrees to radians and rotate so North is at +90°
        pa = np.mod(np.radians(phi), 2 * np.pi)
        return pa, inc

    def _get_vel_obs(self, type: str='gas') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    def _get_radius(self) -> np.ndarray:
        _, radius_h_kpc_map, _ = self.maps_util.get_radius_map()
        return radius_h_kpc_map


    ################################################################################
    # rotation curve fitting procedures
    ################################################################################

    # Formula: V(r) = Vc * tanh(r / Rt) + s_out * r
    def __vel_rot_profile(self, r, Vc, Rt, s_out) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + s_out * r

    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).
    # Formula: V_obs = V_rot * (sin(i) * cos(phi_delta))
    def _calc_vel_obs_from_rot(self, vel_rot: np.ndarray, inc: float, phi: np.ndarray, phi_0: float) -> np.ndarray:
        phi_delta = self._calc_phi_delta(phi, phi_0=0.0)
        correction = np.sin(inc) * np.cos(phi_delta)
        return vel_rot * correction

    #  V_obs = (Vc * tanh(r / Rt) + s_out * r) * (sin(i) * cos(phi - phi_0))
    def _vel_obs_fit_profile(self, r, Vc, Rt, s_out, inc, phi, phi_0) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        vel_rot = self.__vel_rot_profile(r, Vc, Rt, s_out)
        vel_obs = self._calc_vel_obs_from_rot(vel_rot, inc, phi, phi_0)
        return vel_obs

    # used the minimum χ 2 method for fitting
    def _rot_curve_fit(self, radius_map: np.ndarray, vel_obs_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
        """Fit the rotation curve using the experience curve function."""
        # Flatten the maps and remove NaN values
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius_map) & (radius_map > 0.1) & np.isfinite(phi_map)
        radius_valid = radius_map[valid_mask]
        phi_valid = phi_map[valid_mask]
        vel_valid = vel_obs_map[valid_mask]
        vel_valid = np.abs(vel_valid)
        # radius_valid = np.where(vel_valid < 0, -np.abs(radius_valid), np.abs(radius_valid))

        phi_0, inc_0 = self._calc_pa_inc()
        def fit_func_partial(r, Vc, Rt, s_out, inc):
            return self._vel_obs_fit_profile(r, Vc, Rt, s_out, inc, phi=phi_valid, phi_0=phi_0)

        Xdata = radius_valid
        Ydata = vel_valid

        p0 = [100.0, 2.0, 0.1, np.deg2rad(45.0)]  # Initial guesses for Vc, Rt, s_out, inc
        lb = [1e-6, 1e-6, -50.0, 1e-6]
        ub = [500.0, 20, 50.0, np.pi/2]

        popt, _ = curve_fit(fit_func_partial, Xdata, Ydata, p0=p0, bounds=(lb, ub), method='trf')
        Vc_fit, Rt_fit, s_out_fit, inc_fit = popt

        print(f"IFU [{self.PLATE_IFU}] Fitted parameters:")
        print(f"  Vc: {Vc_fit:.3f} km/s")
        print(f"  Rt: {Rt_fit:.3f} kpc/h")
        print(f"  s_out: {s_out_fit:.3f} km/s")
        print(f"  i: {np.degrees(inc_fit):.3f} deg")
        print(f"  phi_0: {np.degrees(phi_0):.3f} deg")

        if radius_fit is None:
            radius_fit = radius_map

        vel_rot_fitted = self.__vel_rot_profile(radius_fit, Vc_fit, Rt_fit, s_out_fit)
        return radius_fit, vel_rot_fitted, (Vc_fit, Rt_fit, s_out_fit)


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
        radius_fitted, vel_rot_fitted, _ =  self._rot_curve_fit(radius_map, vel_obs_map, phi_map, radius_fit=radius_fit)
        return radius_fitted, vel_rot_fitted

    def get_inc_rad(self):
        _, inc_rad = self._calc_pa_inc()
        return inc_rad
    
    def get_radius_fit(self, count: int=100) -> np.ndarray:
        radius_map = self._get_radius()
        radius_max = np.nanmax(radius_map)

        radius_fit = np.linspace(0.0, radius_max, num=count)
        return radius_fit


######################################################
# main function for test
######################################################
def main():
    PLATE_IFU = "7957-3701"

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
    r_rot_fitted, V_rot_fitted = vel_rot.fit_rot_vel(r_obs_map, V_obs_map, phi_map, radius_fit=r_fit)

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

