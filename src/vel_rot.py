from operator import le
from pathlib import Path

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

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


PLATE_IFU = "8723-12705"
# PLATE_IFU = "8723-12703"

# constants definitions
SNR_THRESHOLD = 10.0
PHI_LIMIT_DEG = 60.0
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation

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
    def _calc_phi_delta(phi):
        """Normalize azimuth angles to [-pi/2, +pi/2] relative to the major axis."""
        return (phi + np.pi/2) % np.pi - np.pi/2

    # Filter the velocity map with SNR above the threshold and within ±phi_limit of the major axis.
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, azimuth_map: np.ndarray, snr_threshold: float = 10.0, phi_limit_deg: float = 60.0) -> np.ndarray:
        phi_delta = self._calc_phi_delta(azimuth_map)
        phi_limit_rad = np.radians(phi_limit_deg)
        valid_mask = ((snr_map >= snr_threshold) & (np.abs(phi_delta) <= phi_limit_rad) & np.isfinite(vel_map))

        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]
        return vel_map_filtered

    # Calculate the true rotational velocity V_rot from observed line-of-sight velocity V_obs
    # using the inclination i and azimuthal angle phi.
    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).
    def _calc_vel_rot(self, vel_map: np.ndarray, azimuth_map: np.ndarray, incl_rad: float) -> np.ndarray:
        """
        Calculate the true rotational velocity V_rot from the inclination and azimuth angles.
        Formula: V_rot = V_obs / (sin(i) * cos(phi_delta))
        """
        phi_delta = self._calc_phi_delta(azimuth_map)
        correction = np.sin(incl_rad) * np.cos(phi_delta)

        # Avoid division by zero, mask invalid regions
        valid = np.abs(correction) > 1e-3
        v_rot = np.full_like(vel_map, np.nan, dtype=float)
        v_rot[valid] = vel_map[valid] / correction[valid]

        return v_rot

    ################################################################################
    # main function
    ################################################################################
    def get_vel_rot(self, PLATE_IFU: str):
        print(f"Plate-IFU {PLATE_IFU} MAPS")

        ########################################################
        # get parameters from FITS files
        ########################################################
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

        # PA: The position angle of the major axis of the galaxy, measured from north to east.
        # b/a: The axis ratio (b/a) of the galaxy
        phi, ba_1 = self.maps_util.get_pa_inc()
        ba = 1 - ba_1
        print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")
        # convert PA to radians
        galaxy_pa_rad = np.radians(phi) + np.pi/2  # North is in 90 deg position
        inc_rad = self._calc_inc(ba, ba_0=BA_0)
        print(f"Calculated Inclination i: {np.degrees(inc_rad):.2f} deg from b/a={ba:.3f}")

        ra_map, dec_map = self.maps_util.get_skycoo_map()
        print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

        ########################################################
        # Galaxy spin velocity map
        ########################################################

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
        print("# Correct Velocity Processing")
        print("#######################################################")
        # Velocity correction
        v_obs_map = v_obs_gas_map
        v_unit = _gv_unit
        v_uindx = eml_uindx
        azimuth_rad_map = np.radians(azimuth_map)

        filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG)
        print(f"Filtered Velocity map shape: {filtered_vel_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(filtered_vel_map):.3f}, {np.nanmax(filtered_vel_map):.3f}]")
        print(f"Velocity data before filtering: {np.sum(np.isfinite(v_obs_map))}, after filtering: {np.sum(np.isfinite(filtered_vel_map))}")

        v_rot_map = self._calc_vel_rot(filtered_vel_map, azimuth_rad_map, inc_rad)
        print(f"Rotated Velocity map shape: {v_rot_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(v_rot_map):.3f}, {np.nanmax(v_rot_map):.3f}]")


        # plot rotated velocity map
        if self.plot_util:
            self.plot_util.plot_vel_map(v_obs_gas_map, eml_uindx, ra_map, dec_map, title="H-alpha Emission Line")
            self.plot_util.plot_vel_map(v_obs_stellar_map, stellar_uindx, ra_map, dec_map, pa_rad=galaxy_pa_rad, title="Stellar")
            self.plot_util.plot_vel_map(filtered_vel_map, v_uindx, ra_map, dec_map, title="Filtered Observed")
            self.plot_util.plot_vel_map(v_rot_map, v_uindx, ra_map, dec_map, title="Total Rotational")

        return v_rot_map, radius_h_kpc_map


def main():
    PLATE_IFU = "8723-12705"

    print("#######################################################")
    print("# 1. load necessary files")
    print("#######################################################")
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


    print("#######################################################")
    print("# 3. calculate total rotation velocity V(r)")
    print("#######################################################")
    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=plot_util)
    v_total_map, r_total_map = vel_rot.get_vel_rot(PLATE_IFU)
    print(f"Velocity shape: {v_total_map.shape}, range: [{np.nanmin(v_total_map):.3f}, {np.nanmax(v_total_map):.3f}]")
    print(f"Radius shape: {r_total_map.shape}, range: [{np.nanmin(r_total_map):.3f}, {np.nanmax(r_total_map):.3f}]")

    return


# main entry
if __name__ == "__main__":
    main()

