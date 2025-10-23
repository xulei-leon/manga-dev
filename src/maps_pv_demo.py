
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


plateifu = "8723-12705"
# plateifu = "8723-12703"



  # Intrinsic Axial Ratio for edge-on galaxies, it is assumed value.

################################################################################
# calculate functions
################################################################################

# Calculate the galaxy inclination i (in radians)
# Formula for inclination i
# The inclination is the angle between the galaxy disk normal and the observer's line of sight.
# ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
def calc_inc(ba, BA_0=0.13):
    ba_sq = ba**2
    BA_0_sq = BA_0**2
    
    # Compute the numerator part of cos^2(i)
    numerator = ba_sq - BA_0_sq
    denominator = 1.0 - BA_0_sq

    cos_i_sq = numerator / denominator
    cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)
    
    inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
    return inc_rad

# 将v_map 的值进行绝对值处理
def vel_map_abs(vel_map):
    vel_map = np.asarray(vel_map, dtype=float)
    vel_map_abs = np.abs(vel_map)
    return vel_map_abs

# SNR filtering for velocity map
def vel_snr_filter(vel_map, snr_map, snr_threshold=10.0):
    """
    Filters the velocity map based on a signal-to-noise ratio (SNR) threshold.

    Args:
        vel_map (np.ndarray): The observed velocity map.
        snr_map (np.ndarray): The corresponding SNR map.
        snr_threshold (float): The minimum SNR required to keep a velocity value.

    Returns:
        np.ndarray: The filtered velocity map, with invalid points set to NaN.
    """
    vel_map = np.asarray(vel_map, dtype=float)
    snr_map = np.asarray(snr_map, dtype=float)

    # Create a mask for valid data points
    valid_mask = snr_map >= snr_threshold

    # Initialize the output array with NaNs
    filtered_vel_map = np.full_like(vel_map, np.nan, dtype=float)

    # Keep only the valid points
    filtered_vel_map[valid_mask] = vel_map[valid_mask]

    return filtered_vel_map


def arctan_model(r, Vc, Rt, Vsys=0):
    """Arctangent rotation curve model."""
    return Vsys + (2 / np.pi) * Vc * np.arctan(r / Rt)

# calculate velocity dispersion
def calc_vel_dispersion(ivar_map):
    v_disp = np.sqrt(1 / np.sum(ivar_map, axis=0))
    return v_disp

def calc_v_sys(v_map, size=3):
    """
    Estimate systemic velocity V_sys as the median of the central size x size spaxel box
    from the provided velocity map. `size` should be an odd positive integer
    (commonly 3 or 5). NaN values are ignored; returns np.nan if no valid pixels.
    """
    v_map = np.asarray(v_map, dtype=float)
    if size % 2 == 0 or size < 1:
        raise ValueError("size must be an odd positive integer (e.g., 3 or 5)")

    ny, nx = v_map.shape
    # center indices (rounded in the same spirit as previous code)
    x_c = int(round((nx - 1) / 2.0))
    y_c = int(round((ny - 1) / 2.0))

    half = size // 2
    x0 = max(0, x_c - half)
    x1 = min(nx, x_c + half + 1)
    y0 = max(0, y_c - half)
    y1 = min(ny, y_c + half + 1)

    core = v_map[y0:y1, x0:x1]
    v_sys = np.nanmedian(core)
    return float(v_sys)


# Geometric correction for the MaNGA velocity map
def calc_vel_rot(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0, center_x=None, center_y=None):
    """
    Performs geometric correction on the observed velocity field, combined with
    signal-to-noise and azimuthal angle filtering.

    Args:
        vel_map (np.ndarray): The observed line-of-sight velocity map (km/s).
        pa_rad (float): The position angle (PA) of the galaxy in radians.
        inc_rad (float): The inclination (i) of the galaxy in radians.
        snr_map (np.ndarray): The corresponding signal-to-noise ratio (S/N) map.
        phi_limit_deg (float): The maximum allowed azimuthal angle from the major
                               axis in degrees. Default is 60.0.
        center_x (float, optional): The X coordinate of the galaxy center.
                                    Defaults to (nx - 1) / 2.0.
        center_y (float, optional): The Y coordinate of the galaxy center.
                                    Defaults to (ny - 1) / 2.0.

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - vel_map_corrected (np.ndarray): The corrected true rotation velocity
          (V_rot) map (km/s). Points not meeting filter criteria are NaN.
        - radius_map (np.ndarray): The deprojected radial distance map (in spaxels).
    """
    vel_map = np.asarray(vel_map, dtype=float)
    snr_map = np.asarray(snr_map, dtype=float)
    
    # 1. Prepare inclination parameters
    sin_inc = np.sin(inc_rad)
    cos_inc = np.cos(inc_rad)

    if np.isclose(sin_inc, 0.0):
        # Face-on view, cannot deproject
        print("Warning: Inclination is close to 0 (face-on). Cannot deproject velocity field.")
        nan_map = np.full_like(vel_map, np.nan, dtype=float)
        return nan_map, nan_map

    # 2. Calculate coordinates
    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    
    # Use provided center or geometric center
    x_c = center_x if center_x is not None else (nx - 1) / 2.0
    y_c = center_y if center_y is not None else (ny - 1) / 2.0
    
    x_rel = x - x_c
    y_rel = y - y_c

    # 3. Rotate coordinates to align with the galaxy's major axis
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    # x_rot is along the kinematic major axis
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    # y_rot is along the kinematic minor axis
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    # 4. Deproject and calculate cos(phi)
    # For edge-on, cos_inc is near 0, and y_deproj becomes very large
    y_deproj = y_rot / cos_inc 
    radius_map = np.hypot(x_rot, y_deproj) # True, deprojected radial distance

    # cos(phi) is the cosine of the angle between the velocity vector and the line of sight
    # Use np.divide to avoid division by zero warnings
    cos_phi = np.divide(x_rot, radius_map, out=np.zeros_like(x_rot), where=radius_map > 0)
    
    # 5. Projection factor (V_obs = V_rot * projection)
    projection = sin_inc * cos_phi

    # 6. Determine filtering thresholds
    cos_phi_threshold = np.cos(np.radians(phi_limit_deg))
    snr_threshold = 10.0

    # 7. Apply combined filter mask
    valid = (
        np.isfinite(vel_map) &            # Ensure input velocity is valid
        (radius_map > 0) &                # Exclude the central point
        (snr_map >= snr_threshold) &      # Apply SNR threshold
        (np.abs(cos_phi) >= cos_phi_threshold) # Apply azimuthal angle threshold
    )
    
    # 8. Calculate the corrected velocity map (V_rot = V_obs / projection)
    vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
    # Perform division only on valid data points
    vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    
    return vel_map_corrected, radius_map

################################################################################
# plot functions
################################################################################

# show velocity map and the image_file side-by-side
# load image_file (supports FITS and common image formats)
def plot_galaxy_image(plateifu):
    image_file = fits_util.get_image_file(plateifu)
    if image_file is None or not image_file.exists():
        print(f"Warning: image file for {plateifu} does not exist.")
        return

    try:
        img = plt.imread(str(image_file))
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if img.ndim == 2:
        ax.imshow(img, origin="lower", cmap="gray")
    else:
        ax.imshow(img, origin="lower")

    ax.set_title(f"Galaxy Image ({plateifu})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    plt.show()



# plot velocity map
def plot_velocity_map(vel_map, unit):
    fig, ax = plt.subplots(figsize=(7, 6))
    im0 = ax.imshow(vel_map, origin="lower", cmap="coolwarm")
    fig.colorbar(im0, ax=ax, label=f"H-alpha Line-of-Sight Velocity ({unit})")
    ax.set_title("Galaxy Velocity Map")
    ax.set_xlabel("X Spaxel")
    ax.set_ylabel("Y Spaxel")
    fig.tight_layout()
    plt.show()


# plot r-v curve
def plot_rv_curve(r_flat, v_flat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_flat, v_flat, s=2, color='red', alpha=0.2, label='Data Points')

    ax.set_title("Galaxy Rotation Curve (R-V)")
    ax.set_xlabel("Radius R (spaxel)")
    ax.set_ylabel("Rotation Velocity V_rot (km/s)")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    plt.show()


################################################################################
# main function
################################################################################
def main():
    print(f"Plate-IFU {plateifu} MAPS")

    # get fits files
    maps_file = fits_util.get_maps_file(plateifu)
    drpall_file = fits_util.get_drpall_file()
    drpall_util = DrpallUtil(drpall_file)
    # print(f"DRPALL Info for {plateifu}: {drpall_util.dump_info()}")
    map_util = MapsUtil(maps_file)
    # print(f"MAPS Info for {plateifu}: {map_util.dump_info()}")


    ########################################################
    # get parameters from FITS files
    ########################################################
    spaxel_x, spaxel_y = map_util.get_spaxel_size()
    print(f"Spaxel Size: {spaxel_x:.3f} arcsec (X), {spaxel_y:.3f} arcsec (Y)")

    # R: radial distance map
    r_map, azimuth_map = map_util.get_r_map()
    phi_rad_map = np.radians(azimuth_map) 
    print(f"r_map: [{np.nanmin(r_map):.3f}, {np.nanmax(r_map):.3f}] spaxel,", f"shape: {r_map.shape}")

    # SNR: signal-to-noise ratio map
    snr_map = map_util.get_snr_map()
    print(f"SNR map shape: {snr_map.shape}")
    print(f"SNR: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

    # PA: The position angle of the major axis of the galaxy, measured from north to east.
    # b/a: The axis ratio (b/a) of the galaxy
    phi, ba_1 = map_util.get_pa_inc()
    print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination (1-b/a) from MAPS header: {ba_1:.3f}")
    if ba_1 is not None:
        ba = 1 - ba_1  # convert to b/a
    if phi is None or ba is None:
        phi, ba = drpall_util.get_phi_ba(plateifu)
        print(f"Position Angle PA from DRPALL: {phi:.2f} deg,", f"Axial Ratio b/a from DRPALL: {ba:.3f}")


    ########################################################
    # Galaxy spin velocity map
    ########################################################

    # Get the gas velocity map (H-alpha)
    _gv_map, _gv_unit, _gv_ivar = map_util.get_gvel_map()
    _gv_disp = calc_vel_dispersion(_gv_ivar)
    print(f"Velocity map shape: {_gv_map.shape}, Unit: {_gv_unit}")
    print(f"Velocity: [{np.nanmin(_gv_map):.3f}, {np.nanmax(_gv_map):.3f}] {_gv_unit}")

    _sv_map, _sv_unit, _sv_ivar = map_util.get_stellar_vel_map()
    print(f"Stellar velocity map shape: {_sv_map.shape}, Unit: {_sv_unit}")
    print(f"Stellar Velocity: [{np.nanmin(_sv_map):.3f}, {np.nanmax(_sv_map):.3f}] {_sv_unit}")

    _v_sys = calc_v_sys(_sv_map, size=3)
    print(f"Estimated Systemic Velocity V_sys: {_v_sys:.3f} {_sv_unit}")

    # Velocity Field Centering
    v_obs_map = _gv_map
    v_internal_map = v_obs_map - _v_sys
    v_unit = _gv_unit
    print(f"Internal Velocity map shape: {v_internal_map.shape}, Unit: {v_unit}")
    print(f"Internal Velocity: [{np.nanmin(v_internal_map):.3f}, {np.nanmax(v_internal_map):.3f}] {v_unit}")

    inc_rad = calc_inc(ba)
    pa_rad = np.radians(phi)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    v_rot_map, _r = calc_vel_rot(v_internal_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0)
    print(f"v_rot_map: [{np.nanmin(v_rot_map):.3f}, {np.nanmax(v_rot_map):.3f}] km/s", f"size: {len(v_rot_map)}")
    v_rot_abs = vel_map_abs(v_rot_map)


    ########################################################
    # plot
    ########################################################

    # 1. plot plateifu map
    plot_galaxy_image(plateifu)

    # 2. plot velocity map
    plot_velocity_map(v_internal_map, v_unit)
    plot_velocity_map(v_rot_map, v_unit)


    # 3. plot r-v curve
    valid_idx = ~np.isnan(v_rot_map) & ~np.isnan(r_map)
    r_flat = r_map[valid_idx]
    v_flat = v_rot_map[valid_idx]
    plot_rv_curve(r_flat, v_flat)


# main entry
if __name__ == "__main__":
    main()

