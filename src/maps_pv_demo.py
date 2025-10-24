
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
from matplotlib import colors

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


# PLATE_IFU = "8723-12705"
PLATE_IFU = "8723-12703"

# constants definitions
SNR_THRESHOLD = 10.0
PHI_LIMIT_DEG = 50.0


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
def calc_vel_rot(vel_map, pa_rad, inc_rad, snr_map, snr_threshold=10.0, phi_limit_deg=60.0, center_x=None, center_y=None, apply_projection=True):
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
        apply_projection (bool): If True, perform geometric deprojection of velocities
                                 (V_rot = V_obs / (sin(i)*cos(phi))). If False, do not
                                 deproject; only apply SNR and phi filtering and return
                                 the filtered observed velocities. Default True.

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - vel_map_corrected (np.ndarray): The corrected true rotation velocity
          (V_rot) map (km/s) if apply_projection=True; otherwise the filtered
          observed velocity map. Points not meeting filter criteria are NaN.
        - radius_map (np.ndarray): The radial distance map (in spaxels). This is
          deprojected radius if apply_projection=True, otherwise projected radius.
    """
    vel_map = np.asarray(vel_map, dtype=float)
    snr_map = np.asarray(snr_map, dtype=float)
    
    sin_inc = np.sin(inc_rad)
    cos_inc = np.cos(inc_rad)

    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    
    x_c = center_x if center_x is not None else (nx - 1) / 2.0
    y_c = center_y if center_y is not None else (ny - 1) / 2.0
    
    x_rel = x - x_c
    y_rel = y - y_c

    # Rotate to align with major axis
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * sin_pa - y_rel * cos_pa  # along major axis
    y_rot = x_rel * cos_pa + y_rel * sin_pa  # along minor axis

    # When not deprojecting, use projected radius; otherwise use deprojected radius
    if apply_projection:
        if np.isclose(sin_inc, 0.0) or np.isclose(cos_inc, 0.0):
            # Face-on or nearly so; cannot deproject
            nan_map = np.full_like(vel_map, np.nan, dtype=float)
            return nan_map, nan_map
        y_deproj = y_rot / cos_inc
        radius_map = np.hypot(x_rot, y_deproj)
        cos_phi = np.divide(x_rot, radius_map, out=np.zeros_like(x_rot), where=radius_map > 0)
        projection = sin_inc * cos_phi
    else:
        # 仅进行SNR与phi过滤，不对速度做几何投影/去投影
        radius_map = np.hypot(x_rot, y_rot)
        cos_phi = np.divide(x_rot, radius_map, out=np.zeros_like(x_rot), where=radius_map > 0)
        projection = None  # not used

    # Thresholds
    cos_phi_threshold = np.cos(np.radians(phi_limit_deg))
    

    valid = (
        np.isfinite(vel_map) &
        (radius_map > 0) &
        (snr_map >= snr_threshold) &
        (np.abs(cos_phi) >= cos_phi_threshold)
    )

    vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
    if apply_projection:
        vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    else:
        vel_map_corrected[valid] = vel_map[valid]

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


# plot r-v curve
def plot_rv_curve(r_rot_map, v_rot_map):
    # Keep signs consistent: if v_rot < 0, set r_rot negative; else positive
    r_rot_map = np.asarray(r_rot_map, dtype=float)
    v_rot_map = np.asarray(v_rot_map, dtype=float)
    r_signed = np.where(v_rot_map < 0, -np.abs(r_rot_map), np.abs(r_rot_map))

    # Mask invalid values
    valid = np.isfinite(r_signed) & np.isfinite(v_rot_map)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_signed[valid], v_rot_map[valid], s=2, color='red', alpha=0.2, label='Data Points')

    ax.set_title("Galaxy Rotation Curve (R-V)")
    ax.set_xlabel("Radius R (spaxel)")
    ax.set_ylabel("Rotation Velocity V_rot (km/s)")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    plt.show()


# Plots the binned velocity map using unique bin indices.
def plot_bin_vel_map(vel_map, uindx, ra_map, dec_map, pa_rad=None, title: str=""):
    """
    Plots the binned velocity map using unique bin indices.

    Args:
        vel_map (np.ndarray): The 2D velocity map.
        uindx (np.ndarray): 1D array of unique indices to flatten the maps.
        ra_map (np.ndarray): The 2D Right Ascension map.
        dec_map (np.ndarray): The 2D Declination map.
        pa_rad (float, optional): Position Angle in radians, measured from North to East.
                                  If provided, a line indicating the major axis is drawn.
        title (str, optional): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Flatten maps and select unique binned data
    ra_flat, dec_flat, vel_flat = ra_map.ravel(), dec_map.ravel(), vel_map.ravel()
    ra_u, dec_u, vel_u = ra_flat[uindx], dec_flat[uindx], vel_flat[uindx]
    
    # Filter out non-finite velocity values for color scaling
    valid_vel_mask = np.isfinite(vel_u)
    vel_u_clean = vel_u[valid_vel_mask]
    
    # Set color normalization based on velocity percentiles
    if vel_u_clean.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        p_low, p_high = np.nanpercentile(vel_u_clean, [2, 98])
        vmax = max(abs(p_low), abs(p_high))
        vmin = -vmax
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
    # Create the scatter plot for the binned velocity data
    sc = ax.scatter(ra_u, dec_u, c=vel_u, cmap='RdBu_r', norm=norm, s=30, edgecolors='face', alpha=0.9)
    
    # Add a colorbar
    cbar = fig.colorbar(sc, ax=ax, label="Velocity (km/s)")
    cbar.set_ticks([vmin, 0, vmax])
    
    # Draw the major axis line if pa_rad is provided
    if pa_rad is not None and ra_u[valid_vel_mask].size > 1:
        pa_rad = pa_rad % (2 * np.pi)  # Normalize PA to [0, 2π]
        # Calculate the center of the galaxy from the valid data points
        ra_center = np.mean(ra_u[valid_vel_mask])
        dec_center = np.mean(dec_u[valid_vel_mask])
        
        # Determine the line length based on the data extent
        ra_range = np.ptp(ra_u[valid_vel_mask])
        dec_range = np.ptp(dec_u[valid_vel_mask])
        line_length = 0.6 * np.hypot(ra_range, dec_range)
        
        # Calculate line endpoints. PA is from North (+Dec) to East (-RA, as axis is inverted).
        # This corresponds to a clockwise angle from the positive y-axis.
        dx = -line_length * np.sin(pa_rad)
        dy = line_length * np.cos(pa_rad)
        
        # Plot the line representing the major axis
        ax.plot([ra_center - dx, ra_center + dx], 
                [dec_center - dy, dec_center + dy], 
                color='black', linestyle='--', linewidth=1.5, label='Major Axis (PA)')
        ax.legend()

    # Set plot labels and title
    ax.set_title(f"{title} Binned Velocity Map")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    
    # Invert RA axis for standard astronomical orientation (East to the left)
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    plt.show()

def plot_rotation_curve(r_map, v_map, title: str=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_map, v_map, s=2, color='blue', alpha=0.3, label='Data Points')

    ax.set_title(f"{title} Galaxy Rotation Curve")
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
    print(f"Plate-IFU {PLATE_IFU} MAPS")

    # get fits files
    maps_file = fits_util.get_maps_file(PLATE_IFU)
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
    print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {1-ba_1:.3f}")

    ba = 1 - ba_1
    pa_rad = np.radians(phi)  # convert to radians, from North to East, then to major axis
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    ra_map, dec_map = map_util.get_skycoo_map()
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    ra_map, dec_map = map_util.get_skycoo_map()
    print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

    ########################################################
    # Galaxy spin velocity map
    ########################################################

    ## Get the gas velocity map (H-alpha)
    gas_vel_map, _gv_unit, _gv_ivar = map_util.get_eml_vel_map()
    _gv_disp = calc_vel_dispersion(_gv_ivar)
    print(f"Velocity map shape: {gas_vel_map.shape}, Unit: {_gv_unit}")
    print(f"Velocity: [{np.nanmin(gas_vel_map):.3f}, {np.nanmax(gas_vel_map):.3f}] {_gv_unit}")
    _, eml_uindx = map_util.get_emli_uindx()
    # print(f"Unique bin indices count: {len(eml_uindx)}")
    
    ## Get the stellar velocity map
    stellar_vel_map, _sv_unit, _ = map_util.get_stellar_vel_map()
    print(f"Stellar velocity map shape: {stellar_vel_map.shape}, Unit: {_sv_unit}")
    print(f"Stellar Velocity: [{np.nanmin(stellar_vel_map):.3f}, {np.nanmax(stellar_vel_map):.3f}] {_sv_unit}")
    _, stellar_uindx = map_util.get_stellar_uindx()
    # print(f"Unique bin indices count: {len(stellar_uindx)}")

    # Estimate Systemic Velocity V_sys from stellar velocity map
    vel_sys = calc_v_sys(stellar_vel_map, size=3)
    print(f"Estimated Systemic Velocity V_sys: {vel_sys:.3f} {_sv_unit}")

    # Velocity Field Centering
    v_obs_map = gas_vel_map
    v_internal_map = v_obs_map - vel_sys
    v_unit = _gv_unit
    v_uindx = eml_uindx
    print(f"Internal Velocity map shape: {v_internal_map.shape}, Unit: {v_unit}")
    print(f"Internal Velocity: [{np.nanmin(v_internal_map):.3f}, {np.nanmax(v_internal_map):.3f}] {v_unit}")



    v_rot_map, r_rot_map = calc_vel_rot(v_internal_map, pa_rad, inc_rad, snr_map, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG, apply_projection=False)
    print(f"v_rot_map: [{np.nanmin(v_rot_map):.3f}, {np.nanmax(v_rot_map):.3f}] km/s", f"size: {len(v_rot_map)}")

    ########################################################
    ## plot velocity map
    ########################################################

    # 1. plot galaxy image
    plot_galaxy_image(PLATE_IFU)

    ## 2. plot binned velocity maps (No need to subtract system velocity)
    # plot gas velocity map
    plot_bin_vel_map(gas_vel_map, eml_uindx, ra_map, dec_map, title="H-alpha Emission Line")
    # plot stellar velocity map
    plot_bin_vel_map(stellar_vel_map, stellar_uindx, ra_map, dec_map, title="Stellar")

    # 3. plot internal velocity map
    plot_bin_vel_map(v_internal_map, v_uindx, ra_map, dec_map, pa_rad=pa_rad, title="Internal")
    # 4. plot rotated velocity map
    plot_bin_vel_map(v_rot_map, v_uindx, ra_map, dec_map, title="Rotated")

    # 5. plot r-v curve
    plot_rv_curve(r_rot_map, v_rot_map)


# main entry
if __name__ == "__main__":
    main()

