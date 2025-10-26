from operator import le
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
PHI_LIMIT_DEG = 60.0
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation


################################################################################
# calculate functions
################################################################################

# Calculate the galaxy inclination i (in radians)
# Formula for inclination i
# The inclination is the angle between the galaxy disk normal and the observer's line of sight.
# ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
def calc_inc(ba, ba_0=0.2):
    ba_sq = ba**2
    BA_0_sq = ba_0**2
    
    # Compute the numerator part of cos^2(i)
    numerator = ba_sq - BA_0_sq
    denominator = 1.0 - BA_0_sq

    cos_i_sq = numerator / denominator
    cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)
    
    inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
    return inc_rad


def rotate_to_major_axis(delta_x, delta_y, pa_rad):
    x_prime = delta_x * np.cos(pa_rad) + delta_y * np.sin(pa_rad)
    y_prime = -delta_x * np.sin(pa_rad) + delta_y * np.cos(pa_rad)
    return x_prime, y_prime

def deproject_to_galactic_plane(x_prime, y_prime, inc_rad):
    cos_i = np.cos(inc_rad)
    if np.isclose(cos_i, 0.0):
        cos_i = 1e-6  # Preventing de-zeroing
    x_gal = x_prime
    y_gal = y_prime / cos_i
    return x_gal, y_gal

def calc_rot_angle_phi(x_gal: np.ndarray, y_gal: np.ndarray) -> np.ndarray:
    """
    Calculates the azimuth φ (in radians) in the intrinsic plane of the galaxy relative to the long axis.
    φ = 0 in the range [-π, π] along the positive direction of the long axis.
    """
    x = np.asarray(x_gal, dtype=float)
    y = np.asarray(y_gal, dtype=float)
    phi_array = np.arctan2(y, x)
    invalid_mask = (~np.isfinite(x)) | (~np.isfinite(y)) | ((x == 0.0) & (y == 0.0))
    if np.any(invalid_mask):
        phi_array = np.array(phi_array, dtype=float, copy=True)
        phi_array[invalid_mask] = np.nan

    return phi_array


def vel_map_filter(
    vel_map: np.ndarray,
    snr_map: np.ndarray,
    phi_array: np.ndarray,
    snr_threshold: float = 10.0,
    phi_limit_deg: float = 60.0) -> np.ndarray:
    """
    Filter the velocity map:
    Keep only pixels with SNR above the threshold and within ±phi_limit of the major axis.

    Returns:
        vel_map_filtered: The masked/filtered velocity map
    """
    vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
    # Effective mask
    phi_limit_rad = np.radians(phi_limit_deg)
    valid_mask = (
        (snr_map >= snr_threshold) &
        (np.abs(phi_array) <= phi_limit_rad) &
        np.isfinite(vel_map)
    )

    vel_map_filtered[valid_mask] = vel_map[valid_mask]
    return vel_map_filtered

# Calculate the true rotational velocity V_rot from observed line-of-sight velocity V_obs
# using the inclination i and azimuthal angle phi.
def calc_v_rot(v_obs, phi_array, incl_rad):
    """
    Calculate the true rotational velocity V_rot from the inclination and azimuth angles.
    Formula: V_rot = V_obs / (sin(i) * cos(phi))
    """
    sin_i = np.sin(incl_rad)
    correction = sin_i * np.cos(phi_array)
    
    # Avoid division by zero, mask invalid regions
    valid = np.abs(correction) > 1e-3
    v_rot = np.full_like(v_obs, np.nan, dtype=float)
    v_rot[valid] = v_obs[valid] / correction[valid]
    
    return v_rot

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
        ax.imshow(img, origin="upper", cmap="gray")
    else:
        ax.imshow(img, origin="upper")

    ax.set_title(f"Galaxy Image ({plateifu})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    plt.show()

# Plots the binned velocity map using unique bin indices.
def plot_vel_map(vel_map, uindx, ra_map, dec_map, pa_rad=None, title: str=""):
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
                color='gray', linestyle='--', linewidth=1.5, label='Major Axis (PA)')
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
    offset_x, offset_y = map_util.get_sky_offsets()
    print(f"Sky offsets shape: {offset_x.shape}, X offset: [{np.nanmin(offset_x):.3f}, {np.nanmax(offset_x):.3f}] arcsec")
    print(f"Sky offsets shape: {offset_y.shape}, Y offset: [{np.nanmin(offset_y):.3f}, {np.nanmax(offset_y):.3f}] arcsec")

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
    ba = 1 - ba_1
    print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")

    pa_rad = np.radians(phi)  # convert to radians, from North to East, then to major axis
    inc_rad = calc_inc(ba, ba_0=BA_0)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    ra_map, dec_map = map_util.get_skycoo_map()
    print(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

    ########################################################
    # Galaxy spin velocity map
    ########################################################

    ## Get the gas velocity map (H-alpha)
    gas_vel_map, _gv_unit, _gv_ivar = map_util.get_eml_vel_map()
    print(f"Velocity map shape: {gas_vel_map.shape}, Unit: {_gv_unit}")
    print(f"Velocity: [{np.nanmin(gas_vel_map):.3f}, {np.nanmax(gas_vel_map):.3f}] {_gv_unit}")
    _, eml_uindx = map_util.get_emli_uindx()
    
    ## Get the stellar velocity map
    stellar_vel_map, _sv_unit, _ = map_util.get_stellar_vel_map()
    print(f"Stellar velocity map shape: {stellar_vel_map.shape}, Unit: {_sv_unit}")
    print(f"Stellar Velocity: [{np.nanmin(stellar_vel_map):.3f}, {np.nanmax(stellar_vel_map):.3f}] {_sv_unit}")
    _, stellar_uindx = map_util.get_stellar_uindx()


    # Velocity correction
    v_obs_map = gas_vel_map
    v_unit = _gv_unit
    v_uindx = eml_uindx
    print(f"Obs Velocity map shape: {v_obs_map.shape}, Unit: {v_unit}")
    print(f"Obs Velocity: [{np.nanmin(v_obs_map):.3f}, {np.nanmax(v_obs_map):.3f}] {v_unit}")

    x_p, y_p = rotate_to_major_axis(offset_x, offset_y, pa_rad)
    x_gal, y_gal = deproject_to_galactic_plane(x_p, y_p, inc_rad)
    phi_array = calc_rot_angle_phi(x_gal, y_gal)
    vel_map_filtered = vel_map_filter(v_obs_map, snr_map, phi_array=phi_array, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG)

    v_rot_map = calc_v_rot(vel_map_filtered, phi_array, inc_rad)
    print(f"Rotated Velocity map shape: {v_rot_map.shape}, Unit: {v_unit}")
    print(f"Rotated Velocity: [{np.nanmin(v_rot_map):.3f}, {np.nanmax(v_rot_map):.3f}] {v_unit}")

    ########################################################
    ## plot velocity map
    ########################################################

    # 1. plot galaxy image
    plot_galaxy_image(PLATE_IFU)

    ## 2. plot binned velocity maps (No need to subtract system velocity)
    # plot gas velocity map
    plot_vel_map(gas_vel_map, eml_uindx, ra_map, dec_map, title="H-alpha Emission Line")
    # plot stellar velocity map
    plot_vel_map(stellar_vel_map, stellar_uindx, ra_map, dec_map, pa_rad=pa_rad, title="Stellar")

    # 4. plot rotated velocity map
    plot_vel_map(v_rot_map, v_uindx, ra_map, dec_map, title="Rotated")

    # 5. plot r-v curve
    plot_rv_curve(r_map, v_rot_map)

# main entry
if __name__ == "__main__":
    main()

