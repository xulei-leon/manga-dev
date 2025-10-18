
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit

# my imports
import util.maps_util as mu
import util.drpall_util as du
from util.fits_util import FitsUtil

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


plateifu = "8723-12705"
# get fits files
maps_file = fits_util.get_maps_file(plateifu)
drpall_file = fits_util.get_drpall_file()


################################################################################
# calculate functions
################################################################################

# calculate inclination and PA from DRPALL
def cal_inc_angle(drpall_file, plateifu):
    print("## Calculating Rotation Curve Parameters from DRPALL...")
    # calculate systemic velocity for the galaxy from drpall
    z_sys, pa_deg, ba = du.get_z_sys(drpall_file, plateifu)
    if z_sys is not None:
        print(f"Systemic redshift z_sys from DRPALL: {z_sys:.6f}")
        v_sys = z_sys * const.c.to('km/s').value
        print(f"Systemic velocity v_sys: {v_sys:.3f} km/s")

    if pa_deg is not None:
        pa_rad = np.radians(pa_deg)
        print(f"Position Angle PA from DRPALL: {pa_deg:.2f} deg", f"({pa_rad:.3f} rad)")
    else:
        print("Position Angle PA not found in DRPALL.")
        return None, None

    if ba is not None:
        ba = float(ba)
        print(f"Axis Ratio b/a from DRPALL: {ba:.3f}")
    else:
        print("Axis Ratio b/a not found in DRPALL.")
        return None, None
    
    # calculate inclination angle
    q0 = 0.2  # intrinsic axis ratio for edge-on galaxy
    tmp = (ba**2 - q0**2) / (1 - q0**2)
    tmp = np.clip(tmp, 0.0, 1.0)
    inc_deg = np.degrees(np.arccos(np.sqrt(tmp)))
    inc_rad = np.radians(inc_deg)
    print(f"Inclination angle i: {inc_deg:.2f} deg", f"({inc_rad:.3f} rad)")

    return inc_rad, pa_rad


# calculate Rotation Curve
# v_map / (np.sin(inc) * np.cos(theta))
def cal_rotation_vel(v_map, inc_rad, pa_rad, mask_minor_axis=0.2):
    # fix me
    pixel_scale = 0.5 # arcsec/spaxel

    print("## Calculating Rotation Velocity...")
    ny, nx = v_map.shape
    x0, y0 = nx / 2, ny / 2
    y, x = np.indices(v_map.shape)

    # coordinate rotation (check PA convention!)
    x_rot = (x - x0) * np.cos(-pa_rad) + (y - y0) * np.sin(-pa_rad)
    y_rot = -(x - x0) * np.sin(-pa_rad) + (y - y0) * np.cos(-pa_rad)

    x_rot = (x - x0) * np.cos(pa_rad) + (y - y0) * np.sin(pa_rad)
    # y_rot = -(x - x0) * np.sin(pa_rad) + (y - y0) * np.cos(pa_rad)

    mask_major = (np.abs(y_rot) < 1.0)
    r_major = x_rot[mask_major] * pixel_scale  # arcsec
    v_major = v_map[mask_major]

    mask_valid = np.isfinite(v_major)
    r_major = r_major[mask_valid]
    v_major = v_major[mask_valid]

    # 对半径排序，方便拟合
    order = np.argsort(r_major)
    r_major = r_major[order]
    v_major = v_major[order]

    return r_major, v_major

def arctan_model(r, Vc, Rt, Vsys=0):
    """Arctangent rotation curve model."""
    return Vsys + (2 / np.pi) * Vc * np.arctan(r / Rt)

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
    fig.tight_layout()
    plt.show()


# plot rotation curve
def plot_rotation_curve(r_major, v_major):
    p0 = [150, 3.0, 0.0]
    popt, pcov = curve_fit(arctan_model, r_major, v_major, p0=p0)
    Vc, Rt, Vsys = popt
    print(f"Best-fit parameters:\nVc = {Vc:.1f} km/s, Rt = {Rt:.2f} arcsec, Vsys = {Vsys:.2f} km/s")


    r_fit = np.linspace(np.min(r_major), np.max(r_major), 400)
    v_fit = arctan_model(r_fit, *popt)

    plt.figure(figsize=(6,6))
    plt.scatter(r_major, v_major, s=12, facecolors='none', edgecolors='k', alpha=0.7, label='Observed data')
    plt.plot(r_fit, v_fit, color='deepskyblue', lw=3, label='Arctan fit')

    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel(r'$r\ [\mathrm{arcsec}]$', fontsize=13)
    plt.ylabel(r'$V\ [\mathrm{km\ s^{-1}}]$', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

################################################################################
# main function
################################################################################
def main():
    print(f"Plate-IFU {plateifu} MAPS")

    # get Galaxy spin velocity map from MAPS file
    # This value has been projection-corrected (line-of-sight velocity)
    vel_map, unit = mu.get_spin_vel_map(maps_file)
    print(f"Velocity map shape: {vel_map.shape}, Unit: {unit}")
    print(f"Velocity: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}] {unit}")
    plot_galaxy_image(plateifu)

    in_angle, pa_angle = cal_inc_angle(drpall_file, plateifu)
    print(f"Inclination angle (rad): {in_angle}, PA angle (rad): {pa_angle}")
    plot_velocity_map(vel_map, unit)

    r_ell, v_rot = cal_rotation_vel(vel_map, in_angle, pa_angle)
    print(f"Rotation velocity data points: {len(r_ell)}")
    plot_rotation_curve(r_ell, v_rot)


# main entry
if __name__ == "__main__":
    main()

