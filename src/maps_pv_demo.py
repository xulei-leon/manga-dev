
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
# plateifu = "8723-12703"
# get fits files
maps_file = fits_util.get_maps_file(plateifu)
drpall_file = fits_util.get_drpall_file()


BA_0 = 0.13  # Intrinsic Axial Ratio for edge-on galaxies, it is assumed value.
################################################################################
# calculate functions
################################################################################

# 倾角 i 的计算公式
# 倾角是星系盘法线与观测者视线之间的夹角。
# $$\mathbf{i} = \arccos \left( \sqrt{\frac{(\mathbf{b/a})^2 - (\mathbf{b/a})_0^2}{1 - (\mathbf{b/a})_0^2}} \right)$$
def calc_inc(ba):
    ba_sq = ba**2
    BA_0_sq = BA_0**2
    
    # 计算 cos^2(i) 的分子部分
    numerator = ba_sq - BA_0_sq
    denominator = 1.0 - BA_0_sq

    cos_i_sq = numerator / denominator
    cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)
    
    inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
    return inc_rad

# 真实径向距离 R 的计算公式
def calc_r_ell(x, y, pa_rad, inc_rad):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ny, nx = x.shape
    x_c = (nx - 1) / 2.0
    y_c = (ny - 1) / 2.0

    x_rel = x - x_c
    y_rel = y - y_c

    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    cos_inc = np.cos(inc_rad)
    y_deproj = y_rot / cos_inc

    r_ell = np.hypot(x_rot, y_deproj)
    return r_ell

# 对 MaNGA 的 velocity map 做几何校正
# vel_map (np.ndarray): 视线速度场 (km/s)，shape (ny, nx)。 相对于 systemic velocity。
# pa_rad (float): 位置角 PA（弧度）。
# inc_rad (float): 倾角 i（弧度）。
def vel_correct(vel_map, pa_rad, inc_rad):
    vel_map = np.asarray(vel_map, dtype=float)

    sin_inc = np.sin(inc_rad)
    if np.isclose(sin_inc, 0.0):
        return vel_map.copy()
    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    x_c = (nx - 1) / 2.0
    y_c = (ny - 1) / 2.0

    x_rel = x - x_c
    y_rel = y - y_c

    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    cos_inc = np.cos(inc_rad)
    if np.isclose(cos_inc, 0.0):
        return vel_map.copy()

    y_deproj = y_rot / cos_inc
    radius = np.hypot(x_rot, y_deproj)
    cos_theta = np.divide(x_rot, radius, out=np.zeros_like(x_rot), where=radius > 0)
    projection = sin_inc * cos_theta

    vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
    valid = (projection != 0) & np.isfinite(projection) & np.isfinite(vel_map)
    vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    return vel_map_corrected



def arctan_model(r, Vc, Rt, Vsys=0):
    """Arctangent rotation curve model."""
    return Vsys + (2 / np.pi) * Vc * np.arctan(r / Rt)

# calculate velocity dispersion
def calculate_vel_dispersion(ivar_map):
    v_disp = np.sqrt(1 / np.sum(ivar_map, axis=0))
    return v_disp

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
def plot_rv_curve(r_ell, v_rot):
    plt.figure(figsize=(6,6))
    plt.scatter(r_ell, v_rot, s=12, facecolors='none', edgecolors='k', alpha=0.7, label='Observed data')

    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel(r'$r_{ell}\ [\mathrm{arcsec}]$', fontsize=13)
    plt.ylabel(r'$V_{rot}\ [\mathrm{km\ s^{-1}}]$', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


################################################################################
# main function
################################################################################
def main():
    print(f"Plate-IFU {plateifu} MAPS")
    # plot_galaxy_image(plateifu)

    spaxel_size = mu.get_spaxel_size(maps_file)
    print(f"Spaxel Size: {spaxel_size:.3f} arcsec")

    # get Galaxy spin velocity map from MAPS file
    # This value has been projection-corrected (line-of-sight velocity)
    vel_map, unit, ivar_map = mu.get_gvel_map(maps_file)
    vel_disp = calculate_vel_dispersion(ivar_map)
    print(f"Velocity map shape: {vel_map.shape}, Unit: {unit}")
    print(f"Velocity: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}] {unit}")
    # plot_velocity_map(vel_map, unit)

    # calculate systemic velocity
    # ba: The degree of flattening of galaxies.
    phi, ba = du.get_phi_ba(drpall_file, plateifu)
    print(f"Position Angle PA from DRPALL: {phi:.2f} deg,", f"Axial Ratio b/a from DRPALL: ({ba:.3f})")

    pa_rad = np.radians(phi)
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {np.degrees(inc_rad):.3f} deg")

    x, y = np.indices(vel_map.shape)
    r_arcsec = calc_r_ell(x, y, pa_rad, inc_rad) * 1.0  # arcsec
    print(f"r_map: [{np.nanmin(r_arcsec):.3f}, {np.nanmax(r_arcsec):.3f}] arcsec", f"size: {len(r_arcsec)}")

    vel_map_corrected = vel_correct(vel_map, pa_rad, inc_rad)
    print(f"vel_map_corrected: [{np.nanmin(vel_map_corrected):.3f}, {np.nanmax(vel_map_corrected):.3f}] km/s", f"size: {len(vel_map_corrected)}")
    plot_velocity_map(vel_map_corrected, unit)

# main entry
if __name__ == "__main__":
    main()

