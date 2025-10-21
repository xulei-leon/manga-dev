
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



BA_0 = 0.13  # Intrinsic Axial Ratio for edge-on galaxies, it is assumed value.

################################################################################
# calculate functions
################################################################################

# Calculate the galaxy inclination i (in radians)
# Formula for inclination i
# The inclination is the angle between the galaxy disk normal and the observer's line of sight.
# ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
def calc_inc(ba):
    ba_sq = ba**2
    BA_0_sq = BA_0**2
    
    # Compute the numerator part of cos^2(i)
    numerator = ba_sq - BA_0_sq
    denominator = 1.0 - BA_0_sq

    cos_i_sq = numerator / denominator
    cos_i_sq_clipped = np.clip(cos_i_sq, 0.0, 1.0)
    
    inc_rad = np.arccos(np.sqrt(cos_i_sq_clipped))
    return inc_rad

# Calculate the true radial distance R_ell (in spaxels)
# r_ell: elliptical radius from the galaxy center, in spaxels
# pa_rad: position angle PA (radians)
# inc_rad: inclination i (radians)
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

def r_map_correct(vel_map, pa_rad, inc_rad, x_c=None, y_c=None):
    """
    计算星系盘平面内的真实径向距离 R，并保留正负号，
    其中符号由速度图 vel_map 的符号决定。

    参数:
        vel_map (np.ndarray): 速度图，用于获取形状 (ny, nx) 和符号。
        pa_rad (float): 星系的方位角 (PA)，单位为弧度。
        inc_rad (float): 星系的倾角 (i)，单位为弧度。
        x_c (float, optional): 几何中心 X 坐标 (spaxel)。如果为 None，使用 (nx - 1) / 2.0。
        y_c (float, optional): 几何中心 Y 坐标 (spaxel)。如果为 None，使用 (ny - 1) / 2.0。

    返回:
        np.ndarray: 带有正负号的径向距离 R 场 (单位: spaxel)。
    """
    ny, nx = vel_map.shape
    
    # 1. 确定几何中心
    if x_c is None:
        x_c = (nx - 1) / 2.0
    if y_c is None:
        y_c = (ny - 1) / 2.0

    # 2. 坐标中心化
    y, x = np.indices((ny, nx))
    x_rel = x - x_c
    y_rel = y - y_c

    # 3. 坐标旋转 (对齐到星系主轴)
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    # 4. 去投影 (沿短轴方向)
    cos_inc = np.cos(inc_rad)
    
    if np.isclose(cos_inc, 0.0):
        cos_inc = 1e-6 

    y_deproj = y_rot / cos_inc
    
    # 5. 计算绝对径向距离 |R|
    r_abs = np.hypot(x_rot, y_deproj)
    
    # 6. 应用符号
    # 符号由 vel_map 决定
    # 将 NaN 速度视为 0，其符号为 0
    signs = np.sign(np.nan_to_num(vel_map, nan=0.0))
    r_signed = r_abs * signs
    
    return r_signed

def calc_r_map(vel_map):
    vel_map = np.asarray(vel_map, dtype=float)
    ny, nx = vel_map.shape
    x_c = (nx - 1) / 2.0
    y_c = (ny - 1) / 2.0

    y, x = np.indices((ny, nx))
    x_rel = x - x_c
    y_rel = y - y_c

    r_map = np.hypot(x_rel, y_rel)

    # Use the sign of vel_map; treat NaN velocities as zero (no sign)
    signs = np.sign(np.nan_to_num(vel_map, nan=0.0))
    r_map_signed = r_map * signs
    return r_map_signed

# Geometric correction for the MaNGA velocity map
def vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0, center_x=None, center_y=None):
    """
    对观测到的速度场进行严格几何校正，并结合信噪比和方位角过滤。
    同时生成对应的校正后半径图。

    参数:
        vel_map (np.ndarray): 校准后的视向速度图 (V_internal) (km/s)。
        pa_rad (float): 星系的方位角 (PA)，单位为弧度 (Radians)。
        inc_rad (float): 星系的倾角 (i)，单位为弧度 (Radians)。
        snr_map (np.ndarray): 对应 vel_map 的中值信噪比图 (S/N), 阈值为10。
        phi_limit_deg (float): 运动主轴两侧允许的最大方位角（度）。默认 60.0。
        center_x (float, optional): 图像中心的 X 坐标。如果为 None，则默认为 (nx - 1) / 2.0。
        center_y (float, optional): 图像中心的 Y 坐标。如果为 None，则默认为 (ny - 1) / 2.0。

    返回:
        tuple[np.ndarray, np.ndarray]:
        - vel_map_corrected (np.ndarray): 校正后的真实旋转速度 V_rot 场 (km/s)。
        - r_map_corrected (np.ndarray): 对应的带符号的真实径向距离 R 场 (spaxel)。
        不符合过滤条件的点在两个数组中均为 NaN。
    """
    vel_map = np.asarray(vel_map, dtype=float)
    snr_map = np.asarray(snr_map, dtype=float)
    
    # 1. 倾角参数准备
    sin_inc = np.sin(inc_rad)
    cos_inc = np.cos(inc_rad)

    if np.isclose(sin_inc, 0.0):
        # 面对面观测（Face-on），无法进行去投影
        print("Warning: Inclination is close to 0 (face-on). Cannot deproject velocity field.")
        nan_map = np.full_like(vel_map, np.nan, dtype=float)
        return nan_map, nan_map

    # 2. 坐标计算
    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    
    # 使用提供的中心坐标，否则使用几何中心
    x_c = center_x if center_x is not None else (nx - 1) / 2.0
    y_c = center_y if center_y is not None else (ny - 1) / 2.0
    
    x_rel = x - x_c
    y_rel = y - y_c

    # 3. 坐标旋转 (对齐到星系主轴)
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    # x_rot 沿着星系动力学长轴（运动主轴）
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    # y_rot 沿着星系短轴
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    # 4. 去投影和计算 cos(phi)
    # 对于侧向观测 (edge-on)，cos_inc 接近 0，y_deproj 会变得非常大
    y_deproj = y_rot / cos_inc 
    radius = np.hypot(x_rot, y_deproj) # 真实的、去投影后的径向距离

    # cos(phi) 是运动方向与视线方向夹角的余弦值，这里使用 x_rot / radius
    # 使用 np.divide 避免除以零的警告
    cos_phi = np.divide(x_rot, radius, out=np.zeros_like(x_rot), where=radius > 0)
    
    # 5. 投影因子 (V_obs = V_rot * projection)
    projection = sin_inc * cos_phi

    # 6. 确定过滤阈值
    cos_phi_threshold = np.cos(np.radians(phi_limit_deg))
    snr_threshold = 10.0

    # 7. 应用联合过滤掩码
    valid = (
        np.isfinite(vel_map) &            # 确保输入速度有效
        (radius > 0) &                     # 排除中心点
        (snr_map >= snr_threshold) &       # 应用信噪比阈值
        (np.abs(cos_phi) >= cos_phi_threshold) # 应用方位角阈值 (运动主轴 ± 60度)
    )
    
    # 8. 计算校正后的速度图 (V_rot = V_obs / projection)
    vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
    # 仅对有效数据点进行除法操作
    vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    
    # 9. 计算校正后的半径图（带符号）
    # 半径的符号取决于观测速度的方向（假设速度场已中心化）
    # 使用 nan_to_num(nan=0.0) 避免 np.sign 遇到 NaN 报错
    signs = np.sign(np.nan_to_num(vel_map, nan=0.0))
    r_map_signed = radius * signs
    
    # 对带符号的半径图应用相同的过滤掩码
    r_map_corrected = np.full_like(r_map_signed, np.nan, dtype=float)
    r_map_corrected[valid] = r_map_signed[valid]
    
    return vel_map_corrected, r_map_corrected

import numpy as np

def vel_map_correct_FIXED(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0, center_x=None, center_y=None):
    """
    对观测到的速度场进行严格几何校正，并结合信噪比和方位角过滤。
    同时生成对应的校正后半径图。
    """
    vel_map = np.asarray(vel_map, dtype=float)
    snr_map = np.asarray(snr_map, dtype=float)
    
    sin_inc = np.sin(inc_rad)
    cos_inc = np.cos(inc_rad)

    if np.isclose(sin_inc, 0.0):
        print("Warning: Inclination is close to 0 (face-on). Cannot deproject velocity field.")
        nan_map = np.full_like(vel_map, np.nan, dtype=float)
        return nan_map, nan_map

    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    x_c = center_x if center_x is not None else (nx - 1) / 2.0
    y_c = center_y if center_y is not None else (ny - 1) / 2.0
    x_rel = x - x_c
    y_rel = y - y_c

    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    y_deproj = y_rot / cos_inc 
    radius_physical = np.hypot(x_rot, y_deproj) # 真实的、去投影后的物理距离 (总是正值)

    cos_phi = np.divide(x_rot, radius_physical, out=np.zeros_like(x_rot), where=radius_physical > 0)
    
    projection = sin_inc * cos_phi

    cos_phi_threshold = np.cos(np.radians(phi_limit_deg))
    snr_threshold = 10.0

    valid = (
        np.isfinite(vel_map) &
        (radius_physical > 0) &
        (snr_map >= snr_threshold) &
        (np.abs(cos_phi) >= cos_phi_threshold)
    )
    
    vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
    # V_rot = V_obs / projection. 这个速度V_rot现在带有正确的旋转方向符号（正负）
    vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    
    r_map_corrected = np.full_like(radius_physical, np.nan, dtype=float)
    
    # 核心修复点：使用几何信息（x_rot的符号）来定义带符号的半径，或者直接使用物理半径
    # 如果要生成用于绘图旋转曲线的 R vs V 数据（即 R>0, V_rot有正负）：
    r_map_corrected[valid] = radius_physical[valid] # 保持半径为正值
    
    return vel_map_corrected, r_map_corrected


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
    # plot_galaxy_image(plateifu)

    # get fits files
    maps_file = fits_util.get_maps_file(plateifu)
    drpall_file = fits_util.get_drpall_file()

    drpall_util = DrpallUtil(drpall_file)
    # print(f"DRPALL Info for {plateifu}: {drpall_util.dump_info()}")

    map_util = MapsUtil(maps_file)
    # print(f"MAPS Info for {plateifu}: {map_util.dump_info()}")


    spaxel_x, spaxel_y = map_util.get_spaxel_size()
    print(f"Spaxel Size: {spaxel_x:.3f} arcsec (X), {spaxel_y:.3f} arcsec (Y)")

    # get Galaxy spin velocity map from MAPS file
    # This value has been projection-corrected (line-of-sight velocity)
    vel_map, unit, ivar_map = map_util.get_gvel_map()
    vel_disp = calculate_vel_dispersion(ivar_map)
    print(f"Velocity map shape: {vel_map.shape}, Unit: {unit}")
    print(f"Velocity: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}] {unit}")
    # plot_velocity_map(vel_map, unit)


    #
    # plot r-v curve without correction
    #
    r_map, azimuth_map = map_util.get_r_map()
    print(f"r_map: [{np.nanmin(r_map):.3f}, {np.nanmax(r_map):.3f}] spaxel,", f"shape: {r_map.shape}")
    print(f"Azimuth map: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] deg,", f"shape: {azimuth_map.shape}")

    plot_rv_curve(r_map.flatten(), vel_map.flatten())


    #
    # correct velocity map to get true rotation velocity map
    #
    snr_map = map_util.get_snr_map()
    print(f"SNR map shape: {snr_map.shape}")
    print(f"SNR: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

    # phi: The position angle of the major axis of the galaxy, measured from north to east.
    # ba: The axis ratio (b/a) of the galaxy
    phi, ba_1 = map_util.get_pa_inc()
    print(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination (1-b/a) from MAPS header: {ba_1:.3f}")
    if ba_1 is not None:
        ba = 1 - ba_1  # convert to b/a
    if phi is None or ba is None:
        phi, ba = drpall_util.get_phi_ba(plateifu)
        print(f"Position Angle PA from DRPALL: {phi:.2f} deg,", f"Axial Ratio b/a from DRPALL: {ba:.3f}")


    pa_rad = np.radians(phi) + np.pi 
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    # FIXME: vel_map_correct does not work well
    vel_map_corr, __r = vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0)
    print(f"vel_map_corrected: [{np.nanmin(vel_map_corr):.3f}, {np.nanmax(vel_map_corr):.3f}] km/s", f"size: {len(vel_map_corr)}")

    # plot r-v curve
    valid_idx = ~np.isnan(vel_map_corr) & ~np.isnan(r_map)
    r_flat = r_map[valid_idx]
    v_flat = vel_map_corr[valid_idx]
    plot_rv_curve(r_flat, v_flat)


# main entry
if __name__ == "__main__":
    main()

