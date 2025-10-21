
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
# def vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0):
#     """
#     对观测到的速度场进行严格几何校正，并结合信噪比和方位角过滤。
#     同时生成对应的校正后半径图。

#     参数:
#         vel_map (np.ndarray): 校准后的视向速度图 (V_internal) (km/s)。
#         pa_rad (float): 星系的方位角 (PA)，单位为弧度 (Radians)。
#         inc_rad (float): 星系的倾角 (i)，单位为弧度 (Radians)。
#         snr_map (np.ndarray): 对应 vel_map 的中值信噪比图 (S/N)。
#         phi_limit_deg (float): 运动主轴两侧允许的最大方位角（度）。默认 60.0。

#     返回:
#         tuple[np.ndarray, np.ndarray]:
#         - vel_map_corrected (np.ndarray): 校正后的真实旋转速度 V_rot 场 (km/s)。
#         - r_map_corrected (np.ndarray): 对应的带符号的真实径向距离 R 场 (spaxel)。
#         不符合过滤条件的点在两个数组中均为 NaN。
#     """
#     vel_map = np.asarray(vel_map, dtype=float)
#     snr_map = np.asarray(snr_map, dtype=float)
    
#     # 1. 倾角参数准备
#     sin_inc = np.sin(inc_rad)
#     cos_inc = np.cos(inc_rad)
#     if np.isclose(sin_inc, 0.0):
#         nan_map = np.full_like(vel_map, np.nan, dtype=float)
#         return nan_map, nan_map

#     # 2. 坐标计算
#     ny, nx = vel_map.shape
#     y, x = np.indices((ny, nx))
#     x_c = (nx - 1) / 2.0
#     y_c = (ny - 1) / 2.0
#     x_rel = x - x_c
#     y_rel = y - y_c

#     # 3. 坐标旋转 (对齐到星系主轴)
#     cos_pa = np.cos(pa_rad)
#     sin_pa = np.sin(pa_rad)
#     x_rot = x_rel * cos_pa + y_rel * sin_pa
#     y_rot = -x_rel * sin_pa + y_rel * cos_pa

#     # 4. 去投影和计算 cos(phi)
#     if np.isclose(cos_inc, 0.0):
#         cos_inc = 1e-6 
        
#     y_deproj = y_rot / cos_inc
#     radius = np.hypot(x_rot, y_deproj)
#     cos_theta = np.divide(x_rot, radius, out=np.zeros_like(x_rot), where=radius > 0)
    
#     # 5. 投影因子
#     projection = sin_inc * cos_theta

#     # 6. 确定过滤阈值
#     cos_phi_threshold = np.cos(np.radians(phi_limit_deg))
#     snr_threshold = 10.0

#     # 7. 应用联合过滤掩码
#     valid = (
#         np.isfinite(vel_map) &
#         (radius > 0) &
#         (snr_map >= snr_threshold) &
#         (np.abs(cos_theta) >= cos_phi_threshold)
#     )
    
#     # 8. 计算校正后的速度图
#     vel_map_corrected = np.full_like(vel_map, np.nan, dtype=float)
#     vel_map_corrected[valid] = vel_map[valid] / projection[valid]
    
#     # 9. 计算校正后的半径图
#     # 半径的符号由原始速度图的符号决定
#     signs = np.sign(np.nan_to_num(vel_map, nan=0.0))
#     r_map_signed = radius * signs
    
#     # 对半径图应用相同的过滤掩码
#     r_map_corrected = np.full_like(r_map_signed, np.nan, dtype=float)
#     r_map_corrected[valid] = r_map_signed[valid]
    
#     return vel_map_corrected, r_map_corrected



def vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0, snr_min=3.0):
    """
    对观测到的速度场进行几何校正，并生成与旋转方向一致的带符号半径图。
    """

    ny, nx = vel_map.shape
    y, x = np.indices((ny, nx))
    x_cen = (nx - 1) / 2.0
    y_cen = (ny - 1) / 2.0

    # ---------------- Step 1: 坐标旋转 ----------------
    x_rel = x - x_cen
    y_rel = y - y_cen

    # 注意 FITS 坐标通常 y 轴反向，若你的 vel_map 显示上下颠倒，可尝试 y_rel = -(y - y_cen)
    x_rot = x_rel * np.cos(pa_rad) + y_rel * np.sin(pa_rad)
    y_rot = -x_rel * np.sin(pa_rad) + y_rel * np.cos(pa_rad)

    # ---------------- Step 2: 倾角修正 ----------------
    y_disk = y_rot / np.cos(inc_rad)
    r_map = np.sqrt(x_rot**2 + y_disk**2)
    phi = np.arctan2(y_disk, x_rot)

    phi_limit_rad = np.deg2rad(phi_limit_deg)
    valid_mask = (
        (np.abs(phi) <= phi_limit_rad) &
        (snr_map >= snr_min) &
        np.isfinite(vel_map)
    )

    # ---------------- Step 3: 几何速度校正 ----------------
    vel_map_corrected = np.full_like(vel_map, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        vel_map_corrected[valid_mask] = vel_map[valid_mask] / (
            np.sin(inc_rad) * np.cos(phi[valid_mask])
        )

    # ---------------- Step 4: 通过速度方向确定符号 ----------------
    # 用速度场的符号定义左右区域
    mask_pos = (vel_map_corrected > 0)
    mask_neg = (vel_map_corrected < 0)

    # 初步定义为无符号半径
    r_signed = np.copy(r_map)

    # 判断哪一侧是红移（正速度），哪一侧是蓝移（负速度）
    if np.nanmean(x_rot[mask_pos]) > np.nanmean(x_rot[mask_neg]):
        # 正速度在右 → 正半径在右
        r_signed = np.sign(x_rot) * r_map
    else:
        # 正速度在左 → 反转符号
        r_signed = -np.sign(x_rot) * r_map

    # 屏蔽无效点
    vel_map_corrected[~valid_mask] = np.nan
    r_signed[~valid_mask] = np.nan

    return vel_map_corrected, r_signed


# def vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0, snr_min=3.0):
#     """
#     对观测到的速度场进行严格几何校正，并结合信噪比和方位角过滤。
#     同时生成对应的校正后半径图。

#     此函数假设 vel_map 已经减去了系统速度 (V_sys = 0)，
#     并且星系中心位于图像 (ny/2, nx/2) 处。

#     几何推导:
#     1.  观测到的视向速度 V_los = V_rot(R) * sin(i) * cos(phi)
#         其中 i 是倾角, R 是在星系盘上的真实半径, phi 是在星系盘上的方位角
#         (phi=0 对应星系盘的主轴)。
#     2.  V_rot(R) = V_los / (sin(i) * cos(phi))
#     3.  我们需要计算 R 和 phi。
#         -   (x, y) 是图像坐标 (相对于中心)。
#         -   (x_rot, y_rot) 是在天球平面上旋转 PA 后的坐标：
#             x_rot = x*cos(PA) + y*sin(PA)  (沿天球主轴)
#             y_rot = -x*sin(PA) + y*cos(PA) (沿天球短轴)
#         -   (x', y') 是在星系盘平面上的真实坐标：
#             x' = x_rot
#             y' = y_rot / cos(i)
#         -   真实半径 R = sqrt(x'^2 + y'^2)
#         -   真实方位角 phi = arctan2(y', x')
#     4.  代入 V_rot 公式:
#         cos(phi) = x' / R = x_rot / R
#         V_rot(R) = V_los / (sin(i) * (x_rot / R))
#         V_rot(R) = (V_los * R) / (sin(i) * x_rot)
#     5.  我们返回 V_rot (作为速度图) 和 R_signed = R * sign(x_rot) (作为半径图)。

#     参数:
#         vel_map (np.ndarray): 校准后的视向速度图 (km/s)，shape (ny, nx)。
#         pa_rad (float): 星系的方位角 (Position Angle, PA)，单位弧度。
#         (PA=0 沿+y轴, PA=pi/2 沿+x轴)
#         inc_rad (float): 星系的倾角 (Inclination Angle, i)，单位弧度。
#         (i=0 为 face-on, i=pi/2 为 edge-on)
#         snr_map (np.ndarray): 对应 vel_map 的信噪比图。
#         phi_limit_deg (float): 运动主轴允许的最大方位角 (度)，默认 ±60°。
#         (过滤掉靠近短轴的区域，因为 cos(phi) -> 0 会放大噪声)
#         snr_min (float): S/N 阈值，低于此值的像素剔除。默认 3.0。

#     返回:
#         tuple[np.ndarray, np.ndarray]:
#         - vel_map_corrected (np.ndarray): 校正后的旋转速度图 (km/s)，
#             即 V_rot。不满足条件的像素为 np.nan。
#         - r_map_corrected (np.ndarray): 对应的带符号半径图 (以像素或 spaxel 计)，
#             即 R * sign(x_rot)。不满足条件的像素为 np.nan。
#     """
    
#     # --- 1. 处理数值稳定性和常量 ---
    
#     # 抑制在计算 phi 和 R 时遇到的除零警告 (例如 i=90度 或 x_rot=0)。
#     # 这些情况会产生 Inf 或 NaN，但会被后续的 phi 掩码正确处理。
#     with np.errstate(divide='ignore', invalid='ignore'):
        
#         sin_i = np.sin(inc_rad)
#         cos_i = np.cos(inc_rad)
        
#         # 如果星系是 face-on (i=0)，sin(i)=0，无法进行反投影。
#         # 返回两个填充了 NaN 的图。
#         if np.abs(sin_i) < 1e-9:
#             nan_map = np.full_like(vel_map, np.nan)
#             return nan_map, nan_map

#         # 将 phi 限制从度转换为弧度
#         phi_limit_rad = np.deg2rad(phi_limit_deg)
        
#         # --- 2. 创建相对于中心的坐标网格 ---
#         ny, nx = vel_map.shape
#         # 假设中心在 (ny-1)/2, (nx-1)/2
#         y_c = (ny - 1) / 2.0
#         x_c = (nx - 1) / 2.0
        
#         # y_idx, x_idx 是像素索引
#         y_idx, x_idx = np.indices((ny, nx))
        
#         # y_rel, x_rel 是相对于中心的坐标
#         # 注意：PA 是从+y轴 (North) 转向+x轴 (East)
#         y_rel = y_idx - y_c
#         x_rel = x_idx - x_c
        
#         # --- 3. 旋转天球平面坐标 (x, y) -> (x_rot, y_rot) ---
#         # x_rot 是沿天球平面主轴的距离
#         # y_rot 是沿天球平面短轴的距离
#         sin_pa = np.sin(pa_rad)
#         cos_pa = np.cos(pa_rad)
        
#         # 标准2D旋转矩阵 (逆时针旋转 -PA)
#         # x_rot = x_rel * cos(-PA) - y_rel * sin(-PA) = x_rel * cos(PA) + y_rel * sin(PA)
#         # y_rot = x_rel * sin(-PA) + y_rel * cos(-PA) = -x_rel * sin(PA) + y_rel * cos(PA)
#         # 修正：天文学 PA 是从 N(+y) 向 E(+x) 旋转。
#         # 所以旋转 -PA 应该是：
#         x_rot = x_rel * cos_pa + y_rel * sin_pa
#         y_rot = -x_rel * sin_pa + y_rel * cos_pa
        
#         # --- 4. 反投影坐标到星系盘平面 (x', y') ---
#         # x' 是沿真实主轴的距离
#         # y' 是沿真实短轴的距离
#         x_prime = x_rot
#         # 如果 i = 90度 (edge-on), cos_i = 0, y_prime 会是 Inf (这是正确的)
#         y_prime = y_rot / cos_i
        
#         # --- 5. 计算星系盘平面半径 (R) 和方位角 (phi) ---
        
#         # r_in_plane = R (真实半径)
#         # 如果 i = 90度, y_prime=Inf, R 也会是 Inf
#         r_in_plane = np.sqrt(x_prime**2 + y_prime**2)
        
#         # phi_in_plane = phi (真实方位角)
#         # np.arctan2 正确处理 x_prime=0 (短轴, phi= +/- 90度)
#         # 和 y_prime=Inf (edge-on, phi= +/- 90度)
#         phi_in_plane = np.arctan2(y_prime, x_prime)
        
#         # --- 6. 创建过滤掩码 (Masks) ---
        
#         # a) S/N 掩码
#         mask_snr = (snr_map >= snr_min)
        
#         # b) 方位角掩码
#         # 过滤掉 phi > limit (即靠近短轴的区域)
#         # 这会自动过滤掉 i=90度 (edge-on) 的大部分区域 (phi= +/- 90)
#         # 和 i!=90度 时的短轴 (phi= +/- 90)
#         mask_phi = (np.abs(phi_in_plane) <= phi_limit_rad)
        
#         # c) 总掩码
#         mask_total = mask_snr & mask_phi
        
#         # --- 7. 计算校正后的速度和半径 ---
        
#         # V_rot = (V_los * R) / (sin(i) * x_rot)
#         # V_los = vel_map
#         # R = r_in_plane
#         numerator = vel_map * r_in_plane
#         denominator = sin_i * x_rot
        
#         # 计算 V_rot。
#         # mask_phi 已经排除了 x_rot 接近 0 的情况，因此避免了除零。
#         # V_rot 物理上应为正值 (速度大小)。
#         # 如果 V_los 和 x_rot 符号相反 (例如，本应后退的区域在接近)，
#         # v_rot_raw 会是负值，反映了非圆周运动或噪声。
#         # 我们返回其绝对值，即旋转 *速率* (speed)。
#         v_rot_raw = numerator / denominator
#         v_rot_speed = np.abs(v_rot_raw)
        
#         # 计算带符号的半径 R_signed = R * sign(x_rot)
#         # sign(x_rot) = +1 表示在主轴的一侧，-1 表示在另一侧
#         r_signed = r_in_plane * np.sign(x_rot)
        
#         # --- 8. 应用掩码，生成最终图像 ---
        
#         # 使用 np.where，只在 mask_total 为 True 的地方填充计算值，
#         # 其他地方填充 np.nan。
#         vel_map_corrected = np.where(mask_total, v_rot_speed, np.nan)
#         r_map_corrected = np.where(mask_total, r_signed, np.nan)

#     return vel_map_corrected, r_map_corrected

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
    r_map = calc_r_map(vel_map)
    plot_rv_curve(r_map.flatten(), vel_map.flatten())


    #
    # correct velocity map to get true rotation velocity map
    #
    snr_map = map_util.get_snr_map()
    print(f"SNR map shape: {snr_map.shape}")
    print(f"SNR: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

    # phi: The position angle of the major axis of the galaxy, measured from north to east.
    # ba: The axis ratio (b/a) of the galaxy
    phi, ba = drpall_util.get_phi_ba(plateifu)
    print(f"Position Angle PA from DRPALL: {phi:.2f} deg,", f"Axial Ratio b/a from DRPALL: {ba:.3f}")

    pa_rad = np.radians(phi)
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {pa_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    # FIXME: vel_map_correct does not work well
    vel_map_corr, r_map_corr = vel_map_correct(vel_map, pa_rad, inc_rad, snr_map, phi_limit_deg=60.0)
    print(f"vel_map_corrected: [{np.nanmin(vel_map_corr):.3f}, {np.nanmax(vel_map_corr):.3f}] km/s", f"size: {len(vel_map_corr)}")
    print(f"r_map_corrected: [{np.nanmin(r_map_corr):.3f}, {np.nanmax(r_map_corr):.3f}] arcsec", f"size: {len(r_map_corr)}")

    # plot r-v curve
    valid_idx = ~np.isnan(vel_map_corr) & ~np.isnan(r_map_corr)
    r_flat = r_map_corr[valid_idx]
    v_flat = vel_map_corr[valid_idx]
    plot_rv_curve(r_flat, v_flat)


# main entry
if __name__ == "__main__":
    main()

