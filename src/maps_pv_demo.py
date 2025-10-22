
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

def calc_r_corr(vel_map):
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

    参数:
        vel_map (np.ndarray): 校准后的视向速度图 (V_internal) (km/s)。
        pa_rad (float): 星系的方位角 (PA)，单位为弧度 (Radians)。
        inc_rad (float): 星系的倾角 (i)，单位为弧度 (Radians)。
        snr_map (np.ndarray): 对应 vel_map 的中值信噪比图 (S/N), 阈值为10。
        phi_limit_deg (float): 运动主轴两侧允许的最大方位角（度）。默认 60.0。
        center_x (float, optional): 图像中心的 X 坐标。如果为 None，则默认为 (nx - 1) / 2.0。
        center_y (float, optional): 图像中心的 Y 坐标。如果为 None，则默认为 (ny - 1) / 2.0。

    返回:
        np.ndarray:
        - vel_map_corrected (np.ndarray): 校正后的真实旋转速度 V_rot 场 (km/s)。
        不符合过滤条件的点在数组中均为 NaN。
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
        return nan_map

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
    
    return vel_map_corrected

# 将v_map 的值进行绝对值处理
def vel_map_abs(vel_map):
    vel_map = np.asarray(vel_map, dtype=float)
    vel_map_abs = np.abs(vel_map)
    return vel_map_abs


# 根据vel_map 生成几何校正后的r_map
def calc_r_corr(vel_map, pa_rad, inc_rad):
    """
    根据速度图、方位角和倾角，生成几何校正后的径向距离图 R。
    这个 R 是在星系盘平面内的真实径向距离，并带有由速度图决定的符号。

    参数:
        vel_map (np.ndarray): 速度图，用于获取形状和符号。
        pa_rad (float): 星系的方位角 (PA)，单位为弧度。
        inc_rad (float): 星系的倾角 (i)，单位为弧度。

    返回:
        np.ndarray: 几何校正后且带有符号的径向距离图 (单位: spaxel)。
    """
    vel_map = np.asarray(vel_map, dtype=float)
    ny, nx = vel_map.shape
    x_c = (nx - 1) / 2.0
    y_c = (ny - 1) / 2.0

    y, x = np.indices((ny, nx))
    x_rel = x - x_c
    y_rel = y - y_c

    # 坐标旋转，对齐到星系主轴
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_rot = x_rel * cos_pa + y_rel * sin_pa
    y_rot = -x_rel * sin_pa + y_rel * cos_pa

    # 去投影，计算在星系盘平面内的坐标
    cos_inc = np.cos(inc_rad)
    # 防止除以零
    if np.isclose(cos_inc, 0.0):
        cos_inc = 1e-6  # 对于 edge-on 星系，使用一个很小的值
        
    y_deproj = y_rot / cos_inc

    # 计算去投影后的绝对径向距离 |R|
    r_abs = np.hypot(x_rot, y_deproj)

    # 使用速度图的符号来给径向距离赋符号
    # NaN 速度被视为 0，其符号也为 0
    signs = np.sign(np.nan_to_num(vel_map, nan=0.0))
    r_map_signed = r_abs * signs
    
    return r_map_signed

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

# $$\mathbf{V}_{\text{rot}} = \frac{\mathbf{v}_{\text{internal}}}{\sin(\text{inc\_rad}) \cdot \cos(\text{phi\_rad})}$$
def calc_v_rot(vel_map, r_map, inc_rad, phi_rad_map, phi_limit_deg=60.0):
    """
    Calculates the rotation velocity V_rot, filtering data based on the azimuth angle.

    Args:
        vel_map (np.ndarray): The observed velocity map.
        r_map (np.ndarray): The radius map.
        inc_rad (float): The galaxy inclination in radians.
        phi_rad_map (np.ndarray): The azimuth angle map in radians.
        phi_limit_deg (float): The maximum allowed azimuth angle from the major axis in degrees.

    Returns:
        np.ndarray: The calculated rotation velocity map, with invalid points set to NaN.
    """
    vel_map = np.asarray(vel_map, dtype=float)
    r_map = np.asarray(r_map, dtype=float)
    phi_rad_map = np.asarray(phi_rad_map, dtype=float)

    # Initialize the output array with NaNs
    v_rot = np.full_like(vel_map, np.nan, dtype=float)

    # Calculate the projection factor components
    sin_inc = np.sin(inc_rad)
    cos_phi = np.cos(phi_rad_map)

    # Define the filtering threshold for the azimuth angle
    cos_phi_threshold = np.cos(np.radians(phi_limit_deg))

    # Create a mask for valid data points
    # Valid points are those within the allowed angle from the major axis
    # and where the projection factor is not zero.
    valid_mask = (np.abs(cos_phi) >= cos_phi_threshold) & (sin_inc != 0)

    # Avoid division by zero by creating the projection factor array
    projection_factor = np.zeros_like(vel_map, dtype=float)
    projection_factor[valid_mask] = sin_inc * cos_phi[valid_mask]

    # Calculate V_rot only for valid points to avoid division by zero errors
    # Create a mask to prevent division by zero where projection_factor is zero
    # This is a bit redundant with valid_mask but safer.
    non_zero_proj_mask = projection_factor != 0
    
    v_rot[non_zero_proj_mask] = vel_map[non_zero_proj_mask] / projection_factor[non_zero_proj_mask]

    return v_rot


def vel_map_snr_filter(vel_map, snr_map, snr_threshold=10.0):
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
def calculate_vel_dispersion(ivar_map):
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


import numpy as np

def calc_v_rot2(
    vel_map,
    R_map,
    phi_map,
    inc_rad,
    snr_map=None,
    phi_limit_deg=60.0,
    snr_min=10.0,
    v_sys=None,
    center_kernel_radius=2.0,
    cos_phi_eps=1e-3
):
    """
    使用盘面坐标 (R_map, phi_map) 对视向速度进行几何校正，得到圆周速度 v_rot，
    并返回与 v_rot 空间正负号一致的带符号半径 r_signed。

    参数:
        vel_map (np.ndarray): 2D 视向速度 (km/s)，shape (ny,nx)。
        R_map (np.ndarray): 2D 去投影后的盘内半径（单位由你决定，例如 arcsec 或像素）。
        phi_map (np.ndarray): 2D 盘面方位角 (radians)，主轴处通常为 phi=0。
        inc_rad (float): 倾角 (radians)。
        snr_map (np.ndarray or None): S/N 图；若 None，将不使用 S/N 筛选。
        phi_limit_deg (float): 允许的方位角范围（±度），默认 60°。
        snr_min (float): S/N 最小阈值（若 snr_map 提供）。
        v_sys (float or None): 系统速度（km/s），若 None 自动估计中心区域中位数。
        center_kernel_radius (float): 若 v_sys None，则在 R_map < center_kernel_radius 的区域估计 v_sys。
        cos_phi_eps (float): 当 |cos(phi)| 小于此值时认为无法可靠校正并屏蔽（默认 1e-3）。
    返回:
        v_rot (np.ndarray): 校正后的圆周速度 (km/s)，在无效/被屏蔽处为 np.nan。
        r_signed (np.ndarray): 带符号半径（与 v_rot 空间符号一致），无效处为 np.nan。
        mask_valid (np.ndarray): 布尔数组，表示用于计算的有效像素。
    """

    # 输入检查
    if vel_map.shape != R_map.shape or vel_map.shape != phi_map.shape:
        raise ValueError("vel_map, R_map, phi_map 必须有相同的 shape")

    ny, nx = vel_map.shape

    # 1) 估计系统速度 v_sys（若未给）
    if v_sys is None:
        # 用中心区域（R_map < center_kernel_radius）且有效的像素估计中位数
        central_mask = (R_map <= center_kernel_radius) & np.isfinite(vel_map)
        if np.any(central_mask):
            v_sys_est = np.nanmedian(vel_map[central_mask])
        else:
            # 退而求其次：整体有限值的中位数
            all_mask = np.isfinite(vel_map)
            if np.any(all_mask):
                v_sys_est = np.nanmedian(vel_map[all_mask])
            else:
                v_sys_est = 0.0
        v_sys = v_sys_est

    # 2) 构造有效像素掩码
    phi_limit_rad = np.deg2rad(phi_limit_deg)
    finite_mask = np.isfinite(vel_map) & np.isfinite(R_map) & np.isfinite(phi_map) & np.isfinite(inc_rad)
    angle_mask = np.abs(phi_map) <= phi_limit_rad
    cos_phi = np.cos(phi_map)
    cos_mask = np.abs(cos_phi) > cos_phi_eps
    snr_mask = np.ones_like(vel_map, dtype=bool) if (snr_map is None) else (snr_map >= snr_min)

    mask_valid = finite_mask & angle_mask & cos_mask & snr_mask

    # 3) 速度校正： v_rot = (v_obs - v_sys) / (sin(i) * cos(phi))
    v_rot = np.full_like(vel_map, np.nan, dtype=float)
    sin_i = np.sin(inc_rad)
    if sin_i == 0:
        raise ValueError("inc_rad 对应 sin(i)=0，倾角为 0（面朝观测者），无法校正")

    # 以数值安全的方式计算
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (vel_map - v_sys)
        denom = sin_i * cos_phi
        v_rot_temp = numerator / denom
        v_rot[mask_valid] = v_rot_temp[mask_valid]

    # 把其余位置置为 nan（已经如此）
    v_rot[~mask_valid] = np.nan

    # 4) 生成与速度符号一致的带符号半径
    # 简单且物理合理的实现：按每像素速度的符号给半径赋正负
    r_signed = np.full_like(R_map, np.nan, dtype=float)
    valid_idxs = mask_valid
    # 若 v_rot 在某像素为 nan，则跳过
    idxs = valid_idxs & np.isfinite(v_rot)
    # 将 r_signed = sign(v_rot) * R_map（使得速度为正的一侧半径为正）
    r_signed[idxs] = np.sign(v_rot[idxs]) * R_map[idxs]

    return v_rot, r_signed

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
    # get basic info from FITS files
    ########################################################
    spaxel_x, spaxel_y = map_util.get_spaxel_size()
    print(f"Spaxel Size: {spaxel_x:.3f} arcsec (X), {spaxel_y:.3f} arcsec (Y)")

    # correct velocity map to get true rotation velocity map
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

    phi_rad = np.radians(phi)
    inc_rad = calc_inc(ba)
    print(f"pa_rad: {phi_rad:.3f}, inc_rad = {inc_rad:.3f} ({np.degrees(inc_rad):.2f} deg)")

    # get r map and azimuth map from MAPS file
    r_map, azimuth_map = map_util.get_r_map()
    phi_rad_map = np.radians(azimuth_map) 
    print(f"r_map: [{np.nanmin(r_map):.3f}, {np.nanmax(r_map):.3f}] spaxel,", f"shape: {r_map.shape}")


    ########################################################
    # Galaxy spin velocity map
    ########################################################
    _gv_map, _gv_unit, _gv_ivar = map_util.get_gvel_map()
    _gv_disp = calculate_vel_dispersion(_gv_ivar)
    print(f"Velocity map shape: {_gv_map.shape}, Unit: {_gv_unit}")
    print(f"Velocity: [{np.nanmin(_gv_map):.3f}, {np.nanmax(_gv_map):.3f}] {_gv_unit}")

    _sv_map, _sv_unit, _sv_ivar = map_util.get_stellar_vel_map()
    print(f"Stellar velocity map shape: {_sv_map.shape}, Unit: {_sv_unit}")
    print(f"Stellar Velocity: [{np.nanmin(_sv_map):.3f}, {np.nanmax(_sv_map):.3f}] {_sv_unit}")

    _v_sys = calc_v_sys(_sv_map, size=3)
    print(f"Estimated Systemic Velocity V_sys: {_v_sys:.3f} {_sv_unit}")

    v_line_map = _gv_map - _v_sys
    v_unit = _gv_unit
    print(f"Centered Velocity map shape: {v_line_map.shape}, Unit: {v_unit}")
    print(f"Centered Velocity: [{np.nanmin(v_line_map):.3f}, {np.nanmax(v_line_map):.3f}] {v_unit}")


    ########################################################
    # Correct velocity and R
    ########################################################
    # r_corr = calc_r_corr(_gv_map, phi_rad, inc_rad)
    # print(f"Calculated r_map: [{np.nanmin(r_corr):.3f}, {np.nanmax(r_corr):.3f}] spaxel,", f"shape: {r_corr.shape}")

    # # FIXME: vel_map_correct does not work well
    # vel_corr = vel_map_correct(_gv_map, phi_rad, inc_rad, snr_map, phi_limit_deg=60.0)
    # print(f"vel_map_corrected: [{np.nanmin(vel_corr):.3f}, {np.nanmax(vel_corr):.3f}] km/s", f"size: {len(vel_corr)}")
    # vel_abs = vel_map_abs(vel_corr)

    # v_rot_map = calc_v_rot(v_line_map, r_map, inc_rad, phi_rad_map, phi_limit_deg=60.0)
    # v_rot_map = vel_map_snr_filter(v_rot_map, snr_map, snr_threshold=10.0)
    # print(f"Velocity map after geometric correction and SNR filtering: [{np.nanmin(v_rot_map):.3f}, {np.nanmax(v_rot_map):.3f}] km/s", f"size: {len(v_rot_map)}")

    v_rot_map, r_rot_map = calc_v_rot2(v_line_map, r_map, phi_rad_map, inc_rad, snr_map=snr_map, phi_limit_deg=60.0, snr_min=10.0, v_sys=0.0)

    ########################################################
    # plot
    ########################################################

    # 1. plot plateifu map
    # plot_galaxy_image(plateifu)

    # 2. plot velocity map
    plot_velocity_map(v_line_map, v_unit)


    # 3. plot r-v curve
    # valid_idx = ~np.isnan(v_rot_map)
    # r_flat = r_rot_map[valid_idx]
    # v_flat = v_rot_map[valid_idx]
    valid_idx = ~np.isnan(v_line_map)
    r_flat = r_map[valid_idx]
    v_flat = v_line_map[valid_idx]
    plot_rv_curve(r_flat, v_flat)


# main entry
if __name__ == "__main__":
    main()

