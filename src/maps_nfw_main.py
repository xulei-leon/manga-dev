"""
MaNGA DAP HYB10-MILESHC-MASTARHC2 数据分析脚本
------------------------------------------------
目标:
1. 从 MAPS 文件中提取 Hα 发射线速度场；
2. 进行系统速度校正、倾角与几何投影修正；
3. 提取旋转曲线；
4. 计算包裹质量与质量密度；
5. 拟合 NFW 模型并绘图。
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from astropy.constants import G

plateifu = "7957-3701"
plate, ifu = plateifu.split("-")

import pathlib
source_dir = pathlib.Path(__file__).resolve().parent
data_dir = pathlib.Path(source_dir / "../data")
drpall_path = pathlib.Path(data_dir / "redux/drpall-v3_1_1.fits")
logcube_path = pathlib.Path(data_dir / f"redux/manga-{plate}-{ifu}-LOGCUBE.fits.gz")
maps_path = pathlib.Path(data_dir / f"analysis/manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz")
emiles_template_path = pathlib.Path(data_dir / "ppxf/spectra_emiles_9.0.npz")

# =========================================
# === 1. 用户参数设置 =======================
# =========================================
maps_file = maps_path
galaxy_incl_deg = 60.0    # 倾角 i（来自光学轴比 b/a，可手动估计或用 photometry 拟合）
pa_deg = 45.0             # 位置角 PA（东至北方向度数）
v_sys = 0.0               # 系统速度 (km/s)，若为0则自动估计中值

# =========================================
# === 2. 读取 MAPS FITS 文件 ===============
# =========================================
print(f"Opening MAPS file: {maps_file}")
hdu = fits.open(maps_file)
hdu.info()

# 查找 Hα 所在通道索引
hdr = hdu['EMLINE_GVEL'].header
ha_idx = None
for i in range(hdr['NAXIS3']):
    line_name = hdr.get(f'C{ i + 1 }', '')
    if 'HA' in line_name.upper():
        ha_idx = i
        print(f"Found Hα channel at index {ha_idx}: {line_name}")
        break
if ha_idx is None:
    raise ValueError("未找到 Hα 通道，请检查 MAPS header 中的 C* 关键字")

# 提取 Hα 速度 (km/s)
v_map = hdu['EMLINE_GVEL'].data[ha_idx, :, :]
v_mask = hdu['EMLINE_GVEL_MASK'].data[ha_idx, :, :]
bad = (v_mask > 0)
v_map[bad] = np.nan

# =========================================
# === 3. 系统速度与投影修正 ================
# =========================================
# 若未给系统速度则用中值
if v_sys == 0.0:
    v_sys = np.nanmedian(v_map)
    print(f"自动估计系统速度 v_sys = {v_sys:.2f} km/s")

# 中心坐标（可改为 DAP Header 中的 REFPIX 或通过亮度峰估算）
ny, nx = v_map.shape
x0, y0 = nx / 2, ny / 2

# 倾角与角度弧度
inc = np.radians(galaxy_incl_deg)
pa = np.radians(pa_deg)

# 坐标网格
y, x = np.indices(v_map.shape)
x_rot = (x - x0) * np.cos(pa) + (y - y0) * np.sin(pa)
y_rot = -(x - x0) * np.sin(pa) + (y - y0) * np.cos(pa)

# 投影到星系盘面半径（单位: 像素）
r_ell = np.sqrt(x_rot**2 + (y_rot / np.cos(inc))**2)

# 投影角 θ（沿主轴）
theta = np.arctan2(y_rot / np.cos(inc), x_rot)

# 去投影旋转速度
v_rot = (v_map - v_sys) / (np.sin(inc) * np.cos(theta))
v_rot[np.abs(np.cos(theta)) < 0.2] = np.nan  # 去除近垂直方向像素（不可靠）

# =========================================
# === 4. 构建旋转曲线 =====================
# =========================================
r_bins = np.linspace(0, np.nanmax(r_ell), 20)
r_mids = 0.5 * (r_bins[1:] + r_bins[:-1])
v_circ = np.zeros_like(r_mids)

for i in range(len(r_bins) - 1):
    m = (r_ell >= r_bins[i]) & (r_ell < r_bins[i + 1])
    v_circ[i] = np.nanmedian(v_rot[m])

# 平滑旋转曲线
v_circ_smooth = gaussian_filter1d(v_circ, sigma=1)

# =========================================
# === 5. 计算质量密度 =====================
# =========================================
# 假设像素比例尺 = 0.5 arcsec/pix, 1 arcsec = 0.48 kpc (MaNGA 平均值)
pixscale_kpc = 0.5 * 0.48
r_kpc = r_mids * pixscale_kpc

# G 常数: m^3/kg/s^2 → 转为 kpc, km/s, M_sun
G_kpc = G.to('kpc km2 / (s2 Msun)').value

# 包裹质量 (M(<r))
M_enc = v_circ_smooth**2 * r_kpc / G_kpc  # 单位: M_sun

# 质量密度 ρ(r)
# 数值求导
dvdr = np.gradient(v_circ_smooth, r_kpc)
rho = (1 / (4 * np.pi * G_kpc)) * (2 * v_circ_smooth * dvdr / r_kpc + (v_circ_smooth / r_kpc)**2)
rho = np.abs(rho)  # 去除负值波动

# =========================================
# === 6. NFW 拟合 =========================
# =========================================
def rho_nfw(r, rho_s, r_s):
    return rho_s / ((r / r_s) * (1 + r / r_s)**2)

# 初始猜测
p0 = [1e7, 10]  # rho_s (Msun/kpc^3), r_s (kpc)
valid = (~np.isnan(r_kpc)) & (~np.isnan(rho))
popt, pcov = curve_fit(rho_nfw, r_kpc[valid], rho[valid], p0=p0, maxfev=10000)

rho_fit = rho_nfw(r_kpc, *popt)
print(f"NFW 拟合参数: ρ_s={popt[0]:.2e} Msun/kpc^3, r_s={popt[1]:.2f} kpc")

# =========================================
# === 7. 绘图 =============================
# =========================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# (a) 速度场
im = axs[0].imshow(v_map, origin='lower', cmap='RdBu_r')
axs[0].set_title(f"{plateifu} Hα Velocity (km/s)")
plt.colorbar(im, ax=axs[0], fraction=0.046)

# (b) 旋转曲线
axs[1].plot(r_kpc, v_circ, 'o', label='Raw')
axs[1].plot(r_kpc, v_circ_smooth, '-', label='Smoothed')
axs[1].set_xlabel("R (kpc)")
axs[1].set_ylabel("v_c (km/s)")
axs[1].legend()
axs[1].set_title("Rotation Curve")

# (c) 密度与NFW拟合
axs[2].loglog(r_kpc, rho, 'o', label='Observed ρ(r)')
axs[2].loglog(r_kpc, rho_fit, '-', label='NFW fit')
axs[2].set_xlabel("r (kpc)")
axs[2].set_ylabel("ρ (M☉/kpc³)")
axs[2].legend()
axs[2].set_title("Density Profile vs NFW")

plt.tight_layout()
plt.show()

print("处理完成。")
