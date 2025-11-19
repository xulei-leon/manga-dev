
from math import log
from pathlib import Path
from tkinter import N
from turtle import shape

import numpy as np
from astropy.utils.exceptions import AstropyWarning
from scipy.optimize import curve_fit

import logging
import logging.handlers

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil

# constants definitions
SNR_THRESHOLD = 10.0
PHI_LIMIT_DEG = 60.0
BA_0 = 0.2  # intrinsic axis ratio for inclination calculation
DEFAULT_BETA = 0.5   # sigma_phi^2 / sigma_R^2
DEFAULT_GAMMA = 0.6  # sigma_z / sigma_R

class VelRot:
    drpall_util = None
    firefly_util = None
    maps_util = None
    plot_util = None
    log_level = logging.INFO
    
    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil, log_level=logging.INFO) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util
        self.log_level = log_level

        self._log_init()


    def _log_init(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        self.logger = logger

    def _debug(self, *args):
        if self.log_level <= logging.DEBUG:
            self.logger.debug(" ".join(str(arg) for arg in args))

    def _info(self, *args):
        if self.log_level <= logging.INFO:
            self.logger.info(" ".join(str(arg) for arg in args))

    def _err(self, *args):
        self.logger.error(" ".join(str(arg) for arg in args))

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
    def _calc_phi_delta(phi, phi_0=0.0):
        """Normalize azimuth angles to [-pi/2, +pi/2] relative to the major axis."""
        return ((phi - phi_0) + np.pi/2) % np.pi - np.pi/2
    
    # PA: The position angle of the major axis of the galaxy, measured from north to east.
    # b/a: The axis ratio (b/a) of the galaxy
    def _calc_pa_inc(self) -> float:
        phi, ba_1 = self.maps_util.get_pa_inc()
        ba = 1 - ba_1
        self._debug(f"Position Angle PA from MAPS header: {phi:.2f} deg,", f"Inclination b/a from MAPS header: {ba:.3f}")
        # convert PA to radians
        pa = np.radians(phi) + np.pi/2  # North is in 90 deg position
        inc = self._calc_inc(ba, ba_0=BA_0)
        self._debug(f"Calculated Inclination i: {np.degrees(inc):.2f} deg from b/a={ba:.3f}")

        return pa, inc

    # Filter the velocity map with SNR above the threshold and within ±phi_limit of the major axis.
    def _vel_map_filter(self, vel_map: np.ndarray, snr_map: np.ndarray, azimuth_map: np.ndarray, snr_threshold: float = 10.0, phi_limit_deg: float = 60.0) -> np.ndarray:
        phi_delta = self._calc_phi_delta(azimuth_map)
        phi_limit_rad = np.radians(phi_limit_deg)
        valid_mask = ((snr_map >= snr_threshold) & (np.abs(phi_delta) <= phi_limit_rad) & np.isfinite(vel_map))

        vel_map_filtered = np.full_like(vel_map, np.nan, dtype=float)
        vel_map_filtered[valid_mask] = vel_map[valid_mask]
        return vel_map_filtered
    
    def _get_vel_los(self, type: str='gas') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset_x, offset_y = self.maps_util.get_sky_offsets()
        self._debug(f"Sky offsets shape: {offset_x.shape}, X offset: [{np.nanmin(offset_x):.3f}, {np.nanmax(offset_x):.3f}] arcsec")

        # R: radial distance map
        radius_map, radius_h_kpc_map, azimuth_map = self.maps_util.get_radius_map()
        self._debug(f"r_map: [{np.nanmin(radius_map):.3f}, {np.nanmax(radius_map):.3f}] spaxel,", f"shape: {radius_map.shape}")
        self._debug(f"r_h_kpc_map: [{np.nanmin(radius_h_kpc_map):.3f}, {np.nanmax(radius_h_kpc_map):.3f}] kpc,", f"shape: {radius_h_kpc_map.shape}")
        self._debug(f"azimuth_map: [{np.nanmin(azimuth_map):.3f}, {np.nanmax(azimuth_map):.3f}] deg,", f"shape: {azimuth_map.shape}")

        # SNR: signal-to-noise ratio map
        snr_map = self.maps_util.get_snr_map()
        self._debug(f"SNR map shape: {snr_map.shape}, SNR range: [{np.nanmin(snr_map):.3f}, {np.nanmax(snr_map):.3f}]")

        ra_map, dec_map = self.maps_util.get_skycoo_map()
        self._debug(f"RA map: [{np.nanmin(ra_map):.6f}, {np.nanmax(ra_map):.6f}] deg,", f"Dec map: [{np.nanmin(dec_map):.6f}, {np.nanmax(dec_map):.6f}] deg")

        ## Get the gas velocity map (H-alpha)
        v_obs_gas_map, _gv_unit, _gv_ivar = self.maps_util.get_eml_vel_map()
        self._debug(f"Gas velocity map shape: {v_obs_gas_map.shape}, Unit: {_gv_unit}, Velocity: [{np.nanmin(v_obs_gas_map):.3f}, {np.nanmax(v_obs_gas_map):.3f}] {_gv_unit}")
        eml_uindx = self.maps_util.get_emli_uindx()
        self._debug(f"Gas Unique indices shape: {eml_uindx.shape}")

        ## Get the stellar velocity map
        v_obs_stellar_map, _sv_unit, _ = self.maps_util.get_stellar_vel_map()
        self._debug(f"Stellar velocity map shape: {v_obs_stellar_map.shape}, Unit: {_sv_unit}, Velocity: [{np.nanmin(v_obs_stellar_map):.3f}, {np.nanmax(v_obs_stellar_map):.3f}] {_sv_unit}")
        stellar_uindx = self.maps_util.get_stellar_uindx()
        self._debug(f"Stellar Unique indices shape: {stellar_uindx.shape}")

        # Velocity correction
        if type == 'gas':
            v_obs_map = v_obs_gas_map
            v_unit = _gv_unit
            v_uindx = eml_uindx
        else:
            v_obs_map = v_obs_stellar_map
            v_unit = _sv_unit
            v_uindx = stellar_uindx
        
        azimuth_rad_map = np.radians(azimuth_map)

        filtered_vel_map = self._vel_map_filter(v_obs_map, snr_map, azimuth_rad_map, snr_threshold=SNR_THRESHOLD, phi_limit_deg=PHI_LIMIT_DEG)
        self._debug(f"Filtered Velocity map shape: {filtered_vel_map.shape}, Unit: {v_unit}, Range: [{np.nanmin(filtered_vel_map):.3f}, {np.nanmax(filtered_vel_map):.3f}]")
        self._debug(f"Velocity data before filtering: {np.sum(np.isfinite(v_obs_map))}, after filtering: {np.sum(np.isfinite(filtered_vel_map))}")

        r_obs_map = radius_h_kpc_map
        v_obs_map = filtered_vel_map
        phi_obs_map = azimuth_rad_map

        return r_obs_map, v_obs_map, phi_obs_map
    
    def _get_radius(self) -> np.ndarray:
        _, radius_h_kpc_map, _ = self.maps_util.get_radius_map()
        return radius_h_kpc_map


    ################################################################################
    # rotation curve fitting procedures
    ################################################################################

    # Formula: V(r) = Vc * tanh(r / Rt) + s_out * r
    # sout is the slope of the RC at large radii r >> Rt
    # Rt is the turnover radius where the rotation curve transitions from rising to flat.
    def _vel_rot_profile(self, r, Vc, Rt, s_out) -> np.ndarray:
        return Vc * np.tanh(r / Rt) + s_out * r

    # Inclination Angle: The angle between the galaxy's disk and the plane of the sky.
    # Azimuthal Angle: The angle of the dataset within the galaxy's disk relative to the kinematic major axis (i.e., the line where the line-of-sight velocity is zero).
    # Formula: V_obs = V_rot * (sin(i) * cos(phi_delta))
    def _calc_vel_los_from_rot(self, vel_rot: np.ndarray, inc: float, phi: np.ndarray) -> np.ndarray:
        phi_delta = self._calc_phi_delta(phi, phi_0=0.0)
        correction = np.sin(inc) * np.cos(phi_delta)
        return vel_rot * correction

    #  V_obs = (Vc * tanh(r / Rt) + s_out * r) * (sin(i) * cos(phi_delta))
    def _vel_los_fit_model(self, r, Vc, Rt, s_out, inc, phi) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        vel_rot = self._vel_rot_profile(r, Vc, Rt, s_out)
        vel_los = self._calc_vel_los_from_rot(vel_rot, inc, phi)
        return vel_los
    
    # used the minimum χ 2 method for fitting
    def _rot_curve_fit(self, radius_map: np.ndarray, vel_los_map: np.ndarray, phi_map: np.ndarray, radius_fit: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Fit the rotation curve using the experience curve function."""
        
        # 1. 数据清洗
        # 确保移除 NaN 和 Inf
        valid_mask = np.isfinite(vel_los_map) & np.isfinite(radius_map) & np.isfinite(phi_map)
        
        # 提取有效数据
        # 注意：物理半径 r 必须为正数用于计算 V_rot (tanh函数定义域)
        radius_valid_abs = np.abs(radius_map[valid_mask]) 
        phi_valid = phi_map[valid_mask]
        vel_valid = np.abs(vel_los_map[valid_mask])

        # 2. 计算固定的倾角 (Inc)
        _, inc_fixed = self._calc_pa_inc()
        
        # 保护性检查：如果倾角过小（接近面朝上），投影修正会产生巨大的数字
        sin_inc = np.abs(np.sin(inc_fixed))
        if sin_inc < 0.1: # 约 5.7 度
            self._debug(f"Warning: Inclination is very small ({np.degrees(inc_fixed):.1f} deg). V_rot calculation might be unstable.")
            sin_inc = 0.1 # 避免除零或过度放大

        # 3. 初始猜测 (Initial Guess)
        # 使用 99分位数代替 max，防止极端噪点影响初始值
        v_max_guess = np.nanpercentile(np.abs(vel_valid), 99)
        Vc0 = v_max_guess / sin_inc
        
        # Rt0: 猜测转折半径。通常在有效半径的中位数附近
        Rt0 = np.nanmedian(radius_valid_abs) / 2.0
        if Rt0 == 0: Rt0 = 1.0
        
        # s_out0: 外围斜率。初始化为 0 (平坦) 是最安全的
        s_out0 = 0.0 

        initial_guess = [Vc0, Rt0, s_out0]

        # 4. 构建多变量输入数据 (X_data)
        # 将 半径(abs) 和 角度(phi) 堆叠成 (2, N) 的数组
        # 这样 curve_fit 会同时传入 r 和 phi，确保几何投影的符号计算正确
        X_data = np.vstack((radius_valid_abs, phi_valid))

        # 定义拟合包装函数：解包 X 得到 r 和 phi
        def fit_wrapper(X, Vc, Rt, s_out):
            r = X[0]   # 绝对值半径
            phi = X[1] # 对应的方位角
            # 调用您现有的物理模型
            return self._vel_los_fit_model(r, Vc, Rt, s_out, inc_fixed, phi)
        
        # 5. 设置边界 (Bounds)
        # Vc: [0, 3000] - 速度必须为正
        # Rt: [0.01, 1000] - 转折半径必须为正且非零
        # s_out: [-50, 50] - 斜率可正可负，限制范围防止过拟合
        bounds = ([0.0, 1e-3, -50.0], [3000.0, 1000.0, 50.0]) 

        try:
            # maxfev 增加到 10000 以确保在数据点很多时能收敛
            popt, pcov = curve_fit(fit_wrapper, X_data, vel_valid, p0=initial_guess, bounds=bounds, maxfev=10000)
            Vc_fit, Rt_fit, s_out_fit = popt
            
            # 计算参数误差 (标准差)
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            self._debug(f"Fitting failed: {e}")
            # 拟合失败时的默认回退值
            Vc_fit, Rt_fit, s_out_fit = 0.0, 1.0, 0.0
            perr = [0, 0, 0]

        self._debug("Fitted parameters:")
        self._debug(f"  Vc:    {Vc_fit:.3f} +/- {perr[0]:.3f} km/s")
        self._debug(f"  Rt:    {Rt_fit:.3f} +/- {perr[1]:.3f} kpc/h")
        self._debug(f"  s_out: {s_out_fit:.3f} +/- {perr[2]:.3f} (km/s)/kpc")

        # 6. 生成拟合结果曲线
        if radius_fit is None:
            radius_fit = radius_map # 使用原始输入
        
        # 计算纯物理旋转曲线 (用于展示 V_rot，均为正值)
        # 注意：这里传入 abs(radius_fit) 确保输出仅展示物理速度大小
        radius_fit_abs = np.abs(radius_fit)
        vel_rot_fitted = self._vel_rot_profile(radius_fit_abs, Vc_fit, Rt_fit, s_out_fit)
        
        # 如果需要返回拟合的观测速度(带符号)，则需要传入对应的 phi_fit
        # 但通常 _rot_curve_fit 返回的是去投影后的 V_rot 曲线
        
        return radius_fit, vel_rot_fitted

    ################################################################################
    # public methods
    ################################################################################

    # Line-of-Sight Velocity Maps
    def get_vel_los(self, type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_map, vel_los_map, phi_map = self._get_vel_los(type=type)
        return radius_map, vel_los_map, phi_map

    def fit_vel_rot(self, radius_map, vel_los_map, phi_map, radius_fit=None):
        radius_fitted, vel_rot_fitted =  self._rot_curve_fit(radius_map, vel_los_map, phi_map, radius_fit=radius_fit)
        return radius_fitted, vel_rot_fitted

    def get_inc_rad(self):
        _, inc_rad = self._calc_pa_inc()
        return inc_rad
    
    def get_radius_fit(self, count: int=100) -> np.ndarray:
        radius_map = self._get_radius()
        radius_max = np.nanmax(radius_map)

        radius_fit = np.linspace(0.0, radius_max, num=count)
        return radius_fit


######################################################
# main function for test
######################################################
def main():
    PLATE_IFU = "8723-12705"

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
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

    vel_rot = VelRot(drpall_util, firefly_util, maps_util, log_level=logging.DEBUG)
    r_obs_map, V_obs_map, phi_map = vel_rot.get_vel_los("gas")
    r_rot_fitted, V_rot_fitted = vel_rot.fit_vel_rot(r_obs_map, V_obs_map, phi_map)

    print("#######################################################")
    print("# calculate results")
    print("#######################################################")
    print(f"Obs Velocity shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):.3f}, {np.nanmax(V_obs_map):.3f}]")
    print(f"Fitted Rot Velocity shape: {V_rot_fitted.shape}, range: [{np.nanmin(V_rot_fitted):.3f}, {np.nanmax(V_rot_fitted):.3f}]")

    plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs")
    plot_util.plot_rv_curve(r_obs_map, V_obs_map, title="Obs Fitted", r_rot2_map=r_rot_fitted, v_rot2_map=V_rot_fitted, title2="Rot Fitted")
    return


# main entry
if __name__ == "__main__":
    main()

