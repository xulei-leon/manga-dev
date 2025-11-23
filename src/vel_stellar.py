from ast import mod
from bdb import effective
from math import log
from operator import le
from pathlib import Path
from tkinter import N

from deprecated import deprecated
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
import scipy.special as special
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


from matplotlib import colors

# my imports
from util.fits_util import FitsUtil
from util.drpall_util import DrpallUtil
from util.firefly_util import FireflyUtil
from util.maps_util import MapsUtil
from util.plot_util import PlotUtil

RADIUS_MIN_KPC = 0.1  # kpc
G = const.G.to('kpc km2 / (Msun s2)').value  # gravitational constant in kpc km^2 / (M_sun s^2)

class Stellar:
    drpall_util = None
    firefly_util = None
    maps_util = None

    def __init__(self, drpall_util: DrpallUtil, firefly_util: FireflyUtil, maps_util: MapsUtil) -> None:
        self.drpall_util = drpall_util
        self.firefly_util = firefly_util
        self.maps_util = maps_util

    @staticmethod
    def calc_r_ratio_to_h_kpc(r_arcsec: np.ndarray, r_h_kpc: np.ndarray) -> float:
        r_arcsec = np.asarray(r_arcsec)
        r_h_kpc = np.asarray(r_h_kpc)
        if r_arcsec.shape != r_h_kpc.shape:
            raise ValueError("r_arcsec and r_h_kpc must have the same shape")
        r_h_kpc = np.where(r_h_kpc == 0, np.nan, r_h_kpc)
        r_arcsec = np.where((r_arcsec < 0) | (np.isnan(r_arcsec)), np.nan, r_arcsec)
        r_h_kpc = np.where((r_h_kpc < 0) | (np.isnan(r_h_kpc)), np.nan, r_h_kpc)
        ratio_r = r_h_kpc / r_arcsec
        ratio = np.median(ratio_r[~np.isnan(ratio_r)])
        return ratio

    def _calc_radius_to_h_kpc(self, PLATE_IFU, radius_eff_map):
        _r_arcsec_map, _r_h_kpc_map, _ = self.maps_util.get_radius_map()
        print(f"Radius (MAPS) shape: {_r_arcsec_map.shape}, Unit: arcsec, range [{np.nanmin(_r_arcsec_map[_r_arcsec_map>=0]):.3f}, {np.nanmax(_r_arcsec_map):.3f}]")
        print(f"Radius (MAPS) shape: {_r_h_kpc_map.shape}, Unit: kpc/h, range [{np.nanmin(_r_h_kpc_map[_r_h_kpc_map>=0]):.3f}, {np.nanmax(_r_h_kpc_map):.3f}]")

        effective_radius = self.drpall_util.get_effective_radius(PLATE_IFU)
        print(f"Effective Radius (DRPALL): {effective_radius:.3f} arcsec")

        radius_arcsec_map = radius_eff_map * effective_radius
        print(f"Radius (arcsec) shape: {radius_arcsec_map.shape}, Unit: arcsec, range [{np.nanmin(radius_arcsec_map[radius_arcsec_map>=0]):.3f}, {np.nanmax(radius_arcsec_map):.3f}]")

        _r_err_percent = np.abs(np.nanmax(radius_arcsec_map) - np.nanmax(_r_arcsec_map)) / np.nanmax(_r_arcsec_map)
        print(f"  Verification with MAPS radius:")
        if np.nanmax(_r_err_percent) < 0.01:
            print(f"  Calculated radius matches MAPS radius within {np.nanmax(_r_err_percent):.1%}")
        else:
            print("  WARNING: Calculated radius does not match MAPS radius within 1%")

        ratio_r = self.calc_r_ratio_to_h_kpc(_r_arcsec_map, _r_h_kpc_map)
        print(f"Radius ratio (arcsec to kpc/h): {ratio_r:.3f}")

        radius_h_kpc_map = radius_arcsec_map * ratio_r
        print(f"Radius (kpc/h) shape: {radius_h_kpc_map.shape}, Unit: kpc/h, range [min: {np.nanmin(radius_h_kpc_map[radius_h_kpc_map>=0]):.3f}, max: {np.nanmax(radius_h_kpc_map):.3f}]")
        _r_err_percent = np.abs(np.nanmax(radius_h_kpc_map) - np.nanmax(_r_h_kpc_map)) / np.nanmax(_r_h_kpc_map)
        print(f"  Verification with MAPS radius:")
        if np.nanmax(_r_err_percent) < 0.01:
            print(f"  Calculated radius matches MAPS radius within {np.nanmax(_r_err_percent):.1%}")
        else:
            print("  WARNING: Calculated radius does not match MAPS radius within 1%")
        return radius_h_kpc_map

    ################################################################################
    # Stellar Mass M(r)
    # Unused for now
    ################################################################################

    # The RC of the baryonic disk model (V_baryon) was derived by a method in Noordermeer (2008).

    # Hernquist bulge + Freeman thin exponential disk
    # V_bulge^2(r) = (G * MB * r) / (r + a)^2
    def _vel_sq_bulge_hernquist(self, r: np.ndarray, MB: float, a: float) -> np.ndarray:
        r = np.where(r == 0, 1e-6, r)  # avoid division by zero
        v_sq = G * MB * r / (r + a)**2
        return v_sq
    
    # V_disk^2(r) = (2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    def _vel_sq_disk_freeman(self, r: np.ndarray, M_d: float, Rd: float) -> np.ndarray:
        r = np.where(r == 0, 1e-6, r)  # avoid division by zero
        y = r / (2.0 * Rd)
        I_0 = special.i0(y)
        I_1 = special.i1(y)
        K_0 = special.k0(y)
        K_1 = special.k1(y)
        v_sq = (2.0 * G * M_d / Rd) * (np.square(y)) * (I_0 * K_0 - I_1 * K_1)
        return v_sq
    
    # formula: V_baryon^2 =  (G * MB * r) / (r + a)^2 + (2 * G * MD / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    def _stellar_vel_sq_mass_parts_profile(self, r: np.ndarray, MB: float, a: float, MD: float, Rd: float) -> np.ndarray:
        v_bulge_sq = self._vel_sq_bulge_hernquist(r, MB, a)
        v_disk_sq = self._vel_sq_disk_freeman(r, MD, Rd)
        v_baryon_sq = v_bulge_sq + v_disk_sq
        return v_baryon_sq
    

    def _stellar_vel_sq_mass_profile(self, r: np.ndarray, M_star: float, b_d_ratio: float, a: float, Rd: float) -> np.ndarray:
        MB = b_d_ratio * M_star
        MD = (1 - b_d_ratio) * M_star
        v_baryon_sq = self._stellar_vel_sq_mass_parts_profile(r, MB, a, MD, Rd)
        return v_baryon_sq
    
    # M_star: total mass of star
    # Re: Half-mass radius
    def _stellar_vel_sq_disk_profile(self, r: np.ndarray, M_star: float, Re: float):
        Rd = Re / 1.678
        return self._vel_sq_disk_freeman(r, M_star, Rd)


    @staticmethod
    def _calc_mass_of_radius(mass_cell: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mass_cell = np.asarray(mass_cell)
        radius = np.asarray(radius)

        if mass_cell.shape != radius.shape:
            raise ValueError("mass_cell and radius must have the same shape")

        valid_mask = (~np.isnan(mass_cell)) & (~np.isnan(radius)) & (mass_cell >= 0) & (radius >= 0)

        if not np.any(valid_mask):
            return np.array([], dtype=float), np.array([], dtype=float)

        filtered_mass = mass_cell[valid_mask]
        filtered_radius = radius[valid_mask]

        sorted_idx = np.argsort(filtered_radius)

        sorted_radius = filtered_radius[sorted_idx]
        sorted_masses = filtered_mass[sorted_idx]

        mass_r = np.cumsum(sorted_masses)
        r_bins = sorted_radius

        return mass_r, r_bins

    def _get_stellar_mass(self, PLATE_IFU: str) -> tuple[np.ndarray, np.ndarray]:
        mass_stellar_cell, mass_stellar_cell_err = self.firefly_util.get_stellar_mass_cell(PLATE_IFU)
        print(f"Stellar Mass shape: {mass_stellar_cell.shape}, Unit: M solar, total: {np.nansum(mass_stellar_cell):,.1f} M solar")

        # This mass use h = 1
        total_stellar_mass_1, total_stellar_mass_2 = self.drpall_util.get_stellar_mass(PLATE_IFU)
        print("Verification with DRPALL stellar mass:")
        _mass_err2_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_2) / total_stellar_mass_2
        _mass_err1_percent = (np.nansum(mass_stellar_cell) - total_stellar_mass_1) / total_stellar_mass_1
        print(f"  Stellar Mass (DRPALL): (Sersic) {total_stellar_mass_1:,} M solar, (Elpetro) {total_stellar_mass_2:,} M solar")
        if abs(_mass_err2_percent) < 0.03:
            print(f"  FIREFLY total stellar mass matches DRPALL Elpetro stellar mass within {abs(_mass_err2_percent):.1%}")
        elif abs(_mass_err1_percent) < 0.03:
            print(f"  FIREFLY total stellar mass matches DRPALL Sersic stellar mass within {abs(_mass_err1_percent):.1%}")
        else:
            print("  WARNING: FIREFLY total stellar mass does not match DRPALL stellar masses within 3%")

        _radius_eff, azimuth = self.firefly_util.get_radius_eff(PLATE_IFU)
        print(f"Radius Eff shape: {_radius_eff.shape}, Unit: effective radius, range [{np.nanmin(_radius_eff[azimuth>=0]):.3f}, {np.nanmax(_radius_eff):.3f}]")
        print(f"Azimuth shape: {azimuth.shape}, Unit: degrees, range [{np.nanmin(azimuth[azimuth>=0]):.3f}, {np.nanmax(azimuth):.3f}]")

        mass_map, radius_eff_map = self._calc_mass_of_radius(mass_stellar_cell, _radius_eff)
        print(f"Cumulative Mass shape: {mass_map.shape}, Unit: M solar, range [{np.nanmin(mass_map):.3f}, {np.nanmax(mass_map):,.1f}]")
        print(f"Radius bins shape: {radius_eff_map.shape}, Unit: effective radius, range [{np.nanmin(radius_eff_map):.3f}, {np.nanmax(radius_eff_map):.3f}]")

        radius_h_kpc_map = self._calc_radius_to_h_kpc(PLATE_IFU, radius_eff_map)

        return radius_h_kpc_map, mass_map

    # Formula: M(r) = MB * r^2 / (r + a)^2 + MD * (1 - (1 + r / rd) * exp(-r / rd))
    def _stellar_mass_model(self, r: np.ndarray, log_MB: float, a: float, log_MD: float, rd: float) -> np.ndarray:
        MB = 10 ** log_MB
        MD = 10 ** log_MD
        bulge_mass = MB * np.square(r) / np.square(r + a)
        disk_mass = MD * (1.0 - (1.0 + r / rd) * np.exp(-r / rd))
        total_mass = bulge_mass + disk_mass
        return total_mass

    # used the minimum χ 2 method for fitting
    def _stellar_mass_fit(self, radius: np.ndarray, mass: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> np.ndarray:
        radius_filter = radius[radius>r_min]
        mass_filter = mass[radius>r_min]

        radius_max = np.nanmax(radius)
        radius_mean = np.nanmean(radius)
        mass_max = np.nanmax(mass)
        print(f"Fitting Stellar Mass M(r): radius max {radius_max:.3f} kpc, mass  {mass_max:.3e} M solar")

        p0 = [np.log10(mass_max * 0.3) - 1, 0.5, np.log10(mass_max * 0.7), radius_mean]  # Initial guess for log_MB, a, log_MD, rd
        lb = [-np.inf, 0.5, -np.inf, 1e-4]  # Lower bound
        ub = [np.log10(mass_max * 0.5), radius_max, np.log10(mass_max * 0.9), radius_max]  # Upper bound

        popt, pcov = curve_fit(self._stellar_mass_model, radius_filter, mass_filter, p0=p0, bounds=(lb, ub), method='trf')
        log_mb_fit, a_fit, log_md_fit, rd_fit = popt
        mb_fit = 10**log_mb_fit
        md_fit = 10**log_md_fit
        print(f"Fitted parameters: MB={mb_fit:.3e}, a={a_fit:.3f}, MD={md_fit:.3e}, rd={rd_fit:.3f}")
        print(f"  Parameter uncertainties: {np.sqrt(np.diag(pcov))}")

        mass_fit = self._stellar_mass_model(radius_fitted, *popt)
        mass_fit_max = np.nanmax(mass_fit)
        print(f"Fitted mass max: {mass_fit_max:.3e} M solar")
        
        return mass_fit, mb_fit, a_fit, md_fit, rd_fit
    
    # Model function (Normalization version)
    # M(r) = MB * r^2 / (r + a)^2 + MD * (1 - (1 + r / rd) * exp(-r / rd))
    def _stellar_mass_model_norm(self, MB, a, MD, Rd, r):
        r_safe = np.maximum(r, 1e-6)
        bulge_mass = MB * np.square(r_safe) / np.square(r_safe + a)
        disk_mass = MD * (1.0 - (1.0 + r_safe / Rd) * np.exp(-r_safe / Rd))
        return bulge_mass + disk_mass
        
    def _stellar_mass_fit_minimize(self, radius: np.ndarray, mass: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> tuple:
        # --- 1. 数据预处理 ---
        mask = (radius > r_min) & np.isfinite(mass) & np.isfinite(radius)
        radius_filter = radius[mask]
        mass_filter = mass[mask]

        radius_max = np.nanmax(radius)
        radius_mean = np.nanmean(radius)
        mass_max = np.nanmax(mass)

        print(f"Fitting Stellar Mass: R_max {radius_max:.3f} kpc, M_max {mass_max:.3e} M_sun")

        # --- 2. 归一化 (Normalization) ---
        # 将质量缩放到 0~1 之间，大幅提高拟合稳定性
        mass_norm = mass_filter / mass_max
        
        def _residuals(params):
            # 计算残差
            model_norm = self._stellar_mass_model_norm(*params, radius_filter)
            return np.sum((mass_norm - model_norm) ** 2)

        # --- 3. 设定 Bounds 和 Initial Guess ---
        
        # 初始猜测：
        # 关键调整：假设观测边界处的质量只占渐进总质量的 60%，所以总倍数猜测为 1.0/0.6 ≈ 1.6
        # MB_norm ~ 0.4, MD_norm ~ 1.2
        p0 = [0.4, 1.0, 1.2, radius_mean]

        bounds = [
            (0, 10.0),           # MB_norm: 允许核球是观测最大值的10倍 (归一化后就是10)
            (0.01, radius_max),  # a: 核球尺度
            (0, 20.0),           # MD_norm: 允许盘是观测最大值的20倍 (关键！)
            (0.01, radius_max * 5.0) # rd: 盘尺度
        ]

        # --- 4. 执行拟合 ---
        # 移除 constraints，完全依靠 data 驱动
        result = minimize(_residuals, p0, bounds=bounds, method='L-BFGS-B', tol=1e-6)

        mb_norm_fit, a_fit, md_norm_fit, rd_fit = result.x

        # --- 5. 还原物理量 ---
        # 将归一化的参数乘回 mass_max
        mb_fit = mb_norm_fit * mass_max
        md_fit = md_norm_fit * mass_max
        
        print(f"Fitted params: MB={mb_fit:.2e}, a={a_fit:.2f}, MD={md_fit:.2e}, rd={rd_fit:.2f}")
        print(f"Asymptotic Total: {mb_fit + md_fit:.2e} (Ratio to Obs Max: {(mb_fit+md_fit)/mass_max:.2f}x)")

        # --- 6. 生成最终曲线 (排序后) ---
        sort_idx = np.argsort(radius_fitted)
        radius_fitted_sorted = radius_fitted[sort_idx]
        
        # 使用还原后的真实物理量计算最终曲线
        def _mass_model_real(MB, a, MD, Rd, r):
            r_safe = np.maximum(r, 1e-6)
            bulge = MB * np.square(r_safe) / np.square(r_safe + a)
            disk = MD * (1.0 - (1.0 + r_safe / Rd) * np.exp(-r_safe / Rd))
            return bulge + disk

        mass_fit = _mass_model_real(mb_fit, a_fit, md_fit, rd_fit, radius_fitted_sorted)
        
        fit_max_at_obs = np.max(mass_fit[radius_fitted_sorted <= radius_max])
        print(f"Model mass at R_max: {fit_max_at_obs:.2e} (Target: {mass_max:.2e})")

        return mass_fit, mb_fit, a_fit, md_fit, rd_fit

    ################################################################################
    # Mass Density Method
    ################################################################################

    # Exponential Disk rotation velocity formula
    # V^2(R) = 4 * pi * G * Sigma_0 * Rd * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    # where y = R / (2 * Rd)
    def _stellar_vel_sq_density_profile(self, R: np.ndarray, Sigma_0: float, Rd: float) -> np.ndarray:
        # FIXME: handle R = 0 case
        R = np.where(R == 0, 1e-6, R)  # avoid division by zero

        y = R / (2.0 * Rd)
        I_0 = special.i0(y)
        I_1 = special.i1(y)
        K_0 = special.k0(y)
        K_1 = special.k1(y)

        V2 = 4.0 * np.pi * G * Sigma_0 * Rd * (np.square(y)) * (I_0 * K_0 - I_1 * K_1)
        return V2


    # Exponential Disk Model fitting function
    # Sigma(R) = Sigma_0 * exp(-R / Rd)
    def _stellar_density_profile(self, R: np.ndarray, Sigma_0: float, Rd: float) -> np.ndarray:
        return Sigma_0 * np.exp(-R / Rd)

    def _stellar_density_fit_profile(self, R: np.ndarray, log_Sigma_0: float, Rd: float) -> np.ndarray:
        Sigma_0 = 10**log_Sigma_0
        return Sigma_0 * np.exp(-R / Rd)
    
    # Central Surface Mass Density Fitting
    def _stellar_central_density_fit(self, radius: np.ndarray, density: np.ndarray, r_min: float, radius_fitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius_filter = radius[radius>r_min]
        density_filter = density[radius>r_min]

        p0 = [7, 3.0]  # Initial guess for Sigma_0, Rd
        lb = [6, 0.1]  # Lower bound
        ub = [10, 10.0]  # Upper bound
        popt, pcov = curve_fit(self._stellar_density_fit_profile, radius_filter, density_filter, p0=p0, bounds=(lb, ub), method='trf')

        log_sigma_0_fitted, r_d_fitted = popt
        sigma_0_fitted = 10**log_sigma_0_fitted
        print(f"Fitted parameters: Sigma_0={sigma_0_fitted:.3e}, Rd={r_d_fitted:.3f}")
        return sigma_0_fitted, r_d_fitted

    def _calc_stellar_central_density(self, PLATE_IFU: str, radius_fitted: np.ndarray) -> tuple[float, float]:
        _radius_eff, _ = self.firefly_util.get_radius_eff(PLATE_IFU)
        _radius_h_kpc = self._calc_radius_to_h_kpc(PLATE_IFU, _radius_eff)

        _density, _ = self.firefly_util.get_stellar_density_cell(PLATE_IFU)
        print(f"Stellar Mass Density shape: {_density.shape}, Unit: M solar / kpc^2, range [{np.nanmin(_density[_density>=0]):.3f}, {np.nanmax(_density):,.1f}] M solar / kpc^2")

        sigma_0_fitted, r_d_fitted = self._stellar_central_density_fit(_radius_h_kpc, _density, r_min=RADIUS_MIN_KPC, radius_fitted=radius_fitted)
        return sigma_0_fitted, r_d_fitted

    ################################################################################
    # Stellar Pressure Support Correction:
    ################################################################################

    # formula: sigma_ast^2 = sigma_stellar_obs^2 - sigma_stellar_inst^2
    def _get_stellar_sigma_ast_sq(self) -> np.ndarray:
        sigma_obs, sigma_inst = self.maps_util.get_stellar_sigma_map()
        print(f"Stellar observed sigma shape: {sigma_obs.shape}, range: [{np.nanmin(sigma_obs):.3f}, {np.nanmax(sigma_obs):.3f}] km/s")
        print(f"Stellar instrumental sigma shape: {sigma_inst.shape}, range: [{np.nanmin(sigma_inst):.3f}, {np.nanmax(sigma_inst):.3f}] km/s")
        # If observed dispersion is smaller than instrumental, do not subtract instrumental term.
        sigma_sq = np.where(sigma_obs >= sigma_inst,
                    np.square(sigma_obs) - np.square(sigma_inst),
                    np.square(sigma_obs))

        return sigma_sq
    
    
    def _deproject_sigmaR(self, sigma_ast, incl, beta=0.5, gamma=0.6):
        D = np.sqrt(0.5*(1+beta)*np.sin(incl)**2 + (gamma**2)*np.cos(incl)**2)
        D = np.where(D>0, D, np.nan)
        return sigma_ast / D

    # sigma_R ^ 2 = sigma_0 ^ 2 * exp(−2R / Rd​)
    def _sigmaR_sq_fit_profile(self, radius, log_sigma0, R_sigma):
        sigma0 = 10**log_sigma0
        return sigma0 * np.exp(-2 * radius / R_sigma)
    

    def _fit_stellar_sigmaR(self, radius, sigma_ast_sq_map, incl, radius_fit: np.ndarray) -> np.ndarray:
        valid_mask = ~np.isnan(sigma_ast_sq_map) & (radius > RADIUS_MIN_KPC)
        radius_valid = radius[valid_mask]
        sigma_ast_sq_valid = sigma_ast_sq_map[valid_mask]

        p0 = [2.0, 3.0]  # log_sigma0, R_sigma
        lb = [0.0, 0.1]  # log_sigma0, R_sigma
        ub = [3.0, 10.0]  # log_sigma0, R_sigma

        sigmaR_deprojected = self._deproject_sigmaR(np.sqrt(sigma_ast_sq_valid), incl)

        xdata = radius_valid
        ydata = np.square(sigmaR_deprojected)
        popt, pcov = curve_fit(self._sigmaR_sq_fit_profile, xdata, ydata, p0=p0, bounds=(lb, ub))

        log_sigma0_fit, R_sigma_fit = popt
        sigma0_fitted = 10**log_sigma0_fit
        print(f"Fitted parameters: sigma0={sigma0_fitted:.3f}, R_sigma={R_sigma_fit:.3f}")
        sigmaR_sq_fit = self._sigmaR_sq_fit_profile(radius_fit, log_sigma0_fit, R_sigma_fit)

        return sigmaR_sq_fit

    # Radial Jeans Equation
    # V_drift^2(R) = (R / Density) * [ d(Density * sigma_R^2) / dR + (Density * sigma_R^2 / R) * (1 - (sigma_phi^2 / sigma_R^2)) ]
    def _calc_stellar_vel_drift_sq(self, radius_fit: np.ndarray, incl: float):
        """
        使用解析求导法计算不对称漂移速度 V_drift。
        假设 Density 和 sigma_R^2 均遵循指数衰减分布。
        """
        
        # 1. 获取基础数据和拟合参数
        # ------------------------------------------------------
        # 获取密度参数: Density(R) = Sigma_0 * exp(-R / Rd)
        density_0, r_d = self._calc_stellar_central_density(self.PLATE_IFU, radius_fit)
        density_fit = self._stellar_density_profile(radius_fit, density_0, r_d)
        
        # 获取速度弥散参数: sigma_R^2(R) = sigma_0^2 * exp(-R / h_sigma)
        # 【重要】：您需要确保 _fit_stellar_sigmaR 能返回拟合出的尺度长度 h_sigma
        # 如果您的拟合模型固定了 h_sigma = Rd / 2，则直接使用 Rd / 2
        sigma_ast_sq_map = self._get_stellar_sigma_ast_sq()
        _, radius_sigma_map, _ = self.maps_util.get_radius_map()
        
        # 假设这个函数现在返回拟合数组和尺度参数 (sigma_0_sq, h_sigma)
        # 如果您现在的代码只返回数组，您需要修改该函数或在此处手动指定衰减关系
        sigma_R_sq = self._fit_stellar_sigmaR(radius_sigma_map, sigma_ast_sq_map, incl, radius_fit=radius_fit)
        
        # 【假设场景 A】：完全基于理论假设 (sigma_R^2 随 Rd/2 衰减)
        # 这是最物理、最平滑的做法，不需要从 sigma 数据中拟合尺度
        h_sigma = r_d / 2.0 
        
        # 【假设场景 B】：如果您是从数据中独立拟合了 sigma 的衰减长度
        # h_sigma = fitted_sigma_scale_length 

        # 2. 计算乘积 S
        # ------------------------------------------------------
        S = density_fit * sigma_R_sq

        # 3. 解析求导 (Analytical Derivative)
        # ------------------------------------------------------
        # 公式: dS/dR = -S * (1/Rd + 1/h_sigma)
        # 这一步完全消除了数值差分带来的锯齿噪声
        decay_factor = (1.0 / r_d) + (1.0 / h_sigma)
        dS_dR = -S * decay_factor  # 这是一个平滑的负值数组

        # 4. 计算 V_drift^2
        # ------------------------------------------------------
        # 定义各向异性比率 (beta = sigma_phi^2 / sigma_R^2)
        # 假设 beta = 0.5 (即 sigma_phi ~ 0.707 * sigma_R)
        beta_ratio = 0.5 

        # 防止除以零
        radius_safe = np.where(radius_fit == 0, 1e-9, radius_fit)

        # Jeans 方程项:
        # Term 1: 压力梯度项。注意这里使用了 负号 来抵消 dS_dR 的负值
        # V_drift_term1 = - (R / Density) * (dS/dR)
        term1_contribution = - (radius_safe / density_fit) * dS_dR
        
        # 优化: 代入 dS_dR 的解析式，这一项简化为:
        # term1 = - (R/D) * (-S * decay) = R * sigma_R^2 * (1/Rd + 1/h_sigma)
        # term1_contribution = radius_safe * sigma_R_sq * decay_factor

        # Term 2: 各向异性项
        # V_drift_term2 = (R / Density) * (S / R) * (1 - beta)
        #               = sigma_R^2 * (1 - beta)
        term2_contribution = sigma_R_sq * (1.0 - beta_ratio)

        # 总 V_drift^2
        V_drift_sq = term1_contribution + term2_contribution

        # 5. 清理结果
        # ------------------------------------------------------
        V_drift_sq = np.where(V_drift_sq < 0, 0.0, V_drift_sq) # 物理上不应小于0
        
        return V_drift_sq

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def get_stellar_vel_sq_by_density(self, radius_fitted: np.ndarray) -> np.ndarray:
        sigma_0_fitted, r_d_fitted = self._calc_stellar_central_density(self.PLATE_IFU, radius_fitted)
        vel_sq = self._stellar_vel_sq_density_profile(radius_fitted, sigma_0_fitted, r_d_fitted)
        return radius_fitted, vel_sq

    def get_stellar_vel_by_density(self, radius_fitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius, vel_sq = self.get_stellar_vel_sq_by_density(radius_fitted)
        vel_map = np.sqrt(vel_sq)
        return radius, vel_map

    def get_stellar_vel_sq_by_mass(self, radius_fitted: np.ndarray):
        radius, mass = self._get_stellar_mass(self.PLATE_IFU)
        mass_fitted, MB_fit, a_fit, MD_fit, rd_fit = self._stellar_mass_fit(radius, mass, r_min=RADIUS_MIN_KPC, radius_fitted=radius_fitted)

        vel_sq = self._stellar_vel_sq_mass_parts_profile(radius_fitted, MB_fit, a_fit, MD_fit, rd_fit)
        return radius_fitted, vel_sq
    
    def get_stellar_total_mass(self):
        _, mass = self._get_stellar_mass(self.PLATE_IFU)
        total_mass = np.nanmax(mass)
        return total_mass
    
    def get_stellar_vel_by_mass(self, radius_fitted: np.ndarray):
        radius, vel_sq = self.get_stellar_vel_sq_by_mass(radius_fitted)
        vel_map = np.sqrt(vel_sq)
        return radius, vel_map
    
    def get_stellar_vel_sq_by_mass2(self, radius_fitted: np.ndarray):
        radius, mass = self._get_stellar_mass(self.PLATE_IFU)
        mass_fitted, MB_fit, a_fit, MD_fit, rd_fit = self._stellar_mass_fit_minimize(radius, mass, r_min=RADIUS_MIN_KPC, radius_fitted=radius_fitted)

        vel_sq = self._stellar_vel_sq_mass_parts_profile(radius_fitted, MB_fit, a_fit, MD_fit, rd_fit)
        return radius_fitted, vel_sq
    
    def get_stellar_vel_by_mass2(self, radius_fitted: np.ndarray):
        radius, vel_sq = self.get_stellar_vel_sq_by_mass2(radius_fitted)
        vel_map = np.sqrt(vel_sq)
        return radius, vel_map
    

    def get_stellar_vel_drift_sq(self, radius_fitted: np.ndarray, incl:float) -> np.ndarray:
        vel_drift_sq = self._calc_stellar_vel_drift_sq(radius_fitted, incl)
        return radius_fitted,vel_drift_sq
    
    def get_stellar_vel_drift(self, radius_fitted: np.ndarray, incl:float) -> np.ndarray:
        vel_drift_sq = self._calc_stellar_vel_drift_sq(radius_fitted, incl)
        vel_drift = np.sqrt(vel_drift_sq)
        return radius_fitted, vel_drift


######################################################
# main function for test
######################################################
def main() -> None:
    from vel_rot import VelRot

    PLATE_IFU = "8723-12705"

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)
    plot_util = PlotUtil(fits_util)

    stellar = Stellar(drpall_util, firefly_util, maps_util)
    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=None)

    _, radius_h_kpc_map, _ = maps_util.get_radius_map()
    radius_max = np.nanmax(radius_h_kpc_map)
    radius_fit = np.linspace(0.0, radius_max, num=1000)
    incl = vel_rot.get_inc_rad()

    stellar.set_PLATE_IFU(PLATE_IFU)
    _, mass_map = stellar._get_stellar_mass(PLATE_IFU)
    r_map_by_mass, vel_map_by_mass = stellar.get_stellar_vel_by_mass(radius_fitted=radius_fit)
    r_map_by_mass2, vel_map_by_mass2 = stellar.get_stellar_vel_by_mass2(radius_fitted=radius_fit)
    
    print("#######################################################")
    print("# calculate stellar mass M(r)")
    print("#######################################################")
    print(f"Mass shape: {mass_map.shape}, range: [{np.nanmin(mass_map):.3f}, {np.nanmax(mass_map):,.1f}] M solar")
    print(f"Velocity stellar shape by mass: {vel_map_by_mass.shape}, range: [{np.nanmin(vel_map_by_mass):.3f}, {np.nanmax(vel_map_by_mass):.3f}]")
    print("")



    r_map, vel_map = stellar.get_stellar_vel_by_density(radius_fitted=radius_fit)
    r_drift, vel_drift_map = stellar.get_stellar_vel_drift(radius_fitted=radius_fit, incl=incl)
    print("#######################################################")
    print("# calculate stellar rotation velocity V(r)")
    print("#######################################################")
    print(f"Velocity stellar shape: {vel_map.shape}, range: [{np.nanmin(vel_map):.3f}, {np.nanmax(vel_map):.3f}]")
    print(f"Velocity Drift shape: {vel_drift_map.shape}, range: [{np.nanmin(vel_drift_map):.3f}, {np.nanmax(vel_drift_map):.3f}]")

    # valid = ~np.isnan(r_drift) & ~np.isnan(vel_drift_map)
    # if not np.any(valid):
    #     print("No valid (r_drift, vel_drift) pairs")
    # else:
    #     pairs = np.column_stack((r_drift[valid], vel_drift_map[valid]))
    #     print("(r_drift, vel_drift) pairs:")
    #     # sample 20 radii uniformly between min and max r and interpolate velocities
    #     idx_sort = np.argsort(pairs[:, 0])
    #     r_vals = pairs[idx_sort, 0]
    #     v_vals = pairs[idx_sort, 1]
    #     r_uniform = np.linspace(r_vals[0], r_vals[-1], 20)
    #     v_uniform = np.interp(r_uniform, r_vals, v_vals)
    #     for r, v in zip(r_uniform, v_uniform):
    #         print(f"{r:.6f}, {v:.6f}")
    

    plot_util.plot_rv_curve(r_map, vel_map, title="Stellar by density",
                            r_rot2_map=r_map_by_mass2, v_rot2_map=vel_map_by_mass2, title2="Stellar by mass2")

    plot_util.plot_rv_curve(r_map_by_mass, vel_map_by_mass, title="Stellar by mass",
                            r_rot2_map=r_map_by_mass2, v_rot2_map=vel_map_by_mass2, title2="Stellar by mass2")
    
    return

if __name__ == "__main__":
    main()
