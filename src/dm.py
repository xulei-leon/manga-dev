from copyreg import clear_extension_cache
from pathlib import Path

from re import M, S
import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize, basinhopping, differential_evolution
from astropy import constants as const
from astropy import units as u

from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from vel_stellar import G, Stellar

H = 0.674  # assuming H0 = 67.4 km/s/Mpc


class DmNfw:

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util

    ########################################################################################
    # NFW Dark Matter Halo Profile:
    ########################################################################################
    # --- Navarro-Frenk-White (NFW) Dark Matter Halo Rotational Velocity Squared ---
    #
    # Formula:
    # V_DM^2(r) = (V_200^2 / x) * [ (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c)) ]
    # V_DM(r): rotational velocity due to the dark matter halo at radius r.
    # V_200(r): circular speed at the virial radius r_200.
    # ln        : The natural logarithm function.
    #
    # --- Key Parameters and Context ---
    # 1. Normalized Radius (x):
    #    x = r / r_200.
    #
    # 2. Virial Radius (r_200):
    #    r_200 is the radius within which the mean density is 200 times the critical density.
    #    It is related to V_200 and the Hubble parameter H(z) by: r_200 = V_200 / (10 * H(z)).
    #
    # 3. Halo Mass (M_200) and V_200 Relation:
    #    M_200 is the halo mass within r_200. V_200 is connected to M_200 via:
    #    V_200^3 = 10 * G * H(z) * M_200, where G is the gravitational constant.
    #
    # 4. Concentration Parameter (c):
    #    c is the concentration parameter of the NFW profile. It relates to the scale radius
    #    r_s through r_s = r_200 / c.
    #
    # 5. c - M_200 Mass-Concentration Relation (Duffy et al. 2008):
    #    c is not independent; it correlates with M_200 (low-mass halos are more concentrated).
    #    The relation used here is:
    #    c = 5.74 * ( M_200 / (2 * 10^12 * h^-1 * M_sun) )^(-0.097)
    #
    # 6. Hubble Parameter (H(z)):
    #    H(z) = H_0 * sqrt( Omega_m*(1 + z)^3 + Omega_Lambda )
    #    (Using typical redshift z=0.04 for the sample.)
    #
    # Conclusion:
    # In this simplified model, the entire V_DM(r) profile is determined by a single parameter: the halo mass M_200.
    ########################################################################################

    def _get_z(self) -> float:
        z = self.drpall_util.get_redshift(self.PLATE_IFU)
        print(f"Redshift z from DRPALL: {z:.5f}")
        return z

    # hubble parameter
    # H(z) = H0 * sqrt( Omega_m*(1 + z)^3 + Omega_Lambda )
    def _calc_Hz_kpc(self, z: float, H0=67.4, Om=0.315, Ol=0.685) -> float:
        Hz = H0 * np.sqrt(Om * (1 + z)**3 + Ol)
        return Hz # in km/s/Mpc

    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        r200_Mpc = V200 / (10 * Hz)  # in Mpc
        r200_kpc = r200_Mpc * 1e3  # convert to kpc
        return r200_kpc # in kpc

    # x = r / r200
    def _calc_x_from_r200(self, radius_kpc: np.ndarray, r200_kpc: float) -> np.ndarray:
        return radius_kpc / r200_kpc

   # c = r200 / rss
   # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        M_pivot = 2e12 / h
        return 5.74 * (M200 / M_pivot)**(-0.097)

    # formula: V200^3 = 10 * G * H(z) * M200
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        G = const.G.to('Mpc km^2 / s^2 Msun').value  # Mpc km^2 / s^2 / Msun
        V200 = (10 * G * Hz * M200)**(1/3)  # in km/s
        return V200

    def _calc_M200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        G = const.G.to('Mpc km^2 / s^2 Msun').value  # Mpc km^2 / s^2 / Msun
        M200 = V200**3 / (10 * G * Hz)  # in Msun
        return M200

    ################################################################################
    # profile
    ################################################################################

    #
    # Drift Correction
    # V_drift^2 = 2 * sigma_0^2 * (R / R_d)
    #
    def _vel_drift_sq_profile(self, radius: np.ndarray, sigma_0:float, Re: float):
        Rd = Re / 1.678
        vel_drift_sq = 2 * sigma_0**2 * (radius / Rd)
        return vel_drift_sq

    # formula: Vdm ^ 2 = (V200 ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
    def _vel_dm_sq_profile_V200(self, radius_kpc: np.ndarray, V200: float, c: float, z: float) -> np.ndarray:
        r200 = self._calc_r200_from_V200(V200, z)
        x = self._calc_x_from_r200(radius_kpc, r200)
        x = np.where(x == 0, 1e-6, x)  # avoid division by zero

        num = np.log(1 + c*x) - (c*x)/(1 + c*x)
        den = np.log(1 + c) - c/(1 + c)

        V_dm_sq = (V200**2 / x) * (num / den)
        return V_dm_sq

    # formula: Vdm ^ 2 = ((10 * G * H(z) * M200) ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
    def _vel_dm_sq_profile_M200(self, radius_kpc, M200, c, z):
        V200 = self._calc_V200_from_M200(M200, z)
        r200 = self._calc_r200_from_V200(V200, z)

        if c is None:
            c = self._calc_c_from_M200(M200, h=H)

        x = self._calc_x_from_r200(radius_kpc, r200)
        x = np.where(x == 0, 1e-6, x)

        num = np.log(1 + c*x) - (c*x)/(1 + c*x)
        den = np.log(1 + c) - c/(1 + c)

        V_dm_sq = (V200**2 / x) * (num / den)
        return V_dm_sq


    ########################################################################################
    # Use the following equation to fit DM profile:
    # ignoring gas contribution for simplification
    ########################################################################################
    # V_obs^2  =  V_star^2 + V_dm^2 - V_drift^2
    #
    # Vdm ^ 2 = ((10 * G * H(z) * M200) ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
    # V_star^2 = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
    # V_drift^2 = 2 * sigma_0^2 * (R / R_d)
    ########################################################################################

    # V_rot^2 = V_star^2 + V_dm^2 - V_drift^2
    def _vel_rot_sq_profile(self, radius: np.ndarray, M200: float, c: float, z: float, M_star: float, Re:float, sigma_0:float) -> np.ndarray:
        v_dm_sq = self._vel_dm_sq_profile_M200(radius, M200, c, z)
        v_star_sq = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)
        v_drift_sq = self._vel_drift_sq_profile(radius, sigma_0, Re)
        v_rot_sq = v_dm_sq + v_star_sq  - v_drift_sq
        return v_rot_sq

    def _vel_rot_fit_model(self, radius: np.ndarray, M200: float, c, z: float, M_star: float, Re:float, sigma_0:float) -> np.ndarray:
        v_rot_sq = self._vel_rot_sq_profile(radius, M200, c, z, M_star, Re, sigma_0)
        v_rot_sq[v_rot_sq < 0] = 0.0
        v_rot = np.sqrt(v_rot_sq)
        return v_rot

    ################################################################################
    # Error functions
    ################################################################################
    # sigma: Standard Deviation of the Errors
    def _calc_loss(self, y_data: np.ndarray, y_model: np.ndarray, sigma: np.ndarray) -> float:
        residuals = (y_data - y_model) / sigma
        loss = np.nansum(residuals**2)
        return loss

    def _calc_chi_sq_v(self, y_data: np.ndarray, y_model: np.ndarray, sigma: np.ndarray, dof: int) -> float:
        chi_sq = self._calc_loss(y_data, y_model, sigma)
        chi_sq_v = chi_sq / dof
        return chi_sq_v

    def _calc_R_sq_adj(self, y_data: np.ndarray, y_model: np.ndarray, sigma: np.ndarray, dof: int) -> float:
        ss_total = np.nansum((y_data - np.nanmean(y_data))**2)
        ss_residual = self._calc_loss(y_data, y_model, sigma)
        r_sq = 1 - (ss_residual / ss_total)
        n = len(y_data)
        r_sq_adj = 1 - (1 - r_sq) * (n - 1) / dof
        return r_sq_adj

    ################################################################################
    # Fitting methods
    ################################################################################

    #
    # Least Squares Fitting (TRF)
    def _fit_dm_nfw_ls(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) &
                    (radius > 0.1) & (radius < 0.9 * np.nanmax(radius)))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot[valid_mask]
        vel_rot_err_valid = vel_rot_err[valid_mask]

        radius_max = np.nanmax(radius_valid)
        M_star = self.stellar_util.get_stellar_mass(radius_max)
        z = self._get_z()
        y_data = vel_rot_valid

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'M200': (1e10, 1e14),
            'Re': (1.0, 20.0),
            'sigma_0': (5.0, 100.0),
            'f_bulge': (1e-3, 0.5),
            'a': (0.01, 10.0),
        }

        def _denormalize_params(params_n):
            _M200_n, _Re_n, _sigma_0_n, f_bulge_n, a_n = params_n
            _M200 = _M200_n * (params_range['M200'][1] - params_range['M200'][0]) + params_range['M200'][0]
            _Re = _Re_n * (params_range['Re'][1] - params_range['Re'][0]) + params_range['Re'][0]
            _sigma_0 = _sigma_0_n * (params_range['sigma_0'][1] - params_range['sigma_0'][0]) + params_range['sigma_0'][0]
            _f_bulge = f_bulge_n * (params_range['f_bulge'][1] - params_range['f_bulge'][0]) + params_range['f_bulge'][0]
            _a = a_n * (params_range['a'][1] - params_range['a'][0]) + params_range['a'][0]
            return _M200, _Re, _sigma_0, _f_bulge, _a

        ######################################
        # Residual function for least_squares
        ######################################
        def _residual_function(params):
            _M200, _Re, _sigma_0, f_bulge, a = _denormalize_params(params)
            _y_model = self._vel_rot_fit_model(radius_valid, _M200, z, M_star, _Re, _sigma_0, f_bulge, a)
            # Residuals weighted by sigma
            residuals = (y_data - _y_model) / vel_rot_err_valid
            return residuals

        ######################################
        # Fitting process
        ######################################
        initial_guess = [0.2, 0.1, 0.1, 0.5, 0.5]  # normalized initial guess
        # Bounds for least_squares: ([lower_bounds], [upper_bounds])
        bounds = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])

        result = least_squares(_residual_function, initial_guess, bounds=bounds, method='trf', loss='huber', f_scale=1.5)

        M200_fit, Re_fit, sigma_0_fit, f_bulge_fit, a_fit = _denormalize_params(result.x)
        V200_fit = self._calc_V200_from_M200(M200_fit, z)
        r200_fit = self._calc_r200_from_V200(V200_fit, z)

        ######################################
        # Error estimation
        ######################################
        # Reduced Chi-Squared
        dof = len(y_data) - len(result.x)
        y_model = self._vel_rot_fit_model(radius_valid, M200_fit, z, M_star, Re_fit, sigma_0_fit, f_bulge_fit, a_fit)
        CHI_SQ_V = self._calc_chi_sq_v(y_data, y_model, vel_rot_err_valid, dof)
        F_factor = np.maximum(np.sqrt(CHI_SQ_V), 1.0)

        # Adjusted Coefficient of Determination (R²)
        R_SQ_ADJ = self._calc_R_sq_adj(y_data, y_model, vel_rot_err_valid, dof)
        # RMSE: Root Mean Square Error
        RMSE = np.sqrt(self._calc_loss(y_data, y_model, vel_rot_err_valid) / len(y_data))
        # MAE: Mean Absolute Error
        MAE = np.nanmean(np.abs(y_data - y_model))
        mask_pos = (y_data > 1.0)
        # MAPE: Mean Absolute Percentage Error
        MAPE = np.nanmean(np.abs((y_data[mask_pos] - y_model[mask_pos]) / y_data[mask_pos])) * 100.0
        # SMAPE: Symmetric Mean Absolute Percentage Error
        SMAPE = np.nanmean(2.0 * np.abs(y_data[mask_pos] - y_model[mask_pos]) / (np.abs(y_data[mask_pos]) + np.abs(y_model[mask_pos]))) * 100.0

        # Covariance Matrix
        if result.success is False:
            print("Warning: Optimization did not converge; cannot compute covariance matrix.")
            C = None
        else:
            # Jacobian matrix at the solution
            J = result.jac
            # Approximate Hessian: H ≈ J.T @ J
            # Covariance matrix: C = (J.T @ J)^-1
            try:
                C = np.linalg.inv(J.T @ J)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix encountered; cannot compute covariance matrix.")
                C = None

        if C is not None:
            # Correlation Matrix
            COR_MATRIX = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))

            # Standard Errors of the Parameters
            param_errors = np.sqrt(np.diag(C)) * F_factor  # scaled by F_factor
            M200_norm_err, Re_norm_err, sigma_0_norm_err, f_bulge_norm_err, a_norm_err = param_errors
            M200_err, Re_err, sigma_0_err, f_bulge_err, a_err = (
                M200_norm_err * (params_range['M200'][1] - params_range['M200'][0]),
                Re_norm_err * (params_range['Re'][1] - params_range['Re'][0]),
                sigma_0_norm_err * (params_range['sigma_0'][1] - params_range['sigma_0'][0]),
                f_bulge_norm_err * (params_range['f_bulge'][1] - params_range['f_bulge'][0]),
                a_norm_err * (params_range['a'][1] - params_range['a'][0]),
            )

            M200_err_pct = (M200_err / M200_fit) * 100.0 if M200_fit != 0 else np.nan
            Re_err_pct = (Re_err / Re_fit) * 100.0 if Re_fit != 0 else np.nan
            sigma_0_err_pct = (sigma_0_err / sigma_0_fit) * 100.0 if sigma_0_fit != 0 else np.nan
            f_bulge_err_pct = (f_bulge_err / f_bulge_fit) * 100.0 if f_bulge_fit != 0 else np.nan
            a_err_pct = (a_err / a_fit) * 100.0 if a_fit != 0 else np.nan
        else:
            M200_err = Re_err = sigma_0_err = f_bulge_err = a_err = np.nan
            M200_err_pct = Re_err_pct = sigma_0_err_pct = f_bulge_err_pct = a_err_pct = np.nan
            COR_MATRIX = None


        print(f"\n------------ Fitted Dark Matter NFW (least_squares TRF) ------------")
        print(f" IFU                : {self.PLATE_IFU}")
        print(f" Fitted: M200       : {M200_fit:.3e} Msun, ± {M200_err:.3e} Msun ({M200_err_pct:.2f} %)")
        print(f" Fitted: Re         : {Re_fit:.3f} kpc, ± {Re_err:.3f} kpc ({Re_err_pct:.2f} %)")
        print(f" Fitted: sigma_0    : {sigma_0_fit:.3f} km/s, ± {sigma_0_err:.3f} km/s ({sigma_0_err_pct:.2f} %)")
        print(f" Fitted: f_bulge    : {f_bulge_fit:.3f}, ± {f_bulge_err:.3f} ({f_bulge_err_pct:.2f} %)") if f_bulge_fit is not None else None
        print(f" Fitted: a          : {a_fit:.3f} kpc, ± {a_err:.3f} kpc ({a_err_pct:.2f} %)") if a_fit is not None else None
        print(f" Calculated: V200   : {V200_fit:.3f} km/s")
        print(f" Calculated: r200   : {r200_fit:.3f} kpc")
        print("--- Error --------------------------------")
        print(f" Reduced Chi-Squared    : {CHI_SQ_V:.3f}")
        print(f" Adjusted R²            : {R_SQ_ADJ:.3f}")
        print(f" RMSE                   : {RMSE:.3f} (km/s)^2")
        print(f" MAE                    : {MAE:.3f} (km/s)^2")
        print(f" MAPE                   : {MAPE:.3f} %")
        print(f" SMAPE                  : {SMAPE:.3f} %")
        print(f" Correlation Matrix     : \n{COR_MATRIX}")
        print("------------------------------------------------------------\n")

        ######################################
        # Return fitted velocity profile
        ######################################
        vel_rot_sq_fit = self._vel_rot_sq_profile(radius, M200_fit, z, M_star, Re_fit, sigma_0_fit, f_bulge_fit, a_fit)
        vel_dm_sq_fit = self._vel_dm_sq_profile_M200(radius, M200_fit, c=None, z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re_fit, f_bulge_fit, a_fit)

        vel_total_fit = np.sqrt(vel_rot_sq_fit)
        vel_dm_fit = np.sqrt(vel_dm_sq_fit)
        vel_star_fit = np.sqrt(vel_star_sq_fit)

        # calculate error estimate
        return radius, vel_total_fit, vel_dm_fit, vel_star_fit


    # Differential Evolution Fitting
    def _fit_dm_nfw_de(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) &
                    (radius > 0.1) & (radius < 1.0 * np.nanmax(radius)))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot[valid_mask]
        vel_rot_err_valid = vel_rot_err[valid_mask]

        if len(radius_valid) < 10:
            print("Not enough valid data points for fitting.")
            return None

        radius_max = np.nanmax(radius_valid)
        M_star, Re = self.stellar_util.get_stellar_mass(radius_max)
        z = self._get_z()
        y_data = vel_rot_valid

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'M200': (5e10, 1e14),
            'c': (1.0, 25.0),
            'sigma_0': (0.0, 80.0),
        }

        def _denormalize_params(params_norm):
            _M200_n, _c_n, _sigma_0_n = params_norm
            _M200 = _M200_n * (params_range['M200'][1] - params_range['M200'][0]) + params_range['M200'][0]
            _c = _c_n * (params_range['c'][1] - params_range['c'][0]) + params_range['c'][0]
            _sigma_0 = _sigma_0_n * (params_range['sigma_0'][1] - params_range['sigma_0'][0]) + params_range['sigma_0'][0]
            return _M200, _sigma_0, _c

        #######################################################
        # First stage: Global optimize - Differential Evolution
        #######################################################
        # Loss function: Penalize unphysical parameters
        def _global_loss_function(params_norm):
            _M200, _c, _sigma_0 = _denormalize_params(params_norm)
            _y_model = self._vel_rot_fit_model(radius_valid, _M200, _c, z, M_star, Re, _sigma_0)
            loss = self._calc_loss(y_data, _y_model, vel_rot_err_valid)

            # ---------- penalties (scalar) - balanced and soft ----------
            # 1) negative V^2 penalty (linear penalty on negative part)
            Vtot_sq = self._vel_rot_sq_profile(radius_valid, _M200, _c, z, M_star, Re, _sigma_0)
            if not np.all(np.isfinite(Vtot_sq)):
                return 1e15

            # extract magnitudes of negative V^2 values (zero where V^2 is non-negative)
            negatives = np.where(Vtot_sq < 0.0, -Vtot_sq, 0.0)
            if np.any(negatives):
                loss += 1e6 + np.sum(negatives) * 1e4  # large penalty to avoid unphysical

            # 2) cosmological c-M soft prior: use expected scatter ~0.2 dex
            c_expected = 10.0 * (_M200 / 1e12) ** (-0.1)
            sigma_logc = 0.2  # dex typical scatter
            logc_dev = (np.log(_c) - np.log(c_expected)) / sigma_logc
            loss += 1e2 * np.sum(logc_dev ** 2)  # χ²-like prior

            # 3) V_drift vs V_rot: allow up to frac_drift (0.4), penalize linear excess
            V_drift_sq = self._vel_drift_sq_profile(radius_valid, _sigma_0, Re)
            if not np.all(np.isfinite(V_drift_sq)):
                return 1e15

            # excess = np.maximum(0.0, V_drift_sq - (0.4 * _y_model ** 2))
            # if np.any(excess > 0):
            #     loss += 1e3 * np.sum(excess)  # soft linear penalty

            # 4) V_star shouldn't hugely exceed observed: allow factor 2, penalize linear excess
            vstar_sq = self.stellar_util.stellar_vel_sq_profile(radius_valid, M_star, Re)
            if not np.all(np.isfinite(vstar_sq)):
                return 1e15
            excess_star = np.maximum(0.0, vstar_sq - 2.0 * _y_model ** 2)
            if np.any(excess_star > 0):
                loss += 1e2 * np.sum(excess_star)

            # 5) small penalty to discourage extremes of sigma_0 near bounds (but not force)
            if (_sigma_0 <= params_range['sigma_0'][0]) or (_sigma_0 >= params_range['sigma_0'][1]):
                loss += 1e3


            fb = M_star / (_M200 + M_star)
            loss += 30 * (np.log(fb) - np.log(0.05))**2  # target baryon fraction ~0.15

            V200_fit = self._calc_V200_from_M200(_M200, z)
            if V200_fit > 500.0:
                loss += 1e4

            if not np.isfinite(loss):
                return 1e15

            return float(loss)

        # Fitting process
        global_bounds = [(0.0, 1.0)] * 3 # normalized bounds

        # differential_evolution does not take an initial guess in the same way, it explores the bounds
        global_result = differential_evolution(
            _global_loss_function,
            global_bounds,
            strategy='best1bin',
            maxiter=150,
            popsize=20,
            tol=0.1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False
        )

        global_M200, global_c, global_sigma_0 = _denormalize_params(global_result.x)

        #######################################################
        # Second stage: Local refine - least_squares
        #######################################################
        def _residual_function(params_norm):
            _M200, _c, _sigma_0 = _denormalize_params(params_norm)

            # model prediction
            y_model = self._vel_rot_fit_model(radius_valid, _M200, _c, z, M_star, Re, _sigma_0)

            # data residuals (weighted)
            data_resid = (y_data - y_model) / vel_rot_err_valid  # shape (N,)

            penalty_list = []

            # (a) negative V^2 penalty -> scaled to typical measurement uncertainty
            Vtot_sq = self._vel_rot_sq_profile(radius_valid, _M200, _c, z, M_star, Re, _sigma_0)
            neg = -np.minimum(Vtot_sq, 0.0)
            # compress into single scalar (mean normalized by a velocity scale)
            if np.any(neg > 0):
                penalty_list.append( (np.mean(neg) / (np.mean(y_data**2) + 1e-8)) * 50.0 )

            # (b) c-M soft prior (1 residual, normalized by scatter ~0.2 dex)
            c_expected = 10.0 * (_M200 / 1e12) ** (-0.1)
            logc_dev = (np.log(_c) - np.log(c_expected)) / 0.2
            penalty_list.append(logc_dev)  # this will be treated like a residual ~ N(0,1)

            # (c) V_drift excess (mean fractional excess)
            V_drift_sq = self._vel_drift_sq_profile(radius_valid, _sigma_0, Re)
            # excess = np.maximum(0.0, V_drift_sq - 0.4 * y_model ** 2)
            # penalty_list.append( np.mean(excess) / (np.mean(y_data**2) + 1e-8) * 10.0 )

            # (d) sigma_0 soft-range: if outside preferred range, add penalty residual
            # sigma_low, sigma_high = 8.0, 70.0
            # if _sigma_0 < sigma_low:
            #     penalty_list.append( (sigma_low - _sigma_0) / (sigma_low + 1e-8) * 5.0 )
            # elif _sigma_0 > sigma_high:
            #     penalty_list.append( (_sigma_0 - sigma_high) / (sigma_high + 1e-8) * 5.0 )
            # else:
            #     penalty_list.append(0.0)

            # combine: data_resid (N) + penalty_list (M small)
            if len(penalty_list) > 0:
                return np.concatenate([data_resid, np.array(penalty_list, dtype=float)])
            else:
                return data_resid

        local_initial_guess = global_result.x.copy()
        local_bounds = (np.zeros(3), np.ones(3))
        local_result = least_squares(
            _residual_function,
            local_initial_guess,
            bounds=local_bounds,
            method='trf',
            loss='huber', # robust to outliers
            f_scale=1.5,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=2000
        )
        M200_fit, c_fit, sigma_0_fit = _denormalize_params(local_result.x)
        V200_fit = self._calc_V200_from_M200(M200_fit, z)
        r200_fit = self._calc_r200_from_V200(V200_fit, z)

        ######################################
        # Error estimation
        ######################################
        # Reduced Chi-Squared
        y_model_final = self._vel_rot_fit_model(radius_valid, M200_fit, c_fit, z, M_star, Re, sigma_0_fit)
        data_resid_final = (y_data - y_model_final) / vel_rot_err_valid
        chi2 = np.sum(data_resid_final**2)
        dof = max(1, len(y_data) - len(local_result.x))  # number of data points - number of fitted parameters
        redchi2 = chi2 / dof

        # attempt covariance estimation from jacobian
        cov = None
        try:
            J = local_result.jac  # shape (N+M, 4)
            # use only the data rows (first len(y_data)) to estimate J_data
            J_data = J[:len(y_data), :]
            JTJ = J_data.T.dot(J_data)
            cov = np.linalg.inv(JTJ) * (chi2 / dof)
            perr_norm = np.sqrt(np.diag(cov))  # in normalized parameter units
            # convert normalized errors to physical units
            M200_err = perr_norm[0] * (params_range['M200'][1] - params_range['M200'][0])
            sigma_0_err = perr_norm[1] * (params_range['sigma_0'][1] - params_range['sigma_0'][0])
            c_err = perr_norm[2] * (params_range['c'][1] - params_range['c'][0])

            M200_err_pct = (M200_err / M200_fit) * 100.0 if M200_fit != 0 else np.nan
            sigma_0_err_pct = (sigma_0_err / sigma_0_fit) * 100.0 if sigma_0_fit != 0 else np.nan
            c_err_pct = (c_err / c_fit) * 100.0 if c_fit != 0 else np.nan

        except Exception:
            M200_err = Re_err = sigma_0_err = c_err = np.nan
            cov = None


        print(f"\n------------ Fitted Dark Matter NFW (least_squares) ------------")
        print(f"--- Global optimize (differential_evolution) ---")
        print(f" Global Fit: M200    : {global_M200:.3e} Msun")
        print(f" Global Fit: c       : {global_c:.3f}")
        print(f" Global Fit: sigma_0 : {global_sigma_0:.3f} km/s")
        print(f"--- Local optimize (least_squares) ---")
        print(f" IFU                : {self.PLATE_IFU}")
        print(f" Fitted: M200       : {M200_fit:.3e} Msun, ± {M200_err:.3e} Msun ({M200_err_pct:.2f} %)")
        print(f" Fitted: c          : {c_fit:.3f}, ± {c_err:.3f} ({c_err_pct:.2f} %)")
        print(f" Fitted: sigma_0    : {sigma_0_fit:.3f} km/s, ± {sigma_0_err:.3f} km/s ({sigma_0_err_pct:.2f} %)")
        print("--- Calculated ---")
        print(f" Stellar Mass       : {M_star:.3e} Msun")
        print(f" Half-Mass R(Re)    : {Re:.3f} kpc")
        print(f" Calculated: V200   : {V200_fit:.3f} km/s")
        print(f" Calculated: r200   : {r200_fit:.3f} kpc")
        print("--- Error ---")
        print(f" Reduced Chi-Squared    : {redchi2:.3f}")
        print("------------------------------------------------------------\n")

        ######################################
        # Return fitted velocity profile
        ######################################
        vel_rot_sq_fit = self._vel_rot_sq_profile(radius, M200_fit, c_fit, z, M_star, Re, sigma_0_fit)
        vel_dm_sq_fit = self._vel_dm_sq_profile_M200(radius, M200_fit, c=c_fit, z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)

        vel_total_fit = np.sqrt(np.clip(vel_rot_sq_fit, a_min=0, a_max=None))
        vel_dm_fit = np.sqrt(np.clip(vel_dm_sq_fit, a_min=0, a_max=None))
        vel_star_fit = np.sqrt(np.clip(vel_star_sq_fit, a_min=0, a_max=None))

        return radius, vel_total_fit, vel_dm_fit, vel_star_fit

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def set_stellar_util(self, stellar_util: Stellar) -> None:
        self.stellar_util = stellar_util
        return

    def fit_dm_nfw(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z()
        radius_fit, vel_total, vel_dm_fit, vel_star_fit = self._fit_dm_nfw_de(radius, vel_rot, vel_rot_err)
        return radius_fit, vel_total, vel_dm_fit, vel_star_fit


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

    dm_nfw = DmNfw(drpall_util)
    dm_nfw.set_PLATE_IFU(PLATE_IFU)

    _, radius_h_kpc_map, _ = maps_util.get_radius_map()
    radius_max = np.nanmax(radius_h_kpc_map)
    radius_fit = np.linspace(0.0, radius_max, num=1000)

    print("### Test: DM based on M200")
    z = dm_nfw._get_z()
    M200 = 3*10**12 # example halo mass in Msun
    print(f"M200: {M200:.3e} Msun")
    V200 = dm_nfw._calc_V200_from_M200(M200, z)
    r200 = dm_nfw._calc_r200_from_V200(V200, z)
    c = dm_nfw._calc_c_from_M200(M200, h=0.7)
    print(f"Calculated V200: {V200:.2f} km/s, r200: {r200:.2f} kpc, c: {c:.2f}")

    vel_dm_sq = dm_nfw._vel_dm_sq_profile_M200(radius_fit, M200, c, z)  # Example usage
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")

    print("### Test: DM based on V200")
    V200 = 200
    c = 10
    print(f"V200: {V200:.2f} km/s, c: {c:.2f}")
    Hz = dm_nfw._calc_Hz_kpc(z)
    print(f"Hz: {Hz:.2f} km/s/Mpc")
    r200 = dm_nfw._calc_r200_from_V200(V200, z)
    M200 = dm_nfw._calc_M200_from_V200(V200, z)
    print(f"Calculated r200: {r200:.2f} kpc, M200 from V200: {M200:.3e} Msun")
    x = dm_nfw._calc_x_from_r200(radius_fit, r200)
    print(f"Calculated x  shape: {x.shape}, range: {np.nanmin(x):.5f} - {np.nanmax(x):.5f}")
    c = dm_nfw._calc_c_from_M200(M200, h=0.7)
    print(f"Calculated c from M200: {c:.2f}")

    vel_dm_sq = dm_nfw._vel_dm_sq_profile_V200(radius_fit, V200, c, z)
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")

# main entry
if __name__ == "__main__":
    main()