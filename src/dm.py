from copyreg import clear_extension_cache
from pathlib import Path

from re import M, S
import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize, basinhopping, differential_evolution
from scipy.optimize import brentq
import emcee
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt

from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from vel_stellar import G, Stellar

H = 0.674  # assuming H0 = 67.4 km/s/Mpc

# physical constant used (kpc, km/s, Msun)
G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value

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
        Hz = Hz / 1000
        return Hz # in km/s/kpc

    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        r200_kpc = V200 / (10 * Hz)  # in kpc
        return r200_kpc # in kpc

    # x = r / r200
    def _calc_x_from_r200(self, radius_kpc: np.ndarray, r200_kpc: float) -> np.ndarray:
        return radius_kpc / r200_kpc

   # c = r200 / rss
   # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        M_pivot_h_inv = 2e12 # in Msun/h
        mass_ratio = M200 / (M_pivot_h_inv / h)
        return 5.74 * (mass_ratio)**(-0.097)

    # formula: V200^3 = 10 * G * H(z) * M200
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        G = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun
        V200 = (10 * G * Hz * M200)**(1/3)  # in km/s
        return V200

    def _calc_M200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        G = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun
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

    def _vel_rot_sq_V200(self, radius: np.ndarray, V200: float, c: float, z: float, M_star: float, Re:float, sigma_0:float) -> np.ndarray:
        v_dm_sq = self._vel_dm_sq_profile_V200(radius, V200, c, z)
        v_star_sq = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)
        v_drift_sq = self._vel_drift_sq_profile(radius, sigma_0, Re)
        v_rot_sq = v_dm_sq + v_star_sq  - v_drift_sq
        return v_rot_sq

    def _vel_rot_fit_model_V200(self, radius: np.ndarray, V200: float, c: float, z: float, M_star: float, Re:float, sigma_0:float) -> np.ndarray:
        v_rot_sq = self._vel_rot_sq_V200(radius, V200, c, z, M_star, Re, sigma_0)
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

    # Differential Evolution Fitting
    def _fit_dm_nfw(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) &
                    (radius > 0.01) & (radius < 1.0 * np.nanmax(radius)))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot[valid_mask]
        vel_rot_err_valid = vel_rot_err[valid_mask]

        if len(radius_valid) < 10:
            print("Not enough valid data points for fitting.")
            return None

        M_star, Re = self.stellar_util.fit_stellar_mass()
        z = self._get_z()
        y_data = vel_rot_valid

        ######################################
        # normal all fit parameters
        ######################################
        params_range = {
            'V200': (10.0, 800.0),
            'c': (1.0, 30.0),
            'sigma_0': (0.0, 300.0),
        }

        def _denormalize_params(params_norm):
            _V200_n, _c_n, _sigma_0_n = params_norm
            _V200 = _V200_n * (params_range['V200'][1] - params_range['V200'][0]) + params_range['V200'][0]
            _c = _c_n * (params_range['c'][1] - params_range['c'][0]) + params_range['c'][0]
            _sigma_0 = _sigma_0_n * (params_range['sigma_0'][1] - params_range['sigma_0'][0]) + params_range['sigma_0'][0]
            return _V200, _c, _sigma_0

        #######################################################
        # First stage: Global optimize - Differential Evolution
        #######################################################
        # Loss function: Penalize unphysical parameters
        def _global_loss_function(params_norm):
            # unpack normalized params
            _V200, _c, _sigma_0 = _denormalize_params(params_norm)

            # precompute components
            v_star_sq_base = self.stellar_util.stellar_vel_sq_profile(radius_valid, M_star, Re)
            v_star_sq = v_star_sq_base
            v_dm_sq = self._vel_dm_sq_profile_V200(radius_valid, _V200, _c, z)
            v_drift_sq = self._vel_drift_sq_profile(radius_valid, _sigma_0, Re)

            # total model velocity^2
            v_model_sq = v_star_sq + v_dm_sq - v_drift_sq
            valid_mask = v_model_sq >= 0
            v_model = np.zeros_like(v_model_sq)
            v_model[valid_mask] = np.sqrt(v_model_sq[valid_mask])

            chi2 = np.sum(((y_data - v_model) / vel_rot_err_valid) ** 2)
            loss = chi2

            P_FACTOR = 0.0 # 1e3
            P_BASE = 0.0 # 1e5

            # penalty A: negative V^2
            if not np.all(valid_mask):
                negative_sum = np.sum(-v_model_sq[~valid_mask])
                loss += P_BASE + negative_sum * P_FACTOR

            # penalty B: V_dm must not be negative
            if np.any(v_dm_sq < 0):
                loss += P_BASE + np.sum(-v_dm_sq[v_dm_sq < 0]) * P_FACTOR

            # penalty C: M200 / M_star prior
            _M200 = self._calc_M200_from_V200(_V200, z)
            m_ratio = _M200 / M_star
            log_m_ratio = np.log10(m_ratio + 1e-10)

            if log_m_ratio < 1.0:
                loss += P_BASE + P_FACTOR * (1.0 - log_m_ratio) ** 2 * 10
            elif log_m_ratio > 3.0:
                loss += P_BASE + P_FACTOR * (log_m_ratio - 3.0) ** 2

            # penalty D: drift dominance
            excess_drift = v_drift_sq - (1.5 * v_star_sq)
            if np.any(excess_drift > 0):
                loss += P_BASE + np.sum(excess_drift[excess_drift > 0]) * 100

            # penalty E: soft prior on c
            if _c < 1.0:
                loss += P_BASE + P_FACTOR * (1.0 - _c) ** 2 * 5000
            elif _c > 30.0:
                loss += P_BASE + P_FACTOR * (_c - 30.0) ** 2

            if not np.isfinite(loss):
                return 1e15

            return float(loss)


        # Fitting process
        global_bounds = [(0.0, 1.0)] * 4 # normalized bounds

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

        V200_fit, c_fit, sigma_0_fit = _denormalize_params(global_result.x)

        M200_calc = self._calc_M200_from_V200(V200_fit, z)
        r200_calc = self._calc_r200_from_V200(V200_fit, z)
        c_calc = self._calc_c_from_M200(M200_calc, h=H)

        print(f"\n------------ Fitted Dark Matter NFW (differential_evolution) ------------")
        print(f" Global Fit: V200    : {V200_fit:.3e} km/s")
        print(f" Global Fit: c       : {c_fit:.3f}")
        print(f" Global Fit: sigma_0 : {sigma_0_fit:.3f} km/s")
        print("--- Calculated ---")
        print(f" Stellar Mass       : {M_star:.3e} Msun")
        print(f" Half-Mass R(Re)    : {Re:.3f} kpc")
        print(f" Calculated: M200   : {M200_calc:.3e} Msun")
        print(f" Calculated: r200   : {r200_calc:.3f} kpc")
        print(f" Calculated: c      : {c_calc:.3f}")
        print("------------------------------------------------------------\n")

        ######################################
        # Return fitted velocity profile
        ######################################
        vel_rot_sq_fit = self._vel_rot_sq_V200(radius, V200_fit, c_fit, z, M_star, Re, sigma_0_fit)
        vel_dm_sq_fit = self._vel_dm_sq_profile_V200(radius, V200_fit, c=c_fit, z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)

        vel_total_fit = np.sqrt(np.clip(vel_rot_sq_fit, a_min=0, a_max=None))
        vel_dm_fit = np.sqrt(np.clip(vel_dm_sq_fit, a_min=0, a_max=None))
        vel_star_fit = np.sqrt(np.clip(vel_star_sq_fit, a_min=0, a_max=None))

        return radius, vel_total_fit, vel_dm_fit, vel_star_fit


    ################################################################################
    # MCMC emcee inference methods
    # emcee is a bayesian MCMC sampler, which can provide posterior distributions of parameters.
    ################################################################################

    def _inf_dm_nfw_emcee(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) &
                    (radius > 0.01) & (radius < 1.0 * np.nanmax(radius)))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot[valid_mask]
        vel_rot_err_valid = vel_rot_err[valid_mask]

        if len(radius_valid) < 10:
            print("Not enough valid data points for fitting.")
            return None

        M_star, Re = self.stellar_util.fit_stellar_mass()
        z = self._get_z()

        # Normalization helpers
        params_range = {
            'V200': (10.0, 800.0),
            'c': (1.0, 30.0),
            'sigma_0': (0.1, 300.0),
        }

        def _denormalize_params(params_norm):
            _V200_n, _c_n, _sigma_0_n = params_norm
            _V200 = params_range['V200'][0] * (params_range['V200'][1] / params_range['V200'][0]) ** _V200_n
            _c = params_range['c'][0] * (params_range['c'][1] / params_range['c'][0]) ** _c_n
            _sigma_0 = params_range['sigma_0'][0] * (params_range['sigma_0'][1] / params_range['sigma_0'][0]) ** _sigma_0_n
            return _V200, _c, _sigma_0

        def _normalize_params(params):
            _V200, _c, _sigma_0 = params
            _V200_n = np.log(_V200 / params_range['V200'][0]) / np.log(params_range['V200'][1] / params_range['V200'][0])
            _c_n = np.log(_c / params_range['c'][0]) / np.log(params_range['c'][1] / params_range['c'][0])
            _sigma_0_val = max(_sigma_0, params_range['sigma_0'][0])
            _sigma_0_n = np.log(_sigma_0_val / params_range['sigma_0'][0]) / np.log(params_range['sigma_0'][1] / params_range['sigma_0'][0])
            return [_V200_n, _c_n, _sigma_0_n]

        def _log_prior(params_norm: list[float]) -> float:
            # Check if normalized parameters are within [0, 1]
            if np.all((np.array(params_norm) > 0.0) & (np.array(params_norm) < 1.0)):
                return 0.0
            return -np.inf

        def _log_likelihood(params_norm: list[float], radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray, z: float, M_star: float, Re: float) -> float:
            V200, c, sigma_0 = _denormalize_params(params_norm)
            v_model = self._vel_rot_fit_model_V200(radius, V200, c, z, M_star, Re, sigma_0)
            residuals = (vel_rot - v_model) / vel_rot_err
            log_likelihood = -0.5 * np.sum(residuals**2)
            return log_likelihood

        def _log_posterior(params_norm: list[float], radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray, z: float, M_star: float, Re: float) -> float:
            lp = _log_prior(params_norm)
            if not np.isfinite(lp):
                return -np.inf
            ll = _log_likelihood(params_norm, radius, vel_rot, vel_rot_err, z, M_star, Re)
            return lp + ll

        def _initialize_walkers(n_walkers: int, initial_guess_norm: list[float], spread: float = 1e-4) -> np.ndarray:
            initial_pos = []
            for _ in range(n_walkers):
                pos = np.array(initial_guess_norm) + spread * np.random.randn(len(initial_guess_norm))
                # Ensure initial positions are within bounds [0, 1]
                pos = np.clip(pos, 0.001, 0.999)
                initial_pos.append(pos)
            return np.array(initial_pos)

        # Initial guess (physical units) -> Normalized
        initial_guess_phys = [200.0, 4.0, 0.1]  # V200, c, sigma_0
        initial_guess_norm = _normalize_params(initial_guess_phys)

        n_walkers = 50
        n_steps = 4000
        discard = int(0.3 * n_steps)

        initial_pos = _initialize_walkers(n_walkers, initial_guess_norm, spread=1e-2)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            3,
            _log_posterior,
            args=(radius_valid, vel_rot_valid, vel_rot_err_valid, z, M_star, Re)
        )

        print("Running MCMC sampling...")
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

        # Get samples (normalized)
        samples_norm = sampler.get_chain(discard=discard, thin=15, flat=True)

        # Denormalize samples for statistics
        samples_phys = np.array([_denormalize_params(p) for p in samples_norm])

        V200_mcmc, c_mcmc, sigma_0_mcmc = map(
            lambda v: (np.percentile(v, 16), np.percentile(v, 50), np.percentile(v, 84)),
            samples_phys.T
        )

        M200_calc = self._calc_M200_from_V200(V200_mcmc[1], z)
        r200_calc = self._calc_r200_from_V200(V200_mcmc[1], z)
        c_calc = self._calc_c_from_M200(M200_calc, h=H)

        tau = sampler.get_autocorr_time(tol=0)

        print(f"\n------------ Infer Dark Matter NFW (emcee) ------------")
        print(f" Infer V200         : {V200_mcmc[1]:.3e} (+{V200_mcmc[2]-V200_mcmc[1]:.3e}/-{V200_mcmc[1]-V200_mcmc[0]:.3e}) km/s")
        print(f" Infer c            : {c_mcmc[1]:.3f} (+{c_mcmc[2]-c_mcmc[1]:.3f}/-{c_mcmc[1]-c_mcmc[0]:.3f})")
        print(f" Infer sigma_0      : {sigma_0_mcmc[1]:.3f} (+{sigma_0_mcmc[2]-sigma_0_mcmc[1]:.3f}/-{sigma_0_mcmc[1]-sigma_0_mcmc[0]:.3f}) km/s")
        print("---------------------")
        print(f" Stellar Mass       : {M_star:.3e} Msun")
        print(f" Half-Mass R(Re)    : {Re:.3f} kpc")
        print(f" Calculated: M200   : {M200_calc:.3e} Msun")
        print(f" Calculated: r200   : {r200_calc:.3f} kpc")
        print(f" Calculated: c      : {c_calc:.3f}")
        print("---------------------")
        print(f" mean acceptance fraction       : {np.mean(sampler.acceptance_fraction):.3f}")
        print(f" autocorrelation time estimates : {tau}")
        print("------------------------------------------------------------\n")
        # Return fitted velocity profile using median parameters
        vel_rot_sq_fit = self._vel_rot_sq_V200(
            radius,
            V200_mcmc[1],
            c_mcmc[1],
            z,
            M_star,
            Re,
            sigma_0_mcmc[1]
        )
        vel_dm_sq_fit = self._vel_dm_sq_profile_V200(radius, V200_mcmc[1], c=c_mcmc[1], z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)
        vel_total_fit = np.sqrt(np.clip(vel_rot_sq_fit, a_min=0, a_max=None))
        vel_dm_fit = np.sqrt(np.clip(vel_dm_sq_fit, a_min=0, a_max=None))
        vel_star_fit = np.sqrt(np.clip(vel_star_sq_fit, a_min=0, a_max=None))
        return radius, vel_total_fit, vel_dm_fit, vel_star_fit

    ################################################################################
    # MCMC PyMC inference methods
    ################################################################################
    def _inf_dm_nfw_pymc(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        range_parameters = {
            'V200': (10.0, 2000.0),   # km/s
            'c': (1.0, 50.0),         # dimensionless
            'sigma_0': (0.0, 100.0),  # km/s
        }

        # ---------------------
        # 1) data selection / precompute
        # ---------------------
        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) &
                    (radius > 0.01) & (radius < 1.0 * np.nanmax(radius)))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot[valid_mask]
        vel_rot_err_valid = vel_rot_err[valid_mask]

        if len(radius_valid) < 10:
            print("Not enough valid data points for fitting.")
            return None

        # stellar quantities
        M_star, Re = self.stellar_util.fit_stellar_mass()
        z = self._get_z()

        # precompute stellar contribution v_star^2 (numpy array)
        v_star_sq = self.stellar_util.stellar_vel_sq_profile(radius_valid, M_star, Re)

        # get H(z) in km/s/kpc
        Hz = self._calc_Hz_kpc(z)
        G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun

        # ---------------------
        # 2) model bounds and parameterization
        #    we sample log10(V200) and log10(c) to handle scales nicely
        # ---------------------
        # use range_parameters defined above
        c_min, c_max = range_parameters['c']
        sigma0_min, sigma0_max = range_parameters['sigma_0']

        logc_min, logc_max = np.log10(c_min), np.log10(c_max)
        # sigma0 in linear space (we will put a prior on it)
        sigma0_min, sigma0_max = sigma0_min, sigma0_max

        # ---------------------
        # helper functions (closures)
        # ---------------------

        # Moster-like SHMR
        def Mstar_from_Mhalo(Mhalo, M1=10**11.59, N=0.0351, beta=1.376, gamma=0.608):
            x = Mhalo / M1
            f = 2.0 * N / (x**(-beta) + x**(gamma))
            return f * Mhalo

        # invert (in log10 space)
        def M200_log_from_Mstar(Mstar, Mmin=1e9, Mmax=1e15):
            def f(logM):
                M = 10**logM
                return Mstar_from_Mhalo(M) - Mstar

            return brentq(f, float(np.log10(Mmin)), float(np.log10(Mmax)))

        # r200 closure (kpc) from M200 and Hz
        def r200_from_M200(M200):
            return (G_kpc_kms_Msun * M200 / (100.0 * Hz ** 2)) ** (1.0 / 3.0)

        # normalized radius x = r / r200
        def x_from_M200(r_arr, M200):
            return r_arr / r200_from_M200(M200)

        # numerator/denominator for NFW profile
        def nfw_num_den(x_arr, c):
            cx = c * x_arr
            num = pt.log1p(cx) - (cx) / (1.0 + cx)
            den = pt.log1p(c) - c / (1.0 + c)
            return num, den

        # V_dm^2 closure (uses V200 deterministic)
        def v_dm_sq_profile(r_arr, M200, c):
            x = x_from_M200(r_arr, M200)
            num, den = nfw_num_den(x, c)
            x_safe = pt.maximum(x, 1e-6)
            den_safe = pt.maximum(den, 1e-6)
            return (V200_t**2 / x_safe) * (num / den_safe)

        # V_drift^2 closure
        def v_drift_sq_profile(r_arr, sigma_0):
            return 2.0 * (sigma_0 ** 2) * (r_arr / Re)

        # total v_rot^2 closure
        def v_rot_sq_profile(r_arr, M200, c, sigma_0, v_star_sq):
            v_dm = v_dm_sq_profile(r_arr, M200, c)
            # convert precomputed stellar array to tensor so ops are all in pytensor
            v_star_tensor = pt.as_tensor_variable(v_star_sq)
            v_drift = v_drift_sq_profile(r_arr, sigma_0)
            return v_dm + v_star_tensor - v_drift

        # ---------------------
        # 3) PyMC model
        # ---------------------
        with pm.Model() as model:

            # ---------------------
            # prior distributions
            # ---------------------
            # M200 prior: from SHMR
            M200_log_mu = M200_log_from_Mstar(M_star)  # expected log10(M200) from Mstar
            M200_log_sigma = 0.2
            M200_log_t = pm.Normal("M200_log", mu=M200_log_mu, sigma=M200_log_sigma)  # auxiliary variable for prior

            # c prior: log-normal prior
            log_c_mu = np.log10(5.0)
            c_log_t = pm.TruncatedNormal("c_log", mu=log_c_mu, sigma=0.3, lower=logc_min, upper=logc_max)

            # sigma_0 prior: half normal
            sigma_0_t = pm.HalfNormal("sigma_0", sigma=10.0)

            # ---------------------
            # deterministic relations
            # ---------------------
            # V200: derived from M200
            M200_t = pm.Deterministic("M200", 10 ** M200_log_t)
            c_t = pm.Deterministic("c", 10 ** c_log_t)
            V200_t = pm.Deterministic("V200", (10 * G_kpc_kms_Msun * Hz * M200_t) ** (1.0 / 3.0))

            r = radius_valid  # numpy array
            v_dm_sq = pm.Deterministic("v_dm_sq", v_dm_sq_profile(r, M200_t, c_t))
            v_drift_sq = pm.Deterministic("v_drift_sq", v_drift_sq_profile(r, sigma_0_t))
            v_rot_sq = pm.Deterministic("v_rot_sq", v_rot_sq_profile(r, M200_t, c_t, sigma_0_t, v_star_sq))

            # ---------------------
            # likelihood
            # ---------------------
            # model velocity: ensure non-negative argument to sqrt
            v_rot_sq_pos = pt.maximum(v_rot_sq, 1e-6)
            v_model = pt.sqrt(v_rot_sq_pos)

            # likelihood: observed rotation velocities
            v_rot_obs_sigma = vel_rot_err_valid
            v_rot_obs = vel_rot_valid
            pm.Normal("v_rot_obs", mu=v_model, sigma=v_rot_obs_sigma, observed=v_rot_obs)

            # ---------------------
            # potential
            # ---------------------

            # Add SHMR as a potential (log-prior)
            pm.Potential("shmr_penalty", -0.5 * ((M200_log_t - M200_log_mu) / M200_log_sigma) ** 2)

            # Optionally add a c-M prior (cosmological relation) if desired:
            # Add c-M prior (cosmological relation) as a potential (log-prior)
            c_expected = 10.0 * (M200_t / 1e12)**(-0.1)
            sigma_logc = 0.2  # dex
            # pm.Potential("c_M200_penalty", -0.5 * ((pt.log10(c_t) - pt.log10(c_expected)) / sigma_logc)**2)

            # Penalize regions where v_rot_sq < 0 to discourage unphysical solutions.
            #
            neg_term = pt.maximum(-v_rot_sq, 0.0)
            tau_penalty = 1e-4
            penalty_val = -1.0 * (neg_term**2) / (2.0 * tau_penalty**2)
            pm.Potential("vrot_sq_penalty", pt.sum(penalty_val))

            # ---------------------
            # 4) sampling options & run
            # ---------------------
            draws=2000
            tune=1000
            chains=4
            target_accept=0.9

            print("Starting PyMC sampling (NUTS)... this may take time.")
            trace = pm.sample(init="adapt_diag", draws=draws, tune=tune, chains=chains, nuts_sampler='nutpie', target_accept=target_accept, cores=min(chains, 4),
                              progressbar=True,
                              return_inferencedata=True, compute_convergence_checks=True)

            ppc = pm.sample_posterior_predictive(trace, var_names=["v_rot_obs"], random_seed=42, extend_inferencedata=True)

        # ---------------------
        # 5) postprocess
        # ---------------------
        # summary with diagnostics
        summary = az.summary(trace, var_names=["M200", "c", "sigma_0"], round_to=3)

        # ---------------------
        # plot
        # ---------------------
        plot_mcmc = True
        if plot_mcmc:
            az.plot_trace(trace, var_names=["M200", "c", "sigma_0"])
            az.plot_posterior(trace, var_names=["M200", "c", "sigma_0"], hdi_prob=0.94)

            idata = pm.to_inference_data(trace, posterior_predictive=ppc)
            az.plot_pair(idata, var_names=["M200", "c", "sigma_0"], kind='kde', marginals=True)
            az.plot_ppc(idata, data_pairs={"v_rot_obs": "v_rot_obs"}, mean=True, kind='cumulative', num_pp_samples=200)
            plt.show()

        # median estimates
        M200_mean = float(trace.posterior["M200"].mean().values)
        c_mean = float(trace.posterior["c"].mean().values)
        sigma0_mean = float(trace.posterior["sigma_0"].mean().values)

        M200_sd = float(trace.posterior["M200"].std().values)
        c_sd = float(trace.posterior["c"].std().values)
        sigma0_sd = float(trace.posterior["sigma_0"].std().values)

        # derived quantities using your helper functions (they expect numeric inputs)
        V200_calc = self._calc_V200_from_M200(M200_mean, z)
        r200_calc = self._calc_r200_from_V200(V200_calc, z)
        c_calc = self._calc_c_from_M200(M200_mean, h=getattr(self, "H", 0.7))

        print("\n------------ Infer Dark Matter NFW (PyMC) ------------")
        print(summary)
        print(f"--- median estimates ---")
        print(f" Infer M200         : {M200_mean:.3e} ± {M200_sd:.3e} Msun ({M200_sd/M200_mean:.2%})")
        print(f" Infer c            : {c_mean:.3f} ± {c_sd:.3f} ({c_sd/c_mean:.2%})")
        print(f" Infer sigma_0      : {sigma0_mean:.3f} ± {sigma0_sd:.3f} km/s ({sigma0_sd/sigma0_mean:.2%} km/s)")
        print(f"--- caculate ---")
        print(f" Stellar Mass       : {M_star:.3e} Msun")
        print(f" Half-Mass R(Re)    : {Re:.3f} kpc")
        print(f" Calc: V200         : {V200_calc:.3f} km/s")
        print(f" Calc: r200         : {r200_calc:.3f} kpc")
        print(f" Calc: c            : {c_calc:.3f}")
        print("------------------------------------------------------------\n")

        # compute fitted velocity profiles using median params (use your helpers)
        vel_rot_sq_fit = self._vel_rot_sq_profile(radius, M200_mean, c_mean, z, M_star, Re, sigma0_mean)
        vel_dm_sq_fit = self._vel_dm_sq_profile_M200(radius, M200_mean, c=c_mean, z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)

        vel_total_fit = np.sqrt(np.clip(vel_rot_sq_fit, a_min=0.0, a_max=None))
        vel_dm_fit = np.sqrt(np.clip(vel_dm_sq_fit, a_min=0.0, a_max=None)) if vel_dm_sq_fit is not None else None
        vel_star_fit = np.sqrt(np.clip(vel_star_sq_fit, a_min=0.0, a_max=None))

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
        radius_fit, vel_total, vel_dm_fit, vel_star_fit = self._fit_dm_nfw(radius, vel_rot, vel_rot_err)
        return radius_fit, vel_total, vel_dm_fit, vel_star_fit

    def inf_dm_nfw(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z()
        radius_fit, vel_total, vel_dm_fit, vel_star_fit = self._inf_dm_nfw_pymc(radius, vel_rot, vel_rot_err)
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
    print(f"Hz: {Hz:.2f} km/s/kpc")
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