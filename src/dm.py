import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from astropy import constants as const
import matplotlib.pyplot as plt

from util.drpall_util import DrpallUtil
from vel_stellar import Stellar

H = 0.674  # assuming H0 = 67.4 km/s/Mpc
G = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun


class DmNfw:
    drpall_util: DrpallUtil
    PLATE_IFU: str
    plot_enable: bool
    fit_debug: bool = False

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util
        self.PLATE_IFU = None
        self.plot_enable = False

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

    # Moster-like SHMR
    def _calc_Mstar_from_Mhalo(self, Mhalo: float, M1=10**11.59, N=0.0351, beta=1.376, gamma=0.608):
        x = Mhalo / M1
        f = 2.0 * N / (x**(-beta) + x**(gamma))
        return f * Mhalo

    # invert (in log10 space)
    def _calc_M200_log_from_Mstar(self, Mstar: float, Mmin=1e9, Mmax=1e15):
        def f(logM):
            M = 10**logM
            return self._calc_Mstar_from_Mhalo(M) - Mstar

        return brentq(f, float(np.log10(Mmin)), float(np.log10(Mmax)))

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


    ################################################################################
    # MCMC PyMC inference methods
    ################################################################################
    def _inf_dm_nfw_pymc(self, radius_obs: np.ndarray, vel_obs: np.ndarray, ivar_obs: np.ndarray, vel_sys: float, inc_rad: float, phi_map: np.ndarray):
        # ---------------------
        # 1) data selection / precompute
        # ---------------------
        valid_mask = (np.isfinite(vel_obs) & np.isfinite(radius_obs) & np.isfinite(ivar_obs) & np.isfinite(phi_map) &
                    (radius_obs > 0.01) & (radius_obs < 1.0 * np.nanmax(radius_obs)))
        radius_valid = radius_obs[valid_mask]
        vel_obs_valid = vel_obs[valid_mask]
        ivar_obs_valid = ivar_obs[valid_mask]
        phi_map_valid = phi_map[valid_mask]

        if len(radius_valid) < 50:
            print("Not enough valid data points for fitting.")
            return None

        print(f"NFW pymc radius valid: range=[{np.min(radius_valid):.2f}, {np.max(radius_valid):.2f}] kpc")
        print(f"NFW pymc vel obs valid {len(vel_obs_valid)}: range=[{np.min(vel_obs_valid):.2f}, {np.max(vel_obs_valid):.2f}] km/s")

        # calculate stderr from ivar
        stderr_obs_valid = 1.0 / np.sqrt(ivar_obs_valid)

        # stellar quantities
        Mstar, Re = self.stellar_util.fit_stellar_mass()
        z = self._get_z()

        Mmin=1e9
        Mmax=1e15
        M200_log_mu = self._calc_M200_log_from_Mstar(Mstar, Mmin=Mmin, Mmax=Mmax)  # expected log10(M200) from Mstar

        # precompute stellar contribution v_star^2 (numpy array)
        # Do not convert to tensor yet, do it inside the model
        # v_star_sq = self.stellar_util.stellar_vel_sq_profile(radius_valid, M_star, Re)

        # get H(z) in km/s/kpc
        Hz = self._calc_Hz_kpc(z)
        G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun

        # ---------------------
        # 2) model bounds and parameterization
        #    we sample log10(V200) and log10(c) to handle scales nicely
        # ---------------------
        # use range_parameters defined above
        c_min, c_max = (1.0, 50.0)
        logc_min, logc_max = np.log10(c_min), np.log10(c_max)

        # ---------------------
        # helper functions (closures)
        # Note: use pytensor operations instead of numpy/scipy inside the model
        # ---------------------

        # Mstar
        def v_star_sq_bulge_hernquist(r, MB, a):
            r = pt.where(r == 0, 1e-6, r)  # avoid division by zero
            v_sq = G * MB * r / (r + a)**2
            return v_sq

        # V_disk^2(r) = (2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
        # Calculate V_disk^2 using piecewise polynomial approximation
        def v_star_sq_disk_freeman(r, M_d, Rd):
            r_safe = pt.where(pt.eq(r, 0), 1e-6, r)
            x = r_safe / Rd

            # inner
            f1 = 0.5 * x**2 - 0.0625 * x**4

            # mid
            a1, a2, a3 = 0.1935, 0.0480, -0.0019
            b1, b2, b3 = 0.8215, 0.1936, 0.0103

            num = a1 * x**2 + a2 * x**3 + a3 * x**4
            den = 1.0 + b1*x + b2*x**2 + b3*x**3
            f2 = num / den

            # outer
            f3= (1.0/x) * (1.0 - 0.5/x + 0.375/x**2)

            f = pt.switch(pt.le(x, 1.5), f1, pt.switch(pt.le(x, 4.0), f2, f3))
            v_sq = (G * M_d / Rd) * f
            return v_sq


        # M_star: total mass of star
        # Re: Half-mass radius
        # f_bulge: bulge mass fraction
        # a: Hernquist scale radius
        # V_star^2 = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
        def v_star_sq_profile(r, Mstar, Re, f_bulge, a):
            Rd = Re / 1.678
            MB = f_bulge * Mstar
            MD = (1 - f_bulge) * Mstar

            v_bulge_sq = v_star_sq_bulge_hernquist(r, MB, a)
            v_disk_sq = v_star_sq_disk_freeman(r, MD, Rd)
            v_baryon_sq = v_bulge_sq + v_disk_sq
            return v_baryon_sq

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

        def v_dm_sq_profile(r_arr, M200, c, V200):
            x = x_from_M200(r_arr, M200)
            num, den = nfw_num_den(x, c)
            x_safe = pt.maximum(x, 1e-6)
            den_safe = pt.maximum(den, 1e-6)
            return (V200**2 / x_safe) * (num / den_safe)

        # V_drift^2 closure
        def v_drift_sq_profile(r_arr, sigma_0):
            return 2.0 * (sigma_0 ** 2) * (r_arr / Re)

        # total v_rot^2 closure
        def v_rot_sq_profile(v_dm_sq, v_star_sq, v_drift_sq):
            return v_dm_sq + v_star_sq - v_drift_sq

        # Formula: v_obs = v_sys + v_rot * (sin(i) * cos(phi - phi_0))
        # Warning: The sign of the calculated velocity may be different from the observed velocity.
        def v_obs_project_profile(v_rot, v_sys, inc, phi_map):
            phi_delta = (phi_map + pt.pi) % (2 * pt.pi)  # phi_map is (phi - phi_0)
            correction = pt.sin(inc) * pt.cos(phi_delta)
            v_obs = v_sys + v_rot * correction
            return v_obs

        # ---------------------
        # 3) PyMC model
        # ---------------------
        with pm.Model() as model:
            # ---------------------
            # prior distributions
            # ---------------------
            # M200 prior: from SHMR
            M200_log_sigma = 0.25
            M200_log_t = pm.TruncatedNormal("M200_log", mu=M200_log_mu, sigma=M200_log_sigma, lower=pt.log10(Mmin), upper=pt.log10(Mmax))
            # student-t prior for robustness
            # M200_log_t = pm.StudentT("M200_log", nu=3, mu=M200_log_mu, sigma=M200_log_sigma)

            # c prior: log-normal prior
            log_c_mu = pt.log10(5.0)
            c_log_t = pm.TruncatedNormal("c_log", mu=log_c_mu, sigma=0.25, lower=logc_min, upper=logc_max)

            # sigma_0 prior: half normal
            # sigma_0 > 0
            sigma_0_t = pm.HalfNormal("sigma_0", sigma=10.0)

            # v_sys prior: normal prior around measured value
            v_sys_delta = 20.0
            v_sys_t = pm.TruncatedNormal("v_sys", mu=vel_sys, sigma=5.0, lower=vel_sys - v_sys_delta, upper=vel_sys + v_sys_delta)

            # inc prior: normal prior around measured value
            inc_delta = pt.deg2rad(5.0)
            inc_t = pm.TruncatedNormal("inc", mu=inc_rad, sigma=0.1, lower=pt.maximum(0.0, inc_rad - inc_delta), upper=pt.minimum(pt.pi/2, inc_rad + inc_delta))

            # Re prior
            # Re_t = pm.TruncatedNormal("Re", mu=Re, sigma=0.1 * Re, lower=0.1 * Re, upper=10.0 * Re)

            # f_bulge prior
            f_bulge_t = pm.Beta("f_bulge", alpha=2.0, beta=5.0)

            # a prior
            a_mu = Re / 1.8
            a_t = pm.TruncatedNormal("a", mu=a_mu, sigma=0.1, lower=0.01, upper=10.0)

            # ---------------------
            # deterministic relations
            # ---------------------
            # V200: derived from M200
            M200_t = pm.Deterministic("M200", 10 ** M200_log_t)
            c_t = pm.Deterministic("c", 10 ** c_log_t)
            V200_t = pm.Deterministic("V200", (10 * G_kpc_kms_Msun * Hz * M200_t) ** (1.0 / 3.0))

            r = radius_valid  # numpy array
            v_star_sq_t = pm.Deterministic("v_star_sq", v_star_sq_profile(r, Mstar, Re, f_bulge_t, a_t))
            v_dm_sq_t = pm.Deterministic("v_dm_sq", v_dm_sq_profile(r, M200_t, c_t, V200_t))
            v_drift_sq_t = pm.Deterministic("v_drift_sq", v_drift_sq_profile(r, sigma_0_t))
            v_rot_sq_t = pm.Deterministic("v_rot_sq", v_rot_sq_profile(v_dm_sq_t, v_star_sq_t, v_drift_sq_t))

            # ---------------------
            # likelihood
            # ---------------------
            # model velocity: ensure non-negative argument to sqrt
            v_rot_sq_pos = pt.maximum(v_rot_sq_t, 1e-6)
            v_rot_model = pt.sqrt(v_rot_sq_pos)
            v_obs_model =  v_obs_project_profile(v_rot_model, v_sys_t, inc_t, phi_map_valid)

            # likelihood: observed rotation velocities
            v_obs_sigma = stderr_obs_valid

            # pm.Normal("v_obs", mu=v_obs_model, sigma=v_obs_sigma, observed=vel_obs_valid)
            pm.StudentT("v_obs", nu=5, mu=v_obs_model, sigma=v_obs_sigma, observed=vel_obs_valid)

            # ---------------------
            # potential
            # ---------------------

            # Add SHMR as a potential (log-prior)
            # Do not double count if using truncated normal prior above
            # pm.Potential("shmr_penalty", -0.5 * ((M200_log_t - M200_log_mu) / M200_log_sigma) ** 2)

            # Optionally add a c-M prior (cosmological relation) if desired:
            # Add c-M prior (cosmological relation) as a potential (log-prior)
            # c_expected = 10.0 * (M200_t / 1e12)**(-0.1)
            # sigma_logc = 0.2  # dex
            # pm.Potential("c_M200_penalty", -0.5 * ((pt.log10(c_t) - pt.log10(c_expected)) / sigma_logc)**2)

            # Penalize regions where v_rot_sq < 0 to discourage unphysical solutions.
            #
            neg_term = pt.maximum(-v_rot_sq_t, 0.0)
            tau_penalty = 1e-4
            penalty_val = -1.0 * (neg_term**2) / (2.0 * tau_penalty**2)
            pm.Potential("v_rot_sq_penalty", pt.sum(penalty_val))

            # ---------------------
            # 4) sampling options & run
            # ---------------------
            draws=2000
            tune=1000
            chains=4
            # draws = 1000
            # tune = 500
            # chains = 2
            target_accept=0.95

            print("Starting PyMC sampling (NUTS)... this may take time.")
            if self.fit_debug:
                displaybar = True
            else:
                displaybar = False

            trace = pm.sample(init="jitter+adapt_diag", draws=draws, tune=tune, chains=chains, nuts_sampler='nutpie', target_accept=target_accept, cores=min(chains, 4),
                              progressbar=displaybar,
                              return_inferencedata=True, compute_convergence_checks=True)

            pm.compute_log_likelihood(trace)
            ppc_idata = pm.sample_posterior_predictive(trace, random_seed=42, extend_inferencedata=True)

        # ---------------------
        # 5) postprocess
        # ---------------------
        # summary with diagnostics
        # variable in the posterior. Exclude it from the az.summary var_names list.
        summary = az.summary(trace, var_names=["M200", "c", "sigma_0", "v_sys", "inc", 'a'], round_to=3)

        M200_mean = float(summary.loc["M200", "mean"])
        c_mean = float(summary.loc["c", "mean"])
        sigma0_mean = float(summary.loc["sigma_0", "mean"])
        v_sys_mean = float(summary.loc["v_sys", "mean"])
        inc_mean = float(summary.loc["inc", "mean"])
        a_mean = float(summary.loc["a", "mean"])

        M200_sd = float(summary.loc["M200", "sd"])
        c_sd = float(summary.loc["c", "sd"])
        sigma0_sd = float(summary.loc["sigma_0", "sd"])
        v_sys_sd = float(summary.loc["v_sys", "sd"])
        inc_sd = float(summary.loc["inc", "sd"])
        a_sd = float(summary.loc["a", "sd"])

        M200_r_hat = float(summary.loc["M200", "r_hat"])
        c_r_hat = float(summary.loc["c", "r_hat"])
        sigma0_r_hat = float(summary.loc["sigma_0", "r_hat"])
        v_sys_r_hat = float(summary.loc["v_sys", "r_hat"])
        inc_r_hat = float(summary.loc["inc", "r_hat"])
        a_r_hat = float(summary.loc["a", "r_hat"])

        # extract posterior samples
        posterior = trace.posterior
        # v_obs_samples = posterior["v_obs"].stack(samples=("chain", "draw")).values
        # v_obs_mean = np.mean(v_obs_samples, axis=1)
        v_dm_sq_samples = posterior["v_dm_sq"].stack(samples=("chain", "draw")).values
        v_dm_sq_mean = np.mean(v_dm_sq_samples, axis=1)
        v_dm_mean = np.sqrt(np.clip(v_dm_sq_mean, a_min=0.0, a_max=None))
        v_star_sq_samples = posterior["v_star_sq"].stack(samples=("chain", "draw")).values
        v_star_sq_mean = np.mean(v_star_sq_samples, axis=1)
        v_star_mean = np.sqrt(np.clip(v_star_sq_mean, a_min=0.0, a_max=None))
        v_drift_sq_samples = posterior["v_drift_sq"].stack(samples=("chain", "draw")).values
        v_drift_sq_mean = np.mean(v_drift_sq_samples, axis=1)
        v_drift_mean = np.sqrt(np.clip(v_drift_sq_mean, a_min=0.0, a_max=None))
        v_rot_sq_samples = posterior["v_rot_sq"].stack(samples=("chain", "draw")).values
        v_rot_sq_mean = np.mean(v_rot_sq_samples, axis=1)
        v_rot_mean = np.sqrt(np.clip(v_rot_sq_mean, a_min=0.0, a_max=None))

        # LOO
        # Request pointwise LOO to include pareto_k values for diagnostics
        model_loo = az.loo(trace, pointwise=True)
        elpd_loo_est = float(model_loo.elpd_loo)
        elpd_loo_se = float(model_loo.se)
        p_loo = float(model_loo.p_loo)
        max_k = float(model_loo.pareto_k.max().values)
        mean_k = float(model_loo.pareto_k.mean().values)
        good_k_fraction = float(np.sum(model_loo.pareto_k.values < 0.7) / len(model_loo.pareto_k.values))

        V200_calc = self._calc_V200_from_M200(M200_mean, z)
        r200_calc = self._calc_r200_from_V200(V200_calc, z)
        c_calc = self._calc_c_from_M200(M200_mean, h=getattr(self, "H", 0.7))

        # ---------------------
        # residual diagnostics
        # ---------------------
        # posterior predictive v_obs
        v_obs_ppc_raw = ppc_idata.posterior_predictive["v_obs"].stack(sample=("chain", "draw")).values

        # Standardize to shape (n_samples, n_points) to avoid axis-order surprises
        n_points = len(radius_valid)
        if v_obs_ppc_raw.ndim != 2:
            raise ValueError(f"Unexpected v_obs_ppc shape: {v_obs_ppc_raw.shape}")
        if v_obs_ppc_raw.shape[-1] == n_points:
            v_obs_ppc = v_obs_ppc_raw
        elif v_obs_ppc_raw.shape[0] == n_points:
            v_obs_ppc = v_obs_ppc_raw.T
        else:
            raise ValueError(
                f"Posterior predictive v_obs shape {v_obs_ppc_raw.shape} does not match n_points={n_points}."
            )

        # Compute posterior predictive mean at the observed radius_valid points
        v_obs_mean = np.mean(v_obs_ppc, axis=0)

        # ---------------------
        # residuals
        # ---------------------
        mask = np.isfinite(vel_obs_valid) & np.isfinite(v_obs_mean) & np.isfinite(stderr_obs_valid) & (stderr_obs_valid > 0)
        residual = vel_obs_valid - v_obs_mean
        residual_std = np.full_like(residual, np.nan, dtype=float)
        residual_std[mask] = residual[mask] / stderr_obs_valid[mask]

        res_use = residual[mask]
        rmse = float(np.sqrt(np.mean(res_use**2)))
        nrmse = float(rmse / np.mean(np.abs(vel_obs_valid[mask])))
        mae = float(np.mean(np.abs(res_use)))
        bias = float(np.mean(res_use))

        # chi^2 diagnostics (approx.)
        chi2 = float(np.sum(residual_std[mask] ** 2))
        # parameters in model: M200_log, c_log, sigma_0, v_sys, inc, f_bulge, a  -> ~7
        dof = int(max(np.sum(mask) - 7, 1))
        redchi = float(np.sqrt(chi2 / dof))

        # ---------------------
        # Inference summary info
        # ---------------------
        if self.fit_debug:
            print("\n------------ Infer Dark Matter NFW (PyMC) ------------")
            print("--- Summary ---")
            print(summary)
            print("--- LOO ---")
            print(f"{model_loo}")
            print(f"--- median estimates ---")
            print(f" Infer M200         : {M200_mean:.3e} ± {M200_sd:.3e} Msun ({M200_sd/M200_mean:.2%})")
            print(f" Infer c            : {c_mean:.3f} ± {c_sd:.3f} ({c_sd/c_mean:.2%})")
            print(f" Infer sigma_0      : {sigma0_mean:.3f} ± {sigma0_sd:.3f} km/s ({sigma0_sd/sigma0_mean:.2%})")
            print(f" Infer v_sys        : {v_sys_mean:.3f} ± {v_sys_sd:.3f} km/s ({v_sys_sd/v_sys_mean:.2%})")
            print(f" Infer inc          : {np.degrees(inc_mean):.3f} ± {np.degrees(inc_sd):.3f} deg ({inc_sd/inc_mean:.2%})")
            print(f" Infer a            : {a_mean:.3f} ± {a_sd:.3f} kpc ({a_sd/a_mean:.2%})")
            print(f"--- caculate ---")
            print(f" Calc: V200         : {V200_calc:.3f} km/s")
            print(f" Calc: r200         : {r200_calc:.3f} kpc")
            print(f" Calc: c            : {c_calc:.3f}")
            print(f" Calc: v_sys        : {vel_sys:.3f} km/s")
            print(f" Calc: inc          : {np.degrees(inc_rad):.3f} deg")
            print(f" Stellar Mass       : {Mstar:.3e} Msun")
            print(f" Half-Mass R(Re)    : {Re:.3f} kpc")
            print("--- diagnostics ---")
            print(f" Reduced Chi        : {redchi:.3f}")
            print(f" RMSE               : {rmse:.3f} km/s")
            print(f" NRMSE              : {nrmse:.3f}")
            print(f" MAE                : {mae:.3f} km/s")
            print(f" Bias               : {bias:.3f} km/s")
            print("------------------------------------------------------------\n")

        # ---------------------
        # plot
        # ---------------------
        if self.plot_enable:
            axes_trace = az.plot_trace(trace, var_names=["M200", "c", "sigma_0", "v_sys", "inc"])
            # set the units and credible interval
            # axes_trace is usually (n_vars, 2) for trace plots (left: pdf, right: trace)
            # Accessing rows for variables
            if axes_trace.shape[0] >= 3:
                axes_trace[0,0].set_xlabel("Msun")
                axes_trace[2,0].set_xlabel("km/s")
            plt.tight_layout()

            # az.plot_posterior(trace, var_names=["M200", "c", "sigma_0"], hdi_prob=0.94)

            az.plot_pair(ppc_idata, var_names=["M200", "c", "sigma_0", "v_sys", "inc"], kind='kde', marginals=True)
            az.plot_ppc(ppc_idata, data_pairs={"v_obs": "v_obs"}, mean=True, kind='cumulative', num_pp_samples=200)
            plt.tight_layout()
            plt.show()

            # v_obs fit plot
            plt.figure(figsize=(8,6))
            plt.errorbar(radius_valid, vel_obs_valid, yerr=stderr_obs_valid, fmt='o', label='Observed V_obs', alpha=0.5)
            r_plot = np.linspace(0.0, np.nanmax(radius_valid), num=500)

            # compute posterior predictive mean and credible intervals
            v_obs_ppc_raw = ppc_idata.posterior_predictive["v_obs"].stack(sample=("chain", "draw")).values
            n_points = len(radius_valid)
            if v_obs_ppc_raw.ndim != 2:
                raise ValueError(f"Unexpected v_obs_ppc shape: {v_obs_ppc_raw.shape}")
            if v_obs_ppc_raw.shape[-1] == n_points:
                v_obs_ppc = v_obs_ppc_raw
            elif v_obs_ppc_raw.shape[0] == n_points:
                v_obs_ppc = v_obs_ppc_raw.T
            else:
                raise ValueError(
                    f"Posterior predictive v_obs shape {v_obs_ppc_raw.shape} does not match n_points={n_points}."
                )

            # Interpolate each posterior predictive sample onto r_plot
            v_obs_interp = np.array([
                np.interp(r_plot, radius_valid, v_obs_ppc[s, :]) for s in range(v_obs_ppc.shape[0])
            ])
            v_obs_mean = np.mean(v_obs_interp, axis=0)
            v_obs_hdi = az.hdi(v_obs_interp, hdi_prob=0.94)
            plt.plot(r_plot, v_obs_mean, color='red', label='Posterior Predictive Mean V_obs')
            plt.fill_between(r_plot, v_obs_hdi[:,0], v_obs_hdi[:,1], color='red', alpha=0.3, label='94% Credible Interval')
            plt.xlabel('Radius (kpc)')
            plt.ylabel('Rotation Velocity V_obs (km/s)')
            plt.title(f'Fitted Rotation Curve for {self.PLATE_IFU}')
            plt.legend()

            plt.tight_layout()
            plt.show()

        success = True
        inf_result = {
            'radius': radius_valid,
            'v_rot': v_rot_mean,
            'v_dm': v_dm_mean,
            'v_star': v_star_mean,
            'v_drift': v_drift_mean,
        }

        inf_params = {
            'M200': M200_mean,
            'M200_std': M200_sd,
            'M200_r_hat': M200_r_hat,
            'c': c_mean,
            'c_std': c_sd,
            'c_r_hat': c_r_hat,
            'sigma_0': sigma0_mean,
            'sigma_0_std': sigma0_sd,
            'sigma_0_r_hat': sigma0_r_hat,
            'v_sys': v_sys_mean,
            'v_sys_std': v_sys_sd,
            'v_sys_r_hat': v_sys_r_hat,
            'inc': inc_mean,
            'inc_std': inc_sd,
            'inc_r_hat': inc_r_hat,
            'a': a_mean,
            'a_std': a_sd,
            'a_r_hat': a_r_hat,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'chi2': chi2,
            'chi2_red': redchi,
            'elpd_loo_est': elpd_loo_est,
            'elpd_loo_se': elpd_loo_se,
            'p_loo': p_loo,
            'max_k': max_k,
            'mean_k': mean_k,
            'good_k_fraction': f"{good_k_fraction:.2f}",
        }

        return success, inf_result, inf_params


    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def set_plot_enable(self, plot_enable: bool) -> None:
        self.plot_enable = plot_enable
        return

    def set_fit_debug(self, fit_debug: bool) -> None:
        self.fit_debug = fit_debug
        return

    def set_stellar_util(self, stellar_util: Stellar) -> None:
        self.stellar_util = stellar_util
        return

    def inf_dm_nfw(self, radius_obs: np.ndarray, vel_obs: np.ndarray, ivar_obs: np.ndarray, vel_sys: float, inc_rad: float, phi_map: np.ndarray):
        return self._inf_dm_nfw_pymc(radius_obs, vel_obs, ivar_obs, vel_sys, inc_rad, phi_map)

