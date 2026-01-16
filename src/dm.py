'''
Dark Matter NFW profile inference module

Use the following equation to fit DM profile:
Vobs^2(r)   =  Vstar^2(r) + Vdm^2(r) - Vdrift^2(r)
Vdm^2(r) = ((10 * G * H(z) * M200) ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
Vstar^2(r)  = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
Vdrift^2{r) = 2 * sigma_0^2 * (r / R_d)
'''

import os
from pathlib import Path
from pickletools import read_stringnl_noescape
from time import sleep

import numpy as np
from scipy.optimize import brentq
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from astropy import constants as const
import matplotlib.pyplot as plt

from util.drpall_util import DrpallUtil
from stellar import Stellar

H = 1 #0.674  # assuming H0 = 67.4 km/s/Mpc
G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun

INFER_RHAT_THRESHOLD = 1.05
VEL_SYSTEM_ERROR = 5.0  # km/s, floor error as systematic uncertainty in velocity measurements


class DmNfw:
    drpall_util: DrpallUtil
    PLATE_IFU: str
    plot_enable: bool
    inf_debug: bool = False
    inf_drift: bool = True
    inf_Re: bool = True # Notice: Infer Re will take too much time
    inf_inc: bool = False # Notice: Infer inc may be degenerate with c
    inf_phi_delta: bool = True

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util
        self.PLATE_IFU = None
        self.plot_enable = False

    ################################################################################
    # Helper calculation methods
    ################################################################################
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

   # c = r200 / rss
   # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        M_pivot_h_inv = 2e12 # in Msun/h
        mass_ratio = M200 / (M_pivot_h_inv / h)
        return 5.74 * (mass_ratio)**(-0.097)

    # formula: V200^3 = 10 * G * H(z) * M200
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/kpc
        V200 = (10 * G_kpc_kms_Msun * Hz * M200)**(1/3)  # in km/s
        return V200

    # Moster-like SHMR
    def _calc_Mstar_from_Mhalo(self, Mhalo: float, M1=10**11.59, N=0.0351, beta=1.376, gamma=0.608):
        x = Mhalo / M1
        f = 2.0 * N / (x**(-beta) + x**(gamma))
        return f * Mhalo

    def _calc_M200_from_Mstar(self, Mstar: float, Mmin=1e9, Mmax=1e15):
        def f(M):
            return self._calc_Mstar_from_Mhalo(M) - Mstar

        return brentq(f, Mmin, Mmax)

    ################################################################################
    # MCMC PyMC inference methods
    ################################################################################
    def _inf_dm_nfw_pymc(self, radius_obs: np.ndarray, vel_obs: np.ndarray, ivar_obs: np.ndarray, vel_sys: float, inc_rad: float, phi_map: np.ndarray):
        # ------------------------------------------
        # data selection / precompute
        # ------------------------------------------
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

        # Convert inverse-variance to 1-sigma error
        stderr_obs_valid = np.sqrt(1.0 / ivar_obs_valid)
        print(f"vel_obs stderr: {np.nanmean(stderr_obs_valid):.2f} km/s")

        r_max = np.nanmax(radius_valid)

        # stellar mass
        if self.inf_Re:
            _Mstar_total = self.stellar_util.get_stellar_mass_total()
            _Mstar_elpetro, _Mstar_sersic = self.drpall_util.get_stellar_mass(self.PLATE_IFU)
            print (f"Stellar mass from DRPALL: Mstar_elpetro={_Mstar_elpetro:.2e} Msun, Mstar_sersic={_Mstar_sersic:.2e} Msun, Mstar_total={_Mstar_total:.2e} Msun")
            Mstar = _Mstar_elpetro if _Mstar_elpetro is not None else _Mstar_sersic if _Mstar_sersic is not None else _Mstar_total
        else:
            fit_stellar_mass_results = self.stellar_util.fit_stellar_mass()
            Mstar = fit_stellar_mass_results['Mstar']
            Re = fit_stellar_mass_results['Re']

        z = self._get_z()

        Mmin=1e9
        Mmax=1e15
        M200_expect = self._calc_M200_from_Mstar(Mstar, Mmin=Mmin, Mmax=Mmax)  # expected M200 from Mstar
        print(f"M200_mu: {M200_expect:.2e}")
        c_expect = self._calc_c_from_M200(M200_expect, h=H)
        print(f"c_expect: {c_expect:.2f}")

        # precompute stellar contribution v_star^2 (numpy array)
        # Do not convert to tensor yet, do it inside the model
        # v_star_sq = self.stellar_util.stellar_vel_sq_profile(radius_valid, M_star, Re)

        # get H(z) in km/s/kpc
        Hz = self._calc_Hz_kpc(z)

        # ------------------------------------------
        # helper functions (closures)
        # Note: use pytensor operations instead of numpy/scipy inside the model
        # ------------------------------------------

        # Mstar
        def v_star_sq_bulge_hernquist(r, MB, a):
            r = pt.where(r == 0, 1e-6, r)  # avoid division by zero
            v_sq = (G_kpc_kms_Msun * MB * r) / (r + a)**2
            return v_sq

        # V_disk^2(r) = (2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
        def v_star_sq_disk_freeman(r, M_d, Rd):
            r_safe = pt.where(pt.eq(r, 0), 1e-6, r)
            y = r_safe / (2.0 * Rd)

            # PyMC does not expose bessel_i0/i1/k0/k1 in all versions.
            # Use PyTensor's generic modified Bessel functions:
            # I_n(y) = iv(n, y), K_n(y) = kv(n, y)
            I0 = pt.iv(0, y)
            I1 = pt.iv(1, y)
            K0 = pt.kv(0, y)
            K1 = pt.kv(1, y)

            v_sq = (2.0 * G_kpc_kms_Msun * M_d / Rd) * (y**2) * (I0 * K0 - I1 * K1)
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
        def x_from_M200(r, M200):
            return r / r200_from_M200(M200)

        # numerator/denominator for NFW profile
        def nfw_num_den(x, c):
            cx = c * x
            num = pt.log1p(cx) - (cx) / (1.0 + cx)
            den = pt.log1p(c) - c / (1.0 + c)
            return num, den

        def v_dm_sq_profile(r, M200, c, V200):
            x = x_from_M200(r, M200)
            num, den = nfw_num_den(x, c)
            x_safe = pt.maximum(x, 1e-6)
            den_safe = pt.maximum(den, 1e-6)
            return (V200**2 / x_safe) * (num / den_safe)

        # V_drift^2 = 2 * sigma_0^2 * (R / R_d)
        # Re is equivalent to the half-mass radius
        # Re = 1.68 * Rd.
        def v_drift_sq_profile(r, sigma_0, Re):
            R_d = Re / 1.678
            return 2.0 * (sigma_0 ** 2) * (r / R_d)

        # total v_rot^2 closure
        def v_rot_sq_profile(v_dm, v_star, v_drift):
            return v_dm**2 + v_star**2 - v_drift**2

        # Formula: v_obs = v_sys + v_rot * (sin(i) * cos(phi - phi_0))
        # Warning: The sign of the calculated velocity may be different from the observed velocity.
        def v_obs_project_profile(v_rot, v_sys, inc, phi_map):
            # phi_delta = (phi_map + pt.pi) % (2 * pt.pi)  # phi_map is (phi - phi_0)
            phi_delta = phi_map + pt.pi
            correction = pt.sin(inc) * pt.cos(phi_delta)
            v_obs = v_sys + v_rot * correction
            return v_obs

        # PyMC model
        with pm.Model() as model:
            # ------------------------------------------
            # prior distributions
            # ------------------------------------------
            # M200 prior: from SHMR
            M200_log_sigma = 0.25
            M200_log_mu = pt.log(M200_expect)
            M200_t = pm.LogNormal("M200", mu=M200_log_mu, sigma=M200_log_sigma)

            # c prior: log-normal prior
            c_log_mu = pt.log(c_expect)
            c_t = pm.LogNormal("c", mu=c_log_mu, sigma=0.25)

            # sigma_0 prior: log-normal (strictly positive scale)
            if self.inf_drift:
                sigma_0_t = pm.LogNormal("sigma_0", mu=pt.log(5.0), sigma=0.5)
            else:
                sigma_0_t = pm.Deterministic("sigma_0", pt.as_tensor_variable(0.0))

            # v_sys prior: normal prior around measured value
            v_sys_delta = 20.0
            v_sys_t = pm.TruncatedNormal("v_sys", mu=vel_sys, sigma=5.0, lower=vel_sys - v_sys_delta, upper=vel_sys + v_sys_delta)

            # inc prior: normal prior around measured value
            if self.inf_inc:
                inc_delta = pt.deg2rad(5.0)
                inc_t = pm.TruncatedNormal("inc", mu=inc_rad, sigma=0.1, lower=pt.maximum(0.0, inc_rad - inc_delta), upper=pt.minimum(pt.pi/2, inc_rad + inc_delta))
            else:
                inc_t = pm.Deterministic("inc", pt.as_tensor_variable(inc_rad))

            # phi_delta prior
            if self.inf_phi_delta:
                phi_delta_t = pm.TruncatedNormal("phi_delta", mu=0.0, sigma=pt.deg2rad(5.0), lower=-pt.deg2rad(10.0), upper=pt.deg2rad(10.0))
            else:
                phi_delta_t = pm.Deterministic("phi_delta", pt.as_tensor_variable(0.0))

            # Notice: Do not use vbecause it is strongly degenerate with other parameters.
            if self.inf_Re:
                Re_mu = r_max * 0.25
                Re_t = pm.LogNormal('Re', mu=pt.log(Re_mu), sigma=0.4)
            else:
                Re_t = pm.Deterministic("Re", pt.as_tensor_variable(Re))

            # f_bulge prior
            f_bulge_t = pm.Beta("f_bulge", alpha=1.2, beta=4.0)
            # f_bulge_t = pm.Uniform("f_bulge", lower=0.0, upper=0.2)

            # a prior
            a_mu = r_max * 0.1
            a_t = pm.LogNormal("a", mu=pt.log(a_mu), sigma=0.5)

            # sigma_obs_scale prior: to scale the measurement errors
            stderr_obs_valid_mean = np.nanmean(stderr_obs_valid)
            sigma_obs_scale_mu = float(np.sqrt(stderr_obs_valid_mean**2 + VEL_SYSTEM_ERROR**2))
            sigma_obs_scale_t = pm.LogNormal(
                "sigma_obs_scale",
                mu=pt.log(pt.as_tensor_variable(max(sigma_obs_scale_mu, 1e-6))),
                sigma=0.3,
            )

            # ------------------------------------------
            # deterministic relations
            # ------------------------------------------
            # V200: derived from M200
            V200_t = pm.Deterministic("V200", (10 * G_kpc_kms_Msun * Hz * M200_t) ** (1.0 / 3.0))

            r = radius_valid  # numpy array
            v_star_t = pm.Deterministic("v_star", pt.sqrt(v_star_sq_profile(r, Mstar, Re_t, f_bulge_t, a_t)))
            v_dm_t = pm.Deterministic("v_dm", pt.sqrt(v_dm_sq_profile(r, M200_t, c_t, V200_t)))
            v_drift_t = pm.Deterministic("v_drift", pt.sqrt(v_drift_sq_profile(r, sigma_0_t, Re_t)))
            v_rot_t = pm.Deterministic("v_rot", pt.sqrt(pt.maximum(1e-9, v_rot_sq_profile(v_dm_t, v_star_t, v_drift_t))))
            v_obs_model =  v_obs_project_profile(v_rot_t, v_sys_t, inc_t, phi_map_valid-phi_delta_t)

            # ------------------------------------------
            # likelihood: observed rotation velocities
            # ------------------------------------------

            # predicted observed velocities
            v_obs_mu_t = pm.Deterministic("v_obs_mu", pt.as_tensor_variable(v_obs_model))

            # Measurement error model: start from ivar-derived sigma (plus floor),
            # then allow a global scaling factor to absorb underestimated/overestimated uncertainties.
            # This is often more realistic than treating ivar as perfectly calibrated.
            sigma_obs_t = pm.Deterministic("sigma_obs", sigma_obs_scale_t * stderr_obs_valid)

            pm.Normal("v_obs", mu=v_obs_mu_t, sigma=sigma_obs_t, observed=vel_obs_valid)

            # ------------------------------------------
            # potential
            # ------------------------------------------

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
            # neg_term = pt.maximum(-v_rot_sq_t, 0.0)
            # tau_penalty = 1e-4
            # penalty_val = -1.0 * (neg_term**2) / (2.0 * tau_penalty**2)
            # pm.Potential("v_rot_sq_penalty", pt.sum(penalty_val))


            # MAP estimate (optional diagnostic)
            map_estimate = pm.find_MAP()
            if self.inf_debug:
                print("--- MAP estimate ---")
                print(f"M200    : {map_estimate['M200']:.3e} Msun")
                print(f"c       : {map_estimate['c']:.3f}")
                print("--------------------\n")

            # ------------------------------------------
            # sampling options & run
            # ------------------------------------------
            print(">>> Starting PyMC sampling (NUTS)... this may take time.\n")

            draws = 1000
            tune = 500
            chains = min(4, os.cpu_count())
            target_accept = 0.95

            if self.inf_debug:
                displaybar = True
            else:
                displaybar = False

            sampler = "nutpie" # 'nutpie', 'numpyro'
            init = "jitter+adapt_full" # 'jitter+adapt_diag', 'jitter+adapt_full'
            random_seed = 42
            trace = pm.sample(init=init, draws=draws, tune=tune, chains=chains, cores=chains,
                              nuts_sampler=sampler, target_accept=target_accept,
                              progressbar=displaybar,
                              random_seed=random_seed,
                              return_inferencedata=True, compute_convergence_checks=True)

            if self.inf_debug:
                print("\n\n")
                print(">>> Sampling completed.\n")

            pm.compute_log_likelihood(trace)
            ppc_idata = pm.sample_posterior_predictive(trace, random_seed=random_seed, extend_inferencedata=True)

        # ------------------------------------------
        # postprocess
        # ------------------------------------------
        # summary with diagnostics
        # variable in the posterior. Exclude it from the az.summary var_names list.
        var_names = ["M200", "c", "v_sys", "f_bulge", "a", "sigma_obs_scale"]
        if self.inf_Re:
            var_names.append("Re")
        if self.inf_drift:
            var_names.append("sigma_0")
        if self.inf_inc:
            var_names.append("inc")
        if self.inf_phi_delta:
            var_names.append("phi_delta")

        summary = az.summary(trace, var_names=var_names, round_to=3)

        M200_mean = float(summary.loc["M200", "mean"])
        c_mean = float(summary.loc["c", "mean"])
        sigma0_mean = float(summary.loc["sigma_0", "mean"]) if self.inf_drift else 0.0
        v_sys_mean = float(summary.loc["v_sys", "mean"])
        inc_mean = float(summary.loc["inc", "mean"]) if self.inf_inc else inc_rad
        phi_delta_mean = float(summary.loc["phi_delta", "mean"]) if self.inf_phi_delta else 0.0
        Re_mean = float(summary.loc["Re", "mean"]) if self.inf_Re else Re
        f_bulge_mean = float(summary.loc["f_bulge", "mean"])
        a_mean = float(summary.loc["a", "mean"])
        sigma_obs_scale_mean = float(summary.loc["sigma_obs_scale", "mean"])

        M200_sd = float(summary.loc["M200", "sd"])
        c_sd = float(summary.loc["c", "sd"])
        sigma0_sd = float(summary.loc["sigma_0", "sd"]) if self.inf_drift else 0.0
        v_sys_sd = float(summary.loc["v_sys", "sd"])
        inc_sd = float(summary.loc["inc", "sd"]) if self.inf_inc else 0.0
        phi_delta_sd = float(summary.loc["phi_delta", "sd"]) if self.inf_phi_delta else 0.0
        Re_sd = float(summary.loc["Re", "sd"]) if self.inf_Re else 0.0
        f_bulge_sd = float(summary.loc["f_bulge", "sd"])
        a_sd = float(summary.loc["a", "sd"])
        sigma_obs_scale_sd = float(summary.loc["sigma_obs_scale", "sd"])

        M200_r_hat = float(summary.loc["M200", "r_hat"])
        c_r_hat = float(summary.loc["c", "r_hat"])
        sigma0_r_hat = float(summary.loc["sigma_0", "r_hat"]) if self.inf_drift else 1.0
        v_sys_r_hat = float(summary.loc["v_sys", "r_hat"])
        inc_r_hat = float(summary.loc["inc", "r_hat"]) if self.inf_inc else 1.0
        phi_delta_r_hat = float(summary.loc["phi_delta", "r_hat"]) if self.inf_phi_delta else 1.0
        Re_r_hat = float(summary.loc["Re", "r_hat"]) if self.inf_Re else 1.0
        f_bulge_r_hat = float(summary.loc["f_bulge", "r_hat"])
        a_r_hat = float(summary.loc["a", "r_hat"])
        sigma_obs_scale_r_hat = float(summary.loc["sigma_obs_scale", "r_hat"])

        # extract posterior samples
        posterior = trace.posterior
        v_dm_samples = posterior["v_dm"].stack(samples=("chain", "draw")).values
        v_dm_mean = np.mean(v_dm_samples, axis=1)
        v_star_samples = posterior["v_star"].stack(samples=("chain", "draw")).values
        v_star_mean = np.mean(v_star_samples, axis=1)
        v_drift_samples = posterior["v_drift"].stack(samples=("chain", "draw")).values
        v_drift_mean = np.mean(v_drift_samples, axis=1)
        v_rot_samples = posterior["v_rot"].stack(samples=("chain", "draw")).values
        v_rot_mean = np.mean(v_rot_samples, axis=1)
        sigma_obs_samples = posterior["sigma_obs"].stack(samples=("chain", "draw")).values
        sigma_obs_mean = np.mean(sigma_obs_samples, axis=1)

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
        c_calc = self._calc_c_from_M200(M200_mean, h=H)

        # ------------------------------------------
        # residual diagnostics
        # ------------------------------------------
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

        # ------------------------------------------
        # posterior predictive chi^2 diagnostics
        # ------------------------------------------
        # Extract noise-free posterior mean (mu) per draw
        # Use the same InferenceData object as v_obs_ppc to keep draws aligned.
        v_obs_mu_raw = ppc_idata.posterior["v_obs_mu"].stack(sample=("chain", "draw")).values
        if v_obs_mu_raw.ndim != 2:
            raise ValueError(f"Unexpected v_obs_mu shape: {v_obs_mu_raw.shape}")
        if v_obs_mu_raw.shape[-1] == n_points:
            v_obs_mu = v_obs_mu_raw
        elif v_obs_mu_raw.shape[0] == n_points:
            v_obs_mu = v_obs_mu_raw.T
        else:
            raise ValueError(
                f"Posterior v_obs_mu shape {v_obs_mu_raw.shape} does not match n_points={n_points}."
            )

        # Extract per-draw sigma and standardize to (n_samples, n_points)
        if sigma_obs_samples.ndim != 2:
            raise ValueError(f"Unexpected sigma_obs shape: {sigma_obs_samples.shape}")
        if sigma_obs_samples.shape[-1] == n_points:
            sigma_obs = sigma_obs_samples
        elif sigma_obs_samples.shape[0] == n_points:
            sigma_obs = sigma_obs_samples.T
        else:
            raise ValueError(
                f"Posterior sigma_obs shape {sigma_obs_samples.shape} does not match n_points={n_points}."
            )

        # Ensure we align sample counts between mu/sigma and posterior predictive draws
        n_samp = int(min(v_obs_ppc.shape[0], v_obs_mu.shape[0], sigma_obs.shape[0]))
        v_obs_ppc_use = v_obs_ppc[:n_samp, :]
        v_obs_mu_use = v_obs_mu[:n_samp, :]
        sigma_obs_use = sigma_obs[:n_samp, :]

        sigma = np.where(np.isfinite(sigma_obs_use) & (sigma_obs_use > 0), sigma_obs_use, np.nan)

        # Discrepancy measures
        # T_obs(theta) = sum(((y - mu_theta)/sigma)^2)
        # T_rep(theta) = sum(((y_rep - mu_theta)/sigma)^2)
        resid_obs = (vel_obs_valid[None, :] - v_obs_mu_use) / sigma

        # mask invalid points (if any) in sigma or data
        valid_cols = np.isfinite(vel_obs_valid) & np.all(np.isfinite(sigma), axis=0)
        resid_obs = resid_obs[:, valid_cols]

        chi2_obs_draws = np.sum(resid_obs**2, axis=1)

        # Summaries
        chi2_obs_mean = float(np.mean(chi2_obs_draws))

        # parameters in model: M200_log, c_log, sigma_0, v_sys, inc, phi_delta, Re, f_bulge, a, sigma_obs
        params_num = 10
        if not self.inf_drift:
            params_num -= 1
        if not self.inf_Re:
            params_num -= 1
        if not self.inf_inc:
            params_num -= 1
        if not self.inf_phi_delta:
            params_num -= 1
        dof = int(max(np.sum(valid_cols) - params_num, 1))
        # Reduced Chi-squared (use posterior mean chi2)
        redchi = float(chi2_obs_mean / dof)

        # ------------------------------------------
        # PPC discrepancy for Normal likelihood: deviance = -2 * log p(y|theta)
        # ------------------------------------------
        sigma_use = sigma[:, valid_cols]
        const_n = -0.5 * np.log(2.0 * np.pi)

        z_obs = (vel_obs_valid[None, valid_cols] - v_obs_mu_use[:, valid_cols]) / sigma_use
        z_rep = (v_obs_ppc_use[:, valid_cols] - v_obs_mu_use[:, valid_cols]) / sigma_use

        logp_obs = const_n - np.log(sigma_use) - 0.5 * (z_obs**2)
        logp_rep = const_n - np.log(sigma_use) - 0.5 * (z_rep**2)

        dev_obs_draws = -2.0 * np.sum(logp_obs, axis=1)
        dev_rep_draws = -2.0 * np.sum(logp_rep, axis=1)
        dev_ppc_p = float(np.mean(dev_rep_draws > dev_obs_draws))

        # For legacy residual metrics, keep a point estimate based on mu averaged across draws
        v_obs_mean = np.mean(v_obs_mu_use, axis=0)

        # ------------------------------------------
        # residuals
        # ------------------------------------------
        mask = np.isfinite(vel_obs_valid) & np.isfinite(v_obs_mean)
        res_obs = vel_obs_valid - v_obs_mean
        res_norm = np.full_like(res_obs, np.nan, dtype=float)
        sigma_obs_use = np.nanmean(sigma_obs_use, axis=0)
        res_norm[mask] = res_obs[mask] / sigma_obs_use[mask]

        res_map = res_obs[mask]
        rmse = float(np.sqrt(np.mean(res_map**2)))
        nrmse = float(rmse / np.mean(np.abs(vel_obs_valid[mask])))

        # ---------------------
        # Inference summary info
        # ---------------------
        if self.inf_debug:
            print("\n------------ Infer Dark Matter NFW (PyMC) ------------")
            print("--- Summary ---")
            print(summary)
            print("--- LOO ---")
            print(f"{model_loo}")
            print(f"--- median estimates ---")
            print(f" Infer M200         : {M200_mean:.3e} ± {M200_sd:.3e} Msun ({M200_sd/M200_mean:.2%})")
            print(f" Infer c            : {c_mean:.3f} ± {c_sd:.3f} ({c_sd/c_mean:.2%})")
            print(f" Infer sigma_0      : {sigma0_mean:.3f} ± {sigma0_sd:.3f} km/s ({sigma0_sd/sigma0_mean:.2%})") if self.inf_drift else None
            print(f" Infer v_sys        : {v_sys_mean:.3f} ± {v_sys_sd:.3f} km/s ({v_sys_sd/max(v_sys_mean, 1e-3):.2%})")
            print(f" Infer inc          : {np.degrees(inc_mean):.3f} ± {np.degrees(inc_sd):.3f} deg ({inc_sd/max(inc_mean, 1e-3):.2%})") if self.inf_inc else None
            print(f" Infer phi_delta    : {np.degrees(phi_delta_mean):.3f} ± {np.degrees(phi_delta_sd):.3f} deg ({phi_delta_sd/max(phi_delta_mean, 1e-3):.2%})") if self.inf_phi_delta else None
            print(f" Infer Re           : {Re_mean:.3f} ± {Re_sd:.3f} kpc ({Re_sd/max(Re_mean, 1e-3):.2%})") if self.inf_Re else None
            print(f" Infer f_bulge      : {f_bulge_mean:.3f} ± {f_bulge_sd:.3f} ({f_bulge_sd/max(f_bulge_mean, 1e-3):.2%})")
            print(f" Infer a            : {a_mean:.3f} ± {a_sd:.3f} kpc ({a_sd/max(a_mean, 1e-3):.2%})")
            print(f" Infer sigma_obs_scale : {sigma_obs_scale_mean:.3f} ± {sigma_obs_scale_sd:.3f} ({sigma_obs_scale_sd/sigma_obs_scale_mean:.2%})")
            print(f"--- caculate ---")
            print(f" Calc: V200         : {V200_calc:.3f} km/s")
            print(f" Calc: r200         : {r200_calc:.3f} kpc")
            print(f" Calc: c            : {c_calc:.3f}")
            print(f" Calc: v_sys        : {vel_sys:.3f} km/s")
            print(f" Calc: inc          : {np.degrees(inc_rad):.3f} deg")
            print(f" Stellar Mass       : {Mstar:.3e} Msun")
            print("--- diagnostics ---")
            print(f" Reduced Chi        : {redchi:.3f}")
            print(f" Deviance PPC p-val : {dev_ppc_p:.3f}")
            print(f" NRMSE              : {nrmse:.3f}")
            print("------------------------------------------------------------\n")

        # ---------------------
        # plot
        # ---------------------
        if self.plot_enable:
            # trace plots
            # az.plot_trace(trace, var_names=var_names)

            # corner plot
            az.rcParams['plot.max_subplots'] = 50
            az.plot_pair(trace, var_names=var_names, kind='kde', marginals=True)
            plt.tight_layout()
            plt.show()


        if M200_r_hat < INFER_RHAT_THRESHOLD and \
            c_r_hat < INFER_RHAT_THRESHOLD and \
            sigma0_r_hat < INFER_RHAT_THRESHOLD and \
            v_sys_r_hat < INFER_RHAT_THRESHOLD and \
            inc_r_hat < INFER_RHAT_THRESHOLD and \
            phi_delta_r_hat < INFER_RHAT_THRESHOLD and \
            Re_r_hat < INFER_RHAT_THRESHOLD and \
            f_bulge_r_hat < INFER_RHAT_THRESHOLD and \
            a_r_hat < INFER_RHAT_THRESHOLD and \
            sigma_obs_scale_r_hat < INFER_RHAT_THRESHOLD:
            success = True
        else:
            print("Inference did not converge (R-hat too high).")
            success = False

        inf_result = {
            'radius': radius_valid,
            'v_rot': v_rot_mean,
            'v_dm': v_dm_mean,
            'v_star': v_star_mean,
            'v_drift': v_drift_mean,
            'sigma_obs': sigma_obs_mean,
            'res_obs': res_obs,
        }

        inf_params = {
            'result': 'success' if success else 'failure',
            'M200': M200_mean,
            'M200_std': M200_sd,
            'M200_r_hat': M200_r_hat,
            'c': c_mean,
            'c_std': c_sd,
            'c_r_hat': c_r_hat,
            'nrmse': nrmse,
            'chi2_red': redchi,
            'dev_ppc_p': dev_ppc_p,
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

    def set_inf_debug(self, inf_debug: bool) -> None:
        self.inf_debug = inf_debug
        return

    def set_stellar_util(self, stellar_util: Stellar) -> None:
        self.stellar_util = stellar_util
        return

    def inf_dm_nfw(self, radius_obs: np.ndarray, vel_obs: np.ndarray, ivar_obs: np.ndarray, vel_sys: float, inc_rad: float, phi_map: np.ndarray):
        return self._inf_dm_nfw_pymc(radius_obs, vel_obs, ivar_obs, vel_sys, inc_rad, phi_map)

