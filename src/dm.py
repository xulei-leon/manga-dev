'''
Dark Matter NFW profile inference module

Use the following equation to fit DM profile:
Vobs^2(r)   =  Vstar^2(r) + Vdm^2(r) - Vdrift^2(r)
Vdm^2(r) = ((10 * G * H(z) * M200) ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
Vstar^2(r)  = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
Vdrift^2{r) = 2 * sigma_0^2 * (r / R_d)
'''

from math import log
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

H = 1 #0.674  # assuming H0 = 67.4 km/s/Mpc
G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun

INFER_RHAT_THRESHOLD = 1.05
VEL_SYSTEM_ERROR = 5.0  # km/s, floor error as systematic uncertainty in velocity measurements


def _get_arviz_api():
    if hasattr(az, "preview"):
        preview = az.preview
        if hasattr(preview, "summary"):
            return preview
    return az


def _set_arviz_ci_defaults():
    try:
        if "stats.ci_prob" in az.rcParams:
            az.rcParams["stats.ci_prob"] = 0.94
        if "stats.ci_kind" in az.rcParams:
            az.rcParams["stats.ci_kind"] = "hdi"
    except Exception:
        pass


def _get_posterior_dataset(idata):
    if hasattr(idata, "posterior"):
        return idata.posterior
    try:
        posterior = idata["posterior"]
        if hasattr(posterior, "dataset"):
            posterior = posterior.dataset
        return posterior
    except Exception as exc:
        raise AttributeError("posterior group not found on inference data") from exc


def _get_ppc_dataset(idata):
    if hasattr(idata, "posterior_predictive"):
        return idata.posterior_predictive
    try:
        ppc = idata["posterior_predictive"]
        if hasattr(ppc, "dataset"):
            ppc = ppc.dataset
        return ppc
    except Exception as exc:
        raise AttributeError("posterior_predictive group not found on inference data") from exc


class DmNfw:
    drpall_util: DrpallUtil
    PLATE_IFU: str
    plot_enable: bool
    inf_debug: bool = False
    inf_inc: bool = False # Notice: Infer inc may be degenerate with c
    inf_phi_delta: bool = False

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
    def _calc_Mstar_from_Mhalo(self, M200: float, M1=10**11.59, N=0.0351, beta=1.376, gamma=0.608):
        x = M200 / M1
        f = 2.0 * N / (x**(-beta) + x**(gamma))
        return f * M200

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
        _Mstar_elpetro, _Mstar_sersic = self.drpall_util.get_stellar_mass(self.PLATE_IFU)
        print (f"Stellar mass from DRPALL: Mstar_elpetro={_Mstar_elpetro:.2e} Msun, Mstar_sersic={_Mstar_sersic:.2e} Msun")
        Mstar_expect = _Mstar_elpetro if _Mstar_elpetro is not None else _Mstar_sersic

        z = self._get_z()

        Mmin=1e9
        Mmax=1e15
        M200_expect = self._calc_M200_from_Mstar(Mstar_expect, Mmin=Mmin, Mmax=Mmax)  # expected M200 from Mstar

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

        # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
        def c_from_M200(M200, h):
            M_pivot_h_inv = 2e12 # in Msun/h
            mass_ratio = M200 / (M_pivot_h_inv / h)
            return 5.74 * (mass_ratio)**(-0.097)

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
            Mstar_log_mu = pt.log10(Mstar_expect)
            Mstar_log_t = pm.Normal("Mstar_log10", mu=Mstar_log_mu, sigma=0.1)
            Mstar_t = pm.Deterministic("Mstar", 10**Mstar_log_t)

            # M200 prior:
            # M200 mu from SHMR relation
            M200_log_mu = pt.log10(M200_expect)
            M200_log_t = pm.Normal("M200_log10", mu=M200_log_mu, sigma=0.3)  # 0.3 dex scatter in SHMR
            M200_t = pm.Deterministic("M200", 10**M200_log_t)

            # c prior: log-normal prior
            # soft constraint on c from M200-c relation
            # log10(c) ~ N(a + b * (log10(M200) - 12), sigma_dex)
            intercept = pm.Normal("intercept", mu=0.9, sigma=0.1)
            slop = pm.Normal("slop", mu=-0.097, sigma=0.02)
            sigma_c = pm.HalfNormal("sigma_c", sigma=0.15)
            c_log_mu = intercept + slop * (M200_log_t - 12.0)
            c_log_t = pm.Normal("c_log10", mu=c_log_mu, sigma=sigma_c)
            c_t = pm.Deterministic("c", 10**c_log_t)

            # sigma_0 prior: log-normal (strictly positive scale)
            sigma_0_t = pm.LogNormal("sigma_0", mu=pt.log(5.0), sigma=0.5)

            # v_sys prior: normal prior around measured value
            v_sys_delta = 20.0
            v_sys_t = pm.TruncatedNormal("v_sys", mu=vel_sys, sigma=5.0, lower=vel_sys - v_sys_delta, upper=vel_sys + v_sys_delta)

            # inc prior: normal prior around measured value
            if self.inf_inc:
                inc_t = pm.Normal("inc", mu=inc_rad, sigma=pt.deg2rad(2.0))
            else:
                inc_t = pm.Deterministic("inc", pt.as_tensor_variable(inc_rad))

            # phi_delta prior
            if self.inf_phi_delta:
                phi_delta_t = pm.TruncatedNormal("phi_delta", mu=0.0, sigma=pt.deg2rad(5.0), lower=-pt.deg2rad(10.0), upper=pt.deg2rad(10.0))
            else:
                phi_delta_t = pm.Deterministic("phi_delta", pt.as_tensor_variable(0.0))


            # Perior for Re: log-normal prior
            Re_mu = r_max * 0.25
            Re_t = pm.LogNormal('Re', mu=pt.log(Re_mu), sigma=0.3*log(10)) # 0.3 dex scatter in Re


            # f_bulge prior
            f_bulge_t = pm.Beta("f_bulge", alpha=1.2, beta=4.0)
            # f_bulge_t = pm.Uniform("f_bulge", lower=0.0, upper=0.2)

            # a prior
            a_mu = r_max * 0.1
            a_t = pm.LogNormal("a", mu=pt.log(a_mu), sigma=0.5)

            # sigma_scale prior: to scale the measurement errors
            stderr_obs_valid_mean = np.nanmean(stderr_obs_valid)
            sigma_scale_mu = float(np.sqrt(stderr_obs_valid_mean**2 + VEL_SYSTEM_ERROR**2))
            sigma_scale_t = pm.LogNormal(
                "sigma_scale",
                mu=pt.log(pt.as_tensor_variable(max(sigma_scale_mu, 1e-6))),
                sigma=0.3,
            )

            # ------------------------------------------
            # deterministic relations
            # ------------------------------------------
            # V200: derived from M200
            V200_t = pm.Deterministic("V200", (10 * G_kpc_kms_Msun * Hz * M200_t) ** (1.0 / 3.0))

            r = radius_valid  # numpy array
            v_star_t = pm.Deterministic("v_star", pt.sqrt(v_star_sq_profile(r, Mstar_t, Re_t, f_bulge_t, a_t)))
            v_dm_t = pm.Deterministic("v_dm", pt.sqrt(v_dm_sq_profile(r, M200_t, c_t, V200_t)))
            v_drift_t = pm.Deterministic("v_drift", pt.sqrt(v_drift_sq_profile(r, sigma_0_t, Re_t)))
            v_rot_t = pm.Deterministic("v_rot", pt.sqrt(pt.maximum(1e-9, v_rot_sq_profile(v_dm_t, v_star_t, v_drift_t))))
            v_obs_model_t = pm.Deterministic("v_obs_model", v_obs_project_profile(v_rot_t, v_sys_t, inc_t, phi_map_valid-phi_delta_t))

            # ------------------------------------------
            # likelihood: observed rotation velocities
            # ------------------------------------------

            # Measurement error model: start from ivar-derived sigma (plus floor),
            # then allow a global scaling factor to absorb underestimated/overestimated uncertainties.
            # This is often more realistic than treating ivar as perfectly calibrated.
            sigma_obs_t = pm.Deterministic("sigma_obs", sigma_scale_t * stderr_obs_valid)

            pm.Normal("v_obs", mu=v_obs_model_t, sigma=sigma_obs_t, observed=vel_obs_valid)

            # ------------------------------------------
            # potential
            # ------------------------------------------

            # Penalize regions where v_rot_sq < 0 to discourage unphysical solutions.
            #
            # neg_term = pt.maximum(-v_rot_sq_t, 0.0)
            # tau_penalty = 1e-4
            # penalty_val = -1.0 * (neg_term**2) / (2.0 * tau_penalty**2)
            # pm.Potential("v_rot_sq_penalty", pt.sum(penalty_val))


            # MAP estimate (optional diagnostic)
            map_estimate = pm.find_MAP()

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

            try:
                pm.compute_log_likelihood(trace)
            except Exception as exc:
                print(f"Warning: compute_log_likelihood failed: {exc}")

            ppc_idata = pm.sample_posterior_predictive(trace, random_seed=random_seed, extend_inferencedata=True)

        # ------------------------------------------
        # postprocess
        # ------------------------------------------
        # summary with diagnostics
        # variable in the posterior. Exclude it from the az.summary var_names list.
        var_names = ["M200", "c", "v_sys", "Re", "f_bulge", "a", "sigma_scale", "Mstar", "sigma_0"]
        if self.inf_inc:
            var_names.append("inc")
        if self.inf_phi_delta:
            var_names.append("phi_delta")

        az_api = _get_arviz_api()
        _set_arviz_ci_defaults()

        summary = az_api.summary(trace, var_names=var_names, round_to=3, stat_funcs={"median": np.median})

        Mstar_median = float(summary.loc["Mstar", "median"])
        M200_median = float(summary.loc["M200", "median"])
        c_median = float(summary.loc["c", "median"])
        sigma0_median = float(summary.loc["sigma_0", "median"])
        v_sys_median = float(summary.loc["v_sys", "median"])
        inc_median = float(summary.loc["inc", "median"]) if self.inf_inc else inc_rad
        phi_delta_median = float(summary.loc["phi_delta", "median"]) if self.inf_phi_delta else 0.0
        Re_median = float(summary.loc["Re", "median"])
        f_bulge_median = float(summary.loc["f_bulge", "median"])
        a_median = float(summary.loc["a", "median"])
        sigma_scale_median = float(summary.loc["sigma_scale", "median"])

        Mstar_sd = float(summary.loc["Mstar", "sd"])
        M200_sd = float(summary.loc["M200", "sd"])
        c_sd = float(summary.loc["c", "sd"])
        sigma0_sd = float(summary.loc["sigma_0", "sd"])
        v_sys_sd = float(summary.loc["v_sys", "sd"])
        inc_sd = float(summary.loc["inc", "sd"]) if self.inf_inc else 0.0
        phi_delta_sd = float(summary.loc["phi_delta", "sd"]) if self.inf_phi_delta else 0.0
        Re_sd = float(summary.loc["Re", "sd"])
        f_bulge_sd = float(summary.loc["f_bulge", "sd"])
        a_sd = float(summary.loc["a", "sd"])
        sigma_scale_sd = float(summary.loc["sigma_scale", "sd"])

        success = True
        for var in var_names:
            r_hat = float(summary.loc[var, "r_hat"])
            if r_hat > INFER_RHAT_THRESHOLD:
                print(f"Warning: R-hat for variable {var} is {r_hat:.3f} > {INFER_RHAT_THRESHOLD}, indicating potential non-convergence.")
                success = False


        # extract posterior samples
        posterior = _get_posterior_dataset(trace)
        flat_trace = posterior.stack(sample=("chain", "draw"))
        v_obs_samples = flat_trace["v_obs_model"].values
        v_dm_samples = flat_trace["v_dm"].values
        v_star_samples = flat_trace["v_star"].values
        v_drift_samples = flat_trace["v_drift"].values
        v_rot_samples = flat_trace["v_rot"].values
        sigma_obs_samples = flat_trace["sigma_obs"].values
        Mstar_samples = flat_trace["Mstar"].values
        M200_samples = flat_trace["M200"].values
        c_samples = flat_trace["c"].values
        v_sys_samples = flat_trace["v_sys"].values
        Re_samples = flat_trace["Re"].values
        f_bulge_samples = flat_trace["f_bulge"].values
        a_samples = flat_trace["a"].values
        sigma_scale_samples = flat_trace["sigma_scale"].values
        inc_samples = flat_trace["inc"].values if self.inf_inc else None
        phi_delta_samples = flat_trace["phi_delta"].values if self.inf_phi_delta else None
        sigma0_samples = flat_trace["sigma_0"].values


        # ------------------------------------------
        # Representative Samples
        # ------------------------------------------
        # Method 1: Closest to median profile
        # v_rot_median = np.median(v_rot_samples, axis=1)
        # # Calculate distance of each sample from the median profile
        # v_rot_distance = np.linalg.norm(v_rot_samples - v_rot_median[:, None], axis=0)
        # best_idx = np.argmin(v_rot_distance)

        # Method 2: Maximum A Posteriori
        lp_stacked = trace.sample_stats["logp"].stack(sample=("chain", "draw"))
        best_idx = int(lp_stacked.argmax("sample").values)

        # Select the sample profile that is closest to median
        v_obs_best = v_obs_samples[:, best_idx]
        v_rot_best = v_rot_samples[:, best_idx]
        v_star_best = v_star_samples[:, best_idx]
        v_dm_best = v_dm_samples[:, best_idx]
        v_drift_best = v_drift_samples[:, best_idx]
        sigma_obs_best = sigma_obs_samples[:, best_idx]

        Mstar_best = Mstar_samples[best_idx]
        M200_best = M200_samples[best_idx]
        c_best = c_samples[best_idx]
        v_sys_best = v_sys_samples[best_idx]
        Re_best = Re_samples[best_idx]
        f_bulge_best = f_bulge_samples[best_idx]
        a_best = a_samples[best_idx]
        sigma_scale_best = sigma_scale_samples[best_idx]
        inc_best = inc_samples[best_idx] if self.inf_inc else inc_rad
        phi_delta_best = phi_delta_samples[best_idx] if self.inf_phi_delta else 0.0
        sigma0_best = sigma0_samples[best_idx]

        V200_calc = self._calc_V200_from_M200(M200_best, z)
        r200_calc = self._calc_r200_from_V200(V200_calc, z)
        c_calc = c_from_M200(M200_best, h=H)

        # LOO
        # Request pointwise LOO to include pareto_k values for diagnostics
        model_loo = az_api.loo(trace, pointwise=True)
        elpd_loo_est = float(model_loo.elpd_loo)
        elpd_loo_se = float(model_loo.se)
        p_loo = float(model_loo.p_loo)
        max_k = float(model_loo.pareto_k.max().values)
        mean_k = float(model_loo.pareto_k.mean().values)
        good_k_fraction = float(np.sum(model_loo.pareto_k.values < 0.7) / len(model_loo.pareto_k.values))


        # ------------------------------------------
        # posterior predictive checks (dev_ppc_p)
        # ------------------------------------------
        ppc = _get_ppc_dataset(ppc_idata)
        v_obs_ppc = ppc["v_obs"].stack(sample=("chain", "draw")).values
        n_points = len(radius_valid)

        # Standardize to (n_samples, n_points)
        if v_obs_ppc.shape[-1] != n_points and v_obs_ppc.shape[0] == n_points:
            v_obs_ppc = v_obs_ppc.T

        # Prepare model samples (n_samples, n_points)
        v_obs_model = v_obs_samples.T
        sigma_obs = sigma_obs_samples.T

        # Align samples
        n_samp = min(v_obs_ppc.shape[0], v_obs_model.shape[0], sigma_obs.shape[0])
        y_rep = v_obs_ppc[:n_samp, :]
        mu = v_obs_model[:n_samp, :]
        sigma = sigma_obs[:n_samp, :]

        # Mask invalid points
        valid_cols = np.isfinite(vel_obs_valid) & np.all(np.isfinite(sigma) & (sigma > 0), axis=0)

        # Calculate Deviance = -2 * logp
        # We only use valid columns for the sum
        y_obs_valid = vel_obs_valid[valid_cols]
        y_rep_valid = y_rep[:, valid_cols]
        mu_valid = mu[:, valid_cols]
        sigma_valid = sigma[:, valid_cols]

        const_term = -0.5 * np.log(2.0 * np.pi) - np.log(sigma_valid)
        logp_obs = const_term - 0.5 * ((y_obs_valid - mu_valid) / sigma_valid)**2
        logp_rep = const_term - 0.5 * ((y_rep_valid - mu_valid) / sigma_valid)**2

        dev_obs = -2.0 * np.sum(logp_obs, axis=1)
        dev_rep = -2.0 * np.sum(logp_rep, axis=1)
        dev_ppc_p = float(np.mean(dev_rep > dev_obs))

        # ------------------------------------------
        # residuals
        # ------------------------------------------
        mask = np.isfinite(vel_obs_valid) & np.isfinite(v_obs_best)
        res_obs_best = vel_obs_valid - v_obs_best
        res_norm_best = np.full_like(res_obs_best, np.nan, dtype=float)

        # Use the sigma corresponding to the best fit
        res_norm_best[mask] = res_obs_best[mask] / sigma_obs_best[mask]

        res_map_best = res_obs_best[mask]
        rmse_best = float(np.sqrt(np.mean(res_map_best**2)))
        nrmse_best = float(rmse_best / np.mean(np.abs(vel_obs_valid[mask])))

        # ------------------------------------------
        # Recalculate Reduced Chi2 for the best fit
        # ------------------------------------------
        # parameters in model: M200, c, sigma_0, v_sys, inc, phi_delta, Re, f_bulge, a, sigma_obs
        params_num = 10
        if not self.inf_inc:
            params_num -= 1
        if not self.inf_phi_delta:
            params_num -= 1
        dof = int(max(np.sum(valid_cols) - params_num, 1))

        chi2_best = np.sum(res_norm_best[mask]**2)
        redchi_best = float(chi2_best / dof)

        # ---------------------
        # Inference summary info
        # ---------------------
        if self.inf_debug:
            print("\n------------ Infer Dark Matter NFW (PyMC) ------------")
            print("--- Summary ---")
            print(summary)
            print("--- LOO ---")
            print(f"{model_loo}")
            print("--- Expectation ---")
            print(f"Mstar Expect        : {Mstar_expect:.3e} Msun")
            print(f"M200 Expect         : {M200_expect:.3e} Msun")
            print("--- MAP estimate ---")
            print(f"Mstar Estimate      : {map_estimate['Mstar']:.3e} Msun")
            print(f"M200 Estimate       : {map_estimate['M200']:.3e} Msun")
            print(f"c Estimate          : {map_estimate['c']:.3f}")
            print(f"--- Best ---")
            print(f" Best Mstar         : {Mstar_best:.3e} Msun")
            print(f" Best M200          : {M200_best:.3e} Msun")
            print(f" Best c             : {c_best:.3f}")
            print(f" Best sigma_0       : {sigma0_best:.3f} km/s")
            print(f" Best v_sys         : {v_sys_best:.3f} km/s")
            print(f" Best inc           : {np.degrees(inc_best):.3f} deg") if self.inf_inc else None
            print(f" Best phi_delta     : {np.degrees(phi_delta_best):.3f} deg") if self.inf_phi_delta else None
            print(f" Best Re            : {Re_best:.3f} kpc")
            print(f" Best f_bulge       : {f_bulge_best:.3f}")
            print(f" Best a             : {a_best:.3f} kpc")
            print(f" Best sigma_scale   : {sigma_scale_best:.3f}")
            print(f"--- median estimates ---")
            print(f" Median Mstar       : {Mstar_median:.3e} ± {Mstar_sd:.3e} Msun ({Mstar_sd/max(Mstar_median, 1e-12):.2%})")
            print(f" Median M200        : {M200_median:.3e} ± {M200_sd:.3e} Msun ({M200_sd/max(M200_median, 1e-12):.2%})")
            print(f" Median c           : {c_median:.3f} ± {c_sd:.3f} ({c_sd/max(c_median, 1e-12):.2%})")
            print(f" Median sigma_0     : {sigma0_median:.3f} ± {sigma0_sd:.3f} km/s ({sigma0_sd/max(sigma0_median, 1e-12):.2%})")
            print(f" Median v_sys       : {v_sys_median:.3f} ± {v_sys_sd:.3f} km/s ({v_sys_sd/max(v_sys_median, 1e-12):.2%})")
            print(f" Median inc         : {np.degrees(inc_median):.3f} ± {np.degrees(inc_sd):.3f} deg ({inc_sd/max(inc_median, 1e-12):.2%})") if self.inf_inc else None
            print(f" Median phi_delta   : {np.degrees(phi_delta_median):.3f} ± {np.degrees(phi_delta_sd):.3f} deg ({phi_delta_sd/max(phi_delta_median, 1e-12):.2%})") if self.inf_phi_delta else None
            print(f" Median Re          : {Re_median:.3f} ± {Re_sd:.3f} kpc ({Re_sd/max(Re_median, 1e-12):.2%})")
            print(f" Median f_bulge     : {f_bulge_median:.3f} ± {f_bulge_sd:.3f} ({f_bulge_sd/max(f_bulge_median, 1e-12):.2%})")
            print(f" Median a           : {a_median:.3f} ± {a_sd:.3f} kpc ({a_sd/max(a_median, 1e-12):.2%})")
            print(f" Median sigma_scale : {sigma_scale_median:.3f} ± {sigma_scale_sd:.3f} ({sigma_scale_sd/max(sigma_scale_median, 1e-12):.2%})")
            print(f"--- caculate ---")
            print(f" Calc: V200         : {V200_calc:.3f} km/s")
            print(f" Calc: r200         : {r200_calc:.3f} kpc")
            print(f" Calc: c            : {c_calc:.3f}")
            print(f" Calc: v_sys        : {vel_sys:.3f} km/s")
            print(f" Calc: inc          : {np.degrees(inc_rad):.3f} deg")
            print(f" Stellar Mass       : {Mstar_expect:.3e} Msun")
            print("--- diagnostics ---")
            print(f" Reduced Chi (Best) : {redchi_best:.3f}")
            print(f" NRMSE (Best)       : {nrmse_best:.3f}")
            print(f" Deviance PPC p-val : {dev_ppc_p:.3f}")
            print("------------------------------------------------------------\n")

        # ---------------------
        # plot
        # ---------------------
        if self.plot_enable:
            # trace plots
            # az.plot_trace(trace, var_names=var_names)

            # corner plot
            az.rcParams['plot.max_subplots'] = 100
            az_api.plot_pair(trace, var_names=var_names, kind='kde', marginals=True)
            plt.gcf().set_size_inches(12, 10)
            plt.tight_layout()
            plt.show()


        inf_result = {
            'radius': radius_valid,
            'v_rot': v_rot_best,
            'v_dm': v_dm_best,
            'v_star': v_star_best,
            'v_drift': v_drift_best,
            'sigma_obs': sigma_obs_best,
            'res_obs': res_obs_best,
        }

        inf_params = {
            'result': 'success' if success else 'failure',
            'Mstar': Mstar_best,
            'Mstar_std': Mstar_sd,
            'M200': M200_best,
            'M200_std': M200_sd,
            'c': c_best,
            'c_std': c_sd,
            'nrmse': nrmse_best,
            'chi2_red': redchi_best,
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

    def inf_dm_nfw(self, radius_obs: np.ndarray, vel_obs: np.ndarray, ivar_obs: np.ndarray, vel_sys: float, inc_rad: float, phi_map: np.ndarray):
        return self._inf_dm_nfw_pymc(radius_obs, vel_obs, ivar_obs, vel_sys, inc_rad, phi_map)

