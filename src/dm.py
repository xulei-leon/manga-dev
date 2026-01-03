from pathlib import Path

from re import M, S
import numpy as np
from scipy.optimize import brentq
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
    def _inf_dm_nfw_pymc(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray):
        # ---------------------
        # 1) data selection / precompute
        # ---------------------
        vel_rot = np.where(vel_rot < 0, np.nan, vel_rot)
        radius = np.where(radius < 0, np.nan, radius)

        valid_mask = (np.isfinite(vel_rot) & np.isfinite(radius) & np.isfinite(vel_rot_err) &
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
        c_min, c_max = (1.0, 50.0)
        logc_min, logc_max = np.log10(c_min), np.log10(c_max)

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
        def v_rot_sq_profile(r_arr, M200, c, sigma_0, v_star_sq, V200):
            v_dm = v_dm_sq_profile(r_arr, M200, c, V200)
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
            M200_log_sigma = 0.5
            M200_log_t = pm.Normal("M200_log", mu=M200_log_mu, sigma=M200_log_sigma)  # auxiliary variable for prior
            # student-t prior for robustness
            # M200_log_t = pm.StudentT("M200_log", nu=3, mu=M200_log_mu, sigma=M200_log_sigma)

            # c prior: log-normal prior
            log_c_mu = np.log10(5.0)
            c_log_t = pm.TruncatedNormal("c_log", mu=log_c_mu, sigma=0.5, lower=logc_min, upper=logc_max)

            # sigma_0 prior: half normal
            # sigma_0 > 0
            sigma_0_t = pm.HalfNormal("sigma_0", sigma=10.0)

            # ---------------------
            # deterministic relations
            # ---------------------
            # V200: derived from M200
            M200_t = pm.Deterministic("M200", 10 ** M200_log_t)
            c_t = pm.Deterministic("c", 10 ** c_log_t)
            V200_t = pm.Deterministic("V200", (10 * G_kpc_kms_Msun * Hz * M200_t) ** (1.0 / 3.0))

            r = radius_valid  # numpy array
            v_dm_sq = pm.Deterministic("v_dm_sq", v_dm_sq_profile(r, M200_t, c_t, V200_t))
            v_drift_sq = pm.Deterministic("v_drift_sq", v_drift_sq_profile(r, sigma_0_t))
            v_rot_sq = pm.Deterministic("v_rot_sq", v_rot_sq_profile(r, M200_t, c_t, sigma_0_t, v_star_sq, V200_t))

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
            # c_expected = 10.0 * (M200_t / 1e12)**(-0.1)
            # sigma_logc = 0.2  # dex
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
            if self.plot_enable:
                ppc = pm.sample_posterior_predictive(trace, var_names=["v_rot_obs"], random_seed=42, extend_inferencedata=True)

        # ---------------------
        # 5) postprocess
        # ---------------------

        # summary with diagnostics
        summary = az.summary(trace, var_names=["M200", "c", "sigma_0"], round_to=3)
        M200_mean = float(summary.loc["M200", "mean"])
        c_mean = float(summary.loc["c", "mean"])
        sigma0_mean = float(summary.loc["sigma_0", "mean"])
        M200_sd = float(summary.loc["M200", "sd"])
        c_sd = float(summary.loc["c", "sd"])
        sigma0_sd = float(summary.loc["sigma_0", "sd"])
        M200_r_hat = float(summary.loc["M200", "r_hat"])
        c_r_hat = float(summary.loc["c", "r_hat"])
        sigma0_r_hat = float(summary.loc["sigma_0", "r_hat"])

        # LOO
        # Request pointwise LOO to include pareto_k values for diagnostics
        model_loo = az.loo(trace, pointwise=True)
        elpd_loo_est = float(model_loo.elpd_loo)
        elpd_loo_se = float(model_loo.se)
        p_loo = float(model_loo.p_loo)
        max_k = float(model_loo.pareto_k.max().values)
        mean_k = float(model_loo.pareto_k.mean().values)
        good_k_fraction = float(np.sum(model_loo.pareto_k.values < 0.7) / len(model_loo.pareto_k.values))


        # the posterior mean/std was used as expected values
        # M200_mean = float(trace.posterior["M200"].mean().values)
        # c_mean = float(trace.posterior["c"].mean().values)
        # sigma0_mean = float(trace.posterior["sigma_0"].mean().values)
        # M200_sd = float(trace.posterior["M200"].std().values)
        # c_sd = float(trace.posterior["c"].std().values)
        # sigma0_sd = float(trace.posterior["sigma_0"].std().values)

        # derived quantities using your helper functions (they expect numeric inputs)
        V200_calc = self._calc_V200_from_M200(M200_mean, z)
        r200_calc = self._calc_r200_from_V200(V200_calc, z)
        c_calc = self._calc_c_from_M200(M200_mean, h=getattr(self, "H", 0.7))

        # compute fitted velocity profiles using mean params (use your helpers)
        vel_rot_sq_fit = self._vel_rot_sq_profile(radius, M200_mean, c_mean, z, M_star, Re, sigma0_mean)
        vel_dm_sq_fit = self._vel_dm_sq_profile_M200(radius, M200_mean, c=c_mean, z=z)
        vel_star_sq_fit = self.stellar_util.stellar_vel_sq_profile(radius, M_star, Re)

        vel_total_fit = np.sqrt(np.clip(vel_rot_sq_fit, a_min=0.0, a_max=None))
        vel_dm_fit = np.sqrt(np.clip(vel_dm_sq_fit, a_min=0.0, a_max=None)) if vel_dm_sq_fit is not None else None
        vel_star_fit = np.sqrt(np.clip(vel_star_sq_fit, a_min=0.0, a_max=None))

        # ---------------------
        # Inference summary
        # ---------------------
        if self.fit_debug:
            # RMSE vel_rot
            vel_total_fit_valid = vel_total_fit[valid_mask]
            vel_rot_fit_mean = np.nanmean(vel_rot_valid)
            rmse_vel_rot = np.sqrt(np.nanmean((vel_rot_valid - vel_total_fit_valid) ** 2))

            print("\n------------ Infer Dark Matter NFW (PyMC) ------------")
            print("--- Summary ---")
            print(summary)
            print("--- LOO ---")
            print(f"{model_loo}")
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
            print("---------------------")
            print(f" Vel rot RMSE       : {rmse_vel_rot:.2f} km/s ({rmse_vel_rot/vel_rot_fit_mean:.2%})")
            print("------------------------------------------------------------\n")

        # ---------------------
        # plot
        # ---------------------
        if self.plot_enable:
            axes_trace = az.plot_trace(trace, var_names=["M200", "c", "sigma_0"])
            # set the units and credible interval
            # axes_trace is usually (n_vars, 2) for trace plots (left: pdf, right: trace)
            # Accessing rows for variables
            if axes_trace.shape[0] >= 3:
                axes_trace[0,0].set_xlabel("Msun")
                axes_trace[2,0].set_xlabel("km/s")
            plt.tight_layout()

            # az.plot_posterior(trace, var_names=["M200", "c", "sigma_0"], hdi_prob=0.94)

            idata = pm.to_inference_data(trace, posterior_predictive=ppc)
            az.plot_pair(idata, var_names=["M200", "c", "sigma_0"], kind='kde', marginals=True)
            az.plot_ppc(idata, data_pairs={"v_rot_obs": "v_rot_obs"}, mean=True, kind='cumulative', num_pp_samples=200)

            plt.tight_layout()
            plt.show()

            # v_rot fit plot
            plt.figure(figsize=(8,6))
            plt.errorbar(radius_valid, vel_rot_valid, yerr=vel_rot_err_valid, fmt='o', label='Observed V_rot', alpha=0.5)
            r_plot = np.linspace(0.0, np.nanmax(radius_valid), num=500)

            # compute posterior predictive mean and credible intervals
            v_rot_ppc = ppc.posterior_predictive["v_rot_obs"].stack(sample=("chain", "draw")).values
            v_rot_interp = np.array([np.interp(r_plot, radius_valid, v_rot_ppc[:,i]) for i in range(v_rot_ppc.shape[1])])
            v_rot_mean = np.mean(v_rot_interp, axis=0)
            v_rot_hdi = az.hdi(v_rot_interp, hdi_prob=0.94)
            plt.plot(r_plot, v_rot_mean, color='red', label='Posterior Predictive Mean V_rot')
            plt.fill_between(r_plot, v_rot_hdi[:,0], v_rot_hdi[:,1], color='red', alpha=0.3, label='94% Credible Interval')
            plt.xlabel('Radius (kpc)')
            plt.ylabel('Rotation Velocity V_rot (km/s)')
            plt.title(f'Fitted Rotation Curve for {self.PLATE_IFU}')
            plt.legend()

            plt.tight_layout()
            plt.show()

        success = True
        inf_result = {
            'radius': radius,
            'vel_rot': vel_total_fit,
            'vel_dm': vel_dm_fit,
            'vel_star': vel_star_fit,
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
            'elpd_loo_est': elpd_loo_est,
            'elpd_loo_se': elpd_loo_se,
            'p_loo': p_loo,
            'max_k': max_k,
            'mean_k': mean_k,
            'good_k_fraction': good_k_fraction,
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

    def inf_dm_nfw(self, radius: np.ndarray, vel_rot: np.ndarray, vel_rot_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z()
        return self._inf_dm_nfw_pymc(radius, vel_rot, vel_rot_err)


