import os
from statistics import mean
import tomllib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Constants
M_PIVOT_H_INV = 2e12  # in Msun/h, pivot mass for c-M200 relation
H_0 = 0.674           # Hubble parameter

# Plotting Colors (colorblind-safe palette + neutral grays)
COLOR_DATA_POINTS = '#4D4D4D'
COLOR_BOOTSTRAP_LINES = '#7F7F7F'
COLOR_SIGMA_BAND = '#D9D9D9'
COLOR_HDI_BAND = '#BDBDBD'
COLOR_HIGH_N = '#D55E00'
COLOR_LOW_N = '#0072B2'

def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    if not config_path.exists():
        raise FileNotFoundError("Error: config.toml file not found")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        if not config:
            raise ValueError("Error: config.toml file is empty")
    return config

# Initialize paths from config
config = load_config()
data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")
NFW_PARAM_CM200_FILENAME = config.get("file", {}).get("nfw_param_cm200_filename", "nfw_param_cm200.csv")

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / data_directory
result_dir = data_dir / result_directory

def c_m200_profile(M200: np.ndarray, c0: float, alpha: float, h: float = H_0) -> np.ndarray:
    """
    Theoretical non-linear model for c-M200 relation.
    c = c0 * (M200 / (M_pivot_h_inv / h))^alpha
    """
    M_pivot = M_PIVOT_H_INV / h
    return c0 * (M200 / M_pivot)**alpha


def get_m200_c_data():
    """
    Load data from the results CSV file.
    Returns a dict of arrays keyed by column name.
    """
    nfw_param_file = result_dir / NFW_PARAM_CM200_FILENAME
    if not nfw_param_file.exists():
        print(f"Warning: Data file {nfw_param_file} not found.")
        return None

    df = pd.read_csv(nfw_param_file, index_col=0)

    # Filter for successful fits
    if 'result' in df.columns:
        df = df[df['result'] == 'success']

    if df.empty:
        print("Warning: No successful fits found in data.")
        return None

    import ast
    sersic_n = df['sersic_n'].values if 'sersic_n' in df.columns else np.zeros(len(df))
    nrmse = df['nrmse'].values if 'nrmse' in df.columns else None

    log10_mu_obs = None
    log10_cov_obs = None
    log10_mu_obs_2 = None
    log10_cov_obs_2 = None
    if 'log10_mu_obs' in df.columns and 'log10_cov_obs' in df.columns:
        try:
            log10_mu_obs = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_mu_obs']])
            log10_cov_obs = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_cov_obs']])
        except Exception as e:
            print(f"Warning: Could not parse log10_mu_obs or log10_cov_obs: {e}")
    if 'log10_mu_obs_2' in df.columns and 'log10_cov_obs_2' in df.columns:
        try:
            log10_mu_obs_2 = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_mu_obs_2']])
            log10_cov_obs_2 = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_cov_obs_2']])
        except Exception as e:
            print(f"Warning: Could not parse log10_mu_obs_2 or log10_cov_obs_2: {e}")

    return {
        'sersic_n': sersic_n,
        'log10_mu_obs': log10_mu_obs,
        'log10_cov_obs': log10_cov_obs,
        'nrmse': nrmse,
        'log10_mu_obs_2': log10_mu_obs_2,
        'log10_cov_obs_2': log10_cov_obs_2,
    }


def filter_by_nrmse(nrmse: np.ndarray, drop_fraction: float = None, threshold: float = None) -> np.ndarray:
    """
    Filter observations based on the nrmse column.
    - If threshold is provided, keep points with nrmse <= threshold.
    - Else if drop_fraction is provided, drop the worst fraction by nrmse.
    """
    if nrmse is None:
        return None

    nrmse = np.array(nrmse, dtype=float)
    valid_mask = np.isfinite(nrmse)
    if not np.any(valid_mask):
        return valid_mask

    nrmse_valid = nrmse[valid_mask]

    if threshold is not None:
        print(f"Filtering data: keeping points with nrmse <= {threshold:.4f}")
        keep_mask = nrmse_valid <= threshold
    elif drop_fraction is not None:
        cutoff = np.quantile(nrmse_valid, 1.0 - drop_fraction)
        print(f"Filtering data: dropping {drop_fraction*100:.1f}% of points with nrmse above {cutoff:.4f}")
        keep_mask = nrmse_valid <= cutoff
    else:
        keep_mask = np.ones_like(nrmse_valid, dtype=bool)

    final_mask = np.zeros_like(valid_mask, dtype=bool)
    final_mask[valid_mask] = keep_mask
    return final_mask

def fit_m200_c_mcmc(log10_mu_obs: np.ndarray, log10_cov_obs: np.ndarray, verbose: bool = True):
    """
    Fit the non-linear c-M200 relation using a Hierarchical Bayesian Model (HBM).
    This uses the full covariance matrix from the first stage inference.
    """
    print("\n--- Fitting using Hierarchical Bayesian Model (HBM) ---")

    # Filter out any data points where covariance is not available or invalid
    valid_cov_mask = np.array([cov is not None and np.shape(cov) == (2, 2) and np.all(np.isfinite(cov)) for cov in log10_cov_obs])
    valid_mu_mask = np.array([mu is not None and np.shape(mu) == (2,) and np.all(np.isfinite(mu)) for mu in log10_mu_obs])
    valid_mask = valid_cov_mask & valid_mu_mask

    if not np.all(valid_mask):
        print(f"Warning: Dropping {np.sum(~valid_mask)} points due to invalid mu/cov data.")
        log10_mu_obs = log10_mu_obs[valid_mask]
        log10_cov_obs = log10_cov_obs[valid_mask]

    if len(log10_mu_obs) < 3:
        print("Not enough valid data points for MCMC fitting.")
        return None

    N_galaxies = len(log10_mu_obs)
    log_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    # Stack mu and cov for PyMC
    mu_obs_stacked = np.stack(log10_mu_obs)
    cov_obs_stacked = np.stack(log10_cov_obs)

    # Derive M200 and c from the observed log values for diagnostics/metrics
    log_M200_obs = mu_obs_stacked[:, 0]
    log_c_obs = mu_obs_stacked[:, 1]
    M200 = 10**log_M200_obs
    c = 10**log_c_obs

    # Shift the observed M200 by pivot to make sampling easier and more stable
    mu_obs_shifted = mu_obs_stacked.copy()
    mu_obs_shifted[:, 0] -= log_M_pivot

    with pm.Model() as model:
        # 1. Hyper-priors for the population parameters
        # Population distribution of M200 (shifted by pivot)
        mu_M = pm.Normal('mu_M', mu=np.mean(mu_obs_shifted[:, 0]), sigma=1.0)
        sigma_M = pm.HalfNormal('sigma_M', sigma=1.0)

        # Relation parameters
        log_c0_t = pm.Normal('log_c0', mu=np.log10(8.5), sigma=0.5)
        alpha_t = pm.Normal('alpha', mu=-0.1, sigma=0.3)

        # Intrinsic scatter (sigma_int)
        sigma_int = pm.HalfCauchy('sigma_int', beta=0.2)

        # 2. Marginalized Likelihood
        # Instead of sampling latent variables for each galaxy, we marginalize them out.
        # The true values [M_true, c_true] follow a population distribution:
        # M_true ~ N(mu_M, sigma_M^2)
        # c_true | M_true ~ N(log_c0 + alpha * M_true, sigma_int^2)
        # This implies the joint distribution of [M_true, c_true] is a 2D Normal:
        # Mean: [mu_M, log_c0 + alpha * mu_M]
        # Covariance: [[sigma_M^2, alpha * sigma_M^2], [alpha * sigma_M^2, alpha^2 * sigma_M^2 + sigma_int^2]]

        mu_pop = pt.stack([mu_M, log_c0_t + alpha_t * mu_M])

        cov_pop = pt.stack([
            pt.stack([sigma_M**2, alpha_t * sigma_M**2]),
            pt.stack([alpha_t * sigma_M**2, alpha_t**2 * sigma_M**2 + sigma_int**2])
        ])

        # The observed values are the true values plus observation noise:
        # obs ~ N(true, cov_obs)
        # Therefore, the marginal distribution of obs is:
        # obs ~ N(mu_pop, cov_pop + cov_obs)

        total_cov = cov_pop + cov_obs_stacked

        # Use a StudentT likelihood to be more robust to potential outliers in the observed data
        # obs = pm.MvStudentT('obs', mu=mu_pop, cov=total_cov, observed=mu_obs_shifted, nu=4)
        # Use a Normal likelihood for simplicity, but be aware it may be sensitive to outliers
        obs = pm.MvNormal('obs', mu=mu_pop, cov=total_cov, observed=mu_obs_shifted)

        # Sampling
        draws = 3000
        tune = 2000
        chains = min(4, os.cpu_count())
        target_accept = 0.95
        sampler = 'numpyro'
        init = "jitter+adapt_full"

        trace = pm.sample(init=init, draws=draws, tune=tune, chains=chains, cores=chains,
                          nuts_sampler=sampler, target_accept=target_accept,
                          random_seed=42, progressbar=True)
        try:
            pm.compute_log_likelihood(trace)
        except Exception as exc:
            print(f"Warning: compute_log_likelihood failed: {exc}")

    # summarize results
    var_names = ["log_c0", "alpha", "sigma_int"]
    summary = az.summary(trace, var_names=var_names, round_to=4)

    # LOO
    loo = az.loo(trace, pointwise=True)

    # mean and std
    log_c0_mean = summary.loc['log_c0', 'mean']
    alpha_mean = summary.loc['alpha', 'mean']
    log_c0_sd = summary.loc['log_c0', 'sd']
    alpha_std = summary.loc['alpha', 'sd']
    sigma_int_mean = summary.loc['sigma_int', 'mean']
    sigma_int_std = summary.loc['sigma_int', 'sd']

    # extract posterior samples
    posterior = trace.posterior
    log_c0_samples = posterior['log_c0'].values.flatten()
    alpha_samples = posterior['alpha'].values.flatten()
    sigma_int_samples = posterior['sigma_int'].values.flatten()

    # KDE curves and pair plot for diagnostics
    # try:
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    #     az.plot_kde(log_c0_samples, ax=axes[0])
    #     axes[0].set_title('KDE: log_c0')
    #     az.plot_kde(alpha_samples, ax=axes[1])
    #     axes[1].set_title('KDE: alpha')
    #     az.plot_kde(sigma_int_samples, ax=axes[2])
    #     axes[2].set_title('KDE: sigma_int')
    #     fig.tight_layout()
    #     kde_path = result_dir / "m200_c_hbm_kde.png"
    #     fig.savefig(kde_path, dpi=300, bbox_inches='tight')
    #     print(f"KDE plot saved to {kde_path}")
    # except Exception as exc:
    #     print(f"Warning: KDE plot failed: {exc}")

    # try:
    #     pair_fig = az.plot_pair(trace, var_names=["log_c0", "alpha"], kind="kde", marginals=True)
    #     pair_path = result_dir / "m200_c_hbm_pair.png"
    #     pair_fig.figure.savefig(pair_path, dpi=300, bbox_inches='tight')
    #     print(f"Pair plot saved to {pair_path}")
    # except Exception as exc:
    #     print(f"Warning: Pair plot failed: {exc}")

    # Convert back to linear c0
    c0_mean = 10**log_c0_mean
    c0_mean_std = c0_mean * np.log(10) * log_c0_sd

    # Calculate RMSE/NRMSE using the mean parameters (linear space)
    c_pred = c_m200_profile(M200, c0_mean, alpha_mean, h=H_0)
    residuals_mean = c - c_pred
    rmse_mean = np.sqrt(np.mean(residuals_mean**2))
    nrmse_mean = rmse_mean / (np.mean(c) if np.mean(c) > 0 else 1)

    # Recalculate Reduced Chi2 for the mean fit (log space)
    log_residuals_mean = np.log10(c) - np.log10(c_pred)
    log_rmse = np.sqrt(np.mean(log_residuals_mean**2))
    log_c_err_total = np.sqrt(cov_obs_stacked[:, 1, 1] + sigma_int_mean**2)
    dof = int(max(len(M200) - 2, 1))  # 2 parameters: log_c0 and alpha
    chi2_mean = np.sum((log_residuals_mean / log_c_err_total)**2)
    redchi_mean = chi2_mean / dof

    if verbose:
        print("--------- MCMC M200-c fit results ---------")
        print("--- Summary ---")
        print(summary)
        print("--- LOO ---")
        print(loo)
        print("--- mean ---")
        print(f" c0 mean        : {c0_mean:.4f} ± {c0_mean_std:.4f}")
        print(f" alpha mean     : {alpha_mean:.4f} ± {alpha_std:.4f}")
        print(f" sigma_int      : {sigma_int_mean:.4f}")
        print(f" Log RMSE       : {log_rmse:.3f}")
        print(f" NRMSE (mean)   : {nrmse_mean:.3f}")
        print(f" Reduced Chi2 (mean) : {redchi_mean:.3f}")
        print("------------------------------------------")

    return {
        'c0_mean': c0_mean,
        'alpha_mean': alpha_mean,
        'c0_mean_std': c0_mean_std,
        'alpha_std': alpha_std,
        'log_rmse': log_rmse,
        'sigma_int_mean': sigma_int_mean,
        'sigma_int_std': sigma_int_std,
        'alpha_samples': alpha_samples,
        'log_c0_samples': log_c0_samples
    }

def infer_sersic_n_threshold_mcmc(M200: np.ndarray, c: np.ndarray, sersic_n: np.ndarray, log10_cov_obs: np.ndarray = None):
    """
    Use a Bayesian switchpoint model to infer the optimal Sersic n threshold.
    """
    print("\n--- Inferring optimal Sersic n threshold using MCMC ---")
    log_M200 = np.log10(M200)
    log_c = np.log10(c)
    log_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    if log10_cov_obs is not None:
        log_c_err = np.sqrt(log10_cov_obs[:, 1, 1])
        total_log_err = log_c_err
    else:
        total_log_err = np.full_like(log_c, 0.15) # Fallback if no cov provided

    with pm.Model() as model:
        # Threshold prior: restrict to 10th-90th percentile to avoid edge effects
        lower_bound = np.percentile(sersic_n, 10)
        upper_bound = np.percentile(sersic_n, 90)
        threshold = pm.Uniform('threshold', lower=lower_bound, upper=upper_bound)

        # Steepness of the sigmoid transition
        k = pm.HalfNormal('k', sigma=10)

        # Priors for high n group
        log_c0_high = pm.Normal('log_c0_high', mu=np.log10(8.5), sigma=0.5)
        alpha_high = pm.Normal('alpha_high', mu=-0.1, sigma=0.2)

        # Priors for low n group
        log_c0_low = pm.Normal('log_c0_low', mu=np.log10(8.5), sigma=0.5)
        alpha_low = pm.Normal('alpha_low', mu=-0.1, sigma=0.2)

        # Weight for high group (1 when sersic_n > threshold, 0 when sersic_n < threshold)
        w = pm.math.sigmoid(k * (sersic_n - threshold))

        # Expected value
        log_c_expect_high = log_c0_high + alpha_high * (log_M200 - log_M_pivot)
        log_c_expect_low = log_c0_low + alpha_low * (log_M200 - log_M_pivot)

        log_c_expect = w * log_c_expect_high + (1 - w) * log_c_expect_low

        # Likelihood
        log_c_obs = pm.Normal('log_c_obs', mu=log_c_expect, sigma=total_log_err, observed=log_c)

        # Sampling
        draws = 3000
        tune = 2000
        chains = min(4, os.cpu_count())
        sampler = 'numpyro'
        init = "jitter+adapt_full"

        trace = pm.sample(init=init, draws=draws, tune=tune, chains=chains, cores=chains,
                          nuts_sampler=sampler, target_accept=0.95,
                          random_seed=42, progressbar=True,
                          return_inferencedata=True)

    summary = az.summary(trace, var_names=["threshold", "log_c0_high", "alpha_high", "log_c0_low", "alpha_low"], round_to=4)
    print("\n--- Threshold MCMC Summary ---")
    print(summary)

    threshold_mean = summary.loc['threshold', 'mean']
    threshold_sd = summary.loc['threshold', 'sd']

    print("--------- MCMC Threshold Inference Results ---------")
    print(f" Inferred Sersic n threshold: {threshold_mean:.4f} ± {threshold_sd:.4f}")
    print(f" High n - log_c0: {summary.loc['log_c0_high', 'mean']:.4f}, alpha: {summary.loc['alpha_high', 'mean']:.4f}")
    print(f" Low n  - log_c0: {summary.loc['log_c0_low', 'mean']:.4f}, alpha: {summary.loc['alpha_low', 'mean']:.4f}")
    print("----------------------------------------------------")

    return threshold_mean


def plot_m200_c_spaghetti_all(
    M200: np.ndarray,
    c: np.ndarray,
    log10_cov_obs: np.ndarray = None,
    fit_results: dict = None,
    n_boot: int = 50, # number of bootstrap samples for the spaghetti lines
    plot_suffix: str = "",
    show_error_bars: bool = False,
):
    """
    Spaghetti plot of c-M200 relation for all data (no Sersic n split).
    """
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return

    M200 = M200[valid_mask]
    c = c[valid_mask]
    if log10_cov_obs is not None:
        log10_cov_obs = log10_cov_obs[valid_mask]
    mask_all = np.ones_like(M200, dtype=bool)

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)
    log_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    if log10_cov_obs is not None and show_error_bars:
        M200_err = M200 * np.log(10) * np.sqrt(log10_cov_obs[:, 0, 0])
        c_err = c * np.log(10) * np.sqrt(log10_cov_obs[:, 1, 1])
        ax_top.errorbar(M200, c, xerr=M200_err, yerr=c_err, fmt='o', color=COLOR_DATA_POINTS, alpha=0.5, label='Data (all)', markersize=4, capsize=2, elinewidth=1)
    else:
        ax_top.scatter(M200, c, color=COLOR_DATA_POINTS, alpha=0.7, label='Data (all)', s=20, edgecolors='none')

    def bootstrap_lines(mask, color):
        if np.sum(mask) < 3:
            return
        log_M = np.log10(M200[mask])
        log_c = np.log10(c[mask])
        x = log_M - log_M_pivot
        n = len(x)
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            X = np.column_stack([x[idx], np.ones_like(x[idx])])
            coeffs, *_ = np.linalg.lstsq(X, log_c[idx], rcond=None)
            alpha = coeffs[0]
            log_c0 = coeffs[1]
            c_plot = c_m200_profile(m_plot, 10**log_c0, alpha, h=H_0)
            ax_top.plot(m_plot, c_plot, color=color, alpha=0.3, linewidth=1)

    bootstrap_lines(mask_all, COLOR_BOOTSTRAP_LINES)

    c0_fit = fit_results.get('c0_mean') if fit_results else None
    alpha_fit = fit_results.get('alpha_mean') if fit_results else None
    c0_fit_std = fit_results.get('c0_mean_std') if fit_results else None
    alpha_fit_std = fit_results.get('alpha_std') if fit_results else None
    sigma_int = fit_results.get('sigma_int_mean') if fit_results else None
    sigma_int_std = fit_results.get('sigma_int_std') if fit_results else None

    if c0_fit is None or alpha_fit is None:
        log_M = np.log10(M200)
        log_c = np.log10(c)
        x = log_M - log_M_pivot
        X = np.column_stack([x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(X, log_c, rcond=None)
        alpha_fit = coeffs[0]
        c0_fit = 10**coeffs[1]

    if c0_fit is not None and alpha_fit is not None:
        c_mean = c_m200_profile(m_plot, c0_fit, alpha_fit, h=H_0)
        ax_top.plot(m_plot, c_mean, color='black', linewidth=2, label='Mean Line (all)')
        sigma_band = sigma_int if sigma_int is not None else 0.15
        log_c_mean = np.log10(c_mean)
        c_lower = 10**(log_c_mean - sigma_band)
        c_upper = 10**(log_c_mean + sigma_band)
        ax_top.fill_between(m_plot, c_lower, c_upper, color=COLOR_SIGMA_BAND, alpha=0.4, label=r'$1\sigma_{int}$ (all)')

    log_c_obs = np.log10(c)
    log_c_pred = np.log10(c_m200_profile(M200, c0_fit, alpha_fit, h=H_0))
    residuals = log_c_obs - log_c_pred
    if log10_cov_obs is not None and show_error_bars:
        res_err = np.sqrt(log10_cov_obs[:, 1, 1])
        ax_bottom.errorbar(M200, residuals, xerr=M200_err, yerr=res_err, fmt='o', color=COLOR_DATA_POINTS, alpha=0.5, markersize=4, capsize=2, elinewidth=1)
    else:
        ax_bottom.scatter(M200, residuals, color=COLOR_DATA_POINTS, alpha=0.7, s=20, edgecolors='none')
    ax_bottom.axhline(0.0, color='black', linestyle='--', linewidth=1.2)
    sigma_band = sigma_int if sigma_int is not None else 0.15
    ax_bottom.axhspan(-sigma_band, sigma_band, color=COLOR_SIGMA_BAND, alpha=0.4)

    m_min, m_max = np.min(M200), np.max(M200)
    c_min, c_max = np.min(c), np.max(c)
    title = f"Dark Matter: Halo Concentration vs Mass\n"

    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'$c$', fontsize=12)
    ax_top.set_title(title, fontsize=13)
    ax_top.legend(fontsize=10, loc='lower left')
    ax_top.grid(True, which="both", ls="--", alpha=0.5)

    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel(r'$M_{200} \ [M_\odot]$', fontsize=12)
    ax_bottom.set_ylabel(r'$\Delta \log_{10} c$', fontsize=11)
    ax_bottom.grid(True, which="both", ls="--", alpha=0.5)

    c0_text = f"{c0_fit:.2f}" if c0_fit is not None else "n/a"
    c0_std_text = f"{c0_fit_std:.2f}" if c0_fit_std is not None else "n/a"
    alpha_text = f"{alpha_fit:.3f}" if alpha_fit is not None else "n/a"
    alpha_std_text = f"{alpha_fit_std:.3f}" if alpha_fit_std is not None else "n/a"
    sigma_text = f"{sigma_int:.3f}" if sigma_int is not None else "0.150"
    sigma_std_text = f"{sigma_int_std:.3f}" if sigma_int_std is not None else "n/a"
    infer_text = (
        rf"$c_0 = {c0_text} \pm {c0_std_text}$" "\n"
        rf"$\alpha = {alpha_text} \pm {alpha_std_text}$" "\n"
        rf"$\sigma_{{int}} = {sigma_text} \pm {sigma_std_text}$"
    )
    ax_top.text(
        0.98,
        0.02,
        infer_text,
        transform=ax_top.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f"m200_c_spaghetti{plot_suffix}_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Spaghetti plot saved to {plot_path}")


def plot_m200_c_spaghetti_split(
    M200: np.ndarray,
    c: np.ndarray,
    sersic_n: np.ndarray,
    log10_cov_obs: np.ndarray = None,
    fit_results_high: dict = None,
    fit_results_low: dict = None,
    sersic_n_threshold: float = 2.5,
    count_boot: int = 50, # count of bootstrap samples for the spaghetti lines
    plot_suffix: str = "",
    show_error_bars: bool = False,
):
    """
    Spaghetti plot of c-M200 relations split by Sersic n groups using bootstrap fits.
    """
    valid_mask = (M200 > 0) & (c > 0) & np.isfinite(M200) & np.isfinite(c)
    if not np.any(valid_mask):
        return

    M200 = M200[valid_mask]
    c = c[valid_mask]
    sersic_n = sersic_n[valid_mask]
    if log10_cov_obs is not None:
        log10_cov_obs = log10_cov_obs[valid_mask]

    mask_high_n = sersic_n >= sersic_n_threshold
    mask_low_n = sersic_n < sersic_n_threshold

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)
    log_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    if log10_cov_obs is not None and show_error_bars:
        M200_err = M200 * np.log(10) * np.sqrt(log10_cov_obs[:, 0, 0])
        c_err = c * np.log(10) * np.sqrt(log10_cov_obs[:, 1, 1])
        ax_top.errorbar(M200[mask_high_n], c[mask_high_n], xerr=M200_err[mask_high_n], yerr=c_err[mask_high_n], fmt='s', color=COLOR_HIGH_N, alpha=0.4,
                       label=rf'Data ($n \geq {sersic_n_threshold:.2f}$)', markersize=4, capsize=2, elinewidth=1)
        ax_top.errorbar(M200[mask_low_n], c[mask_low_n], xerr=M200_err[mask_low_n], yerr=c_err[mask_low_n], fmt='o', color=COLOR_LOW_N, alpha=0.4,
                       label=rf'Data ($n < {sersic_n_threshold:.2f}$)', markersize=4, capsize=2, elinewidth=1)
    else:
        ax_top.scatter(M200[mask_high_n], c[mask_high_n], color=COLOR_HIGH_N, alpha=0.4,
                       label=rf'Data ($n \geq {sersic_n_threshold:.2f}$)', s=20, marker='s', edgecolors='none')
        ax_top.scatter(M200[mask_low_n], c[mask_low_n], color=COLOR_LOW_N, alpha=0.4,
                       label=rf'Data ($n < {sersic_n_threshold:.2f}$)', s=20, marker='o', edgecolors='none')

    def bootstrap_lines(mask, color, linestyle):
        if np.sum(mask) < 3:
            return
        log_M = np.log10(M200[mask])
        log_c = np.log10(c[mask])
        x = log_M - log_M_pivot
        n = len(x)
        for _ in range(count_boot):
            idx = np.random.randint(0, n, n)
            X = np.column_stack([x[idx], np.ones_like(x[idx])])
            coeffs, *_ = np.linalg.lstsq(X, log_c[idx], rcond=None)
            alpha = coeffs[0]
            log_c0 = coeffs[1]
            c_plot = c_m200_profile(m_plot, 10**log_c0, alpha, h=H_0)
            ax_top.plot(m_plot, c_plot, color=color, alpha=0.15, linewidth=1, linestyle=linestyle)

    bootstrap_lines(mask_high_n, COLOR_HIGH_N, '-')
    bootstrap_lines(mask_low_n, COLOR_LOW_N, '-')

    def fit_group_mean(mask):
        if np.sum(mask) < 3:
            return None, None
        log_M = np.log10(M200[mask])
        log_c = np.log10(c[mask])
        x = log_M - log_M_pivot
        X = np.column_stack([x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(X, log_c, rcond=None)
        return 10**coeffs[1], coeffs[0]

    c0_high = fit_results_high.get('c0_mean') if fit_results_high else None
    alpha_high = fit_results_high.get('alpha_mean') if fit_results_high else None
    sigma_int_high = fit_results_high.get('sigma_int_mean') if fit_results_high else None

    c0_low = fit_results_low.get('c0_mean') if fit_results_low else None
    alpha_low = fit_results_low.get('alpha_mean') if fit_results_low else None
    sigma_int_low = fit_results_low.get('sigma_int_mean') if fit_results_low else None

    if c0_high is None or alpha_high is None:
        c0_high, alpha_high = fit_group_mean(mask_high_n)
    if c0_low is None or alpha_low is None:
        c0_low, alpha_low = fit_group_mean(mask_low_n)

    def fit_all_mean():
        log_M = np.log10(M200)
        log_c = np.log10(c)
        x = log_M - log_M_pivot
        X = np.column_stack([x, np.ones_like(x)])
        coeffs, *_ = np.linalg.lstsq(X, log_c, rcond=None)
        return 10**coeffs[1], coeffs[0]

    sigma_band_high = sigma_int_high if sigma_int_high is not None else 0.15
    sigma_band_low = sigma_int_low if sigma_int_low is not None else 0.15

    if c0_high is not None and alpha_high is not None:
        c_mean_high = c_m200_profile(m_plot, c0_high, alpha_high, h=H_0)
        ax_top.plot(m_plot, c_mean_high, color=COLOR_HIGH_N, linewidth=2, linestyle='-', label=rf'Mean Line ($n \geq {sersic_n_threshold:.2f}$)')
        log_c_mean = np.log10(c_mean_high)
        c_lower = 10**(log_c_mean - sigma_band_high)
        c_upper = 10**(log_c_mean + sigma_band_high)
        ax_top.fill_between(m_plot, c_lower, c_upper, color=COLOR_HIGH_N, alpha=0.15, label=rf'$1\sigma_{{int}}$ ($n \geq {sersic_n_threshold:.2f}$)')

    if c0_low is not None and alpha_low is not None:
        c_mean_low = c_m200_profile(m_plot, c0_low, alpha_low, h=H_0)
        ax_top.plot(m_plot, c_mean_low, color=COLOR_LOW_N, linewidth=2, linestyle='--', label=rf'Mean Line (n < {sersic_n_threshold:.2f})')
        log_c_mean = np.log10(c_mean_low)
        c_lower = 10**(log_c_mean - sigma_band_low)
        c_upper = 10**(log_c_mean + sigma_band_low)
        ax_top.fill_between(m_plot, c_lower, c_upper, color=COLOR_LOW_N, alpha=0.15, label=rf'$1\sigma_{{int}}$ (n < {sersic_n_threshold:.2f})')

    log_c_obs = np.log10(c)
    log_c_pred = np.full_like(log_c_obs, np.nan)
    if c0_high is not None and alpha_high is not None:
        log_c_pred[mask_high_n] = np.log10(c_m200_profile(M200[mask_high_n], c0_high, alpha_high, h=H_0))
    if c0_low is not None and alpha_low is not None:
        log_c_pred[mask_low_n] = np.log10(c_m200_profile(M200[mask_low_n], c0_low, alpha_low, h=H_0))
    if np.any(~np.isfinite(log_c_pred)):
        c0_all, alpha_all = fit_all_mean()
        log_c_pred = np.log10(c_m200_profile(M200, c0_all, alpha_all, h=H_0))

    residuals = log_c_obs - log_c_pred
    if log10_cov_obs is not None and show_error_bars:
        res_err = np.sqrt(log10_cov_obs[:, 1, 1])
        ax_bottom.errorbar(M200[mask_high_n], residuals[mask_high_n], xerr=M200_err[mask_high_n], yerr=res_err[mask_high_n], fmt='s', color=COLOR_HIGH_N, alpha=0.4, markersize=4, capsize=2, elinewidth=1)
        ax_bottom.errorbar(M200[mask_low_n], residuals[mask_low_n], xerr=M200_err[mask_low_n], yerr=res_err[mask_low_n], fmt='o', color=COLOR_LOW_N, alpha=0.4, markersize=4, capsize=2, elinewidth=1)
    else:
        ax_bottom.scatter(M200[mask_high_n], residuals[mask_high_n], color=COLOR_HIGH_N, alpha=0.4, s=20, marker='s', edgecolors='none')
        ax_bottom.scatter(M200[mask_low_n], residuals[mask_low_n], color=COLOR_LOW_N, alpha=0.4, s=20, marker='o', edgecolors='none')
    ax_bottom.axhline(0.0, color='black', linestyle='--', linewidth=1.2)
    ax_bottom.axhspan(-sigma_band_high, sigma_band_high, color=COLOR_HIGH_N, alpha=0.1)
    ax_bottom.axhspan(-sigma_band_low, sigma_band_low, color=COLOR_LOW_N, alpha=0.1)

    title = f"Dark Matter: Halo Concentration vs Mass (Split by Sersic n={sersic_n_threshold:.2f})\n"

    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'$c$', fontsize=12)
    ax_top.set_title(title, fontsize=13)
    ax_top.legend(fontsize=10, loc='lower left')
    ax_top.grid(True, which="both", ls="--", alpha=0.5)

    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel(r'$M_{200} \ [M_\odot]$', fontsize=12)
    ax_bottom.set_ylabel(r'$\Delta \log_{10} c$', fontsize=11)
    ax_bottom.grid(True, which="both", ls="--", alpha=0.5)

    c0_high_text = f"{c0_high:.2f}" if c0_high is not None else "n/a"
    alpha_high_text = f"{alpha_high:.3f}" if alpha_high is not None else "n/a"
    sigma_high_text = f"{sigma_band_high:.3f}"

    c0_low_text = f"{c0_low:.2f}" if c0_low is not None else "n/a"
    alpha_low_text = f"{alpha_low:.3f}" if alpha_low is not None else "n/a"
    sigma_low_text = f"{sigma_band_low:.3f}"

    infer_text = (
        rf"$n \geq {sersic_n_threshold:.2f}$: $c_0 = {c0_high_text}$, $\alpha = {alpha_high_text}$, $\sigma_{{int}} = {sigma_high_text}$" "\n"
        rf"$n < {sersic_n_threshold:.2f}$: $c_0 = {c0_low_text}$, $\alpha = {alpha_low_text}$, $\sigma_{{int}} = {sigma_low_text}$"
    )
    ax_top.text(
        0.98,
        0.02,
        infer_text,
        transform=ax_top.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f"m200_c_spaghetti{plot_suffix}_split.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Spaghetti plot saved to {plot_path}")


def plot_alpha_posterior_difference(alpha_high_samples: np.ndarray, alpha_low_samples: np.ndarray, plot_suffix: str = ""):
    """
    Plot posterior difference distribution for alpha_high - alpha_low.
    """
    diff = alpha_high_samples - alpha_low_samples
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    hist, edges = np.histogram(diff, bins=200, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, color='black', linewidth=1.5)

    diff_mean = float(np.mean(diff))
    hdi_low, hdi_high = az.hdi(diff, hdi_prob=0.94)

    hdi_mask = (centers >= hdi_low) & (centers <= hdi_high)
    ax.fill_between(centers, 0.0, hist, where=hdi_mask, color=COLOR_HDI_BAND, alpha=0.25)

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Posterior Difference: alpha_high - alpha_low', fontsize=12)
    ax.set_xlabel(r'$\Delta \alpha$')
    ax.set_ylabel('Density')

    y_top = np.max(hist) * 1.02
    tick_height = np.max(hist) * 0.06
    ax.hlines(y_top, hdi_low, hdi_high, color='black', linewidth=3)
    ax.vlines([hdi_low, hdi_high], y_top - tick_height, y_top + tick_height, color='black', linewidth=2)

    ax.text(
        0.02,
        0.98,
        f"mean = {diff_mean:.4f}\n94% HDI = [{hdi_low:.4f}, {hdi_high:.4f}]",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f"m200_alpha_posterior_diff{plot_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Posterior difference plot saved to {plot_path}")


def plot_logc0_posterior_difference(log_c0_high_samples: np.ndarray, log_c0_low_samples: np.ndarray, plot_suffix: str = ""):
    """
    Plot posterior difference distribution for log_c0_high - log_c0_low.
    """
    diff = log_c0_high_samples - log_c0_low_samples
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    hist, edges = np.histogram(diff, bins=200, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, color='black', linewidth=1.5)

    diff_mean = float(np.mean(diff))
    hdi_low, hdi_high = az.hdi(diff, hdi_prob=0.94)

    hdi_mask = (centers >= hdi_low) & (centers <= hdi_high)
    ax.fill_between(centers, 0.0, hist, where=hdi_mask, color=COLOR_HDI_BAND, alpha=0.25)

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Posterior Difference: log_c0_high - log_c0_low', fontsize=12)
    ax.set_xlabel(r'$\Delta \log c_0$')
    ax.set_ylabel('Density')

    y_top = np.max(hist) * 1.02
    tick_height = np.max(hist) * 0.06
    ax.hlines(y_top, hdi_low, hdi_high, color='black', linewidth=3)
    ax.vlines([hdi_low, hdi_high], y_top - tick_height, y_top + tick_height, color='black', linewidth=2)

    ax.text(
        0.02,
        0.98,
        f"mean = {diff_mean:.4f}\n94% HDI = [{hdi_low:.4f}, {hdi_high:.4f}]",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f"m200_logc0_posterior_diff{plot_suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Posterior difference plot saved to {plot_path}")


def main(
    n_threshold: str = 'auto',
    nrmse_threshold: float = None,
    sigma: int = 1,
    show_error_bars: bool = False,
):
    """
    Main execution function.
    n_threshold: 'auto', 'all', or a float value
    sigma: integer selector for mu/cov columns (e.g., 1 uses log10_mu_obs)
    """
    # 1. Load data
    data = get_m200_c_data()
    if not data:
        print("Failed to load data. Exiting.")
        return

    sersic_n_raw = np.array(data['sersic_n'], dtype=float)
    nrmse = np.array(data['nrmse'], dtype=float) if 'nrmse' in data else None

    sersic_n_raw = np.where(np.isfinite(sersic_n_raw), sersic_n_raw, 0.0)

    log10_mu_obs_raw = data.get('log10_mu_obs')
    log10_cov_obs_raw = data.get('log10_cov_obs')
    log10_mu_obs_2_raw = data.get('log10_mu_obs_2')
    log10_cov_obs_2_raw = data.get('log10_cov_obs_2')

    # 2. Filter outliers (disabled by default)
    if nrmse_threshold is not None:
        mask = filter_by_nrmse(nrmse, threshold=nrmse_threshold)
        if mask is None:
            print("Warning: nrmse column missing; skipping NRMSE filtering.")
            mask = np.ones_like(sersic_n_raw, dtype=bool)
    else:
        mask = np.ones_like(sersic_n_raw, dtype=bool)

    def compute_threshold(M200, c, sersic_n, log10_cov_obs):
        mode_lower = n_threshold.lower()
        if mode_lower == 'auto':
            return infer_sersic_n_threshold_mcmc(M200, c, sersic_n, log10_cov_obs=log10_cov_obs)

        if mode_lower == 'all':
            threshold_val = 2.5
            print(f"\n--- Using default Sersic n threshold: {threshold_val:.2f} ---")
            return threshold_val

        try:
            threshold_val = float(n_threshold)
            print(f"\n--- Using fixed Sersic n threshold: {threshold_val:.2f} ---")
            return threshold_val
        except ValueError:
            print(f"\n--- Invalid Sersic n threshold '{n_threshold}', falling back to auto ---")
            return infer_sersic_n_threshold_mcmc(M200, c, sersic_n, log10_cov_obs=log10_cov_obs)

    def fit_all_data(log10_mu_obs, log10_cov_obs):
        if n_threshold.lower() != 'all':
            print("\n# 1. Skipping All Data fit (use --n=all to enable)")
            return None

        print("\n# 1. Fitting All Data")
        print("\nUsing MCMC for fitting...")
        return fit_m200_c_mcmc(log10_mu_obs, log10_cov_obs)

    def fit_group(label, log10_mu_obs, log10_cov_obs):
        print(label)
        return fit_m200_c_mcmc(log10_mu_obs, log10_cov_obs)

    def run_pipeline(log10_mu_obs_raw, log10_cov_obs_raw, plot_suffix: str, label: str):
        print(f"\n=== Running {label} data ===")
        if log10_mu_obs_raw is None or log10_cov_obs_raw is None:
            print(f"Warning: Missing log10_mu_obs/log10_cov_obs for {label}; skipping.")
            return

        log10_mu_obs = np.array(log10_mu_obs_raw[mask].tolist(), dtype=float)
        log10_cov_obs = np.array(log10_cov_obs_raw[mask].tolist(), dtype=float)

        M200 = 10 ** log10_mu_obs[:, 0]
        c = 10 ** log10_mu_obs[:, 1]
        sersic_n = sersic_n_raw[mask]

        print(f"Data points after filtering: {len(M200)} (dropped {len(sersic_n_raw) - len(M200)}), nrmse_threshold={nrmse_threshold}")

        if len(M200) < 3:
            print("Not enough valid data points for fitting.")
            return

        if n_threshold.lower() == 'all':
            fit_results_all = fit_all_data(log10_mu_obs, log10_cov_obs)
            plot_m200_c_spaghetti_all(
                M200,
                c,
                log10_cov_obs=log10_cov_obs,
                fit_results=fit_results_all,
                plot_suffix=plot_suffix,
                show_error_bars=show_error_bars,
            )
            plt.show()
            return

        # split by Sersic n using the threshold
        threshold = compute_threshold(M200, c, sersic_n, log10_cov_obs)
        mask_high_n = sersic_n >= threshold
        mask_low_n = sersic_n < threshold

        fit_results_high = fit_group(
            f"\n#2. Fitting High Sersic n (>= {threshold:.2f})",
            log10_mu_obs[mask_high_n],
            log10_cov_obs[mask_high_n],
        )

        fit_results_low = fit_group(
            f"\n#3. Fitting Low Sersic n (< {threshold:.2f})",
            log10_mu_obs[mask_low_n],
            log10_cov_obs[mask_low_n],
        )

        if fit_results_high and fit_results_low and fit_results_high.get('alpha_samples') is not None and fit_results_low.get('alpha_samples') is not None:
            plot_alpha_posterior_difference(fit_results_high['alpha_samples'], fit_results_low['alpha_samples'], plot_suffix=plot_suffix)
        if fit_results_high and fit_results_low and fit_results_high.get('log_c0_samples') is not None and fit_results_low.get('log_c0_samples') is not None:
            plot_logc0_posterior_difference(fit_results_high['log_c0_samples'], fit_results_low['log_c0_samples'], plot_suffix=plot_suffix)

        plot_m200_c_spaghetti_split(
            M200,
            c,
            sersic_n,
            log10_cov_obs=log10_cov_obs,
            fit_results_high=fit_results_high,
            fit_results_low=fit_results_low,
            sersic_n_threshold=threshold,
            plot_suffix=plot_suffix,
            show_error_bars=show_error_bars,
        )

        plt.show()
        plt.close()

    sigma_map = {
        1: (log10_mu_obs_raw, log10_cov_obs_raw, "_1sigma", "1-sigma"),
        2: (log10_mu_obs_2_raw, log10_cov_obs_2_raw, "_2sigma", "2-sigma"),
    }

    if sigma in sigma_map:
        mu_raw, cov_raw, suffix, label = sigma_map[sigma]
        run_pipeline(mu_raw, cov_raw, suffix, label)
    else:
        print(f"Unsupported sigma={sigma}; available options are 1 or 2.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit c-M200 relation.")
    parser.add_argument('--n', type=str, default='2.5',
                        help="Sersic n threshold: 'all', 'auto', or a float value (e.g., '2.5')")
    parser.add_argument('--nmrse', '--nrmse', dest='nrmse', type=float, default=0.10,
                        help="NRMSE threshold: keep points with nrmse <= value")
    parser.add_argument('--sigma', type=int, default=2,
                        help="Sigma selector for mu/cov columns (e.g., 1 or 2)")
    parser.add_argument('--show-error-bars', action='store_true',
                        help="Show error bars for data points in plots")
    args = parser.parse_args()

    main(n_threshold=args.n, nrmse_threshold=args.nrmse, sigma=args.sigma, show_error_bars=args.show_error_bars)
