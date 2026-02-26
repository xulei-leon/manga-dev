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
DM_INTRINSIC_SIGMA_DEX = 0.15  # Intrinsic scatter in dex for c

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


def log_c_m200_profile(log_M200, log_c0, alpha, h=H_0):
    log_M_pivot = np.log10(M_PIVOT_H_INV / h)
    return log_c0 + alpha * (log_M200 - log_M_pivot)

def get_m200_c_data():
    """
    Load M200 and c data from the results CSV file.
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
    M200 = df['M200'].values
    M200_std = df['M200_std'].values if 'M200_std' in df.columns else np.zeros_like(M200)
    c = df['c'].values
    c_std = df['c_std'].values if 'c_std' in df.columns else np.zeros_like(c)
    sersic_n = df['sersic_n'].values if 'sersic_n' in df.columns else np.zeros_like(c)
    nrmse = df['nrmse'].values if 'nrmse' in df.columns else None

    log10_mu_obs = None
    log10_cov_obs = None
    if 'log10_mu_obs' in df.columns and 'log10_cov_obs' in df.columns:
        try:
            log10_mu_obs = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_mu_obs']])
            log10_cov_obs = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df['log10_cov_obs']])
        except Exception as e:
            print(f"Warning: Could not parse log10_mu_obs or log10_cov_obs: {e}")

    return {
        'M200': M200,
        'M200_std': M200_std,
        'c': c,
        'c_std': c_std,
        'sersic_n': sersic_n,
        'log10_mu_obs': log10_mu_obs,
        'log10_cov_obs': log10_cov_obs,
        'nrmse': nrmse,
    }

def filter_by_cov(log10_cov_obs: np.ndarray, drop_fraction: float = 0.10) -> np.ndarray:
    """
    Filter out the worst observations based on the provided covariance matrices.
    The "worst" entries are defined as those with the largest total variance
    (trace of the 2x2 covariance in log space). By default, drop the worst 10%.
    """
    valid_mask = np.ones(len(log10_cov_obs), dtype=bool)
    if log10_cov_obs is None:
        return valid_mask

    cov_valid_mask = np.array([
        cov is not None and np.shape(cov) == (2, 2) and np.all(np.isfinite(cov))
        for cov in log10_cov_obs
    ])
    valid_mask = valid_mask & cov_valid_mask
    if not np.any(valid_mask):
        return valid_mask

    cov_valid = log10_cov_obs[valid_mask]
    var_score = np.array([np.trace(cov) for cov in cov_valid])
    if var_score.size == 0:
        return valid_mask

    cutoff = np.quantile(var_score, 1.0 - drop_fraction)
    print(f"Filtering data: dropping {drop_fraction*100:.1f}% of points with variance score above {cutoff:.4f}")
    keep_mask = var_score <= cutoff

    final_mask = np.zeros_like(valid_mask, dtype=bool)
    final_mask[valid_mask] = keep_mask
    return final_mask

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

def fit_m200_c_nonlinear(M200: np.ndarray, c: np.ndarray, c_std: np.ndarray = None, intrinsic_scatter_dex: float = DM_INTRINSIC_SIGMA_DEX, verbose: bool = True):
    """
    Fit the non-linear c-M200 relation using scipy.optimize.curve_fit.
    Includes an intrinsic scatter (in dex) added in quadrature to the measurement errors.
    Returns c0_fit, alpha_fit, c0_err, alpha_err, log_rmse.
    """
    # Initial guess for c0 and alpha based on typical values (e.g., Dutton & Macciò 2014)
    p0 = [8.5, -0.10]

    # Use c_std for weighting if available and valid
    sigma = None
    absolute_sigma = False
    if c_std is not None and np.all(c_std > 0):
        # Convert intrinsic scatter from dex to linear scale error approximately
        # delta_c / c = ln(10) * delta_log10_c
        intrinsic_scatter_linear = c * np.log(10) * intrinsic_scatter_dex
        # Add measurement error and intrinsic scatter in quadrature
        sigma = np.sqrt(c_std**2 + intrinsic_scatter_linear**2)
        absolute_sigma = True
    else:
        # If no measurement errors, just use intrinsic scatter
        sigma = c * np.log(10) * intrinsic_scatter_dex
        absolute_sigma = True

    # Wrapper to fix h parameter
    def model_to_fit(M, c0, alpha):
        return c_m200_profile(M, c0, alpha, h=H_0)

    try:
        popt, pcov = curve_fit(
            model_to_fit,
            M200,
            c,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            maxfev=10000
        )
        c0_fit, alpha_fit = popt
        c0_err, alpha_err = np.sqrt(np.diag(pcov))

        c_pred = model_to_fit(M200, c0_fit, alpha_fit)

        # Calculate metrics in log space for better representation in log-log plots
        log_c = np.log10(c)
        log_c_pred = np.log10(c_pred)
        log_residuals = log_c - log_c_pred
        log_rmse = np.sqrt(np.mean(log_residuals**2))

        if sigma is not None:
            residuals = c - c_pred
            chi_squared = np.sum((residuals / sigma)**2)
            dof = len(M200) - len(popt)
            chi_squared_reduced = chi_squared / max(dof, 1)
        else:
            chi_squared_reduced = np.nan

        if verbose:
            print("--------- Non-linear M200-c fit results ---------")
            print(f" c0             : {c0_fit:.4f} ± {c0_err:.4f}")
            print(f" alpha          : {alpha_fit:.4f} ± {alpha_err:.4f}")
            print("---------------")
            print(f" chi² reduced   : {chi_squared_reduced:.3f}")
            print(f" Log RMSE       : {log_rmse:.3f}")
            print("-------------------------------------------------")

        return c0_fit, alpha_fit, c0_err, alpha_err, log_rmse

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None, None, None

def fit_m200_c_mcmc(log10_mu_obs: np.ndarray, log10_cov_obs: np.ndarray, intrinsic_scatter_dex: float = DM_INTRINSIC_SIGMA_DEX, verbose: bool = True):
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
        return None, None, None, None, None

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
        obs = pm.MvStudentT('obs', mu=mu_pop, cov=total_cov, observed=mu_obs_shifted, nu=4)

        # Sampling
        draws = 3000
        tune = 2000
        chains = min(4, os.cpu_count())
        target_accept = 0.98
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

    # extract posterior samples
    posterior = trace.posterior
    log_c0_samples = posterior['log_c0'].values.flatten()
    alpha_samples = posterior['alpha'].values.flatten()

    # Maximum A Posteriori (MAP) estimates
    if "logp" in trace.sample_stats:
        lp_stacked = trace.sample_stats["logp"].stack(sample=("chain", "draw"))
    elif "lp" in trace.sample_stats:
        lp_stacked = trace.sample_stats["lp"].stack(sample=("chain", "draw"))
    else:
        raise KeyError("Could not find 'logp' or 'lp' in trace.sample_stats")
    best_idx = int(lp_stacked.argmax("sample").values)
    log_c0_best = log_c0_samples[best_idx]
    alpha_best = alpha_samples[best_idx]


    # Convert back to linear c0
    c0_mean = 10**log_c0_mean
    c0_mean_std = c0_mean * np.log(10) * log_c0_sd
    c0_best = 10**log_c0_best
    c0_best_sd = c0_best * np.log(10) * log_c0_sd

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

    # Recalculate Reduced Chi2 for the best fit
    c_pred_best = c_m200_profile(M200, 10**log_c0_best, alpha_best, h=H_0)
    log_residuals_best = np.log10(c) - np.log10(c_pred_best)
    rmse_best = np.sqrt(np.mean(log_residuals_best**2))
    nrmse_best = rmse_best / (np.mean(c) if np.mean(c) > 0 else 1)
    dof = int(max(len(M200) - 2, 1))  # 2 parameters: log_c0 and alpha

    # Calculate total error in log space for chi2
    # We use the observed variance in log_c (index 1,1) + intrinsic scatter
    log_c_err_total = np.sqrt(cov_obs_stacked[:, 1, 1] + sigma_int_mean**2)
    chi2_best = np.sum((log_residuals_best / log_c_err_total)**2)
    redchi_best = chi2_best / dof


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
        print("--- best ---")
        print(f" c0 best        : {c0_best:.4f} ± {c0_best_sd:.4f}")
        print(f" alpha best     : {alpha_best:.4f} ± {alpha_std:.4f}")
        print(f" RMSE (best)    : {rmse_best:.3f}")
        print(f" NRMSE (best)   : {nrmse_best:.3f}")
        print(f" Reduced Chi2 (best) : {redchi_best:.3f}")
        print("--- metrics ---")

        print("------------------------------------------")

    return c0_mean, alpha_mean, c0_mean_std, alpha_std, log_rmse

def infer_sersic_n_threshold_mcmc(M200: np.ndarray, c: np.ndarray, sersic_n: np.ndarray, c_std: np.ndarray = None, intrinsic_scatter_dex: float = DM_INTRINSIC_SIGMA_DEX):
    """
    Use a Bayesian switchpoint model to infer the optimal Sersic n threshold.
    """
    print("\n--- Inferring optimal Sersic n threshold using MCMC ---")
    log_M200 = np.log10(M200)
    log_c = np.log10(c)
    log_M_pivot = np.log10(M_PIVOT_H_INV / H_0)

    if c_std is not None and np.all(c_std > 0):
        log_c_err = c_std / (c * np.log(10))
        total_log_err = np.sqrt(log_c_err**2 + intrinsic_scatter_dex**2)
    else:
        total_log_err = np.full_like(log_c, intrinsic_scatter_dex)

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

def infer_sersic_n_threshold_grid(M200: np.ndarray, c: np.ndarray, sersic_n: np.ndarray, c_std: np.ndarray = None, intrinsic_scatter_dex: float = DM_INTRINSIC_SIGMA_DEX):
    """
    Infer the optimal Sersic n threshold by grid search, minimizing the combined RMSE.
    """
    print("\n--- Inferring optimal Sersic n threshold using Grid Search ---")
    lower_bound = np.percentile(sersic_n, 15)
    upper_bound = np.percentile(sersic_n, 85)

    thresholds = np.linspace(lower_bound, upper_bound, 50)
    best_threshold = 2.5
    min_rmse = np.inf

    for t in thresholds:
        mask_high = sersic_n >= t
        mask_low = sersic_n < t

        if np.sum(mask_high) < 5 or np.sum(mask_low) < 5:
            continue

        # Fit high
        res_high = fit_m200_c_nonlinear(M200[mask_high], c[mask_high],
                                        c_std[mask_high] if c_std is not None else None,
                                        intrinsic_scatter_dex, verbose=False)
        # Fit low
        res_low = fit_m200_c_nonlinear(M200[mask_low], c[mask_low],
                                       c_std[mask_low] if c_std is not None else None,
                                       intrinsic_scatter_dex, verbose=False)

        if res_high[0] is None or res_low[0] is None:
            continue

        # Calculate combined RMSE
        rmse_high = res_high[4]
        rmse_low = res_low[4]

        # Weighted average of RMSE squared
        n_high = np.sum(mask_high)
        n_low = np.sum(mask_low)
        combined_rmse = np.sqrt((n_high * rmse_high**2 + n_low * rmse_low**2) / (n_high + n_low))

        if combined_rmse < min_rmse:
            min_rmse = combined_rmse
            best_threshold = t

    print(f"--------- Grid Search Threshold Inference Results ---------")
    print(f" Best Sersic n threshold: {best_threshold:.4f}")
    print(f" Minimum combined Log RMSE: {min_rmse:.4f}")
    print("-----------------------------------------------------------")

    return best_threshold

def plot_m200_c_all(
    M200: np.ndarray,
    c: np.ndarray,
    sersic_n: np.ndarray,
    c0_fit: float,
    alpha_fit: float,
    log_rmse: float,
    c0_fit_std: float = None,
    alpha_fit_std: float = None,
    threshold: float = 2.5,
):
    """
    Plot the M200 vs c data and the overall non-linear fit with residuals, color-coded by Sersic n.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # Split data by Sersic n
    mask_high_n = sersic_n >= threshold
    mask_low_n = sersic_n < threshold

    # Plot raw data with color mapping based on sersic_n
    ax1.scatter(M200[mask_high_n], c[mask_high_n], color='red', alpha=0.6, label=rf'Data ($n \geq {threshold:.2f}$)', s=20, edgecolors='none')
    ax1.scatter(M200[mask_low_n], c[mask_low_n], color='blue', alpha=0.6, label=rf'Data ($n < {threshold:.2f}$)', s=20, edgecolors='none')

    # Plot fit
    if c0_fit is not None and alpha_fit is not None:
        # Generate smooth curve for plotting
        m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)
        c_plot = c_m200_profile(m_plot, c0_fit, alpha_fit, h=H_0)

        c0_label = f"{c0_fit:.2f}"
        alpha_label = f"{alpha_fit:.3f}"
        if c0_fit_std is not None:
            c0_label = f"{c0_label} ± {c0_fit_std:.2f}"
        if alpha_fit_std is not None:
            alpha_label = f"{alpha_label} ± {alpha_fit_std:.3f}"
        ax1.plot(m_plot, c_plot, color='black', linewidth=2,
                 label=rf'Fit (All): $c_0={c0_label}$, $\alpha={alpha_label}$')

        # Plot ±1 sigma band in log space
        if log_rmse is not None and not np.isnan(log_rmse):
            c_upper = 10**(np.log10(c_plot) + log_rmse)
            c_lower = 10**(np.log10(c_plot) - log_rmse)
            ax1.fill_between(m_plot, c_lower, c_upper,
                             color='black', alpha=0.2, label=r'Fit $\pm 1\sigma$ (log)')

        # Calculate residuals (in log space)
        c_pred = c_m200_profile(M200, c0_fit, alpha_fit, h=H_0)
        log_residuals = np.log10(c) - np.log10(c_pred)

        # Plot residuals color-coded by sersic_n
        ax2.scatter(M200[mask_high_n], log_residuals[mask_high_n], color='red', alpha=0.6, s=20, edgecolors='none')
        ax2.scatter(M200[mask_low_n], log_residuals[mask_low_n], color='blue', alpha=0.6, s=20, edgecolors='none')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)

        if log_rmse is not None and not np.isnan(log_rmse):
            ax2.axhline(log_rmse, color='black', linestyle=':', alpha=0.5)
            ax2.axhline(-log_rmse, color='black', linestyle=':', alpha=0.5)
            ax2.fill_between(m_plot, -log_rmse, log_rmse, color='black', alpha=0.1)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$c$', fontsize=12)
    ax1.set_title('DM NFW: $M_{200}$ vs $c$ (All Data)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$M_{200} \ [M_\odot]$', fontsize=12)
    ax2.set_ylabel(r'$\Delta \log_{10} c$', fontsize=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Save plot
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "m200_c_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Overall plot saved to {plot_path}")

def plot_m200_c_split(
    M200: np.ndarray,
    c: np.ndarray,
    sersic_n: np.ndarray,
    c0_high: float,
    alpha_high: float,
    rmse_high: float,
    c0_low: float,
    alpha_low: float,
    rmse_low: float,
    c0_high_std: float = None,
    alpha_high_std: float = None,
    c0_low_std: float = None,
    alpha_low_std: float = None,
    threshold: float = 2.5,
):
    """
    Plot the M200 vs c data and the non-linear fits with residuals, separated by Sersic n.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # Split data by Sersic n
    mask_high_n = sersic_n >= threshold
    mask_low_n = sersic_n < threshold

    # Plot raw data with color mapping based on sersic_n
    ax1.scatter(M200[mask_high_n], c[mask_high_n], color='red', alpha=0.6, label=rf'Data ($n \geq {threshold:.2f}$)', s=20, edgecolors='none')
    ax1.scatter(M200[mask_low_n], c[mask_low_n], color='blue', alpha=0.6, label=rf'Data ($n < {threshold:.2f}$)', s=20, edgecolors='none')

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)

    # Plot high n fit
    if c0_high is not None and alpha_high is not None:
        c_plot_high = c_m200_profile(m_plot, c0_high, alpha_high, h=H_0)
        c0_high_label = f"{c0_high:.2f}"
        alpha_high_label = f"{alpha_high:.3f}"
        if c0_high_std is not None:
            c0_high_label = f"{c0_high_label} ± {c0_high_std:.2f}"
        if alpha_high_std is not None:
            alpha_high_label = f"{alpha_high_label} ± {alpha_high_std:.3f}"
        ax1.plot(m_plot, c_plot_high, color='darkred', linewidth=2,
                 label=rf'Fit ($n \geq {threshold:.2f}$): $c_0={c0_high_label}$, $\alpha={alpha_high_label}$')

        if rmse_high is not None and not np.isnan(rmse_high):
            c_upper = 10**(np.log10(c_plot_high) + rmse_high)
            c_lower = 10**(np.log10(c_plot_high) - rmse_high)
            ax1.fill_between(m_plot, c_lower, c_upper, color='red', alpha=0.15)

        c_pred_high = c_m200_profile(M200[mask_high_n], c0_high, alpha_high, h=H_0)
        log_res_high = np.log10(c[mask_high_n]) - np.log10(c_pred_high)
        ax2.scatter(M200[mask_high_n], log_res_high, color='red', alpha=0.6, s=20, edgecolors='none')

    # Plot low n fit
    if c0_low is not None and alpha_low is not None:
        c_plot_low = c_m200_profile(m_plot, c0_low, alpha_low, h=H_0)
        c0_low_label = f"{c0_low:.2f}"
        alpha_low_label = f"{alpha_low:.3f}"
        if c0_low_std is not None:
            c0_low_label = f"{c0_low_label} ± {c0_low_std:.2f}"
        if alpha_low_std is not None:
            alpha_low_label = f"{alpha_low_label} ± {alpha_low_std:.3f}"
        ax1.plot(m_plot, c_plot_low, color='darkblue', linewidth=2,
                 label=rf'Fit ($n < {threshold:.2f}$): $c_0={c0_low_label}$, $\alpha={alpha_low_label}$')

        if rmse_low is not None and not np.isnan(rmse_low):
            c_upper = 10**(np.log10(c_plot_low) + rmse_low)
            c_lower = 10**(np.log10(c_plot_low) - rmse_low)
            ax1.fill_between(m_plot, c_lower, c_upper, color='blue', alpha=0.15)

        c_pred_low = c_m200_profile(M200[mask_low_n], c0_low, alpha_low, h=H_0)
        log_res_low = np.log10(c[mask_low_n]) - np.log10(c_pred_low)
        ax2.scatter(M200[mask_low_n], log_res_low, color='blue', alpha=0.6, s=20, edgecolors='none')

    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$c$', fontsize=12)
    ax1.set_title('DM NFW: $M_{200}$ vs $c$', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$M_{200} \ [M_\odot]$', fontsize=12)
    ax2.set_ylabel(r'$\Delta \log_{10} c$', fontsize=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Save plot
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "m200_c_split.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Split plot saved to {plot_path}")


def main(
    mode: str = 'curve_fit',
    n_threshold: str = 'auto',
    filter_drop_fraction: float = None,
    nrmse_threshold: float = None,
):
    """
    Main execution function.
    mode: 'curve_fit' (default) or 'mcmc'
    n_threshold: 'auto' or a float value
    """
    # 1. Load data
    data = get_m200_c_data()
    if not data:
        print("Failed to load data. Exiting.")
        return

    M200_raw = np.array(data['M200'], dtype=float)
    M200_std = np.array(data['M200_std'], dtype=float)
    c_raw = np.array(data['c'], dtype=float)
    c_std = np.array(data['c_std'], dtype=float)
    sersic_n_raw = np.array(data['sersic_n'], dtype=float)
    nrmse = np.array(data['nrmse'], dtype=float) if 'nrmse' in data else None

    M200_std = np.where(np.isfinite(M200_std), M200_std, 0.0)
    c_std = np.where(np.isfinite(c_std), c_std, 0.0)
    sersic_n_raw = np.where(np.isfinite(sersic_n_raw), sersic_n_raw, 0.0)

    log10_mu_obs_raw = data.get('log10_mu_obs')
    log10_cov_obs_raw = data.get('log10_cov_obs')

    # 2. Filter outliers (disabled by default)
    if nrmse_threshold is not None or filter_drop_fraction is not None:
        # mask = filter_by_cov(log10_cov_obs_raw, drop_fraction=filter_drop_fraction)
        mask = filter_by_nrmse(nrmse, drop_fraction=filter_drop_fraction, threshold=nrmse_threshold)
        if mask is None:
            print("Warning: nrmse column missing; skipping NRMSE filtering.")
            mask = np.ones_like(M200_raw, dtype=bool)
    else:
        mask = np.ones_like(M200_raw, dtype=bool)
    M200 = M200_raw[mask]
    c = c_raw[mask]
    c_err = c_std[mask] if c_std is not None else None
    sersic_n = sersic_n_raw[mask]
    log10_mu_obs = log10_mu_obs_raw[mask] if log10_mu_obs_raw is not None else None
    log10_cov_obs = log10_cov_obs_raw[mask] if log10_cov_obs_raw is not None else None
    print(f"Data points after filtering: {len(M200)} (dropped {len(M200_raw) - len(M200)})")

    if len(M200) < 3:
        print("Not enough valid data points for fitting.")
        return

    # 3. Perform overall non-linear fit
    print("\n# 1. Fitting All Data")
    if mode == 'mcmc':
        print("\nUsing MCMC for fitting...")
        c0_all, alpha_all, c0_std_all, alpha_std_all, log_rmse_all = fit_m200_c_mcmc(log10_mu_obs, log10_cov_obs)
    else:
        print("\nUsing curve_fit for fitting...")
        c0_all, alpha_all, c0_std_all, alpha_std_all, log_rmse_all = fit_m200_c_nonlinear(M200, c, c_std=c_err)

    # 4. Infer optimal Sersic n threshold
    if n_threshold.lower() == 'auto':
        if mode == 'mcmc':
            threshold = infer_sersic_n_threshold_mcmc(M200, c, sersic_n, c_std=c_err)
        else:
            threshold = infer_sersic_n_threshold_grid(M200, c, sersic_n, c_std=c_err)
    else:
        try:
            threshold = float(n_threshold)
            print(f"\n--- Using fixed Sersic n threshold: {threshold:.2f} ---")
        except ValueError:
            print(f"\n--- Invalid Sersic n threshold '{n_threshold}', falling back to auto ---")
            if mode == 'mcmc':
                threshold = infer_sersic_n_threshold_mcmc(M200, c, sersic_n, c_std=c_err)
            else:
                threshold = infer_sersic_n_threshold_grid(M200, c, sersic_n, c_std=c_err)

    # 5. Plot overall results
    plot_m200_c_all(
        M200,
        c,
        sersic_n,
        c0_all,
        alpha_all,
        log_rmse_all,
        c0_fit_std=c0_std_all,
        alpha_fit_std=alpha_std_all,
        threshold=threshold,
    )

    # 6. Split data and perform non-linear fits separately
    mask_high_n = sersic_n >= threshold
    mask_low_n = sersic_n < threshold

    print(f"\n#2. Fitting High Sersic n (>= {threshold:.2f})")
    if mode == 'mcmc':
        c0_high, alpha_high, c0_std_high, alpha_std_high, log_rmse_high = fit_m200_c_mcmc(
            log10_mu_obs[mask_high_n] if log10_mu_obs is not None else None,
            log10_cov_obs[mask_high_n] if log10_cov_obs is not None else None
        )
    else:
        c0_high, alpha_high, c0_std_high, alpha_std_high, log_rmse_high = fit_m200_c_nonlinear(
            M200[mask_high_n], c[mask_high_n],
            c_std=c_err[mask_high_n] if c_err is not None else None
        )

    print(f"\n#3. Fitting Low Sersic n (< {threshold:.2f})")
    if mode == 'mcmc':
        c0_low, alpha_low, c0_std_low, alpha_std_low, log_rmse_low = fit_m200_c_mcmc(
            log10_mu_obs[mask_low_n] if log10_mu_obs is not None else None,
            log10_cov_obs[mask_low_n] if log10_cov_obs is not None else None
        )
    else:
        c0_low, alpha_low, c0_std_low, alpha_std_low, log_rmse_low = fit_m200_c_nonlinear(
            M200[mask_low_n], c[mask_low_n],
            c_std=c_err[mask_low_n] if c_err is not None else None
        )

    # 7. Plot split results
    plot_m200_c_split(
        M200,
        c,
        sersic_n,
        c0_high,
        alpha_high,
        log_rmse_high,
        c0_low,
        alpha_low,
        log_rmse_low,
        c0_high_std=c0_std_high,
        alpha_high_std=alpha_std_high,
        c0_low_std=c0_std_low,
        alpha_low_std=alpha_std_low,
        threshold=threshold,
    )

    plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit c-M200 relation.")
    parser.add_argument('--mode', type=str, choices=['fit', 'mcmc'], default='fit',
                        help="Fitting method to use: 'fit' or 'mcmc'")
    parser.add_argument('--n', type=str, default='auto',
                        help="Sersic n threshold: 'auto' or a float value (e.g., '2.5')")
    parser.add_argument('--filter', type=float, default=None,
                        help="Drop fraction for NRMSE filtering (e.g., 0.10)")
    parser.add_argument('--nmrse', '--nrmse', dest='nrmse', type=float, default=None,
                        help="NRMSE threshold: keep points with nrmse <= value")
    args = parser.parse_args()

    main(mode=args.mode, n_threshold=args.n, filter_drop_fraction=args.filter, nrmse_threshold=args.nrmse)
