import tomllib
from xml.sax.saxutils import prepare_input_source
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import pymc as pm

# Load configuration file
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    if not config:
        raise ValueError("Error: config.toml file is empty")

# get settings from config
data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")
NFW_PARAM_FILENAME = {}
NFW_PARAM_FILENAME['c-m200'] = config.get("file", {}).get("nfw_param_cm200_filename", "nfw_param_cm200.csv")
NFW_PARAM_FILENAME['shmr'] = config.get("file", {}).get("nfw_param_shmr_filename", "nfw_param_shmr.csv")

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / data_directory
result_dir = data_dir / result_directory


# c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
def _calc_c_from_M200(M200: float, h: float=0.674) -> float:
    M_pivot_h_inv = 2e12 # in Msun/h
    mass_ratio = M200 / (M_pivot_h_inv / h)
    return 5.74 * (mass_ratio)**(-0.097)

# Moster-like SHMR
# Mstar = 2 * N * Mhalo / ( (Mhalo/M1)**(-beta) + (Mhalo/M1)**gamma )
def _calc_Mstar_from_Mhalo(M200: float, M1=10**11.59, N=0.0351, beta=1.376, gamma=0.608):
    x = M200 / M1
    f = 2.0 * N / (x**(-beta) + x**(gamma))
    return f * M200


def fit_m200_mstar_mcmc(M200: np.ndarray, Mstar: np.ndarray, draws: int = 800, tune: int = 800):
    valid_mask = (M200 > 0) & (Mstar > 0)
    if not np.any(valid_mask):
        return None

    log_m200 = np.log10(M200[valid_mask])
    log_mstar = np.log10(Mstar[valid_mask])

    with pm.Model() as model:
        a = pm.Normal("a", mu=0.0, sigma=5.0)
        b = pm.Normal("b", mu=0.0, sigma=5.0)
        c = pm.Normal("c", mu=0.0, sigma=5.0)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        mu = a + b * log_m200 + c * log_m200**2
        pm.Normal("obs", mu=mu, sigma=sigma, observed=log_mstar)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.95,
            progressbar=True,
        )

    return idata

def get_all_nfw_params(mode: str = 'c-m200'):
    nfw_param_file = result_dir / NFW_PARAM_FILENAME[mode]
    nfw_params = pd.read_csv(nfw_param_file, index_col=0).to_dict(orient='index')
    # remove result is not success
    nfw_params = {k: v for k, v in nfw_params.items() if v.get('result') == 'success'}
    return nfw_params

def get_m200_c():
    nfw_params = get_all_nfw_params("c-m200")
    if nfw_params is None:
        return None

    # extract M200 and c as numpy arrays
    M200 = np.array([nfw_params[PLATE_IFU]['M200'] for PLATE_IFU in nfw_params])
    M200_std = np.array([nfw_params[PLATE_IFU]['M200_std'] for PLATE_IFU in nfw_params])
    c = np.array([nfw_params[PLATE_IFU]['c'] for PLATE_IFU in nfw_params])
    c_std = np.array([nfw_params[PLATE_IFU]['c_std'] for PLATE_IFU in nfw_params])
    return M200, M200_std, c, c_std

def get_m200_mstar():
    nfw_params = get_all_nfw_params("shmr")
    if nfw_params is None:
        return None

    # extract M200 and Mstar as numpy arrays
    M200 = np.array([nfw_params[PLATE_IFU]['M200'] for PLATE_IFU in nfw_params])
    Mstar = np.array([nfw_params[PLATE_IFU]['Mstar'] for PLATE_IFU in nfw_params])
    return M200, Mstar

def _fit_loglog_linear(M200: np.ndarray, c: np.ndarray):
    log_m200 = np.log10(M200)
    log_c = np.log10(c)
    A = np.column_stack([log_m200, np.ones_like(log_m200)])
    coeffs, *_ = np.linalg.lstsq(A, log_c, rcond=None)
    slope, intercept = coeffs
    return slope, intercept


def fit_loglog_linear_scipy(M200: np.ndarray, c: np.ndarray):
    """Fit log10(c) = slope * log10(M200) + intercept using scipy.stats.linregress.

    Returns: slope, intercept, slope_err, intercept_err, rvalue, pvalue, sigma_y
    """
    log_m200 = np.log10(M200)
    log_c = np.log10(c)
    n = len(log_m200)
    res = stats.linregress(log_m200, log_c)
    slope = res.slope
    intercept = res.intercept
    slope_err = res.stderr
    rvalue = res.rvalue
    pvalue = res.pvalue

    # residuals and standard error of the regression
    pred = slope * log_m200 + intercept
    resid = log_c - pred
    dof = n - 2
    if dof > 0:
        sigma_y = np.sqrt(np.sum(resid ** 2) / dof)
    else:
        sigma_y = float('nan')

    xbar = np.mean(log_m200)
    Sxx = np.sum((log_m200 - xbar) ** 2)
    if Sxx > 0 and dof > 0:
        intercept_err = sigma_y * np.sqrt(1.0 / n + xbar ** 2 / Sxx)
    else:
        intercept_err = float('nan')

    return slope, intercept, slope_err, intercept_err, rvalue, pvalue, sigma_y


def filter_m200_c_outliers(M200: np.ndarray, c: np.ndarray, sigma: float = 3.0):
    valid_mask = (M200 > 0) & (c > 0)
    if not np.any(valid_mask):
        return None

    M200_valid = M200[valid_mask]
    c_valid = c[valid_mask]

    slope, intercept = _fit_loglog_linear(M200_valid, c_valid)
    log_m200 = np.log10(M200_valid)
    log_c = np.log10(c_valid)
    residuals = log_c - (slope * log_m200 + intercept)

    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        # keep all valid entries
        keep_mask_full = np.zeros_like(valid_mask, dtype=bool)
        keep_mask_full[valid_mask] = True
        return keep_mask_full

    robust_sigma = 1.4826 * mad
    keep_mask = np.abs(residuals) <= sigma * robust_sigma
    keep_mask_full = np.zeros_like(valid_mask, dtype=bool)
    keep_mask_full[valid_mask] = keep_mask
    return keep_mask_full


def fit_m200_c_linear(M200: np.ndarray, c: np.ndarray, M200_std: np.ndarray = None, c_std: np.ndarray = None):
    """Fit log10(c) = slope * log10(M200) + intercept and compute fit metrics.

    Parameters
    ----------
    M200, c       : data arrays (linear units)
    M200_std, c_std : optional 1-sigma uncertainties in the same linear units.
                    When provided, chi-squared and RMSE are uncertainty-weighted.
                    When absent, sigma_y (std of residuals, dof-corrected) is used.

    Returns
    -------
    (slope, intercept, slope_err, intercept_err, chi_squared, rmse, nrmse)

    All residual-based metrics are in log10 space.
    NRMSE is normalised by the full observed log10(c) range.
    """
    if M200 is None or c is None or len(M200) == 0:
        return (None, None, None, None, None, None, None)

    slope, intercept, slope_err, intercept_err, rvalue, pvalue, sigma_y = fit_loglog_linear_scipy(M200, c)

    # residuals in log10 space
    log_m200 = np.log10(M200)
    log_c_obs = np.log10(c)
    log_c_fit = slope * log_m200 + intercept
    residuals = log_c_obs - log_c_fit

    # normalisation range — always the full dataset range
    log_c_range = np.max(log_c_obs) - np.min(log_c_obs)

    if M200_std is not None and c_std is not None:
        # propagate absolute errors → log10-space uncertainty
        # sigma_log10(x) ≈ sigma_x / (x * ln 10)
        log_m200_err = np.where(
            (M200_std > 0) & (M200 > 0),
            M200_std / (M200 * np.log(10)),
            0.0,
        )
        log_c_err = np.where(
            (c_std > 0) & (c > 0),
            c_std / (c * np.log(10)),
            0.0,
        )
        # total effective y-error: propagate x-uncertainty through the slope
        sigma_eff = np.sqrt(log_c_err ** 2 + (slope * log_m200_err) ** 2)
        valid = sigma_eff > 0
        if np.any(valid):
            # weighted chi-squared
            chi_squared = float(np.sum((residuals[valid] / sigma_eff[valid]) ** 2))
            # weighted RMSE: sum(w_i * r_i^2) / sum(w_i), w_i = 1/sigma_i^2
            weights = 1.0 / sigma_eff[valid] ** 2
            chi_squared_reduced = chi_squared / max(np.sum(valid) - 2, 1)
            rmse = float(np.sqrt(np.sum(weights * residuals[valid] ** 2) / np.sum(weights)))
        else:
            # all errors zero — fall through to unweighted
            chi_squared = float(np.sum((residuals / sigma_y) ** 2))
            chi_squared_reduced = chi_squared / max(len(residuals) - 2, 1)
            rmse = float(sigma_y)
    else:
        # no measurement errors: use sigma_y (dof-corrected regression std) as per-point error
        chi_squared = float(np.sum((residuals / sigma_y) ** 2))   # = n - 2 by construction
        chi_squared_reduced = chi_squared / max(len(residuals) - 2, 1)
        rmse = float(sigma_y)   # dof-corrected std of residuals

    nrmse = float(rmse / log_c_range) if log_c_range > 0 else float('nan')

    # print concise summary
    print("--------- M200-c fit results ---------")
    print(f"Fit params")
    print(f" slope      : {slope:.6f} ± {slope_err:.6f}")
    print(f" intercept  : {intercept:.6f} ± {intercept_err:.6f}")
    print("---------------")
    print(f"Fit metrics")
    # print(f" χ²         : {chi_squared:.3e}")
    print(f" χ² reduced : {chi_squared_reduced:.3f}")
    # print(f" RMSE       : {rmse:.3e}")
    print(f" NRMSE      : {nrmse:.3f}")
    # print(f" r          : {rvalue:.3f}")
    # print(f" p          : {pvalue:.3e}")
    print("--------------------------------------")

    return (slope, intercept, slope_err, intercept_err, chi_squared, rmse, nrmse)


# plot M200 vs c
def plot_m200_c():
    M200_raw, M200_std, c_raw, c_std = get_m200_c()
    mask = filter_m200_c_outliers(M200_raw, c_raw)
    if mask is not None:
        M200_raw = M200_raw[mask]
        M200_std = M200_std[mask]
        c_raw = c_raw[mask]
        c_std = c_std[mask]

    if M200_raw is None or c_raw is None:
        print("No DM NFW parameters found.")
        return

    c_calc = _calc_c_from_M200(M200_raw)  # Use the concentration-mass relation function

    slope, intercept, slope_err, intercept_err, chi_squared, rmse, nrmse = fit_m200_c_linear(
        M200_raw, c_raw, M200_std=M200_std, c_std=c_std
    )
    c_fit = None
    c_fit_upper = None
    c_fit_lower = None
    sigma = None
    if slope is not None:
        log_m200 = np.log10(M200_raw)
        log_c_fit = slope * log_m200 + intercept
        c_fit = 10 ** log_c_fit

        log_c_obs = np.log10(c_raw)
        residuals = log_c_obs - log_c_fit
        sigma = np.std(residuals)
        c_fit_upper = 10 ** (log_c_fit + sigma)
        c_fit_lower = 10 ** (log_c_fit - sigma)

    plt.figure(figsize=(12, 6))
    plt.scatter(M200_raw, c_raw, alpha=0.7, label='Raw', color='black', s=20, linewidths=0.2, edgecolors='k')
    # plt.scatter(M200_raw, c_calc, alpha=0.7, label='Calc (Dutton+14)', color='green', s=20, linewidths=0.2, edgecolors='k')
    if c_fit is not None:
        sort_idx = np.argsort(M200_raw)
        m_sorted = M200_raw[sort_idx]
        c_sorted = c_fit[sort_idx]
        plt.plot(m_sorted, c_sorted, color='black', label='Fit (log-log)', linestyle='-')
        if c_fit_upper is not None and c_fit_lower is not None:
            c_upper_sorted = c_fit_upper[sort_idx]
            c_lower_sorted = c_fit_lower[sort_idx]
            plt.fill_between(
                m_sorted,
                c_lower_sorted,
                c_upper_sorted,
                color='red',
                alpha=0.2,
                label='Fit ±1σ'
            )
        # fit metrics are computed and printed inside `fit_m200_c_linear`
    plt.xscale('log')
    plt.xlabel('M200 [Msun]')
    plt.ylabel('c')
    plt.title('DM NFW: M200 vs c')
    # plt.legend()
    # plt.grid(True, which="both", ls="--")
    plt.savefig(result_dir / "m200_c.png")

def plot_m200_mstar():
    M200, Mstar = get_m200_mstar()
    if M200 is None or Mstar is None:
        print("No DM NFW parameters found.")
        return

    Mstar_calc = _calc_Mstar_from_Mhalo(M200)  # Use the Moster-like SHMR function

    idata = fit_m200_mstar_mcmc(M200, Mstar)
    m_sorted = None
    mstar_fit = None
    mstar_lower = None
    mstar_upper = None
    if idata is not None:
        sort_idx = np.argsort(M200)
        m_sorted = M200[sort_idx]
        log_m200_sorted = np.log10(m_sorted)

        posterior = idata.posterior
        a = posterior["a"].values.reshape(-1)
        b = posterior["b"].values.reshape(-1)
        c = posterior["c"].values.reshape(-1)

        mu_samples = (
            a[:, None]
            + b[:, None] * log_m200_sorted[None, :]
            + c[:, None] * log_m200_sorted[None, :] ** 2
        )
        mu_mean = np.mean(mu_samples, axis=0)
        mu_p16 = np.percentile(mu_samples, 16, axis=0)
        mu_p84 = np.percentile(mu_samples, 84, axis=0)

        mstar_fit = 10 ** mu_mean
        mstar_lower = 10 ** mu_p16
        mstar_upper = 10 ** mu_p84

    plt.figure(figsize=(12, 6))
    plt.scatter(M200, Mstar, alpha=0.7, label='Raw', color='black', s=20, linewidths=0.2, edgecolors='k')
    # plt.scatter(M200, Mstar_calc, alpha=0.7, label='Calc (SHMR)', color='green', linewidths=0.2, edgecolors='k')
    if mstar_fit is not None:
        plt.plot(m_sorted, mstar_fit, color='black', linestyle='-', label='MCMC fit (log-log poly)')
        if mstar_lower is not None and mstar_upper is not None:
            plt.fill_between(
                m_sorted,
                mstar_lower,
                mstar_upper,
                color='red',
                alpha=0.2,
                label='Fit ±1σ'
            )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M200 [Msun]')
    plt.ylabel('Mstar [Msun]')
    plt.title('DM NFW: M200 vs Mstar')
    # plt.legend()
    # plt.grid(True, which="both", ls="--")
    plt.savefig(result_dir / "m200_mstar.png")


def main():
    plot_m200_c()
    # plot_m200_mstar()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
