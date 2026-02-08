import tomllib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pymc as pm

# Load configuration file
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    if not config:
        raise ValueError("Error: config.toml file is empty")

# get settings from config
data_directory = config.get("file", {}).get("data_directory", "data")
result_directory = config.get("file", {}).get("result_directory", "results")
NFW_PARAM_FILENAME = config.get("file", {}).get("nfw_param_filename", "nfw_param.csv")

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
            chains=2,
            cores=1,
            target_accept=0.9,
            progressbar=False,
        )

    return idata

def get_all_nfw_params():
    nfw_param_file = result_dir / NFW_PARAM_FILENAME
    nfw_params = pd.read_csv(nfw_param_file, index_col=0).to_dict(orient='index')
    # remove result is not success
    nfw_params = {k: v for k, v in nfw_params.items() if v.get('result') == 'success'}
    return nfw_params

def get_m200_c():
    nfw_params = get_all_nfw_params()
    if nfw_params is None:
        return None

    # extract M200 and c as numpy arrays
    M200 = np.array([nfw_params[PLATE_IFU]['M200'] for PLATE_IFU in nfw_params])
    c = np.array([nfw_params[PLATE_IFU]['c'] for PLATE_IFU in nfw_params])
    return M200, c

def get_m200_mstar():
    nfw_params = get_all_nfw_params()
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


def filter_m200_c_outliers(M200: np.ndarray, c: np.ndarray, sigma: float = 3.0):
    valid_mask = (M200 > 0) & (c > 0)
    if not np.any(valid_mask):
        return None, None

    M200_valid = M200[valid_mask]
    c_valid = c[valid_mask]

    slope, intercept = _fit_loglog_linear(M200_valid, c_valid)
    log_m200 = np.log10(M200_valid)
    log_c = np.log10(c_valid)
    residuals = log_c - (slope * log_m200 + intercept)

    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        return M200_valid, c_valid

    robust_sigma = 1.4826 * mad
    keep_mask = np.abs(residuals) <= sigma * robust_sigma
    return M200_valid[keep_mask], c_valid[keep_mask]


def fit_m200_c_linear(M200: np.ndarray, c: np.ndarray):
    if M200 is None or c is None or len(M200) == 0:
        return None, None
    return _fit_loglog_linear(M200, c)


# plot M200 vs c
def plot_m200_c():
    M200_raw, c_raw = get_m200_c()
    M200_raw, c_raw = filter_m200_c_outliers(M200_raw, c_raw)

    if M200_raw is None or c_raw is None:
        print("No DM NFW parameters found.")
        return

    c_calc = _calc_c_from_M200(M200_raw)  # Use the concentration-mass relation function

    slope, intercept = fit_m200_c_linear(M200_raw, c_raw)
    c_fit = None
    c_fit_upper = None
    c_fit_lower = None
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
    plt.scatter(M200_raw, c_calc, alpha=0.7, label='Calc (Dutton+14)', color='green', s=20, linewidths=0.2, edgecolors='k')
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
    plt.scatter(M200, Mstar_calc, alpha=0.7, label='Calc (SHMR)', color='green', linewidths=0.2, edgecolors='k')
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
    plot_m200_mstar()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
