import tomllib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
M_PIVOT_H_INV = 2e12  # in Msun/h, pivot mass for c-M200 relation
H_0 = 0.674           # Hubble parameter
DM_INTRINSIC_SIGMA_DEX = 0.10  # Intrinsic scatter in dex for c

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

def c_m200_model(M200: np.ndarray, c0: float, alpha: float, h: float = H_0) -> np.ndarray:
    """
    Theoretical non-linear model for c-M200 relation.
    c = c0 * (M200 / (M_pivot_h_inv / h))^alpha
    """
    mass_ratio = M200 / (M_PIVOT_H_INV / h)
    return c0 * (mass_ratio)**alpha

def get_m200_c_data():
    """
    Load M200 and c data from the results CSV file.
    Returns M200, M200_std, c, c_std, sersic_n arrays.
    """
    nfw_param_file = result_dir / NFW_PARAM_CM200_FILENAME
    if not nfw_param_file.exists():
        print(f"Warning: Data file {nfw_param_file} not found.")
        return None, None, None, None, None

    df = pd.read_csv(nfw_param_file, index_col=0)

    # Filter for successful fits
    if 'result' in df.columns:
        df = df[df['result'] == 'success']

    if df.empty:
        print("Warning: No successful fits found in data.")
        return None, None, None, None, None

    M200 = df['M200'].values
    M200_std = df['M200_std'].values if 'M200_std' in df.columns else np.zeros_like(M200)
    c = df['c'].values
    c_std = df['c_std'].values if 'c_std' in df.columns else np.zeros_like(c)
    sersic_n = df['sersic_n'].values if 'sersic_n' in df.columns else np.zeros_like(c)

    return M200, M200_std, c, c_std, sersic_n

def filter_outliers(M200: np.ndarray, c: np.ndarray, sigma_threshold: float = 3.0) -> np.ndarray:
    """
    Filter out outliers based on Median Absolute Deviation (MAD) of residuals in log-log space.
    Returns a boolean mask of valid data points to keep.
    """
    valid_mask = (M200 > 0) & (c > 0)
    if not np.any(valid_mask):
        return valid_mask

    M200_valid = M200[valid_mask]
    c_valid = c[valid_mask]

    # Simple log-log linear fit for outlier detection
    log_m200 = np.log10(M200_valid)
    log_c = np.log10(c_valid)
    A = np.column_stack([log_m200, np.ones_like(log_m200)])
    coeffs, *_ = np.linalg.lstsq(A, log_c, rcond=None)
    slope, intercept = coeffs

    residuals = log_c - (slope * log_m200 + intercept)
    mad = np.median(np.abs(residuals - np.median(residuals)))

    if mad == 0:
        return valid_mask

    robust_sigma = 1.4826 * mad
    keep_mask = np.abs(residuals) <= sigma_threshold * robust_sigma

    final_mask = np.zeros_like(valid_mask, dtype=bool)
    final_mask[valid_mask] = keep_mask
    return final_mask

def fit_m200_c_nonlinear(M200: np.ndarray, c: np.ndarray, c_std: np.ndarray = None, intrinsic_scatter_dex: float = DM_INTRINSIC_SIGMA_DEX):
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
        return c_m200_model(M, c0, alpha, h=H_0)

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

def plot_m200_c_all(M200: np.ndarray, c: np.ndarray, sersic_n: np.ndarray, c0_fit: float, alpha_fit: float, log_rmse: float):
    """
    Plot the M200 vs c data and the overall non-linear fit with residuals, color-coded by Sersic n.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # Split data by Sersic n
    mask_high_n = sersic_n >= 2.5
    mask_low_n = sersic_n < 2.5

    # Plot raw data with color mapping based on sersic_n
    ax1.scatter(M200[mask_high_n], c[mask_high_n], color='red', alpha=0.6, label=r'Data ($n \geq 2.5$)', s=20, edgecolors='none')
    ax1.scatter(M200[mask_low_n], c[mask_low_n], color='blue', alpha=0.6, label=r'Data ($n < 2.5$)', s=20, edgecolors='none')

    # Plot fit
    if c0_fit is not None and alpha_fit is not None:
        # Generate smooth curve for plotting
        m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)
        c_plot = c_m200_model(m_plot, c0_fit, alpha_fit, h=H_0)

        ax1.plot(m_plot, c_plot, color='black', linewidth=2,
                 label=rf'Fit (All): $c_0={c0_fit:.2f}$, $\alpha={alpha_fit:.3f}$')

        # Plot ±1 sigma band in log space
        if log_rmse is not None and not np.isnan(log_rmse):
            c_upper = 10**(np.log10(c_plot) + log_rmse)
            c_lower = 10**(np.log10(c_plot) - log_rmse)
            ax1.fill_between(m_plot, c_lower, c_upper,
                             color='black', alpha=0.2, label=r'Fit $\pm 1\sigma$ (log)')

        # Calculate residuals (in log space)
        c_pred = c_m200_model(M200, c0_fit, alpha_fit, h=H_0)
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

def plot_m200_c_split(M200: np.ndarray, c: np.ndarray, sersic_n: np.ndarray,
                c0_high: float, alpha_high: float, rmse_high: float,
                c0_low: float, alpha_low: float, rmse_low: float):
    """
    Plot the M200 vs c data and the non-linear fits with residuals, separated by Sersic n.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # Split data by Sersic n
    mask_high_n = sersic_n >= 2.5
    mask_low_n = sersic_n < 2.5

    # Plot raw data with color mapping based on sersic_n
    ax1.scatter(M200[mask_high_n], c[mask_high_n], color='red', alpha=0.6, label=r'Data ($n \geq 2.5$)', s=20, edgecolors='none')
    ax1.scatter(M200[mask_low_n], c[mask_low_n], color='blue', alpha=0.6, label=r'Data ($n < 2.5$)', s=20, edgecolors='none')

    m_plot = np.logspace(np.log10(np.min(M200)), np.log10(np.max(M200)), 100)

    # Plot high n fit
    if c0_high is not None and alpha_high is not None:
        c_plot_high = c_m200_model(m_plot, c0_high, alpha_high, h=H_0)
        ax1.plot(m_plot, c_plot_high, color='darkred', linewidth=2,
                 label=rf'Fit ($n \geq 2.5$): $c_0={c0_high:.2f}$, $\alpha={alpha_high:.3f}$')

        if rmse_high is not None and not np.isnan(rmse_high):
            c_upper = 10**(np.log10(c_plot_high) + rmse_high)
            c_lower = 10**(np.log10(c_plot_high) - rmse_high)
            ax1.fill_between(m_plot, c_lower, c_upper, color='red', alpha=0.15)

        c_pred_high = c_m200_model(M200[mask_high_n], c0_high, alpha_high, h=H_0)
        log_res_high = np.log10(c[mask_high_n]) - np.log10(c_pred_high)
        ax2.scatter(M200[mask_high_n], log_res_high, color='red', alpha=0.6, s=20, edgecolors='none')

    # Plot low n fit
    if c0_low is not None and alpha_low is not None:
        c_plot_low = c_m200_model(m_plot, c0_low, alpha_low, h=H_0)
        ax1.plot(m_plot, c_plot_low, color='darkblue', linewidth=2,
                 label=rf'Fit ($n < 2.5$): $c_0={c0_low:.2f}$, $\alpha={alpha_low:.3f}$')

        if rmse_low is not None and not np.isnan(rmse_low):
            c_upper = 10**(np.log10(c_plot_low) + rmse_low)
            c_lower = 10**(np.log10(c_plot_low) - rmse_low)
            ax1.fill_between(m_plot, c_lower, c_upper, color='blue', alpha=0.15)

        c_pred_low = c_m200_model(M200[mask_low_n], c0_low, alpha_low, h=H_0)
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


def main():
    # 1. Load data
    M200_raw, M200_std, c_raw, c_std, sersic_n_raw = get_m200_c_data()
    if M200_raw is None or c_raw is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Filter outliers
    mask = filter_outliers(M200_raw, c_raw)
    M200 = M200_raw[mask]
    c = c_raw[mask]
    c_err = c_std[mask] if c_std is not None else None
    sersic_n = sersic_n_raw[mask]

    if len(M200) < 3:
        print("Not enough valid data points for fitting.")
        return

    # 3. Perform overall non-linear fit
    print("\n--- Fitting All Data ---")
    c0_all, alpha_all, _, _, rmse_all = fit_m200_c_nonlinear(M200, c, c_std=c_err)

    # 4. Plot overall results
    plot_m200_c_all(M200, c, sersic_n, c0_all, alpha_all, rmse_all)

    # 5. Split data and perform non-linear fits separately
    mask_high_n = sersic_n >= 2.5
    mask_low_n = sersic_n < 2.5

    print("\n--- Fitting High Sersic n (>= 2.5) ---")
    c0_high, alpha_high, _, _, rmse_high = fit_m200_c_nonlinear(
        M200[mask_high_n], c[mask_high_n],
        c_std=c_err[mask_high_n] if c_err is not None else None
    )

    print("\n--- Fitting Low Sersic n (< 2.5) ---")
    c0_low, alpha_low, _, _, rmse_low = fit_m200_c_nonlinear(
        M200[mask_low_n], c[mask_low_n],
        c_std=c_err[mask_low_n] if c_err is not None else None
    )

    # 6. Plot split results
    plot_m200_c_split(M200, c, sersic_n, c0_high, alpha_high, rmse_high, c0_low, alpha_low, rmse_low)

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
