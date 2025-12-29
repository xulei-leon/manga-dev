from turtle import width
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent

  # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
def _calc_c_from_M200(M200: float, h: float=0.674) -> float:
    M_pivot_h_inv = 2e12 # in Msun/h
    mass_ratio = M200 / (M_pivot_h_inv / h)
    return 5.74 * (mass_ratio)**(-0.097)

def get_dm_nfw_params(PLATE_IFU: str):
    data_dir = ROOT_DIR / "data"
    dm_nfw_param_file = data_dir / "dm_nfw_param.csv"
    dm_nfw_params = pd.read_csv(dm_nfw_param_file, index_col=0).to_dict(orient='index')

    if PLATE_IFU in dm_nfw_params:
        return dm_nfw_params[PLATE_IFU]
    else:
        return None

def get_all_dm_nfw_params():
    data_dir = ROOT_DIR / "data"
    dm_nfw_param_file = data_dir / "dm_nfw_param.csv"
    dm_nfw_params = pd.read_csv(dm_nfw_param_file, index_col=0).to_dict(orient='index')
    return dm_nfw_params

def get_m200_c():
    nfw_params = get_all_dm_nfw_params()
    if nfw_params is None:
        return None

    # extract M200 and c as numpy arrays
    M200 = np.array([nfw_params[PLATE_IFU]['M200'] for PLATE_IFU in nfw_params])
    c = np.array([nfw_params[PLATE_IFU]['c'] for PLATE_IFU in nfw_params])
    return M200, c


# plot M200 vs c
def plot_m200_c():
    M200_fit, c_fit = get_m200_c()
    if M200_fit is None or c_fit is None:
        print("No DM NFW parameters found.")
        return

    c_calc = _calc_c_from_M200(M200_fit)  # Use the concentration-mass relation function

    plt.figure(figsize=(8, 6))
    plt.scatter(M200_fit, c_fit, alpha=0.7, label='Fitted c', color='blue', linewidths=0.2, edgecolors='k')
    plt.scatter(M200_fit, c_calc, alpha=0.7, label='Calculated c (Dutton+14)', color='red', linewidths=0.2, edgecolors='k')
    plt.xscale('log')
    plt.xlabel('M200 [Msun]')
    plt.ylabel('Concentration c')
    plt.title('DM NFW Parameters: M200 vs Concentration c')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    # plt.savefig(ROOT_DIR / "results" / "m200_vs_c.png")
    plt.show()
    plt.close()

def main():
    plot_m200_c()

if __name__ == "__main__":
    main()