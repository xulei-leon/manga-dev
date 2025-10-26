from operator import le
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from matplotlib import colors
from scipy.special import gamma, gammainc


root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


# PLATE_IFU = "8723-12705"
PLATE_IFU = "8723-12703"

# Gravitational constant in kpc (km/s)^2 / Msun
G = 4.30091727003628e-6 # kpc (km/s)^2 / Msun

# Sersic b_n approximation
def b_n(n):
    return 2.0*n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n)

# Stellar mass to velocity conversion
# Prugniel-Simien approximation by assuming spherical symmetry for the total masses stellar_mass, Re, n -> v(R)
def stellar__mass_to_velocity(stellar_mass: float,  R_kpc: np.ndarray, Re_kpc, n) -> np.ndarray:
    bn = b_n(n)
    # Sigma_e from total mass:
    Sigma_e = stellar_mass / (2.0 * np.pi * Re_kpc**2 * n * np.exp(bn) * bn**(-2.0*n) * gamma(2.0*n))
    p = 1.0 - 0.6097/n + 0.05463/(n*n)
    rho0 = (Sigma_e * bn**(n*(1.0 - p)) / Re_kpc) * (gamma(2.0*n) / (2.0 * gamma(n*(3.0 - p))))
    const = 4.0 * np.pi * rho0 * Re_kpc**3 * n * bn**(-n*(3.0 - p))
    R = np.atleast_1d(R_kpc)
    x = bn * (R / Re_kpc)**(1.0 / n)
    s = n * (3.0 - p)
    lower_gamma = gammainc(s, x) * gamma(s)
    M_enc = const * lower_gamma
    v2 = G * M_enc / R
    v2 = np.where(R==0, 0.0, v2)  # avoid divide by zero
    v2 = np.maximum(v2, 0.0)
    return np.sqrt(v2)


# v_DM²(r) = v_total²(r) - v_stars²(r) - v_gas²(r)
# v_DM(r) = √[max(0, v_total² - v_stars² - v_gas²)]
def calc_dm_velocity_map(total_vel_map: np.ndarray, star_vel_map: np.ndarray, gas_vel_map: np.ndarray) -> np.ndarray:
    total_sq = np.square(total_vel_map)
    star_sq = np.square(star_vel_map)
    gas_sq = np.square(gas_vel_map)
    dm_sq = total_sq - star_sq - gas_sq
    # Ensure no negative values under the square root
    dm_sq = np.maximum(dm_sq, 0)
    dm_vel_map = np.sqrt(dm_sq)
    return dm_vel_map


def main():
    drpall_file = fits_util.get_drpall_file()

    drpall_util = DrpallUtil(drpall_file)
    stellar_mass_1, stellar_mass_2 = drpall_util.get_stellar_mass(PLATE_IFU)
    print(f"Stellar Mass: (Sersic) {stellar_mass_1:,} M solar, (Elpetro) {stellar_mass_2:,} M solar")

    v_stellar = stellar__mass_to_velocity(stellar_mass_1, R_kpc=np.array([1.0, 5.0, 10.0]), Re_kpc=5.0, n=4.0)
    print(f"Stellar Velocity at R=1,5,10 kpc: {v_stellar} km/s")

    return

if __name__ == "__main__":
    main()

