from operator import le
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.constants as const
from scipy import stats
from scipy.optimize import curve_fit


from matplotlib import colors
from scipy.special import gamma, gammainc

# my imports
from util.fits_util import FitsUtil
from util.drpall_util import DrpallUtil
from util.firefly_util import FireflyUtil
from util.maps_util import MapsUtil

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


# PLATE_IFU = "8723-12705"
PLATE_IFU = "8723-12703"

# Gravitational constant in kpc (km/s)^2 / Msun
G = 4.30091727003628e-6 # kpc (km/s)^2 / Msun

# Sersic b_n approximation
def b_n(n):
    return 2.0*n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n)


def calc_vel_stellar(density_stellar, radius_map):
    return


def main():
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)

    stellar_mass_1, stellar_mass_2 = drpall_util.get_stellar_mass(PLATE_IFU)
    print(f"Stellar Mass (DRPALL): (Sersic) {stellar_mass_1:,} M solar, (Elpetro) {stellar_mass_2:,} M solar")

    mass_stellar, mass_stellar_err = firefly_util.get_stellar_mass(PLATE_IFU)
    print(f"Stellar Mass shape: {mass_stellar.shape}")
    print(f"Stellar Mass (FIREFLY) total: {np.nansum(mass_stellar):,.1f} M solar")

    density_stellar, density_stellar_err = firefly_util.get_stellar_density(PLATE_IFU)
    print(f"Stellar Surface Mass Density shape: {density_stellar.shape}")

    uindx = firefly_util.get_voronoi_binid(PLATE_IFU)
    print(f"Unique indices shape: {uindx.shape}")

    radius, azimuth = firefly_util.get_radius_eff(PLATE_IFU)
    print(f"Radius shape: {radius.shape}, Azimuth shape: {azimuth.shape}")

    # radius, r_h_kpc, azimuth = maps_util.get_radius_map()
    # print(f"Radius shape: {radius.shape}, r_h_kpc shape: {r_h_kpc.shape}, azimuth shape: {azimuth.shape}")

    return

if __name__ == "__main__":
    main()

