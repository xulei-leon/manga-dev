from pathlib import Path
import numpy as np

# my imports
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil


INC_MIN = 30.0  # minimum inclination angle in degrees
INC_MAX = 60.0  # maximum inclination angle in degrees

STAR_MASS_MIN = 1e10  # minimum stellar mass in solar masses
STAR_MASS_MAX = 1e11  # maximum stellar mass in solar masses


root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


def main():

    drpall_file = fits_util.get_drpall_file()
    print(f"DRPALL file: {drpall_file}")

    drpall_util = DrpallUtil(drpall_file)

    inc_list = drpall_util.search_plateifu_by_inc(INC_MIN, INC_MAX)
    print(f"Galaxies with inclination between {INC_MIN} and {INC_MAX} degrees:")
    print(f"  Total found: {len(inc_list)}")

    stellar_mass_list = drpall_util.search_plateifu_by_stellar_mass(STAR_MASS_MIN, STAR_MASS_MAX)
    print(f"Galaxies with stellar mass between {STAR_MASS_MIN:.2e} and {STAR_MASS_MAX:.2e}:")
    print(f"  Total found: {len(stellar_mass_list)}")

    return

if __name__ == "__main__":
    main()