from pathlib import Path
import numpy as np

# my imports
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil


INC_MIN = 20.0  # minimum inclination angle in degrees
INC_MAX = 75.0  # maximum inclination angle in degrees



root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


def main():

    drpall_file = fits_util.get_drpall_file()
    print(f"DRPALL file: {drpall_file}")

    drpall_util = DrpallUtil(drpall_file)

    plateifu_list = drpall_util.search_galaxy_by_inc(INC_MIN, INC_MAX)
    print(f"Galaxies with inclination between {INC_MIN} and {INC_MAX} degrees:")
    print(f"  Total found: {len(plateifu_list)}")
    print("----------------------------------")
    # for row in plateifu_list:
    #     print(f"  PLATEIFU: {row}")
    print("----------------------------------")


    return

if __name__ == "__main__":
    main()