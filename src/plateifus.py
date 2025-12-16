from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# my imports
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil


INC_MIN = 25.0  # minimum inclination angle in degrees
INC_MAX = 65.0  # maximum inclination angle in degrees

IQR_FACTOR = 0.0  # interquartile range factor for stellar mass and effective radius


root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")


def galaxy_filter():

    drpall_file = fits_util.get_drpall_file()
    print(f"DRPALL file: {drpall_file}")

    drpall_util = DrpallUtil(drpall_file)

    inc_list, _ = drpall_util.search_plateifu_by_inc(INC_MIN, INC_MAX)
    print(f"-- Galaxies with inclination between {INC_MIN} and {INC_MAX} degrees:")
    print(f"  Total found: {len(inc_list)}")
    print()

    _, mass_values = drpall_util.search_plateifu_by_stellar_mass(0.0, np.inf)
    mass_Q2 = np.nanmedian(mass_values)
    mass_Q1 = np.nanpercentile(mass_values, 25)
    mass_Q3 = np.nanpercentile(mass_values, 75)
    mass_median = mass_Q2
    mass_IQR = mass_Q3 - mass_Q1
    mass_min= mass_Q1 - IQR_FACTOR * mass_IQR
    mass_max = mass_Q3 + IQR_FACTOR * mass_IQR

    mass_list, _ = drpall_util.search_plateifu_by_stellar_mass(mass_min, mass_max)

    print(f"-- Galaxies with stellar mass between {mass_min:.2e} and {mass_max:.2e} Msun:")
    print(f"  Min stellar mass : {mass_min:.2e} Msun")
    print(f"  Max stellar mass : {mass_max:.2e} Msun")
    print(f"  Median stellar mass: {mass_median:.2e} Msun")
    print(f"  Total found: {len(mass_list)}")
    print()


    # plot histogram of stellar mass
    # plt.hist(mass_values, bins=100, edgecolor='black')
    # plt.title("Histogram of Stellar Mass")
    # plt.xlabel("Stellar Mass (Msun)")
    # plt.ylabel("Number of Galaxies")
    # plt.yscale('log')
    # plt.grid(True)
    # plt.show()

    _, r_eff_values = drpall_util.search_plateifu_by_effective_radius(0.0, np.inf)
    r_eff_Q2 = np.nanmedian(r_eff_values)
    r_eff_Q1 = np.nanpercentile(r_eff_values, 25)
    r_eff_Q3 = np.nanpercentile(r_eff_values, 75)
    r_eff_median = r_eff_Q2
    r_eff_IQR = r_eff_Q3 - r_eff_Q1
    r_eff_min= r_eff_Q1 - IQR_FACTOR * r_eff_IQR
    r_eff_max = r_eff_Q3 + IQR_FACTOR * r_eff_IQR

    r_eff_list, _ = drpall_util.search_plateifu_by_effective_radius(r_eff_min, r_eff_max)
    print(f"-- Galaxies with effective radius between {r_eff_min:.2f} and {r_eff_max:.2f} arcsec:")
    print(f"  Min effective radius : {r_eff_min:.2f} arcsec")
    print(f"  Max effective radius : {r_eff_max:.2f} arcsec")
    print(f"  Median effective radius: {r_eff_median:.2f} arcsec")
    print(f"  Total found: {len(r_eff_list)}")
    print()

    # plot histogram of effective radius
    # plt.hist(r_eff_values, bins=100, edgecolor='black')
    # plt.title("Histogram of Effective Radius")
    # plt.xlabel("Effective Radius (arcsec)")
    # plt.ylabel("Number of Galaxies")
    # plt.grid(True)
    # plt.show()

    # Final selection: intersection of all three criteria
    final_plateifus = set(inc_list) & set(mass_list) & set(r_eff_list)
    final_plateifus = sorted(list(final_plateifus))

    return final_plateifus

def maps_download(fits_util: FitsUtil, plateifu_list: list[str]):
    total = len(plateifu_list)
    if total == 0:
        print("No plateifu to download.")
        return

    # Use tqdm to display a progress bar and ensure maps_file is used (e.g., print saved path)
    for plateifu in tqdm(plateifu_list, desc="Downloading maps", unit="galaxy"):
        try:
            maps_file = fits_util.get_maps_file(plateifu)
            # use the returned maps_file (print path or perform further processing)
            if maps_file:
                tqdm.write(f"Saved: {maps_file}")
        except Exception as e:
            tqdm.write(f"Error processing {plateifu}: {e}")
            print(f"Error processing {plateifu}: {e}")



def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    fits_util = FitsUtil(data_dir)

    plateifus = galaxy_filter()
    print(f"== Filter selection of galaxies:")
    print(f"  Total selected galaxies: {len(plateifus)}")

    # dump selected plateifu to a text file
    output_file = data_dir / "plateifus.txt"
    with open(output_file, "w") as f:
        for plateifu in plateifus:
            f.write(f"{plateifu}\n")
    print(f"  Selected plateifus saved to: {output_file}")

    maps_download(fits_util, plateifus)
    return


if __name__ == "__main__":
    main()