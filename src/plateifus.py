from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# my imports
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil


INC_MIN = 25.0  # minimum inclination angle in degrees
INC_MAX = 70.0  # maximum inclination angle in degrees


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
    mass_mean = np.nanmean(mass_values)
    mass_std = np.nanstd(mass_values)
    mass_median = np.nanmedian(mass_values)
    mass_min = mass_mean - 2.0 * mass_std
    mass_max = mass_mean + 2.0 * mass_std

    mass_list, _ = drpall_util.search_plateifu_by_stellar_mass(mass_min, mass_max)

    print(f"-- Galaxies with stellar mass between {mass_min:.2e} and {mass_max:.2e} Msun:")
    print(f"  Min stellar mass : {mass_min:.2e} Msun")
    print(f"  Max stellar mass : {mass_max:.2e} Msun")
    print(f"  Median stellar mass: {mass_median:.2e} Msun")
    print(f"  Total found: {len(mass_list)}")
    print()

    _, r_eff_values = drpall_util.search_plateifu_by_effective_radius(0.0, np.inf)
    r_eff_Q2 = np.nanmedian(r_eff_values)
    _, r_eff_values = drpall_util.search_plateifu_by_effective_radius(0.0, np.inf)
    r_eff_mean = np.nanmean(r_eff_values)
    r_eff_std = np.nanstd(r_eff_values)
    r_eff_median = np.nanmedian(r_eff_values)
    r_eff_min = r_eff_mean - 2.0 * r_eff_std
    r_eff_max = r_eff_mean + 2.0 * r_eff_std

    r_eff_list, _ = drpall_util.search_plateifu_by_effective_radius(r_eff_min, r_eff_max)
    print(f"  Max effective radius : {r_eff_max:.2f} arcsec")
    print(f"  Median effective radius: {r_eff_median:.2f} arcsec")
    print(f"  Total found: {len(r_eff_list)}")
    print()

    # Final selection: intersection of all three criteria
    final_plateifus = set(inc_list)
    # final_plateifus &= set(mass_list)
    # final_plateifus &= set(r_eff_list)

    final_plateifus = sorted(list(final_plateifus))

    return final_plateifus

def fits_download(fits_util: FitsUtil, plateifu_list: list[str]):
    total = len(plateifu_list)
    if total == 0:
        print("No plateifu to download.")
        return

    # Use tqdm to display a progress bar and ensure maps_file is used (e.g., print saved path)
    for plateifu in tqdm(plateifu_list, desc="Downloading maps", unit="galaxy"):
        print(f"\n# {plateifu}")

        try:
            maps_file = fits_util.get_maps_file(plateifu, checksum=True)
            # use the returned maps_file (print path or perform further processing)
            if maps_file:
                tqdm.write(f"Saved: {maps_file}")
        except Exception as e:
            tqdm.write(f"Error processing {plateifu}: {e}")
            print(f"Error processing {plateifu}: {e}")

        try:
            image_file = fits_util.get_image_file(plateifu)
            if image_file:
                tqdm.write(f"Saved: {image_file}")
        except Exception as e:
            tqdm.write(f"Error processing {plateifu} image: {e}")
            print(f"Error processing {plateifu} image: {e}")


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

    fits_download(fits_util, plateifus)
    return


if __name__ == "__main__":
    main()