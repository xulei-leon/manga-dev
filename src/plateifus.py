from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# my imports
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil


INC_MIN = 25.0  # minimum inclination angle in degrees
INC_MAX = 70.0  # maximum inclination angle in degrees

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"
fits_util = FitsUtil(data_dir)

def galaxy_filter():

    drpall_file = fits_util.get_drpall_file()
    print(f"DRPALL file: {drpall_file}")

    drpall_util = DrpallUtil(drpall_file)

    inc_list, _ = drpall_util.search_plateifu_by_inc(INC_MIN, INC_MAX)
    print(f"-- Galaxies with inclination between {INC_MIN} and {INC_MAX} degrees:")
    print(f"  Total found: {len(inc_list)}")
    print()

    # Final selection: intersection of all three criteria
    final_plateifus = set(inc_list)
    final_plateifus = sorted(list(final_plateifus))

    return final_plateifus

def fits_download(fits_util: FitsUtil, plateifu_list: list[str]):
    total = len(plateifu_list)
    if total == 0:
        print("No plateifu to download.")
        return
    # Use a thread pool to download maps and images concurrently
    max_workers = min(8, total)

    def _process(plateifu: str):
        errors = []
        try:
            fits_util.get_maps_file(plateifu, checksum=True)
        except Exception as e:
            errors.append(f"maps:{e}")

        try:
            fits_util.get_image_file(plateifu)
        except Exception as e:
            errors.append(f"image:{e}")

        return plateifu, errors

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_process, p): p for p in plateifu_list}
        for fut in tqdm(as_completed(futures), total=total, desc="Downloading maps", unit="galaxy"):
            try:
                plateifu, errors = fut.result()
                if errors:
                    tqdm.write(f"Errors for {plateifu}: {', '.join(errors)}")
            except Exception as e:
                plateifu = futures.get(fut, "unknown")
                tqdm.write(f"Unhandled error for {plateifu}: {e}")

def main(is_download: bool = False):
    plateifus = galaxy_filter()
    print(f"== Filter selection of galaxies:")
    print(f"  Total selected galaxies: {len(plateifus)}")

    # dump selected plateifu to a text file
    output_file = data_dir / "plateifus.txt"
    with open(output_file, "w") as f:
        for plateifu in plateifus:
            f.write(f"{plateifu}\n")
    print(f"  Selected plateifus saved to: {output_file}")

    if is_download:
        fits_download(fits_util, plateifus)

    return


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter MaNGA plateifus and download FITS files.")
    parser.add_argument("--download", action="store_true", help="Download FITS files for selected plateifus.")
    args = parser.parse_args()

    is_download = args.download
    main(is_download=is_download)