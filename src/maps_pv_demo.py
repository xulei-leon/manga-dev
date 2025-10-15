
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import warnings
import util.maps_util as mu
from util.fits_util import FitsUtil
from pathlib import Path

plateifu = "8550-12704"

root_dir = Path(__file__).resolve().parent.parent
fits_util = FitsUtil(root_dir / "data")
maps_file = fits_util.get_maps_file(plateifu)
drpall_file = fits_util.get_drpall_file()
image_file = fits_util.get_image_file(plateifu)

print(f"Plate-IFU {plateifu} MAPS")
velocity_map, unit = mu.get_vel_from_map(maps_file)
print(f"Velocity map shape: {velocity_map.shape}, Unit: {unit}")
print(f"Velocity: ({np.nanmin(velocity_map):.3f} - {np.nanmax(velocity_map):.3f}) {unit}")

# Optional visualization
# show velocity map and the image_file side-by-side
# load image_file (supports FITS and common image formats)
img_path = Path(image_file)
img_data = None
try:
    if img_path.suffix.lower() in (".fits", ".fit", ".fts"):
        img_data = fits.getdata(str(img_path))
        # drop singleton dimensions if present
        img_data = np.squeeze(img_data)
    else:
        img_data = plt.imread(str(img_path))
except Exception as e:
    print(f"Could not read image_file {image_file}: {e}")
    img_data = None

# create subplots with image on the left and velocity map on the right
fig, (ax_img, ax_vel) = plt.subplots(1, 2, figsize=(14, 6))

# image (left)
if img_data is None:
    ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center")
    ax_img.set_axis_off()
else:
    # handle grayscale and RGB; handle channel-first FITS cubes
    if img_data.ndim == 2:
        ax_img.imshow(img_data, origin="lower", cmap="gray")
    elif img_data.ndim == 3:
        # if shape is (channels, y, x) move channels to last axis
        if img_data.shape[0] in (3, 4) and img_data.shape[0] != img_data.shape[-1]:
            img_disp = np.moveaxis(img_data, 0, -1)
        else:
            img_disp = img_data
        ax_img.imshow(img_disp, origin="lower")
    else:
        ax_img.imshow(np.squeeze(img_data), origin="lower", cmap="gray")
    ax_img.set_title("Galaxy Image")
    ax_img.set_xlabel("X")
    ax_img.set_ylabel("Y")

# velocity map (right)
im0 = ax_vel.imshow(velocity_map, origin="lower", cmap="coolwarm")
fig.colorbar(im0, ax=ax_vel, label=f'H-alpha Line-of-Sight Velocity ({unit})')
ax_vel.set_title("Galaxy Velocity Map")
ax_vel.set_xlabel("X Spaxel")
ax_vel.set_ylabel("Y Spaxel")

plt.tight_layout()
plt.show()