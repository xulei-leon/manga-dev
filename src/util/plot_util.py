import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
try:
    # Prefer relative import when used as a package
    from .fits_util import FitsUtil
except ImportError:
    # Fallback for direct script execution or non-package layouts
    from fits_util import FitsUtil

class PlotUtil:
    fits_util = None

    def __init__(self, fits_util: FitsUtil) -> None:
        self.fits_util = fits_util

    # show velocity map and the image_file side-by-side
    # load image_file (supports FITS and common image formats)
    def plot_galaxy_image(self, plateifu):
        image_file = self.fits_util.get_image_file(plateifu)
        if image_file is None or not image_file.exists():
            print(f"Warning: image file for {plateifu} does not exist.")
            return

        try:
            img = plt.imread(str(image_file))
        except Exception:
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        if img.ndim == 2:
            ax.imshow(img, origin="upper", cmap="gray")
        else:
            ax.imshow(img, origin="upper")

        ax.set_title(f"Galaxy Image ({plateifu})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        plt.show()

    # Plots the binned velocity map using unique bin indices.
    def plot_vel_map(self, vel_map, uindx, ra_map, dec_map, pa_rad=None, title: str=""):
        """
        Plots the binned velocity map using unique bin indices.

        Args:
            vel_map (np.ndarray): The 2D velocity map.
            uindx (np.ndarray): 1D array of unique indices to flatten the maps.
            ra_map (np.ndarray): The 2D Right Ascension map.
            dec_map (np.ndarray): The 2D Declination map.
            pa_rad (float, optional): Position Angle in radians, measured from North to East.
                                    If provided, a line indicating the major axis is drawn.
            title (str, optional): The title for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Flatten maps and select unique binned data
        ra_flat, dec_flat, vel_flat = ra_map.ravel(), dec_map.ravel(), vel_map.ravel()
        ra_u, dec_u, vel_u = ra_flat[uindx], dec_flat[uindx], vel_flat[uindx]

        # Filter out non-finite velocity values for color scaling
        valid_vel_mask = np.isfinite(vel_u)
        vel_u_clean = vel_u[valid_vel_mask]

        # Set color normalization based on velocity percentiles
        if vel_u_clean.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            p_low, p_high = np.nanpercentile(vel_u_clean, [2, 98])
            vmax = max(abs(p_low), abs(p_high))
            vmin = -vmax
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        # Create the scatter plot for the binned velocity data
        sc = ax.scatter(ra_u, dec_u, c=vel_u, cmap='RdBu_r', norm=norm, s=30, edgecolors='face', alpha=0.9)

        # Add a colorbar
        cbar = fig.colorbar(sc, ax=ax, label="Velocity (km/s)")
        cbar.set_ticks([vmin, 0, vmax])

        # Draw the major axis line if pa_rad is provided
        if pa_rad is not None and ra_u[valid_vel_mask].size > 1:
            pa_rad = pa_rad % (2 * np.pi)  # Normalize PA to [0, 2π]
            # Calculate the center of the galaxy from the valid data points
            ra_center = np.mean(ra_u[valid_vel_mask])
            dec_center = np.mean(dec_u[valid_vel_mask])

            # Determine the line length based on the data extent
            ra_range = np.ptp(ra_u[valid_vel_mask])
            dec_range = np.ptp(dec_u[valid_vel_mask])
            line_length = 0.6 * np.hypot(ra_range, dec_range)

            # Calculate line endpoints. PA is from North (+Dec) to East (-RA, as axis is inverted).
            # This corresponds to a clockwise angle from the positive y-axis.
            dx = -line_length * np.sin(pa_rad)
            dy = line_length * np.cos(pa_rad)

            # Plot the line representing the major axis
            ax.plot([ra_center - dx, ra_center + dx],
                    [dec_center - dy, dec_center + dy],
                    color='gray', linestyle='--', linewidth=1.5, label='Major Axis (PA)')
            ax.legend()

        # Set plot labels and title
        ax.set_title(f"{title} Binned Velocity Map")
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")

        # Invert RA axis for standard astronomical orientation (East to the left)
        ax.invert_xaxis()
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()
        plt.show()

    # plot r-v curve
    def plot_rv_curve(self, r_rot_map: np.ndarray, v_rot_map: np.ndarray, title: str="", v_rot2_map: np.ndarray=None, title2: str=""):
        # Keep signs consistent: if v_rot < 0, set r_rot negative; else positive
        r_rot_map = np.asarray(r_rot_map, dtype=float)
        v_rot_map = np.asarray(v_rot_map, dtype=float)
        r_signed = np.where(v_rot_map < 0, -np.abs(r_rot_map), np.abs(r_rot_map))

        # Mask invalid values for gas rotation
        valid_gas = np.isfinite(r_signed) & np.isfinite(v_rot_map)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(r_signed[valid_gas], v_rot_map[valid_gas], s=2, color='red', alpha=0.2, label=f'{title} Rotation Velocity')

        # Plot stellar velocity if provided
        if v_rot2_map is not None:
            v_rot2_map = np.asarray(v_rot2_map, dtype=float)
            # Mask invalid values for stellar velocity
            valid_stellar = np.isfinite(r_signed) & np.isfinite(v_rot2_map)
            ax.scatter(r_signed[valid_stellar], v_rot2_map[valid_stellar], s=2, color='blue', alpha=0.2, label=f'{title2} Velocity')

        ax.set_title(f"Galaxy Rotation Curve (R-V)")
        ax.set_xlabel("Radius R (kpc/h)")
        ax.set_ylabel("Velocity V (km/s)")
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        plt.show()
