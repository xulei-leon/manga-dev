from pathlib import Path
import logging
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Configure logging (keeps output concise)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class FireflyUtil:
    firefly_file: Path
    hdu: fits.HDUList

    def __init__(self, firefly_file: str):
        self.firefly_file = Path(firefly_file)
        if not self.firefly_file.exists():
            raise FileNotFoundError(f"FITS file not found: {self.firefly_file}")

        try:
            self.hdu = fits.open(self.firefly_file)
        except Exception as e:
            raise Exception(f"Error opening FITS file: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.hdu:
            self.hdu.close()

    ##############################################################################
    # Load FITS table
    ##############################################################################

    def _find_row_index(self, plateifu: str):
        """Return the integer row index that matches plateifu.
        If `columns` is provided, return the value(s) for those column(s) at the found row.
        """
        table = Table(self.hdu[1].data)
        matches = np.where(table['PLATEIFU'] == plateifu)[0]
        if matches.size == 0:
            raise ValueError(f"plateifu not found: {plateifu}")
        if matches.size > 1:
            log.warning("Multiple matches for plateifu %s; using the first match", plateifu)
        idx = int(matches[0])
        return idx


    ##############################################################################
    # interface methods
    ##############################################################################

    # HDU13: SURFACE MASS DENSITY
    # Surface Mass Density, and associated error, derived from the full spectral fit for each Voronoi cell.
    # shape: (10735, 2800, 2)
    def get_stellar_density_cell(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the surface mass density (in solar masses per kpc^2) for the given plateifu."""
        hdu_index = 13
        row_idx = self._find_row_index(plateifu)  # integer row index for galaxy
        data = self.hdu[hdu_index].data  # shape: (10735, 2800, 2)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}")

        data_row = data[row_idx, :, :]  # shapes (2800, 2)

        density = data_row[:, 0]  # log(M⊙/kpc^2)
        density_err = data_row[:, 1]  # error in log(M⊙/kpc^2)

        linear_density = 10**density  # convert log(M⊙/kpc^2) to M⊙/kpc^2
        linear_density_err = linear_density * np.log(10) * density_err  # propagate error
        return linear_density, linear_density_err

    # HDU11: STELLAR MASS
    # Stellar mass, and associated error, derived from the full spectral fit for each Voronoi cell. Different to the global stellar mass.
    # The first two channels give the stellar mass and error per spaxel, the last two channels give the total stellar mass and error of the Voronoi cell.
    # shape: (10735, 2800, 4)
    def get_stellar_mass_cell(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the stellar mass (in solar masses) for the given plateifu."""
        hdu_index = 11
        row_idx = self._find_row_index(plateifu)
        data = self.hdu[hdu_index].data  # shape: (10735, 2800, 4)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}")

        data_row = data[row_idx, :, :]  # shape: (2800, 4)

        mass_log = data_row[:, 2]  # total stellar mass of Voronoi cell in log(M⊙)
        mass_log_err = data_row[:, 3]  # error in log(M⊙)

        mass = 10**mass_log  # convert log(M⊙) to M⊙
        mass_err = mass * np.log(10) * mass_log_err  # propagate error
        return mass, mass_err


    # HDU4: SPATIAL INFORMATION (VORONOI CELL)
    # bin number
    # x-position, y-position and, in elliptical polar coordinates,
    # radius (in units of effective radius) and azimuth for each Voronoi cell.
    # shape: (10735, 2800, 5)
    def get_spatial_info(self, plateifu: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the Voronoi cell information for the given plateifu."""
        hdu_index = 4
        row_idx = self._find_row_index(plateifu)
        data = self.hdu[hdu_index].data  # shape: (10735, 2800, 5)
        if not (0 <= row_idx < data.shape[0]):
            raise ValueError(f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}")

        data_row = data[row_idx, :, :]  # shape: (2800, 5)

        bin_number = data_row[:, 0]
        x_pos = data_row[:, 1]
        y_pos = data_row[:, 2]
        radius_eff = data_row[:, 3]
        azimuth = data_row[:, 4]

        return bin_number, x_pos, y_pos, radius_eff, azimuth

    def get_binid(self, plateifu: str) -> np.ndarray:
        """Get the bin ID map for the given plateifu."""
        binid, _, _, _, _ = self.get_spatial_info(plateifu)
        return binid

    def get_radius_eff(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the effective radius map for the given plateifu."""
        _, _, _, radius_eff, azimuth = self.get_spatial_info(plateifu)
        return radius_eff, azimuth