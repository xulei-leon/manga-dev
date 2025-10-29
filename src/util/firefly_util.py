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
    def get_stellar_density(self, plateifu: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the surface mass density (in solar masses per kpc^2) for the given plateifu."""
        hdu_index = 13
        row_idx = self._find_row_index(plateifu)  # boolean mask for galaxies
        if not np.any(row_idx):
            raise ValueError(f"Row index {row_idx} out of bounds for HDU{hdu_index} with shape {data.shape}")
        
        data = self.hdu[hdu_index].data  # shape: (10735, 2800, 2)
        data_row = data[row_idx, :, :]  # shapes (2800, 2)

        density = data_row[:, 0]  # log(M⊙/kpc^2)
        density_err = data_row[:, 1]  # error in log(M⊙/kpc^2)

        linear_density = 10**density  # convert log(M⊙/kpc^2) to M⊙/kpc^2
        linear_density_err = linear_density * np.log(10) * density_err  # propagate error
        return linear_density, linear_density_err