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
        
    def load_table(self, ext: int = 1) -> Table:
        """Load FITS table from the specified extension and return an astropy Table."""
        return Table(self.hdu[ext].data)
        
    def _fetch_scalar_column_value(self, plateifu: str, candidates: Iterable[str]):
        """Return the scalar value from the first matching column name for the given plateifu."""
        table = self.load_table()
        row = table[table['PLATEIFU'] == plateifu]
        if len(row) == 0:
            raise ValueError(f"plateifu {plateifu} not found in Firefly data.")
        for name in candidates:
            if name in row.colnames:
                return row[name][0]
        raise ValueError(f"No matching column found for candidates: {candidates}")


    # key: STELLAR_MASS_VORONOI
    # HDU11: STELLAR MASS
    # Stellar mass, and associated error, derived from the full spectral fit for each Voronoi cell. Different to the global stellar mass. 
    # The first two channels give the stellar mass and error per spaxel, the last two channels give the total stellar mass and error of the Voronoi cell.
    # Units of log(M⊙/spaxel), log(M⊙). Dimensions (4, 2800, 10735).
    def get_stellar_mass(self, plateifu: str) -> float:
        """Get the stellar mass (in solar masses) for the given plateifu."""
        return self._fetch_scalar_column_value(plateifu, ['STELLAR_MASS_VORONOI'])


    # key: SURFACE_MASS_DENSITY_VORONOI
    # HDU13: SURFACE MASS DENSITY
    # Surface Mass Density, and associated error, derived from the full spectral fit for each Voronoi cell.
    # Units of log(M⊙/kpc^2). Dimensions (2, 2800, 10735).
    def get_stellar_surface_mass_density(self, plateifu: str) -> float:
        """Get the surface mass density (in solar masses per kpc^2) for the given plateifu."""
        return self._fetch_scalar_column_value(plateifu, ['SURFACE_MASS_DENSITY_VORONOI'])

    # key: SPATIAL_INFO
    # Contains the spatial information, such as bin number, x-position, y-position and, in elliptical polar coordinates, radius (in units of effective radius) and azimuth for each Voronoi cell.
    def get_spatial_info(self, plateifu: str) -> np.ndarray:
        """Get the 2D spatial info map for the given plateifu."""
        spatial_info = self._fetch_scalar_column_value(plateifu, ['SPATIAL_INFO'])
        radius = spatial_info['RADIUS'] # radius (in units of effective radius) 
        x_pos = spatial_info['X_POS']
        y_pos = spatial_info['Y_POS']
        return radius, x_pos, y_pos

