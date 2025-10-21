from pathlib import Path
import argparse
import logging
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Configure logging (keeps output concise)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class DrpallUtil:
    # Bit masks (use bitwise-or for clarity)
    BIT_MASK_3_EXCLUDE = (1 << 19) | (1 << 20) | (1 << 21) | (1 << 27)  # exclude bits in MNGTARG3
    BIT_MASK_DRP_FAIL = (1 << 14) | (1 << 30)                         # failure bits in DRP3QUAL

    def __init__(self, drpall_file: str):
        self.drpall_file = Path(drpall_file)
        if not self.drpall_file.exists():
            raise FileNotFoundError(f"FITS file not found: {self.drpall_file}")

    @staticmethod
    def _load_table(fits_path: Path, ext: int = 1) -> Table:
        """Load FITS table from the specified extension and return an astropy Table."""
        with fits.open(fits_path) as hdul:
            return Table(hdul[ext].data)

    @staticmethod
    def _find_colname(table: Table, candidates: Iterable[str]) -> Optional[str]:
        """Return the first matching column name from candidates or None if none found."""
        colset = set(table.colnames)
        for name in candidates:
            if name in colset:
                return name
        return None

    def _get_col_as_int(self, table: Table, candidates: Iterable[str], length_fallback: int, dtype=np.int64):
        """
        Return a numpy integer array for the first matching column name.
        If no column is found, return an array of zeros with length length_fallback.
        Masked values are filled with 0 before casting.
        """
        name = self._find_colname(table, candidates)
        if name is None:
            return np.zeros(length_fallback, dtype=dtype)
        col = table[name]
        # If it's a masked column, fill masked entries with 0
        try:
            arr = np.asarray(col.filled(0))
        except Exception:
            arr = np.asarray(col)
        return arr.astype(dtype, copy=False)

    def select_target_galaxies(self, drpall: Table) -> Table:
        """Select galaxies that are MANGATARG (mngtarg1 or mngtarg3) AND do not have excluded bits."""
        nrows = len(drpall)
        mngtarg1 = self._get_col_as_int(drpall, ['MNGTARG1', 'mngtarg1', 'MNGTARG_1'], nrows)
        mngtarg3 = self._get_col_as_int(drpall, ['MNGTARG3', 'mngtarg3', 'MNGTARG_3'], nrows)

        cond_a = (mngtarg1 != 0) | (mngtarg3 != 0)
        cond_b = (np.bitwise_and(mngtarg3, self.BIT_MASK_3_EXCLUDE) == 0)

        sel = cond_a & cond_b
        log.info(f"Remaining galaxies after target selection: {np.count_nonzero(sel)}")
        return drpall[sel]

    def select_high_quality(self, galaxies: Table) -> Table:
        """Filter out galaxies with DRP3QUAL failure bits set."""
        nrows = len(galaxies)
        drp3qual = self._get_col_as_int(galaxies, ['DRP3QUAL', 'drp3qual', 'DRP_3_QUAL'], nrows)
        sel = (np.bitwise_and(drp3qual, self.BIT_MASK_DRP_FAIL) == 0)
        log.info(f"Remaining galaxies after high-quality selection: {np.count_nonzero(sel)}")
        return galaxies[sel]

    def unique_by_id(self, table: Table, id_candidates: Iterable[str]) -> Table:
        """Return a table with unique entries by the first existing id column in id_candidates.
        Keeps the first occurrence of each id (stable). If no id column exists, returns the input table.
        """
        id_col = self._find_colname(table, id_candidates)
        if id_col is None:
            log.warning("Warning: 'MANGAID' column not found, skipping deduplication.")
            return table

        ids = np.asarray(table[id_col])
        # Preserve order of first occurrence
        _, idx = np.unique(ids, return_index=True)
        idx_sorted = np.sort(idx)
        unique_table = table[idx_sorted]
        log.info(f"Final count of unique high-quality galaxies: {len(unique_table)}")
        return unique_table

    def get_all_fits(self):
        drpall = self._load_table(self.drpall_file)

        galaxies = self.select_target_galaxies(drpall)
        highqual = self.select_high_quality(galaxies)
        uniquegals = self.unique_by_id(highqual, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])

        log.info("--- Selection completed ---")
        return uniquegals

    def _fetch_scalar_column_value(self, plateifu: str, candidates: Iterable[str]):
        """Open drpall, find row matching plateifu and return first available candidate column value or None."""
        with fits.open(self.drpall_file) as hdul:
            # get original column names robustly
            try:
                orig_names = list(hdul[1].columns.names)
            except Exception:
                orig_names = list(getattr(hdul[1].data, "dtype").names or [])
            lower_names = [n.lower() for n in orig_names]

            # find plateifu column
            if "plateifu" not in lower_names:
                log.info("The 'plateifu' column was not found in the drpall file")
                return None
            plateifu_col = orig_names[lower_names.index("plateifu")]

            data = hdul[1].data
            match = data[plateifu_col] == plateifu
            if not np.any(match):
                log.info(f"No match found for {plateifu} in drpall")
                return None

            # find first existing candidate column and return its scalar value
            for cand in candidates:
                lc = cand.lower()
                if lc in lower_names:
                    colname = orig_names[lower_names.index(lc)]
                    return data[colname][match][0]
            return None

    # z_sys: The systemic redshift of the galaxy.
    def get_z_sys(self, plateifu: str) -> float | None:
        """Return z_sys for plateifu using available columns (nsa_z, nsa_zdist, z) or None."""
        val = self._fetch_scalar_column_value(plateifu, ["nsa_z", "nsa_zdist", "z"])
        if val is not None:
            log.info(f"Using z sys value: {val}")
        return float(val) if val is not None else None

    # phi: The position angle of the major axis of the galaxy, measured from north to east.
    # ba: The axis ratio (b/a) of the galaxy, where 'b' is the length of the minor axis and 'a' is the length of the major axis.
    def get_phi_ba(self, plateifu: str) -> tuple[float | None, float | None]:
        """Return (position angle in degrees, axis ratio b/a) for plateifu using available columns or (None, None)."""
        phi_val = self._fetch_scalar_column_value(plateifu, ["nsa_elpetro_phi"])
        ba_val = self._fetch_scalar_column_value(plateifu, ["nsa_elpetro_ba"])

        phi = float(phi_val) if phi_val is not None else None
        ba = float(ba_val) if ba_val is not None else None
        return (phi, ba)

    # find all columns containing the keywords
    def search_columns(self, plateifu: str, keywords: str) -> list[str]:
        """Return a list of column names in drpall that contain the specified keywords for the given plateifu."""
        with fits.open(self.drpall_file) as hdul:
            # get original column names robustly
            try:
                orig_names = list(hdul[1].columns.names)
            except Exception:
                orig_names = list(getattr(hdul[1].data, "dtype").names or [])
            lower_names = [n.lower() for n in orig_names]

            # find plateifu column
            if "plateifu" not in lower_names:
                log.info("The 'plateifu' column was not found in the drpall file")
                return []
            plateifu_col = orig_names[lower_names.index("plateifu")]

            data = hdul[1].data
            match = data[plateifu_col] == plateifu
            if not np.any(match):
                log.info(f"No match found for {plateifu} in drpall")
                return []

            # find all columns containing the keywords
            keywords_lower = keywords.lower()
            matching_cols = [name for name in orig_names if keywords_lower in name.lower()]
            return matching_cols
        
    def dump_info(self) -> str:
        """Return a string summary of the DRPALL file."""
        try:
            drpall = self._load_table(self.drpall_file)
            nrows = len(drpall)
            info = f"DRPALL file: {self.drpall_file}\n"
            info += f"Number of entries: {nrows}\n"
            info += f"Columns: {', '.join(drpall.colnames)}\n"
            return info
        except Exception as e:
            raise Exception(f"Error dumping DRPALL info: {e}")