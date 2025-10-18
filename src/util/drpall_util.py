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

# Bit masks (use bitwise-or for clarity)
BIT_MASK_3_EXCLUDE = (1 << 19) | (1 << 20) | (1 << 21) | (1 << 27)  # exclude bits in MNGTARG3
BIT_MASK_DRP_FAIL = (1 << 14) | (1 << 30)                         # failure bits in DRP3QUAL


def load_table(fits_path: Path, ext: int = 1) -> Table:
    """Load FITS table from the specified extension and return an astropy Table."""
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
    with fits.open(fits_path) as hdul:
        return Table(hdul[ext].data)


def _find_colname(table: Table, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from candidates or None if none found."""
    colset = set(table.colnames)
    for name in candidates:
        if name in colset:
            return name
    return None


def _get_col_as_int(table: Table, candidates: Iterable[str], length_fallback: int, dtype=np.int64):
    """
    Return a numpy integer array for the first matching column name.
    If no column is found, return an array of zeros with length length_fallback.
    Masked values are filled with 0 before casting.
    """
    name = _find_colname(table, candidates)
    if name is None:
        return np.zeros(length_fallback, dtype=dtype)
    col = table[name]
    # If it's a masked column, fill masked entries with 0
    try:
        arr = np.asarray(col.filled(0))
    except Exception:
        arr = np.asarray(col)
    return arr.astype(dtype, copy=False)


def select_target_galaxies(drpall: Table) -> Table:
    """Select galaxies that are MANGATARG (mngtarg1 or mngtarg3) AND do not have excluded bits."""
    nrows = len(drpall)
    mngtarg1 = _get_col_as_int(drpall, ['MNGTARG1', 'mngtarg1', 'MNGTARG_1'], nrows)
    mngtarg3 = _get_col_as_int(drpall, ['MNGTARG3', 'mngtarg3', 'MNGTARG_3'], nrows)

    cond_a = (mngtarg1 != 0) | (mngtarg3 != 0)
    cond_b = (np.bitwise_and(mngtarg3, BIT_MASK_3_EXCLUDE) == 0)

    sel = cond_a & cond_b
    log.info(f"筛选目标星系后剩余数量: {np.count_nonzero(sel)}")
    return drpall[sel]


def select_high_quality(galaxies: Table) -> Table:
    """Filter out galaxies with DRP3QUAL failure bits set."""
    nrows = len(galaxies)
    drp3qual = _get_col_as_int(galaxies, ['DRP3QUAL', 'drp3qual', 'DRP_3_QUAL'], nrows)
    sel = (np.bitwise_and(drp3qual, BIT_MASK_DRP_FAIL) == 0)
    log.info(f"筛选高质量星系后剩余数量: {np.count_nonzero(sel)}")
    return galaxies[sel]


def unique_by_id(table: Table, id_candidates: Iterable[str]) -> Table:
    """Return a table with unique entries by the first existing id column in id_candidates.
    Keeps the first occurrence of each id (stable). If no id column exists, returns the input table.
    """
    id_col = _find_colname(table, id_candidates)
    if id_col is None:
        log.warning("警告：未找到 'MANGAID' 列，跳过去重步骤。")
        return table

    ids = np.asarray(table[id_col])
    # Preserve order of first occurrence
    _, idx = np.unique(ids, return_index=True)
    idx_sorted = np.sort(idx)
    unique_table = table[idx_sorted]
    log.info(f"最终唯一高质量星系数量: {len(unique_table)}")
    return unique_table


def get_all_fits(drpall_file: str):
    fits_path = Path(drpall_file)
    drpall = load_table(fits_path)

    galaxies = select_target_galaxies(drpall)
    highqual = select_high_quality(galaxies)
    uniquegals = unique_by_id(highqual, ['MANGAID', 'mangaid', 'MANGA_ID', 'MANGA_Id'])

    log.info("--- 筛选完成 ---")
    return uniquegals

# get z_sys from drpall
def get_z_sys(drpall_file, plateifu) -> float | None:
    drpall_hdu = fits.open(drpall_file)
    try:
        drpall_data = drpall_hdu[1].data
        match = drpall_data['plateifu'] == plateifu
        if not np.any(match):
            print(f"No match found for {plateifu} in drpall")
            return None

        # determine available column names in a robust way (avoid comparing Column objects with strings)
        try:
            colnames = [n.lower() for n in drpall_hdu[1].columns.names]
        except Exception:
            # fallback for numpy recarray
            colnames = [n.lower() for n in getattr(drpall_data, 'dtype').names] if getattr(drpall_data, 'dtype', None) else []

        # find z_sys
        z_sys = None
        if 'nsa_z' in colnames:
                        print("Using nsa_z for z_sys")
                        z_sys = drpall_data['nsa_z'][match][0]
        elif 'nsa_zdist' in colnames:
            print("Using nsa_zdist for z_sys")
            z_sys = drpall_data['nsa_zdist'][match][0]
        elif 'z' in colnames:
            print("Using z for z_sys")
            z_sys = drpall_data['z'][match][0]

        # find pa
        pa_deg = None
        if 'nsa_sersic_phi' in colnames:
            print("Using nsa_sersic_phi for pa_deg")
            pa_deg = drpall_data['nsa_sersic_phi'][match][0]
        elif 'nsa_petro_phi' in colnames:
            print("Using nsa_petro_phi for pa_deg")
            pa_deg = drpall_data['nsa_petro_phi'][match][0]
        elif 'elpetro_phi' in colnames:
            print("Using elpetro_phi for pa_deg")
            pa_deg = drpall_data['elpetro_phi'][match][0]

        # find ba
        ba = None
        if 'nsa_sersic_ba' in colnames:
            print("Using nsa_sersic_ba for ba")
            ba = drpall_data['nsa_sersic_ba'][match][0]
        elif 'nsa_petro_ba' in colnames:
            print("Using nsa_petro_ba for ba")
            ba = drpall_data['nsa_petro_ba'][match][0]
        elif 'elpetro_ba' in colnames:
            print("Using elpetro_ba for ba")
            ba = drpall_data['elpetro_ba'][match][0]

        return z_sys, pa_deg, ba


    finally:
        drpall_hdu.close()

