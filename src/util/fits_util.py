from argparse import FileType
from fileinput import filename
from pathlib import Path
import requests
import hashlib

from astropy.io import fits
from astropy.table import Table
import numpy as np

MAPS_BASE_URL = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2"
REDUX_BASE_URL = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1"
FIREFLY_BASE_URL = "https://data.sdss.org/sas/dr17/manga/spectro/firefly/v3_1_1"

class FitsUtil:
    data_dir: Path
    drp_dir: Path
    dap_dir: Path
    images_dir: Path
    firefly_dir: Path

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.drp_dir = self.data_dir / "redux"
        self.dap_dir = self.data_dir / "analysis"
        self.images_dir = self.data_dir / "images"
        self.firefly_dir = self.data_dir / "firefly"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.drp_dir.mkdir(parents=True, exist_ok=True)
        self.dap_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # example plateifu:
    # "8550-12704"
    # manga-8550-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz
    def get_maps_file(self, plateifu: str) -> Path:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '8550-12704'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        ret_path = Path(self.dap_dir / filename)

        if (ret_path).exists() and (ret_path.with_suffix('.sha256')).exists():
            # checksum for file
            sha256_checksum = self._compute_sha256(ret_path)
            checksum_file = ret_path.with_suffix('.sha256')
            with open(checksum_file, 'r', encoding='utf-8') as cf:
                line = cf.readline().strip()

            parts = line.split(maxsplit=1)
            stored_checksum = parts[0] if parts else ""
            stored_name = ""
            if len(parts) > 1:
                # sha256 files often have: "<hash> *<filename>" (binary mode marker '*')
                stored_name = parts[1].lstrip("*").strip()

            if stored_name and stored_name != filename:
                print(f"Checksum file name mismatch for {ret_path}: expected {filename}, got {stored_name}; re-downloading.")
            elif stored_checksum and sha256_checksum == stored_checksum:
                print(f"MAPS file checksum success: {ret_path}")
                return ret_path
            else:
                print(f"Checksum mismatch for {ret_path}; re-downloading.")
        else:
            print(f"MAPS file or checksum missing: {ret_path}")


        print(f"Warning: file {filename} need to be downloaded.")
        # remove existing file if any
        try:
            if ret_path.exists():
                ret_path.unlink()
            if ret_path.with_suffix('.sha256').exists():
                ret_path.with_suffix('.sha256').unlink()
        except Exception:
            pass

        dl_success = self.dl_maps(plateifu, filename)
        if not dl_success:
            raise FileNotFoundError(f"Unable to obtain MAPS file: {filename}")

        # Create sha256 checksum for file
        sha256_checksum = self._compute_sha256(ret_path)
        checksum_file = ret_path.with_suffix('.sha256')
        # Ensure the checksum file uses Unix newlines (no '\r' at line end on Windows)
        with open(checksum_file, 'w', encoding='utf-8', newline='\n') as cf:
            cf.write(f"{sha256_checksum} *{filename}\n")

        return ret_path

    # drpall file is always the same name
    # drpall-v3_1_1.fits
    def get_drpall_file(self) -> Path:
        filename = "drpall-v3_1_1.fits"
        ret_path = Path(self.drp_dir / filename)
        if not (ret_path).exists():
            print(f"Warning: file {filename} does not exist; it may need to be downloaded first.")
            self.dl_drpall(filename)
        return ret_path

    def get_firefly_file(self) -> Path:
        filename = "manga-firefly-v3_1_1-mastar.fits"
        ret_path = Path(self.firefly_dir / filename)
        if not (ret_path).exists():
            print(f"Warning: file {filename} does not exist; it may need to be downloaded first.")
            self.dl_firefly_mastar(filename)
        return ret_path

    # get image file path
    # example plateifu:
    # "7957-3701"
    # manga-7957-3701.png
    def get_image_file(self, plateifu: str) -> Path:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7957-3701'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}.png"
        ret_path = Path(self.images_dir / filename)

        if not (ret_path).exists():
            print(f"Warning: file {filename} does not exist; it may need to be downloaded first.")
            dl_success = self.dl_image(plateifu, filename)
            if not dl_success:
                raise FileNotFoundError(f"Unable to obtain image file: {filename}")
        return ret_path

    # single file, always the same name
    # https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits
    def dl_drpall(self, filename: str) -> bool:
        url = f"{REDUX_BASE_URL}/{filename}"
        target_path = self.drp_dir / filename
        return self._download_file(url, target_path, file_type_str="DRPALL")

    # https://data.sdss.org/sas/dr17/manga/spectro/firefly/v3_1_1/manga-firefly-v3_1_1-mastar.fits
    def dl_firefly_mastar(self, filename: str) -> bool:
        url = f"{FIREFLY_BASE_URL}/{filename}"
        target_path = self.firefly_dir / filename
        return self._download_file(url, target_path, file_type_str="FIREFLY MASTAR")

    # example plateifu:
    # "7957-3701"
    # https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7957/images/3701.png
    def dl_image(self, plateifu: str, filename: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7957-3701'")
        plate, ifu = plateifu.split("-", 1)
        if not filename:
            filename = f"manga-{plate}-{ifu}.png"
        url = f"{REDUX_BASE_URL}/{plate}/images/{ifu}.png"
        target_path = self.images_dir / filename
        return self._download_file(url, target_path, file_type_str="image")

    # DAPTYPE: HYB10-MILESHC-MASTARHC2
    # https://sdss-mangadap.readthedocs.io/en/latest/gettingstarted.html#gettingstarted-daptype
    # HYB10-MILESHC-MASTARHC2: Same as the above except the hierarchically clustered MaStar stellar spectra are used to fit the stellar continuum in the emission-line module.
    # example plateifu:
    # "87443-12703"
    # https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/7443/12703/manga-7443-12703-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz
    def dl_maps(self, plateifu: str, filename: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7443-12703'")
        plate, ifu = plateifu.split("-", 1)
        if not filename:
            filename = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        url = f"{MAPS_BASE_URL}/{plate}/{ifu}/{filename}"
        target_path = self.dap_dir / filename
        return self._download_file(url, target_path, file_type_str="MAPS")

    # TODO: only used galaxies with Rmax/Rt ≥ 3
    def find_galaxies(self) -> list[str]:
        """Find galaxies that meet criteria from DRPALL file."""
        drpall_file = self.get_drpall_file()
        selected_plateifus = []

        with fits.open(drpall_file) as hdul:
            table = Table(hdul[1].data)
            plateifus = table['PLATEIFU']
            # Placeholder: implement actual criteria check here
            selected_plateifus = [str(pifu) for pifu in plateifus]
        return selected_plateifus

    ##############################################################################
    # Private methods
    ##############################################################################
    def _download_file(self, url: str, target_path: Path, file_type_str: str = "file") -> bool:
        if target_path.exists():
            print(f"{file_type_str} already exists: {target_path}")
            return True

        try:
            print(f"Downloading {file_type_str} from {url}...")
            resp = requests.get(url, stream=True, timeout=60)
        except requests.RequestException as exc:
            print(f"Request for {file_type_str} failed: {exc}")
            return False

        if resp.status_code != 200:
            print(f"Download of {file_type_str} failed HTTP {resp.status_code}: {url}")
            return False

        try:
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except OSError as exc:
            print(f"Failed to write {file_type_str}: {exc}")
            # Try to remove incomplete file
            try:
                if target_path.exists():
                    target_path.unlink()
            except Exception:
                pass
            return False

        print(f"{file_type_str} download complete: {target_path}")
        return True

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 checksum of a file and return the hex digest."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError as exc:
            raise IOError(f"Unable to read file for sha256 computation: {path}: {exc}") from exc
        return h.hexdigest()
