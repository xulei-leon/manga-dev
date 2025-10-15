from pathlib import Path
import requests

BASE_URL = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2"
REDUX_URL_PREFIX = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1"

class FitsUtil:
    data_dir: Path
    drp_dir: Path
    dap_dir: Path
    images_dir: Path

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.drp_dir = self.data_dir / "redux"
        self.dap_dir = self.data_dir / "analysis"
        self.images_dir = self.data_dir / "images"

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

        if not (ret_path).exists():
            print(f"Warning: file {filename} does not exist; it may need to be downloaded first.")
            dl_success = self.dl_maps(plateifu)
            if not dl_success:
                raise FileNotFoundError(f"Unable to obtain MAPS file: {filename}")
        return ret_path

    # drpall file is always the same name
    # drpall-v3_1_1.fits
    def get_drpall_file(self) -> Path:
        ret_path = Path(self.drp_dir / "drpall-v3_1_1.fits")
        if not (ret_path).exists():
            print(f"Warning: file drpall-v3_1_1.fits does not exist; it may need to be downloaded first.")
            self.dl_drpall()
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
            dl_success = self.dl_image(plateifu)
            if not dl_success:
                raise FileNotFoundError(f"Unable to obtain image file: {filename}")
        return ret_path

    # single file, always the same name
    # https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits
    def dl_drpall(self) -> bool:
        url = f"{REDUX_URL_PREFIX}/drpall-v3_1_1.fits"
        target_path = self.drp_dir / "drpall-v3_1_1.fits"
        return self._download_file(url, target_path)


    # example plateifu:
    # "7957-3701"
    # https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7957/images/3701.png
    def dl_image(self, plateifu: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7957-3701'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}.png"
        url = f"{REDUX_URL_PREFIX}/{plate}/images/{ifu}.png"
        target_path = self.images_dir / filename
        return self._download_file(url, target_path, file_type_str="image")

    # example plateifu:
    # "87443-12703"
    # https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/7443/12703/manga-7443-12703-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz 
    def dl_maps(self, plateifu: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7443-12703'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        url = f"{BASE_URL}/{plate}/{ifu}/{filename}"
        target_path = self.dap_dir / filename
        return self._download_file(url, target_path)

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
