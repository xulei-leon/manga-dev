from warnings import catch_warnings
import matplotlib.pyplot as plt
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
import numpy as np

class MapsUtil:
    def __init__(self, maps_file_path: str):
        self.maps_file_path = maps_file_path
        try:
            self.hdu = fits.open(self.maps_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.maps_file_path}. Please check the path.")
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
    # EMLINE_GVEL: Line-of-sight velocity in km/s of the ionized gas relative to the input guess redshift
    # EMLINE_GVEL_IVAR (Emission Line Gaussian Velocity Inverse Variance)
    # channel_name: 'Ha' for Hα, 'OIII' for [O III], etc.
    ##############################################################################
    def get_eml_vel_map(self, channel_name='Ha-6564') -> tuple[np.ndarray, str, np.ndarray]:
        gas_vel_hdr = self.hdu['EMLINE_GVEL'].header
        velocity_unit = gas_vel_hdr['BUNIT']

        gas_vel_data = self.hdu['EMLINE_GVEL'].data
        gas_mask_data = self.hdu['EMLINE_GVEL_MASK'].data
        gas_ivar_data = self.hdu['EMLINE_GVEL_IVAR'].data

        channel_index = self._channel_dictionary('EMLINE_GVEL').get(channel_name)
        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        gas_vel_channel = gas_vel_data[channel_index, ...]
        gas_mask_channel = gas_mask_data[channel_index, ...]
        masked_velocity_map = np.where(gas_mask_channel == 0, gas_vel_channel, np.nan)

        gas_ivar_channel = gas_ivar_data[channel_index, ...]
        masked_ivar_map = np.where(gas_ivar_channel > 0, gas_ivar_channel, np.nan)

        return masked_velocity_map, velocity_unit, masked_ivar_map


    # STELLAR_VEL
    # Line-of-sight stellar velocity in km/s, relative to the input guess redshift
    def get_stellar_vel_map(self) -> tuple[np.ndarray, str, np.ndarray]:
        """Return the stellar velocity map, its unit, and masked IVAR from the MAPS file."""
        hdr = self.hdu['STELLAR_VEL'].header
        velocity_unit = hdr.get('BUNIT', '')

        vel_data = self.hdu['STELLAR_VEL'].data
        mask_data = self.hdu['STELLAR_VEL_MASK'].data
        ivar_data = self.hdu['STELLAR_VEL_IVAR'].data

        masked_velocity_map = np.where(mask_data == 0, vel_data, np.nan)
        masked_ivar_map = np.where(ivar_data > 0, ivar_data, np.nan)

        return masked_velocity_map, velocity_unit, masked_ivar_map

    # SPX_SKYCOO
    # Sky-right offsets in arcsec
    def get_sky_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        data_skycoo = self.hdu['SPX_SKYCOO'].data
        offset_x = data_skycoo[0, ...]  # X offset in arcsec
        offset_y = data_skycoo[1, ...]  # Y offset in arcsec
        return offset_x, offset_y

    #BIN_SNR
    def get_snr_map(self) -> np.ndarray:
        """Return the SNR map from the MAPS file."""
        snr_data = self.hdu['BIN_SNR'].data
        return snr_data

    #  ECOOPA: Position angle for ellip. coo
    #  ECOOELL: Ellipticity (1-b/a) for ellip. coo
    def get_pa(self) -> tuple[float | None, float | None]:
        hdr = self.hdu['PRIMARY'].header
        pa_val = hdr.get('ECOOPA', None)
        return pa_val
    
    def get_ba(self) -> float | None:
        hdr = self.hdu['PRIMARY'].header
        ellip_val = hdr.get('ECOOELL', None)
        if ellip_val is not None:
            return 1 - ellip_val
        return None


    # BIN_LWELLCOO
    # Light-weighted elliptical polar coordinates of each bin from the galaxy center based on the on-sky coordinates in BIN_LWSKYCOO and the ECOOPA and ECOOELL parameters (typically taken from the NASA-Sloan atlas) in the primary header.
    # r: Lum. weighted elliptical radius (arcsec)
    # R: R h/kpc (kpc/h)
    # azimuth: Lum. weighted elliptical azimuth
    def get_radius_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the radial map from the MAPS file."""
        data = self.hdu['BIN_LWELLCOO'].data
        radius = data[0, ...]
        r_h_kpc = data[2, ...]
        azimuth = data[3, ...]

        return radius, r_h_kpc, azimuth

    # BIN_LWSKYCOO
    def get_skycoo_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the sky coordinate maps (RA, Dec) from the MAPS file."""
        skycoo_data = self.hdu['BIN_LWSKYCOO'].data
        ra_map = skycoo_data[0, ...]
        dec_map = skycoo_data[1, ...]

        return ra_map, dec_map

    def _get_unique_bins(self, layer_index: int) -> np.ndarray:
        """
        Internal helper function to get unique bins and their indices from BINID data.
        """
        bin_id_map = self.hdu['BINID'].data[layer_index]
        flat = bin_id_map.ravel()

        # Exclude invalid bins (-1)
        valid_mask = flat >= 0
        flat_valid = flat[valid_mask]

        _, uindx_valid = np.unique(flat_valid, return_index=True)
        # Map back to the original array indices
        uindx = np.nonzero(valid_mask)[0][uindx_valid]

        return uindx


    def get_stellar_uindx(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_unique_bins(1)

    def get_emli_uindx(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_unique_bins(3)

    # EMLINE_GSIGMA
    # EMLINE_INSTSIGMA
    # sigma_gas^2 = EMLINE_GSIGMA^2 - EMLINE_INSTSIGMA^2
    def get_eml_sigma_map(self, channel_name='Ha-6564') -> tuple[np.ndarray, np.ndarray]:
        """Return the gas velocity dispersion map, its unit, and masked IVAR from the MAPS file."""
        hdr = self.hdu['EMLINE_GSIGMA'].header
        sigma_unit = hdr.get('BUNIT', '')

        sigma_data = self.hdu['EMLINE_GSIGMA'].data
        mask_data = self.hdu['EMLINE_GSIGMA_MASK'].data
        sigma_inst_data = self.hdu['EMLINE_INSTSIGMA'].data

        channel_index = self._channel_dictionary('EMLINE_GSIGMA').get(channel_name)
        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        sigma_channel = sigma_data[channel_index, ...]
        mask_channel = mask_data[channel_index, ...]
        sigma_inst_channel = sigma_inst_data[channel_index, ...]

        sigma_obs = np.where(mask_channel == 0, sigma_channel, np.nan)
        sigma_inst = np.where(mask_channel == 0, sigma_inst_channel, np.nan)

        return sigma_obs, sigma_inst


    # STELLAR_SIGMA: Raw line-of-sight stellar velocity dispersion measurements in km/s. 
    # STELLAR_SIGMACORR: Quadrature correction for STELLAR_SIGMA to obtain the astrophysical velocity dispersion. 
    # sigma_ast^2 = STELLAR_SIGMA^2 - STELLAR_SIGMACORR^2
    def get_stellar_sigma_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stellar velocity dispersion map, its unit, and masked IVAR from the MAPS file."""
        hdr = self.hdu['STELLAR_SIGMA'].header
        sigma_unit = hdr.get('BUNIT', '')

        sigma_data = self.hdu['STELLAR_SIGMA'].data
        mask_data = self.hdu['STELLAR_SIGMA_MASK'].data
        sigma_corr_data = self.hdu['STELLAR_SIGMACORR'].data[0, ...]

        sigma = np.where(mask_data == 0, sigma_data, np.nan)
        sigma_corr = np.where(mask_data == 0, sigma_corr_data, np.nan)

        return sigma, sigma_corr

    ###############################################################################
    # internal utility functions for channel handling
    ###############################################################################

    # Declare a function that creates a dictionary for the columns in the
    # multi-channel extensions
    def _channel_dictionary(self, ext, prefix='C'):
        """
        Construct a dictionary of the channels in a MAPS file.
        """
        channel_dict = {}
        for k, v in self.hdu[ext].header.items():
            if k[:len(prefix)] == prefix:
                try:
                    i = int(k[len(prefix):])-1
                except ValueError:
                    continue
                channel_dict[v] = i
        return channel_dict

    def _channel_units(self, ext, prefix='U'):
        """
        Construct an array with the channel units.
        """
        cu = {}
        for k, v in self.hdu[ext].header.items():
            if k[:len(prefix)] == prefix:
                try:
                    i = int(k[len(prefix):])-1
                except ValueError:
                    continue
                cu[i] = v.strip()
        channel_units = np.empty(max(cu.keys())+1, dtype=object)
        for k, v in cu.items():
            channel_units[k] = v
        return channel_units.astype(str)
