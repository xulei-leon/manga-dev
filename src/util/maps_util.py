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
    def get_gvel_map(self, channel_name='Ha') -> tuple[np.ndarray, str, np.ndarray]:
        hdr = self.hdu['EMLINE_GVEL'].header
        channel_index = None
        naxis3 = int(hdr.get('NAXIS3', 0))
        for i in range(naxis3):
            line_name = hdr.get(f'C{i+1}', '')
            if isinstance(line_name, str) and channel_name.strip().upper() in line_name.upper():
                channel_index = i
                print(f"Found Hα channel at index {channel_index}: {line_name}")
                break

        if channel_index is None:
            raise ValueError(f"Channel {channel_name} not found in MAPS file.")

        # Extract velocity data
        gas_vel_data = self.hdu['EMLINE_GVEL'].data
        gas_vel_channel = gas_vel_data[channel_index, :, :]

        # Extract mask data
        gas_mask_data = self.hdu['EMLINE_GVEL_MASK'].data
        gas_mask_channel = gas_mask_data[channel_index, :, :]

        # Get velocity unit
        velocity_unit = self.hdu['EMLINE_GVEL'].header['BUNIT']

        # Apply mask: True for good data (mask == 0)
        good_data_mask = (gas_mask_channel == 0)

        # Mask bad pixels with NaN
        masked_velocity_map = gas_vel_channel.copy()
        masked_velocity_map[~good_data_mask] = np.nan

        # Extract IVAR data
        gas_ivar_data = self.hdu['EMLINE_GVEL_IVAR'].data
        gas_ivar_channel = gas_ivar_data[channel_index, :, :]

        # IVAR is inverse variance: positive values indicate good measurements
        good_ivar_mask = (gas_ivar_channel > 0)

        masked_ivar_map = gas_ivar_channel.copy()
        masked_ivar_map[~good_ivar_mask] = np.nan

        return masked_velocity_map, velocity_unit, masked_ivar_map


    # STELLAR_VEL
    # Line-of-sight stellar velocity in km/s, relative to the input guess redshift
    def get_stellar_vel_map(self) -> tuple[np.ndarray, str, np.ndarray]:
        """Return the stellar velocity map from the MAPS file."""
        # Extract velocity data
        stellar_vel_data = self.hdu['STELLAR_VEL'].data
        stellar_vel_map = stellar_vel_data[:, :]

        # Extract mask data
        stellar_mask_data = self.hdu['STELLAR_VEL_MASK'].data
        stellar_mask_map = stellar_mask_data[:, :]

        # Get velocity unit
        velocity_unit = self.hdu['STELLAR_VEL'].header['BUNIT']

        # Apply mask: True for good data (mask == 0)
        good_data_mask = (stellar_mask_map == 0)

        # Mask bad pixels with NaN
        masked_velocity_map = stellar_vel_map.copy()
        masked_velocity_map[~good_data_mask] = np.nan

        # Extract IVAR data
        stellar_ivar_data = self.hdu['STELLAR_VEL_IVAR'].data
        stellar_ivar_map = stellar_ivar_data[:, :]

        # IVAR is inverse variance: positive values indicate good measurements
        good_ivar_mask = (stellar_ivar_map > 0)

        masked_ivar_map = stellar_ivar_map.copy()
        masked_ivar_map[~good_ivar_mask] = np.nan

        return masked_velocity_map, velocity_unit, masked_ivar_map

    # get Spaxel Size
    def get_spaxel_size(self) -> tuple[float, float]:
        """Return the spaxel size in arcseconds from the MAPS file header."""
        hdr = self.hdu['SPX_SKYCOO'].header
        x = abs(float(hdr.get('CDELT1')))
        y = abs(float(hdr.get('CDELT2')))
        return x, y

    #BIN_SNR
    def get_snr_map(self) -> np.ndarray:
        """Return the SNR map from the MAPS file."""
        snr_data = self.hdu['BIN_SNR'].data
        return snr_data
    
    #  ECOOPA: Position angle for ellip. coo
    #  ECOOELL: Ellipticity (1-b/a) for ellip. coo
    def get_pa_inc(self) -> tuple[float | None, float | None]:
        """Return (position angle in degrees, inclination in degrees) from MAPS header or (None, None)."""
        hdr = self.hdu['PRIMARY'].header
        pa_val = hdr.get('ECOOPA', None)
        ellip_val = hdr.get('ECOOELL', None)
        return pa_val, ellip_val
    

    # BIN_LWELLCOO
    # Light-weighted elliptical polar coordinates of each bin from the galaxy center based on the on-sky coordinates in BIN_LWSKYCOO and the ECOOPA and ECOOELL parameters (typically taken from the NASA-Sloan atlas) in the primary header. 
    # SPX_ELLCOO
    # Elliptical polar coordinates of each spaxel from the galaxy center based on the on-sky coordinates in SPX_SKYCOO and the ECOOPA and ECOOELL parameters (typically taken from the NASA-Sloan atlas) in the primary header. 
    def get_r_map(self) -> np.ndarray:
        """Return the radial map from the MAPS file."""
        r_data = self.hdu['BIN_LWELLCOO'].data
        r_map = r_data[0, :, :]
        azimuth = r_data[3, :, :]

        return r_map, azimuth

    def dump_info(self):
        """Print basic information about the MAPS file."""
        print(f"MAPS File: {self.maps_file_path}")
        print("HDU List:")
        self.hdu.info()
