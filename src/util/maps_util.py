import matplotlib.pyplot as plt
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
import numpy as np


# get velocity map from MAPS file
##############################################################################
# EMLINE_GVEL (Emission Line Gaussian Velocity)
# Line-of-Sight Velocity of Ionized Gas
# km/s
##############################################################################
# EMLINE_GVEL_IVAR (Emission Line Gaussian Velocity Inverse Variance)
##############################################################################
# Emission Line Gaussian Velocity (EMLINE_GVEL) extension
# channel_name: 'Ha' for Hα, 'OIII' for [O III], etc.
def get_gvel_map(maps_file_path, channel_name='Ha') -> tuple[np.ndarray, str, np.ndarray]:
    try:
        with fits.open(maps_file_path) as hdu:
            hdr = hdu['EMLINE_GVEL'].header
            channel_index = None
            naxis3 = int(hdr.get('NAXIS3', 0))
            for i in range(naxis3):
                line_name = hdr.get(f'C{i+1}', '')
                if isinstance(line_name, str) and  channel_name.strip().upper() in line_name.upper():
                    channel_index = i
                    print(f"Found Hα channel at index {channel_index}: {line_name}")
                    break

            if channel_index is None:
                raise ValueError(f"Channel {channel_name} not found in MAPS file.")
            
            # Extract velocity data
            gas_vel_data = hdu['EMLINE_GVEL'].data
            gas_vel_channel = gas_vel_data[channel_index, :, :]
            
            # Extract mask data
            gas_mask_data = hdu['EMLINE_GVEL_MASK'].data
            gas_mask_channel = gas_mask_data[channel_index, :, :]
            
            # Get velocity unit
            velocity_unit = hdu['EMLINE_GVEL'].header['BUNIT']
            
            # print(f"Successfully extracted {channel_name} velocity map, shape: {gas_vel_channel.shape}")
            # print(f"Velocity unit: {velocity_unit}")
            
            # Apply mask: True for good data (mask == 0)
            good_data_mask = (gas_mask_channel == 0)
            
            # Mask bad pixels with NaN
            masked_velocity_map = gas_vel_channel.copy()
            masked_velocity_map[~good_data_mask] = np.nan

            # Extract IVAR data
            gas_ivar_data = hdu['EMLINE_GVEL_IVAR'].data
            gas_ivar_channel = gas_ivar_data[channel_index, :, :]

            # IVAR is inverse variance: positive values indicate good measurements
            good_ivar_mask = (gas_ivar_channel > 0)

            masked_ivar_map = gas_ivar_channel.copy()
            masked_ivar_map[~good_ivar_mask] = np.nan
            
            return masked_velocity_map, velocity_unit, masked_ivar_map
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {maps_file_path}. Please check the path.")
    except KeyError:
        raise KeyError("MAPS file structure incorrect or missing required extensions. Ensure it's a DAP MAPS file.")
    except ValueError:
        raise  # Re-raise the ValueError for channel not found
    except IndexError:
        raise IndexError(f"Channel index for {channel_name} out of range. Check the channel name.")
    except Exception as e:
        raise Exception(f"Unknown error: {e}")


# get Spaxel Size
def get_spaxel_size(maps_file_path) -> float:
    """Return the spaxel size in arcseconds from the MAPS file header."""
    try:
        with fits.open(maps_file_path) as hdu:
            hdr = hdu['SPX_SKYCOO'].header
            cd1_1 = hdr.get('CDELT1', None)
            if cd1_1 is None:
                raise KeyError("CD1_1 keyword not found in SPX_SKYCOO header.")
            spaxel_size = abs(cd1_1) * 3600.0  # degrees to arcseconds
            return spaxel_size
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {maps_file_path}. Please check the path.")
    except KeyError as e:
        raise KeyError(f"MAPS file structure incorrect or missing required keywords: {e}")
    except Exception as e:
        raise Exception(f"Unknown error: {e}")