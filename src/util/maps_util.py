import matplotlib.pyplot as plt
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
import numpy as np


# get velocity map from MAPS file
# Emission Line Gaussian Velocity (EMLINE_GVEL) extension
# channel_name: 'Ha' for Hα, 'OIII' for [O III], etc.
def get_spin_vel_map(maps_file_path, channel_name='Ha') -> tuple[np.ndarray, str]:
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
            
            # Extract IVAR data (not used in this function, but extracted for completeness)
            gas_ivar_data = hdu['EMLINE_GVEL_IVAR'].data
            gas_ivar_channel = gas_ivar_data[channel_index, :, :]
            
            # Get velocity unit
            velocity_unit = hdu['EMLINE_GVEL'].header['BUNIT']
            
            # print(f"Successfully extracted {channel_name} velocity map, shape: {gas_vel_channel.shape}")
            # print(f"Velocity unit: {velocity_unit}")
            
            # Apply mask: True for good data (mask == 0)
            good_data_mask = (gas_mask_channel == 0)
            
            # Mask bad pixels with NaN
            masked_velocity_map = gas_vel_channel.copy()
            masked_velocity_map[~good_data_mask] = np.nan
            
            return masked_velocity_map, velocity_unit
            
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
