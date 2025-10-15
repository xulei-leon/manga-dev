import matplotlib.pyplot as plt
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
import numpy as np

# get z_sys from drpall
def get_z_sys(drpall_path, plateifu):
    drpall_hdu = fits.open(drpall_path)
    try:
        drpall_data = drpall_hdu[1].data
        match = drpall_data['plateifu'] == plateifu
        if np.any(match):
            z_sys = drpall_data['nsa_z'][match][0]
            return z_sys
        else:
            print(f"No match found for {plateifu} in drpall")
            return None
    finally:
        drpall_hdu.close()


# get velocity map from MAPS file
def get_vel_from_map(maps_file_path, channel_name='Ha'):
    try:
        with fits.open(maps_file_path) as hdul:
            # Find the channel index based on the channel name
            channel_index = None
            for key, value in hdul['EMLINE_GVEL'].header.items():
                if key.startswith('C') and isinstance(value, str):
                    if channel_name.upper() in value.upper():
                        channel_index = int(key[1:]) - 1
                        print(f"Found channel {channel_name} at index {channel_index}")
                        break

            if channel_index is None:
                raise ValueError(f"Channel {channel_name} not found in MAPS file.")
            
            # Extract velocity data
            gas_vel_data = hdul['EMLINE_GVEL'].data
            gas_vel_channel = gas_vel_data[channel_index, :, :]
            
            # Extract mask data
            gas_mask_data = hdul['EMLINE_GVEL_MASK'].data
            gas_mask_channel = gas_mask_data[channel_index, :, :]
            
            # Extract IVAR data (not used in this function, but extracted for completeness)
            gas_ivar_data = hdul['EMLINE_GVEL_IVAR'].data
            gas_ivar_channel = gas_ivar_data[channel_index, :, :]
            
            # Get velocity unit
            velocity_unit = hdul['EMLINE_GVEL'].header['BUNIT']
            
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
