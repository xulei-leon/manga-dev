from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil
from vel_stellar import Stellar
from vel_rot import VelRot
from dm import DmNfw

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"
fits_util = FitsUtil(data_dir)

VEL_FIT_PARAM_FILENAME = "vel_rot_param.csv"
VEL_FIT_PARAM_ALL_FILENAME = "vel_rot_param_all.csv"
DM_NFW_PARAM_FILENAME = "dm_nfw_param.csv"


# Store vel rot fit parameters as CSV file
def store_params_file(PLATE_IFU: str, fit_parameters: dict, filename:str):
    output_file = data_dir / filename

    if output_file.exists():
        try:
            all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
        except pd.errors.EmptyDataError:
            all_fit_parameters = {}
    else:
        all_fit_parameters = {}

    all_fit_parameters[PLATE_IFU] = fit_parameters

    df = pd.DataFrame.from_dict(all_fit_parameters, orient='index')
    df.rename_axis('PLATE_IFU', inplace=True)
    df.to_csv(output_file)
    return

def get_params_file(PLATE_IFU: str, filename:str):
    output_file = data_dir / filename

    if not output_file.exists():
        return None

    try:
        all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
    except pd.errors.EmptyDataError:
        return None

    if PLATE_IFU in all_fit_parameters:
        return all_fit_parameters[PLATE_IFU]
    else:
        return None


def process_plate_ifu(PLATE_IFU, plot_enable:bool=False, process_nfw: bool=True):
    nfw_param = get_params_file(PLATE_IFU, DM_NFW_PARAM_FILENAME)
    if process_nfw and nfw_param is not None:
        print(f"DM NFW parameters already exist for {PLATE_IFU}. Skipping processing.")
        return

    print("#######################################################")
    print("# 1. load necessary files")
    print("#######################################################")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU, checksum=False, download=False)
    if maps_file is None:
        print(f"MAPS file for {PLATE_IFU} not found locally. Skipping processing.")
        return

    print(f"DRPALL file: {drpall_file}")
    print(f"FIREFLY file: {firefly_file}")
    print(f"MAPS file: {maps_file}")

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)
    plot_util = PlotUtil(fits_util)

    print("")
    print("#######################################################")
    print("# 2. calculate rot rotation velocity V(r)")
    print("#######################################################")
    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=None)
    vel_rot.set_PLATE_IFU(PLATE_IFU)

    r_obs_map, V_obs_map, ivar_map, phi_map = vel_rot.get_vel_obs()
    r_disp_map, V_disp_map, _ = vel_rot.get_vel_obs_disp()
    radius_fit = vel_rot.get_radius_fit(np.nanmax(r_disp_map), count=1000)

    if process_nfw:
        fit_check = True
        vel_rot_filename = VEL_FIT_PARAM_FILENAME
    else:
        fit_check = False
        vel_rot_filename = VEL_FIT_PARAM_ALL_FILENAME

    success, fit_result, fit_params = vel_rot.fit_vel_rot(r_obs_map, V_obs_map, ivar_map, phi_map, radius_fit=radius_fit, fit_check=fit_check)
    store_params_file(PLATE_IFU, fit_params, filename=vel_rot_filename)

    if not success:
        print(f"Fitting rotational velocity failed for {PLATE_IFU}")
        return

    if not process_nfw:
        return

    r_rot_fit = fit_result['radius']
    V_rot_fit = fit_result['vel_rot']
    V_rot_err = fit_result['vel_err']

    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    stellar = Stellar(drpall_util, firefly_util, maps_util)
    stellar.set_PLATE_IFU(PLATE_IFU)

    print("#######################################################")
    print("# 4. calculate dark-matter rotation velocity V(r)")
    print("#######################################################")
    dm_nfw = DmNfw(drpall_util)
    dm_nfw.set_PLATE_IFU(PLATE_IFU)
    dm_nfw.set_stellar_util(stellar)
    dm_nfw.set_plot_enable(plot_enable)
    dm_nfw.set_fit_debug(False)

    # r_dm_fit, V_total_fit, V_dm_fit, V_stellar_fit = dm_nfw.fit_dm_nfw(r_rot_fit, V_rot_fit, V_rot_err)
    success, inf_result, inf_params = dm_nfw.inf_dm_nfw(r_rot_fit, V_rot_fit, V_rot_err)
    store_params_file(PLATE_IFU, inf_params, filename=DM_NFW_PARAM_FILENAME)
    if not success:
        print(f"Inferring dark matter NFW failed for {PLATE_IFU}")
        return

    r_dm_fit = inf_result['radius']
    V_total_fit = inf_result['vel_rot']
    V_dm_fit = inf_result['vel_dm']
    V_stellar_fit = inf_result['vel_star']


    print("#######################################################")
    print("# Results")
    print("#######################################################")
    print(f"V_obs_map shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):,.1f}, {np.nanmax(V_disp_map):,.1f}] km/s")
    print(f"V_obs_fitted shape: {V_rot_fit.shape}, range: [{np.nanmin(V_rot_fit):,.1f}, {np.nanmax(V_rot_fit):,.1f}] km/s")
    print(f"V_total_fit shape: {V_total_fit.shape}, range: [{np.nanmin(V_total_fit):,.1f}, {np.nanmax(V_total_fit):,.1f}] km/s")
    print(f"V_dm_fit shape: {V_dm_fit.shape}, range: [{np.nanmin(V_dm_fit):,.1f}, {np.nanmax(V_dm_fit):,.1f}] km/s")
    print(f"V_stellar_fit shape: {V_stellar_fit.shape}, range: [{np.nanmin(V_stellar_fit):,.1f}, {np.nanmax(V_stellar_fit):,.1f}] km/s")

    ########################################################
    ## plot velocity map
    ########################################################
    if  plot_enable:
        # plot galaxy image
        plot_util.plot_galaxy_image(PLATE_IFU)

        # plot RC curves
        plot_util.plot_rv_curve(r_rot_map=r_disp_map, v_rot_map=V_disp_map, title="Observed Deproject",
                                r_rot2_map=r_rot_fit, v_rot2_map=V_rot_fit, title2="Observed Fit")

        plot_util.plot_rv_curve(r_rot_map=r_disp_map, v_rot_map=V_disp_map, title="Observed Deproject",
                                r_rot2_map=r_dm_fit, v_rot2_map=V_total_fit, title2="Fitted Total")

        plot_util.plot_rv_curve(r_rot_map=r_rot_fit, v_rot_map=V_rot_fit, title="Observed Fit",
                                r_rot2_map=r_dm_fit, v_rot2_map=V_total_fit, title2="Fitted Total")

        plot_util.plot_rv_curve(r_rot_map=r_dm_fit, v_rot_map=V_total_fit, title="Fit Total",
                            r_rot2_map=r_dm_fit, v_rot2_map=V_dm_fit, title2="Fit DM")

        plot_util.plot_rv_curve(r_rot_map=r_dm_fit, v_rot_map=V_total_fit, title="Fit Total",
                                r_rot2_map=r_dm_fit, v_rot2_map=V_stellar_fit, title2="Fit Star")

    return

TEST_PLATE_IFUS = [
    "7957-3701",
    "8078-1902",
    "10218-6102",
    "8329-6103",
    "8723-12703",
    "8723-12705",
    "7495-12704",
    "10220-12705"
]

PLATES_FILENAME = "plateifus.txt"
def get_plate_ifu_list():
    plate_ifu_file = data_dir / PLATES_FILENAME

    with open(plate_ifu_file, 'r') as f:
        plate_ifu_list = [line.strip() for line in f if line.strip()]

    # sort the list
    plate_ifu_list.sort()
    return plate_ifu_list


def main():
    plate_ifu_list = get_plate_ifu_list()
    if not plate_ifu_list:
        plate_ifu_list = TEST_PLATE_IFUS

    def _process(plate_ifu):
        print(f"\n\n########## Processing PLATE_IFU: {plate_ifu} ##########")
        try:
            process_plate_ifu(plate_ifu, plot_enable=False, process_nfw=RUN_NFW)
        except Exception as e:
            print(f"Error processing {plate_ifu}: {e}")
        finally:
            # Clear pymc internal cache to free up memory
            gc.collect()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process, plate_ifu) for plate_ifu in plate_ifu_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing galaxies", unit="galaxy"):
            pass

RUN_NFW = True

if __name__ == "__main__":
    main()
