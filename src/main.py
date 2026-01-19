from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from threading import Lock
import gc

# my imports
from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from util.plot_util import PlotUtil
from rc import RotCurve
from dm import DmNfw

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"
result_dir = data_dir / "results"
data_dir.mkdir(parents=True, exist_ok=True)
result_dir.mkdir(parents=True, exist_ok=True)


fits_util = FitsUtil(data_dir)

VEL_FIT_PARAM_FILENAME = "vel_rot_param.csv"
VEL_FIT_PARAM_ALL_FILENAME = "vel_rot_param_all.csv"
DM_NFW_PARAM_FILENAME = "dm_nfw_param.csv"

# Thresholds for filtering fitting results
NRMSE_THRESHOLD1 = 0.07  # threshold for first fitting
NRMSE_THRESHOLD2 = 0.05  # tighter threshold for second fitting
CHI_SQ_V_THRESHOLD1 = 5.0  # looser threshold for first fitting
CHI_SQ_V_THRESHOLD2 = 3.0  # threshold for reduced chi-squared to filter weak fitting
VEL_OBS_COUNT_THRESHOLD1 = 150  # minimum number of valid velocity data points
VEL_OBS_COUNT_THRESHOLD2 = 100  # minimum number of valid velocity data points
RMAX_RT_FACTOR = 2.5  # factor to determine maximum radius for fitting

csv_lock = Lock()

# Store vel rot fit parameters as CSV file
def store_params_file(PLATE_IFU: str, fit_parameters: dict, filename:str):
    output_file = result_dir / filename

    with csv_lock:
        if output_file.exists():
            try:
                all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
            except pd.errors.EmptyDataError:
                all_fit_parameters = {}
        else:
            all_fit_parameters = {}

        # clean previous entry
        if PLATE_IFU in all_fit_parameters:
            del all_fit_parameters[PLATE_IFU]

        all_fit_parameters[PLATE_IFU] = fit_parameters

        df = pd.DataFrame.from_dict(all_fit_parameters, orient='index')
        df.rename_axis('PLATE_IFU', inplace=True)
        df.to_csv(output_file)
    return

def get_params_file(PLATE_IFU: str, filename:str):
    output_file = result_dir / filename

    if not output_file.exists():
        return None

    with csv_lock:
        try:
            all_fit_parameters = pd.read_csv(output_file, index_col=0).to_dict(orient='index')
        except pd.errors.EmptyDataError:
            return None

    if PLATE_IFU in all_fit_parameters:
        return all_fit_parameters[PLATE_IFU]
    else:
        return None


def process_plate_ifu(PLATE_IFU, process_nfw: bool=True, debug: bool=False):
    nfw_param = get_params_file(PLATE_IFU, DM_NFW_PARAM_FILENAME)
    if debug is None and process_nfw and nfw_param is not None:
        print(f"DM NFW parameters already exist for {PLATE_IFU}. Skipping processing.")
        return

    vel_rot_param = get_params_file(PLATE_IFU, VEL_FIT_PARAM_FILENAME)
    if not process_nfw and vel_rot_param is not None:
        if vel_rot_param['result'] != 'success':
            print(f"Velocity rotation fit previously failed for {PLATE_IFU}. Skipping processing.")
            return

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

    vel_rot = RotCurve(drpall_util, firefly_util, maps_util, plot_util=None)
    vel_rot.set_PLATE_IFU(PLATE_IFU)

    r_obs_map, V_obs_map, ivar_map, phi_map = vel_rot.get_vel_obs()
    radius_fit = vel_rot.get_radius_fit(np.nanmax(r_obs_map), count=1000)

    if process_nfw:
        vel_rot_filename = VEL_FIT_PARAM_FILENAME
    else:
        vel_rot_filename = VEL_FIT_PARAM_ALL_FILENAME

    #----------------------------------------------------------------------
    # First fitting
    #----------------------------------------------------------------------
    print(f"## First fitting {PLATE_IFU} ##")
    success, fit_result, fit_params = vel_rot.fit_vel_rot(r_obs_map, V_obs_map, ivar_map, phi_map, radius_fit=radius_fit)
    if not success:
        print(f"Fitting rotational velocity failed for {PLATE_IFU}")
        return

    r_obs_map = fit_result['radius_obs']
    V_obs_map = fit_result['vel_obs']
    ivar_obs_map = fit_result['ivar_obs']
    r_rot_fit = fit_result['radius_rot']
    V_rot_fit = fit_result['vel_rot']
    stderr_rot_fit = fit_result['stderr_rot']
    inc_rad_fit = float(fit_params['inc'])
    vel_sys_fit = float(fit_params['Vsys'])
    phi_delta_fit = float(fit_params['phi_delta'])
    Rmax = float(fit_params['Rmax'])
    Rt = float(fit_params['Rt'])

    # Filter fitting parameters
    data_count = np.sum(np.isfinite(V_obs_map))
    NRMSE = float(fit_params['NRMSE'])
    CHI_SQ_V = float(fit_params['CHI_SQ_V'])
    if data_count < VEL_OBS_COUNT_THRESHOLD1 or \
        (NRMSE > NRMSE_THRESHOLD1) or \
        (CHI_SQ_V > CHI_SQ_V_THRESHOLD1) or \
        (Rmax < Rt * RMAX_RT_FACTOR):
        print(f"First fitting results failure for {PLATE_IFU}, data amount: {data_count}, "
              f"NRMSE: {NRMSE:.3f}, CHI_SQ_V: {CHI_SQ_V:.3f}, "
              f"Rmax: {Rmax:.3f}, Rt: {Rt:.3f}, skipping...")
        return

    store_params_file(PLATE_IFU, fit_params, filename=vel_rot_filename)
    if not process_nfw:
        return

    r_disp_map, V_disp_map, _ = vel_rot.get_vel_obs_disp(inc_rad=inc_rad_fit, vel_sys=vel_sys_fit, phi_delta=phi_delta_fit)


    #--------------------------------------------------------
    # DM NFW fitting
    #--------------------------------------------------------
    dm_nfw = DmNfw(drpall_util)
    dm_nfw.set_PLATE_IFU(PLATE_IFU)
    dm_nfw.set_plot_enable(False)
    dm_nfw.set_inf_debug(debug)

    success, inf_result, inf_params = dm_nfw.inf_dm_nfw(radius_obs=r_obs_map,
                                                        vel_obs=V_obs_map,
                                                        ivar_obs=ivar_obs_map,
                                                        vel_sys=vel_sys_fit,
                                                        inc_rad=inc_rad_fit,
                                                        phi_map=phi_map)
    store_params_file(PLATE_IFU, inf_params, filename=DM_NFW_PARAM_FILENAME)
    if not success:
        print(f"Inferring dark matter NFW failed for {PLATE_IFU}")
        return

    r_inf = inf_result['radius']
    V_rot_inf = inf_result['v_rot']
    V_dm_inf = inf_result['v_dm']
    V_star_inf = inf_result['v_star']
    V_drift_inf = inf_result['v_drift']
    sigma_obs_inf = inf_result['sigma_obs']
    res_obs_inf = inf_result['res_obs']


    #--------------------------------------------------------
    # DM NFW fitting
    #--------------------------------------------------------
    print(f"V_obs_map shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):,.1f}, {np.nanmax(V_disp_map):,.1f}] km/s")
    print(f"V_obs_fitted shape: {V_rot_fit.shape}, range: [{np.nanmin(V_rot_fit):,.1f}, {np.nanmax(V_rot_fit):,.1f}] km/s")
    print(f"V_total_fit shape: {V_rot_inf.shape}, range: [{np.nanmin(V_rot_inf):,.1f}, {np.nanmax(V_rot_inf):,.1f}] km/s")
    print(f"V_dm_fit shape: {V_dm_inf.shape}, range: [{np.nanmin(V_dm_inf):,.1f}, {np.nanmax(V_dm_inf):,.1f}] km/s")
    print(f"V_star_fit shape: {V_star_inf.shape}, range: [{np.nanmin(V_star_inf):,.1f}, {np.nanmax(V_star_inf):,.1f}] km/s")
    print(f"V_drift_fit shape: {V_drift_inf.shape}, range: [{np.nanmin(V_drift_inf):,.1f}, {np.nanmax(V_drift_inf):,.1f}] km/s")

    ########################################################
    ## plot velocity map
    ########################################################
    plot_util.plot_rv_curves([
        {'r_map': r_disp_map, 'V_map': V_disp_map, 'title': "Observe", 'color': 'gray', 'linestyle': None, 'size': 5},
        {'r_map': r_rot_fit, 'V_map': V_rot_fit, 'title': "Fit rot", 'color': 'black', 'linestyle': '-'},
        {'r_map': r_inf, 'V_map': V_rot_inf, 'title': "Inf rot", 'color': 'red', 'linestyle': '-'},
        {'r_map': r_inf, 'V_map': V_dm_inf, 'title': "Inf DM", 'color': 'black', 'linestyle': '--'},
        {'r_map': r_inf, 'V_map': V_star_inf, 'title': "Inf Stellar", 'color': 'blue', 'linestyle': '-'},
        {'r_map': r_inf, 'V_map': V_drift_inf, 'title': "Inf Drift", 'color': 'green', 'linestyle': '-'},
    ], plateifu=PLATE_IFU, savedir=result_dir if not debug else None)

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

import pandas as pd
VEL_ROT_PARAM_FILE = "vel_rot_param_all.csv"
def get_plate_list_from_fit():
    param_file = data_dir / VEL_ROT_PARAM_FILE
    df = pd.read_csv(param_file)
    df = df[df['result'] == 'success']
    plate_ifu_list = df['PLATE_IFU'].tolist()
    plate_ifu_list = [str(plate_ifu) for plate_ifu in plate_ifu_list]
    plate_ifu_list.sort()
    return plate_ifu_list

def main(run_nfw: bool = True, ifu: str = None, debug: bool = False):
    plate_ifu_list = []

    if ifu == "all":
        plate_ifu_list = get_plate_ifu_list()
    elif ifu == "test":
        plate_ifu_list = TEST_PLATE_IFUS
    else:
        plate_ifu_list = [ifu]

    if not plate_ifu_list or len(plate_ifu_list) == 0:
        plate_ifu_list = TEST_PLATE_IFUS

    def _process(plate_ifu):
        print(f"\n\n########## Processing PLATE_IFU: {plate_ifu} ##########")
        if debug:
            process_plate_ifu(plate_ifu, process_nfw=run_nfw, debug=debug)
        else:
            try:
                process_plate_ifu(plate_ifu, process_nfw=run_nfw, debug=debug)
            except Exception as e:
                print(f"Error processing {plate_ifu}: {e}")

        # Clear pymc internal cache to free up memory
        gc.collect()

    for plate_ifu in tqdm(plate_ifu_list, total=len(plate_ifu_list), desc="Processing galaxies", unit="galaxy"):
        _process(plate_ifu)
    return

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MaNGA galaxies for velocity rotation and DM NFW fitting.")
    parser.add_argument('--nfw', type=str, default="on", help='Run dark matter NFW fitting.')
    parser.add_argument('--ifu', type=str, default="all", help='Type of data to process (all, fit, test.)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--result', type=str, default="results", help='Result directory name.')

    args = parser.parse_args()

    nfw_enable = args.nfw.lower() in ['on', 'true', 'enable' ,'1']
    result_dir = data_dir / args.result if args.result else root_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    main(run_nfw=nfw_enable, ifu=args.ifu, debug=args.debug)