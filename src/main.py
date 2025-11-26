from pathlib import Path
import numpy as np

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
fits_util = FitsUtil(root_dir / "data")


def main():
    PLATE_IFU = "8723-12703"

    print("#######################################################")
    print("# 1. load necessary files")
    print("#######################################################")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

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
    r_obs_fitted, V_obs_fitted = vel_rot.fit_vel_rot(r_obs_map, V_obs_map, ivar_map, phi_map, radius_fit=radius_fit)


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

    r_dm_fit,  V_total_fit, V_dm_fit, V_stellar_fit = dm_nfw.fit_dm_nfw(r_obs_fitted, V_obs_fitted)


    print("#######################################################")
    print("# Results")
    print("#######################################################")
    print(f"V_obs_map shape: {V_disp_map.shape}, range: [{np.nanmin(V_disp_map):,.1f}, {np.nanmax(V_disp_map):,.1f}] km/s")
    print(f"V_obs_fitted shape: {V_obs_fitted.shape}, range: [{np.nanmin(V_obs_fitted):,.1f}, {np.nanmax(V_obs_fitted):,.1f}] km/s")
    print(f"V_total_fit shape: {V_total_fit.shape}, range: [{np.nanmin(V_total_fit):,.1f}, {np.nanmax(V_total_fit):,.1f}] km/s")
    print(f"V_dm_fit shape: {V_dm_fit.shape}, range: [{np.nanmin(V_dm_fit):,.1f}, {np.nanmax(V_dm_fit):,.1f}] km/s")
    print(f"V_stellar_fit shape: {V_stellar_fit.shape}, range: [{np.nanmin(V_stellar_fit):,.1f}, {np.nanmax(V_stellar_fit):,.1f}] km/s")

    ########################################################
    ## plot velocity map
    ########################################################

    # plot galaxy image
    plot_util.plot_galaxy_image(PLATE_IFU)

    # plot RC curves
    plot_util.plot_rv_curve(r_rot_map=r_disp_map, v_rot_map=V_disp_map, title="Observed Deproject",
                            r_rot2_map=r_obs_fitted, v_rot2_map=V_obs_fitted, title2="Observed Fit")

    plot_util.plot_rv_curve(r_rot_map=r_disp_map, v_rot_map=V_disp_map, title="Observed Deproject",
                            r_rot2_map=r_dm_fit, v_rot2_map=V_total_fit, title2="Fitted Total")

    plot_util.plot_rv_curve(r_rot_map=r_obs_fitted, v_rot_map=V_obs_fitted, title="Observed Fit",
                            r_rot2_map=r_dm_fit, v_rot2_map=V_total_fit, title2="Fitted Total")

    plot_util.plot_rv_curve(r_rot_map=r_dm_fit, v_rot_map=V_total_fit, title="Fit Total", 
                        r_rot2_map=r_dm_fit, v_rot2_map=V_dm_fit, title2="Fit DM")

    plot_util.plot_rv_curve(r_rot_map=r_dm_fit, v_rot_map=V_total_fit, title="Fit Total", 
                            r_rot2_map=r_dm_fit, v_rot2_map=V_stellar_fit, title2="Fit Star")

    return

if __name__ == "__main__":
    main()