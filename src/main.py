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
    PLATE_IFU = "8723-12705"

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

    r_sorted = vel_rot.get_radius_sort()

    # r_gas_obs_map, V_gas_obs_map, phi_gas_map = vel_rot.get_gas_vel_obs()
    # r_gas_rot_fitted, V_gas_rot_fitted = vel_rot.fit_rot_vel(r_gas_obs_map, V_gas_obs_map, phi_gas_map, radius_fit=r_sorted)

    r_stellar_obs_map, V_stellar_obs_map, phi_stellar_map = vel_rot.get_stellar_vel_obs()
    r_stellar_rot_fitted, V_stellar_rot_fitted = vel_rot.fit_rot_vel(r_stellar_obs_map, V_stellar_obs_map, phi_stellar_map, radius_fit=r_sorted)

    r_obs_map = r_stellar_obs_map
    V_obs_map = V_stellar_obs_map
    r_obs_fitted = r_stellar_rot_fitted
    V_obs_fitted = V_stellar_rot_fitted

    inc_rad = vel_rot.get_inc_rad()

    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    stellar = Stellar(drpall_util, firefly_util, maps_util)
    stellar.set_PLATE_IFU(PLATE_IFU)
    r_stellar, V_stellar = stellar.get_stellar_vel(radius_fitted=r_obs_fitted)
    r_stellar, V_stellar_sq, _, r_d = stellar.get_stellar_vel_sq(radius_fitted=r_obs_fitted)
    _, stellar_density_map = stellar.calc_stellar_density(radius_fitted=r_obs_fitted)

    r_drift, V_drift_sq = stellar.get_stellar_vel_drift_sq(r_obs_fitted, inc_rad)
    V_drift = np.sqrt(V_drift_sq)
    print(f"V_drift values at specific radii:")
    for r_target in [1, 2, 3, 4, 5, 6, 7, 8]:
        idx = np.argmin(np.abs(r_drift - r_target))
        actual_r = r_drift[idx]
        actual_v = V_drift[idx]
        print(f"  r ≈ {actual_r:.2f}: V_drift = {actual_v:.2f} km/s")


    print("#######################################################")
    print("# 4. calculate dark-matter rotation velocity V(r)")
    print("#######################################################")
    dm_nfw = DmNfw(drpall_util)
    dm_nfw.set_PLATE_IFU(PLATE_IFU)
    M200_fit, r_dm_fit, V_total_fit, V_dm_fit = dm_nfw.fit_dm_nfw(r_obs_fitted, V_obs_fitted, V_stellar_sq, V_drift_sq, r_d)



    print("#######################################################")
    print("# Results")
    print("#######################################################")
    print(f"V_obs_map shape: {V_obs_map.shape}, range: [{np.nanmin(V_obs_map):,.1f}, {np.nanmax(V_obs_map):,.1f}] km/s")
    print(f"V_obs_fitted shape: {V_obs_fitted.shape}, range: [{np.nanmin(V_obs_fitted):,.1f}, {np.nanmax(V_obs_fitted):,.1f}] km/s")
    print(f"V_stellar shape: {V_stellar.shape}, range: [{np.nanmin(V_stellar):,.1f}, {np.nanmax(V_stellar):,.1f}] km/s")
    print(f"V_drift shape: {V_drift.shape}, range: [{np.nanmin(V_drift):,.1f}, {np.nanmax(V_drift):,.1f}] km/s")
    print(f"V_total_fit shape: {V_total_fit.shape}, range: [{np.nanmin(V_total_fit):,.1f}, {np.nanmax(V_total_fit):,.1f}] km/s")
    print(f"V_dm_fit shape: {V_dm_fit.shape}, range: [{np.nanmin(V_dm_fit):,.1f}, {np.nanmax(V_dm_fit):,.1f}] km/s")

    ########################################################
    ## plot velocity map
    ########################################################

    # plot galaxy image
    plot_util.plot_galaxy_image(PLATE_IFU)

    # 
    plot_util.plot_rv_curve(r_rot_map=r_stellar, v_rot_map=V_stellar, title="Star Circular")

    plot_util.plot_rv_curve(r_rot_map=r_obs_fitted, v_rot_map=V_obs_fitted, title="Fitted Rotational", 
                            r_rot2_map=r_obs_fitted, v_rot2_map=V_drift, title2="Drift Rotational")

    plot_util.plot_rv_curve(r_rot_map=r_dm_fit, v_rot_map=V_total_fit, title="Fitted Total", 
                        r_rot2_map=r_dm_fit, v_rot2_map=V_dm_fit, title2="Dark Matter")
    return

if __name__ == "__main__":
    main()