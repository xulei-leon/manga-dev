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
    r_gas_obs_map, V_gas_obs_map, phi_gas_map = vel_rot.get_gas_vel_obs(PLATE_IFU)
    r_gas_rot_fitted, V_gas_rot_fitted = vel_rot.fit_vel_rot(r_gas_obs_map, V_gas_obs_map, phi_gas_map)

    r_stellar_obs_map, V_stellar_obs_map, phi_stellar_map = vel_rot.get_stellar_vel_obs(PLATE_IFU)
    r_stellar_rot_fitted, V_stellar_rot_fitted = vel_rot.fit_vel_rot(r_stellar_obs_map, V_stellar_obs_map, phi_stellar_map)


    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    stellar = Stellar(drpall_util, firefly_util, maps_util)
    r_stellar, V_stellar = stellar.get_stellar_vel(PLATE_IFU, radius_fitted=r_gas_rot_fitted)
    r_stellar, V_stellar_sq = stellar.get_stellar_vel_sq(PLATE_IFU, radius_fitted=r_gas_rot_fitted)

    print("#######################################################")
    print("# 4. calculate dark-matter rotation velocity V(r)")
    print("#######################################################")
    r_dm, V_dm, V_total = vel_rot.fit_vel_dm(PLATE_IFU, r_gas_rot_fitted, V_gas_rot_fitted, v_star=V_stellar)



    print("#######################################################")
    print("# Results")
    print("#######################################################")
    print(f"V_gas_obs_map shape: {V_gas_obs_map.shape}, range: [{np.nanmin(V_gas_obs_map):,.1f}, {np.nanmax(V_gas_obs_map):,.1f}] km/s")
    print(f"V_gas_rot_fitted shape: {V_gas_rot_fitted.shape}, range: [{np.nanmin(V_gas_rot_fitted):,.1f}, {np.nanmax(V_gas_rot_fitted):,.1f}] km/s")
    print(f"V_stellar shape: {V_stellar.shape}, range: [{np.nanmin(V_stellar):,.1f}, {np.nanmax(V_stellar):,.1f}] km/s")
    print(f"V_dm shape: {V_dm.shape}, range: [{np.nanmin(V_dm):,.1f}, {np.nanmax(V_dm):,.1f}] km/s")
    print(f"V_total shape: {V_total.shape}, range: [{np.nanmin(V_total):,.1f}, {np.nanmax(V_total):,.1f}] km/s")

    ########################################################
    ## plot velocity map
    ########################################################

    # plot galaxy image
    plot_util.plot_galaxy_image(PLATE_IFU)

    # 
    plot_util.plot_rv_curve(r_rot_map=r_stellar, v_rot_map=V_stellar, title="Star Circular")

    plot_util.plot_rv_curve(r_rot_map=r_gas_rot_fitted, v_rot_map=np.abs(V_gas_rot_fitted), title="Fitted Rotational", 
                            r_rot2_map=r_stellar, v_rot2_map=V_stellar, title2="Star Rotational")

    plot_util.plot_rv_curve(r_rot_map=r_gas_rot_fitted, v_rot_map=np.abs(V_gas_rot_fitted), title="Fitted Rotational", 
                        r_rot2_map=r_dm, v_rot2_map=V_dm, title2="Dark Matter Rotational")
    return

if __name__ == "__main__":
    main()