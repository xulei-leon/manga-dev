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
    print(f"V_stellar_rot_fitted shape: {V_stellar_rot_fitted.shape}, range: [{np.nanmin(V_stellar_rot_fitted):,.1f}, {np.nanmax(V_stellar_rot_fitted):,.1f}] km/s")


    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    stellar = Stellar(drpall_util, firefly_util, maps_util)
    r_stellar, V_stellar = stellar.get_stellar_vel(PLATE_IFU)

    r_stellar, V_stellar_sq, stellar_sigma_0, stellar_r_d = stellar.get_stellar_vel_sq(PLATE_IFU)
    stellar_density = stellar.get_stellar_density(r_stellar_obs_map, stellar_sigma_0, stellar_r_d)
    print(f"stellar density shape: {stellar_density.shape}, range: [{np.nanmin(stellar_density):,.1f}, {np.nanmax(stellar_density):,.1f}] Msun/kpc^2")

    print("#######################################################")
    print("# 4. calculate stellar circular velocity V(r)")
    print("#######################################################")
    r_stellar_circular, V_stellar_circular = vel_rot.calc_vel_circ(r_stellar_rot_fitted, V_stellar_rot_fitted, stellar_density)
    print(f"V_stellar_circular shape: {V_stellar_circular.shape}, range: [{np.nanmin(V_stellar_circular):,.1f}, {np.nanmax(V_stellar_circular):,.1f}] km/s")


    ########################################################
    ## plot velocity map
    ########################################################

    # plot galaxy image
    plot_util.plot_galaxy_image(PLATE_IFU)

    # 
    plot_util.plot_rv_curve(r_rot_map=r_stellar, v_rot_map=V_stellar, title="Star Circular")

    # compare rotational fitted velocity vs observed fitted velocity
    plot_util.plot_rv_curve(r_rot_map=r_gas_rot_fitted, v_rot_map=V_gas_rot_fitted, title="Rotational gas", 
                            r_rot2_map=r_stellar_rot_fitted, v_rot2_map=V_stellar_rot_fitted, title2="Rotational stellar")
    
    # plot_util.plot_rv_curve(r_rot_map=r_stellar_rot_fitted, v_rot_map=V_stellar_rot_fitted, title="Rotational stellar", 
    #                         r_rot2_map=r_stellar_rot_fitted, v_rot2_map=V_stellar_obs_fitted, title2="Observed stellar")
    
    plot_util.plot_rv_curve(r_rot_map=r_stellar_rot_fitted, v_rot_map=V_stellar_rot_fitted, title="Rotational stellar", 
                            r_rot2_map=r_stellar_circular, v_rot2_map=V_stellar_circular, title2="Circular stellar")
 
    # compare rotational fitted velocity (abs) vs stellar fitted velocity
    plot_util.plot_rv_curve(r_rot_map=r_stellar_circular, v_rot_map=np.abs(V_stellar_circular), title="Total stellar", 
                            r_rot2_map=r_stellar, v_rot2_map=V_stellar, title2="Star Circular")
    return

if __name__ == "__main__":
    main()