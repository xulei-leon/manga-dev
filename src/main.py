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
    print("# 3. calculate rot rotation velocity V(r)")
    print("#######################################################")
    vel_rot = VelRot(drpall_util, firefly_util, maps_util, plot_util=None)
    r_obs_map, V_obs_map, _ = vel_rot.get_vel_obs(PLATE_IFU)
    r_rot_fitted, V_rot_fitted, V_obs_fitted = vel_rot.fit_vel_rot(PLATE_IFU)
    

    print("#######################################################")
    print("# 3. calculate stellar rotation velocity V(r)")
    print("#######################################################")
    stellar = Stellar(drpall_util, firefly_util, maps_util)
    r_stellar, V_stellar = stellar.get_vel_stellar(PLATE_IFU)
    r_stellar_fitted, V_stellar_fitted = stellar.fit_vel_stellar(PLATE_IFU, r_rot_fitted)

    ########################################################
    ## plot velocity map
    ########################################################

    # plot galaxy image
    plot_util.plot_galaxy_image(PLATE_IFU)

    # plot rotational radius-velocity curve
    plot_util.plot_rv_curve(r_rot_map=r_rot_fitted, v_rot_map=V_rot_fitted, title="Rotational Fitted", r_rot2_map=r_rot_fitted, v_rot2_map=V_obs_fitted, title2="Observed Fitted")
    plot_util.plot_rv_curve(r_rot_map=r_rot_fitted, v_rot_map=np.abs(V_rot_fitted), title="Rotational Fitted abs", r_rot2_map=r_stellar_fitted, v_rot2_map=V_stellar_fitted, title2="Stellar")

    return

if __name__ == "__main__":
    main()