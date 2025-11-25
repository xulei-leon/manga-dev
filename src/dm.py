from pathlib import Path

from re import M, S
import numpy as np
from scipy.optimize import curve_fit, minimize
from astropy import constants as const
from astropy import units as u

from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from vel_stellar import G, Stellar
from vel_rot import PLATE_IFU, VelRot


class DmNfw:

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util


    ########################################################################################
    # Use the following equation to fit DM profile:
    # ignoring gas contribution for simplification
    ########################################################################################
    # V_obs^2  =  V_star^2 + V_dm^2 - V_drift^2
    ########################################################################################
        

    ########################################################################################
    # NFW Dark Matter Halo Profile:
    ########################################################################################
    # --- Navarro-Frenk-White (NFW) Dark Matter Halo Rotational Velocity Squared ---
    #
    # Formula:
    # V_DM^2(r) = (V_200^2 / x) * [ (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c)) ]
    # V_DM(r): rotational velocity due to the dark matter halo at radius r.
    # V_200(r): circular speed at the virial radius r_200.
    # ln        : The natural logarithm function.
    #
    # --- Key Parameters and Context ---
    # 1. Normalized Radius (x):
    #    x = r / r_200.
    #
    # 2. Virial Radius (r_200):
    #    r_200 is the radius within which the mean density is 200 times the critical density.
    #    It is related to V_200 and the Hubble parameter H(z) by: r_200 = V_200 / (10 * H(z)).
    #
    # 3. Halo Mass (M_200) and V_200 Relation:
    #    M_200 is the halo mass within r_200. V_200 is connected to M_200 via:
    #    V_200^3 = 10 * G * H(z) * M_200, where G is the gravitational constant.
    #
    # 4. Concentration Parameter (c):
    #    c is the concentration parameter of the NFW profile. It relates to the scale radius
    #    r_s through r_s = r_200 / c.
    #
    # 5. c - M_200 Mass-Concentration Relation (Duffy et al. 2008):
    #    c is not independent; it correlates with M_200 (low-mass halos are more concentrated).
    #    The relation used here is:
    #    c = 5.74 * ( M_200 / (2 * 10^12 * h^-1 * M_sun) )^(-0.097)
    #
    # 6. Hubble Parameter (H(z)):
    #    H(z) = H_0 * sqrt( Omega_m*(1 + z)^3 + Omega_Lambda )
    #    (Using typical redshift z=0.04 for the sample.)
    #
    # Conclusion:
    # In this simplified model, the entire V_DM(r) profile is determined by a single parameter: the halo mass M_200.
    ########################################################################################

    def _get_z(self) -> float:
        z = self.drpall_util.get_redshift(self.PLATE_IFU)
        print(f"Redshift z from DRPALL: {z:.5f}")
        return z

    # hubble parameter
    # H(z) = H0 * sqrt( Omega_m*(1 + z)^3 + Omega_Lambda )
    def _calc_Hz_kpc(self, z: float, H0=67.4, Om=0.315, Ol=0.685) -> float:
        Hz = H0 * np.sqrt(Om * (1 + z)**3 + Ol)
        return Hz # in km/s/Mpc

    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        r200_Mpc = V200 / (10 * Hz)  # in Mpc
        r200_kpc = r200_Mpc * 1e3  # convert to kpc
        return r200_kpc # in kpc
    
    # x = r / r200
    def _calc_x_from_r200(self, radius_kpc: np.ndarray, r200_kpc: float) -> np.ndarray:
        return radius_kpc / r200_kpc

   # c = r200 / rss
   # c = 5.74 * ( M200 / (2 * 10^12 * h^-1 * Msun) )^(-0.097)
    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        M_pivot = 2e12 / h
        return 5.74 * (M200 / M_pivot)**(-0.097)
    
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        G = const.G.to('Mpc km^2 / s^2 Msun').value  # Mpc km^2 / s^2 / Msun
        V200 = (10 * G * Hz * M200)**(1/3)  # in km/s
        return V200
    
    def _calc_M200_from_V200(self, V200: float, z: float) -> float:
        Hz = self._calc_Hz_kpc(z)  # in km/s/Mpc
        G = const.G.to('Mpc km^2 / s^2 Msun').value  # Mpc km^2 / s^2 / Msun
        M200 = V200**3 / (10 * G * Hz)  # in Msun
        return M200
    

    def _calc_M200_from_norm(self, M200_norm: float, mass_norm: float) -> float:
        return M200_norm * mass_norm
    
    def _calc_M200_norm(self, M200: float, mass_norm: float) -> float:
        return M200 / mass_norm

    ################################################################################
    # profile
    ################################################################################

    #
    # Drift Correction: V_drift^2 = 2 * sigma_0^2 * (R / R_d)
    #
    def _vel_drift_sq_profile(self, radius: np.ndarray, sigma_0:float, Re: float):
        Rd = Re / 1.678
        vel_drift_sq = 2 * sigma_0**2 * (radius / Rd)
        return vel_drift_sq
    

    # formula: Vdm ^ 2 = (V200 ^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
    def _vel_dm_sq_profile_V200(self, radius_kpc: np.ndarray, V200: float, c: float, z: float) -> np.ndarray:
        r200 = self._calc_r200_from_V200(V200, z)
        M200 = self._calc_M200_from_V200(V200, z)
        x = self._calc_x_from_r200(radius_kpc, r200)
        x = np.where(x == 0, 1e-6, x)  # avoid division by zero

        num = np.log(1 + c*x) - (c*x)/(1 + c*x)
        den = np.log(1 + c) - c/(1 + c)

        V_dm_sq = (V200**2 / x) * (num / den)
        return V_dm_sq

 
    def _vel_dm_sq_profile_M200(self, radius_kpc, M200, c, z):
        V200 = self._calc_V200_from_M200(M200, z)
        r200 = self._calc_r200_from_V200(V200, z)

        x = self._calc_x_from_r200(radius_kpc, r200)
        x = np.where(x == 0, 1e-6, x)

        num = np.log(1 + c*x) - (c*x)/(1 + c*x)
        den = np.log(1 + c) - c/(1 + c)

        V_dm_sq = (V200**2 / x) * (num / den)
        return V_dm_sq


    # V_rot^2 = V_star^2 + V_dm^2 - V_drift^2
    def _V_rot_sq_fit_model(self, radius: np.ndarray, M200: float, c: float, z: float, M_star: float, Re:float, f_bulge:float, a:float, sigma_0:float) -> np.ndarray:
        v_dm_sq = self._vel_dm_sq_profile_M200(radius, M200, c=c, z=z)
        v_star_sq = self.stellar_util._stellar_vel_sq_mass_profile(radius, M_star, Re, f_bulge, a)
        v_drift_sq = self._vel_drift_sq_profile(radius, sigma_0, Re)
        v_rot_sq = v_dm_sq + v_star_sq  - v_drift_sq
        v_rot_sq = np.where(v_rot_sq <= 0, 1e-6, v_rot_sq)  # avoid negative values

        return v_rot_sq
        

    ################################################################################
    # Fitting methods
    ################################################################################

    def _dm_nfw_fit_minimize(self, radius: np.ndarray, vel_obs: np.ndarray):
        # 1. Data Masking
        # Ensure we don't have NaNs and avoid the very center (resolution limits)
        valid_mask = (np.isfinite(vel_obs) & np.isfinite(radius) & 
                    (radius > 0.1) & (radius < 0.9 * np.nanmax(radius)))
        
        # Apply mask to ALL component arrays
        radius_valid = radius[valid_mask]
        vel_obs_valid = vel_obs[valid_mask]

        # geg parameters from stellar_util
        radius_max = np.nanmax(radius_valid)
        M_star = self.stellar_util.get_stellar_total_mass(radius_max)
        z = self._get_z()
        ydata = vel_obs_valid**2
        
        # Check if we have enough data points
        if len(ydata) < 5:
            print("Warning: Not enough valid data points for fitting.")
            return np.nan, radius, np.full_like(radius, np.nan), np.full_like(radius, np.nan)

        def fit_func_partial(params):         
            M200_norm, c, f_bulge, a, Re, sigma_0 = params
            M200 = self._calc_M200_from_norm(M200_norm, M_star)
            vals = self._V_rot_sq_fit_model(radius_valid, M200, c, z, M_star, f_bulge, a, Re, sigma_0)
            residuals = vals - ydata
            return np.sum(residuals**2)

        # to normalize the mass of M200
        p0 = [300, 10.0, 0.1, 1.0, 3.0, 20.0]  # Initial guess: M200_norm, c, f_bulge, a, Re, sigma_0
         # Lower and upper bounds
        bounds = [(1e1, 1e4),     # M200_norm bounds
                  (0.5, 100.0),     # c bounds
                  (0.01, 1.0),      # f_bulge ratios bounds
                  (0.1, 10.0),      # a bounds
                  (0.1, 20.0),      # Re bounds
                  (5.0, 100.0)]     # sigma_0 bounds

        try:
            # Perform Fit
            result = minimize(fit_func_partial, p0, bounds=bounds, method='L-BFGS-B')
            M200_norm_fit, c_fit, f_bulge_fit, a_fit, Re_fit, sigma_0_fit = result.x
            M200_fit = self._calc_M200_from_norm(M200_norm_fit, M_star)
        except RuntimeError:
            print("Fitting failed: Optimal parameters not found.")
            return np.nan, radius, np.zeros_like(radius), np.zeros_like(radius)
        
        V200_fit = self._calc_V200_from_M200(M200_fit, z)
        r200_fit = self._calc_r200_from_V200(V200_fit, z)

        vel_obs_sq_fit = self._V_rot_sq_fit_model(radius, M200_fit, c_fit, z, M_star, Re_fit, f_bulge_fit, a_fit, sigma_0_fit)
        vel_dm_sq_fit = self._vel_dm_sq_profile_M200(radius, M200_fit, c=c_fit, z=z)
        vel_star_sq_fit = self.stellar_util._stellar_vel_sq_mass_profile(radius, M_star, Re_fit, f_bulge_fit, a_fit)

        vel_total_fit = np.sqrt(np.maximum(vel_obs_sq_fit, 0))
        vel_dm_fit = np.sqrt(np.maximum(vel_dm_sq_fit, 0))
        vel_star_fit = np.sqrt(np.maximum(vel_star_sq_fit, 0))

        print("Fitted DM NFW parameters (minimize):")
        print(f" Fitted: M200: {M200_fit:.3e} Msun")
        print(f" Fitted: c: {c_fit:.3f}")
        print(f" Fitted: f_bulge: {f_bulge_fit:.3f}")
        print(f" Fitted: a: {a_fit:.3f} kpc")
        print(f" Fitted: Re: {Re_fit:.3f} kpc")
        print(f" Fitted: sigma_0: {sigma_0_fit:.3f} km/s")
        print(f" Calculated: V200: {V200_fit:.3f} km/s")
        print(f" Calculated: r200: {r200_fit:.3f} kpc")

        # calculate error estimate
        return radius, vel_total_fit, vel_dm_fit, vel_star_fit


    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return
    
    def set_stellar_util(self, stellar_util: Stellar) -> None:
        self.stellar_util = stellar_util
        return
        
    def fit_dm_nfw(self, radius: np.ndarray, vel_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z()
        radius_fit, vel_total, vel_dm_fit, vel_star_fit = self._dm_nfw_fit_minimize(radius, vel_obs)
        return radius_fit, vel_total, vel_dm_fit, vel_star_fit


######################################################
# main function for test
######################################################
def main():
    PLATE_IFU = "8723-12705"

    root_dir = Path(__file__).resolve().parent.parent
    fits_util = FitsUtil(root_dir / "data")
    drpall_file = fits_util.get_drpall_file()
    firefly_file = fits_util.get_firefly_file()
    maps_file = fits_util.get_maps_file(PLATE_IFU)

    print(f"DRPALL file: {drpall_file}")
    print(f"FIREFLY file: {firefly_file}")
    print(f"MAPS file: {maps_file}")

    drpall_util = DrpallUtil(drpall_file)
    firefly_util = FireflyUtil(firefly_file)
    maps_util = MapsUtil(maps_file)

    dm_nfw = DmNfw(drpall_util)
    dm_nfw.set_PLATE_IFU(PLATE_IFU)

    _, radius_h_kpc_map, _ = maps_util.get_radius_map()
    radius_max = np.nanmax(radius_h_kpc_map)
    radius_fit = np.linspace(0.0, radius_max, num=1000)
    
    print("### Test: DM based on M200")
    z = dm_nfw._get_z()
    M200 = 3*10**12 # example halo mass in Msun
    print(f"M200: {M200:.3e} Msun")
    V200 = dm_nfw._calc_V200_from_M200(M200, z)
    r200 = dm_nfw._calc_r200_from_V200(V200, z)
    c = dm_nfw._calc_c_from_M200(M200, h=0.7)
    print(f"Calculated V200: {V200:.2f} km/s, r200: {r200:.2f} kpc, c: {c:.2f}")

    vel_dm_sq = dm_nfw._vel_dm_sq_profile_M200(radius_fit, M200, c, z)  # Example usage
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")

    print("### Test: DM based on V200")
    V200 = 200
    c = 10
    print(f"V200: {V200:.2f} km/s, c: {c:.2f}")
    Hz = dm_nfw._calc_Hz_kpc(z)
    print(f"Hz: {Hz:.2f} km/s/Mpc")
    r200 = dm_nfw._calc_r200_from_V200(V200, z)
    M200 = dm_nfw._calc_M200_from_V200(V200, z)
    print(f"Calculated r200: {r200:.2f} kpc, M200 from V200: {M200:.3e} Msun")
    x = dm_nfw._calc_x_from_r200(radius_fit, r200)
    print(f"Calculated x  shape: {x.shape}, range: {np.nanmin(x):.5f} - {np.nanmax(x):.5f}")
    c = dm_nfw._calc_c_from_M200(M200, h=0.7)
    print(f"Calculated c from M200: {c:.2f}")
    
    vel_dm_sq = dm_nfw._vel_dm_sq_profile_V200(radius_fit, V200, c, z)
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")

# main entry
if __name__ == "__main__":
    main()