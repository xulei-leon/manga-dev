from pathlib import Path

from re import M
import numpy as np
from scipy.optimize import curve_fit
from astropy import constants as const
from astropy import units as u

from util.maps_util import MapsUtil
from util.drpall_util import DrpallUtil
from util.fits_util import FitsUtil
from util.firefly_util import FireflyUtil
from vel_stellar import Stellar
from vel_rot import PLATE_IFU, VelRot


class DmNfw:

    def __init__(self, drpall_util: DrpallUtil):
        self.drpall_util = drpall_util


    ########################################################################################
    # Use the following equation to fit DM profile:
    ########################################################################################
    # V_obs^2  =  V_star^2 + V_gas^2 + V_dm^2 - V_drift^2
    # V_obs: has been calculated
    # V_star: has been calculated
    # V_dm: use NFW profile to fit
    # V_drift^2 = 2 * sigma_0^2 * (R / R_d) : sigma_0  will be fitted, R_d has been calculated
    ########################################################################################

    # V_obs^2 = V_circ^2 - V_drift^2
    # V_drift^2 = 2 * sigma_0^2 * (R / R_d)
    def _v_drift_sq_profile(self, radius: np.ndarray, sigma_0: float, r_d: float) -> np.ndarray:
        v_drift_sq = 2 * np.square(sigma_0) * (radius / r_d)
        return v_drift_sq
    
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

    def _calc_H_from_z(self, z: float) -> float:
        H0 = 70.0
        Om0 = 0.3
        Ol0 = 0.7
        Hz_km_s_Mpc = H0 * np.sqrt(Om0 * (1 + z)**3 + Ol0)
        return Hz_km_s_Mpc

    def _calc_H_si(self, z: float) -> float:
        '''Return H(z) in s^-1.'''
        H_km_s_Mpc = self._calc_H_from_z(z)
        H = H_km_s_Mpc * (u.km / u.s / u.Mpc)
        return H.to(1/u.s).value

    # V200^3 = 10 * G * M200 * H(z)
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        '''V200 in km/s.'''
        # Use G in km^3 / (s^2 Msun) so that V200 is returned in km/s.
        G = const.G.to(u.km**3 / (u.s**2 * u.Msun)).value  # km^3 / (s^2 Msun)
        H_si = self._calc_H_si(z)                    # s^-1

        V200 = (10.0 * G * M200 * H_si)**(1/3)      # km/s
        return V200

    # R200 = V200 / (10 * H(z))
    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        '''r200 in kpc.'''
        H_si = self._calc_H_si(z)   # s^-1
        V200_m_s = V200 * 1000.0    # km/s → m/s
        r200_m = V200_m_s / (10.0 * H_si) # m
        r200_kpc = r200_m / u.kpc.to(u.m) # kpc
        return r200_kpc


    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        M_pivot = 2e12 / h
        return 5.74 * (M200 / M_pivot)**(-0.097)


    def _vel_dm_sq_profile(self, radius_kpc, M200, z=0.04, h=0.7):
        V200 = self._calc_V200_from_M200(M200, z)
        r200 = self._calc_r200_from_V200(V200, z)
        c = self._calc_c_from_M200(M200, h)

        x = radius_kpc / r200
        x = np.where(x == 0, 1e-6, x)

        num = np.log(1 + c*x) - (c*x)/(1 + c*x)
        den = np.log(1 + c) - c/(1 + c)

        V_dm_sq = (V200**2 / x) * (num / den)
        V_dm = np.sqrt(V_dm_sq)
        V_dm_max_val = float(np.nanmax(V_dm))
        radius_max_val = float(np.nanmax(radius_kpc))
        print(f"M200={M200:.3e} Msun, V200={V200:.2f} km/s, r200={r200:.2f} kpc, c={c:.2f}, radius={radius_max_val:.2f} kpc -> V_dm={V_dm_max_val:.2f} km/s")

        return V_dm_sq
    
    def _calc_M200_from_rho_rs(self, rho_s: float, r_s: float, z: float=0.04, h: float=0.7):
        # M200 = 4 * pi * rho_s * r_s^3 * [ ln(1 + c) - c/(1 + c) ]
        H_si = self._calc_H_si(z)   # s^-1
        H_kpc = (H_si * u.s**-1).to(u.km/u.s/u.kpc).value
        r200 = r_s * ( (3 * rho_s) / (200 * (3 * H_kpc**2) / (8 * np.pi * const.G.to('kpc km^2 / s^2 Msun').value)) )**(1/3)
        c = r200 / r_s
        M200 = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + c) - c/(1 + c))
        return M200

    def _vel_dm_sq_rho_rs_profile(self, radius: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
        # V_DM^2(r) = 4 * pi * G * rho_s * r_s^3 * [ ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s) ] / r
        G = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun
        x = radius / r_s
        x = np.where(x == 0, 1e-6, x)  # avoid division by zero

        num = np.log(1 + x) - (x)/(1 + x)
        vel_dm_sq = (4 * np.pi * G * rho_s * r_s**3 * num) / radius
        return vel_dm_sq
    

    # V_rot^2 = V_star^2 + V_dm^2 - V_drift^2
    def _V_rot_sq_fit_model(self, radius: np.ndarray, log_M200: float, v_star_sq: np.ndarray, v_drift_sq: np.ndarray, z: float=0.04, h: float=0.7) -> np.ndarray:
        M200 = 10**log_M200
        v_dm_sq = self._vel_dm_sq_profile(radius, M200, z=z, h=h)
        v_rot_sq = v_star_sq + v_dm_sq - v_drift_sq
        v_rot_sq = np.where(v_rot_sq <= 0, 1e-6, v_rot_sq)  # avoid negative values

        print(f"Fit M200={M200:.3e} Msun -> V_rot: {np.sqrt(np.nanmax(v_rot_sq)):.2f}, V_dm: {np.sqrt(np.nanmax(v_dm_sq)):.2f}, V_star: {np.sqrt(np.nanmax(v_star_sq)):.2f}, V_drift: {np.sqrt(np.nanmax(v_drift_sq)):.2f}")
        return v_rot_sq

    # used the minimum χ 2 method for fitting
    def _dm_nfw_fit(self, radius: np.ndarray, vel_obs: np.ndarray, vel_star_sq: np.ndarray, vel_drift_sq: np.ndarray, r_d:float, z: float=0.04, h: float=0.7):
        valid_mask = np.isfinite(vel_obs) & np.isfinite(radius) & (radius > 0.1) & (radius < 0.9 * np.nanmax(radius))
        radius_valid = radius[valid_mask]
        vel_obs_valid = vel_obs[valid_mask]
        vel_star_sq_valid = vel_star_sq[valid_mask]
        vel_drift_sq_valid = vel_drift_sq[valid_mask]

        # Initial guess for parameters: log_M200
        p0 = [11.5]  # log_M200
        lb = [10.0]  # log_M200
        ub = [15.0]  # log_M200
        xdata = radius_valid
        ydata = vel_obs_valid**2

        # Ensure ydata are finite and positive
        ydata = np.where(~np.isfinite(ydata) | (ydata <= 0), 1e-6, ydata)

        # Perform the fit; ensure model output is finite at initial p0
        def fit_func_partial(r, log_M200):
            vals = self._V_rot_sq_fit_model(r, log_M200, vel_star_sq_valid, vel_drift_sq_valid, z=z, h=h)
            return np.where(~np.isfinite(vals) | (vals <= 0), 1e-6, vals)

        popt, pcov = curve_fit(fit_func_partial, xdata, ydata, p0=p0, bounds=(lb, ub))
        # popt is an array (shape (1,)); extract scalar value
        log_M200_fit = float(popt[0])
        M200_fit = 10**log_M200_fit

        # Return the fitted DM velocity profile
        vel_obs_sq_fit = self._V_rot_sq_fit_model(radius, log_M200_fit, vel_star_sq, vel_drift_sq, z=z, h=h)
        vel_dm_sq_fit = self._vel_dm_sq_profile(radius, M200_fit, z=z, h=h)
        vel_total_fit = np.sqrt(vel_obs_sq_fit)
        vel_dm_fit = np.sqrt(vel_dm_sq_fit)

        print("Fitted DM NFW parameters:")
        print(f"  M200_fit: {M200_fit:.3e} Msun")
        print(f"pcovariance matrix:\n{pcov}")

        return M200_fit, radius, vel_total_fit, vel_dm_fit
    

    ################################################################################
    # public methods
    ################################################################################
    def set_PLATE_IFU(self, PLATE_IFU: str) -> None:
        self.PLATE_IFU = PLATE_IFU
        return

    def fit_dm_nfw(self, radius: np.ndarray, vel_obs: np.ndarray, vel_star_sq: np.ndarray, V_drift_sq: np.ndarray, r_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z()
        M200_fit, radius_fit, vel_total, vel_dm = self._dm_nfw_fit(radius, vel_obs, vel_star_sq, V_drift_sq, r_d, z=z)
        return M200_fit, radius_fit, vel_total, vel_dm
    

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
    z = dm_nfw._get_z()
    M200 = 10**11.5 # example halo mass in Msun
    vel_dm_sq = dm_nfw._vel_dm_sq_profile(radius_h_kpc_map, M200, z=z)  # Example usage
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")

    M200 = 10**13 # example halo mass in Msun
    vel_dm_sq = dm_nfw._vel_dm_sq_profile(radius_h_kpc_map, M200, z=z)  # Example usage
    vel_dm = np.sqrt(vel_dm_sq)
    print(f"Calculated V_DM  shape: {vel_dm.shape}, range: {np.nanmin(vel_dm):.2f} - {np.nanmax(vel_dm):.2f} km/s")


# main entry
if __name__ == "__main__":
    main()