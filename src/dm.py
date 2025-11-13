import numpy as np
from scipy.optimize import curve_fit
from astropy import constants as const

from util.drpall_util import DrpallUtil
from vel_stellar import Stellar
from vel_rot import PLATE_IFU, VelRot


class DmNfw:

    def __init__(self, drpall_util: DrpallUtil, stellar: Stellar, vel_rot: VelRot):
        self.drpall_util = drpall_util
        self.stellar = stellar
        self.vel_rot = vel_rot

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

    def _get_z(self, PLATE_IFU: str) -> float:
        z = self.drpall_util.get_redshift(PLATE_IFU)
        print(f"Redshift z from DRPALL: {z:.5f}")
        return z

    def _calc_H_from_z(self, z: float) -> float:
        """Calculate the Hubble parameter at redshift z."""
        H0 = 70.0 # km/s/Mpc
        Om0 = 0.3
        Ol0 = 0.7
        Hz = H0 * np.sqrt(Om0 * np.power(1 + z, 3) + Ol0)
        return Hz
    
    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        # V200 = (10 G H(z) M200)^(1/3)
        G_kpc_kms_Msun = const.G.to('kpc km^2 / s^2 Msun').value
        H_kpc = self._calc_H_from_z(z) / 1000.0  # convert H from km/s/Mpc to km/s/kpc
        V200 = (10.0 * G_kpc_kms_Msun * H_kpc * M200)**(1/3)  # in km/s
        return V200
    
    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        """Calculate the virial radius r200 from V200 at redshift z."""
        H = self._calc_H_from_z(z)
        return V200 / (10.0 * H) * 1000.0  # convert Mpc to kpc

    def _calc_c_from_M200(self, M200: float, h: float) -> float:
        # c = 5.74 * ( M_200 / (2 * 10^12 * h^-1 * M_sun) )^(-0.097)
        M_pivot = 2e12 / h  # in Msun
        c = 5.74 * np.power(M200 / M_pivot, -0.097)
        return c
    
    def _vel_dm_sq_profile(self, radius_kpc: np.ndarray, M200_Msun: float, z: float=0.04, h: float=0.7) -> np.ndarray:
        """Calculate the NFW dark matter halo rotational velocity squared profile."""
        V200 = self._calc_V200_from_M200(M200_Msun, z)  # in km/s
        r200_kpc = self._calc_r200_from_V200(V200, z)  # in kpc
        c = self._calc_c_from_M200(M200_Msun, h)  # concentration parameter

        x = radius_kpc / r200_kpc
        x = np.where(x == 0, 1e-6, x)  # avoid division by zero

        numerator = np.log(1 + c * x) - (c * x) / (1 + c * x)
        denominator = np.log(1 + c) - c / (1 + c)

        V_dm_sq = (V200**2 / x) * (numerator / denominator)
        return V_dm_sq
    

    # V_rot^2 = V_star^2 + V_dm^2 - V_drift^2
    def _dm_nfw_vel_rot_fit_profile(self, radius: np.ndarray, M200: float, sigma_0: float, r_d: float, star_mass: float, z: float=0.04, h: float=0.7) -> np.ndarray:
        v_star_sq = self.stellar._stellar_vel_sq_mass_profile(radius, star_mass, r_d)
        v_dm_sq = self._vel_dm_sq_profile(radius, M200, z=z, h=h)
        v_drift_sq = self._v_drift_sq_profile(radius, sigma_0, r_d)
        v_rot_sq = v_star_sq + v_dm_sq - v_drift_sq
        v_rot_sq = np.where(v_rot_sq <= 0, 1e-6, v_rot_sq)  # avoid negative values
        v_rot = np.sqrt(v_rot_sq)
        return v_rot

    # used the minimum χ 2 method for fitting
    def _dm_nfw_fit(self, radius: np.ndarray, vel_obs_map, star_mass: float, z: float=0.04, h: float=0.7):
        valid_mask = np.isfinite(vel_obs_map) & np.isfinite(radius) & (radius > 0.1)
        radius_valid = radius[valid_mask]
        vel_obs_valid = vel_obs_map[valid_mask]

        # Initial guess for parameters: M200, sigma_0
        initial_guess = [1e12, 70.0, 3.0]  # M200 in Msun, sigma_0 in km/s, r_d in kpc
        bounds = ([1e8, 1.0, 0.1], [1e17, 350.0, 10.0])  # M200 in Msun, sigma_0 in km/s

        # Perform the fit
        fit_func_partial = lambda r, M200, sigma_0, r_d: self._dm_nfw_vel_rot_fit_profile(r, M200, sigma_0, r_d, star_mass=star_mass, z=z, h=h)
        popt, _ = curve_fit(fit_func_partial, radius_valid, vel_obs_valid, p0=initial_guess, bounds=bounds)
        M200_fit, sigma_0_fit, r_d_fit = popt

        # Return the fitted DM velocity profile
        vel_total_fit = self._dm_nfw_vel_rot_fit_profile(radius, M200_fit, sigma_0_fit, r_d_fit, star_mass=star_mass, z=z, h=h)
        vel_dm_sq_fit = self._vel_dm_sq_profile(radius, M200_fit, z=z, h=h)
        vel_dm_fit = np.sqrt(vel_dm_sq_fit)

        print("Fitted DM NFW parameters:")
        print(f"  M200: {M200_fit:.3e} Msun")
        print(f"  sigma_0: {sigma_0_fit:.3f} km/s")
        print(f"  r_d: {r_d_fit:.3f} kpc")

        return M200_fit, radius, vel_total_fit, vel_dm_fit
    

    ################################################################################
    # public methods
    ################################################################################

    def fit_dm_vel(self, PLATE_IFU: str, radius_map: np.ndarray, vel_obs_map: np.ndarray, vel_stellar_sq: np.ndarray, r_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z(PLATE_IFU)
        _, star_mass = self.drpall_util.get_stellar_mass(PLATE_IFU)
        M200_fit, radius_fit, vel_total, vel_dm = self._dm_nfw_fit(radius_map, vel_obs_map, star_mass, z=z)
        return M200_fit, radius_fit, vel_total, vel_dm