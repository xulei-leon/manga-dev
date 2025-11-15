from re import M
import numpy as np
from scipy.optimize import curve_fit
from astropy import constants as const
from astropy import units as u

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
        H0 = 70.0
        Om0 = 0.3
        Ol0 = 0.7
        Hz_km_s_Mpc = H0 * np.sqrt(Om0 * (1 + z)**3 + Ol0)
        return Hz_km_s_Mpc


    def _calc_H_si(self, z: float) -> float:
        """Return H(z) in s^-1."""
        H_km_s_Mpc = self._calc_H_from_z(z)
        H = H_km_s_Mpc * (u.km / u.s / u.Mpc)
        return H.to(1/u.s).value


    def _calc_V200_from_M200(self, M200: float, z: float) -> float:
        G = const.G.to('kpc km^2 / s^2 Msun').value  # kpc km^2 / s^2 / Msun
        H_si = self._calc_H_si(z)                    # s^-1
        # Convert H_si to km/s/kpc:
        H_kpc = (H_si * u.s**-1).to(u.km/u.s/u.kpc).value
        V200 = (10 * G * H_kpc * M200)**(1/3)
        return V200


    def _calc_r200_from_V200(self, V200: float, z: float) -> float:
        """r200 in kpc."""
        H_si = self._calc_H_si(z)   # s^-1
        V200_m_s = V200 * 1000.0    # km/s → m/s
        r200_m = V200_m_s / (10 * H_si)
        r200_kpc = (r200_m / u.kpc.to(u.m))
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

        return (V200**2 / x) * (num / den)
    
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
        return v_rot_sq

    # used the minimum χ 2 method for fitting
    def _dm_nfw_fit(self, radius: np.ndarray, vel_rot_sq: np.ndarray, vel_star_sq: np.ndarray, vel_drift_sq: np.ndarray, r_d:float, z: float=0.04, h: float=0.7):
        valid_mask = np.isfinite(vel_rot_sq) & np.isfinite(radius) & (radius > 0.1) & (radius < 0.9 * np.nanmax(radius))
        radius_valid = radius[valid_mask]
        vel_rot_valid = vel_rot_sq[valid_mask]
        vel_star_sq_valid = vel_star_sq[valid_mask]
        vel_drift_sq_valid = vel_drift_sq[valid_mask]

        # Initial guess for parameters: log_M200, sigma_0
        p0 = [11.5]  # log_M200
        lb = [10.0]  # log_M200
        ub = [15.0]  # log_M200
        xdata = radius_valid
        ydata = vel_rot_valid**2

        # Perform the fit
        fit_func_partial = lambda r, log_M200: self._V_rot_sq_fit_model(r, log_M200, vel_star_sq_valid, vel_drift_sq_valid, z=z, h=h)
        popt, pcov = curve_fit(fit_func_partial, xdata, ydata, p0=p0, bounds=(lb, ub))
        # popt is an array (shape (1,)); extract scalar value
        log_M200_fit = float(popt[0])
        M200_fit = 10**log_M200_fit

        # Return the fitted DM velocity profile
        vel_rot_sq_fit = self._V_rot_sq_fit_model(radius, log_M200_fit, vel_star_sq, vel_drift_sq, z=z, h=h)
        vel_dm_sq_fit = self._vel_dm_sq_profile(radius, M200_fit, z=z, h=h)
        vel_total_fit = np.sqrt(vel_rot_sq_fit)
        vel_dm_fit = np.sqrt(vel_dm_sq_fit)

        print("Fitted DM NFW parameters:")
        print(f"  M200_fit: {M200_fit:.3e} Msun")
        print(f"pcovariance matrix:\n{pcov}")

        return M200_fit, radius, vel_total_fit, vel_dm_fit
    

    ################################################################################
    # public methods
    ################################################################################

    def fit_dm_nfw(self, PLATE_IFU: str, radius: np.ndarray, vel_rot_sq: np.ndarray, vel_star_sq: np.ndarray, V_drift_sq: np.ndarray, r_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self._get_z(PLATE_IFU)
        M200_fit, radius_fit, vel_total, vel_dm = self._dm_nfw_fit(radius, vel_rot_sq, vel_star_sq, V_drift_sq, r_d, z=z)
        return M200_fit, radius_fit, vel_total, vel_dm