# Data Processing Pipeline for MaNGA Rotation-Curve Fitting and NFW Inference

## 1. Overview

This document outlines the end-to-end data processing pipeline used to derive gas rotation curves from MaNGA Data Analysis Pipeline (DAP) products and to infer the dark matter halo parameters (specifically the NFW profile) using Bayesian methods. The pipeline transforms 2D integral field unit (IFU) kinematic maps into 1D rotation curves, performs dynamical mass decomposition, and samples the posterior distribution of the halo parameters.

## 2. Data Sources and Sample Selection

### 2.1 Primary Data Products
- **MaNGA DRPALL**: Summary table providing galaxy metadata, redshifts, and morphological proxies (e.g., Sérsic index, axis ratio).
- **MaNGA MAPS (DAP)**: Emission-line gas velocity fields (H$\alpha$), inverse variance (IVAR) maps, and signal-to-noise ratio (SNR) maps.
- **MaNGA Firefly**: Spatially resolved stellar population properties, providing stellar mass density maps.

### 2.2 Sample Selection and Quality Control
To ensure reliable kinematic modeling, galaxies are filtered based on their geometric and observational properties:
- **Inclination Cut**: Galaxies are restricted to intermediate inclinations, $25^\circ \le i \le 70^\circ$. Face-on galaxies ($i < 25^\circ$) suffer from severe deprojection uncertainties, while edge-on galaxies ($i > 70^\circ$) are heavily affected by line-of-sight integration and dust extinction.
- **Spaxel Filtering**: Within the 2D velocity maps, we only retain spaxels that meet the following criteria:
  - $\text{SNR} \ge 10.0$ to ensure robust velocity measurements.
  - Azimuthal angle within $\pm 45^\circ$ of the kinematic major axis to minimize the impact of non-circular motions and radial flows.
  - The lowest 10% of IVAR values are discarded to remove anomalously noisy spaxels.

## 3. Rotation Curve Extraction

### 3.1 Geometric Projection Model
The observed 2D line-of-sight velocity field $V_{\mathrm{obs}}(r, \phi)$ is modeled as a projection of the intrinsic 1D rotation curve $V_{\mathrm{rot}}(r)$:
$$ V_{\mathrm{obs}}(r, \phi) = V_{\mathrm{sys}} + V_{\mathrm{rot}}(r) \sin i \cos(\phi - \phi_0) $$
where:
- $V_{\mathrm{sys}}$ is the systemic velocity of the galaxy.
- $i$ is the disk inclination angle.
- $\phi$ is the azimuthal angle in the plane of the sky.
- $\phi_0$ is the position angle of the kinematic major axis.
- $r$ is the deprojected galactocentric radius.

### 3.2 Empirical Rotation Curve Fitting
To extract the global kinematic structure, we fit the 2D velocity field using an empirical asymptotic function:
$$ V_{\mathrm{rot}}(r) = V_c \tanh\left(\frac{r}{R_t}\right) + s_{\mathrm{out}} r $$
where $V_c$ is the asymptotic velocity amplitude, $R_t$ is the turnover radius characterizing the inner velocity gradient, and $s_{\mathrm{out}}$ allows for a linearly rising or falling outer rotation curve. The fit is performed using a Levenberg-Marquardt optimizer, weighted by the IVAR of each spaxel. Galaxies with poor fits (e.g., normalized root-mean-square error $\text{NRMSE} > 0.1$ or reduced $\chi^2 > 10$) are excluded from further analysis.

## 4. Dynamical Mass Decomposition

The intrinsic gas rotation velocity $V_{\mathrm{rot}}(r)$ traces the total gravitational potential, modified by pressure support (asymmetric drift). The total circular velocity squared is the sum of the contributions from the stellar mass and the dark matter halo:
$$ V_{\mathrm{rot}}^2(r) = V_{\star}^2(r) + V_{\mathrm{dm}}^2(r) - V_{\mathrm{drift}}^2(r) $$

### 4.1 Stellar Component
The stellar mass distribution is modeled as a composite of a spherical Hernquist bulge and a Freeman exponential disk:
$$ V_{\star}^2(r) = V_{\mathrm{bulge}}^2(r) + V_{\mathrm{disk}}^2(r) $$
- **Hernquist Bulge**:
  $$ V_{\mathrm{bulge}}^2(r) = \frac{G M_{\mathrm{bulge}} r}{(r+a)^2} $$
  where $M_{\mathrm{bulge}} = f_{\mathrm{bulge}} M_{\star}$ is the bulge mass, and $a$ is the scale radius.
- **Exponential Disk**:
  $$ V_{\mathrm{disk}}^2(r) = \frac{2 G M_{\mathrm{disk}}}{R_d} y^2 \left[ I_0(y)K_0(y) - I_1(y)K_1(y) \right] $$
  where $M_{\mathrm{disk}} = (1 - f_{\mathrm{bulge}}) M_{\star}$ is the disk mass, $R_d$ is the disk scale length, $y = r / (2 R_d)$, and $I_n, K_n$ are modified Bessel functions.

### 4.2 Dark Matter Halo
The dark matter halo is modeled using the Navarro-Frenk-White (NFW) profile:
$$ V_{\mathrm{dm}}^2(r) = \frac{V_{200}^2}{x} \frac{\ln(1+cx) - \frac{cx}{1+cx}}{\ln(1+c) - \frac{c}{1+c}} $$
where $x = r/R_{200}$, $c$ is the concentration parameter, and $V_{200}$ is the circular velocity at the virial radius $R_{200}$, which is directly related to the halo mass $M_{200}$ via $V_{200} = (10 G H(z) M_{200})^{1/3}$.

### 4.3 Asymmetric Drift Correction
Because the gas is not perfectly collisionless, thermal pressure provides partial support against gravity. This is corrected using an asymmetric drift term, approximated as linear in radius:
$$ V_{\mathrm{drift}}^2(r) = 2 \sigma_0^2 \left(\frac{r}{R_d}\right) $$
where $\sigma_0$ is the characteristic velocity dispersion scale.

## 5. Bayesian Inference Framework

We employ a Bayesian framework implemented in PyMC (using the NUTS sampler) to infer the posterior distributions of the mass model parameters. This approach naturally handles the well-known disk-halo degeneracy by incorporating physically motivated priors.

### 5.1 Prior Specifications

- **Stellar Mass ($M_{\star}$)**: $\mathcal{LN}(\mu = \ln M_{\star,\mathrm{obs}}, \sigma = 0.05 \ln 10)$. A tight log-normal prior anchored to the observed photometric mass (from NSA/Firefly), allowing 0.05 dex flexibility.
- **Halo Mass ($M_{200}$)**: Depending on the inference mode, this is either a wide independent prior (Truncated Normal in log space, $\mu=12.0, \sigma=1.0$ dex) or anchored to the Stellar-to-Halo Mass Relation (SHMR) via a log-normal prior ($\sigma=0.2$ dex).
- **Concentration ($c$)**: Modeled as a log-normal distribution. In the SHMR-anchored mode, it is tightly constrained by the theoretical $c-M_{200}$ relation ($\sigma=0.11$ dex). In the independent mode, it is centered around $c \approx 9$ with $\sigma=0.2$ dex.
- **Bulge Mass Fraction ($f_{\mathrm{bulge}}$)**: To avoid the stochasticity of population-level relations, we anchor this prior to the galaxy's specific Sérsic index $n$. Using an empirical logit-linear relation: $\text{logit}(f_{\mathrm{bulge}}) \sim \mathcal{N}(\mu = 1.2(n - 2.5), \sigma = 0.2)$.
- **Bulge Scale Radius ($a$)**: $\mathcal{LN}(\mu = \ln(0.13 R_e), \sigma = 0.3)$. Centered on empirical scaling relations for bulge sizes, with a width that prevents unphysical runaway during MCMC sampling.
- **Velocity Dispersion ($\sigma_0$)**: $\mathcal{LN}(\mu = \ln 5, \sigma = 0.3 \ln 10)$ km/s, providing a weakly informative prior for the asymmetric drift.
- **Geometric Parameters**: The systemic velocity $V_{\mathrm{sys}}$, inclination $i$, and kinematic position angle offset $\phi_{\mathrm{delta}}$ are assigned narrow normal or truncated normal priors centered on the values derived from the 2D geometric fit.
- **Error Scale ($\sigma_{\mathrm{scale}}$)**: A multiplicative factor $\mathcal{LN}(\mu = \ln\sqrt{\bar{\sigma}_{\mathrm{ivar}}^2 + \sigma_{\mathrm{sys}}^2}, \sigma = 0.3)$ to absorb potential underestimation of the IFU velocity uncertainties.

### 5.2 Likelihood and Spatial Weighting

The likelihood of the observed velocity field assumes Gaussian errors scaled by $\sigma_{\mathrm{scale}}$:
$$ V_{\mathrm{obs}} \sim \mathcal{N}(V_{\mathrm{obs,model}}, \sigma_{\mathrm{scale}} \sigma_{\mathrm{ivar}}) $$

To mitigate the influence of non-circular motions and beam smearing in the central regions, we apply a smooth radial down-weighting via an auxiliary potential in the log-probability. A logistic ramp function reduces the statistical weight of spaxels within the inner 30% of the observed radial extent, ensuring that the global halo parameters are primarily driven by the more reliable outer rotation curve.

## 6. Known Limitations and Physical Considerations

- **Disk-Halo Degeneracy**: The fundamental degeneracy between the stellar mass-to-light ratio and the dark matter halo concentration is mitigated by our strong, object-specific priors on $M_{\star}$ and $f_{\mathrm{bulge}}$ (derived from Firefly and Sérsic index $n$).
- **Beam Smearing**: The spatial resolution of MaNGA ($\sim 2.5''$ FWHM) can artificially flatten inner velocity gradients. While our radial down-weighting scheme reduces its impact on the global fit, the inner bulge parameters may still be affected.
- **Adiabatic Contraction**: The current model assumes a pure dark-matter-only NFW profile and does not explicitly account for halo contraction due to baryonic cooling, which may slightly alter the inferred inner dark matter density.
