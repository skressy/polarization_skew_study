"""
magnetic_fields.py
==================
Magnetic field geometry generators and synthetic polarization pipeline.

Public API
----------
simulate_field(geometry, turbulence, M_A, alpha, iaxis, g_size, **kwargs)
    -> P_obs, phi_obs, cos2g_obs

Geometries  : 'uniform', 'wavy', 'helical', 'hourglass'
Turbulence  : 'gs95', 'kolmogorov', 'random'

Each geometry returns (Bx0, By0, Bz0, b_hat, B0_ref) where:
  b_hat  — unit vector of mean field direction for GS95 anisotropy
  B0_ref — reference field amplitude for turbulence normalisation
            (RMS(δB) = M_A * B0_ref)
"""

import numpy as np
from scipy.special import jn_zeros, j0, j1, erfc


# ============================================================
# 1.  GRID
# ============================================================

def _make_grid(g_size):
    x = np.linspace(-np.pi, np.pi, g_size)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return x, X, Y, Z


# ============================================================
# 2.  GEOMETRIES  — each returns (Bx0, By0, Bz0, b_hat, B0_ref)
# ============================================================

def _los_b_hat(iaxis):
    """Return unit vector along the LOS axis."""
    return [np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([0., 0., 1.])][iaxis]


def _geometry_uniform(X, Y, Z, alpha, iaxis=2, **_):
    """
    Uniform POS field at angle alpha in the XY plane.
    Bx = cos(alpha), By = sin(alpha), Bz = 0
    b_hat : along LOS — avoids stripe artefacts from POS-aligned anisotropy axis
    B0_ref: 1.0 (unit field)
    """
    Bx0 = np.full_like(X, np.cos(alpha))
    By0 = np.full_like(X, np.sin(alpha))
    Bz0 = np.zeros_like(X)
    b_hat  = _los_b_hat(iaxis)
    B0_ref = 1.0
    return Bx0, By0, Bz0, b_hat, B0_ref


def _geometry_wavy(X, Y, Z, alpha, iaxis=2,
                   frequency=1.0, amplitude_range=(1.0, 1.0), **_):
    """
    Wavy field: field direction rotates as a cosine wave in X across the POS.
    theta(x) = amplitude * cos(frequency * x)
    Bx = cos(theta), By = sin(theta), Bz = 0  — same for every Z slice.
    b_hat : along LOS — no meaningful single POS mean direction
    B0_ref: RMS(|B0|) over the grid
    """
    g_size    = X.shape[2]
    amplitude = np.linspace(amplitude_range[0], amplitude_range[1], g_size)

    Bx0 = np.zeros_like(X)
    By0 = np.zeros_like(X)
    Bz0 = np.zeros_like(X)

    for iz in range(g_size):
        theta         = amplitude[iz] * np.cos(frequency * X[:, :, iz])
        Bx0[:, :, iz] = np.cos(theta)
        By0[:, :, iz] = np.sin(theta)

    b_hat  = _los_b_hat(iaxis)
    B0_ref = np.sqrt(np.mean(Bx0**2 + By0**2 + Bz0**2))
    return Bx0, By0, Bz0, b_hat, B0_ref


def _geometry_helical(X, Y, Z, alpha, iaxis=2, **_):
    """
    Helical / toroidal field.
    B_phi = cos(alpha)/r,  B_z = sin(alpha)
    b_hat : along LOS — toroidal components cancel so no net POS mean direction
    B0_ref: RMS(|B0|) — avoids 1/r divergence dominating the scale
    """
    r   = np.sqrt(X**2 + Y**2) + 1e-8
    Bx0 = (-Y * np.cos(alpha)) / r
    By0 = ( X * np.cos(alpha)) / r
    Bz0 = np.full_like(X, np.sin(alpha))
    b_hat  = _los_b_hat(iaxis)
    B0_ref = np.sqrt(np.mean(Bx0**2 + By0**2 + Bz0**2))
    return Bx0, By0, Bz0, b_hat, B0_ref

def _geometry_hourglass(X, Y, Z, alpha, iaxis=2, **_):
    # Parameters
    g_size    = X.shape[2]
    R_outer = 5.0
    h = 1.0
    B0 = 1.0
    Bm = 1.0
    betam = Bm / B0
    N_terms = 3
    roots = jn_zeros(1, N_terms)

    # CARTESIAN GRID
    x = np.linspace(-0.25, 0.25, g_size)
    y = np.linspace(-0.25, 0.25, g_size)
    z = np.linspace(-0.25, 0.25, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    R = np.sqrt(X**2 + Y**2)
    R_safe = np.where(R == 0, 1e-12, R)

    # BESSEL SERIES
    weights = 2 / (roots * j0(roots)**2)
    eta = h / R_outer

    Br0 = np.zeros_like(R)
    Bz0 = np.zeros_like(R)

    for n in range(N_terms):

        lam = roots[n]
        w = weights[n]

        J0_eval = j0(lam * R)
        J1_eval = j1(lam * R)

        arg_neg = lam * eta / 2 - Z / eta
        arg_pos = lam * eta / 2 + Z / eta

        er_neg = erfc(arg_neg)
        er_pos = erfc(arg_pos)

        exp_neg = np.exp(-lam * Z)
        exp_pos = np.exp(lam * Z)

        brackets0 = er_pos * exp_pos + er_neg * exp_neg
        brackets1 = er_neg * exp_neg - er_pos * exp_pos

        Br0 += w * J1_eval * betam * brackets1
        Bz0 += w * J0_eval * betam * brackets0
        
    Bz0 += 1.0

    # CYLINDRICAL → CARTESIAN
    Bx0 = Br0 * (X/ R_safe)
    By0 = Br0 * (Y/ R_safe)

    b_hat  = _los_b_hat(iaxis)
    B0_ref = np.sqrt(np.mean(Bx0**2 + By0**2 + Bz0**2))
    return Bx0, By0, Bz0, b_hat, B0_ref

_GEOMETRIES = {
    'uniform'   : _geometry_uniform,
    'wavy'      : _geometry_wavy,
    'helical'   : _geometry_helical,
    'hourglass' : _geometry_hourglass,
}


# ============================================================
# 3.  SHARED TURBULENCE HELPERS
# ============================================================

def _normalise_turbulence(dBx, dBy, dBz, M_A, B0_ref):
    """Scale δB so that RMS(δB) = M_A * B0_ref."""
    rms = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    if rms > 0 and B0_ref > 0:
        scale = M_A * B0_ref / rms
        dBx  *= scale
        dBy  *= scale
        dBz  *= scale
    return dBx, dBy, dBz


def _make_k_grid(g_size, dx):
    kx = np.fft.fftfreq(g_size, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k2   = KX**2 + KY**2 + KZ**2
    k    = np.sqrt(k2)
    k2_s = np.where(k2 > 0, k2, 1.0)
    return KX, KY, KZ, k, k2, k2_s


def _solenoidal_project(Fx, Fy, Fz, KX, KY, KZ, k2_s):
    """Remove longitudinal component so ∇·δB = 0."""
    kdF = KX*Fx + KY*Fy + KZ*Fz
    Fx -= KX * kdF / k2_s
    Fy -= KY * kdF / k2_s
    Fz -= KZ * kdF / k2_s
    return Fx, Fy, Fz


def _rand_c(shape, rng):
    return (rng.standard_normal(shape) +
            1j * rng.standard_normal(shape)) / np.sqrt(2)


# ============================================================
# 4.  TURBULENCE MODELS
# ============================================================

def _turbulence_gs95(X, Y, Z, Bx0, By0, Bz0, M_A, B0_ref, b_hat,
                     anisotropy_strength=1.0, seed=None, **_):
    """
    GS95 anisotropic MHD turbulence.
    Sub-Alfvénic  : weak (k_⊥ < k_trans) + GS95 (k_⊥ > k_trans)
    Super-Alfvénic: hydro (k < k_trans)  + GS95 (k   > k_trans)
    References: GS95, CL02, L06
    b_hat is passed in from the geometry — NOT inferred from the field arrays.
    """
    g_size = X.shape[0]
    rng    = np.random.default_rng(seed)

    dx    = (X.max() - X.min()) / (g_size - 1)
    L     = g_size * dx
    k_box = 2.0 * np.pi / L

    KX, KY, KZ, k, k2, k2_s = _make_k_grid(g_size, dx)

    k_par  = KX*b_hat[0] + KY*b_hat[1] + KZ*b_hat[2]
    k_perp = np.sqrt(np.maximum(k2 - k_par**2, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        W_hydro = np.where(k > 0, k**(-11./6.), 0.)
        W_weak  = np.where(k_perp > 0, k_perp**(-2.), W_hydro)
        k_p_cb  = np.where(k_perp > 0, k_perp**(2./3.), 1.)
        cb_arg  = np.where(k_perp > 0,
                           (k_par / k_p_cb)**2 * anisotropy_strength, 0.)
        W_gs95  = np.where(k_perp > 0,
                           k_perp**(-11./6.) * np.exp(-cb_arg), W_hydro)

    if M_A <= 1.0:
        k_trans = k_box / M_A**2
        W = np.where(k_perp < k_trans, W_weak, W_gs95)
    else:
        k_trans = k_box * M_A**3
        W = np.where(k < k_trans, W_hydro, W_gs95)

    W = np.where(k2 > 0, W, 0.)
    W_rms = np.sqrt(np.mean(W**2))
    if W_rms > 0:
        W /= W_rms

    Fx = _rand_c(KX.shape, rng) * W
    Fy = _rand_c(KX.shape, rng) * W
    Fz = _rand_c(KX.shape, rng) * W
    Fx, Fy, Fz = _solenoidal_project(Fx, Fy, Fz, KX, KY, KZ, k2_s)

    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    return _normalise_turbulence(dBx, dBy, dBz, M_A, B0_ref)


def _turbulence_kolmogorov(X, Y, Z, Bx0, By0, Bz0, M_A, B0_ref, b_hat,
                           seed=None, **_):
    """
    Isotropic Kolmogorov turbulence (k^{-11/6} amplitude spectrum).
    """
    g_size = X.shape[0]
    rng    = np.random.default_rng(seed)

    dx = (X.max() - X.min()) / (g_size - 1)
    KX, KY, KZ, k, k2, k2_s = _make_k_grid(g_size, dx)

    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(k > 0, k**(-11./6.), 0.)

    Fx = _rand_c(KX.shape, rng) * W
    Fy = _rand_c(KX.shape, rng) * W
    Fz = _rand_c(KX.shape, rng) * W
    Fx, Fy, Fz = _solenoidal_project(Fx, Fy, Fz, KX, KY, KZ, k2_s)

    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    return _normalise_turbulence(dBx, dBy, dBz, M_A, B0_ref)


def _turbulence_random(X, Y, Z, Bx0, By0, Bz0, M_A, B0_ref, b_hat,
                       iaxis=2, corr_r=0.9, seed=None, **_):
    """
    LOS-correlated random rotation turbulence.
    Field is rotated by a correlated random angle along each LOS pencil.
    """
    g_size   = X.shape[0]
    rng      = np.random.default_rng(seed)
    turb_sig = M_A

    los_axes = [0, 1, 2]
    los_axes.remove(iaxis)
    i1, i2  = los_axes

    delta_theta = np.zeros((g_size, g_size, g_size))
    for i in range(g_size):
        for j in range(g_size):
            val = rng.normal(scale=turb_sig)
            for k_idx in range(g_size):
                idx        = [0, 0, 0]
                idx[iaxis] = k_idx
                idx[i1]    = i
                idx[i2]    = j
                delta_theta[tuple(idx)] = val
                val = (corr_r * val +
                       rng.normal(scale=turb_sig * np.sqrt(1 - corr_r**2)))

    if iaxis == 0:
        dBx = np.zeros_like(Bx0)
        dBy = By0*(np.cos(delta_theta) - 1) - Bz0*np.sin(delta_theta)
        dBz = By0* np.sin(delta_theta)      + Bz0*(np.cos(delta_theta) - 1)
    elif iaxis == 1:
        dBx = Bx0*(np.cos(delta_theta) - 1) - Bz0*np.sin(delta_theta)
        dBy = np.zeros_like(By0)
        dBz = Bx0* np.sin(delta_theta)      + Bz0*(np.cos(delta_theta) - 1)
    else:
        dBx = Bx0*(np.cos(delta_theta) - 1) - By0*np.sin(delta_theta)
        dBy = Bx0* np.sin(delta_theta)      + By0*(np.cos(delta_theta) - 1)
        dBz = np.zeros_like(Bz0)

    return dBx, dBy, dBz


_TURBULENCE = {
    'gs95'       : _turbulence_gs95,
    'kolmogorov' : _turbulence_kolmogorov,
    'random'     : _turbulence_random,
}


# ============================================================
# 5.  SHARED POST-PROCESSING
# ============================================================

def _enforce_divB_zero(Bx, By, Bz, x):
    """Project total field onto divergence-free subspace (Helmholtz)."""
    g_size = Bx.shape[0]
    dx     = x[1] - x[0]
    KX, KY, KZ, _, k2, _ = _make_k_grid(g_size, dx)

    Bx_k = np.fft.fftn(Bx)
    By_k = np.fft.fftn(By)
    Bz_k = np.fft.fftn(Bz)

    kdB  = KX*Bx_k + KY*By_k + KZ*Bz_k
    mask = k2 > 0
    Bx_k[mask] -= KX[mask] * kdB[mask] / k2[mask]
    By_k[mask] -= KY[mask] * kdB[mask] / k2[mask]
    Bz_k[mask] -= KZ[mask] * kdB[mask] / k2[mask]

    return (np.fft.ifftn(Bx_k).real,
            np.fft.ifftn(By_k).real,
            np.fft.ifftn(Bz_k).real)


def _stokes_and_polarization(Bx, By, Bz, iaxis):
    """
    Integrate Stokes Q, U along iaxis and return P, PA, cos2gamma.

    Stokes convention (matches original working per-geometry code):
      Q = cos2g * (B2² - B1²) / B_perp²
      U = cos2g * (2 B1 B2)   / B_perp²
    where B1, B2 are the two POS components.
    """
    if iaxis == 0:       # LOS = x  →  POS = (y, z)
        B_los  = Bx
        B1, B2 = By, Bz
    elif iaxis == 1:     # LOS = y  →  POS = (x, z)
        B_los  = By
        B1, B2 = Bx, Bz
    else:                # LOS = z  →  POS = (x, y)
        B_los  = Bz
        B1, B2 = Bx, By

    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los**2
    cos2g   = B_perp2 / (Btot2 + 1e-12)

    mask = B_perp2 > 1e-12
    Q    = np.zeros_like(B1)
    U    = np.zeros_like(B1)
    Q[mask] = cos2g[mask] * (B2[mask]**2 - B1[mask]**2) / B_perp2[mask]
    U[mask] = cos2g[mask] * (2*B1[mask]*B2[mask])        / B_perp2[mask]

    I_obs   = np.ones_like(B1).sum(axis=iaxis)
    Q_obs   = Q.sum(axis=iaxis)
    U_obs   = U.sum(axis=iaxis)

    P_obs   = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    return P_obs, phi_obs, cos2g.mean(axis=iaxis)


# ============================================================
# 6.  PUBLIC ENTRY POINT
# ============================================================

def simulate_field(geometry='helical', turbulence='gs95',
                   M_A=1.0, alpha=0.0, iaxis=2, g_size=64,
                   anisotropy_strength=1.0, seed=None, **kwargs):
    """
    Generate a synthetic polarization map.

    Parameters
    ----------
    geometry    : str    'uniform' | 'wavy' | 'helical' | 'hourglass'
    turbulence  : str    'gs95'    | 'kolmogorov' | 'random'
    M_A         : float  Alfvénic Mach number
    alpha       : float  Field angle [rad]  (geometry-dependent)
    iaxis       : int    LOS axis  0=x, 1=y, 2=z
    g_size      : int    Grid size (g_size³ cube)
    anisotropy_strength : float  GS95 cb-anisotropy suppression factor
    seed        : int | None  RNG seed

    Returns
    -------
    P_obs   : 2-D array  Polarization fraction
    phi_obs : 2-D array  Polarization angle [rad]
    cos2g   : 2-D array  Mean geometric depolarization factor
    """
    if geometry not in _GEOMETRIES:
        raise ValueError(f"Unknown geometry '{geometry}'. "
                         f"Choose from {list(_GEOMETRIES)}")
    if turbulence not in _TURBULENCE:
        raise ValueError(f"Unknown turbulence '{turbulence}'. "
                         f"Choose from {list(_TURBULENCE)}")

    x, X, Y, Z = _make_grid(g_size)

    # --- Mean field (geometry provides b_hat and B0_ref explicitly) ---
    Bx0, By0, Bz0, b_hat, B0_ref = _GEOMETRIES[geometry](
        X, Y, Z, alpha, iaxis=iaxis, **kwargs)

    # --- Turbulence ---
    dBx, dBy, dBz = _TURBULENCE[turbulence](
        X, Y, Z, Bx0, By0, Bz0, M_A, B0_ref, b_hat,
        anisotropy_strength=anisotropy_strength,
        iaxis=iaxis, seed=seed, **kwargs)

    # --- Total field ---
    Bx = Bx0 + dBx
    By = By0 + dBy
    Bz = Bz0 + dBz

    # --- Enforce ∇·B = 0 on total field ---
    Bx, By, Bz = _enforce_divB_zero(Bx, By, Bz, x)

    # --- Stokes / polarization ---
    return _stokes_and_polarization(Bx, By, Bz, iaxis)