# Functions:
# _gs95_turbulence
# hourglass_field_gs95
# compare_kolmo_vs_gs95
# hourglass_field_kolmo
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0, j1, erfc


# =============================================================================
#  GS95 / LAZARIAN 2006 SCALE-DEPENDENT ANISOTROPIC TURBULENCE
# =============================================================================
#
#  This module implements a synthetic turbulent magnetic field whose spectral
#  properties vary with scale according to the Alfvén Mach number M_A,
#  following the theoretical framework of:
#
#    Goldreich & Sridhar 1995, ApJ 438, 763          (GS95)
#    Cho & Lazarian 2002, PRL 88, 245001             (CL02)
#    Cho & Lazarian 2003, MNRAS 345, 325             (CL03)
#    Lazarian 2006, ApJ 645, L25                     (L06)
#
#  PHYSICAL REGIMES  (L06, Table 1 and surrounding text)
#  -------------------------------------------------------
#
#  SUB-ALFVÉNIC  (M_A < 1)
#  -----------------------
#  Two regimes separated by the transition scale l_trans ~ L * M_A^2
#  (L06, Eq. 1), where L is the turbulent injection scale:
#
#    WEAK regime  (l > l_trans, i.e. k_perp < k_trans):
#      - Cascade exclusively perpendicular to B
#      - Power spectrum: E(k_perp) ~ k_perp^{-2}           (L06)
#      - Parallel scale frozen at injection scale: l_par = L
#
#    STRONG (GS95) regime  (l < l_trans, i.e. k_perp > k_trans):
#      - Full GS95 critical balance cascade
#      - E(k_perp) ~ k_perp^{-5/3}                        (GS95; CL02; CL03)
#      - Critical balance: k_par ~ k_perp^{2/3}            (GS95)
#      - Anisotropy kernel: W ~ k_perp^{-11/6}
#                             * exp(-(k_par/k_perp^{2/3})^2)  (CL02)
#
#  SUPER-ALFVÉNIC  (M_A > 1)
#  -------------------------
#  Two regimes separated by the Alfvénic scale l_A ~ L * M_A^{-3}
#  (L06, Eq. 3):
#
#    HYDRODYNAMIC regime  (l > l_A, i.e. k < k_A):
#      - Magnetic field dynamically unimportant
#      - Isotropic Kolmogorov: E(k) ~ k^{-5/3}             (CL03; L06)
#
#    MHD (GS95) regime  (l < l_A, i.e. k > k_A):
#      - Full GS95 cascade; injection scale replaced by l_A  (L06)
#
#  TRANS-ALFVÉNIC  (M_A ~ 1)
#  -------------------------
#  GS95 applies throughout the inertial range.
#
# =============================================================================


def _gs95_turbulence(X, Y, Z, Bx0, By0, Bz0, M_A,
                     anisotropy_strength=1.0,
                     seed=None):
    """
    Generate a divergence-free turbulent magnetic field with scale-dependent
    spectral properties determined by M_A  (Lazarian 2006; Cho & Lazarian 2002).

    Parameters
    ----------
    X, Y, Z             : 3-D coordinate arrays (g_size^3)
    Bx0, By0, Bz0       : background field — same shape
    M_A                 : Alfven Mach number — controls amplitude AND regime
    anisotropy_strength : GS95 Gaussian kernel scale (1.0 = CL02 standard)
    seed                : optional RNG seed

    Returns
    -------
    dBx, dBy, dBz       : turbulent field, RMS = M_A * |B0|_mean
    """

    g_size = X.shape[0]
    rng    = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1.  Mean field direction  (global-frame approximation, CL02)
    # ------------------------------------------------------------------
    B0_mean = np.array([Bx0.mean(), By0.mean(), Bz0.mean()])
    B0_mag  = np.linalg.norm(B0_mean)
    b_hat   = B0_mean / B0_mag if B0_mag > 1e-12 else np.array([0., 0., 1.])

    # ------------------------------------------------------------------
    # 2.  Fourier k-grid
    # ------------------------------------------------------------------
    dx = X[1, 0, 0] - X[0, 0, 0]
    L  = g_size * dx        # physical box size = injection scale L

    kx = np.fft.fftfreq(g_size, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")

    k2   = KX**2 + KY**2 + KZ**2
    k    = np.sqrt(k2)
    k2_s = np.where(k2 > 0, k2, 1.0)

    # ------------------------------------------------------------------
    # 3.  k_par and k_perp relative to mean field  (CL02)
    # ------------------------------------------------------------------
    k_par  = KX * b_hat[0] + KY * b_hat[1] + KZ * b_hat[2]
    k_perp = np.sqrt(np.maximum(k2 - k_par**2, 0.0))

    # ------------------------------------------------------------------
    # 4.  Transition wavenumbers  (L06)
    #
    # Sub-Alfvenic:
    #   l_trans ~ L * M_A^2   (Lazarian 2006, p. L26, text preceding Eq. 5)
    #   k_trans = 2*pi / l_trans = (2*pi/L) * M_A^{-2}   [derived]
    #
    # Super-Alfvenic:
    #   l_A ~ L * M_A^{-3}    (Lazarian 2006, Eq. 2)
    #   k_A = 2*pi / l_A = (2*pi/L) * M_A^3              [derived]
    # ------------------------------------------------------------------
    k_box = 2.0 * np.pi / L

    if M_A < 1.0:
        k_trans = k_box / (M_A**2 + 1e-12)
    else:
        k_trans = k_box * M_A**3

    # ------------------------------------------------------------------
    # 5.  Spectral amplitude weights
    #
    #  W_hydro:  k^{-11/6}            isotropic Kolmogorov  (L06)
    #  W_weak:   k_perp^{-1}          weak perpendicular cascade, E~k_perp^{-2}  (L06)
    #  W_gs95:   k_perp^{-11/6}
    #            * exp(-(k_par/k_perp^{2/3})^2)     strong GS95  (CL02)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        W_hydro   = np.where(k > 0, k**(-11.0/6.0), 0.0)
        W_weak    = np.where(k_perp > 0, k_perp**(-1.0), W_hydro)
        k_perp_cb = np.where(k_perp > 0, k_perp**(2.0/3.0), 1.0)
        cb_arg    = np.where(k_perp > 0,
                             (k_par / k_perp_cb)**2 * anisotropy_strength,
                             0.0)
        W_perp    = np.where(k_perp > 0, k_perp**(-11.0/6.0), 0.0)
        W_gs95    = np.where(k_perp > 0, W_perp * np.exp(-cb_arg), W_hydro)

    # ------------------------------------------------------------------
    # 6.  Scale-dependent blend  (L06)
    #
    #  Sigmoid transition over 1 decade in log k, centred on k_trans.
    #  This avoids a hard spectral break while faithfully encoding
    #  the physics of the weak-to-strong or hydro-to-MHD transition.
    # ------------------------------------------------------------------
    sigmoid_width = np.log(10.0)

    if M_A < 1.0:
        # Sub-Alfvenic: weak at large scales, GS95 at small scales
        # Transition variable: k_perp  (L06: perpendicular cascade)
        k_perp_safe = np.where(k_perp > 0, k_perp, k_trans)
        log_ratio   = np.log(k_perp_safe / k_trans) / sigmoid_width
        f_gs95      = 1.0 / (1.0 + np.exp(-log_ratio))
        W           = (1.0 - f_gs95) * W_weak + f_gs95 * W_gs95
    else:
        # Super-Alfvenic: hydrodynamic at large scales, GS95 at small scales
        # Transition variable: isotropic k  (L06: hydrodynamic -> MHD)
        k_safe    = np.where(k > 0, k, k_trans)
        log_ratio = np.log(k_safe / k_trans) / sigmoid_width
        f_gs95    = 1.0 / (1.0 + np.exp(-log_ratio))
        W         = (1.0 - f_gs95) * W_hydro + f_gs95 * W_gs95

    W = np.where(k2 > 0, W, 0.0)

    # ------------------------------------------------------------------
    # 7.  Random amplitudes + solenoidal projection
    # ------------------------------------------------------------------
    def rand_c(shape):
        return (rng.standard_normal(shape) +
                1j * rng.standard_normal(shape)) / np.sqrt(2)

    Fx = rand_c(KX.shape) * W
    Fy = rand_c(KX.shape) * W
    Fz = rand_c(KX.shape) * W

    k_dot_F = KX*Fx + KY*Fy + KZ*Fz
    Fx -= KX * k_dot_F / k2_s
    Fy -= KY * k_dot_F / k2_s
    Fz -= KZ * k_dot_F / k2_s

    # ------------------------------------------------------------------
    # 8.  Real space + normalise RMS to M_A * |B0|
    # ------------------------------------------------------------------
    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    rms = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    if rms > 0:
        scale = M_A * B0_mag / rms
        dBx *= scale; dBy *= scale; dBz *= scale

    return dBx, dBy, dBz


# =============================================================================
#  MAIN HOURGLASS FUNCTION
# =============================================================================

def hourglass_field_gs95(M_A, g_size, iaxis, plotting,
                         anisotropy_strength=1.0,
                         seed=None):
    """
    Hourglass magnetic field + scale-dependent GS95 turbulence.

    Parameters
    ----------
    M_A                 : Alfven Mach number (amplitude + spectral regime)
    g_size              : grid size
    iaxis               : LOS axis (0=x, 1=y, 2=z)
    plotting            : bool
    anisotropy_strength : GS95 kernel scale (1.0 = CL02 standard)
    seed                : RNG seed

    Returns
    -------
    P_obs, phi_obs, cos2g
    """

    R_outer = 5.0;  h = 1.0;  B0 = 1.0;  Bm = 1.0
    betam   = Bm / B0;  N_terms = 3;  I0 = 1.0

    x = np.linspace(-0.25, 0.25, g_size)
    y = np.linspace(-0.25, 0.25, g_size)
    z = np.linspace(-0.25, 0.25, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R      = np.sqrt(X**2 + Y**2)
    R_safe = np.where(R == 0, 1e-12, R)

    # Hourglass Bessel series
    roots   = jn_zeros(1, N_terms)
    weights = 2 / (roots * j0(roots)**2)
    eta     = h / R_outer
    Br0 = np.zeros_like(R);  Bz0 = np.zeros_like(R)

    for n in range(N_terms):
        lam = roots[n];  w = weights[n]
        arg_neg = lam*eta/2 - Z/eta;  arg_pos = lam*eta/2 + Z/eta
        er_neg  = erfc(arg_neg);      er_pos  = erfc(arg_pos)
        exp_neg = np.exp(-lam*Z);     exp_pos = np.exp(lam*Z)
        Br0 += w * j1(lam*R) * betam * (er_neg*exp_neg - er_pos*exp_pos)
        Bz0 += w * j0(lam*R) * betam * (er_pos*exp_pos + er_neg*exp_neg)

    Bz0 += 1.0
    Bx0 = Br0 * (X / R_safe)
    By0 = Br0 * (Y / R_safe)

    # Turbulence
    dBx, dBy, dBz = _gs95_turbulence(
        X, Y, Z, Bx0, By0, Bz0, M_A,
        anisotropy_strength=anisotropy_strength, seed=seed)

    # Total field + divergence cleaning
    Bx = Bx0 + dBx;  By = By0 + dBy;  Bz = Bz0 + dBz

    kx = np.fft.fftfreq(g_size, d=x[1]-x[0]) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2

    Bx_k = np.fft.fftn(Bx);  By_k = np.fft.fftn(By);  Bz_k = np.fft.fftn(Bz)
    k_dot_B = KX*Bx_k + KY*By_k + KZ*Bz_k
    mask = k2 > 0
    Bx_k[mask] -= KX[mask]*k_dot_B[mask]/k2[mask]
    By_k[mask] -= KY[mask]*k_dot_B[mask]/k2[mask]
    Bz_k[mask] -= KZ[mask]*k_dot_B[mask]/k2[mask]
    Bx = np.fft.ifftn(Bx_k).real
    By = np.fft.ifftn(By_k).real
    Bz = np.fft.ifftn(Bz_k).real

    # Stokes
    if iaxis == 0:   B_los = Bx;  B1, B2 = By, Bz
    elif iaxis == 1: B_los = By;  B1, B2 = Bx, Bz
    else:            B_los = Bz;  B1, B2 = Bx, By

    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los**2
    cos2g   = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * cos2g

    mask_q = B_perp2 > 1e-12
    Q = np.zeros_like(B1);  U = np.zeros_like(B1)
    Q[mask_q] = I_local[mask_q]*(B2[mask_q]**2-B1[mask_q]**2)/B_perp2[mask_q]
    U[mask_q] = I_local[mask_q]*(2*B1[mask_q]*B2[mask_q])/B_perp2[mask_q]

    I_obs   = I_local.sum(axis=iaxis)
    Q_obs   = Q.sum(axis=iaxis)
    U_obs   = U.sum(axis=iaxis)
    P_obs   = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    # Diagnostic
    B0_mean = np.array([Bx0.mean(), By0.mean(), Bz0.mean()])
    B0_mag  = np.linalg.norm(B0_mean)
    b_hat   = B0_mean / B0_mag if B0_mag > 1e-12 else np.array([0.,0.,1.])
    k_par_d  = KX*b_hat[0] + KY*b_hat[1] + KZ*b_hat[2]
    k_safe_d = np.where(k2 > 0, np.sqrt(k2), 1.0)
    cos_theta = np.abs(k_par_d) / k_safe_d

    dBx_k = np.fft.fftn(dBx);  dBy_k = np.fft.fftn(dBy);  dBz_k = np.fft.fftn(dBz)
    power   = np.abs(dBx_k)**2 + np.abs(dBy_k)**2 + np.abs(dBz_k)**2
    power_n = power / power.max()
    par_mask  = (cos_theta > 0.7) & (k_safe_d > 0)
    perp_mask = (cos_theta < 0.3) & (k_safe_d > 0)
    aniso_ratio = (np.median(power_n[par_mask]) / np.median(power_n[perp_mask])
                   if par_mask.any() and perp_mask.any() else np.nan)

    L_box = g_size * (x[1]-x[0])
    k_box_val = 2*np.pi / L_box
    if M_A < 1.0:
        k_trans_val = k_box_val / M_A**2
        regime = "weak→GS95 (sub-Alfvenic, L06)"
    else:
        k_trans_val = k_box_val * M_A**3
        regime = "hydro→GS95 (super-Alfvenic, L06)"

    print(f"[GS95] M_A={M_A:.2f}  regime={regime}  "
          f"k_trans/k_box={k_trans_val/k_box_val:.2f}  "
          f"aniso_ratio={aniso_ratio:.4f}  (<1 = par suppressed)")

    if plotting:
        skip = max(1, g_size//10);  mid = g_size//2
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.quiver(X[::skip,::skip,::skip], Y[::skip,::skip,::skip],
                  Z[::skip,::skip,::skip], Bx[::skip,::skip,::skip],
                  By[::skip,::skip,::skip], Bz[::skip,::skip,::skip],
                  length=0.05, normalize=True, linewidth=1.5, arrow_length_ratio=0)
        ax.set_title(f"Hourglass + GS95  M_A={M_A}");  plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].quiver(Y[mid,:,:],Z[mid,:,:],By[mid,:,:],Bz[mid,:,:],
                       headaxislength=0,headlength=0,headwidth=1,pivot='middle')
        axes[0].set_title("LOS X")
        axes[1].quiver(X[:,mid,:],Z[:,mid,:],Bx[:,mid,:],Bz[:,mid,:],
                       headaxislength=0,headlength=0,headwidth=1,pivot='middle')
        axes[1].set_title("LOS Y")
        axes[2].quiver(X[:,:,mid],Y[:,:,mid],Bx[:,:,mid],By[:,:,mid],
                       headaxislength=0,headlength=0,headwidth=1,pivot='middle')
        axes[2].set_title("LOS Z")
        plt.tight_layout();  plt.show()

    return P_obs, phi_obs, cos2g.mean(axis=iaxis)



def hourglass_field_kolmo(M_A, g_size, iaxis, plotting):

    # Parameters
    R_outer = 5.0
    h = 1.0
    B0 = 1.0
    Bm = 1.0
    betam = Bm / B0
    N_terms = 3

    # Stokes
    I0 = 1.0

    # Turbulence
    turb_sigma = M_A
    corr_r = 0.9

    # DCF Parameters
    rho = 1.0              # mass density (set to 1 for toy model)
    sigma_v = 1.0          # LOS velocity dispersion (required by DCF formula)
    cf_prefactor = np.sqrt(4 * np.pi * rho) * sigma_v

    # =========================
    # CARTESIAN GRID
    # =========================
    x = np.linspace(-0.25, 0.25, g_size)
    y = np.linspace(-0.25, 0.25, g_size)
    z = np.linspace(-0.25, 0.25, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    R = np.sqrt(X**2 + Y**2)
    R_safe = np.where(R == 0, 1e-12, R)

    # =========================
    # BESSEL SERIES
    # =========================
    roots = jn_zeros(1, N_terms)
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

    # =========================
    # CYLINDRICAL → CARTESIAN
    # =========================
    Bx0 = Br0 * (X/ R_safe)
    By0 = Br0 * (Y/ R_safe)

    # Bx0[R == 0] = 0
    # By0[R == 0] = 0

    # =========================
    # 3D KOLMOGOROV TURBULENCE (DIVERGENCE-FREE)
    # =========================

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    kx = np.fft.fftfreq(g_size, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(g_size, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(g_size, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    k2 = KX**2 + KY**2 + KZ**2
    k = np.sqrt(k2)

    rng = np.random.default_rng()

    def rand_complex(shape):
        return (rng.standard_normal(shape) +
                1j * rng.standard_normal(shape)) / np.sqrt(2)

    Fx = rand_complex(X.shape)
    Fy = rand_complex(X.shape)
    Fz = rand_complex(X.shape)

    # Kolmogorov amplitude scaling
    with np.errstate(divide="ignore", invalid="ignore"):
        kolmogorov_weight = np.where(k > 0, k ** (-11.0 / 6.0), 0.0)

    Fx *= kolmogorov_weight
    Fy *= kolmogorov_weight
    Fz *= kolmogorov_weight

    # Project out longitudinal component (∇·δB = 0)
    k2_safe = np.where(k2 > 0, k2, 1.0)

    k_dot_F = KX * Fx + KY * Fy + KZ * Fz

    Fx -= KX * k_dot_F / k2_safe
    Fy -= KY * k_dot_F / k2_safe
    Fz -= KZ * k_dot_F / k2_safe

    # Back to real space
    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    # Normalise RMS turbulence amplitude to M_A
    rms = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    if rms > 0:
        dBx *= M_A / rms
        dBy *= M_A / rms
        dBz *= M_A / rms

    # Add turbulence to large-scale field
    Bx = Bx0 + dBx
    By = By0 + dBy
    Bz = Bz0 + dBz

    # =========================
    # ENFORCE ∇·B = 0 ON TOTAL FIELD
    # =========================
    Bx_k = np.fft.fftn(Bx)
    By_k = np.fft.fftn(By)
    Bz_k = np.fft.fftn(Bz)

    k_dot_B = KX * Bx_k + KY * By_k + KZ * Bz_k

    mask = k2 > 0
    Bx_k[mask] -= KX[mask] * k_dot_B[mask] / k2[mask]
    By_k[mask] -= KY[mask] * k_dot_B[mask] / k2[mask]
    Bz_k[mask] -= KZ[mask] * k_dot_B[mask] / k2[mask]

    Bx = np.fft.ifftn(Bx_k).real
    By = np.fft.ifftn(By_k).real
    Bz = np.fft.ifftn(Bz_k).real

    # =========================
    # STOKES
    # =========================

    # Define LOS and POS components
    if iaxis == 0:        # LOS = x → POS = (y,z)
        B_los = Bx
        B1, B2 = By, Bz

    elif iaxis == 1:      # LOS = y → POS = (x,z)
        B_los = By
        B1, B2 = Bx, Bz

    else:                 # LOS = z → POS = (x,y)
        B_los = Bz
        B1, B2 = Bx, By

    B_los2  = B_los**2
    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los2

    # Geometrical depolarization
    cos2g = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * cos2g

    # Safe division mask
    mask = B_perp2 > 1e-12

    Q = np.zeros_like(B1)
    U = np.zeros_like(B1)

    # Standard dust polarization definition
    Q[mask] = I_local[mask] * (B2[mask]**2 - B1[mask]**2) / B_perp2[mask]
    U[mask] = I_local[mask] * (2 * B1[mask] * B2[mask]) / B_perp2[mask]

    # LOS integration
    I_obs = I_local.sum(axis=iaxis)
    Q_obs = Q.sum(axis=iaxis)
    U_obs = U.sum(axis=iaxis)

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    # =========================
    # PLOTTING
    # =========================
    if plotting:

        skip = g_size // 10

        # 3D visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(
            X[::skip, ::skip, ::skip],
            Y[::skip, ::skip, ::skip],
            Z[::skip, ::skip, ::skip],
            Bx[::skip, ::skip, ::skip],
            By[::skip, ::skip, ::skip],
            Bz[::skip, ::skip, ::skip],
            length=0.05, normalize=True,linewidth=1.5, arrow_length_ratio=0)
        ax.set_title("3D Hourglass + Turbulence")
        plt.show()

        mid = g_size // 2

        fig, axes = plt.subplots(1,3,figsize=(12,4))

        # LOS = x → POS = (y,z)
        axes[0].quiver(
            Y[mid, :, :],
            Z[mid, :, :],
            By[mid, :, :],
            Bz[mid, :, :],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("Z")
        axes[0].set_title("LOS X")

        # LOS = y → POS = (x,z)
        axes[1].quiver(
            X[:, mid, :],
            Z[:, mid, :],
            Bx[:, mid, :],
            Bz[:, mid, :],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[1].set_title("LOS Y")

        # LOS = z → POS = (x,y)
        axes[2].quiver(
            X[:, :, mid],
            Y[:, :, mid],
            Bx[:, :, mid],
            By[:, :, mid],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_title("LOS Z")

        plt.tight_layout()
        plt.show()


    return P_obs, phi_obs, cos2g.mean(axis=iaxis)


# =============================================================================
#  QUICK COMPARISON HELPER
# =============================================================================

def compare_kolmo_vs_gs95(M_A=0.5, g_size=64, iaxis=2):
    """
    Side-by-side polarization maps: Kolmogorov vs GS95 for the same M_A.
    Requires hourglass_field_kolmo to be importable from the same directory.
    """

    print("Running Kolmogorov model...")
    P_k, phi_k, cg_k = hourglass_field_kolmo(M_A, g_size, iaxis, plotting=False)

    print("Running GS95 model...")
    P_g, phi_g, cg_g = hourglass_field_gs95(M_A, g_size, iaxis, plotting=False,
                                             anisotropy_strength=1.0, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"Kolmogorov vs GS95  |  M_A = {M_A}  |  LOS axis = {iaxis}")

    im0 = axes[0, 0].imshow(P_k, origin='lower', cmap='inferno')
    axes[0, 0].set_title("Kolmogorov — polarization fraction P")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(P_g, origin='lower', cmap='inferno')
    axes[0, 1].set_title("GS95 — polarization fraction P")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(phi_k, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
    axes[1, 0].set_title("Kolmogorov — PA φ [rad]")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(phi_g, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
    axes[1, 1].set_title("GS95 — PA φ [rad]")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    print(f"\n  Mean P  :  Kolmogorov = {P_k.mean():.4f}   GS95 = {P_g.mean():.4f}")
    print(f"  Std  P  :  Kolmogorov = {P_k.std():.4f}    GS95 = {P_g.std():.4f}")
    print(f"  Mean cg :  Kolmogorov = {cg_k.mean():.4f}   GS95 = {cg_g.mean():.4f}")



# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    for MA in [0.2, 0.5, 1.0, 2.0]:
        P, phi, cg = hourglass_field_gs95(
            M_A=MA, g_size=64, iaxis=2, plotting=False,
            anisotropy_strength=1.0, seed=42)
        print(f"  Mean P = {P.mean():.4f}   cos2g = {cg.mean():.4f}\n")
        