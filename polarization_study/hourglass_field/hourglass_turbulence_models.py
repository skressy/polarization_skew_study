# Functions:
# _gs95_turbulence
# hourglass_field_gs95
# hourglass_field_kolmo

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0, j1, erfc


def _gs95_turbulence(X, Y, Z, Bx0, By0, Bz0, M_A,
                     anisotropy_strength=1.0,
                     seed=None):

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
    dx = (X.max() - X.min()) / (X.shape[0] - 1)
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
    k_box = 2.0 * np.pi / L

    # ------------------------------------------------------------------
    # 5.  Spectral amplitude weights  (spherical shell, see header)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):

        # W_hydro: k^{-11/6}  (Kolmogorov 1941)
        W_hydro   = np.where(k > 0, k**(-11.0/6.0), 0.0)

        # W_weak: k_⊥^{-2}  (LV99 App. A Eq. A2, spherical shell)
        W_weak    = np.where(k_perp > 0, k_perp**(-2.0), W_hydro)

        # W_gs95: k_⊥^{-11/6} * exp(-(k_∥/k_⊥^{2/3})²)
        k_perp_cb = np.where(k_perp > 0, k_perp**(2.0/3.0), 1.0)
        cb_arg    = np.where(k_perp > 0,
                             (k_par / k_perp_cb)**2 * anisotropy_strength,
                             0.0)
        W_perp    = np.where(k_perp > 0, k_perp**(-11.0/6.0), 0.0)
        W_gs95    = np.where(k_perp > 0, W_perp * np.exp(-cb_arg), W_hydro)

    # ------------------------------------------------------------------
    # 6.  Hard regime assignment per L06
    # ------------------------------------------------------------------

    if M_A <= 1.0:
        # Sub-Alfvenic: weak at large scales, GS95 at small scales
        # Transition variable: k_perp  (perpendicular cascade, L06 p.L26)
        k_trans = k_box / (M_A**2)
        W       = np.where(k_perp < k_trans, W_weak, W_gs95)

    elif M_A > 1.0:
        # Super-Alfvenic: hydrodynamic at large scales, GS95 at small scales
        # Transition variable: isotropic k  (L06, Eq. 2)
        k_trans = k_box * M_A**3
        W       = np.where(k < k_trans, W_hydro, W_gs95)

    # else:
    #     # Trans-Alfvenic: GS95 throughout
    #     k_trans = k_box
    #     W       = W_gs95

    W = np.where(k2 > 0, W, 0.0)

    # ------------------------------------------------------------------
    # 7.  Random complex Fourier amplitudes + solenoidal projection (∇·δB=0)
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
    # 8.  Transform to real space + normalise RMS to M_A * |B0|
    # ------------------------------------------------------------------
    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    rms = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    if rms > 0:
        scale = M_A * B0_mag / rms
        dBx  *= scale
        dBy  *= scale
        dBz  *= scale

    return dBx, dBy, dBz


def hourglass_field_gs95(M_A, g_size, iaxis, plotting,
                         alfven_fraction=0.6,
                         slow_fraction=0.2,
                         fast_fraction=0.2,
                         anisotropy_strength=1.0,
                         seed=None):

    # -------------------------------------------------------------------------
    # PARAMETERS  (unchanged from Kolmogorov version)
    # -------------------------------------------------------------------------
    R_outer  = 5.0
    h        = 1.0
    B0       = 1.0
    Bm       = 1.0
    betam    = Bm / B0
    N_terms  = 3
    I0       = 1.0

    rho      = 1.0
    sigma_v  = 1.0
    cf_prefactor = np.sqrt(4 * np.pi * rho) * sigma_v

    # -------------------------------------------------------------------------
    # CARTESIAN GRID
    # -------------------------------------------------------------------------
    x = np.linspace(-0.25, 0.25, g_size)
    y = np.linspace(-0.25, 0.25, g_size)
    z = np.linspace(-0.25, 0.25, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    R      = np.sqrt(X**2 + Y**2)
    R_safe = np.where(R == 0, 1e-12, R)

    # -------------------------------------------------------------------------
    # BESSEL SERIES HOURGLASS FIELD  (unchanged)
    # -------------------------------------------------------------------------
    roots   = jn_zeros(1, N_terms)
    weights = 2 / (roots * j0(roots)**2)
    eta     = h / R_outer

    Br0 = np.zeros_like(R)
    Bz0 = np.zeros_like(R)

    for n in range(N_terms):
        lam   = roots[n]
        w     = weights[n]

        J0_eval = j0(lam * R)
        J1_eval = j1(lam * R)

        arg_neg = lam * eta / 2 - Z / eta
        arg_pos = lam * eta / 2 + Z / eta

        er_neg  = erfc(arg_neg)
        er_pos  = erfc(arg_pos)

        exp_neg = np.exp(-lam * Z)
        exp_pos = np.exp( lam * Z)

        brackets0 = er_pos * exp_pos + er_neg * exp_neg
        brackets1 = er_neg * exp_neg - er_pos * exp_pos

        Br0 += w * J1_eval * betam * brackets1
        Bz0 += w * J0_eval * betam * brackets0

    Bz0 += 1.0

    Bx0 = Br0 * (X / R_safe)
    By0 = Br0 * (Y / R_safe)

    # -------------------------------------------------------------------------
    # GS95 ANISOTROPIC TURBULENCE  (replaces Kolmogorov block)
    # -------------------------------------------------------------------------
    dBx, dBy, dBz = _gs95_turbulence(
        X, Y, Z, Bx0, By0, Bz0, M_A,
        alfven_fraction=alfven_fraction,
        slow_fraction=slow_fraction,
        fast_fraction=fast_fraction,
        anisotropy_strength=anisotropy_strength,
        seed=seed
    )

    # -------------------------------------------------------------------------
    # ADD TURBULENCE TO BACKGROUND FIELD
    # -------------------------------------------------------------------------
    Bx = Bx0 + dBx
    By = By0 + dBy
    Bz = Bz0 + dBz

    # -------------------------------------------------------------------------
    # ENFORCE ∇·B = 0 ON TOTAL FIELD
    # -------------------------------------------------------------------------
    kx = np.fft.fftfreq(g_size, d=x[1]-x[0]) * 2 * np.pi
    ky = np.fft.fftfreq(g_size, d=y[1]-y[0]) * 2 * np.pi
    kz = np.fft.fftfreq(g_size, d=z[1]-z[0]) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k    = np.sqrt(k2)
    
    Bx_k = np.fft.fftn(Bx)
    By_k = np.fft.fftn(By)
    Bz_k = np.fft.fftn(Bz)

    k_dot_B = KX * Bx_k + KY * By_k + KZ * Bz_k
    mask    = k2 > 0
    Bx_k[mask] -= KX[mask] * k_dot_B[mask] / k2[mask]
    By_k[mask] -= KY[mask] * k_dot_B[mask] / k2[mask]
    Bz_k[mask] -= KZ[mask] * k_dot_B[mask] / k2[mask]

    Bx = np.fft.ifftn(Bx_k).real
    By = np.fft.ifftn(By_k).real
    Bz = np.fft.ifftn(Bz_k).real

    # -------------------------------------------------------------------------
    # STOKES PARAMETERS  (unchanged)
    # -------------------------------------------------------------------------
    if iaxis == 0:
        B_los     = Bx
        B1, B2    = By, Bz
    elif iaxis == 1:
        B_los     = By
        B1, B2    = Bx, Bz
    else:
        B_los     = Bz
        B1, B2    = Bx, By

    B_los2  = B_los**2
    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los2

    cos2g   = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * cos2g

    mask_q  = B_perp2 > 1e-12
    Q = np.zeros_like(B1)
    U = np.zeros_like(B1)
    Q[mask_q] = I_local[mask_q] * (B2[mask_q]**2 - B1[mask_q]**2) / B_perp2[mask_q]
    U[mask_q] = I_local[mask_q] * (2 * B1[mask_q] * B2[mask_q])   / B_perp2[mask_q]

    I_obs   = I_local.sum(axis=iaxis)
    Q_obs   = Q.sum(axis=iaxis)
    U_obs   = U.sum(axis=iaxis)

    P_obs   = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    # -------------------------------------------------------------------------
    # DIAGNOSTIC: anisotropy ratio  (should differ from Kolmogorov run)
    # -------------------------------------------------------------------------
    # Measure power in k_∥ vs k_⊥ slabs to confirm anisotropy was injected
    B0_mean = np.array([Bx0.mean(), By0.mean(), Bz0.mean()])
    B0_mag  = np.linalg.norm(B0_mean)
    b_hat   = B0_mean / B0_mag if B0_mag > 1e-12 else np.array([0., 0., 1.])

    k_par  = KX * b_hat[0] + KY * b_hat[1] + KZ * b_hat[2]
    k_perp = np.sqrt(np.maximum(k2 - k_par**2, 0.0))

    dBx_k = np.fft.fftn(dBx)
    dBy_k = np.fft.fftn(dBy)
    dBz_k = np.fft.fftn(dBz)
    power  = (np.abs(dBx_k)**2 + np.abs(dBy_k)**2 + np.abs(dBz_k)**2)

    # Correlation length ratio: Λ_∥ / Λ_⊥  (>1 means elongated along B → GS95)
    par_scale  = np.sum(power / (np.abs(k_par)  + 1e-12))
    perp_scale = np.sum(power / (np.abs(k_perp) + 1e-12))
    aniso_ratio = par_scale / perp_scale
    print(f"[GS95] Anisotropy ratio Λ∥/Λ⊥ = {aniso_ratio:.3f}  "
          f"(>1 → field-aligned eddies as expected)")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    if plotting:
        skip = max(1, g_size // 10)

        # 3-D overview
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.quiver(
            X[::skip, ::skip, ::skip],
            Y[::skip, ::skip, ::skip],
            Z[::skip, ::skip, ::skip],
            Bx[::skip, ::skip, ::skip],
            By[::skip, ::skip, ::skip],
            Bz[::skip, ::skip, ::skip],
            length=0.05, normalize=True, linewidth=1.5, arrow_length_ratio=0)
        ax.set_title("3D Hourglass + GS95 Turbulence")
        plt.show()

        mid = g_size // 2

        # 2-D field slices
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].quiver(Y[mid, :, :], Z[mid, :, :],
                       By[mid, :, :], Bz[mid, :, :],
                       headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[0].set_xlabel("Y"); axes[0].set_ylabel("Z"); axes[0].set_title("LOS X")

        axes[1].quiver(X[:, mid, :], Z[:, mid, :],
                       Bx[:, mid, :], Bz[:, mid, :],
                       headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[1].set_xlabel("X"); axes[1].set_ylabel("Z"); axes[1].set_title("LOS Y")

        axes[2].quiver(X[:, :, mid], Y[:, :, mid],
                       Bx[:, :, mid], By[:, :, mid],
                       headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[2].set_xlabel("X"); axes[2].set_ylabel("Y"); axes[2].set_title("LOS Z")

        plt.tight_layout()
        plt.show()

        # Power spectra comparison
        k_bins   = np.linspace(0, k.max(), g_size // 2)
        k_mid    = 0.5 * (k_bins[:-1] + k_bins[1:])

        E_perp   = np.zeros(len(k_mid))
        E_par    = np.zeros(len(k_mid))

        for i, (k0, k1) in enumerate(zip(k_bins[:-1], k_bins[1:])):
            m_perp = (k_perp >= k0) & (k_perp < k1)
            m_par  = (np.abs(k_par) >= k0) & (np.abs(k_par) < k1)
            E_perp[i] = power[m_perp].sum() if m_perp.any() else 0
            E_par[i]  = power[m_par ].sum() if m_par.any()  else 0

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(k_mid[E_perp > 0], E_perp[E_perp > 0],
                  label=r'$E(k_\perp)$ — GS95 slope $-5/3$', color='steelblue')
        ax.loglog(k_mid[E_par > 0],  E_par[E_par > 0],
                  label=r'$E(k_\parallel)$', color='tomato', linestyle='--')

        # Reference slopes
        k_ref  = k_mid[k_mid > k_mid[1]]
        ax.loglog(k_ref, 3e6 * k_ref**(-5/3), 'k:',  alpha=0.5, label=r'$k^{-5/3}$')
        ax.loglog(k_ref, 3e6 * k_ref**(-3/2), 'k--', alpha=0.5, label=r'$k^{-3/2}$')
        ax.set_xlabel(r'$k$'); ax.set_ylabel(r'Power')
        ax.set_title('GS95 Power Spectra: perpendicular vs parallel')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

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


def hourglass_field_uniform(M_A, g_size, iaxis, plotting):

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
    # ADD LOS TURBULENCE
    # =========================
    delta_theta = np.zeros_like(Bx0)

    axes = [0, 1, 2]
    axes.remove(iaxis)
    i1, i2 = axes

    for i in range(g_size):
        for j in range(g_size):
            val = np.random.normal(scale=turb_sigma)
            for k in range(g_size):
                idx = [0, 0, 0]
                idx[iaxis] = k
                idx[i1] = i
                idx[i2] = j
                delta_theta[tuple(idx)] = val
                val = (corr_r * val + np.random.normal(scale=turb_sigma * np.sqrt(1 - corr_r**2)))

    # add turbulence to appropriate POS components
    if iaxis == 0:   # LOS = x
        By_new = By0*np.cos(delta_theta) - Bz0*np.sin(delta_theta)
        Bz_new = By0*np.sin(delta_theta) + Bz0*np.cos(delta_theta)
        Bx_new = Bx0
    elif iaxis == 1: # LOS = y
        Bx_new = Bx0*np.cos(delta_theta) - Bz0*np.sin(delta_theta)
        Bz_new = Bx0*np.sin(delta_theta) + Bz0*np.cos(delta_theta)
        By_new = By0
    else:            # LOS = z
        Bx_new = Bx0*np.cos(delta_theta) - By0*np.sin(delta_theta)
        By_new = Bx0*np.sin(delta_theta) + By0*np.cos(delta_theta)
        Bz_new = Bz0

    Bx, By, Bz = Bx_new, By_new, Bz_new

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

    # Local intensity weighted by geometry
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

