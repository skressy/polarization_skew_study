# Functions:
# _gs95_turbulence
# helical_field_gs95
# helical_field_kolmo

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0, j1, erfc


def _gs95_turbulence(X, Y, Z, Bx0, By0, Bz0, M_A, anisotropy_strength=1.0, seed=None):

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

    # In _gs95_turbulence, after computing W:
    W_rms = np.sqrt(np.mean(W**2))
    if W_rms > 0:
        W /= W_rms

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
    B0_rms = np.sqrt(np.mean(Bx0**2 + By0**2 + Bz0**2))
    if rms > 0:
        dBx *= M_A * B0_rms / rms
        dBy *= M_A * B0_rms / rms
        dBz *= M_A * B0_rms / rms
    
    # print(f"  B0_rms={B0_rms:.4f}  dB_rms_before={rms:.4f}  dB_rms_after={np.sqrt(np.mean(dBx**2+dBy**2+dBz**2)):.4f}")


    return dBx, dBy, dBz

def helical_field_gs95(M_A, alpha, iaxis, g_size, anisotropy_strength=1.0, seed=None, plotting=False):

    I0 = 1.0

    x = np.linspace(-1, 1, g_size)
    y = np.linspace(-1, 1, g_size)
    z = np.linspace(-1, 1, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # avoid singularity at r=0
    r = np.sqrt(X**2 + Y**2) + 1e-8
    # r = np.clip(np.sqrt(X**2 + Y**2), 0.00000005, None)

    # base toroidal field (unrotated in z)
    Bx0 = (-Y * np.cos(alpha)) / r
    By0 = ( X * np.cos(alpha)) / r
    Bz0 = np.full_like(X, np.sin(alpha))

    # ------------------------------------------------------------------
    # GS95 TURBULENCE
    # Scale-dependent anisotropic spectrum following:
    #   Goldreich & Sridhar 1995, ApJ 438, 763
    #   Cho & Lazarian 2002, PRL 88, 245001
    #   Lazarian 2006, ApJ 645, L25
    # ------------------------------------------------------------------
    dBx, dBy, dBz = _gs95_turbulence(X, Y, Z, Bx0, By0, Bz0, M_A, anisotropy_strength=anisotropy_strength, seed=seed)

    # ------------------------------------------------------------------
    # TOTAL FIELD
    # ------------------------------------------------------------------
    Bx = Bx0 + dBx
    By = By0 + dBy
    Bz = Bz0 + dBz

    # ------------------------------------------------------------------
    # ENFORCE ∇·B = 0 ON TOTAL FIELD
    # ------------------------------------------------------------------
    kx = np.fft.fftfreq(g_size, d=x[1]-x[0]) * 2 * np.pi
    ky = np.fft.fftfreq(g_size, d=y[1]-y[0]) * 2 * np.pi
    kz = np.fft.fftfreq(g_size, d=z[1]-z[0]) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2

    Bx_k = np.fft.fftn(Bx)
    By_k = np.fft.fftn(By)
    Bz_k = np.fft.fftn(Bz)

    k_dot_B = KX*Bx_k + KY*By_k + KZ*Bz_k
    mask    = k2 > 0
    Bx_k[mask] -= KX[mask] * k_dot_B[mask] / k2[mask]
    By_k[mask] -= KY[mask] * k_dot_B[mask] / k2[mask]
    Bz_k[mask] -= KZ[mask] * k_dot_B[mask] / k2[mask]

    Bx = np.fft.ifftn(Bx_k).real
    By = np.fft.ifftn(By_k).real
    Bz = np.fft.ifftn(Bz_k).real


    # =========================
    # Calculate Stokes for appropriate POS components
    # =========================
    # Define LOS and POS components
    if iaxis == 0:        # LOS = x → POS = (y, z)
        B_los = Bx
        B1, B2 = By, Bz
        # Q = (Bz² - By²), U = 2 By Bz  ← check convention matches iaxis=1,2

    elif iaxis == 1:      # LOS = y → POS = (x, z)
        B_los = By
        B1, B2 = Bx, Bz

    else:                 # LOS = z → POS = (x, y)
        B_los = Bz
        B1, B2 = Bx, By

    B_los2  = B_los**2
    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los2

    # Geometrical depolarization
    cos2g = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * np.ones_like(Btot2)

    # Safe division mask
    mask = B_perp2 > 1e-12

    Q = np.zeros_like(B1)
    U = np.zeros_like(B1)

    Q[mask] = cos2g[mask] * (B1[mask]**2 - B2[mask]**2) / B_perp2[mask]
    U[mask] = cos2g[mask] * (2 * B1[mask] * B2[mask])   / B_perp2[mask]

    # LOS integration
    I_obs = I_local.sum(axis=iaxis)
    Q_obs = Q.sum(axis=iaxis)
    U_obs = U.sum(axis=iaxis)

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

        # ------------------------------------------------------------------
    if plotting:

        P_k, phi_k, cg_k = helical_field_kolmo(M_A, alpha, iaxis, g_size, plotting)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle(f"Helical Field Kolmogorov vs GS95  |  M_A = {M_A} | Axis = {iaxis}", fontsize=14)

        im0 = axes[0, 0].imshow(P_k.T, origin='lower', cmap='inferno')
        axes[0, 0].set_title("Kolmogorov — polarization fraction P")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(P_obs.T, origin='lower', cmap='inferno')
        axes[0, 1].set_title("GS95 — polarization fraction P")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(phi_k.T, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        axes[1, 0].set_title("Kolmogorov — PA φ [rad]")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(phi_obs.T, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        axes[1, 1].set_title("GS95 — PA φ [rad]")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

        print(f"\n  Mean P  :  Kolmogorov = {P_k.mean():.4f}   GS95 = {P_obs.mean():.4f}")
        print(f"  Std  P  :  Kolmogorov = {P_k.std():.4f}    GS95 = {P_obs.std():.4f}")
        print(f"  Mean Phi : Kolmogorov = {phi_k.mean():.4f}   GS95 = {phi_obs.mean():.4f}")


    return P_obs, phi_obs, cos2g.mean(axis=iaxis)

def helical_field_kolmo(M_A, alpha, iaxis, g_size, plotting):

    # =========================
    # PARAMETERS
    # =========================
    I0 = 1.0
    corr_r = 0.9   # LOS correlation coefficient
    turb_sigma = M_A

    # =========================
    # GRID
    # =========================
    x = np.linspace(-1, 1, g_size)
    y = np.linspace(-1, 1, g_size)
    z = np.linspace(-1, 1, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # avoid singularity at r=0
    r = np.sqrt(X**2 + Y**2) + 1e-8

    # base toroidal field (unrotated in z)
    Bx0 = (-Y * np.cos(alpha)) / r
    By0 = ( X * np.cos(alpha)) / r
    Bz0 = np.full_like(X, np.sin(alpha))

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
    rms    = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    B0_rms = np.sqrt(np.mean(Bx0**2 + By0**2 + Bz0**2))
    if rms > 0:
        dBx *= M_A * B0_rms / rms
        dBy *= M_A * B0_rms / rms
        dBz *= M_A * B0_rms / rms

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
    # Calculate Stokes for appropriate POS components
    # =========================
    # Define LOS and POS components
    if iaxis == 0:        # LOS = x → POS = (y, z)
        B_los = Bx
        B1, B2 = By, Bz
        # Q = (Bz² - By²), U = 2 By Bz  ← check convention matches iaxis=1,2

    elif iaxis == 1:      # LOS = y → POS = (x, z)
        B_los = By
        B1, B2 = Bx, Bz

    else:                 # LOS = z → POS = (x, y)
        B_los = Bz
        B1, B2 = Bx, By


    B_los2  = B_los**2
    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los2

    # Geometrical depolarization
    cos2g = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * np.ones_like(Btot2)

    # Safe division mask
    mask = B_perp2 > 1e-12

    Q = np.zeros_like(B1)
    U = np.zeros_like(B1)

    # Fix: Q should be (B1² - B2²), not (B2² - B1²)
    Q[mask] = cos2g[mask] * (B1[mask]**2 - B2[mask]**2) / B_perp2[mask]
    U[mask] = cos2g[mask] * (2 * B1[mask] * B2[mask])   / B_perp2[mask]

    # LOS integration
    I_obs = I_local.sum(axis=iaxis)
    Q_obs = Q.sum(axis=iaxis)
    U_obs = U.sum(axis=iaxis)

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    # plotting
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
            length=0.25, normalize=True,linewidth=1.5, arrow_length_ratio=0)
        ax.set_title("3D Helical")
        plt.show()

        mid = g_size // 2

        fig, axes = plt.subplots(1,3,figsize=(12,4))

        n =  10 # take every nth index

        # LOS = x → POS = (y,z)
        axes[0].quiver(
            Y[mid, ::n, ::n],
            Z[mid, ::n, ::n],
            By[mid, ::n, ::n],
            Bz[mid, ::n, ::n],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("Z")
        axes[0].set_title("LOS X")

        # LOS = y → POS = (x,z)
        axes[1].quiver(
            X[::n, mid, ::n],
            Z[::n, mid, ::n],
            Bx[::n, mid, ::n],
            Bz[::n, mid, ::n],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[1].set_title("LOS Y")

        # LOS = z → POS = (x,y)
        axes[2].quiver(
            X[::n, ::n, mid],
            Y[::n, ::n, mid],
            Bx[::n, ::n, mid],
            By[::n, ::n, mid],
            headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_title("LOS Z")

        plt.tight_layout()
        plt.show()

    return P_obs, phi_obs, cos2g.mean(axis=iaxis)

def helical_field_random(M_A, alpha, iaxis, g_size=64, plotting=False):

    # =========================
    # PARAMETERS
    # =========================
    I0 = 1.0
    corr_r = 0.9   # LOS correlation coefficient
    turb_sigma = M_A

    # =========================
    # GRID
    # =========================
    x = np.linspace(-1, 1, g_size)
    y = np.linspace(-1, 1, g_size)
    z = np.linspace(-1, 1, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # avoid singularity at r=0
    r = np.sqrt(X**2 + Y**2) + 1e-8

    # base toroidal field (unrotated in z)
    Bx0 = (-Y * np.cos(alpha)) / r
    By0 = ( X * np.cos(alpha)) / r
    Bz0 = np.full_like(X, np.sin(alpha))

    # =========================
    # Turbulence
    # =========================
    delta_theta = np.zeros((g_size, g_size, g_size))
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
                val = corr_r * val + np.random.normal(scale=turb_sigma * np.sqrt(1 - corr_r**2))

    # =========================
    # APPLY TURBULENT ROTATION
    # =========================
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
    # Calculate Stokes for appropriate POS components
    # =========================
    # Define LOS and POS components
    if iaxis == 0:        # LOS = x → POS = (y, z)
        B_los = Bx
        B1, B2 = By, Bz
        # Q = (Bz² - By²), U = 2 By Bz  ← check convention matches iaxis=1,2

    elif iaxis == 1:      # LOS = y → POS = (x, z)
        B_los = By
        B1, B2 = Bx, Bz

    else:                 # LOS = z → POS = (x, y)
        B_los = Bz
        B1, B2 = Bx, By

    B_los2  = B_los**2
    B_perp2 = B1**2 + B2**2
    Btot2   = B_perp2 + B_los2

    # Geometrical depolarization
    cos2g = B_perp2 / (Btot2 + 1e-12)
    I_local = I0 * np.ones_like(Btot2)

    # Safe division mask
    mask = B_perp2 > 1e-12

    Q = np.zeros_like(B1)
    U = np.zeros_like(B1)

    Q[mask] = cos2g[mask] * (B1[mask]**2 - B2[mask]**2) / B_perp2[mask]
    U[mask] = cos2g[mask] * (2 * B1[mask] * B2[mask])   / B_perp2[mask]

    # LOS integration
    I_obs = I_local.sum(axis=iaxis)
    Q_obs = Q.sum(axis=iaxis)
    U_obs = U.sum(axis=iaxis)

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-12)
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)

    # plotting
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
            length=0.25, normalize=True,linewidth=1.5, arrow_length_ratio=0)
        ax.set_title("3D Helical")
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


