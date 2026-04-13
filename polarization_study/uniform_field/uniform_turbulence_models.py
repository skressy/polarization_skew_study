# Functions:
# _gs95_turbulence
# uniform_field_gs95
# uniform_field_kolmo

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

def uniform_field_gs95(M_A, Bz_true, angle_range, g_size, anisotropy_strength=1.0, seed=None, plotting=False):

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    I0   = 1.0
    Bx0  = 1.0
    By0  = 1.0
    angle_range = np.linspace(angle_range[0], angle_range[1], g_size)

    # ------------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------------
    x = np.linspace(-np.pi, np.pi, g_size)
    y = np.linspace(-np.pi, np.pi, g_size)
    z = np.linspace(-np.pi, np.pi, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    # ------------------------------------------------------------------
    # LARGE-SCALE UNIFORM BACKGROUND FIELD
    # ------------------------------------------------------------------
    Bx_bg = np.zeros_like(X)
    By_bg = np.zeros_like(Y)
    Bz_bg = np.zeros_like(Z)

    for iz in range(g_size):
        itheta = angle_range[iz]
        Bx_bg[:, :, iz] = Bx0 * np.cos(itheta)
        By_bg[:, :, iz] = By0 * np.sin(itheta)
        Bz_bg[:, :, iz] = Bz_true

    # ------------------------------------------------------------------
    # GS95 TURBULENCE
    # Scale-dependent anisotropic spectrum following:
    #   Goldreich & Sridhar 1995, ApJ 438, 763
    #   Cho & Lazarian 2002, PRL 88, 245001
    #   Lazarian 2006, ApJ 645, L25
    # ------------------------------------------------------------------
    dBx, dBy, dBz = _gs95_turbulence(X, Y, Z, Bx_bg, By_bg, Bz_bg, M_A, anisotropy_strength=anisotropy_strength, seed=seed)

    # ------------------------------------------------------------------
    # TOTAL FIELD
    # ------------------------------------------------------------------
    Bx = Bx_bg + dBx
    By = By_bg + dBy
    Bz = Bz_bg + dBz

    # ------------------------------------------------------------------
    # ENFORCE ∇·B = 0 ON TOTAL FIELD
    # ------------------------------------------------------------------
    kx = np.fft.fftfreq(g_size, d=x[1]-x[0]) * 2 * np.pi
    ky = np.fft.fftfreq(g_size, d=y[1]-y[0]) * 2 * np.pi
    kz = np.fft.fftfreq(g_size, d=z[1]-z[0]) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")
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

    # ------------------------------------------------------------------
    # STOKES PARAMETERS
    # LOS = z-axis (axis=2), POS = (x, y)
    # ------------------------------------------------------------------
    Bperp2 = Bx**2 + By**2
    Btot2  = Bperp2 + Bz**2 + 1e-12

    cos2g = Bperp2 / Btot2
    q     = (By**2 - Bx**2) / Btot2
    u     = (2 * Bx * By)   / Btot2

    Q3 = I0 * q
    U3 = I0 * u

    # ------------------------------------------------------------------
    # LOS INTEGRATION
    # ------------------------------------------------------------------
    Q_obs = Q3.sum(axis=2)
    U_obs = U3.sum(axis=2)
    I_obs = I0 * g_size

    P_obs     = np.sqrt(Q_obs**2 + U_obs**2) / I_obs
    phi_obs   = 0.5 * np.arctan2(U_obs, Q_obs)
    cos2g_LOS = cos2g.mean(axis=2)

        # ------------------------------------------------------------------
    if plotting:

        P_k, phi_k, cg_k = uniform_field_kolmo(M_A, Bz_true, angle_range, g_size, plotting=False)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle(f"Uniform Field Kolmogorov vs GS95  |  M_A = {M_A} ")

        im0 = axes[0, 0].imshow(P_k, origin='lower', cmap='inferno')
        axes[0, 0].set_title("Kolmogorov — polarization fraction P")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(P_obs, origin='lower', cmap='inferno')
        axes[0, 1].set_title("GS95 — polarization fraction P")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(phi_k, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        axes[1, 0].set_title("Kolmogorov — PA φ [rad]")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(phi_obs, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        axes[1, 1].set_title("GS95 — PA φ [rad]")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

        print(f"\n  Mean P  :  Kolmogorov = {P_k.mean():.4f}   GS95 = {P_obs.mean():.4f}")
        print(f"  Std  P  :  Kolmogorov = {P_k.std():.4f}    GS95 = {P_obs.std():.4f}")
        print(f"  Mean cg :  Kolmogorov = {cg_k.mean():.4f}   GS95 = {cos2g_LOS.mean():.4f}")
        print(f"  Mean Phi : Kolmogorov = {phi_k.mean():.4f}   GS95 = {phi_obs.mean():.4f}")


    return P_obs, phi_obs, cos2g_LOS

def uniform_field_kolmo(M_A, Bz_true, angle_range, g_size, plotting=False):
    """
    ISM toy polarization model with LOS turbulence parameterized
    by Alfvénic Mach number M_A.
    """

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    sigma_phi = M_A
    I0 = 1.0
    Bx0, By0 = 1.0, 1.0
    angle_range = np.linspace(angle_range[0], angle_range[1], g_size)

    # =========================
    # GRID
    # =========================
    x = np.linspace(-np.pi, np.pi, g_size)
    y = np.linspace(-np.pi, np.pi, g_size)
    z = np.linspace(-np.pi, np.pi, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(Z)

    # =========================
    # LARGE-SCALE POS FIELD
    # =========================
    for iz in range(g_size):
        itheta = angle_range[iz]
        Bx[:, :, iz] = Bx0 * np.cos(itheta)
        By[:, :, iz] = By0 * np.sin(itheta)
        Bz[:, :, iz] = Bz_true

    # ------------------------------------------------------------------
    # 3D KOLMOGOROV TURBULENCE
    # ------------------------------------------------------------------
    kx = np.fft.fftfreq(g_size, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(g_size, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(g_size, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="xy")

    k2 = KX**2 + KY**2 + KZ**2
    k  = np.sqrt(k2)

    # --- random complex Gaussian vector field (unit variance per mode) ---
    rng = np.random.default_rng()

    def rand_complex(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    Fx = rand_complex(X.shape)
    Fy = rand_complex(X.shape)
    Fz = rand_complex(X.shape)

    # --- Kolmogorov amplitude weighting: |B_k| ∝ k^(-11/6) ---
    # Power spectrum E(k) ∝ k^(-11/3) in 3D  →  amplitude ∝ k^(-11/6)
    # Zero-mode is left at zero (no mean turbulent field).
    with np.errstate(divide="ignore", invalid="ignore"):
        kolmogorov_weight = np.where(k > 0, k ** (-11.0 / 6.0), 0.0)

    Fx *= kolmogorov_weight
    Fy *= kolmogorov_weight
    Fz *= kolmogorov_weight

    # --- Enforce Hermitian symmetry so IFFT is real ---
    # np.fft.ifftn handles this automatically as long as we don't break it;
    # taking .real after ifftn is sufficient for our purposes.

    # --- Project out compressive (longitudinal) component: F → F_perp ---
    # This guarantees ∇·δB = 0 in the continuous sense.
    k2_safe = np.where(k2 > 0, k2, 1.0)   # avoid divide-by-zero at k=0

    k_dot_F = KX * Fx + KY * Fy + KZ * Fz

    Fx -= KX * k_dot_F / k2_safe
    Fy -= KY * k_dot_F / k2_safe
    Fz -= KZ * k_dot_F / k2_safe

    # --- Back to real space ---
    dBx = np.fft.ifftn(Fx).real
    dBy = np.fft.ifftn(Fy).real
    dBz = np.fft.ifftn(Fz).real

    # --- Normalise so that rms(|δB|) = M_A ---
    rms = np.sqrt(np.mean(dBx**2 + dBy**2 + dBz**2))
    if rms > 0:
        dBx *= M_A / rms
        dBy *= M_A / rms
        dBz *= M_A / rms

    # --- Add turbulence to large-scale field ---
    Bx += dBx
    By += dBy
    Bz += dBz

    # ------------------------------------------------------------------
    # ENFORCE DIVERGENCE-FREE CONDITION ON FULL FIELD
    # Project out the longitudinal part of the total field in Fourier space.
    # ------------------------------------------------------------------
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
    # STOKES FROM FIELD
    # =========================
    Bperp2 = Bx**2 + By**2
    Btot2 = Bperp2 + Bz**2 + 1e-12

    cos2g = Bperp2 / Btot2
    q = (By**2 - Bx**2) / Btot2
    u = (2 * Bx * By) / Btot2

    Q3 = I0 * q
    U3 = I0 * u

    # =========================
    # LOS INTEGRATION
    # =========================
    Q_obs = Q3.sum(axis=2)
    U_obs = U3.sum(axis=2)
    I_obs = I0 * g_size

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / I_obs
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)
    cos2g_LOS = cos2g.mean(axis=2)



    return P_obs, phi_obs, cos2g_LOS

def uniform_field_random(M_A, Bz_true, angle_range, g_size, plotting=False):
    """
    ISM toy polarization model with LOS turbulence parameterized
    by Alfvénic Mach number M_A.
    """
    # =========================
    # PARAMETERS
    # =========================
    I0 = 1.0
    Bx0, By0 = 1.0, 1.0
    corr_r = 0.9   # LOS correlation coefficient
    angle_range = np.linspace(angle_range[0], angle_range[1], g_size)
    sigma_phi = M_A

    # =========================
    # GRID
    # =========================
    x = np.linspace(-np.pi, np.pi, g_size)
    y = np.linspace(-np.pi, np.pi, g_size)
    z = np.linspace(-np.pi, np.pi, g_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(Z)

    # =========================
    # LARGE-SCALE POS FIELD
    # =========================
    for iz in range(g_size):
        itheta = angle_range[iz]
        Bx[:, :, iz] = Bx0 * np.cos(itheta)
        By[:, :, iz] = By0 * np.sin(itheta)
        Bz[:, :, iz] = Bz_true

    # =========================
    # LOS TURBULENCE (AR(1))
    # =========================
    delta_theta = np.zeros((g_size, g_size, g_size))

    for ix in range(g_size):
        for iy in range(g_size):
            val = np.random.normal(scale=sigma_phi)
            for iz in range(g_size):
                delta_theta[ix, iy, iz] = val
                val = corr_r * val + np.random.normal(
                    scale=sigma_phi * np.sqrt(1 - corr_r**2)
                )

    # =========================
    # APPLY TURBULENT ROTATION
    # =========================
    Bx_rot = Bx * np.cos(delta_theta) - By * np.sin(delta_theta)
    By_rot = Bx * np.sin(delta_theta) + By * np.cos(delta_theta)

    Bx, By = Bx_rot, By_rot

    # =========================
    # OPTIONAL FIELD PLOTTING
    # =========================
    if plotting:
        fig, ax = plt.subplots(1, g_size, figsize=(18, 2))
        for iz in range(g_size):
            ax[iz].quiver(
                X[:, :, iz], Y[:, :, iz],
                Bx[:, :, iz], By[:, :, iz],
                pivot="middle",
                scale=2,
                scale_units="xy",
                headlength=0,
                headaxislength=0
            )
            ax[iz].set_title(f"z={iz}")
        plt.tight_layout()
        plt.show()

    # =========================
    # STOKES FROM FIELD
    # =========================
    Bperp2 = Bx**2 + By**2
    Btot2 = Bperp2 + Bz**2 + 1e-12

    cos2g = Bperp2 / Btot2
    q = (By**2 - Bx**2) / Btot2
    u = (2 * Bx * By) / Btot2

    Q3 = I0 * q
    U3 = I0 * u

    # =========================
    # LOS INTEGRATION
    # =========================
    Q_obs = Q3.sum(axis=2)
    U_obs = U3.sum(axis=2)
    I_obs = I0 * g_size

    P_obs = np.sqrt(Q_obs**2 + U_obs**2) / I_obs
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)
    cos2g_LOS = cos2g.mean(axis=2)

    return P_obs, phi_obs, cos2g_LOS

