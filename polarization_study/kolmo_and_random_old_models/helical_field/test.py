import numpy as np
import matplotlib.pyplot as plt
import warnings

def helical_field_depol_test(M_A, alpha, iaxis, g_size=64, plotting=False):

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

P,PHI,COS = helical_field_depol_test(M_A=0, alpha=np.pi/4, iaxis=1, g_size=10, plotting=True)
