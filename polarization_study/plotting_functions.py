import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from magnetic_fields import simulate_field
from scipy.stats import skew, skewnorm
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import math

def plot_model_comparison(geometry, turbulence_a, turbulence_b,
                    M_A=0.5, alpha=0, iaxis=2, g_size=256):
    """
    Side-by-side P and PA maps for two turbulence models.
    """
    P_a, phi_a, _ = simulate_field(geometry, turbulence_a, M_A=M_A, alpha=alpha, iaxis=iaxis, g_size=g_size)
    P_b, phi_b, _ = simulate_field(geometry, turbulence_b, M_A=M_A, alpha=alpha, iaxis=iaxis, g_size=g_size)

    axis_label = ['x', 'y', 'z'][iaxis]
    fig, axes  = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f"{geometry.capitalize()} field  |  M_A = {M_A}  |  LOS = {axis_label}", fontsize=14)

    im0 = axes[0,0].imshow(P_a.T,   origin='lower', cmap='inferno')
    im1 = axes[0,1].imshow(P_b.T,   origin='lower', cmap='inferno')
    im2 = axes[1,0].imshow(phi_a.T, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
    im3 = axes[1,1].imshow(phi_b.T, origin='lower', cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)

    titles = [f'{turbulence_a} — P', f'{turbulence_b} — P', f'{turbulence_a} — PA [rad]', f'{turbulence_b} — PA [rad]']
    for ax, im, t in zip(axes.flat, [im0,im1,im2,im3], titles):
        ax.set_title(t)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    print(f"\n  Mean P : {turbulence_a} = {P_a.mean():.4f}     {turbulence_b} = {P_b.mean():.4f}")
    print(f"  Std  P : {turbulence_a} = {P_a.std():.4f}    {turbulence_b} = {P_b.std():.4f}")

def plot_gs95_turb_model(geometry, turbulence='gs95', M_A=0.5, alpha=0, iaxis=2, g_size=256):

    P, phi, _ = simulate_field(geometry, turbulence, M_A=M_A, alpha=alpha, iaxis=iaxis, g_size=g_size)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(P.T, origin='lower', cmap='inferno')

    ax.set_title(f"{geometry.capitalize()} Field  |  $M_A$ = {M_A}", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Polarization Fraction $P$", fontsize=11)

    # ax.set_xticks([])
    # ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    print(f"\n  Mean P : {P.mean():.4f}")
    print(f"  Std  P  : {P.std():.4f}")

def plot_P_dist(geometry='helical', turbulence='gs95',
                         alpha=0, iaxis=2, g_size=128,
                         mach_vals=None):
    if mach_vals is None:
        mach_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                     0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]

    axis_label = ['x', 'y', 'z'][iaxis]
    plt.figure(figsize=(9, 5))
    skews = []

    for idx, imach in enumerate(mach_vals):
        P, _, _ = simulate_field(geometry, turbulence,
                                 M_A=imach, alpha=alpha,
                                 iaxis=iaxis, g_size=g_size)
        Pflat    = P.flatten()
        Pflat    = Pflat[Pflat > 0]
        skew_val = skew(Pflat, bias=False)
        skews.append(skew_val)

        color = plt.cm.rainbow(idx / len(mach_vals))
        a, loc, scale = skewnorm.fit(Pflat)
        x_fit = np.linspace(Pflat.min(), Pflat.max(), 300)

        plt.hist(Pflat, bins=30, density=True, alpha=0.3, color=color)
        plt.plot(x_fit, skewnorm.pdf(x_fit, a, loc=loc, scale=scale),
                 lw=2, label=rf"$M_A={imach:.2f}$, skew={skew_val:.2f}",
                 color=color)

    plt.xlabel('P', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title(rf'{geometry.capitalize()} | {turbulence} | '
              rf'$\alpha$={alpha} | LOS={axis_label}', fontsize=14)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return mach_vals, skews


def plot_pol_distributions():

    mach_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    g_size = 128
    titles = ['Uniform', 'Wavy', 'Helical Axis=1', 'Helical Axis=2', 'Hourglass Axis=1', 'Hourglass Axis=2']
    fig, axes = plt.subplots(6, 1, figsize=(9, 14), constrained_layout=True)

    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(vmin=min(mach_vals), vmax=max(mach_vals))

    for idx, imach in enumerate(mach_vals):
        P_U,  PHI_U,  COS_U     = simulate_field(geometry='uniform',   turbulence='gs95', M_A=imach, alpha=np.pi/4, g_size=g_size, anisotropy_strength=1.0, seed=None)
        P_W,  PHI_W,  COS_W     = simulate_field(geometry='wavy',      turbulence='gs95', M_A=imach,                g_size=g_size, anisotropy_strength=1.0, seed=None)
        P_He1, PHI_He1, COS_He1 = simulate_field(geometry='helical',   turbulence='gs95', M_A=imach, alpha=0.0,     iaxis=1, g_size=g_size, anisotropy_strength=1.0, seed=None)
        P_He2, PHI_He2, COS_He2 = simulate_field(geometry='helical',   turbulence='gs95', M_A=imach, alpha=0.0,     iaxis=2, g_size=g_size, anisotropy_strength=1.0, seed=None)
        P_Ho1, PHI_Ho1, COS_Ho1 = simulate_field(geometry='hourglass', turbulence='gs95', M_A=imach,                iaxis=1, g_size=g_size, anisotropy_strength=1.0, seed=None)
        P_Ho2, PHI_Ho2, COS_Ho2 = simulate_field(geometry='hourglass', turbulence='gs95', M_A=imach,                iaxis=2, g_size=g_size, anisotropy_strength=1.0, seed=None)

        for P, ax, title in zip([P_U, P_W, P_He1, P_He2, P_Ho1, P_Ho2], axes, titles):
            Pflat    = P.flatten()
            Pflat    = Pflat[Pflat > 0]
            skew_val = skew(Pflat, bias=False)

            color = plt.cm.plasma(idx / len(mach_vals))
            a, loc, scale = skewnorm.fit(Pflat)
            x_fit = np.linspace(Pflat.min(), Pflat.max(), 300)

            counts, bin_edges = np.histogram(Pflat, bins=30)
            ax.hist(Pflat, bins=bin_edges, density=True, alpha=0.4, color=color)
            ax.plot(x_fit, skewnorm.pdf(x_fit, a, loc=loc, scale=scale), lw=2, label=rf"{skew_val:.2f}", color=color, alpha=0.8)
            ax.set_title(title, fontsize=16, y=0.8)
            if title == 'Hourglass Axis=2' or title=='Helical Axis=1':
                ax.legend(fontsize=8, loc='upper right', ncol=2, title='Skew')
            else:
                ax.legend(fontsize=8, loc='upper left', ncol=2, title='Skew')

    # Shared colorbar across all axes with lines at each M_A value
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label(r'$M_A$', fontsize=14)
    cbar.ax.hlines(mach_vals, 0, 1, colors='white', linewidth=0.8, transform=cbar.ax.get_yaxis_transform())

    plt.xlabel('p', fontsize=20, labelpad=10)
    axes[2].set_ylabel('Number Density', fontsize=20, labelpad=10)
    plt.savefig('pol_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.show()



def model_tanh(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d

def model_rational(x, a, b, c, d):
    return a - b / (x**c + d)

def model_recip(x, a, b, c):
    return a - b * np.exp(-c * x) / x

_FIT_MODELS = {
    'tanh'    : (model_tanh,     [1.0, 2.0, -1.0, 0.0], (-np.inf, np.inf)),
    'rational': (model_rational, [0.6, 2.0,  1.0, 0.1],
                 ([0, 0, 0, 1e-6], [2, 10, 3, 1])),
    'recip'   : (model_recip,    [0.6, 0.15, 8.0],
                 ([0.4, 0.01, 1.0], [1.2, 1.0, 20.0])),
}


def fit_and_plot(all_skews, mach_vals, mach_dense,
                 color='blue', label='', fit_method='tanh'):
    """
    Fit mean skewness curve and overlay ±1σ band from per-run fits.
    fit_method : 'tanh' | 'rational' | 'recip'
    """
    model, p0, bounds = _FIT_MODELS[fit_method]
    mean_skews        = np.mean(all_skews, axis=0)

    popt, _ = curve_fit(model, mach_vals, mean_skews,
                        p0=p0, bounds=bounds,
                        method='trf', maxfev=10000)
    mean_fit = model(mach_dense, *popt)

    fitted_curves = []
    for run_skews in all_skews:
        try:
            popt_i, _ = curve_fit(model, mach_vals, run_skews,
                                  p0=popt, bounds=bounds,
                                  method='trf', maxfev=10000)
            fitted_curves.append(model(mach_dense, *popt_i))
        except RuntimeError:
            pass   # skip failed runs silently

    std_fit = np.std(np.array(fitted_curves), axis=0)

    plt.plot(mach_dense, mean_fit, color=color, linewidth=2.5, label=label)
    plt.fill_between(mach_dense,
                     mean_fit - std_fit, mean_fit + std_fit,
                     color=color, alpha=0.25)


def run_skewness_sweep(geometry='helical', turbulence='gs95',
                       alpha=0, iaxis=2, g_size=64,
                       n_runs=50, mach_vals=None,
                       fit_method='recip',
                       color='blue', label=None):
    """
    Collect skewness(M_A) over n_runs realisations, plot raw traces
    and fitted mean ± 1σ.
    """
    if mach_vals is None:
        mach_vals = np.linspace(0.1, 3, 50)
    if label is None:
        axis_label = ['x', 'y', 'z'][iaxis]
        label = rf'{geometry}/{turbulence} LOS={axis_label} $\pm1\sigma$'

    rng       = np.random.default_rng()
    all_skews = []

    for i in range(n_runs):
        run_skews = []
        for imach in mach_vals:
            P, _, _ = simulate_field(geometry, turbulence,
                                     M_A=imach, alpha=alpha,
                                     iaxis=iaxis, g_size=g_size)
            run_skews.append(skew(P.flatten(), bias=False))
        all_skews.append(run_skews)
        plt.plot(mach_vals, run_skews, color=color, alpha=0.1, linewidth=0.8)

    all_skews  = np.array(all_skews)
    mach_dense = np.linspace(mach_vals[0], mach_vals[-1], 300)
    fit_and_plot(all_skews, mach_vals, mach_dense,
                 color=color, label=label, fit_method=fit_method)
    return all_skews


## BELOW ALL PAPER PLOTS ###

def plot_synthetic_pol_maps(M_A = 0.01, g_size=32, step=4, vscale=0.5):
    P_U,  PHI_U,  COS_U     = simulate_field(geometry='uniform',   turbulence='gs95', M_A=M_A, alpha=np.pi/4, g_size=g_size, anisotropy_strength=1.0, seed=None)
    P_W,  PHI_W,  COS_W     = simulate_field(geometry='wavy',      turbulence='gs95', M_A=M_A,                g_size=g_size, anisotropy_strength=1.0, seed=None)
    P_He1, PHI_He1, COS_He1 = simulate_field(geometry='helical',   turbulence='gs95', M_A=M_A, alpha=0.0,     iaxis=1, g_size=g_size, anisotropy_strength=1.0, seed=None)
    P_He2, PHI_He2, COS_He2 = simulate_field(geometry='helical',   turbulence='gs95', M_A=M_A, alpha=0.0,     iaxis=2, g_size=g_size, anisotropy_strength=1.0, seed=None)
    P_Ho1, PHI_Ho1, COS_Ho1 = simulate_field(geometry='hourglass', turbulence='gs95', M_A=M_A,                iaxis=1, g_size=g_size, anisotropy_strength=1.0, seed=None)
    P_Ho2, PHI_Ho2, COS_Ho2 = simulate_field(geometry='hourglass', turbulence='gs95', M_A=M_A,                iaxis=2, g_size=g_size, anisotropy_strength=1.0, seed=None)

    geometries = [
        ('Uniform',   P_U, PHI_U),
        ('Wavy', P_W, PHI_W),
        ('Helical axis=1', P_He1,  PHI_He1),
        ('Helical axis=2', P_He2,  PHI_He2),
        ('Hourglass axis=1', P_Ho1,  PHI_Ho1),
        ('Hourglass axis=2', P_Ho2,  PHI_Ho2)]

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    for ax, (label, P, PHI) in zip(axes.flat, geometries):

        P_map   = P
        PHI_map = PHI
        nx, ny = P_map.shape

        Bx = P_map * np.sin(PHI_map)
        By = P_map * np.cos(PHI_map)

        X, Y = np.meshgrid(np.arange(nx) + 2, np.arange(ny) + 2, indexing='ij')

        im = ax.imshow(P_map.T, origin='lower', extent=[0, nx, 0, ny],
                       cmap='plasma', alpha=1,
                       vmin=P_map.min(), vmax=P_map.max())

        # fig.colorbar(im, ax=ax, label='P %', fraction=0.046, pad=0.04)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))
        if ax == axes[0, 1] or ax == axes[1, 1] or ax == axes[2, 1]:
            cbar.ax.set_ylabel('P %', fontsize=12)

        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  Bx[::step, ::step], By[::step, ::step],
                  pivot='middle', scale=vscale, scale_units='xy',
                  headlength=0, headaxislength=0, width=0.008, color='white')

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_title(label, fontsize=16)
        ax.set_aspect('equal')

    
    axes[2,0].set_xlabel('X', fontsize=14)
    axes[2,1].set_xlabel('X', fontsize=14)
    axes[0,0].set_ylabel('Y', fontsize=14)
    axes[1,0].set_ylabel('Y', fontsize=14)
    axes[2,0].set_ylabel('Y', fontsize=14)

    plt.tight_layout()
    plt.savefig('synthetic_pol_maps.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_skewness_trendlines():
    geometries = ['uniform', 'wavy', 'helical', 'helical', 'hourglass', 'hourglass']
    axes = [None, None, 1, 2, 1, 2]                 # paired 1-to-1 with geometries
    turbulence = 'gs95'
    color = ['#0072B2', '#E69F00', '#D55E00', '#56B4E9', '#009E73', '#F0E442']
    labels = ['Uniform', 'Wavy', 'Helical Axis = 1', 'Helical Axis = 2', 'Hourglass Axis = 1', 'Hourglass Axis = 2']

    mach_vals = np.linspace(0.1, 3, 50)
    g_size = 64
    n_runs = 20

    plt.figure(figsize=(10, 7))
    plt.xlabel(r'$M_A$', fontsize=20)
    plt.ylabel('Skewness of p', fontsize=20)

    # zip iaxis in here so each geometry gets exactly one axis
    for geometry, iaxis, icolor, ilabel in zip(geometries, axes, color, labels):

        if geometry == 'hourglass' or geometry == 'helical':
            alpha = 0.0
            all_skews = []
            for i in range(n_runs):
                run_skews = []
                for imach in mach_vals:
                    P, _, _ = simulate_field(geometry, turbulence, M_A=imach, alpha=alpha, iaxis=iaxis, g_size=g_size)
                    run_skews.append(skew(P.flatten(), bias=False))
                plt.plot(mach_vals, run_skews, color=icolor, alpha=0.15, linewidth=0.8)
                all_skews.append(run_skews)

            all_skews  = np.array(all_skews)
            mach_dense = np.linspace(mach_vals[0], mach_vals[-1], 300)
            fit_and_plot(all_skews, mach_vals, mach_dense,
                         color=icolor, label=ilabel, fit_method='tanh')

        else:
            alpha = np.pi / 4
            all_skews = []
            for i in range(n_runs):
                run_skews = []
                for imach in mach_vals:
                    P, _, _ = simulate_field(geometry, turbulence, M_A=imach, alpha=alpha, g_size=g_size)
                    run_skews.append(skew(P.flatten(), bias=False))
                plt.plot(mach_vals, run_skews, color=icolor, alpha=0.15, linewidth=0.8)
                all_skews.append(run_skews)

            all_skews  = np.array(all_skews)
            mach_dense = np.linspace(mach_vals[0], mach_vals[-1], 300)
            fit_and_plot(all_skews, mach_vals, mach_dense,
                         color=icolor, label=ilabel, fit_method='tanh')
    plt.axvline(x=1.0, color='red', linestyle='--', label='$M_A=1$')
    plt.ylim(-2.5,1.5)
    plt.tight_layout()
    plt.legend(fontsize=12, loc='lower right')
    plt.grid()
    plt.savefig('skewness_trendlines.pdf', dpi=300)
    plt.show()

# def plot_skewness_trendlines():
#     geometries = ['uniform', 'wavy', 'helical', 'helical', 'hourglass', 'hourglass']
#     axes = [None, None, 1, 2, 1, 2]                 # paired 1-to-1 with geometries
#     turbulence = 'gs95'
#     color = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
#     labels = ['Uniform', 'Wavy', 'Helical Axis = 1', 'Helical Axis = 2', 'Hourglass Axis = 1', 'Hourglass Axis = 2']

#     mach_vals = np.linspace(0.1, 3, 50)
#     g_size = 64
#     n_runs = 10

#     plt.figure(figsize=(10, 7))
#     plt.xlabel(r'$M_A$', fontsize=16)
#     plt.ylabel('Skewness of P', fontsize=16)

#     # zip iaxis in here so each geometry gets exactly one axis
#     for geometry, iaxis, icolor, ilabel in zip(geometries, axes, color, labels):

#         if geometry == 'hourglass' or geometry == 'helical':
#             alpha = 0.0
#             all_skews = []
#             for i in range(n_runs):
#                 run_skews = []
#                 for imach in mach_vals:
#                     P, _, _ = simulate_field(geometry, turbulence, M_A=imach,
#                                              alpha=alpha, iaxis=iaxis, g_size=g_size)
#                     run_skews.append(skew(P.flatten(), bias=False))
#                 plt.plot(mach_vals, run_skews, color=icolor, alpha=0.1, linewidth=0.8)
#                 all_skews.append(run_skews)

#             all_skews  = np.array(all_skews)
#             mach_dense = np.linspace(mach_vals[0], mach_vals[-1], 300)
#             fit_and_plot(all_skews, mach_vals, mach_dense,
#                          color=icolor, label=ilabel, fit_method='tanh')

#         else:
#             alpha = np.pi / 4
#             all_skews = []
#             for i in range(n_runs):
#                 run_skews = []
#                 for imach in mach_vals:
#                     P, _, _ = simulate_field(geometry, turbulence, M_A=imach,
#                                              alpha=alpha, g_size=g_size)
#                     run_skews.append(skew(P.flatten(), bias=False))
#                 plt.plot(mach_vals, run_skews, color=icolor, alpha=0.1, linewidth=0.8)
#                 all_skews.append(run_skews)

#             all_skews  = np.array(all_skews)
#             mach_dense = np.linspace(mach_vals[0], mach_vals[-1], 300)
#             fit_and_plot(all_skews, mach_vals, mach_dense,
#                          color=icolor, label=ilabel, fit_method='tanh')

#     plt.ylim(-2.5,1.5)
#     plt.tight_layout()
#     plt.legend(fontsize=10, loc='lower right')
#     plt.show()

def draw_cloud(ax,radius=3,center=(5,5)):
    theta=np.linspace(0,2*np.pi,400)
    r=radius*(1+0.15*np.sin(6*theta)+0.08*np.sin(3*theta))
    x=center[0]+r*np.cos(theta)
    y=center[1]+r*np.sin(theta)
    ax.plot(x,y,color='black',lw=2)

def LOS_contrib_visual():

    # Base field
    x0,y0,z0=0.78,0.78,0.00

    # Field + turb
    x1=x0+0.2
    y1=y0-0.2
    z1=z0+0.2

    # Field + turb + integration LOS
    x2=x1-0.2
    y2=y1-0.2
    z2=z1+0.1

    def project(x,y,z):
        L=np.sqrt(x**2+y**2+z**2)
        if L==0:
            return 0,0
        return x/L,y/L

    vx0,vy0=project(x0,y0,z0)
    vx1,vy1=project(x1,y1,z1)
    vx2,vy2=project(x2,y2,z2)
    vx3,vy3=project(x2,-y2,0.2)

    fig,axs=plt.subplots(1,3,figsize=(14,5))

    xc,yc=4,4
    Ldraw=4

    for ax in axs:
        ax.set_xlim(0,8)
        ax.set_ylim(0,8)
        ax.set_xticks([])
        ax.set_yticks([])
        # draw_cloud(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

    def draw_centered(ax,vx,vy,color,lw,alpha):
        dx=vx*Ldraw
        dy=vy*Ldraw
        ax.plot([xc-dx/2,xc+dx/2],[yc-dy/2,yc+dy/2],color=color,lw=lw,alpha=alpha)

    # Panel 1
    draw_centered(axs[0],vx0,vy0,'black',5,alpha=1)

    # Panel 2
    draw_centered(axs[1],vx1,vy1,'black',5,alpha=1)

    # Panel 3
    draw_centered(axs[2],vx1,vy1,'black',5,alpha=1)

    # Opposite direction for depolarization
    draw_centered(axs[2],vx3,vy3,'purple',5, alpha=0.5)

    axs[0].set_title("FIELD",fontsize=16, fontweight='bold', y=0.9)
    axs[1].set_title("FIELD + TURB",fontsize=16,fontweight='bold', y=0.9)
    axs[2].text(0.62, 0.93, "FIELD + TURB + ", fontsize=16, fontweight='bold',
                ha='right', va='center', transform=axs[2].transAxes, color='black')
    axs[2].text(0.62, 0.93, "INT-LOS", fontsize=16, fontweight='bold',
                ha='left', va='center', transform=axs[2].transAxes, color='purple', alpha=0.5)

    # axs[0].text(0.02,.15,r"$x_0 = 0.78$",transform=axs[0].transAxes, fontsize=12)
    # axs[0].text(0.02,.10,r"$y_0 = 0.78$",transform=axs[0].transAxes, fontsize=12)
    # axs[0].text(0.02,.05,r"$z_0 = 0$",transform=axs[0].transAxes, fontsize=12)

    # axs[1].text(0.02,.15,r"$x_1 = x_0 + 0.2 = 0.98$",transform=axs[1].transAxes, fontsize=12)
    # axs[1].text(0.02,.1,r"$y_1 = y_0 - 0.2 = 0.58$",transform=axs[1].transAxes, fontsize=12)
    # axs[1].text(0.02,.05,r"$z_1 = z_0 + 0.2 = 0.20$",transform=axs[1].transAxes, fontsize=12)

    # axs[2].text(0.02,.15,r"$x_2 = x_1 - 0.2 = 0.78$",transform=axs[2].transAxes, fontsize=12)
    # axs[2].text(0.02,.1,r"$y_2 = y_1 - 0.2 = 0.38$",transform=axs[2].transAxes, fontsize=12)
    # axs[2].text(0.02,.05,r"$z_2 = z_1 + 0.1 = 0.30$",transform=axs[2].transAxes, fontsize=12)

    axs[0].text(0.8, 0.05, "POS", transform=axs[0].transAxes, fontsize=20, color='black')
    axs[1].text(0.8, 0.05, "POS", transform=axs[1].transAxes, fontsize=20, color='black')
    axs[2].text(0.8, 0.05, "POS", transform=axs[2].transAxes, fontsize=20, color='black')

    plt.tight_layout()
    plt.savefig('LOS_contrib_visual.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pol_comparisons(nruns, gsize):
    geometries = ['uniform', 'wavy', 'helical', 'helical', 'hourglass', 'hourglass']
    naxis = [2, 2, 1, 2, 1, 2]
    turbulence = 'gs95'
    # colors  = ['red', 'blue', 'orange', 'green', 'purple', 'cyan']
    # colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
    colors = ['#0072B2', '#E69F00', '#D55E00', '#56B4E9', '#009E73', '#F0E442']
    labels = ['Uniform', 'Wavy', 'Helical Axis = 1', 'Helical Axis = 2', 'Hourglass Axis = 1', 'Hourglass Axis = 2']

    nbins = 30
    mach_numbers = [0.25, 0.75, 2]

    # --- Pre-collect all P values to define global bin edges ---
    all_P_vals = []
    for Mval in mach_numbers:
        for geometry, iaxis in zip(geometries, naxis):
            alpha = np.pi / 4 if geometry == 'uniform' else 0.0
            P, _, _ = simulate_field(geometry, turbulence, M_A=Mval, alpha=alpha, iaxis=iaxis, g_size=gsize)
            all_P_vals.append(P.flatten())

    global_min = np.min([p.min() for p in all_P_vals])
    global_max = np.max([p.max() for p in all_P_vals])
    bin_edges = np.linspace(global_min, global_max, nbins + 1)

    # --- Now plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for iax, Mval in zip(axes, mach_numbers):
        for geometry, iaxis, icolor, ilabel in zip(geometries, naxis, colors, labels):

            alpha = np.pi / 4 if geometry == 'uniform' else 0.0
            all_hists = []

            for i in range(nruns):
                P, _, _ = simulate_field(geometry, turbulence, M_A=Mval, alpha=alpha, iaxis=iaxis, g_size=gsize)
                hist, _ = np.histogram(P.flatten(), bins=bin_edges, density=True)
                all_hists.append(hist)

            all_hists = np.array(all_hists)
            mean_hist = np.mean(all_hists, axis=0)
            std_hist  = np.std(all_hists, axis=0)
            sem_hist  = std_hist / np.sqrt(nruns)

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            iax.plot(bin_centers, mean_hist, color=icolor, label=ilabel, linewidth=3)
            iax.fill_between(bin_centers, mean_hist - std_hist, mean_hist + std_hist, color=icolor, alpha=0.2)
  
            iax.set_title(f"M$_A$ = {Mval}", fontsize=20)

    axes[1].set_xlabel("p", fontsize=22)
    axes[0].set_ylabel("Number Density", fontsize=22)
    axes[0].legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('pol_comparisons.pdf', dpi=300)
    plt.show()


def plot_PA_comparisons(nruns, gsize):
    geometries = ['uniform', 'wavy', 'helical', 'helical', 'hourglass', 'hourglass']
    naxis = [2, 2, 1, 2, 1, 2]
    turbulence = 'gs95'
    # colors  = ['red', 'blue', 'orange', 'green', 'purple', 'cyan']
    colors = ['#0072B2', '#E69F00', '#D55E00', '#56B4E9', '#009E73', '#F0E442']
    labels = ['Uniform', 'Wavy', 'Helical Axis = 1', 'Helical Axis = 2', 'Hourglass Axis = 1', 'Hourglass Axis = 2']

    nbins = 30
    mach_numbers = [0.25, 0.75, 2]

    # --- Pre-collect all P values to define global bin edges ---
    all_PHI_vals = []
    for Mval in mach_numbers:
        for geometry, iaxis in zip(geometries, naxis):
            alpha = np.pi / 4 if geometry == 'uniform' else 0.0
            _, PHI, _ = simulate_field(geometry, turbulence, M_A=Mval, alpha=alpha, iaxis=iaxis, g_size=gsize)
            all_PHI_vals.append(PHI.flatten())

    global_min = np.min([p.min() for p in all_PHI_vals])
    global_max = np.max([p.max() for p in all_PHI_vals])
    bin_edges = np.linspace(global_min, global_max, nbins + 1)

    # --- Now plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for iax, Mval in zip(axes, mach_numbers):
        for geometry, iaxis, icolor, ilabel in zip(geometries, naxis, colors, labels):

                
            alpha = np.pi / 4 if geometry == 'uniform' else 0.0
            all_hists = []

            for i in range(nruns):
                _, PHI, _ = simulate_field(geometry, turbulence, M_A=Mval, alpha=alpha, iaxis=iaxis, g_size=gsize)
                hist, _ = np.histogram(PHI.flatten(), bins=bin_edges, density=True)
                all_hists.append(hist)

            all_hists = np.array(all_hists)
            mean_hist = np.mean(all_hists, axis=0)
            std_hist  = np.std(all_hists, axis=0)
            sem_hist  = std_hist / np.sqrt(nruns)

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            iax.plot(bin_centers, mean_hist, color=icolor, label=ilabel, linewidth=2)
            iax.fill_between(bin_centers, mean_hist - std_hist, mean_hist + std_hist, color=icolor, alpha=0.2)

            iax.set_title(f"M$_A$ = {Mval}", fontsize=20)

    axes[1].set_xlabel("Position Angle", fontsize=22)
    axes[0].set_ylabel("Number Density", fontsize=22)
    axes[0].legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('PA_comparisons.pdf', dpi=300)
    plt.show()
