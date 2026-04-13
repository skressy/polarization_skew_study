import numpy as np
import matplotlib.pyplot as plt
from magnetic_fields import simulate_field
from scipy.stats import skew, skewnorm
from scipy.optimize import curve_fit


def plot_comparison(geometry, turbulence_a, turbulence_b,
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


def plot_P_distributions(geometry='helical', turbulence='gs95',
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

# ── Fit models ────────────────────────────────────────────────────────────────

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
                 color='blue', label='', fit_method='recip'):
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


# ── Run & plot ─────────────────────────────────────────────────────────────────

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

