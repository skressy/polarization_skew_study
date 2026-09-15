import numpy as np
import matplotlib.pyplot as plt
# from model import simulate_field, compute_S, _stokes_and_polarization
from scipy.stats import skew, skewnorm
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import seaborn as sns
from collections import defaultdict


def count_peaks(samples, bandwidth=0.1, prominence=0.05): #keep prominence at 0.05
    samples = np.asarray(samples).flatten()  # ensure 1D
    kde = gaussian_kde(samples, bw_method=bandwidth)
    x = np.linspace(-np.pi/2, np.pi/2, 1000)
    density = kde(x)
    density /= density.max()  # normalize
    peaks, props = find_peaks(density, prominence=prominence)
    return len(peaks), x[peaks], density

def circular_variance(phi_map):
    # Range= [0,1]; 0 = perfectly ordered, 1 = fully random.
    phi_flat = phi_map.flatten()
    R = np.abs(np.mean(np.exp(2j * phi_flat)))  # mean resultant length
    return 1 - R

def decision_tree(P, PHI, BLOS, NH2):

    Pflat = P.flatten()
    Pflat = Pflat[Pflat > 0]

    skew_val = skew(Pflat, bias=False)

    if skew_val > 0.2:
        CV = circular_variance(PHI)
        if CV > 0.3:
            med_P = np.median(Pflat)
            if med_P > 0.13:
                geometry = 'None'
                print('Super-Alfvenic, Geometry unrecoverable')
                # print('Mach no: ', M_A, 'Skew: ', skew_val, 'CV: ', CV, 'P median: ', med_P)

            else:
                geometry = 'Hourglass Axis = 2'
                print('Sub-Alfvenic, geometry Hourglass Axis = 2')
        else:
            geometry = 'Helical Axis = 1'
            print('Sub-Alfvenic, geometry Helical Axis = 1')
            # print('Mach no: ', M_A, 'Skew: ', skew_val, 'CV: ', CV)

    else:
        Pflat = P.flatten()
        Pflat = Pflat[Pflat > 0]

        a, loc, scale = skewnorm.fit(Pflat)
        x_fit = np.linspace(Pflat.min(), Pflat.max(), 300)
        P_peak = x_fit[np.argmax(skewnorm.pdf(x_fit, a, loc=loc, scale=scale))]

        # ── Step 2: Circular Variance ─────────────────────────
        CV = circular_variance(PHI)

        if P_peak <= 0.1:
            # low polarization fraction -> disordered or helical
            if CV > 0.5:
                geometry = 'Hourglass Axis = 2'
            else:
                geometry = 'Helical Axis = 1'

        elif 0.1 <= P_peak < 0.4:
            geometry = 'Helical Axis = 1'
            
        elif 0.4 <= P_peak <= 1:

            if CV < 0.1:
                # ordered field -> uniform or hourglass axis=1
                M_PHI = 7.6e-21 * NH2 / BLOS
                if M_PHI < 1:
                    geometry = 'Uniform'
                else:
                    geometry = 'Hourglass Axis = 1'

            elif 0.1 <= CV < 0.5:
                # moderate disorder -> hourglass axis=2
                geometry = 'Hourglass Axis = 2'

            elif 0.5 <= CV < 0.75:
                # wavy sits here based on observed CV ~ 0.57-0.71
                n_peaks, peak_loc, dens = count_peaks(samples=PHI, bandwidth=0.1, prominence=0.75)
                if n_peaks > 1:
                    geometry = 'Wavy'
                else:
                    geometry = 'Hourglass Axis = 2'

            else:
                # CV >= 0.75 -> fully disordered -> helical axis=2
                geometry = 'Helical Axis = 2'

        else:
            print('Peak P outside [0,1] range.')
            geometry = 'None'

    return geometry

def plot_sstat_heatmaps(sstat_data, noise_vals, M_A_vals, geometries):
    # plots heatmap of Circular Variance with varying M_A and U/Q noise
    geom_iaxis_pairs = [(geom, iaxis) 
                        for geom, cfg in geometries.items() 
                        for iaxis in cfg['iaxis']]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()

    for ax, (geom, iaxis) in zip(axes_flat, geom_iaxis_pairs):
        # build 2D grid: rows = noise, cols = M_A
        grid = np.zeros((len(noise_vals), len(M_A_vals)))

        for i, snoise in enumerate(noise_vals):
            for j, ima in enumerate(M_A_vals):
                vals = sstat_data.get((geom, iaxis, snoise, ima), [])
                grid[i, j] = np.mean(vals) if vals else np.nan

        im = ax.imshow(grid, aspect='auto', origin='lower',
                       cmap='viridis', vmin=0, vmax=1,
                       extent=[M_A_vals.min(), M_A_vals.max(),
                               -0.5, len(noise_vals) - 0.5])

        ax.set_yticks(range(len(noise_vals)))
        ax.set_yticklabels([str(s) for s in noise_vals], fontsize=9)
        ax.set_xlabel('$M_A$', fontsize=12)
        ax.set_ylabel('$\\sigma_{Q,U}$', fontsize=12)
        ax.set_title(f'{geom.capitalize()} (iaxis={iaxis})', fontsize=12)
        plt.colorbar(im, ax=ax, label='Circular Variance (CV)')

    plt.suptitle('Circular Variance vs $M_A$ and $\\sigma_{Q,U}$', fontsize=14)
    plt.tight_layout()
    plt.savefig('sstat_heatmaps.png', dpi=150)
    plt.show()

def plot_recovery_vs_MA(results, SIGMA_VALS, M_A_VALS, INPUT_GEOMETRY, IAXIS):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [cm.plasma(i / len(SIGMA_VALS)) for i in range(len(SIGMA_VALS))]
    for i, sigma in enumerate(SIGMA_VALS):
        ax.plot(M_A_VALS, results[sigma] * 100,
                color=colors[i], lw=2, marker='o', ms=4,
                label=r'$\sigma_{{q,u}}={:.3f}$'.format(sigma))
    ax.axhline(50, color='gray', lw=1, ls=':')
    ax.set_xlabel(r'$M_A$', fontsize=14)
    ax.set_ylabel('Recovery Fraction (%)', fontsize=14)
    ax.set_title('{}, iaxis={}'.format(INPUT_GEOMETRY, IAXIS), fontsize=13)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig('/plots/'+str(INPUT_GEOMETRY)+'_recovery_vs_MA_by_noise.png', dpi=300)
    plt.show()

def plot_recovery_heatmap(results, SIGMA_VALS, M_A_VALS, INPUT_GEOMETRY, IAXIS, N_TRIALS):
    matrix = np.array([results[s] for s in SIGMA_VALS])
    fig, ax = plt.subplots(figsize=(8, 5))
    im      = ax.imshow(matrix * 100, aspect='auto', origin='upper', cmap='RdYlGn', vmin=0, vmax=100,
                        extent=[M_A_VALS.min(), M_A_VALS.max(), len(SIGMA_VALS) - 0.5, -0.5])
    plt.colorbar(im, ax=ax).set_label('Recovery Fraction (%)', fontsize=12)
    ax.set_yticks(range(len(SIGMA_VALS)))
    ax.set_yticklabels([r'$\sigma_{{q,u}}={:.3f}$'.format(s) for s in SIGMA_VALS], fontsize=9)
    ax.set_xlabel(r'$M_A$', fontsize=13)
    ax.set_title('Recovery Heatmap — {}'.format(INPUT_GEOMETRY, IAXIS, N_TRIALS), fontsize=13)
    typical = [i for i, s in enumerate(SIGMA_VALS) if 0.005 <= s <= 0.02]
    if typical:
        ymin, ymax = min(typical) - 0.5, max(typical) + 0.5
        ax.axhspan(ymin, ymax, hatch='////', fill=False, edgecolor='steelblue', lw=1.5, label='Typical POL-2 range')
        ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig('/plots'+str(INPUT_GEOMETRY)+'_recovery_heatmap.png', dpi=300)
    plt.show()

def plot_recovery_curves(recovery, M_A_vals, geometries, expected_output):

    colors = {
        'uniform'  : 'steelblue',
        'wavy'     : 'darkorange',
        'helical'  : 'forestgreen',
        'hourglass': 'crimson'}
    
    linestyles = {1: '--', 2: '-'}

    fig, ax = plt.subplots(figsize=(9, 5))

    for (geom, iaxis), rec_curve in recovery.items():
        label = f"{geometries[geom]['label']} (axis={iaxis})"
        ax.plot(M_A_vals, rec_curve * 100,
                color=colors[geom],
                ls=linestyles[iaxis],
                lw=2, marker='o', ms=4,
                label=label)

    ax.axhline(50, color='gray', lw=1, ls=':', label='50% chance')
    ax.axvline(1.0, color='k', lw=1, ls='--', alpha=0.4, label='$M_A = 1$')
    ax.set_xlabel('Alfvénic Mach Number $M_A$', fontsize=14)
    ax.set_ylabel('Recovery Fraction (%)', fontsize=14)
    ax.set_title('Decision Tree Recovery Fraction vs $M_A$', fontsize=15)
    ax.set_ylim(0, 105)
    ax.set_xlim(M_A_vals.min(), M_A_vals.max())
    ax.legend(fontsize=10, loc='lower left')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('recovery_fraction.png', dpi=150)
    plt.show()

def plot_confusion_matrix(confusion, all_labels, M_A_vals):
    # build ordered label list (only those that appeared)
    labels = sorted(set(
        list(confusion.keys()) +
        [p for preds in confusion.values() for p in preds.keys()]))

    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for i, true in enumerate(labels):
        for j, pred in enumerate(labels):
            matrix[i, j] = confusion[true].get(pred, 0)

    # normalize by row (true class) to get fraction
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid divide by zero
    matrix_norm = matrix / row_sums

    mid_MA = M_A_vals[len(M_A_vals) // 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix_norm, annot=True, fmt='.2f',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Predicted Geometry', fontsize=13)
    ax.set_ylabel('True Geometry', fontsize=13)
    ax.set_title(f'Confusion Matrix at $M_A \\approx {mid_MA:.2f}$', fontsize=14)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
