import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

def load_2d_casa_image(fpath):
    with fits.open(fpath) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    naxis = hdr["NAXIS"]
    ctypes = [hdr.get(f"CTYPE{i}", "") for i in range(1, naxis + 1)]
    sky_axes = [i for i, ct in enumerate(ctypes)
                if ct.upper().startswith(("RA", "DEC", "GLON", "GLAT"))]

    np_axes = [naxis - 1 - i for i in sky_axes]
    other_np_axes = [ax for ax in range(data.ndim) if ax not in np_axes]
    data2d = np.squeeze(data, axis=tuple(other_np_axes)).astype(np.float64)

    return data2d, hdr

def load_casa_stokes(ifile, qfile, ufile):
    I, hdr = load_2d_casa_image(ifile)
    Q, _ = load_2d_casa_image(qfile)
    U, _ = load_2d_casa_image(ufile)

    if not (I.shape == Q.shape == U.shape):
        raise ValueError(f"Shape mismatch: I={I.shape}, Q={Q.shape}, U={U.shape}")

    return I, Q, U, hdr


def estimate_rms(image, region=None, sigma=3.0, maxiters=5):
    data = image[region] if region is not None else image
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("No finite pixels available to estimate RMS.")
    _, _, std = sigma_clipped_stats(data, sigma=sigma, maxiters=maxiters)
    return float(std)


def compute_P_PHI(I, Q, U, sigma_I=None, sigma_Q=None, sigma_U=None, 
                  i_snr=5.0, p_snr=1.0, debias=True, angle_unit="radians", rms_region=None):

    # estimate noise if not provided
    if sigma_I is None:
        sigma_I = estimate_rms(I, region=rms_region)
    if sigma_Q is None:
        sigma_Q = estimate_rms(Q, region=rms_region)
    if sigma_U is None:
        sigma_U = estimate_rms(U, region=rms_region)

    # calculate P and PHI
    # Treatment following: (Serkowski 1958; Wardle & Kronberg 1974; Naghizadeh-Khouei & Clarke 1993; Vaillancourt 2006)
    with np.errstate(invalid="ignore", divide="ignore"):
        P_obs = np.sqrt(Q**2 + U**2) / I
        PHI_obs = 0.5 * np.arctan2(U, Q)
        sigma_QU = 0.5*(sigma_Q + sigma_U) 
        sigma_P = sigma_QU / np.abs(I)
        sigma_PHI = sigma_P/(2*P_obs)

    # add option for degrees
    if angle_unit == "degrees":
        PHI_obs = np.degrees(PHI_obs)
    elif angle_unit != "radians":
        raise ValueError("angle_unit must be 'radians' or 'degrees'")

    # Rice debiasing -- Wardle&Kronberg 1974 = P' = √(Q² + U² − ½(σ_Q² + σ_U²))
    if debias:
        P = np.where(P_obs > sigma_P, np.sqrt(np.clip(P_obs**2 - sigma_P**2, 0, None)), np.nan)
    else:
        P = P_obs

    # SNR masking
    mask = np.isfinite(I) & (I > i_snr * sigma_I)
    if p_snr is not None:
        mask &= P > (p_snr * sigma_P)

    P = np.where(mask, P, np.nan)
    PHI = np.where(mask, PHI_obs, np.nan)

    return P, PHI, sigma_P, sigma_PHI

def fit_powerlaw(P, I):
    valid = np.isfinite(P) & np.isfinite(I) & (P > 0) & (I > 0)
    Pv, Iv = P[valid], I[valid]

    lo, hi = Iv.min(), Iv.max()

    I_ref = np.median(Iv)
    x = np.log(Iv / I_ref)
    y = np.log(Pv)

    A = np.vstack([x, np.ones_like(x)]).T 
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha = -slope
    p0 = np.exp(intercept)

    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return alpha, p0, I_ref, r2, lo, hi
 
 