"""
fetch_polarimetry.py
====================
Downloads polarization data for two targets:
  1. GRSMC 45.60+0.30  -- GPIPS DR4 NIR stellar polarimetry (IRSA / astroquery)
                          Cuts faithfully reproduce Marchwinski+2012 sample:
                            - 2MASS-matched stars (desig_2mass not empty)
                            - H-band magnitude < 12.5
                            - SP < 2% (DR4 high-quality threshold)
                            - P/SP >= 2 (2-sigma detection)
                            - Spatially within cloud boundary
                            - Cloud 2 contamination zone excluded

  2. L1544             -- SCUPOL 850 um dust polarimetry (Matthews+2009, VizieR)

Requirements:
    pip install astropy astroquery

Outputs (in ./pol_data/):
    gpips_GRSMC45.60+0.30.fits
    gpips_GRSMC45.60+0.30.csv
    scupol_L1544.fits
    scupol_L1544.csv
"""

import os
import warnings
import numpy as np
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
from astropy.table import Table
from astroquery.irsa import Irsa
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

OUTDIR = "./pol_data"
os.makedirs(OUTDIR, exist_ok=True)


def circular_pa_stats(pa_deg):
    """
    Circular mean and std for headless position angles on [0, 180].
    Shifts to [-90, 90], doubles to map onto full circle [-180, 180],
    computes circular stats, then halves and shifts back.
    """
    pa = pa_deg[~np.isnan(pa_deg)]
    pa_shifted = pa - 90.0                              # [0,180] -> [-90, 90]
    pa2 = np.deg2rad(2.0 * pa_shifted)                 # double -> full circle
    S = np.mean(np.sin(pa2))
    C = np.mean(np.cos(pa2))
    mean_pa = np.rad2deg(np.arctan2(S, C)) / 2.0 + 90.0   # halve and shift back
    R = np.sqrt(S**2 + C**2)
    std_pa = np.rad2deg(np.sqrt(-2.0 * np.log(np.clip(R, 1e-10, 1.0)))) / 2.0
    return mean_pa, std_pa


# =============================================================================
# 1.  GPIPS DR4  --  GRSMC 45.60+0.30
# =============================================================================
def fetch_gpips(outdir=OUTDIR):
    print("\n" + "=" * 60)
    print("Fetching GPIPS DR4 polarimetry for GRSMC 45.60+0.30")
    print("Applying Marchwinski+2012 selection criteria")
    print("=" * 60)

    gal  = SkyCoord(l=45.6 * u.deg, b=0.3 * u.deg, frame=Galactic)
    icrs = gal.icrs
    print("  Cloud centre: l=45.6, b=0.3  ->  RA={:.4f}, Dec={:.4f}".format(
        icrs.ra.deg, icrs.dec.deg))

    radius = 0.6 * u.deg
    print("  Initial cone search radius: {}".format(radius))
    print("  Querying IRSA catalog: gpipsdr4 ...")

    Irsa.ROW_LIMIT = -1
    result = Irsa.query_region(
        coordinates=icrs,
        catalog="gpipsdr4",
        spatial="Cone",
        radius=radius,
    )

    if len(result) == 0:
        print("  !! No GPIPS sources returned.")
        return None

    print("  Retrieved {} stars in cone before any cuts.".format(len(result)))

    # Extract arrays
    p_arr       = np.ma.filled(np.array(result["p"]).astype(float),      np.nan)
    sp_arr      = np.ma.filled(np.array(result["sp"]).astype(float),     np.nan)
    hmag_arr    = np.ma.filled(np.array(result["h_mag"]).astype(float),  np.nan)
    gal_l_arr   = np.ma.filled(np.array(result["gal_l"]).astype(float),  np.nan)
    gal_b_arr   = np.ma.filled(np.array(result["gal_b"]).astype(float),  np.nan)
    desig_2mass = np.array(result["desig_2mass"]).astype(str)

    # Cut 1: 2MASS-matched stars only
    mask_2mass = np.array([
        s.strip() not in ("", "nan", "--", "b''", "b'--'")
        for s in desig_2mass
    ])
    print("  Cut 1 -- 2MASS matched:              {:>6d} stars".format(mask_2mass.sum()))

    # Cut 2: H-band magnitude < 12.5
    mask_hmag = hmag_arr < 12.5
    print("  Cut 2 -- H_mag < 12.5:               {:>6d} stars".format(
        (mask_2mass & mask_hmag).sum()))

    # Cut 3: SP < 2% (DR4 high-quality threshold)
    mask_sp = sp_arr < 2.0
    print("  Cut 3 -- SP < 2%:                    {:>6d} stars".format(
        (mask_2mass & mask_hmag & mask_sp).sum()))

    # Cut 4: P/SP >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(sp_arr > 0, p_arr / sp_arr, 0.0)
    mask_snr = snr >= 2.0
    print("  Cut 4 -- P/SP >= 2:                  {:>6d} stars".format(
        (mask_2mass & mask_hmag & mask_sp & mask_snr).sum()))

    # Cut 5: Spatial boundary of cloud (approximate GRS CO extent)
    mask_spatial = (
        (gal_l_arr >= 45.0) & (gal_l_arr <= 46.2)
        & (gal_b_arr >= -0.25) & (gal_b_arr <= 0.85)
    )
    print("  Cut 5 -- Cloud spatial boundary:     {:>6d} stars".format(
        (mask_2mass & mask_hmag & mask_sp & mask_snr & mask_spatial).sum()))

    # Cut 6: Exclude Cloud 2 contamination zone (Marchwinski+2012 Figure 4)
    mask_cloud2 = ~((gal_l_arr > 45.7) & (gal_b_arr < 0.15))
    mask_all = mask_2mass & mask_hmag & mask_sp & mask_snr & mask_spatial & mask_cloud2
    print("  Cut 6 -- Cloud 2 exclusion zone:     {:>6d} stars".format(mask_all.sum()))

    filtered = result[mask_all]
    print("\n  Final sample (all cuts):             {:>6d} stars".format(len(filtered)))
    print("  (Marchwinski+2012 reported 2684 stars)")

    # Select columns
    keep_cols = [
        "ra", "dec", "gal_l", "gal_b",
        "h_mag", "shmag",
        "h_2mass", "k_2mass", "j_2mass",
        "p", "sp", "pa_deg", "gpa_deg", "spa",
        "q", "sq", "u", "su",
        "nhwp", "uf",
    ]
    keep_cols = [c for c in keep_cols if c in filtered.colnames]
    out = filtered[keep_cols]

    out.meta["TARGET"]     = "GRSMC 45.60+0.30"
    out.meta["CATALOG"]    = "GPIPS DR4 (Clemens+2020)"
    out.meta["SELECTION"]  = "Marchwinski+2012 criteria"
    out.meta["CENTRE_L"]   = 45.6
    out.meta["CENTRE_B"]   = 0.3
    out.meta["CENTRE_RA"]  = round(icrs.ra.deg, 4)
    out.meta["CENTRE_DEC"] = round(icrs.dec.deg, 4)
    out.meta["CUT_HMAG"]   = "< 12.5"
    out.meta["CUT_SP"]     = "< 2.0 %"
    out.meta["CUT_SNR"]    = ">= 2.0"
    out.meta["WAVELENGTH"] = "1.6 um H-band NIR"
    out.meta["POL_TYPE"]   = "stellar background (plane-of-sky B-field)"
    out.meta["REFERENCE"]  = "Marchwinski+2012 ApJ 755 130; Clemens+2020 ApJS 249 23"

    if len(out) == 0:
        print("  WARNING: Output table is empty -- nothing saved.")
        return None

    p_vals   = np.ma.filled(np.array(out["p"]).astype(float),      np.nan)
    gpa_vals = np.ma.filled(np.array(out["gpa_deg"]).astype(float), np.nan)
    circ_mean, circ_std = circular_pa_stats(gpa_vals)

    print("\n  Summary (Marchwinski+2012 sample):")
    print("    N stars          = {}".format(len(out)))
    print("    Mean   p         = {:.2f} %  (paper: 1.8 +/- 0.6 %)".format(np.nanmean(p_vals)))
    print("    Median p         = {:.2f} %".format(np.nanmedian(p_vals)))
    print("    Max    p         = {:.2f} %".format(np.nanmax(p_vals)))
    print("    Circular mean GPA= {:.1f} deg  (paper: ~75-85 deg)".format(circ_mean))
    print("    Circular std  GPA= {:.1f} deg  (paper: 15 +/- 2 deg)".format(circ_std))

    fits_path = os.path.join(outdir, "gpips_GRSMC45.60+0.30.fits")
    csv_path  = os.path.join(outdir, "gpips_GRSMC45.60+0.30.csv")
    out.write(fits_path, overwrite=True)
    out.write(csv_path,  format="csv", overwrite=True)
    print("\n  Saved FITS: {}".format(fits_path))
    print("  Saved CSV:  {}".format(csv_path))
    return out


# =============================================================================
# 2.  SCUPOL  --  L1544
#     Matthews et al. 2009, ApJS 182 143  ->  VizieR: J/ApJS/182/143
#     Actual VizieR column names: Pol, e_Pol, theta, e_theta, Int, e_Int
# =============================================================================
def fetch_scupol_l1544(outdir=OUTDIR):
    print("\n" + "=" * 60)
    print("Fetching SCUPOL 850 um polarimetry for L1544")
    print("=" * 60)

    l1544 = SkyCoord(ra="05h04m17.21s", dec="+25d10m42.8s", frame="icrs")
    print("  L1544:  RA={:.4f}, Dec={:.4f}".format(l1544.ra.deg, l1544.dec.deg))

    viz = Vizier(columns=["**"], row_limit=-1)
    print("  Querying VizieR catalog J/ApJS/182/143 (Matthews+2009 SCUPOL) ...")

    catalogs = viz.query_region(l1544, radius=5.0 * u.arcmin, catalog="J/ApJS/182/143")

    if catalogs is None or len(catalogs) == 0:
        print("  Cone search returned nothing -- trying full catalog download ...")
        catalogs = viz.get_catalogs("J/ApJS/182/143")

    if catalogs is None or len(catalogs) == 0:
        print("  !! Could not retrieve SCUPOL catalog from VizieR.")
        return None

    print("  Tables returned: {}".format(
        [t.meta.get("name", i) for i, t in enumerate(catalogs)]))

    pol_table = None
    for t in catalogs:
        if "Pol" in t.colnames or "e_Pol" in t.colnames:
            pol_table = t
            break
    if pol_table is None:
        pol_table = max(catalogs, key=lambda t: len(t))
        print("  No Pol column found; using largest table ({} rows).".format(len(pol_table)))
    else:
        print("  Found polarization table with {} rows.".format(len(pol_table)))

    print("  Available columns: {}".format(pol_table.colnames))

    if "Name" in pol_table.colnames:
        name_mask = np.array([
            "L1544" in str(n) or "l1544" in str(n).lower()
            for n in pol_table["Name"]
        ])
        if name_mask.sum() > 0:
            pol_table = pol_table[name_mask]
            print("  Filtered to L1544 rows: {}".format(len(pol_table)))
        else:
            print("  No L1544 match in Name column -- using cone search result as-is.")

    p_col  = "Pol"   if "Pol"   in pol_table.colnames else None
    dp_col = "e_Pol" if "e_Pol" in pol_table.colnames else None
    i_col  = "Int"   if "Int"   in pol_table.colnames else None
    pa_col = "theta" if "theta" in pol_table.colnames else None

    if p_col and dp_col:
        p_arr  = np.ma.filled(pol_table[p_col].data.astype(float),  np.nan)
        dp_arr = np.ma.filled(pol_table[dp_col].data.astype(float), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(dp_arr > 0, p_arr / dp_arr, 0.0)
        mask = (snr >= 2.0) & (dp_arr < 4.0) & (~np.isnan(p_arr))
        if i_col:
            i_arr = np.ma.filled(pol_table[i_col].data.astype(float), np.nan)
            mask &= i_arr > 0
        filtered = pol_table[mask]
        print("  After quality cuts (Pol/e_Pol>=2, e_Pol<4%): {} vectors retained.".format(
            len(filtered)))
    else:
        filtered = pol_table
        print("  Could not apply quality cut -- returning raw table ({} rows).".format(
            len(filtered)))

    filtered.meta["TARGET"]     = "L1544"
    filtered.meta["CATALOG"]    = "SCUPOL Legacy (Matthews+2009)"
    filtered.meta["CENTRE_RA"]  = round(l1544.ra.deg, 4)
    filtered.meta["CENTRE_DEC"] = round(l1544.dec.deg, 4)
    filtered.meta["WAVELENGTH"] = "850 um submillimetre"
    filtered.meta["BEAMSIZE"]   = "15 arcsec FWHM"
    filtered.meta["POL_TYPE"]   = "dust emission (plane-of-sky B-field)"
    filtered.meta["REFERENCE"]  = "Matthews et al. 2009, ApJS 182 143"

    if p_col and len(filtered) > 0:
        p_vals  = np.ma.filled(filtered[p_col].data.astype(float), np.nan)
        pa_vals = np.ma.filled(filtered[pa_col].data.astype(float), np.nan) if pa_col else None
        print("\n  Summary of Pol (polarization %):")
        print("    N vectors = {}".format(len(filtered)))
        print("    Mean   P  = {:.2f} %".format(np.nanmean(p_vals)))
        print("    Median P  = {:.2f} %".format(np.nanmedian(p_vals)))
        print("    Max    P  = {:.2f} %".format(np.nanmax(p_vals)))
        if pa_vals is not None:
            circ_mean, circ_std = circular_pa_stats(pa_vals)
            print("    Circular mean PA = {:.1f} deg".format(circ_mean))
            print("    Circular std  PA = {:.1f} deg".format(circ_std))

    fits_path = os.path.join(outdir, "scupol_L1544.fits")
    csv_path  = os.path.join(outdir, "scupol_L1544.csv")
    filtered.write(fits_path, overwrite=True)
    filtered.write(csv_path, format="csv", overwrite=True)
    print("\n  Saved FITS: {}".format(fits_path))
    print("  Saved CSV:  {}".format(csv_path))
    return filtered


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Polarimetry data downloader")
    print("  Target 1: GRSMC 45.60+0.30  ->  GPIPS DR4 (Marchwinski+2012 cuts)")
    print("  Target 2: L1544             ->  SCUPOL (VizieR Matthews+2009)")

    gpips_data  = fetch_gpips()
    scupol_data = fetch_scupol_l1544()

    print("\n" + "=" * 60)
    print("Done. Output files written to ./pol_data/")
    print("=" * 60)
    if gpips_data is not None:
        print("  GPIPS  : {} stars   -> gpips_GRSMC45.60+0.30.fits/csv".format(len(gpips_data)))
    if scupol_data is not None:
        print("  SCUPOL : {} vectors -> scupol_L1544.fits/csv".format(len(scupol_data)))
    print()
    print("Column reference:")
    print("  GPIPS  -- p: debiased pol %, sp: uncertainty, pa_deg: pos angle E of N")
    print("            gpa_deg: Galactic pos angle (use this for B-field analysis)")
    print("            q/u: Stokes params (% of I)")
    print("            j/h/k_2mass: 2MASS photometry for reddening analysis")
    print("  SCUPOL -- Pol: pol %, e_Pol: uncertainty, theta: pos angle E of N")
    print('            Int: total intensity, all at 850 um / 15" beam')