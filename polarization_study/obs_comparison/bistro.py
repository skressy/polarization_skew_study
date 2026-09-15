"""
bistro_validation.py
====================
Validation workflow for skewness-based M_A diagnostic using BISTRO Pol-2 data.

Steps:
  1. Define target sample with literature M_A and geometry classifications
  2. Fetch 850 um polarimetry for each target from VizieR (BISTRO/SCUPOL papers)
  3. Apply standard quality cuts (P/dP >= 2, dP < 4%, I > 0)
  4. Compute skewness of normalized P/<P> for each target
  5. Output summary table ready for comparison with simulation skewness curve

Targets selected to span M_A ~ 0.1 to 2.0 with known geometry:
  - rho Oph A     : M_A ~ 0.38, ordered/uniform field (HAWC+, Ngoc+2024)
  - OMC-1         : M_A ~ 0.40, hourglass geometry (BISTRO, Hwang+2021)
  - L1689         : M_A ~ 0.5,  ordered perpendicular to filament (BISTRO, Coude+2019)
  - IC 5146       : M_A ~ 0.5,  hub-filament ordered (BISTRO, Wang+2019)
  - L1544         : M_A ~ 0.5,  hourglass (SCUPOL, Ward-Thompson+2000)
  - NGC 1333      : M_A ~ 1.0,  complex/turbulent (BISTRO, Doi+2020)

Requirements:
    pip install astropy astroquery scipy

Outputs (in ./pol_data/bistro/):
    <target>_pol.fits / .csv     -- quality-filtered polarization vectors
    validation_summary.csv       -- one row per target: name, geometry, M_A, N, skewness
"""

import os
import warnings
import numpy as np
from scipy.stats import skew
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

OUTDIR = "./pol_data/bistro"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# Target sample
# Literature M_A and geometry from published BISTRO / SCUPOL CF-method papers.
# Geometry codes:
#   uniform   -- large-scale ordered, low dispersion
#   hourglass -- pinched geometry around dense core
#   filament  -- ordered along or perpendicular to filament spine
#   complex   -- multi-component, high dispersion, turbulent
# =============================================================================
TARGETS = [
    {
        "name"     : "rho_Oph_A",
        "display"  : "rho Oph A",
        "ra"       : "16h26m27s",
        "dec"      : "-24d24m00s",
        "radius_am": 8.0,           # arcmin search radius
        "M_A"      : 0.38,
        "M_A_ref"  : "Ngoc+2024 A&A (HAWC+)",
        "geometry" : "uniform",
        "vizier_cat": "J/ApJ/908/10",   # Kwon+2021 BISTRO rho Oph
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "I",
    },
    {
        "name"     : "OMC1",
        "display"  : "OMC-1",
        "ra"       : "05h35m17s",
        "dec"      : "-05d22m30s",
        "radius_am": 8.0,
        "M_A"      : 0.40,
        "M_A_ref"  : "Hwang+2021 ApJ (BISTRO)",
        "geometry" : "hourglass",
        "vizier_cat": "J/ApJ/913/128",  # Hwang+2021 BISTRO OMC-1
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "I",
    },
    {
        "name"     : "L1689",
        "display"  : "L1689",
        "ra"       : "16h32m29s",
        "dec"      : "-24d28m52s",
        "radius_am": 8.0,
        "M_A"      : 0.50,
        "M_A_ref"  : "Coude+2019 ApJ (BISTRO)",
        "geometry" : "filament",
        "vizier_cat": "J/ApJ/877/88",   # Coude+2019 BISTRO L1689
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "I",
    },
    {
        "name"     : "IC5146",
        "display"  : "IC 5146",
        "ra"       : "21h53m24s",
        "dec"      : "+47d16m00s",
        "radius_am": 10.0,
        "M_A"      : 0.50,
        "M_A_ref"  : "Wang+2019 ApJ (BISTRO)",
        "geometry" : "filament",
        "vizier_cat": "J/ApJ/876/42",   # Wang+2019 BISTRO IC5146
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "I",
    },
    {
        "name"     : "L1544",
        "display"  : "L1544",
        "ra"       : "05h04m17.21s",
        "dec"      : "+25d10m42.8s",
        "radius_am": 5.0,
        "M_A"      : 0.50,
        "M_A_ref"  : "Ward-Thompson+2000 (SCUPOL)",
        "geometry" : "hourglass",
        "vizier_cat": "J/ApJS/182/143",  # Matthews+2009 SCUPOL legacy
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "Int",
    },
    {
        "name"     : "NGC1333",
        "display"  : "NGC 1333",
        "ra"       : "03h29m00s",
        "dec"      : "+31d16m00s",
        "radius_am": 10.0,
        "M_A"      : 1.00,
        "M_A_ref"  : "Doi+2020 ApJ (BISTRO)",
        "geometry" : "complex",
        "vizier_cat": "J/ApJ/899/28",   # Doi+2020 BISTRO NGC1333
        "p_col"    : "Pol",
        "dp_col"   : "e_Pol",
        "pa_col"   : "theta",
        "i_col"    : "I",
    },
]


# =============================================================================
# Helpers
# =============================================================================
def circular_pa_stats(pa_deg):
    """Circular mean and std for headless PA on [0, 180]."""
    pa = pa_deg[~np.isnan(pa_deg)]
    if len(pa) == 0:
        return np.nan, np.nan
    pa2 = np.deg2rad(2.0 * (pa - 90.0))
    S, C = np.mean(np.sin(pa2)), np.mean(np.cos(pa2))
    mean_pa = np.rad2deg(np.arctan2(S, C)) / 2.0 + 90.0
    R = np.sqrt(S**2 + C**2)
    std_pa = np.rad2deg(np.sqrt(-2.0 * np.log(np.clip(R, 1e-10, 1.0)))) / 2.0
    return mean_pa, std_pa


def compute_skewness(p_vals):
    """
    Skewness of normalized P/<P>.
    Normalization removes sensitivity to absolute polarization efficiency,
    making simulations and observations directly comparable.
    Returns (skewness, N) or (nan, 0) if insufficient data.
    """
    p = p_vals[~np.isnan(p_vals)]
    if len(p) < 8:
        print("    WARNING: fewer than 8 vectors -- skewness unreliable.")
        return np.nan, len(p)
    p_mean = np.mean(p)
    if p_mean <= 0:
        return np.nan, len(p)
    p_norm = p / p_mean       # normalize: P / <P>
    return float(skew(p_norm, bias=False)), len(p)


def quality_filter(table, p_col, dp_col, i_col, snr_cut=2.0, dp_cut=4.0):
    """
    Standard BISTRO/SCUPOL quality cuts:
      P/dP >= snr_cut
      dP   <  dp_cut %
      I    >  0  (if i_col present)
    Returns filtered table.
    """
    p_arr  = np.ma.filled(table[p_col].data.astype(float),  np.nan)
    dp_arr = np.ma.filled(table[dp_col].data.astype(float), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(dp_arr > 0, p_arr / dp_arr, 0.0)
    mask = (snr >= snr_cut) & (dp_arr < dp_cut) & (~np.isnan(p_arr))
    if i_col and i_col in table.colnames:
        i_arr = np.ma.filled(table[i_col].data.astype(float), np.nan)
        mask &= i_arr > 0
    return table[mask]


# =============================================================================
# Main fetch + analysis loop
# =============================================================================
def fetch_and_analyse(target, viz):
    name     = target["name"]
    display  = target["display"]
    coord    = SkyCoord(ra=target["ra"], dec=target["dec"], unit=(u.hourangle, u.deg))
    radius   = target["radius_am"] * u.arcmin
    cat      = target["vizier_cat"]
    p_col    = target["p_col"]
    dp_col   = target["dp_col"]
    pa_col   = target["pa_col"]
    i_col    = target["i_col"]

    print("\n" + "-" * 55)
    print("Target: {}  (catalog: {})".format(display, cat))
    print("  RA={:.4f}, Dec={:.4f}, radius={}'".format(
        coord.ra.deg, coord.dec.deg, target["radius_am"]))

    # Query VizieR
    try:
        catalogs = viz.query_region(coord, radius=radius, catalog=cat)
    except Exception as e:
        print("  !! VizieR query failed: {}".format(e))
        return None

    if catalogs is None or len(catalogs) == 0:
        print("  !! No tables returned from VizieR.")
        return None

    print("  Tables returned: {}".format(
        [t.meta.get("name", i) for i, t in enumerate(catalogs)]))

    # Find polarization table
    pol_table = None
    for t in catalogs:
        if p_col in t.colnames and dp_col in t.colnames:
            pol_table = t
            break
    if pol_table is None:
        # Fall back to largest table and print columns for diagnosis
        pol_table = max(catalogs, key=lambda t: len(t))
        print("  Expected columns ({}, {}) not found.".format(p_col, dp_col))
        print("  Available columns: {}".format(pol_table.colnames))
        print("  Skipping quality filter -- check column names for this target.")
        skewness, N = np.nan, 0
        filtered = pol_table
    else:
        print("  Found pol table: {} rows, columns: {}".format(
            len(pol_table), pol_table.colnames))

        # Quality filter
        filtered = quality_filter(pol_table, p_col, dp_col, i_col)
        print("  After quality cuts: {} vectors retained.".format(len(filtered)))

        # Compute skewness
        if len(filtered) > 0 and p_col in filtered.colnames:
            p_vals = np.ma.filled(filtered[p_col].data.astype(float), np.nan)
            skewness, N = compute_skewness(p_vals)

            # PA stats
            if pa_col and pa_col in filtered.colnames:
                pa_vals = np.ma.filled(filtered[pa_col].data.astype(float), np.nan)
                circ_mean, circ_std = circular_pa_stats(pa_vals)
            else:
                circ_mean, circ_std = np.nan, np.nan

            print("  Summary:")
            print("    N vectors        = {}".format(N))
            print("    Mean P           = {:.2f} %".format(np.nanmean(p_vals)))
            print("    Skewness(P/<P>)  = {:.3f}".format(skewness) if not np.isnan(skewness) else "    Skewness = nan (insufficient data)")
            print("    Circular mean PA = {:.1f} deg".format(circ_mean))
            print("    Circular std  PA = {:.1f} deg".format(circ_std))
        else:
            skewness, N = np.nan, 0

    # Save per-target files
    fits_path = os.path.join(OUTDIR, "{}_pol.fits".format(name))
    csv_path  = os.path.join(OUTDIR, "{}_pol.csv".format(name))
    try:
        filtered.write(fits_path, overwrite=True)
        filtered.write(csv_path,  format="csv", overwrite=True)
        print("  Saved: {}_pol.fits/csv".format(name))
    except Exception as e:
        print("  WARNING: Could not save files: {}".format(e))

    return {
        "name"     : display,
        "geometry" : target["geometry"],
        "M_A"      : target["M_A"],
        "M_A_ref"  : target["M_A_ref"],
        "N_vectors": N,
        "skewness" : skewness,
        "catalog"  : cat,
    }


def run_validation():
    print("=" * 55)
    print("BISTRO Polarimetry Validation")
    print("=" * 55)

    viz = Vizier(columns=["**"], row_limit=-1)
    results = []

    for target in TARGETS:
        row = fetch_and_analyse(target, viz)
        if row is not None:
            results.append(row)

    # Build and save summary table
    if len(results) == 0:
        print("\n!! No results to summarize.")
        return None

    summary = Table(
        rows=results,
        names=["name", "geometry", "M_A", "M_A_ref", "N_vectors", "skewness", "catalog"]
    )

    # Sort by M_A for easy reading
    summary.sort("M_A")

    summary_path = os.path.join(OUTDIR, "validation_summary.csv")
    summary.write(summary_path, format="csv", overwrite=True)

    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    print("{:<18} {:<12} {:>6} {:>8} {:>10}".format(
        "Target", "Geometry", "M_A", "N_vec", "Skewness"))
    print("-" * 55)
    for row in summary:
        sk = "{:.3f}".format(row["skewness"]) if not np.isnan(row["skewness"]) else "  nan"
        print("{:<18} {:<12} {:>6.2f} {:>8d} {:>10}".format(
            row["name"], row["geometry"], row["M_A"], row["N_vectors"], sk))
    print("-" * 55)
    print("\nNote: skewness computed on normalized P/<P> for direct")
    print("comparison with simulation skewness curves.")
    print("\nIf skewness increases monotonically with M_A across")
    print("these targets, your decision tree method is observationally")
    print("validated. If not, skewness is geometry-dependent and the")
    print("geometry classification step must come first.")
    print("\nSummary saved to: {}".format(summary_path))
    return summary


# =============================================================================
# Convenience: load already-fetched data without re-querying
# =============================================================================
def load_target(name):
    """
    Load a previously fetched target's FITS file.
    Usage: tbl = load_target('OMC1')
    """
    path = os.path.join(OUTDIR, "{}_pol.fits".format(name))
    if not os.path.exists(path):
        print("File not found: {}".format(path))
        print("Run run_validation() first.")
        return None
    return Table.read(path)


def recompute_skewness(name, p_col="Pol"):
    """
    Recompute skewness for a loaded target -- useful for testing
    different quality cuts without re-fetching from VizieR.
    Usage: sk, N = recompute_skewness('OMC1')
    """
    tbl = load_target(name)
    if tbl is None:
        return np.nan, 0
    p_vals = np.ma.filled(tbl[p_col].data.astype(float), np.nan)
    sk, N = compute_skewness(p_vals)
    print("{}: skewness = {:.3f}, N = {}".format(name, sk, N))
    return sk, N


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    summary = run_validation()