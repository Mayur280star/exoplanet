"""
exo_siamese_step1-5.py

A single-file runnable pipeline that implements Steps 1-5 from the plan:
- Download KOI/TCE labels (NASA Exoplanet Archive) using astroquery
- Download Kepler/TESS light curves using lightkurve
- Preprocess: mask, detrend/flatten, normalize, outlier removal
- Phase-fold and create Global & Local views per candidate
- Build pair & triplet datasets (anchor/positive/negative) and save as .npz

Usage (recommended within a virtualenv):
1) Create env & install:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install lightkurve astroquery numpy pandas scipy tqdm requests

2) Run (example downloads a small sample first):
   python exo_siamese_step1-5.py --mission Kepler --n_candidates 200

Notes:
- This script downloads real light curves from MAST. For large runs use an institutional connection and/or download in batches.
- The script is robust with retries and caching; it saves raw FITS and processed arrays to data/.
- Adjust window_length, global_len, local_len, and other params via CLI args.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Astronomy libs
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Scientific
from scipy import interpolate

# ----------------------------
# Config / Defaults
# ----------------------------
DEFAULTS = {
    "data_dir": "data",
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "pairs_dir": "data/pairs",
    "mission": "Kepler",
    "n_candidates": 500,
    "global_len": 2001,
    "local_len": 201,
    "local_n_durations": 3.0,  # +/- n transit durations
    "flatten_window": 401,
    "outlier_sigma": 5.0,
}

# ----------------------------
# Utilities
# ----------------------------

def ensure_dirs(cfg):
    for p in [cfg["raw_dir"], cfg["interim_dir"], cfg["pairs_dir"]]:
        Path(p).mkdir(parents=True, exist_ok=True)


def safe_download_lc(target, mission="Kepler", cache_dir="data/raw", max_tries=3):
    """Download a lightcurve file using lightkurve, cache by target-mission.
    target: Kepler ID / TIC / target name
    Returns LightCurve object or None if failed.
    """
    fname = Path(cache_dir) / f"{mission}_{str(target)}.fits"
    if fname.exists():
        try:
            lcfile = lk.read(fname)
            # If it's a LightCurveFile, try to return PDCSAP_FLUX when present
            if hasattr(lcfile, "PDCSAP_FLUX"):
                return lcfile.PDCSAP_FLUX
            # if it's a LightCurve (rare) return it directly
            return lcfile
        except Exception:
            # fallback to re-download if file is corrupted
            pass

    for attempt in range(1, max_tries + 1):
        try:
            search = lk.search_lightcurvefile(target, mission=mission)
            if len(search) == 0:
                return None
            # download the first search result (most common)
            lcf = search.download(quality_bitmask="default")
            # write to cache if possible
            try:
                lcf.to_fits(path=str(fname), overwrite=True)
            except Exception:
                pass
            # return PDCSAP when available, else raw LightCurveFile
            try:
                return lcf.PDCSAP_FLUX
            except Exception:
                return lcf
        except Exception:
            if attempt == max_tries:
                return None
            time.sleep(2 * attempt)
    return None


# ----------------------------
# Step 1: Labels & ephemerides
# ----------------------------

def fetch_koi_table(limit=None):
    """
    Fetch KOI (Kepler Objects of Interest) table via astroquery using the current API.
    Returns pandas DataFrame with selected columns.
    """
    print("Fetching KOI table from NASA Exoplanet Archive (this may take a moment)...")

    # Attempt the commonly used KOI table names in order
    table_candidates = ["cumulative", "q1_q17_dr25_koi"]

    last_err = None
    for tname in table_candidates:
        try:
            koi_table = NasaExoplanetArchive.query_criteria(table=tname)
            if koi_table is not None and len(koi_table) > 0:
                koi_df = koi_table.to_pandas()
                print(f"Successfully fetched table '{tname}' with {len(koi_df)} rows.")
                break
        except Exception as e:
            last_err = e
            # try next candidate
            continue
    else:
        # none succeeded
        raise RuntimeError("Failed to fetch KOI table via astroquery: " + str(last_err))

    # Normalize column names robustly (some KOI tables use slightly different names)
    # We'll search for known column name variants.
    col_map_variants = {
        "kepid": ["kepid", "kepid"],
        "kepler_name": ["kepler_name", "kepoi_name", "kepoi_name"],
        "period": ["koi_period", "period", "kepoi_period"],
        "t0": ["koi_time0bk", "koi_time0bk", "koi_time0bk", "koi_time0bk"],
        "duration_hrs": ["koi_duration", "koi_duration_hrs", "koi_duration", "koi_duration"],
        "disposition": ["koi_disposition", "koi_pdisposition", "koi_disposition"],
        "depth_ppm": ["koi_depth", "koi_prad", "koi_depth"],
    }

    # build a mapping from existing columns to desired keys
    rename_map = {}
    for target, variants in col_map_variants.items():
        for v in variants:
            if v in koi_df.columns:
                rename_map[v] = target
                break

    df = koi_df.rename(columns=rename_map)

    # Keep only useful columns that are present
    keep_cols = [c for c in ["kepid", "kepler_name", "period", "t0", "duration_hrs", "disposition", "depth_ppm"] if c in df.columns]
    df = df[keep_cols]

    # We require at least kepid, period, t0, duration to proceed. If duration is not present, try to find other duration-like columns
    required = ["kepid", "period", "t0"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise RuntimeError(f"Required KOI columns missing from fetched table: {missing_required}")

    # If duration missing, try to infer or set a fallback
    if "duration_hrs" not in df.columns:
        # try to find any column with 'duration' in its name and map it
        duration_candidates = [c for c in koi_df.columns if "duration" in c.lower()]
        if duration_candidates:
            df = df.rename(columns={duration_candidates[0]: "duration_hrs"})
            if "duration_hrs" not in df.columns:
                df["duration_hrs"] = np.nan
        else:
            df["duration_hrs"] = np.nan

    # Drop rows missing period or t0 or kepid
    df = df.dropna(subset=[c for c in ["kepid", "period", "t0"] if c in df.columns])

    if limit is not None:
        df = df.head(limit)

    # convert duration to days if present and non-null
    if "duration_hrs" in df.columns:
        try:
            df["duration_days"] = df["duration_hrs"].astype(float) / 24.0
        except Exception:
            df["duration_days"] = pd.to_numeric(df["duration_hrs"], errors="coerce") / 24.0
    else:
        df["duration_days"] = np.nan

    # ensure kepid is integer where possible
    try:
        df["kepid"] = df["kepid"].astype(int)
    except Exception:
        # leave as-is if conversion fails
        pass

    print(f"Prepared KOI dataframe with {len(df)} rows (after filtering).")
    return df


# ----------------------------
# Step 2-4: Download + preprocess + fold + window
# ----------------------------

def preprocess_lc(lc, flatten_window=401, outlier_sigma=5.0):
    """Given a lightkurve.LightCurve, perform cleaning: remove NaNs, outliers, flatten.
    Return flattened normalized LightCurve (LightCurve object or similar).
    """
    # remove NaNs
    try:
        lc = lc.remove_nans()
    except Exception:
        pass
    # remove outliers
    try:
        lc = lc.remove_outliers(sigma=outlier_sigma)
    except Exception:
        pass
    # normalize
    try:
        lc = lc.normalize()
    except Exception:
        # fallback
        try:
            lc.flux = lc.flux / np.nanmedian(lc.flux)
        except Exception:
            pass
    # flatten (detrend)
    try:
        flat = lc.flatten(window_length=flatten_window)
    except Exception:
        # fallback to a median filter
        times = lc.time.value
        kernel = int(min(len(times) // 5, flatten_window))
        if kernel % 2 == 0:
            kernel += 1
        try:
            from scipy.signal import medfilt
            flat_flux = medfilt(lc.flux, kernel_size=max(3, kernel))
            flat = lc.copy()
            flat.flux = lc.flux / (flat_flux + 1e-12)
        except Exception:
            flat = lc
    return flat


def make_global_local_views(lc_flat, period, t0, duration_days, global_len=2001, local_len=201, local_n_durations=3.0):
    """Produce global & local phase views from a flattened LightCurve.
    Returns (global_array, local_array) as numpy float32 arrays with NaNs filled via interpolation.
    lc_flat: LightCurve with time in BKJD or appropriate time; period in days, t0 in BKJD (or same units)
    """
    # phase fold (lightkurve has fold method)
    try:
        folded = lc_flat.fold(period=period, t0=t0)
        phase = folded.phase.value  # in cycles, -0.5..0.5
        flux = folded.flux.value
    except Exception:
        # manual fold
        t = lc_flat.time.value
        phase = ((t - t0) / period + 0.5) % 1.0 - 0.5
        flux = lc_flat.flux.value

    # create global grid
    ggrid = np.linspace(-0.5, 0.5, global_len)
    try:
        ginterp = interpolate.interp1d(phase, flux, kind="linear", bounds_error=False, fill_value=np.nan)
        gvals = ginterp(ggrid)
    except Exception:
        gvals = np.full(global_len, np.nan)

    # local window
    half_phase = local_n_durations * (duration_days / period) if (duration_days is not None and not np.isnan(duration_days) and period != 0) else None
    if (half_phase is None) or (half_phase <= 0) or (np.isnan(half_phase)) or (half_phase > 0.25):
        half_phase = 0.03  # fallback default
    lgrid = np.linspace(-half_phase, half_phase, local_len)
    try:
        linterp = interpolate.interp1d(phase, flux, kind="linear", bounds_error=False, fill_value=np.nan)
        lvals = linterp(lgrid)
    except Exception:
        lvals = np.full(local_len, np.nan)

    # fill NaNs by nearest valid value (simple and robust)
    def fill_nans(arr):
        arr = np.array(arr, dtype=float)
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)
        nans = np.isnan(arr)
        if np.any(nans):
            idx = np.arange(len(arr))
            good = ~nans
            if np.sum(good) == 0:
                return np.zeros_like(arr)
            f = interpolate.interp1d(idx[good], arr[good], kind="nearest", bounds_error=False, fill_value=(arr[good][0], arr[good][-1]))
            arr = f(idx)
        return arr

    gvals = fill_nans(gvals)
    lvals = fill_nans(lvals)

    # standard normalization: subtract median, divide by robust std (MAD)
    def robust_norm(x):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if mad == 0 or np.isnan(mad):
            mad = np.nanstd(x) + 1e-9
        return (x - med) / (mad * 1.4826)

    gnorm = robust_norm(gvals).astype(np.float32)
    lnorm = robust_norm(lvals).astype(np.float32)

    return gnorm, lnorm


# ----------------------------
# Step 5: Build Pairs / Triplets
# ----------------------------

def build_pair_triplet_indices(df_meta, positive_dispositions=("CANDIDATE", "CONFIRMED"), negative_dispositions=("FALSE POSITIVE",)):
    """Given metadata DataFrame with 'kepid' and 'disposition', create mapping lists of positives and negatives.
    Returns dict: {'pos_ids': [...], 'neg_ids': [...]} using kepid integers
    """
    if "disposition" not in df_meta.columns:
        raise RuntimeError("Meta dataframe does not include a 'disposition' column.")
    disp = df_meta["disposition"].astype(str).str.upper()
    pos_mask = disp.isin([d.upper() for d in positive_dispositions])
    neg_mask = disp.isin([d.upper() for d in negative_dispositions])

    pos_ids = df_meta.loc[pos_mask, "kepid"].unique().tolist()
    neg_ids = df_meta.loc[neg_mask, "kepid"].unique().tolist()
    return {"pos_ids": pos_ids, "neg_ids": neg_ids}


def sample_triplets_from_meta(meta_df, cached_views, n_triplets=10000, random_state=42):
    """Create triplets (anchor, positive, negative) from metadata + cached views dictionary
    cached_views: dict mapping kepid -> list of candidate dicts each with keys: {'period','t0','duration','global','local','candidate_id'}
    Returns a list of triplet tuples (a, p, n)
    """
    rng = np.random.RandomState(random_state)
    # first create pools of candidate entries labeled pos/neg
    pos_entries = []
    neg_entries = []
    for kepid, candidates in cached_views.items():
        for c in candidates:
            disp = str(c.get("disposition", "UNKNOWN")).upper()
            if disp in ("CANDIDATE", "CONFIRMED"):
                pos_entries.append(c)
            elif disp in ("FALSE POSITIVE",):
                neg_entries.append(c)
            else:
                # unknown dispositions can be added to negative pool (or skip) — here we add to neg pool
                neg_entries.append(c)

    if len(pos_entries) == 0 or len(neg_entries) == 0:
        raise RuntimeError("Not enough positive or negative examples to build triplets. Make sure your metadata contains candidates with dispositions.")

    # build triplets by drawing anchor from pos_entries and positive another pos (same or different star) and negative from neg_entries
    triplets = []
    for _ in range(n_triplets):
        a = rng.choice(pos_entries)
        # choose positive: prefer different candidate_id
        p_candidates = [x for x in pos_entries if x["candidate_id"] != a["candidate_id"]]
        p = rng.choice(p_candidates) if len(p_candidates) > 0 else a
        n = rng.choice(neg_entries)
        triplets.append((a, p, n))

    return triplets


# ----------------------------
# Main pipeline orchestration
# ----------------------------

def pipeline(cfg):
    ensure_dirs(cfg)

    # Step 1: get KOI table
    koi_df = fetch_koi_table(limit=cfg.get("n_candidates", None))
    print(f"Fetched {len(koi_df)} KOI rows (limited).")

    # save metadata subset
    koi_df.to_csv(Path(cfg["interim_dir"]) / "koi_subset.csv", index=False)

    # Step 2-4: download LC for each kepid and create views
    cached_views = {}  # kepid -> list of candidate dicts
    failures = []

    for idx, row in tqdm(koi_df.iterrows(), total=len(koi_df), desc="Processing KOIs"):
        try:
            kepid = int(row["kepid"])
            period = float(row["period"])
            t0 = float(row["t0"])
        except Exception:
            failures.append(None)
            continue

        duration_days = float(row["duration_days"]) if "duration_days" in row and not pd.isna(row["duration_days"]) else 0.1 * period
        disposition = str(row.get("disposition", "UNKNOWN"))

        lc = safe_download_lc(kepid, mission=cfg.get("mission", "Kepler"), cache_dir=cfg["raw_dir"])
        if lc is None:
            failures.append(kepid)
            continue

        try:
            flat = preprocess_lc(lc, flatten_window=cfg["flatten_window"], outlier_sigma=cfg["outlier_sigma"])
            g, l = make_global_local_views(flat, period=period, t0=t0, duration_days=duration_days, global_len=cfg["global_len"], local_len=cfg["local_len"], local_n_durations=cfg["local_n_durations"])

            cand = {
                "kepid": kepid,
                "period": period,
                "t0": t0,
                "duration_days": duration_days,
                "global": g,
                "local": l,
                "disposition": disposition,
                "candidate_id": f"{kepid}_{int(abs(period)*1e6)}_{int(abs(t0))}",
            }
            cached_views.setdefault(kepid, []).append(cand)
        except Exception:
            failures.append(kepid)
            continue

    print(f"Processed candidates: {sum(len(v) for v in cached_views.values())}; failures: {len(failures)}")

    # Save cached arrays for later quick loading
    # We'll save per-kepid npz files containing arrays for each candidate
    for kepid, clist in cached_views.items():
        arrs = {}
        for i, c in enumerate(clist):
            arrs[f"global_{i}"] = c["global"]
            arrs[f"local_{i}"] = c["local"]
            arrs[f"meta_{i}"] = np.array([c["period"], c["t0"], c["duration_days"]], dtype=np.float32)
        np.savez_compressed(Path(cfg["interim_dir"]) / f"kep_{kepid}.npz", **arrs)

    # Step 5: Build triplets
    # Create a flat meta list of candidates with dispositions
    flat_meta = []
    for kepid, clist in cached_views.items():
        for c in clist:
            meta = {
                "kepid": kepid,
                "candidate_id": c["candidate_id"],
                "period": c["period"],
                "t0": c["t0"],
                "duration_days": c["duration_days"],
                "disposition": c["disposition"],
            }
            flat_meta.append(meta)
    meta_df = pd.DataFrame(flat_meta)
    meta_df.to_csv(Path(cfg["pairs_dir"]) / "candidates_meta.csv", index=False)

    # sample triplets
    try:
        triplets = sample_triplets_from_meta(meta_df, cached_views, n_triplets=min(20000, max(1000, len(meta_df) * 2)))
    except Exception as e:
        print("Triplet sampling failed:", e)
        triplets = []

    # create arrays for triplets: stack into np arrays for fast training loading
    G = cfg["global_len"]
    L = cfg["local_len"]

    def pack_view(entry):
        # returns stacked channel [2, G] where channel0=global (length G) and channel1=centered local in length G
        g = entry["global"]
        l = entry["local"]
        ch0 = g
        ch1 = np.zeros(G, dtype=np.float32)
        start = (G - L) // 2
        ch1[start:start+L] = l
        return np.stack([ch0, ch1], axis=0)

    triplet_arrays = []
    for (a, p, n) in tqdm(triplets, desc="Packing triplets"):
        a_entry = None
        p_entry = None
        n_entry = None
        # brute-force find by candidate_id
        for kepid, clist in cached_views.items():
            for c in clist:
                if c["candidate_id"] == a["candidate_id"]:
                    a_entry = c
                if c["candidate_id"] == p["candidate_id"]:
                    p_entry = c
                if c["candidate_id"] == n["candidate_id"]:
                    n_entry = c
        if a_entry is None or p_entry is None or n_entry is None:
            continue
        A = pack_view(a_entry)
        P = pack_view(p_entry)
        N = pack_view(n_entry)
        triplet_arrays.append((A, P, N))

    # convert to numpy arrays and save
    if len(triplet_arrays) > 0:
        A_stack = np.stack([t[0] for t in triplet_arrays], axis=0)
        P_stack = np.stack([t[1] for t in triplet_arrays], axis=0)
        N_stack = np.stack([t[2] for t in triplet_arrays], axis=0)
        outpath = Path(cfg["pairs_dir"]) / "triplets.npz"
        np.savez_compressed(outpath, A=A_stack, P=P_stack, N=N_stack)
        print(f"Saved triplets to {outpath} with {A_stack.shape[0]} triplets.")
    else:
        print("No triplets created.")

    print("Pipeline finished.")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exoplanet Siamese data pipeline (steps 1-5)")
    p.add_argument("--mission", default=DEFAULTS["mission"], choices=["Kepler", "TESS"], help="Mission to use for LC downloads")
    p.add_argument("--n_candidates", type=int, default=DEFAULTS["n_candidates"], help="Number of KOI rows to fetch (limit)")
    p.add_argument("--global_len", type=int, default=DEFAULTS["global_len"])
    p.add_argument("--local_len", type=int, default=DEFAULTS["local_len"])
    p.add_argument("--flatten_window", type=int, default=DEFAULTS["flatten_window"])
    p.add_argument("--outlier_sigma", type=float, default=DEFAULTS["outlier_sigma"])
    args = p.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))

    pipeline(cfg)
