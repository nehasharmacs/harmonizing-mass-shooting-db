"""
preprocess.py

Harmonize three mass-shooting datasets into a common schema.

This module is the integration contribution of the paper. Every dataset
defines "mass shooting", "venue", and "mental health" slightly differently,
so we document each mapping decision explicitly. A reviewer or future reuser
should be able to read this file and understand exactly what was done.

Common schema (columns of the harmonized output):
    source            : str     — which dataset the row came from
    year              : int     — year of incident
    fatalities        : int     — number killed (excluding shooter)
    injured           : int     — number injured but not killed
    total_victims     : int     — fatalities + injured
    incident_area     : str     — normalized venue category (see AREA_MAP)
    open_close        : str     — {Open, Close, Open/Close, Unknown}
    age               : float   — shooter age (imputed with dataset mean if missing)
    gender            : str     — {M, F, MF, U}
    race              : str     — {White, Black, Latino, Asian, Other, Unknown}
    mental_health     : str     — {Yes, No, Unclear, Unknown}
    multiple_shooters : int     — 0/1 flag

Definitional choice: we keep all rows with total_victims >= 3, matching the
most inclusive common definition (Kaggle and Stanford MSA both use 3+; Mother
Jones uses 3+ killed from 2013 onward and 4+ killed pre-2013, meaning their
rows satisfy the 3+ total victim bound automatically).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Normalization maps
# ---------------------------------------------------------------------------

# Incident-area categories used downstream. Every dataset's raw venue strings
# are mapped onto this set.
AREA_CATEGORIES = [
    "School", "Workplace", "Religious", "Military",
    "Commercial", "Public_Open", "Home", "Other",
]

AREA_MAP = {
    # School
    "school": "School",
    "schools": "School",
    # Workplace / commercial distinction: keep them apart because the original
    # paper found venue type mattered.
    "workplace": "Workplace",
    "work": "Workplace",
    "service building": "Workplace",
    "office": "Workplace",
    # Religious
    "religious": "Religious",
    "religous building": "Religious",   # sic — typo in source data
    "religious building": "Religious",
    "religion": "Religious",
    "place of worship": "Religious",
    # Military / government
    "military": "Military",
    "airport": "Public_Open",   # airports are public open spaces
    # Commercial (stores, restaurants, bars)
    "amenity": "Commercial",
    "store": "Commercial",
    "restaurant": "Commercial",
    "bar": "Commercial",
    # Public open
    "street": "Public_Open",
    "park": "Public_Open",
    "event": "Public_Open",
    "protest": "Public_Open",
    # Home
    "home": "Home",
    "residence": "Home",
    "house": "Home",
    # Other / multiple / unknown
    "other": "Other",
    "multiple": "Other",
}

OPEN_CLOSE_MAP = {
    "open": "Open",
    "close": "Close",
    "closed": "Close",
    "open/close": "Open/Close",
    "close/open": "Open/Close",
}

GENDER_MAP = {
    "m": "M", "male": "M",
    "f": "F", "female": "F",
    "mf": "MF", "male/female": "MF", "male & female": "MF",
    "u": "U", "unknown": "U",
}

RACE_MAP = {
    "white": "White",
    "black": "Black",
    "black american or african american": "Black",
    "latino": "Latino",
    "hispanic": "Latino",
    "asian": "Asian",
    "native american": "Other",
    "other": "Other",
    "unknown": "Unknown",
    "": "Unknown",
}

MH_MAP = {
    "yes": "Yes",
    "no": "No",
    "unclear": "Unclear",
    "unknown": "Unknown",
    "tbd": "Unknown",
    "-": "Unknown",
    "": "Unknown",
    "no evidence of": "No",
}


# ---------------------------------------------------------------------------
# Helper: defensive normalizer
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    """Lowercase-strip. Returns '' for NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).strip().lower()


def _map_or_other(val, mapping: dict, default: str = "Other") -> str:
    key = _norm(val)
    # Allow substring matching for area (e.g. "Multiple venues (school, home)")
    if key in mapping:
        return mapping[key]
    for k, v in mapping.items():
        if k and k in key:
            return v
    return default


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_kaggle(path: Path) -> pd.DataFrame:
    """Load the Kaggle 1965-2019 dataset (original project data)."""
    df = pd.read_csv(path)
    # Column name cleanup: spaces -> underscores
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Total_victims in raw is fatalities + injured; add policemen killed if any
    fatalities = pd.to_numeric(df.get("Fatalities"), errors="coerce").fillna(0)
    injured = pd.to_numeric(df.get("Injured"), errors="coerce").fillna(0)
    police = pd.to_numeric(df.get("Policeman_Killed"), errors="coerce").fillna(0)
    total = fatalities + injured + police

    # Year from Date field (mixed formats)
    year = pd.to_datetime(df.get("Date"), errors="coerce").dt.year

    out = pd.DataFrame({
        "source": "kaggle",
        "year": year,
        "fatalities": (fatalities + police).astype(int),
        "injured": injured.astype(int),
        "total_victims": total.astype(int),
        "incident_area": df.get("Incident_Area").apply(lambda x: _map_or_other(x, AREA_MAP, "Other")),
        "open_close": df.get("Open/Close_Location").apply(lambda x: OPEN_CLOSE_MAP.get(_norm(x), "Unknown")),
        "age": pd.to_numeric(df.get("Age"), errors="coerce"),
        "gender": df.get("Gender").apply(lambda x: GENDER_MAP.get(_norm(x), "U")),
        "race": df.get("Race").apply(lambda x: RACE_MAP.get(_norm(x), "Unknown")),
        "mental_health": df.get("Mental_Health_Issues").apply(lambda x: MH_MAP.get(_norm(x), "Unknown")),
        "multiple_shooters": 0,   # Kaggle schema doesn't mark this; default 0
    })
    return out


def load_mother_jones(path: Path) -> pd.DataFrame:
    """Load the Mother Jones dataset (1982-present)."""
    df = pd.read_csv(path, encoding="latin-1")
    # Mother Jones column names vary between mirrors. Normalize.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def _col(*candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([np.nan] * len(df))

    fatalities = pd.to_numeric(_col("fatalities"), errors="coerce").fillna(0)
    injured = pd.to_numeric(_col("injured"), errors="coerce").fillna(0)
    total = pd.to_numeric(_col("total_victims", "total victims"), errors="coerce")
    total = total.fillna(fatalities + injured)

    year = pd.to_numeric(_col("year"), errors="coerce")
    if year.isna().all():
        year = pd.to_datetime(_col("date"), errors="coerce").dt.year

    # Mother Jones uses a single "location" or "type"/"location.1" column for venue.
    # Some mirrors name it 'location_1' when there are duplicate headers.
    venue = _col("location.1", "location_1", "venue", "location_category")
    # If venue column is missing/empty, fall back to the "type" column or "other"
    if venue.isna().all() or (venue.astype(str).str.strip() == "").all():
        venue = _col("type")

    age = pd.to_numeric(_col("age_of_shooter", "age"), errors="coerce")

    out = pd.DataFrame({
        "source": "mother_jones",
        "year": year,
        "fatalities": fatalities.astype(int),
        "injured": injured.astype(int),
        "total_victims": total.astype(int),
        "incident_area": venue.apply(lambda x: _map_or_other(x, AREA_MAP, "Other")),
        "open_close": "Unknown",   # Mother Jones doesn't record this explicitly
        "age": age,
        "gender": _col("gender").apply(lambda x: GENDER_MAP.get(_norm(x), "U")),
        "race": _col("race").apply(lambda x: RACE_MAP.get(_norm(x), "Unknown")),
        "mental_health": _col("prior_signs_mental_health_issues", "prior_signs_mental_health", "mental_health")
            .apply(lambda x: MH_MAP.get(_norm(x), "Unknown")),
        "multiple_shooters": 0,
    })
    return out


def load_stanford_msa(path: Path) -> pd.DataFrame:
    """Load the Stanford MSA dataset (1966-2016)."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]

    def _col(*candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([np.nan] * len(df))

    fatalities = pd.to_numeric(_col("Total Number of Fatalities"), errors="coerce").fillna(0)
    injured_civ = pd.to_numeric(_col("Number of Civilian Injured"), errors="coerce").fillna(0)
    injured_enf = pd.to_numeric(_col("Number of Enforcement Injured"), errors="coerce").fillna(0)
    injured = injured_civ + injured_enf
    total = pd.to_numeric(_col("Total Number of Victims"), errors="coerce")
    total = total.fillna(fatalities + injured)

    year = pd.to_datetime(_col("Date"), errors="coerce").dt.year

    n_shooters = pd.to_numeric(_col("Number of shooters"), errors="coerce").fillna(1)
    age = pd.to_numeric(_col("Average Shooter Age"), errors="coerce")

    out = pd.DataFrame({
        "source": "stanford_msa",
        "year": year,
        "fatalities": fatalities.astype(int),
        "injured": injured.astype(int),
        "total_victims": total.astype(int),
        "incident_area": _col("Place Type").apply(lambda x: _map_or_other(x, AREA_MAP, "Other")),
        "open_close": "Unknown",   # Not recorded in Stanford MSA
        "age": age,
        "gender": _col("Shooter Sex").apply(lambda x: GENDER_MAP.get(_norm(x), "U")),
        "race": _col("Shooter Race").apply(lambda x: RACE_MAP.get(_norm(x), "Unknown")),
        "mental_health": _col("History of Mental Illness - General")
            .apply(lambda x: MH_MAP.get(_norm(x), "Unknown")),
        "multiple_shooters": (n_shooters > 1).astype(int),
    })
    return out


# ---------------------------------------------------------------------------
# Cleanup + risk label + output
# ---------------------------------------------------------------------------

def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common-schema cleanup after per-dataset loaders."""
    # Drop rows missing critical fields
    df = df.dropna(subset=["year", "total_victims"]).copy()
    df["year"] = df["year"].astype(int)
    # Definition: keep only rows with >= 3 victims
    df = df[df["total_victims"] >= 3].copy()

    # Drop extreme outliers per the original project's decisions.
    # Las Vegas 2017 (586 victims including 546 injured) was the dominant outlier
    # in the original work; keep it dropped to preserve comparability.
    df = df[df["total_victims"] < 100].copy()

    # Impute age with source-level mean (preserves per-dataset distributions)
    df["age"] = df.groupby("source")["age"].transform(
        lambda s: s.fillna(s.mean() if not np.isnan(s.mean()) else 30)
    )
    df["age"] = df["age"].astype(float)

    # Categorical NaN -> Unknown
    for c in ["incident_area", "open_close", "gender", "race", "mental_health"]:
        df[c] = df[c].fillna("Unknown").replace("", "Unknown")

    df = df.reset_index(drop=True)
    return df


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add three risk-level labelings (rule-based, std-dev, quartile).

    Labels are computed on the POOLED dataset so they are directly comparable
    across sources. See paper Section IV for rationale.
    """
    df = df.copy()
    v = df["total_victims"].astype(float)

    # --- Rule-based (the original project's thresholds) ---
    def rule_based(x):
        if x < 10:
            return "Low"
        if 11 <= x <= 20:
            return "Medium"
        if 21 <= x <= 40:
            return "High"
        return "VeryHigh"
    df["risk_rule"] = v.apply(rule_based)

    # --- Standard deviation ---
    mu, sd = v.mean(), v.std()
    def std_based(x):
        if x < mu - sd:
            return "Low"
        if x <= mu + sd:
            return "Medium"
        if x <= mu + 2 * sd:
            return "High"
        return "VeryHigh"
    df["risk_std"] = v.apply(std_based)

    # --- Quartile ---
    q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
    def quart(x):
        if x <= q1:
            return "Low"
        if x <= q2:
            return "Medium"
        if x <= q3:
            return "High"
        return "VeryHigh"
    df["risk_quartile"] = v.apply(quart)

    return df


def run(raw_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_DIR
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    loaders = [
        ("kaggle_1965_2019.csv", load_kaggle),
        ("mother_jones.csv", load_mother_jones),
        ("stanford_msa.csv", load_stanford_msa),
    ]
    for fname, loader in loaders:
        path = raw_dir / fname
        if not path.exists():
            print(f"[SKIP] {fname} missing — run download_data.py first")
            continue
        print(f"[LOAD] {fname}")
        try:
            frames.append(loader(path))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] failed to load {fname}: {type(e).__name__}: {e}")

    if not frames:
        raise SystemExit("No datasets loaded. Did you run download_data.py?")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] combined raw rows: {len(combined)}")

    cleaned = clean_and_filter(combined)
    print(f"[INFO] after clean/filter: {len(cleaned)}")
    print(cleaned.groupby("source").size().to_string())

    labeled = add_risk_labels(cleaned)

    out_path = out_dir / "harmonized.csv"
    labeled.to_csv(out_path, index=False)
    print(f"[DONE] wrote {out_path} ({len(labeled)} rows)")

    # Summary table for the paper
    summary = (labeled.groupby("source")
               .agg(n=("total_victims", "size"),
                    year_min=("year", "min"),
                    year_max=("year", "max"),
                    victims_mean=("total_victims", "mean"),
                    victims_median=("total_victims", "median"))
               .round(2))
    print()
    print("Dataset summary:")
    print(summary.to_string())

    summary.to_csv(out_dir / "dataset_summary.csv")
    return labeled


if __name__ == "__main__":
    run()
