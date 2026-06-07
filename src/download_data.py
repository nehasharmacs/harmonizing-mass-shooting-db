"""
download_data.py

Downloads publicly-available mass shooting datasets to data/raw/.
Does NOT download the Kaggle dataset — that must already be at
data/raw/kaggle_1965_2019.csv (copy your existing data.csv there).

Datasets fetched:
  - Mother Jones Mass Shootings Database (1982-present)
      Mirrored on GitHub. Primary upstream is a Google Sheet that requires
      manual export; we use a GitHub mirror that tracks it.
  - Stanford MSA (1966-2016)
      Canonical GitHub repo from Stanford Geospatial Center.

If a download fails (URL moved, network blocked, etc.), the script prints a
manual-download instruction and exits nonzero for that dataset but continues
with the others.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Dataset URLs (primary + fallback mirrors).
# Order matters: first success wins.
SOURCES = {
    "mother_jones.csv": [
        # Primary: foxnic mirror — tracks Mother Jones through 2019, stable URL.
        "https://raw.githubusercontent.com/foxnic/US-Mass-Shootings-Analysis/master/ShootingsData.csv",
        # Fallback: jmoreau1309 gist (through 2023)
        "https://gist.githubusercontent.com/jmoreau1309/d47166615f1951dc311bd9e4a0de8ea5/raw/Mother%20Jones%20-%20Mass%20Shootings%20Database%2C%201982%20-%202023%20-%20Sheet1.csv",
    ],
    "stanford_msa.csv": [
        "https://raw.githubusercontent.com/StanfordGeospatialCenter/MSA/master/Data/Stanford_MSA_Database.csv",
    ],
}

MANUAL_INSTRUCTIONS = {
    "mother_jones.csv": (
        "Manual download:\n"
        "  1. Visit https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/\n"
        "  2. Click the Google Sheet link and File -> Download -> Comma-separated values\n"
        "  3. Save as data/raw/mother_jones.csv"
    ),
    "stanford_msa.csv": (
        "Manual download:\n"
        "  1. Visit https://github.com/StanfordGeospatialCenter/MSA\n"
        "  2. Download Data/Stanford_MSA_Database.csv\n"
        "  3. Save as data/raw/stanford_msa.csv"
    ),
}


def _try_download(url: str, dest: Path, timeout: int = 60) -> Optional[Path]:
    """Attempt a single download. Returns dest on success, None on failure."""
    try:
        r = requests.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": "victim-forecast-iri/1.0"})
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        # Sanity check: at least 1 KB
        if dest.stat().st_size < 1024:
            print(f"    suspiciously small file ({dest.stat().st_size} bytes); treating as failure")
            dest.unlink(missing_ok=True)
            return None
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"    failed: {type(e).__name__}: {e}")
        return None


def download_all() -> int:
    """Download every dataset. Returns the number of failures."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    # Check for user-supplied Kaggle dataset up front.
    kaggle_path = RAW_DIR / "kaggle_1965_2019.csv"
    if not kaggle_path.exists():
        print(f"[WARN] {kaggle_path} not found.")
        print("       The original Kaggle dataset must be copied manually:")
        print("       cp /path/to/your/data.csv data/raw/kaggle_1965_2019.csv")
        print()
        failures += 1
    else:
        print(f"[ OK ] {kaggle_path.name} present ({kaggle_path.stat().st_size:,} bytes)")

    for filename, urls in SOURCES.items():
        dest = RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"[SKIP] {filename} already present ({dest.stat().st_size:,} bytes)")
            continue

        print(f"[ .. ] downloading {filename}")
        ok = False
        for url in urls:
            print(f"       trying {url[:90]}{'...' if len(url) > 90 else ''}")
            if _try_download(url, dest):
                print(f"[ OK ] {filename} ({dest.stat().st_size:,} bytes)")
                ok = True
                break
        if not ok:
            failures += 1
            print(f"[FAIL] could not download {filename}")
            print(MANUAL_INSTRUCTIONS[filename])
            print()

    print()
    print(f"Done. {failures} failure(s).")
    return failures


if __name__ == "__main__":
    sys.exit(0 if download_all() == 0 else 1)
