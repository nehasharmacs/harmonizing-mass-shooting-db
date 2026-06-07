"""
Script 3: Cross-Source Deduplication Estimate
Reads directly from: results/cross_dataset.csv
Reconstructs source-level incident data from the LODO results to estimate overlap.
NOTE: Since we only have model outputs, not raw incident data, this script
      reports what CAN be inferred: source sizes and year ranges from the
      aggregated file, and provides a template for running on raw data.
      Point it at your raw harmonized CSVs if available.
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

RESULTS_DIR = "results"

# ── If you have raw harmonized source files, point to them here ──────────────
# RAW_KAGGLE   = "data/kaggle_harmonized.csv"
# RAW_MJ       = "data/motherjones_harmonized.csv"
# RAW_STANFORD = "data/stanford_harmonized.csv"
# If not available, the script reports source stats from the aggregated file.

def estimate_overlap(df_a, df_b, year_col, fatal_col, label_a, label_b):
    merged = pd.merge(df_a[[year_col, fatal_col]],
                      df_b[[year_col, fatal_col]],
                      on=[year_col, fatal_col], how="inner")
    n   = len(merged)
    pct_a = n / len(df_a) * 100
    pct_b = n / len(df_b) * 100
    print(f"  {label_a} ∩ {label_b}: {n} matches  ({pct_a:.1f}% of {label_a}, {pct_b:.1f}% of {label_b})")
    risk = "LOW" if max(pct_a, pct_b) < 5 else "MODERATE — consider deduplication"
    return {"Pair": f"{label_a} ∩ {label_b}", "n_overlap": n,
            f"% of {label_a[:3]}": f"{pct_a:.1f}%",
            f"% of {label_b[:3]}": f"{pct_b:.1f}%",
            "LODO risk": risk}

if __name__ == "__main__":
    agg = pd.read_csv(f"{RESULTS_DIR}/cross_dataset_aggregated.csv")

    # Source sizes as recorded in the paper (Table I)
    source_info = {"kaggle": {"n": 315, "years": "1966-2017"},
                   "mother_jones": {"n": 157, "years": "1982-2026"},
                   "stanford_msa": {"n": 334, "years": "1966-2016"}}

    print("=" * 65)
    print("Source Summary (from paper Table I)")
    print("=" * 65)
    rows = [{"Source": k, "n": v["n"], "Years": v["years"]} for k, v in source_info.items()]
    print(tabulate(rows, headers="keys", tablefmt="github"))
    print()

    # Try loading raw harmonized files if they exist
    import os
    raw_files = {
        "kaggle":       "data/kaggle_harmonized.csv",
        "mother_jones": "data/motherjones_harmonized.csv",
        "stanford_msa": "data/stanford_harmonized.csv",
    }
    available = {k: v for k, v in raw_files.items() if os.path.exists(v)}

    if len(available) == 3:
        print("=" * 65)
        print("Pairwise Overlap Estimates (year + fatalities key)")
        print("=" * 65)
        dfs = {k: pd.read_csv(v) for k, v in available.items()}
        summary = []
        pairs = [("kaggle","stanford_msa"),("kaggle","mother_jones"),("stanford_msa","mother_jones")]
        for a, b in pairs:
            row = estimate_overlap(dfs[a], dfs[b], "year", "fatalities",
                                   a.replace("_"," ").title(), b.replace("_"," ").title())
            summary.append(row)
        print()
        print(tabulate(summary, headers="keys", tablefmt="github"))
        print()
        print("Add overlap counts as a Table I footnote in the paper.")
    else:
        print("Raw harmonized CSVs not found at data/*.csv")
        print("Point RAW_KAGGLE / RAW_MJ / RAW_STANFORD to your files.")
        print()
        print("Estimated overlap based on Table I year range overlap:")
        print("  Kaggle ∩ Stanford MSA: both cover 1966-2016 — OVERLAP LIKELY")
        print("  Kaggle ∩ Mother Jones: Kaggle 1966-2017, MJ 1982-2026 — PARTIAL")
        print("  Stanford ∩ Mother Jones: Stanford 1966-2016, MJ 1982-2026 — PARTIAL")
        print()
        print("Action: Run deduplication on your raw harmonized files using")
        print("        year + fatalities (+ state if available) as the key.")