"""
Script 2: Wilcoxon Signed-Rank Test — All 27 Paired Comparisons
Reads directly from: results/cross_dataset_aggregated.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tabulate import tabulate

RESULTS_DIR = "results"

if __name__ == "__main__":
    agg = pd.read_csv(f"{RESULTS_DIR}/cross_dataset_aggregated.csv")

    def min_recall(df, feat):
        return (df[df.features==feat]
                .groupby(["strategy","model","threshold_mode"])
                ["recall_very_high_mean"].min().to_dict())

    ctx_min  = min_recall(agg, "ctx")
    full_min = min_recall(agg, "full")

    rows = []
    for (strat, model, thr), ctx_val in ctx_min.items():
        full_val = full_min.get((strat, model, thr), np.nan)
        delta    = ctx_val - full_val
        rows.append({"strategy":strat,"model":model,"threshold":thr,
                     "ctx_min":round(ctx_val,4),"full_min":round(full_val,4),
                     "delta":round(delta,4),
                     "favors":"ctx" if delta>0 else ("full" if delta<0 else "tie")})

    df = pd.DataFrame(rows)
    ctx_vals  = df.ctx_min.values
    full_vals = df.full_min.values
    deltas    = ctx_vals - full_vals

    n_ctx  = int((deltas>0).sum())
    n_full = int((deltas<0).sum())
    n_ties = int((deltas==0).sum())

    stat_two, p_two = wilcoxon(ctx_vals, full_vals, alternative="two-sided",  zero_method="wilcox")
    stat_gt,  p_gt  = wilcoxon(ctx_vals, full_vals, alternative="greater",    zero_method="wilcox")
    stat_lt,  p_lt  = wilcoxon(ctx_vals, full_vals, alternative="less",       zero_method="wilcox")

    print("=" * 70)
    print("All 27 Paired Comparisons (ctx vs full)")
    print("=" * 70)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))
    print()
    print("=" * 70)
    print("Wilcoxon Signed-Rank Test")
    print("=" * 70)
    print(f"  Favor ctx  : {n_ctx}")
    print(f"  Favor full : {n_full}")
    print(f"  Ties       : {n_ties}")
    print(f"  Two-sided  : W={stat_two:.1f}, p={p_two:.3f}")
    print(f"  ctx > full : W={stat_gt:.1f},  p={p_gt:.3f}")
    print(f"  full > ctx : W={stat_lt:.1f},  p={p_lt:.3f}")
    print()
    alpha = 0.05
    if p_two > alpha:
        conclusion = (f"no significant systematic advantage for either feature set "
                      f"across all 27 paired comparisons "
                      f"(Wilcoxon signed-rank: W\u2009=\u2009{stat_two:.0f}, p\u2009=\u2009{p_two:.3f})")
    elif p_gt <= alpha:
        conclusion = (f"a significant advantage for contextual features "
                      f"(Wilcoxon: W\u2009=\u2009{stat_gt:.0f}, p\u2009=\u2009{p_gt:.3f}, one-sided)")
    else:
        conclusion = (f"a significant advantage for the full feature set "
                      f"(Wilcoxon: W\u2009=\u2009{stat_lt:.0f}, p\u2009=\u2009{p_lt:.3f}, one-sided)")
    print("── Paper sentence (paste into Section IV-B) ──────────────────")
    print(f"Among all 27 paired comparisons, {n_ctx} favour contextual features and "
          f"{n_full} favour the full set, with {n_ties} tie(s); "
          f"we find {conclusion}.")