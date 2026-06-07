"""
PATCH for run_experiments.py
=============================
Add these changes to your existing run_experiments.py in three places.
Each block is marked with WHERE TO ADD IT.

1. Import (at the top, after the existing imports from .evaluation)
2. New argparse flag (inside the ArgumentParser block in main())
3. New execution block (after the existing --skip-fi block in main())
"""

# ---------------------------------------------------------------------------
# 1. ADD THIS IMPORT — after the existing `.evaluation` imports
# ---------------------------------------------------------------------------

# from .temporal_holdout import temporal_holdout_sweep, aggregate_temporal


# ---------------------------------------------------------------------------
# 2. ADD THIS ARGPARSE FLAG — inside main(), after the --verbose argument
# ---------------------------------------------------------------------------

# p.add_argument("--temporal-holdout", action="store_true",
#                help="Run temporal holdout (train pre-2010, test post-2010).")
# p.add_argument("--cutoff-year", type=int, default=2010,
#                help="Year threshold for temporal holdout split.")
# p.add_argument("--n-seeds", type=int, default=5,
#                help="Number of oversampling seeds for temporal holdout.")


# ---------------------------------------------------------------------------
# 3. ADD THIS EXECUTION BLOCK — after the existing --skip-fi block in main()
#    (i.e. just before `return 0`)
# ---------------------------------------------------------------------------

# if args.temporal_holdout:
#     logging.info("=== Temporal holdout (train pre-%d, test post-%d) ===",
#                  args.cutoff_year, args.cutoff_year)
#
#     n_train = (df["year"] < args.cutoff_year).sum()
#     n_test  = (df["year"] >= args.cutoff_year).sum()
#     logging.info("Split: %d train rows, %d test rows", n_train, n_test)
#
#     seeds = [args.random_state + i for i in range(args.n_seeds)]
#     temporal = temporal_holdout_sweep(
#         df=df,
#         cutoff_year=args.cutoff_year,
#         seeds=seeds,
#         random_state=args.random_state,
#         recompute_labels=True,
#         include_youden_val=True,
#     )
#
#     temporal_path = out / "temporal_holdout.csv"
#     temporal.to_csv(temporal_path, index=False)
#     logging.info("Wrote %s (%d rows)", temporal_path, len(temporal))
#
#     temporal_agg = aggregate_temporal(temporal)
#     temporal_agg_path = out / "temporal_holdout_aggregated.csv"
#     temporal_agg.to_csv(temporal_agg_path, index=False)
#     logging.info("Wrote %s", temporal_agg_path)
#
#     # Print top 5 overall configs
#     if "recall_very_high_mean" in temporal_agg.columns:
#         top = (temporal_agg[temporal_agg["test_source"] == "ALL_POST_CUTOFF"]
#                .sort_values("recall_very_high_mean", ascending=False)
#                .head(5)[["strategy", "model", "threshold_mode", "features",
#                           "recall_very_high_mean", "recall_very_high_std",
#                           "precision_very_high_mean"]])
#         logging.info("Top 5 temporal configs:\n%s", top.to_string())
