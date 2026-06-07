# The Victim Forecast — Multi-Dataset Integration (IRI 2026)

Cross-dataset integration of three public mass-shooting databases with a unified
risk-classification pipeline. Extends the original Victim Forecast project from
one Kaggle dataset to three harmonized sources plus cross-dataset evaluation.

## Datasets

| Source | Years | Definition | Access |
|--------|-------|------------|--------|
| Kaggle (original) | 1965-2019 | 3+ victims | `data/raw/kaggle_1965_2019.csv` (user-supplied) |
| Mother Jones | 1982-present | 3+ killed (2013+), 4+ killed (pre-2013), public setting | auto-download |
| Stanford MSA | 1966-2016 | 3+ shooting victims (not just killed) | auto-download |

The Violence Project database (1966-present, 200+ variables) is **not** included
because it requires a research-access application. If you obtain it, drop
`tvp.csv` into `data/raw/` and extend `preprocess.py` to include it.

## Pipeline

```
data/raw/  ->  preprocess.py  ->  data/processed/harmonized.csv
                                        |
                                        v
                              run_experiments.py
                              (5-fold CV over 3 strategies x 3 models
                               x {default thresholds, Youden's J})
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
        results/within_dataset.csv          results/cross_dataset.csv
        (pooled CV across all 3)          (leave-one-dataset-out test)
```

The **cross-dataset** experiment is what makes this an IRI-appropriate
contribution: train on two datasets, test on the held-out third. Poor
generalization across datasets is itself a finding; strong generalization
demonstrates the reusable-pipeline claim.

## Quick start (local smoke test)

```bash
# 1. Copy your existing Kaggle data
cp /path/to/your/data.csv data/raw/kaggle_1965_2019.csv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download external datasets (Mother Jones, Stanford MSA)
python src/download_data.py

# 4. Harmonize into a common schema
python src/preprocess.py

# 5. Run the full experiment grid (~5-15 min on a laptop)
python src/run_experiments.py --output-dir results/
```

## HPC (SLURM)

```bash
sbatch scripts/submit_slurm.sh
```

Edit `scripts/submit_slurm.sh` to match your cluster's partition names,
account codes, and module system. The job is CPU-only and low-memory —
a single node with 4 cores and 8 GB is enough.

## Re-using the original notebooks

This project re-implements the methodology from the `group_project_final_presentation/`
notebooks as importable modules. Your core logic (3 classification strategies,
3 models, RandomOverSampler, Youden's J per-class threshold tuning) is preserved
in `src/classification_strategies.py`, `src/models.py`, and `src/evaluation.py`.
The additions are:

1. Proper 5-fold stratified cross-validation (was: single train/test split)
2. Mean ± std reporting per configuration
3. Multi-dataset harmonization
4. Leave-one-dataset-out cross-dataset evaluation
5. Permutation-based feature importance

## Citation

If the paper is accepted, cite as: Sharma, N., Sharma, R. (2026). The Victim Forecast: Cross-Dataset Risk
Classification for Mass Shooting Incidents. *IEEE IRI 2026*.

## Ethics note

See `paper/paper.tex` Section VI (Limitations and Ethics) for discussion of
reporting bias, the operationalization of mental-health variables, and
deployment considerations. Do not deploy this model for operational risk
assessment without substantial additional validation.
