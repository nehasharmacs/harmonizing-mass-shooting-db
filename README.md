# Harmonizing Mass-Shooting Databases for Cross-Source Risk Classification

Code and harmonized schema accompanying the IEEE IRI 2026 short paper.

This repository integrates three public U.S. mass-shooting databases (Kaggle,
Mother Jones, Stanford MSA; 806 incidents) into a unified 12-variable schema
and evaluates cross-source generalization of risk classifiers under a
leakage-aware leave-one-dataset-out (LODO) protocol.

## Datasets

| Source       | Years     | Inclusion criterion              | Access                                       |
|--------------|-----------|----------------------------------|----------------------------------------------|
| Kaggle       | 1966–2017 | 3+ victims                       | `data/raw/kaggle_1965_2019.csv`              |
| Mother Jones | 1982–2026 | 4+ killed (pre-2013), 3+ (2013+) | `data/raw/mother_jones.csv` (auto-download)  |
| Stanford MSA | 1966–2016 | 3+ shooting victims              | `data/raw/stanford_msa.csv` (auto-download)  |

After harmonization and removal of the Las Vegas 2017 outlier, the pooled
dataset contains **806 incidents** across **12 common variables**
(`source`, `year`, `fatalities`, `injured`, `total_victims`, `incident_area`,
`open_close`, `age`, `gender`, `race`, `mental_health`, `multiple_shooters`).

## Repository layout

```
.
├── data/
│   ├── raw/                    # source CSVs (one per database)
│   └── processed/              # harmonized.csv, dataset_summary.csv
├── src/                        # core pipeline modules
│   ├── preprocess.py           # harmonization into 12-variable schema
│   ├── classification_strategies.py  # rule / std / quartile labeling
│   ├── models.py               # DT, MLR, GNB wrappers
│   ├── evaluation.py           # 5-fold CV, LODO, metrics
│   ├── run_experiments.py      # main experiment driver
│   ├── temporal_holdout.py     # pre/post-2010 temporal split
│   └── download_data.py        # fetch Mother Jones + Stanford MSA
├── scripts/                    # numbered analysis scripts
│   ├── 01_bootstrap_ci.py
│   ├── 02_wilcoxon_ablation.py
│   ├── 03_deduplication.py
│   ├── 04_ablation_depth_sweep.py
│   ├── 05_rf_baseline_youden_breakdown.py
│   ├── 06_rf_depth_pipeline.py
│   └── 07_figures.py           # regenerate paper figures
├── results/                    # experiment outputs (CSV)
├── figures/                    # paper figures (PDF + PNG)
└── README.md
```

## Installation

```bash
# Conda environment (recommended)
conda env create -f environment.yml
conda activate iri

# Or pip
pip install -r requirements.txt
```

Tested with Python 3.9 and 3.11.

## Reproducing the paper

```bash
# 1. Download external datasets (Mother Jones + Stanford MSA)
python src/download_data.py

# 2. Harmonize all three sources into the common schema
python src/preprocess.py

# 3. Run the full experiment grid
#    (5-fold CV + LODO across 3 strategies × 3 models × {default, Youden})
python src/run_experiments.py

# 4. Temporal holdout (pre-2010 vs post-2010)
python src/temporal_holdout.py

# 5. Downstream analyses
python scripts/01_bootstrap_ci.py
python scripts/02_wilcoxon_ablation.py

# 6. Regenerate paper figures
python scripts/07_figures.py
```

Outputs are written to `results/` (CSVs) and `figures/` (PDF + PNG).
On a modern laptop the full pipeline runs in roughly 10–20 minutes.

## Method summary

- **Three risk-stratification schemes** over `total_victims`: fixed
  rule-based thresholds (10/20/40), standard deviation (μ±σ, μ+2σ), and
  quartiles (Q1/Q2/Q3). Under LODO, std and quartile thresholds are
  computed only on the training fold to prevent label leakage.
- **Three classifiers**: depth-3 decision tree, L2-regularized multinomial
  logistic regression, Gaussian naïve Bayes.
- **Two decision rules**: argmax over class probabilities (default) and
  per-class Youden's J thresholding on a 20% stratified validation split.
- **Two evaluation protocols**: stratified 5-fold cross-validation on the
  pooled 806-row dataset (within-dataset), and leave-one-dataset-out
  trained on two sources and tested on the third (cross-source), repeated
  over 5 random seeds.

## Key findings

- Cross-source ranking of feature sets and stratification strategies is
  not reliable at *n*=806 rows: across 27 paired comparisons, neither
  contextual nor full feature sets dominate.
- The best LODO configuration reaches VeryHigh recall of 0.819 on Kaggle
  and 0.886 on Stanford MSA; the higher Mother Jones value (0.989)
  reflects its 4+ fatality inclusion criterion rather than genuine
  generalization.
- Within-dataset VeryHigh precision is only 0.147 — the pipeline is not
  suitable for operational use.
- LODO-optimal and time-optimal configurations diverge sharply: the LODO
  best collapses to recall 0.047 on post-2010 data, while a different
  configuration achieves 0.820. Cross-source and cross-time evaluations
  select different models.

## Ethics

This pipeline is a research artifact, not a deployable risk-assessment
system. Operational use for resource allocation or surveillance would
risk reinforcing media-coverage bias and encoding demographic proxies
as actionable risk signals. The harmonized schema and code are released
for research and audit purposes only. See the paper's Discussion section
for the full ethics and limitations discussion.

## Citation

```bibtex
@inproceedings{sharma2026harmonizing,
  title     = {Harmonizing Mass-Shooting Databases for Cross-Source Risk Classification},
  author    = {Sharma, Neha and Sharma, Ritesh},
  booktitle = {IEEE International Conference on Information Reuse and Integration (IRI)},
  year      = {2026},
  address   = {Seattle, WA, USA}
}
```

## Data attribution

- **Mother Jones**: Follman, M., Aronsen, G., & Pan, D. *U.S. Mass
  Shootings, 1982–2026*. Mother Jones.
  https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/
- **Stanford MSA**: Stanford Geospatial Center. *Stanford Mass Shootings
  in America (MSA) Database, 1966–2016*.
  https://github.com/StanfordGeospatialCenter/MSA
- **Kaggle**: *Mass Shootings in America* dataset, originally compiled by
  the Stanford Geospatial Center.
