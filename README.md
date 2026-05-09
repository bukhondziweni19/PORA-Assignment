# Signature-Informed Transformer for Portfolio Optimisation
## Replication and Extension — PORA9X1 Group Project

### Overview

This repository contains the full replication and extension
of Hwang and Zohren (2025), "Signature-Informed Transformer
for Asset Allocation" (arXiv:2510.03129), completed as part
of the PORA9X1 module at the University of Johannesburg
under the supervision of Prof. Jules Mba.

The original paper proposes a Transformer-based portfolio
optimisation framework that uses path signatures of asset
return histories as input features and trains the model
end-to-end using a CVaR risk objective directly on portfolio
returns, bypassing the traditional return forecasting step.

---

### Research Extension

Our extension asks which drawdown-based training objective
produces the best portfolio outcomes within the SIT framework,
and whether that result is robust across different time periods,
return paths and market crises.

We conduct a horse race comparing five training objectives:

| Model        | Objective                    | Reference                       |

| CVaR         | Conditional Value at Risk    | Rockafellar and Uryasev (2000) |
| AvgDD        | Average Drawdown             | Cajas (2025, Chapter 7)        |
| CDaR         | Conditional Drawdown at Risk | Chekhlov et al. (2005)         |
| UlcerIndex   | Root Mean Square Drawdown    | Martin and McCann (1989)       |
| SmoothMaxDD  | Differentiable Max Drawdown  | Magdon-Ismail and Atiya (2004) |

---

### Classical Benchmarks

Two benchmarks are included for comparison:

| Benchmark | Description |
|---|---|
| Equal Weight (1/N) | Allocates 1/N to each stock every day. No optimisation required. |
| Mean-CVaR | Classical predict-then-optimise using the Rockafellar-Uryasev linear reformulation solved via cvxpy. |

The Mean-CVaR benchmark was added to test whether end-to-end
learning adds genuine value over a classical two-step pipeline
that targets the same CVaR risk objective. If the SIT-CVaR
model cannot beat this benchmark the additional complexity of
the end-to-end architecture is not justified.

---

### Dataset

Daily adjusted closing prices from Yahoo Finance for S&P 100
constituent stocks across five sectors, covering January 2000
to December 2024. Three universe sizes are tested: 20, 30
and 50 stocks.

Four stocks were replaced due to delistings or insufficient
price history:

|Original| Replacement| Reason |

| BCR | BSX | BCR delisted after BDX acquisition in 2017     |
| K   | CPB | K acquired by Mars in August 2024              |
| MMC | CB  | MMC had data gaps on Yahoo Finance around 2000 |
| PSX | OXY | PSX only listed from May 2012                  |

Data split follows the original paper exactly:

| Split      | Period                   | Duration |

| Training   | 2000-01-01 to 2016-12-31 | 17 years |
| Validation | 2017-01-01 to 2019-12-31 | 3 years  |
| Test       | 2020-01-01 to 2024-12-31 | 5 years  |

The test period covers the COVID-19 crash, post-crisis
recovery, the 2022 Federal Reserve rate shock, and the
SVB banking crisis of 2023.

---

### Notebook Structure

The full project is contained in paper_replication.ipynb
with 29 cells organised into five sections:

**Section 1: Data Collection (Cells 0 to 8)**
- Cell 0: Imports and seeds
- Cell 1: Configuration (change DATA_POOL here only)
- Cell 2: Ticker definitions and stock replacements
- Cell 3: Download data from Yahoo Finance
- Cell 4: Clean data and handle missing values
- Cell 5: Compute log returns and winsorise
- Cell 6: Apply train/validation/test split
- Cell 7: Visualise data (3 plots)
- Cell 8: Save all data to disk

**Section 2: Path Signatures (Cells 9 to 11)**
- Cell 9: Compute path signature features
- Cell 10: Normalise features and align with returns
- Cell 11: Save all signature arrays to disk

**Section 3: Model Training (Cells 12 to 22)**
- Cell 12: SIT model architecture definition
- Cell 13: PyTorch Dataset and DataLoaders
- Cell 14: Five differentiable loss functions
- Cell 15: Nine performance metrics including Sortino ratio
- Cell 16: Training loop with early stopping
- Cell 17: Evaluation function using best checkpoint
- Cell 18: Classical benchmarks (Equal Weight and Mean-CVaR)
- Cell 19: Horse race training for all five models
- Cell 20: Composite and drawdown-specific ranking
- Cell 21: Cumulative wealth and drawdown plots
- Cell 22: Training curves for all five models

**Section 4: Robustness Testing (Cells 23 to 27)**
- Cell 23: Lens 1, walk-forward validation (9 windows)
- Cell 24: Walk-forward plots and win rate summary
- Cell 25: Lens 2, block bootstrap (500 samples)
- Cell 26: Bootstrap KDE plots and Wilcoxon test results
- Cell 27: Lens 3, stress scenario testing

**Section 5: Master Summary (Cell 28)**
- Cell 28: Master robustness summary and overall verdict

---

### Key Design Decisions

**Signature order:** Order 2 is used for 30 and 50 stock
universes due to RAM constraints. At order 3 with 50 assets
the feature vector contains 127,550 terms per day which
exceeds available memory. Order 2 produces 2,550 features
and preserves all pairwise cross-asset lead-lag interactions.
Order 3 is used for the 20 stock universe where memory
constraints are not binding.

**Dynamic model assignment:** The winning model is determined
automatically from the horse race composite ranking and saved
to winners.json by Cell 20. All robustness cells (23 to 28)
load from this file rather than hardcoding model names. This
means changing DATA_POOL and rerunning updates everything
automatically.

**Hyperparameter policy:** Hyperparameters are not re-tuned
per walk-forward window. The configuration validated on the
2017 to 2019 period is reused across all windows to prevent
look-ahead bias from re-tuning, following the examiner
recommendation.

**Transaction costs:** Assumed zero throughout all
experiments. Turnover is reported as a metric and included
in the composite ranking. This assumption is stated as a
limitation in the final report.

---

### Performance Metrics

All models are evaluated on nine metrics:

| Metric                | Direction        | Description                                  |

| Annualised Return     | Higher is better | Geometric mean annual return                 |
| Annualised Volatility | Lower is better  | Daily std scaled to annual                   |
| Sharpe Ratio          | Higher is better | Excess return per unit of total risk         |
| Sortino Ratio         | Higher is better | Excess return per unit of downside risk only |
| Maximum Drawdown      | Lower is better  | Largest peak-to-trough decline               |
| Calmar Ratio          | Higher is better | Annualised return divided by max drawdown    |
| Ulcer Index           | Lower is better  | Root mean square of all drawdowns            |
| Time Underwater       | Lower is better  | Fraction of days below previous peak         |
| Turnover              | Lower is better  | Average daily portfolio reallocation         |

---

### Robustness Framework

Three robustness lenses are applied to the composite ranking
winner at each universe size:

**Lens 1: Walk-Forward Validation**
Nine expanding training windows with test years from 2008
to 2016. Training always starts from January 2000. Models
are retrained from scratch on each window. Hyperparameters
are not re-tuned per window.

**Lens 2: Block Bootstrap**
500 resampled test paths using 20-day blocks to preserve
return autocorrelation. The Wilcoxon signed-rank test is
applied to each metric distribution. Null hypothesis: the
median performance difference between the winning model
and the CVaR baseline equals zero. Significance level: p
less than 0.05.

**Lens 3: Stress Scenario Testing**
Three crisis periods evaluated:
- COVID-19 Crash from February 2020 to April 2020
- 2022 Federal Reserve Rate Shock, full calendar year 2022
- SVB Banking Crisis from March 2023 to May 2023

---

### Preliminary Results (50 Stocks)

| Metric            | CVaR SIT| Equal Weight         |

| Annualised Return | 15.23% | Reported positive     |
| Sharpe Ratio      | 0.5863 | 0.3801                |
| Max Drawdown      | 29.97% | 38.56%                |
| Sortino Ratio     | 0.8269 | Not reported          |
| Ulcer Index       | 5.15%  | 8.94%                 |


The SIT-CVaR model achieves a 54 percent improvement in
Sharpe ratio over the equal weight benchmark at the 50
stock scale, confirming the original paper's central claim.

The SIT-CVaR model achieves a 54 percent improvement in
Sharpe ratio over the equal weight benchmark at the 50
stock scale, confirming the original paper's central claim.

---

### Requirements
torch>=2.0
iisignature==0.24
numpy<2
pandas
matplotlib
seaborn
scikit-learn
scipy
yfinance
cvxpy
tqdm
jupyter
ipykernel

Install all dependencies:

```bash
pip install torch iisignature "numpy<2" pandas matplotlib
pip install seaborn scikit-learn scipy yfinance cvxpy
pip install tqdm jupyter ipykernel
```

---

### How to Run

```bash
# Clone the repository
git clone https://github.com/bukhondziweni19/PORA-Assignment.git
cd PORA-Assignment

# Install requirements
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open paper_replication.ipynb in Jupyter. Set DATA_POOL
in Cell 1 to 20, 30 or 50. Run Kernel then Restart and
Run All.

Results save automatically to:
- data_paper_20/ for the 20 stock universe
- data_paper_30/ for the 30 stock universe
- data_paper_50/ for the 50 stock universe

---

### Repository Structure
PORA-Assignment/
│
├── paper_replication.ipynb    main notebook (29 cells)
├── requirements.txt           Python dependencies
├── README.md                  this file
└── .gitignore                 excludes large data folders

The data folders (data_paper_20/, data_paper_30/,
data_paper_50/) are generated automatically when the
notebook runs. They are excluded from the repository
by .gitignore to avoid uploading large files.

---

### Reference

Hwang, Y. and Zohren, S. (2025). Signature-Informed
Transformer for Asset Allocation. arXiv preprint
arXiv:2510.03129.

---

### Authors

PORA9X1 Group Project, University of Johannesburg, 2026

Bukho Ndziweni, Sizwe November,
Tshireletso Sethaiso, Bridget Maposa
