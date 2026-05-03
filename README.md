# Signature-Informed Transformer for Portfolio Optimisation
## Replication and Extension — PORA9X1 Group Project

### Overview

This repository contains the full replication and extension of **Hwang and Zohren (2025)** — *"Signature-Informed Transformer for Asset Allocation"* (arXiv:2510.03129), completed as part of the PORA9X1 module at the **University of Johannesburg** under the supervision of **Prof. Jules Mba**.

The original paper proposes a Transformer-based portfolio optimisation framework that uses **path signatures** of asset return histories as input features and trains the model end-to-end using a **CVaR risk objective** directly on portfolio returns — bypassing the traditional return forecasting step entirely.

---

### Research Extension

Our extension asks:

> *Which drawdown-based training objective produces the best portfolio outcomes within the SIT framework — and is that result robust across different time periods, return paths and market crises?*

We conduct a **horse race** comparing five training objectives:

| Model | Objective | Reference |
|---|---|---|
| CVaR | Conditional Value at Risk | Rockafellar & Uryasev (2000) |
| AvgDD | Average Drawdown | — |
| CDaR | Conditional Drawdown at Risk | Chekhlov et al. (2005) |
| UlcerIndex | Root Mean Square Drawdown | Martin & McCann (1989) |
| SmoothMaxDD | Differentiable Max Drawdown | Magdon-Ismail & Atiya (2004) |

---

### Dataset

The original paper used institutional data across diverse equity universes. We replicate using **Yahoo Finance** adjusted closing prices for a carefully selected universe of **50 S&P 500 stocks** spanning five sectors — Technology, Financials, Healthcare, Energy and Consumer Staples — covering the period **January 2000 to December 2024**.

Four stocks from the original selection were replaced due to delistings or insufficient history:

| Original | Replacement | Reason |
|---|---|---|
| BCR | BSX | BCR delisted after BDX acquisition 2017 |
| K | CPB | K acquired by Mars August 2024 |
| MMC | CB | MMC data gaps on Yahoo Finance around 2000 |
| PSX | OXY | PSX only listed from May 2012 |

The data split follows the original paper exactly:

| Split | Period |
|---|---|
| Training | 2000-01-01 to 2016-12-31 |
| Validation | 2017-01-01 to 2019-12-31 |
| Test | 2020-01-01 to 2024-12-31 |

---

### Notebook Structure

The project is contained in a single notebook `paper_replication.ipynb` with 27 cells organised into five sections:

SECTION 1 — DATA COLLECTION          (Cells 0–8)
Data download, cleaning, log return computation,
train/val/test splitting and visualisation

SECTION 2 — PATH SIGNATURES           (Cells 9–11)
Sliding window signature computation, normalisation,
return alignment and feature saving

SECTION 3 — MODEL AND HORSE RACE      (Cells 12–22)
SIT architecture definition, DataLoaders,
five loss functions, training loop, evaluation,
benchmarks, horse race, composite ranking,
cumulative wealth plots and training curves

SECTION 4 — ROBUSTNESS TESTING        (Cells 23–26)
Lens 1: Walk-forward validation (9 expanding windows)
Lens 2: Stress scenario testing (COVID, Rate Shock, SVB)

SECTION 5 — MASTER SUMMARY            (Cell 27)
Consolidated robustness verdict across all three lenses

---

### Key Design Decisions

- **Signature order reduced from 3 to 2** for the 50-stock universe due to memory constraints. At order 3 the feature vector contains 127,550 terms which exceeded available RAM. Order 2 produces 2,550 features and captures all pairwise cross-asset interactions.
- **Dynamic model assignment** — the winning model is determined automatically from the horse race composite ranking and saved to `winners.json`. All robustness cells load from this file rather than hardcoding model names.
- **DATA_POOL variable** — changing `DATA_POOL` in Cell 1 to 30, 40 or 50 reruns the entire pipeline on a different asset universe with no other changes required.

---

### Performance Metrics

All models are evaluated on nine metrics aligned with the original paper:

- Annualised Return
- Annualised Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Ulcer Index
- Time Underwater
- Turnover

---

### Robustness Framework

Three robustness lenses are applied to the winning model:

**Lens 1 — Walk-Forward Validation**
Nine expanding training windows from 2008 to 2016 test whether the winning model consistently outperforms across different historical periods.

**Lens 2 — Stress Scenario Testing**
Performance during three specific market crises is evaluated:
- COVID-19 Crash (February–April 2020)
- 2022 Federal Reserve Rate Shock (January–December 2022)
- SVB Banking Crisis (March–May 2023)

---

### Requirements
python=3.9
torch>=2.0
iisignature=0.24
numpy<2
pandas
matplotlib
seaborn
scikit-learn
scipy
yfinance
tqdm
jupyter

Install via conda:

```bash
conda create -n sit_portfolio python=3.9
conda activate sit_portfolio
pip install torch iisignature numpy pandas matplotlib seaborn scikit-learn scipy yfinance tqdm jupyter
```

---

### How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/sit-portfolio-replication.git
cd sit-portfolio-replication

# Activate environment
conda activate sit_portfolio

# Launch Jupyter
jupyter notebook

# Open paper_replication.ipynb
# Set DATA_POOL = 30, 40 or 50 in Cell 1
# Run Kernel → Restart & Run All
```

---

### Repository Structure
sit-portfolio-replication/
│
├── paper_replication.ipynb     ← main notebook
│
├── data_paper_50/              ← generated when you run
│   ├── clean_prices.csv
│   ├── log_returns.csv
│   ├── train_returns.csv
│   ├── val_returns.csv
│   ├── test_returns.csv
│   ├── config.json
│   ├── winners.json
│   ├── horse_race_results.csv
│   ├── composite_ranking.csv
│   ├── signatures/
│   ├── models/
│   └── robustness/
│
└── README.md

> **Note:** The `data_paper_50/` folder is generated automatically when you run the notebook. It is not included in the repository. Add it to `.gitignore` to avoid uploading large data files.

---

### Reference

Hwang, Y. and Zohren, S. (2025). *Signature-Informed Transformer for Asset Allocation*. arXiv:2510.03129.

---

### Authors

PORA9X1 Group Project — University of Johannesburg, 2025






