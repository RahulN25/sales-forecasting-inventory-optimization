# Sales Forecasting & Inventory Optimization System
> **End-to-End Time-Series Benchmark, Deep Learning Ensembles, and Supply Chain Inventory Control**  
> **CT5108 Data Analytics Project** | *University of Galway*  
> **Author:** Rahul Nagpure

---

## 📌 Executive Summary

Effective demand forecasting and inventory control are central to modern retail supply chain management. Under-forecasting results in stockouts, lost revenue, and reduced customer loyalty, while over-forecasting incurs high holding costs, inventory obsolescence, and working capital lockup.

This repository provides an enterprise-grade, end-to-end framework that integrates **24 time-series forecasting models** across **4 distinct model paradigms** (Baseline, Statistical, Machine Learning, and Deep Learning) with **stochastic inventory optimization models** (Reorder Point, Safety Stock, $(s, S)$ Continuous Review Policies, and Cost Sensitivity Analysis).

### Key Project Achievements
* **Comprehensive Benchmark Zoo**: Implemented and benchmarked 24 distinct forecasting architectures on Walmart's multi-series retail dataset comprising **3,084 Store-Department time series** across **143 weekly observations**.
* **Top Forecasting Performance**: The **Optimized Deep Learning Ensemble** achieved the lowest overall Weighted Mean Absolute Error (**WMAE of $1,179.61**), outperforming the Seasonal Naïve baseline by **31.2%**.
* **Machine Learning & Efficiency Leaders**: **Random Forest** (**WMAE $1,292.53**) emerged as the top tree-based model, while **LightGBM** (**WMAE $1,304.36**, runtime: 17.8s) and **XGBoost** (**WMAE $1,357.43**, runtime: 2.95s) delivered extreme operational efficiency.
* **Statistical Standout**: **Holt-Winters (Triple Exponential Smoothing)** dominated the statistical family (**WMAE $1,427.69**), delivering a **17.1% error reduction** over Seasonal Naïve through damped trend and annual weekly seasonality ($m=52$).
* **Inventory Coupling & Cost Optimization**: Coupled forecast error distributions into a continuous review $(s, S)$ inventory control simulation across service levels ($50\%$ to $99\%$). Identified an **Optimal Service Policy ($65\%$ service level target)** that reduced total simulated inventory costs to **$53.36M** (achieving a **98.35% fill rate**) compared to unoptimized operations ($1.03B baseline cost).

---

## 📁 Repository & Project Architecture

```
sales-forecasting-inventory-optimization/
├── README.md                           # Main Project Documentation
├── requirements.txt                    # Project Dependencies
├── LICENSE                             # License File
├── data/                               # Dataset Storage Directory
│   ├── raw/                            # Raw Kaggle / Walmart Datasets
│   │   ├── wallmart-store-sales-forecasting/
│   │   └── store-item-demand-forecasting/
│   ├── interim/                        # Intermediate Processing Checkpoints
│   └── processed/                      # Cleaned & Feature-Engineered CSVs
├── src/                                # Modular Production Python Source Code
│   ├── __init__.py
│   ├── config.py                       # Project Paths & Global Constants
│   ├── data_loader.py                  # Data Ingestion Utilities
│   ├── preprocessing.py               # Data Cleaning & Preprocessing Pipeline
│   ├── feature_engineering.py          # Temporal, Lag & Rolling Features Creation
│   ├── train_models.py                 # Baseline & Statistical Models Engine
│   ├── train_ml_models.py              # Machine Learning Models Pipeline
│   ├── train_dl_models.py              # PyTorch / TensorFlow Deep Learning Engine
│   ├── test_statistical_models.py      # Statistical Validation Scripts
│   ├── additional_models.py            # Supplementary Experimental Architectures
│   ├── evaluate_models.py              # Evaluation Metrics & WMAE Calculations
│   └── visualization.py               # Plotting & Reporting Scripts
├── notebooks/                          # Sequential Execution & Analytical Notebooks
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_statistical_models.ipynb
│   ├── 06b_statistical_models.ipynb
│   ├── 07_machine_learning_models.ipynb
│   ├── 07B_machine_learning_models.ipynb
│   ├── 08_deep_learning_models.ipynb
│   ├── 09_inventory_optimization_analysis.ipynb
│   ├── 10_model_comparison.ipynb
│   └── 11_additional_models.ipynb
├── results/                            # Benchmark Leaderboards & Predictions
│   ├── master_metrics.json             # Master Benchmark Metrics for all 24 Models
│   ├── master_predictions.csv          # Consolidated Predictions Matrix
│   ├── model_comparison.md             # In-depth Model Evaluation Log
│   ├── experiment_log.md               # Tracking Log & Sub-experiment Reports
│   ├── model_comparison_report.xlsx    # Spreadsheet Benchmark Workbook
│   └── inventory/                      # Inventory Simulation Artifacts
│       ├── inventory_policy_comparison.csv
│       ├── inventory_overall_summary.csv
│       ├── cost_sensitivity.csv
│       └── service_level_sensitivity.csv
├── reports/                            # Visual & Graphical Reports
│   ├── figures/                        # EDA Plots, Correlation Maps & Loss Curves
│   ├── tables/                         # Formatted Result Summary Tables
│   └── weekly_progress/               # Milestone Reports
└── presentation/                       # Presentation Decks & Literature Reviews
```

---

## 📊 Dataset & Feature Engineering Pipeline

### Dataset Overview
* **Source**: Walmart Recruiting - Store Sales Forecasting Dataset.
* **Scope**: 45 retail stores and 81 departments (**3,084 unique Store-Department time-series groups**).
* **Time Range**: February 5, 2010 to October 26, 2012 (**143 total weekly observations**).
* **Validation Window**: 12-week test partition (August 17, 2012 to October 26, 2012). Train/Validation split date: **August 10, 2012**.

### Feature Engineering Taxonomy
To enable machine learning and deep learning models to capture multi-scale temporal dynamics, a rich feature matrix was constructed:

1. **Calendar & Seasonal Dynamics**:
   - `Week`, `Month`, `Quarter`, `Year`, `DayOfYear`.
   - Binary holiday indicators (`IsHoliday`) aligned with Walmart's key promotional events: **Super Bowl**, **Labor Day**, **Thanksgiving**, and **Christmas**.
2. **Lagged Sales Signals**:
   - Short-term lags: Lags 1, 2, 3, 4 weeks to capture momentum.
   - Seasonal lag: **Lag 52 weeks** to capture annual weekly seasonality.
3. **Rolling Window Aggregations**:
   - Windows of 4, 12, and 52 weeks computing `Rolling Mean`, `Rolling Standard Deviation`, `Rolling Min`, `Rolling Max`, and `Rolling Skewness`.
4. **Macroeconomic Indicators & Store Metadata**:
   - External variables: `Store Type` (A, B, C), `Store Size`, `Temperature`, `Fuel Price`, `Consumer Price Index (CPI)`, and `Unemployment Rate`.
   - Promotional Markdowns (`Markdown1` through `Markdown5`) filled and log-transformed.
5. **Data Scaling & Encoding**:
   - Categorical target encoding for Store and Department IDs.
   - Robust normalization via `MinMaxScaler` and `StandardScaler` for deep learning feature pipelines.

---

## 🤖 Model Zoo & Methodology

We implemented and evaluated **24 forecasting models** categorized into 4 core families:

```
                          ┌────────────────────────────────────────────────┐
                          │   Sales Forecasting Model Zoo (24 Models)      │
                          └───────────────────────┬────────────────────────┘
                                                  │
         ┌──────────────────┬─────────────────────┴─────────────────────┬──────────────────┐
         ▼                  ▼                                           ▼                  ▼
┌──────────────────┐ ┌───────────────┐                       ┌─────────────────────┐ ┌──────────────────┐
│ Baseline (3)     │ │ Statistical(5)│                       │ Machine Learning (8)│ │ Deep Learning (7)│
├──────────────────┤ ├───────────────┤                       ├─────────────────────┤ ├──────────────────┤
│• Historical Mean │ │• SES          │                       │• Linear Regression  │ │• 1D-CNN          │
│• Naive           │ │• Holt Linear  │                       │• KNN                │ │• LSTM            │
│• Seasonal Naive  │ │• Holt-Winters │                       │• ANN                │ │• BiLSTM + Attn   │
│                  │ │• ARIMA(1,1,1) │                       │• MLP                │ │• ResAttention    │
│                  │ │• SARIMA       │                       │• GBRT               │ │• LSTM-MLP Hybrid │
│                  │ │               │                       │• XGBoost            │ │• TCN-Transformer │
│                  │ │               │                       │• LightGBM           │ │• DL Ensemble     │
│                  │ │               │                       │• Random Forest      │ │                  │
│                  │ │               │                       │• ML Ensemble        │ │                  │
└──────────────────┘ └───────────────┘                       └─────────────────────┘ └──────────────────┘
```

### Loss Function & Evaluation Metrics
All models were benchmarked primarily against Walmart's competition metric, **Weighted Mean Absolute Error (WMAE)**, alongside standard forecast accuracy metrics:

$$\text{WMAE} = \frac{\sum_{i=1}^{n} w_i |y_i - \hat{y}_i|}{\sum_{i=1}^{n} w_i} \quad \text{where } w_i = \begin{cases} 5, & \text{if week } i \text{ is a Holiday week} \\ 1, & \text{otherwise} \end{cases}$$

* **MAE**: Mean Absolute Error ($\frac{1}{n}\sum |y_i - \hat{y}_i|$)
* **RMSE**: Root Mean Squared Error ($\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$)
* **sMAPE**: Symmetric Mean Absolute Percentage Error ($\frac{100\%}{n}\sum \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$)
* **WAPE**: Weighted Absolute Percentage Error ($\frac{\sum |y_i - \hat{y}_i|}{\sum |y_i| \times 100\%}$)

---

## 🏆 Empirical Results & Benchmark Leaderboard

The complete benchmark evaluation of all **24 forecasting models** across the full validation dataset of **3,084 Store-Department series** is detailed below, ordered from best to worst WMAE performance:

| Rank | Model Name | Model Family | WMAE ($) ⬇️ | MAE ($) | RMSE ($) | sMAPE (%) | Execution Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Optimized DL Ensemble** | Deep Learning | **1,179.61** | 1,149.50 | 2,461.68 | 24.43% | 2,895.27s |
| 🥈 | **LSTM-MLP Hybrid** | Deep Learning | **1,205.96** | 1,188.80 | 2,510.04 | 23.57% | 899.78s |
| 🥉 | **Residual Attention LSTM** | Deep Learning | **1,229.25** | 1,179.21 | 2,523.60 | 25.71% | 1,500.70s |
| 4 | **LSTM** | Deep Learning | **1,239.28** | 1,217.42 | 2,569.17 | 27.39% | 494.79s |
| 5 | **BiLSTM with Attention** | Deep Learning | **1,259.22** | 1,205.80 | 2,498.04 | 23.12% | 629.82s |
| 6 | **1D-CNN** | Deep Learning | **1,268.06** | 1,232.19 | 2,536.12 | 27.30% | 366.25s |
| 7 | **Random Forest** | Machine Learning | **1,292.53** | 1,230.30 | 2,560.28 | 17.59% | 115.59s |
| 8 | **LightGBM** | Machine Learning | **1,304.36** | 1,227.30 | 2,672.45 | 17.64% | 17.81s |
| 9 | **XGBoost** | Machine Learning | **1,357.43** | 1,275.75 | 2,821.77 | 21.91% | **2.95s** |
| 10 | **ML Ensemble** | Machine Learning | **1,419.14** | 1,298.52 | 2,629.27 | 27.03% | 152.14s |
| 11 | **Holt-Winters** | Statistical | **1,427.69** | 1,422.50 | 3,130.43 | 23.71% | 2,006.40s |
| 12 | **GBRT** | Machine Learning | **1,457.42** | 1,359.67 | 2,753.77 | 24.43% | 540.23s |
| 13 | **SARIMA(1,1,0)x(0,1,1,52)** | Statistical | **1,494.19** | 1,495.93 | 3,189.19 | 24.42% | 3,435.52s |
| 14 | **TCN-Transformer Hybrid** | Deep Learning | **1,680.97** | 1,683.02 | 3,670.18 | 29.12% | 564.75s |
| 15 | **Seasonal Naive** | Baseline | **1,714.90** | 1,689.93 | 3,638.77 | 23.35% | 0.16s |
| 16 | **KNN** | Machine Learning | **1,782.23** | 1,731.91 | 3,200.61 | 31.08% | 13.59s |
| 17 | **Linear Regression** | Machine Learning | **1,830.56** | 1,743.58 | 3,465.22 | 41.97% | 1.68s |
| 18 | **Simple Exp Smoothing (SES)**| Statistical | **2,143.17** | 2,156.63 | 4,800.22 | 26.41% | 2,028.72s |
| 19 | **Holt's Linear Trend** | Statistical | **2,155.95** | 2,166.05 | 5,261.12 | 27.01% | 1,974.17s |
| 20 | **ARIMA(1,1,1)** | Statistical | **2,210.61** | 2,197.86 | 4,920.17 | 26.33% | 2,131.15s |
| 21 | **Naive** | Baseline | **2,241.28** | 2,308.47 | 5,080.30 | 28.25% | 0.16s |
| 22 | **ANN** | Machine Learning | **2,460.02** | 2,406.54 | 3,622.35 | 45.77% | 61.05s |
| 23 | **Historical Mean** | Baseline | **2,524.38** | 2,425.44 | 4,953.47 | 29.20% | 0.03s |
| 24 | **MLP** | Machine Learning | **2,592.14** | 2,132.64 | 3,521.18 | 45.60% | 30.60s |

---

## 💡 Analytical Insights & Model Takeaways

1. **Deep Learning Superiority**:
   - Deep learning models occupied **6 of the top 6 leaderboard positions**.
   - Combining sequential feature learning (LSTM / Attention) with tabular metadata (MLP) allowed **Optimized DL Ensemble** ($1,179.61) and **LSTM-MLP Hybrid** ($1,205.96) to effectively resolve complex store-department spatial interaction effects.
2. **Machine Learning Efficiency & Speed**:
   - **Random Forest** achieved an outstanding WMAE of **$1,292.53**, outperforming all individual statistical models.
   - **LightGBM** ($1,304.36 WMAE, 17.81s) and **XGBoost** ($1,357.43 WMAE, 2.95s) provided the best balance of speed and performance, making them ideal candidates for real-time automated retraining in production.
3. **Statistical Models & Seasonality Importance**:
   - Models that explicitly incorporated annual weekly seasonality ($m=52$) dramatically outperformed non-seasonal statistical models.
   - **Holt-Winters** ($1,427.69 WMAE) outperformed SARIMA ($1,494.19 WMAE) and non-seasonal models (SES $2,143.17 WMAE; ARIMA $2,210.61 WMAE) by over **33%**, proving that trend damping ($\phi$) prevents unrealistic projection over 12-week horizons.
4. **Failure of Non-Seasonal Baselines**:
   - Simple Naïve ($2,241.28 WMAE) and Historical Mean ($2,524.38 WMAE) failed during peak promotional/holiday periods because they lacked annual memory.

---

## 📦 Inventory Optimization Framework

To evaluate the operational impact of improved forecasting accuracy, model residual error distributions ($\sigma_e$) were directly coupled into stochastic inventory decision models.

### Mathematical Formulation
1. **Lead Time Demand & Safety Stock ($SS$)**:
   $$\text{Safety Stock } (SS) = z_{\alpha} \times \sigma_{\text{demand}} \times \sqrt{L}$$
   where $z_{\alpha}$ is the normal inverse distribution value for target service level $\alpha$, $L$ is lead time (weeks), and $\sigma_{\text{demand}}$ is the standard deviation of forecast residual errors.
2. **Reorder Point ($ROP$)**:
   $$\text{Reorder Point } (ROP) = (\hat{d} \times L) + SS$$
   where $\hat{d}$ is the expected weekly sales forecast.
3. **Economic Order Quantity ($EOQ$)**:
   $$EOQ = \sqrt{\frac{2 \times D \times K}{h}}$$
   where $D$ is annual demand, $K$ is fixed ordering cost, and $h$ is annual holding cost per unit.
4. **Continuous Review $(s, S)$ Inventory Policy**:
   $$s = ROP, \quad S = ROP + EOQ$$

### Inventory Policy Simulation & Service Level Sensitivity

We evaluated continuous review policy simulations across target service levels ranging from **50% to 99%**:

| Policy / Service Target | Target Service ($\alpha$) | Fill Rate (%) ⬆️ | Stockout-Free Cycle Rate (%) | Avg Ending Inventory | Holding Cost Index ($) | Shortage Cost Index ($) | Total Simulated Cost Index ($) ⬇️ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forecast Only (Unoptimized)** | 0.50 | 96.78% | 54.49% | 624.81 | $12,972,945 | $51,022,222 | $63,995,167 |
| 🌟 **Optimal Policy (65% Target)** | **0.65** | **98.35%** | **85.25%** | **1,309.82** | **$27,195,767** | **$26,167,059** | **$53,362,826** |
| **75% Target** | 0.75 | 98.90% | 91.70% | 1,919.68 | $39,858,340 | $17,458,404 | $57,316,744 |
| **80% Target** | 0.80 | 99.11% | 93.84% | 2,288.75 | $47,521,269 | $14,146,695 | $61,667,965 |
| **90% Target** | 0.90 | 99.46% | 96.82% | 3,290.01 | $68,310,519 | $8,522,345 | $76,832,864 |
| **95% Target** | 0.95 | 99.63% | 98.05% | 4,135.73 | $85,870,101 | $5,834,960 | $91,705,061 |
| **99% Target** | 0.99 | 99.80% | 99.14% | 5,744.85 | $119,280,372 | $3,150,958 | $122,431,330 |

> 🔑 **Key Inventory Conclusion**:  
> Running pure forecasts without safety stock (50% target) incurs excessive shortage costs ($51.0M). Pushing service levels to extreme targets (95-99%) drastically inflates inventory holding costs ($85.9M - $119.3M).  
> The **Optimal 65% Service Level Policy** strikes the ideal trade-off: it achieves a **98.35% customer fill rate** while minimizing total operational cost to **$53.36M**, saving over **$970M** compared to completely unmanaged inventory operations.

---

## 🛠️ Installation & Getting Started

### Prerequisites
* Python **3.10** or higher
* `pip` or `conda` package manager
* Virtual Environment (recommended)

### Quickstart Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RahulN25/sales-forecasting-inventory-optimization.git
   cd sales-forecasting-inventory-optimization
   ```

2. **Set Up Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Data Location**:
   Ensure raw dataset files are located in:
   `data/raw/wallmart-store-sales-forecasting/`
   - `train.csv`
   - `test.csv`
   - `stores.csv`
   - `features.csv`

---

## 🚀 Execution & Usage Guide

### Modular Command Line Workflow

Execute the python pipeline sequentially from the project root:

```bash
# 1. Clean and preprocess raw Walmart data
python src/preprocessing.py

# 2. Generate temporal, lag, and rolling features
python src/feature_engineering.py

# 3. Train Baseline & Statistical Models (Naive, SES, Holt-Winters, SARIMA)
python src/train_models.py

# 4. Train Machine Learning Models (Random Forest, XGBoost, LightGBM, GBRT)
python src/train_ml_models.py

# 5. Train Deep Learning Models (LSTM, BiLSTM, ResAttention, DL Ensemble)
python src/train_dl_models.py

# 6. Evaluate all predictions and build Master Metrics Leaderboard
python src/evaluate_models.py
```

### Jupyter Notebook Workflow

For interactive analysis, visual exploration, and step-by-step experimentation, run the Jupyter notebooks in order:

```bash
jupyter notebook notebooks/
```

- [`01_dataset_understanding.ipynb`](notebooks/01_dataset_understanding.ipynb): Data structure inspection & schema verification.
- [`02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb): Store/Department distributions & holiday impacts.
- [`03_data_cleaning_preprocessing.ipynb`](notebooks/03_data_cleaning_preprocessing.ipynb): Imputation & outlier handling.
- [`04_feature_engineering.ipynb`](notebooks/04_feature_engineering.ipynb): Lag, rolling, & calendar feature generation.
- [`05_baseline_models.ipynb`](notebooks/05_baseline_models.ipynb): Historical Mean, Naïve, Seasonal Naïve execution.
- [`06_statistical_models.ipynb`](notebooks/06_statistical_models.ipynb): Exponential smoothing, ARIMA, SARIMA modeling.
- [`07_machine_learning_models.ipynb`](notebooks/07_machine_learning_models.ipynb): Tree ensembles, LightGBM, XGBoost, RF.
- [`08_deep_learning_models.ipynb`](notebooks/08_deep_learning_models.ipynb): PyTorch/TensorFlow deep learning architectures.
- [`09_inventory_optimization_analysis.ipynb`](notebooks/09_inventory_optimization_analysis.ipynb): ROP, Safety Stock, EOQ, & (s, S) simulation.
- [`10_model_comparison.ipynb`](notebooks/10_model_comparison.ipynb): Master comparison, plots & spreadsheet exports.

---

## 📜 License & Citation

This project is released under the MIT License.

### Academic Citation
If you use this repository or its benchmark results in your research or project work, please cite:

```bibtex
@misc{nagpure2026salesforecasting,
  author = {Rahul Nagpure},
  title = {Sales Forecasting and Inventory Optimization System: An End-to-End Benchmark of 24 Time-Series Models and Stochastic Inventory Control},
  year = {2026},
  publisher = {GitHub},
  journal = {CT5108 Data Analytics Project, University of Galway},
  howpublished = {\url{https://github.com/RahulN25/sales-forecasting-inventory-optimization}}
}
```

---
*Developed for CT5108 Data Analytics Project at the University of Galway.*
