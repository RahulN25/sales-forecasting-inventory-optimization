# Model Comparison and Experiment Tracking Log

This document serves as the comprehensive, detailed tracking repository for all sales forecasting models trained and evaluated. It tracks exact configurations, execution runtimes, error metrics (WMAE, MAE, RMSE, MAPE, sMAPE), and structural insights for both baseline and statistical models across three distinct evaluation datasets.

---

## 1. Global Experiment Configurations
* **Dataset**: Walmart Weekly Store Sales Dataset
* **Historical Data Range**: 2010-02-05 to 2012-10-26 (143 total weeks)
* **Validation Window**: 12 weeks (2012-08-17 to 2012-10-26)
* **Train/Validation Split Date**: 2012-08-10
* **Evaluation Loss Metric**: Walmart Weighted Mean Absolute Error (WMAE)
  $$\text{WMAE} = \frac{\sum_{i=1}^{n} w_i |y_i - \hat{y}_i|}{\sum_{i=1}^{n} w_i}$$
  where $w_i = 5$ if week $i$ is a designated Holiday week, and $w_i = 1$ otherwise.

---

## 2. Model Implementations & Configurations

### A. Baseline Models
1. **Historical Mean**:
   * **Formula**: $\hat{y}_{t+h|t} = \bar{y}_{\text{train}}$ (group-level historical average).
   * **Purpose**: Establish simple static central tendency benchmark.
2. **Naive**:
   * **Formula**: $\hat{y}_{t+h|t} = y_t$ (uses the last observed sales value in the training series).
   * **Purpose**: Benchmark behavior under random walk assumption.
3. **Seasonal Naive**:
   * **Formula**: $\hat{y}_{t+h|t} = y_{t+h-52}$ (uses the sales value from exactly 52 weeks ago).
   * **Purpose**: Benchmark capability under strict annual weekly seasonality.

### B. Statistical Models
1. **Simple Exponential Smoothing (SES)**:
   * **Level Equation**: $\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}$
   * **Parameters**: Level smoothing coefficient $\alpha \in (0, 1]$ optimized dynamically per group using MLE.
2. **Holt's Linear Trend (Damped)**:
   * **Equations**:
     * $\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + \phi b_{t-1})$
     * $b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)\phi b_{t-1}$
   * **Parameters**: Level smoothing $\alpha$, trend smoothing $\beta$, and trend damping coefficient $\phi$ optimized dynamically per group using MLE.
3. **Holt-Winters (Triple Exponential Smoothing)**:
   * **Type**: Additive trend and additive seasonal components.
   * **Equations**:
     * $\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + \phi b_{t-1})$
     * $b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)\phi b_{t-1}$
     * $s_t = \gamma(y_t - \ell_{t-1} - \phi b_{t-1}) + (1 - \gamma)s_{t-m}$
   * **Parameters**: Level smoothing $\alpha$, trend smoothing $\beta$, seasonal smoothing $\gamma$, damping coefficient $\phi$, and seasonal period $m=52$ weeks. Optimized per group via L-BFGS-B.
4. **ARIMA(1, 1, 1)**:
   * **Formula**: $\Phi_1(B)(1-B) y_t = \Theta_1(B)\epsilon_t$
   * **Parameters**: Autoregressive order $p=1$, differencing $d=1$, moving average order $q=1$ fit per group.
5. **Seasonal ARIMA (SARIMA(1, 1, 0) x (0, 1, 1, 52))**:
   * **Formula**: $\Phi_1(B)(1-B)(1-B^{52}) y_t = \tilde{\Theta}_1(B^{52})\epsilon_t$
   * **Configuration**: Fits a seasonal autoregressive process with simple differencing disabled (`simple_differencing=False`) to integrate predictions directly to level sales values, resolving model outputs differencing scale issues.

---

## 3. Detailed Results & Experiment Logs

### Experiment 1: Primary Evaluation (All Store-Department Groups)
* **Dataset Scope**: All 3,084 unique Store-Department groups in the validation partition.
* **Objective**: Evaluate model capability under full-data complexity, representing a mixture of high-volume, low-volume, and zero-sales intermittent demand.

| Model | WMAE ($) | MAE ($) | RMSE ($) | MAPE (%) | sMAPE (%) | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Holt-Winters** | **1,427.69** | **1,422.50** | **3,130.43** | 289.75 | 23.71 | ~33m (Parallel) |
| **SARIMA(1,1,0)x(0,1,1,52)** | **1,494.19** | **1,495.93** | **3,189.19** | 443.87 | 24.42 | ~57m (Parallel) |
| **Seasonal Naive** | **1,723.13** | **1,698.02** | **3,658.17** | 2400.12 | 23.46 | 0.13 seconds |
| **Simple Exp Smoothing (SES)** | **2,143.17** | **2,156.63** | **4,800.22** | 236.20 | 26.41 | ~33m (Parallel) |
| **Holt's Linear Trend** | **2,155.95** | **2,166.05** | **5,261.12** | 254.83 | 27.01 | ~33m (Parallel) |
| **ARIMA(1,1,1)** | **2,210.61** | **2,197.86** | **4,920.17** | 449.93 | 26.33 | ~35m (Parallel) |
| **Naive** | **2,241.28** | **2,308.47** | **5,080.30** | 416.67 | 28.25 | 0.16 seconds |
| **Historical Mean** | **2,524.38** | **2,425.44** | **4,953.47** | 2800.84 | 29.20 | 0.03 seconds |

---

### Experiment 2: 300 High-Volume Groups Evaluation
* **Dataset Scope**: Subset of 300 Store-Department groups with $\ge 104$ weeks of history and highest total sales.
* **Objective**: Evaluate model scaling and computational feasibility on a larger high-volume subset.

| Model | WMAE ($) | MAE ($) | RMSE ($) | MAPE (%) | sMAPE (%) | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Holt-Winters** | **4,538.40** | **4,557.37** | **6,930.41** | 7.30 | 7.49 | 17.2 seconds |
| **SARIMA(1,1,0)x(0,1,1,52)** | **4,721.37** | **4,761.32** | **7,166.81** | 7.61 | 7.82 | 168.4 seconds |
| **Simple Exp Smoothing (SES)** | **5,842.20** | **5,639.89** | **7,977.69** | 8.90 | 8.86 | 29.5 seconds |
| **Holt's Linear Trend** | **5,882.53** | **5,679.11** | **8,026.30** | 8.91 | 8.91 | 22.7 seconds |
| **ARIMA(1,1,1)** | **5,915.22** | **5,626.12** | **7,813.31** | 9.12 | 8.85 | 8.7 seconds |
| **Seasonal Naive** | **5,969.34** | **5,856.30** | **8,924.13** | 9.14 | 9.07 | 0.04 seconds |

---

### Experiment 3: Controlled 50 High-Volume Groups Evaluation
* **Dataset Scope**: Subset of 50 Store-Department groups with $\ge 104$ weeks of history and highest total sales.
* **Objective**: Establish comparison benchmark with statistical models under identical parameters.

| Model | WMAE ($) | MAE ($) | RMSE ($) | MAPE (%) | sMAPE (%) | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Holt-Winters** | **6,838.89** | **6,753.86** | **9,852.28** | 6.35 | 6.50 | 1.37 seconds |
| **SARIMA(1,1,0)x(0,1,1,52)** | **7,375.57** | **7,465.79** | **11,072.19** | 7.06 | 7.31 | 62.2 seconds |
| **ARIMA(1,1,1)** | **9,148.29** | **8,670.96** | **11,257.67** | 7.88 | 7.93 | 0.80 seconds |
| **Simple Exp Smoothing (SES)** | **9,166.11** | **8,779.46** | **11,333.78** | 8.00 | 8.04 | 1.99 seconds |
| **Holt's Linear Trend** | **9,327.07** | **8,889.63** | **11,473.64** | 8.05 | 8.12 | 0.46 seconds |
| **Seasonal Naive** | **10,154.41** | **9,921.17** | **14,704.32** | 9.29 | 8.80 | 0.01 seconds |

---

## 4. Key Experimental Conclusions & Structural Insights

1. **Holt-Winters Superiority**:
   * Holt-Winters is the undisputed winner across **all** three subsets. On the full dataset, it reduces error relative to the Seasonal Naïve baseline by **17.1%** (WMAE drops from **1,723.13** to **1,427.69**).
   * Damped trend parameters prevent the linear component from projecting unrealistic growth trends over the 12-week horizon, which is critical during retail transitions.
2. **Seasonal vs. Non-Seasonal Performance Gap**:
   * Models that explicitly incorporate seasonal cycles (Holt-Winters, SARIMA, Seasonal Naïve) outperform non-seasonal models (SES, ARIMA, Holt) by a massive margin. Non-seasonal models struggle because they lack memory of annual peak holiday spikes, falling back to recent baseline averages.
3. **Differencing Integration (SARIMA)**:
   * By setting `simple_differencing=False` in statsmodels' `SARIMAX`, predictions are properly integrated back to original sales values instead of remaining differenced sales level. This brings SARIMA error rates down to competitive runner-up ranges (~1,494 WMAE).
4. **Computational Performance & Scalability**:
   * Holt-Winters runs extremely quickly (fitting 300 groups takes only 17.2s). SARIMA is much more computationally expensive (~3m on 300 groups), making Holt-Winters the most operationally viable model for large-scale enterprise integration.
