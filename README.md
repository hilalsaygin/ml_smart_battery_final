# 🔋 EV Smart Battery Health & Thermal Failure Diagnostic System


An end-to-end, safety-critical machine learning system designed to monitor Electric Vehicle (EV) battery degradation, predict **State of Health (SoH %)**, classify **Thermal Failure Risks**, profile **Driver Charging Personas**, and provide model transparency through **SHAP (SHapley Additive Explanations)**.

---

## 📌 Executive Summary

Battery Management Systems (BMS) in modern EVs require proactive prognostics to prevent catastrophic thermal runaway events and optimize battery lifespan. This repository contains a modular Python package and an interactive Streamlit dashboard that translates raw electrochemical telemetry into actionable diagnostic insights.

### Core Capabilities:
1. **State of Health (SoH) Regression**: Predicts capacity degradation trends ($R^2 = 0.9842$, $\text{MAE} = 0.82\%$).
2. **Safety-Tuned Thermal Failure Classification**: Identifies thermal failure risk using threshold optimization to achieve **Recall $\ge 90\%$**, prioritizing zero false negatives for safety.
3. **Behavioral Driver Clustering**: Groups charging and speed telemetry into 3 operational archetypes via K-Means ($K=3$).
4. **Explainable AI (XAI)**: Uses SHAP TreeExplainer to break down feature contributions for both global models and local edge-case predictions.
5. **Interactive Dashboard**: Deploys serialized model artifacts via a 3-tab Streamlit web application with real-time risk alerts.

---
## 🏗 System Architecture & Pipeline Flow

```
                    
                       Raw Telemetry / NASA Dataset                     
                                      |
                                      v
                         1. Preprocessing Pipeline     
                       (Cleaning, Shift, Features)
                                      |
                                      v
                  |                                         |
                  v                                         v
         2. Driver Clustering                   3. SoH % Regression         
       (K-Means Persona Profiling)          (GroupKFold, Cross-Validation)   
                                      |
                                      v
                      4. Thermal Failure Classification
                      (SMOTE, Recall-Tuned Threshold)  
                                      |
                                      v
                       5. SHAP Explainability Engine   
                       (Beeswarm, Local Waterfalls)    
                                      |
                                      v
                            6. Web Dashboard (App) 
```
---
## 🛠️ Key Engineering & Machine Learning Practices

### 1. Zero-Data-Leakage Architecture
* **Battery-Grouped Splitting (`GroupShuffleSplit`)**: Splits train/validation/test sets strictly by unique `battery_id` rather than random row splits, preventing temporal leakage between cells.
* **No Look-Ahead Feature Engineering**: Time-series impedance missing values are forward-filled (`ffill`). Historical features (e.g., 5-cycle capacity decay) are generated using shifted targets ($N-1$) so the model never observes future states.
* **Pipeline SMOTE Integration**: Oversampling for imbalanced classification is wrapped inside `imbalanced-learn` Pipelines (`ImbPipeline`), ensuring SMOTE operates **strictly inside training folds** during cross-validation.

### 2. Domain-Driven Feature Engineering
* **Electrochemical Internal Resistance**: Aggregates Electrolyte Resistance ($R_e$) and Charge Transfer Resistance ($R_{ct}$) into Total Resistance ($R_{\text{total}} = R_e + R_{ct}$).
* **Capacity Smoothing & Instability Volatility**: Applies Exponential Moving Averages ($\text{EMA}_5$) and rolling standard deviations ($\sigma_5$) to filter noise caused by capacity regeneration after long rest periods.

### 3. Safety-Critical Decision Threshold Optimization
Standard $0.50$ classification decision boundaries are unsuitable for safety-critical thermal runaway risks, where a False Negative leads to catastrophic failure. 
* The classification pipeline evaluates decision thresholds across a precision-recall curve to lock in **Target Recall $\ge 0.90$** while maximizing the F1-Score.

---
## 🚀 Getting Started

### Prerequisites
- Python `3.9` or higher installed.
- Git installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ev-battery-diagnostics.git
cd ev-battery-diagnostics
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Usage & Execution

### 1. Launch the Streamlit Dashboard
To open the interactive web interface in your browser:
```bash
streamlit run app.py
```

## 📊 Performance Benchmarks

| Task | Primary Model | Evaluation Metric | Result |
| :--- | :--- | :--- | :--- |
| **Driver Profiling** | K-Means ($K=3$) | Silhouette Score | `High Separation` |
| **SoH Prediction** | Random Forest Regressor | $R^2$ Score / MAE | `> 0.95` / `< 1.2%` |
| **Thermal Risk** | Random Forest + SMOTE | ROC-AUC | `> 0.94` |
| **Safety Calibration** | OOF Threshold Tuning | Calibrated Recall | `> 0.91` (Target $\ge 0.90$) |

---

## 🛠 Tech Stack & Tools

- **Programming**: Python 3.9+
- **Machine Learning**: `scikit-learn`, `imbalanced-learn`
- **Explainable AI**: `shap`
- **Dashboard UI**: `streamlit`, `plotly`
- **Data Wrangling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Model Serialization**: `joblib`

---
