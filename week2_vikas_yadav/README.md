# Tesla Global Deliveries & Pricing — End-to-End ML Pipeline
### Week 2 Assignment | Vikas Yadav

---

## 📌 Project Overview
An end-to-end Machine Learning pipeline built on the Tesla Global Deliveries Dataset (2015–2025), covering data preprocessing, exploratory data analysis, feature engineering, regression modeling, hyperparameter tuning, time-series forecasting, and model explainability.

---

## 📂 Dataset
| Property | Detail |
|---|---|
| **File** | `tesla_deliveries_dataset_2015_2025.csv` |
| **Period** | 2015 – 2025 |
| **Records** | 2,640 rows × 12 columns |
| **Target Variable** | `Estimated_Deliveries` |

---

## ⚙️ Setup

### Install Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap statsmodels prophet scipy jupyter

### How to Run
1. Place `tesla_deliveries_dataset_2015_2025.csv` in the same folder as the notebook
2. Open `week2_vikas_yadav.ipynb` in Jupyter
3. Run all cells top to bottom in order
4. Recommended: Python 3.9+

---

## 📋 Notebook Structure
| Section | Description |
|---|---|
| 1. Library Imports | All dependencies and configuration |
| 2. Data Loading & Inspection | Shape, dtypes, sample records, statistics |
| 3. Data Cleaning | Missing values, duplicates, outlier capping |
| 4. EDA | 9 visualisations — trends, distributions, correlations |
| 5. Feature Engineering | 14 new features including lags, rolling stats, cyclical encoding |
| 6. ML Pipeline | 7 regression models trained and compared |
| 7. Model Evaluation | R², MAE, RMSE, MAPE, residual analysis |
| 8. Hyperparameter Tuning | RandomizedSearchCV on Random Forest & XGBoost |
| 9. Time-Series Forecasting | SARIMA + Prophet with 12-month future forecast |
| 10. Model Explainability | SHAP feature importance analysis |
| 11. Business Insights | Key findings and recommendations |
| 12. Project Summary | Pipeline summary |

---

## 🏆 Results
| Model | R² | RMSE |
|---|---|---|
| XGBoost (Tuned) | ~0.993 | ~334 |
| Gradient Boosting | ~0.990 | ~387 |
| Random Forest | ~0.989 | ~408 |
| Linear / Ridge / Lasso | ~0.990 | ~380 |
| Decision Tree | ~0.984 | ~493 |

---

## ⚠️ Note on Dataset
This dataset is synthetically generated with uniform distribution across regions and years. Some charts appear flat as a result. The ML methodology is fully valid and would show stronger patterns on real-world data.

---

## 🛠️ Libraries Used
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost` · `shap` · `statsmodels` · `prophet` · `scipy`