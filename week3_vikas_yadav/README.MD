# Country Development Clustering

A data-driven unsupervised machine learning project that segments 167 countries into meaningful development tiers using socio-economic and health indicators. Cluster labels are then validated using supervised classification models.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

Vast disparities exist across the world in healthcare, income, and quality of life. This project uses **K-Means clustering** to automatically group countries by their development level — without any predefined labels — and then trains **Random Forest** and **XGBoost** classifiers to validate the cluster structure. The analysis provides actionable insights for international organizations such as UNICEF, WHO, and the World Bank to identify countries in need of urgent aid.

---

## Dataset

**File:** `Country-data.csv`  
**Source:** HELP International (Humanitarian Aid Dataset)  
**Shape:** 167 rows × 10 columns  
**Missing values:** None

| Column | Description |
| --- | --- |
| `country` | Name of the country |
| `child_mort` | Deaths of children under 5 per 1,000 live births |
| `exports` | Exports of goods and services (% of GDP per capita) |
| `health` | Total health spending (% of GDP per capita) |
| `imports` | Imports of goods and services (% of GDP per capita) |
| `income` | Net income per person (USD) |
| `inflation` | Annual GDP growth rate (%) |
| `life_expec` | Average life expectancy at birth (years) |
| `total_fer` | Average number of children per woman |
| `gdpp` | GDP per capita (USD) |

---

## Project Structure

```
├── Country-data.csv          # Raw dataset
├── data-dictionary.csv       # Feature descriptions
├── week3_Vikas_Yadav.ipynb   # Main analysis notebook
└── README.md                 # Project documentation
```

---

## Methodology

### 1. Data Cleaning
- Lowercased and stripped column names
- Removed duplicate rows
- Coerced all numeric columns; filled any missing values with column medians

### 2. Exploratory Data Analysis
- Correlation heatmap across all 9 numeric features
- Boxplots per feature to identify skewness and outliers

### 3. Feature Scaling
- Applied `StandardScaler` to all 9 numeric features
- Required for distance-based algorithms (K-Means, DBSCAN)

### 4. Clustering
- **Elbow Method** used to determine optimal k — elbow at **k = 3**
- **K-Means (k=3)** trained; silhouette score ≈ 0.28
- **DBSCAN (eps=1.5, min_samples=5)** used for outlier detection

### 5. Classification (Cluster Validation)
- K-Means labels used as target classes
- 80/20 stratified train-test split
- **Random Forest** and **XGBoost** trained and evaluated

### 6. Interpretability
- PCA 2D visualization of cluster separation
- Random Forest feature importance ranking
- Cluster profiling (mean of all features per cluster)

---

## Results

### Cluster Summary

| Cluster | Label | Size | Defining Traits |
| --- | --- | --- | --- |
| 0 | Developed | 36 countries | GDP ~$42K, child mortality ~5, life expectancy ~80 yrs |
| 1 | Underdeveloped | 47 countries | GDP ~$1.9K, child mortality ~93, life expectancy ~59 yrs |
| 2 | Developing | 84 countries | GDP ~$6.5K, child mortality ~22, life expectancy ~73 yrs |

### Model Performance

| Model | Accuracy |
| --- | --- |
| Random Forest | 100.00% |
| XGBoost | 97.06% |

### Top Features Driving Segmentation

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | child_mort | 26.2% |
| 2 | gdpp | 20.2% |
| 3 | total_fer | 16.6% |
| 4 | life_expec | 15.6% |
| 5 | income | 14.6% |

---

## Installation

```bash
git clone https://github.com/xishhh/country-development-clustering.git
cd country-development-clustering
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## Usage

```bash
jupyter notebook week3_Vikas_Yadav.ipynb
```

Run all cells top to bottom. The notebook is self-contained — all steps from data loading to cluster profiling execute sequentially with no external dependencies beyond the CSV file.

---

## Dependencies

| Library | Purpose |
| --- | --- |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Scaling, clustering, PCA, Random Forest |
| `xgboost` | Gradient boosted classifier |

---
