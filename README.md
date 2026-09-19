# CodeAlpha ML Pipeline — California Housing Price Prediction

An end-to-end supervised machine learning pipeline: exploratory data analysis, statistical outlier detection, model training, and feature importance analysis, built on a real public dataset.

**Author:** Mawa Wazir
**Dataset:** [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) (1990 U.S. Census, Pace & Barry 1997) — 20,640 samples

---

## 📋 Project Overview

This repository documents a complete ML workflow for predicting median house values across California census block groups, built with the same structure and rigor as a production data science project:

- **Exploratory Data Analysis (EDA)** — distributions, missing values, correlations, and geographic patterns, all visualized
- **Statistical outlier detection** — IQR-based detection on skewed features, plus identification of a top-coding data artifact
- **Supervised learning** — Linear Regression and Random Forest Regressor, compared before/after outlier handling
- **Feature importance analysis** — both Gini (impurity-based) and permutation importance, cross-checked against each other
- **Honest before/after impact reporting** — outlier handling is shown to help some models more than others, with real numbers, not a single flattering headline stat

The full workflow lives in [`notebooks/eda_and_modeling.ipynb`](notebooks/eda_and_modeling.ipynb) — every cell has been executed end-to-end and produces the outputs shown in this README.

## 📊 Dataset

The **California Housing dataset** contains 1990 U.S. Census data at the census block group level: 20,640 rows, 8 numeric features (location, housing age, room/population counts, median income) and 1 categorical feature (`ocean_proximity`), with `median_house_value` as the regression target.

I selected this dataset because it's real (not synthetic), independently verifiable, well above a 10,000-sample threshold, and has genuine, well-documented data quality issues — a top-coded target variable, ~1% missing values in one column, and heavily right-skewed count features — which makes for an honest outlier-detection exercise rather than a manufactured one.

> **A note on methodology:** this project was built from scratch using a public dataset to demonstrate the same EDA → outlier handling → modeling → feature importance workflow described in the internship experience. All metrics below are computed directly from the executed notebook in this repo — nothing is asserted without a corresponding cell producing that number.

## 🔍 Key Findings

### Outlier detection
- **965 rows (4.68%)** had a `median_house_value` of exactly \$500,001 — a Census Bureau top-coding artifact (values were capped during collection), not a genuine outlier. These were removed before further outlier analysis.
- **IQR-based detection** on four skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`) flagged **1,762 rows (9.0%)** of the remaining data as statistical outliers.

### Model performance (Random Forest, final pipeline)
| Metric | Before outlier handling | After outlier handling |
|---|---|---|
| R² | 0.7891 | **0.7966** |
| MAE | \$30,624 | **\$30,048** |
| RMSE | \$45,855 | **\$44,357** |

**5-fold cross-validation** (shuffled, to avoid bias from the dataset's geographic row ordering) confirms this isn't a lucky split: mean R² = **0.7898 (± 0.0126)** across folds.

### The honest outlier-handling story
Outlier handling's impact turned out to be **model-dependent** — which is a more useful, and more truthful, finding than a single percentage:

| Model | R² improvement | MAE improvement |
|---|---|---|
| Linear Regression | **+4.54%** | **+3.16%** |
| Random Forest | +0.95% | +1.88% |

Linear Regression is meaningfully more sensitive to these outliers (its coefficients are pulled by extreme values in skewed features), while Random Forest is largely robust to them by construction, since tree splits depend on relative ordering rather than magnitude. The notebook walks through why, with both models trained and compared side-by-side.

*(If you're comparing this against a specific "8% accuracy improvement" figure: that exact number wasn't reproducible on this dataset/model combination during honest testing — the measured, real improvement for the outlier-sensitive model (Linear Regression) was 4.54% R² / 3.16% MAE. I'd rather report the true number than force-fit one that sounded better. See [Methodology Notes](#-methodology-notes) below.)*

### Feature importance
`median_income` is by far the strongest predictor (Gini importance ≈ 0.43, and the top feature by permutation importance too). Interestingly, `ocean_proximity_INLAND` ranks **second** by Gini importance — ahead of the raw `latitude`/`longitude` coordinates — since it's a single efficient binary split that separates cheaper inland housing early in the trees. The notebook cross-checks this with permutation importance and explains why the two methods rank things slightly differently.

## 📁 Repository Structure

```
codealpha-ml-pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── housing.csv                    # California Housing dataset (20,640 rows)
├── notebooks/
│   └── eda_and_modeling.ipynb         # Full, executed, end-to-end pipeline
└── images/                            # Exported plots (also embedded in the notebook)
    ├── target_distribution.png
    ├── feature_distributions.png
    ├── geographic_distribution.png
    ├── ocean_proximity.png
    ├── correlation_heatmap.png
    ├── outliers_before_after.png
    ├── feature_importance.png
    └── before_after_comparison.png
```

## 🛠️ Tech Stack

- **Python 3.12**
- **pandas** / **numpy** — data manipulation
- **matplotlib** / **seaborn** — visualization
- **scikit-learn** — preprocessing pipelines, `RandomForestRegressor`, `LinearRegression`, cross-validation, permutation importance
- **scipy** — statistical methods

## 🚀 Running This Project

```bash
git clone https://github.com/<your-username>/codealpha-ml-pipeline.git
cd codealpha-ml-pipeline
pip install -r requirements.txt
jupyter notebook notebooks/eda_and_modeling.ipynb
```

The notebook runs top-to-bottom with no external downloads required — the dataset is included in `data/housing.csv`.

## 🧪 Methodology Notes

A few decisions worth being transparent about, for anyone reviewing this as a portfolio piece:

- **IQR over Z-score** for outlier detection: the skewed features here are heavily right-tailed (confirmed visually in the notebook), and Z-score assumes approximate normality — IQR, being quantile-based, is more appropriate for this data shape.
- **Removal over winsorizing** for the final reported pipeline: both were tested during development; winsorizing gave a negligible difference for the Random Forest model specifically, while removal gives a cleaner before/after comparison for this report. A production system would likely default to winsorizing to avoid discarding ~9% of rows.
- **Shuffled K-Fold**: the raw dataset's rows are ordered geographically (confirmed in the EDA section). An unshuffled `KFold` would create geographically homogeneous folds and understate/skew cross-validation performance — this is called out explicitly in the notebook, along with the fix (`shuffle=True`).
- **The two feature-importance methods disagree slightly** on where `ocean_proximity` ranks relative to raw coordinates — the notebook explains why, rather than only reporting the more flattering of the two.

## 📄 License

This project uses the California Housing dataset (Pace, R. Kelley and Ronald Barry, 1997), which is in the public domain / freely redistributable for research and educational use.
