# Cardiovascular Disease Prediction — ML - V2

Machine learning pipeline to detect cardiovascular disease risk using the Cleveland Heart Disease dataset. The project is built around a clinical priority: **minimizing false negatives**, since missing a real case carries far greater consequences than a false alarm.

---

## Results

All models evaluated at threshold 0.3, optimized for recall via GridSearchCV.

| Model | Accuracy | Recall | Precision | F1 | False Negatives | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.82 | 0.96 | 0.73 | 0.83 | 1 | 0.951 |
| SVM | 0.84 | 0.96 | 0.75 | 0.84 | 1 | 0.951 |
| **Random Forest** | **0.80** | **1.00** | **0.70** | **0.82** | **0** | **0.963** |
| Decision Tree | 0.80 | 0.82 | 0.77 | 0.79 | 5 | 0.860 |

**Random Forest is the best clinical model** — the only one to achieve zero false negatives and the highest ROC-AUC across all four models.

---

## Why Threshold Adjustment Matters

Standard ML pipelines optimize for accuracy or F1. In a clinical screening context, that's the wrong objective. A false negative — a sick patient classified as healthy — is not equivalent to a false positive.

Lowering the classification threshold from 0.5 to 0.3 aligns the model's behavior with clinical reasoning. The tradeoff across all models is more false positives (patients flagged unnecessarily) in exchange for fewer missed cases. In a screening context, that is the correct tradeoff.

---

## Key Finding — Feature Importances

The Random Forest feature importance ranking reveals which variables actually drive predictions:

| Rank | Feature | Clinical meaning |
|---|---|---|
| 1 | `thal_7` | Reversible thalassemia defect |
| 2 | `chest_pain_4` | Asymptomatic chest pain |
| 3 | `ca` | Number of major vessels colored by fluoroscopy |
| 4 | `max_heart_rate` | Maximum heart rate achieved |
| 5 | `oldpeak` | ST depression induced by exercise |

`thal_7`, `chest_pain_4`, and `ca` as dominant predictors aligns with established cardiology literature, which adds credibility to the model's behavior. Variables such as `fasting_glucose`, `thal_6`, `slope_3`, and `restecg_1` contributed near zero importance.

---

## Dataset

- **Source:** Cleveland Heart Disease Dataset (UCI ML Repository)
- **Size:** 303 patients, 14 columns (13 features + target)
- **Target:** Binary — presence or absence of heart disease (`diagnosis > 0`)
- **Split:** 80/20 stratified by target class — both sets maintain 54/46 class balance

---

## Pipeline

### 1. Exploratory Data Analysis
Class balance check, feature distributions, boxplots for outlier detection, correlation heatmap including target, unique value check for categorical columns.

### 2. One Hot Encoding
`chest_pain`, `restecg`, `slope`, and `thal` are categorical variables encoded as integers in the raw dataset. Feeding them as continuous values implies an ordinal relationship that does not exist. Applied `pd.get_dummies` with `drop_first=True`, expanding from 13 to 18 features.

### 3. Outlier Capping
`blood_pressure`, `max_heart_rate`, and `oldpeak` clipped at the 1st/99th percentile. Used 1st/99th rather than 5th/95th because with 303 rows, 5% means clipping ~15 patients per tail — too aggressive for a clinical dataset where extreme values may be real.

| Feature | Clipped range |
|---|---|
| `blood_pressure` | [100.0, 180.0] |
| `max_heart_rate` | [95.0, 192.0] |
| `oldpeak` | [0.0, 4.2] |

### 4. Log Transform
`oldpeak` is right-skewed with values bunched near zero and a long tail. Applied `np.log1p` which handles zeros cleanly (`log1p(0) = 0`).

### 5. Train / Test Split
80/20 split with `stratify=y` and `random_state=42`. Stratification confirmed — both sets show identical 54/46 class balance.

### 6. Feature Scaling
`RobustScaler` fitted on `X_train` only, then applied to `X_test`. RobustScaler uses median and IQR instead of mean and standard deviation, making it resistant to the remaining outliers. Fitting only on the training set prevents data leakage.

### 7. Model Training
All models trained with `class_weight='balanced'` and `scoring='recall'` in GridSearchCV (5-fold CV) to maintain clinical priority throughout hyperparameter optimization.

**Best hyperparameters found:**

| Model | Best params |
|---|---|
| SVM | `C=0.1, gamma='scale', kernel='rbf'` |
| Random Forest | `max_depth=5, max_features='sqrt', min_samples_split=5, n_estimators=200` |
| Decision Tree | `criterion='gini', max_depth=5, min_samples_split=10` |

---

## Limitations

These should be understood before drawing any clinical conclusions:

- **Small sample.** 303 patients is not enough for generalization. Results should be validated on a larger, more recent dataset before any clinical application.
- **Fragile recall numbers.** With ~28 positive cases in the test set, zero false negatives represents a very small absolute count. A different random split could produce different results. The directional finding is valid; the exact numbers are not.
- **Historical dataset.** The Cleveland dataset is from the 1980s. Clinical feature distributions may not reflect current patient populations.
- **No external validation.** All evaluation is on a held-out split of the same dataset. Performance on an independent cohort is unknown.

---

## Project Structure

```
cardiovascular-disease-ml/
├── cardiovascular_risk_ml.ipynb   # Full pipeline and modeling notebook
├── heart.csv                      # Cleveland dataset
├── requirements.txt               # Dependencies
└── README.md
```

---

## Setup

```bash
git clone https://github.com/NaujQD/cardiovascular-disease-ml.git
cd cardiovascular-disease-ml
pip install -r requirements.txt
```

Then open `cardiovascular_risk_ml.ipynb` in Jupyter or Google Colab.

---

## Stack

Python · pandas · numpy · scikit-learn · seaborn · matplotlib

---

## Author

Built by [@NaujQD](https://github.com/NaujQD)
