# Cardiovascular Disease Prediction — ML

Logistic Regression model trained on the Cleveland Heart Disease dataset to detect cardiovascular disease risk. The project is built around a clinical priority: **minimizing false negatives**, since missing a real case carries far greater consequences than a false alarm.

---

## Results

| Metric | Threshold 0.5 | Threshold 0.3 |
|---|---|---|
| Accuracy | 0.89 | 0.89 |
| Recall (disease) | 0.91 | **0.97** |
| False Negatives | 3 | **1** |

Lowering the classification threshold from 0.5 to 0.3 reduced false negatives by 67% with no loss in overall accuracy. This is the central result of the project.

---

## Why Threshold Adjustment Matters

Standard ML pipelines optimize for accuracy or F1. In a clinical screening context, that's the wrong objective. A false negative — a sick patient classified as healthy — is not equivalent to a false positive. Adjusting the decision threshold is a low-cost, interpretable intervention that directly aligns the model's behavior with clinical reasoning.

---

## Dataset

- **Source:** Cleveland Heart Disease Dataset (UCI ML Repository)
- **Size:** 303 patients, 13 clinical variables
- **Target:** Binary — presence or absence of heart disease
- **Split:** 80% train / 20% test, stratified by target class

Variables include age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, and thalassemia type.

---

## Methodology

1. **Exploratory analysis** — distribution checks, class balance, feature correlations
2. **Preprocessing** — no missing values in this dataset; features left on original scale for logistic regression interpretability
3. **Model** — Logistic Regression with `class_weight='balanced'` to handle class imbalance
4. **Threshold tuning** — decision boundary moved from 0.5 → 0.3 based on clinical priority
5. **Evaluation** — confusion matrix, precision/recall, ROC-AUC, 5-fold cross-validation

---

## Limitations

These should be understood before drawing any clinical conclusions:

- **Small sample.** 303 patients is not enough for generalization. The gap between test accuracy (~89%) and cross-validation accuracy (~84%) suggests some overfitting.
- **Fragile recall numbers.** Going from 3 to 1 false negative represents 2 patients in an ~60-patient test set. A different random split could produce different results. The directional finding is valid; the exact numbers are not.
- **Single model.** No comparison against tree-based or ensemble methods. Logistic regression was chosen for interpretability, not because it was benchmarked as the best option.
- **Historical dataset.** The Cleveland dataset is from the 1980s. Clinical feature distributions may not reflect current patient populations.

---

## Project Structure

```
cardiovascular-disease-ml/
├── cardiovascular_risk_ml.ipynb   # Full analysis and modeling notebook
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
