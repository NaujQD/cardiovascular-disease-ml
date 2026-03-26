# Cardiovascular Disease Prediction (Machine Learning)

Machine learning model to detect cardiovascular disease using clinical variables from the Cleveland dataset.

**Key results**
- Accuracy: ~89%
- Recall for disease detection: **0.97**
- False negatives reduced from **3 to 1** by adjusting the decision threshold
- Cross-validation accuracy: ~84%

---

## Why This Matters

In clinical practice, missing a patient with cardiovascular disease (false negative) can have severe consequences.  
This project optimizes the model using clinical reasoning, prioritizing recall over raw accuracy.

---

## Dataset

Cleveland Heart Disease Dataset (303 patients, 13 clinical variables).

---

## Methodology

1. Data cleaning and exploration
2. Feature correlation analysis
3. Logistic Regression model with balanced class weights
4. **Decision threshold adjustment (0.3 instead of 0.5)** to reduce false negatives
5. Evaluation using confusion matrix, recall, precision and ROC analysis

---

## Clinical Threshold Adjustment

Lowering the classification threshold from 0.5 to 0.3:

| Metric | Threshold 0.5 | Threshold 0.3 |
|--------|----------------|----------------|
| False Negatives | 3 | **1** |
| Recall (Disease) | 0.91 | **0.97** |
| Accuracy | 0.89 | 0.89 |

This demonstrates how ML models should be adapted to healthcare priorities rather than generic metrics.

---

## Technologies Used

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- Google Colab

---

## Project Structure
cardiovascular-disease-ml/
│
├── cardiovascular_risk_ml.ipynb
├── heart.csv
├── README.md
├── requirements.txt

---

## Conclusions

This project shows how medical knowledge can guide machine learning optimization to create clinically meaningful models. Using the logic behind algorythms and the clincial experience.
