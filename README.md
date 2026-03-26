# Cardiovascular Disease Prediction (ML)

Machine learning model to detect cardiovascular disease using clinical data.

- Accuracy: ~88%
- Recall (disease): ~91%
- Cross-validation: ~84%

The goal is to evaluate how well these models perform and assess their potential applicability in real clinical environments.

Dataset
Public dataset: Cleveland Heart Disease Dataset

~300 patients

14 clinical variables (age, cholesterol, blood pressure, etc.)

Tools & Technologies
Python

pandas

scikit-learn

matplotlib

Methodology

Data cleaning and preprocessing

Feature scaling (StandardScaler)

Train-test split (80/20)

Logistic Regression model

Model evaluation using:
-Accuracy

-Precision / Recall

-F1-score

-Confusion Matrix

### Clinical Threshold Adjustment

By lowering the classification threshold to 0.3, the model reduced false negatives from 3 to 1, increasing recall for disease detection from 0.91 to 0.97, without reducing overall accuracy.

This reflects a clinically-oriented optimization prioritizing patient safety over pure statistical performance.
