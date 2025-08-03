# Credit Default Prediction with Machine Learning
<p align="center">
  <img src="https://i1.wp.com/blog.bankbazaar.com/wp-content/uploads/2016/03/Surviving-a-Credit-Card-Default.png?resize=665%2C266&ssl=1" alt="Credit Default Risk" width="600">
</p>

## Problem Statement
Banks are pivotal in market economies, determining who qualifies for loans and under what terms. Credit scoring models predict the probability of default, enabling lenders to make informed decisions. This project aims to:
- Predict financial distress likelihood within the next two years.
- Empower borrowers with actionable insights to improve creditworthiness.
- Optimize bank decision-making by minimizing costly false negatives (missed defaults).

**Key Focus**: Model interpretability and high recall to prioritize risk reduction for lenders.

---

## Key Observations
- **Binary Classification**: Predict `default` (1) or `no default` (0).
- **Business Impact**: False negatives (missing a default) cost banks more than false positives.
- **Evaluation Metric**: **AUC-ROC** (Area Under the ROC Curve), chosen for its robustness in imbalanced classification.

---

## Evaluation Metric: AUC-ROC Explained
### What is AUC-ROC?
AUC (Area Under the Curve) measures the model's ability to distinguish between classes across all thresholds. Higher AUC = better performance.

#### ROC Curve Components:
| Metric | Formula | Interpretation |
|----------------------|----------------------------------|--------------------------------------------------------------------------------|
| **True Positive Rate (Recall)** | `TPR = TP / (TP + FN)` | "How many true defaults did we correctly predict?" |
| **False Positive Rate** | `FPR = FP / (FP + TN)` | "How many non-defaults were incorrectly flagged as risky?" |

- **AUC = 1**: Perfect classifier.
- **AUC = 0.5**: Random guessing.

---

## Project Workflow
1. **Data Preprocessing**: Handle missing values, outliers, and feature scaling.
2. **Exploratory Analysis (EDA)**: Visualize class imbalance, correlations, and feature distributions.
3. **Model Selection**: Train and compare Logistic Regression, Random Forest, and XGBoost.
4. **Interpretability**: Use SHAP/LIME to explain predictions to borrowers.
5. **Threshold Tuning**: Optimize for recall to reduce false negatives.

---

## Sample Results
**Best Model**: XGBoost (AUC = 0.89)

**Confusion Matrix** (Optimized for Recall):
<p align="center">
  <img src="https://i.imgur.com/nHHmhxt.png" alt="Confusion Matrix" width="400">
</p>

---

## How to Use This Repository
```bash
git clone https://github.com/SurajPaikekar/credit-default-prediction.git
cd credit-default-prediction
pip install -r requirements.txt # numpy, pandas, scikit-learn, xgboost, matplotlib
jupyter notebook # Run the analysis notebook


---

## References
- [Google ML Crash Course: ROC & AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Kaggle Competition: GiveMeSomeCredit](https://www.kaggle.com/competitions/GiveMeSomeCredit)

---

## Contributors
- **Suraj Paikekar** | [LinkedIn](https://www.linkedin.com/in/surajpaikekar/)
