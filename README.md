# Credit Default Prediction with Machine Learning

<p align="center">
  <img src="https://i1.wp.com/blog.bankbazaar.com/wp-content/uploads/2016/03/Surviving-a-Credit-Card-Default.png?resize=665%2C266&ssl=1" alt="Credit Default Risk" width="600">
</p>

## 📌 Problem Statement
Banks are pivotal in market economies, determining who qualifies for loans and under what terms. Credit scoring models predict the probability of default, enabling lenders to make informed decisions. This project aims to:
- Predict financial distress likelihood within the next two years
- Empower borrowers with actionable insights to improve creditworthiness
- Optimize bank decision-making by minimizing costly false negatives (missed defaults)

**Key Focus**: Model interpretability and high recall to prioritize risk reduction for lenders.

---

## 🔍 Key Observations
- **Binary Classification**: Predict `default` (1) or `no default` (0)
- **Business Impact**: False negatives (missing a default) cost banks more than false positives
- **Evaluation Metric**: **AUC-ROC** (Area Under the ROC Curve), chosen for its robustness in imbalanced classification

---

## 📊 Evaluation Metric: AUC-ROC Explained
### What is AUC-ROC?
AUC (Area Under the Curve) measures the model's ability to distinguish between classes across all thresholds. Higher AUC = better performance.

#### ROC Curve Components:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **True Positive Rate (Recall)** | `TPR = TP / (TP + FN)` | "How many true defaults did we correctly predict?" |
| **False Positive Rate** | `FPR = FP / (FP + TN)` | "How many non-defaults were incorrectly flagged as risky?" |

- **AUC = 1**: Perfect classifier
- **AUC = 0.5**: Random guessing

---

## 🛠️ Project Workflow
1. **Data Preprocessing**: Handle missing values, outliers, and feature scaling
2. **Exploratory Analysis (EDA)**: Visualize class imbalance, correlations, and feature distributions
3. **Model Selection**: Train and compare Logistic Regression, Random Forest, and XGBoost
4. **Interpretability**: Use SHAP/LIME to explain predictions to borrowers
5. **Threshold Tuning**: Optimize for recall to reduce false negatives

---

## 📈 Model Performance Results

### Comparative Analysis of Models
| Model | ROCAUC | Accuracy | Precision | Recall | F1_score | Private Score | Public Score |
|-------|--------|----------|-----------|--------|----------|---------------|--------------|
| **LGBM Final Model** | **86.3247** | 79.7610 | 21.4420 | **76.3010** | 33.4760 | **0.8667** | 0.8605 |
| XGBoost Final Model | 86.4370 | 80.3600 | 21.9460 | 75.9920 | 34.0570 | 0.8666 | 0.8596 |
| LogisticRegression | 85.0554 | 93.5700 | 55.7540 | 17.7230 | 26.8960 | 0.8523 | 0.8460 |
| GaussianNB | 84.7506 | 90.3590 | 33.8930 | 46.7800 | 39.3070 | 0.8491 | 0.8491 |
| KNeighborsClassifier | 82.7503 | 93.5390 | 56.2250 | 14.4260 | 22.9600 | 0.8369 | 0.8254 |
| SGDClassifier | 84.2914 | 93.4190 | 51.7290 | 20.8140 | 29.6840 | 0.8523 | 0.8460 |
| RandomForestClassifier | 83.8682 | 93.5120 | 54.0790 | 18.4440 | 27.5070 | 0.8431 | 0.8354 |


**Key Findings**:
- The hyperparameter-tuned LGBM and XGBoost models achieved the best AUC-ROC scores
- Both models show significantly higher recall (76.30% and 75.99% respectively) compared to other models
- LGBM model achieved a competition private score of 0.8667 (top 14% in Kaggle competition)

**Best Model**: LightGBM (LGBM) with:
- Highest competition private score (**0.8667**)
- Top 14% ranking in Kaggle competition (132/924)
- Best balance of recall (**76.30%**) and AUC-ROC (**86.32%**)
- Most suitable for business needs (minimizing false negatives)


## 📈 Sample Results
**Best Model**: XGBoost (AUC = 0.89)

**Confusion Matrix** (Optimized for Recall):
<p align="center">
  <img src="https://i.imgur.com/nHHmhxt.png" alt="Confusion Matrix" width="700">
</p>

---

## 🎯 Conclusion
The project successfully developed a credit default prediction model through:
1. Comprehensive data analysis and preprocessing
2. Extensive evaluation of multiple machine learning models
3. Rigorous hyperparameter tuning of XGBoost and LightGBM models

The final LGBM model with optimized hyperparameters achieved:
- **AUC-ROC of 0.8667**
- **Top 14% performance** in Kaggle competition (132/924 ranking)
- **High recall of 76.30%**, crucial for minimizing false negatives

**Limitations and Future Work**:
- Model performance depends on available dataset features
- Real-world credit factors may change more dynamically than captured in static data
- Potential for improvement with additional research and computational resources
- Exploration of ensemble methods and neural networks could yield further improvements

📚 References
- [Google ML Crash Course: ROC & AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Kaggle Competition: GiveMeSomeCredit](https://www.kaggle.com/competitions/GiveMeSomeCredit)

---

👥 Contributors
- **Suraj Paikekar** | [LinkedIn](https://www.linkedin.com/in/surajpaikekar/)

---

## 💻 How to Use This Repository
```bash
git clone https://github.com/SurajPaikekar/credit-default-prediction.git
cd credit-default-prediction
pip install -r requirements.txt  # numpy, pandas, scikit-learn, xgboost, matplotlib
jupyter notebook  # Run the analysis notebook



