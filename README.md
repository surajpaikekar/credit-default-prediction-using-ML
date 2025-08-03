# Credit Default Prediction With Machine Learning

<p align="center">
  <img src="https://i1.wp.com/blog.bankbazaar.com/wp-content/uploads/2016/03/Surviving-a-Credit-Card-Default.png?resize=665%2C266&ssl=1" alt="Surviving a credit default" width="600" height="400">
</p>

## Project Overview

Banks play a pivotal role in market economies by determining who receives financing and under what conditions. Credit scoring algorithms estimate the likelihood of default, guiding banks' lending decisions. This project aims to enhance credit scoring by developing a machine learning model to predict the probability of a borrower experiencing financial distress within the next two years. The model is designed to empower borrowers with actionable insights to improve their creditworthiness.

## Problem Statement

The objective is to build an interpretable machine learning model that predicts whether a borrower will default on a loan within two years. This is a **binary classification problem** where the model prioritizes **recall** over precision to minimize false negatives, as these are more costly for banks. The model should provide clear, actionable insights for borrowers to enhance their financial decisions.

### Key Goals
- Develop a model with high **explainability** to provide actionable insights.
- Optimize for **recall** to reduce false negatives, minimizing financial risk for banks.
- Achieve a high **Area Under the ROC Curve (AUC)**, the evaluation metric for this Kaggle competition.

## Evaluation Metric: AUC

Submissions are evaluated using the **Area Under the ROC Curve (AUC)**, which measures the model's performance across all classification thresholds.

### What is AUC?
AUC represents the entire two-dimensional area under the **Receiver Operating Characteristic (ROC) curve**. It quantifies the probability that the model ranks a random positive example (default) higher than a random negative example (non-default).

### What is an ROC Curve?
An ROC curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various classification thresholds.

- **True Positive Rate (TPR)** (also called **recall** or **sensitivity**):
  \[
  \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]
  - **TP**: Number of True Positives (correctly predicted defaults)
  - **FN**: Number of False Negatives (missed defaults)

- **False Positive Rate (FPR)**:
  \[
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  \]
  - **FP**: Number of False Positives (incorrectly predicted defaults)
  - **TN**: Number of True Negatives (correctly predicted non-defaults)

### Visualizing the ROC Curve and AUC
The ROC curve illustrates the trade-off between TPR and FPR as the classification threshold varies. A higher AUC indicates better model performance.

<p align="center">
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg" alt="ROC Curve" width="500" height="400">
</p>
<p align="center">
  <em>ROC Curve: TPR vs. FPR at different thresholds</em>
</p>

<p align="center">
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg" alt="AUC" width="500" height="400">
</p>
<p align="center">
  <em>AUC: Area under the ROC curve, measuring overall model performance</em>
</p>

### Confusion Matrix
The following confusion matrix clarifies the concepts of TP, TN, FP, and FN:

<p align="center">
  <img src="https://i.imgur.com/nHHmhxt.png" alt="Confusion Matrix">
</p>

## Project Features
- **Model Explainability**: Provides interpretable outputs to guide borrowers on improving creditworthiness.
- **High Recall Focus**: Prioritizes minimizing false negatives to reduce financial risk for banks.
- **Robust Evaluation**: Optimized for AUC, ensuring strong performance across classification thresholds.
- **Real-World Impact**: Empowers borrowers with insights for better financial decisions and supports banks in making informed lending choices.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, etc.
- Kaggle account to access the competition dataset

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-default-prediction.git
