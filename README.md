# 💵 VaultNet

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)

## 🎯 1. Project Overview & Business Objective
Customer churn is one of the largest financial leaks for modern banks. The objective of this project is to build a highly optimized, end-to-end machine learning pipeline capable of predicting which customers are at risk of leaving the bank. 

By accurately identifying these "flight risks," the bank can proactively allocate retention budgets (such as cash bonuses or targeted marketing) to save accounts, drastically reducing lost lifetime revenue while minimizing wasted spend on false alarms.

## 📥 2. The Dataset
This project uses the industry-standard **Bank Customer Churn Dataset**, which features demographic and financial information for 10,000 customers. 
* **Link:** [Bank Customer Churn Modeling (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
* **Challenge:** The dataset presents a severe **80:20 class imbalance** (80% of customers stay, 20% leave), requiring advanced threshold tuning and algorithmic penalization to accurately capture the minority class.

## 🛠️ 3. Pipeline & Engineering
To prepare the raw data for advanced tree-based modeling, a strict preprocessing pipeline was implemented:
* 🔀 **Categorical Encoding:** Applied Ordinal Encoding to hierarchical features (`Credit_Tier`, `Age_Group`) and One-Hot Encoding to nominal features (`Geography`), actively dropping the first column to avoid the Dummy Variable Trap.
* 📏 **Feature Scaling:** Standardized continuous financial metrics to ensure stability across distance-based evaluation metrics.
* ✂️ **Train/Test Split:** Implemented a stratified split to ensure the 80:20 imbalance was perfectly preserved across the training and holdout sets.

## 🧠 4. Modeling Strategy & Ensembling
Rather than relying on a single algorithm, this project evaluates multiple architectures and combines the top performers into a **Soft Voting Classifier**.

1. 🥇 **XGBoost (Extreme Gradient Boosting):** Tuned with `scale_pos_weight=2` and strict regularization (`reg_alpha=1`) to aggressively hunt churners while avoiding overfitting.
2. 🥈 **Random Forest (Bagging):** Configured with `class_weight={0: 1, 1: 2}` to act as a stable, high-precision anchor.
3. 🤝 **Voting Ensemble:** Averaged the probability distributions of both models, successfully smoothing out false positives while maintaining a high recall rate.

## 💰 5. Business Impact & Results
The final Soft Voting Ensemble achieved state-of-the-art results for this specific imbalanced dataset, breaking the 0.65 F1-Score barrier.

### 📑 Final Classification Report
* **Accuracy:** 86.77%
* **F1-Score:** 0.6541 🌟 *(Primary evaluation metric)*
* **Precision:** 70.03% *(7 out of 10 flagged customers are guaranteed flight risks)*
* **Recall:** 61.36% *(Successfully caught the majority of all actual churners)*


### 🧮 Confusion Matrix Insight
By tuning the probability threshold based on a custom **Precision-Recall Curve**, the model successfully prioritizes True Positives (saved revenue) while strictly clamping down on False Positives (wasted retention budget). 

## 💾 6. Repository Structure & Usage
The fully trained Voting Ensemble and Data Scaler have been serialized using `joblib` for instant deployment without retraining.

```python
import joblib

# Load the pipeline
model = joblib.load('bank_churn_model.pkl')
scaler = joblib.load('bank_churn_scaler.pkl')

# Predict on a new customer
new_customer_scaled = scaler.transform(new_customer_df)
churn_probability = model.predict_proba(new_customer_scaled)[:, 1]
