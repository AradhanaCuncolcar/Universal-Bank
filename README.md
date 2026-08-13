# Universal Bank - AI Loan Prediction Dashboard & Classification Pipeline

An institutional-grade, end-to-end Machine Learning project and interactive web application designed to predict whether bank customers will accept a personal loan offer. This project encompasses comprehensive exploratory data analysis, data cleaning, feature engineering, classification modeling across four key algorithms, hyperparameter tuning via Stratified K-Fold Cross-Validation, and a fully interactive Streamlit dashboard.

---

## 🚀 Project Architecture & Key Features

1. **Descriptive Analytics & EDA**: In-depth analysis of customer demographics, financial behavior (income, CCAvg, mortgage), and correlation matrices.
2. **Robust Preprocessing & Stratification**: 
   - Removal of unique identifiers (`ID`) and high-cardinality features (`ZIP Code`).
   - 80:20 train-test split with strict stratification to preserve the minority class distribution (`Personal Loan = 1`).
   - Standard scaling specifically tailored for distance-based algorithms.
3. **Four Core Classification Algorithms**:
   - K-Nearest Neighbors (KNN)
   - Decision Tree (DT)
   - Random Forest (RF)
   - Gradient Boosting Regressor Tree (GBRT)
4. **Comprehensive Evaluation Metrics**:
   - Accuracy (Train & Test), Precision, Recall, F1 Score, and ROC-AUC.
   - Combined multi-algorithm ROC curve chart.
   - Fully labeled confusion matrices displaying **both counts and percentages** for both training and testing sets.
5. **Feature Importance Analysis**: Visual inspection of feature drivers across the Decision Tree family models.
6. **Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation using `GridSearchCV` optimized for the F1-score.
7. **Interactive Streamlit Dashboard (`app.py`)**: A professional web application featuring sidebar navigation, live model metrics, visual charts, and a side-by-side comparison matrix of pre- vs. post-tuning performance.

---

## 📂 Repository Structure

```text
├── UniversalBank with description.xls   # Raw dataset file
├── app.py                               # Streamlit interactive dashboard
├── requirements.txt                     # Required Python dependencies
└── README.md                            # Project documentation
