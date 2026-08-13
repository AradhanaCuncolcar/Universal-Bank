import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Universal Bank - AI Loan Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: 700;}
    .sub-header {font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;}
    .card {padding: 1.5rem; border-radius: 0.5rem; background-color: #F3F4F6; margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & MODEL CACHING
# ==========================================
@st.cache_data
def load_data():
    file_path = 'UniversalBank with description.xls'
    df = pd.read_excel(file_path, sheet_name='Data')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'UniversalBank with description.xls' is in the same directory.")
    st.stop()

# Preprocessing & Split
X = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
y = df['Personal Loan']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# SIDEBAR CONTROLS & NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=80)
st.sidebar.title("Navigation & Filters")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "🏠 Overview & Descriptive Analytics",
        "⚙️ Initial Model Training & ROC",
        "📊 Confusion Matrices (Train & Test)",
        "🌲 Feature Importance (DT Family)",
        "🔍 Hyperparameter Tuning (GridSearch)",
        "⚖️ Side-by-Side Performance Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("📌 **Target Variable:** Personal Loan (1 = Yes, 0 = No)\n\n📌 **Stratification:** 80:20 Train-Test Split")

# ==========================================
# MODEL TRAINING PIPELINE (Cached/Executed)
# ==========================================
@st.cache_resource
def run_initial_models():
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    trained_models = {}
    roc_data = {}
    
    for name, model in models.items():
        if name == 'KNN':
            model.fit(X_train_scaled, y_train)
            tr_pred = model.predict(X_train_scaled)
            te_pred = model.predict(X_test_scaled)
            tr_proba = model.predict_proba(X_train_scaled)[:, 1]
            te_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            tr_pred = model.predict(X_train)
            te_pred = model.predict(X_test)
            tr_proba = model.predict_proba(X_train)[:, 1]
            te_proba = model.predict_proba(X_test)[:, 1]
            
        trained_models[name] = model
        
        results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, tr_pred),
            'Test Accuracy': accuracy_score(y_test, te_pred),
            'Precision': precision_score(y_test, te_pred),
            'Recall': recall_score(y_test, te_pred),
            'F1 Score': f1_score(y_test, te_pred),
            'AUC': roc_auc_score(y_test, te_proba)
        })
        fpr, tpr, _ = roc_curve(y_test, te_proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, te_proba))
        
    return pd.DataFrame(results), trained_models, roc_data

init_df, trained_models, roc_data = run_initial_models()


@st.cache_resource
def run_tuned_models():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grids = {
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'Decision Tree': {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
        'GBRT': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 0.2]}
    }
    
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    
    tuned_results = []
    tuned_models = {}
    
    for name, model in models.items():
        grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
        if name == 'KNN':
            grid.fit(X_train_scaled, y_train)
            best_m = grid.best_estimator_
            tuned_models[name] = best_m
            tr_pred = best_m.predict(X_train_scaled)
            te_pred = best_m.predict(X_test_scaled)
            te_proba = best_m.predict_proba(X_test_scaled)[:, 1]
        else:
            grid.fit(X_train, y_train)
            best_m = grid.best_estimator_
            tuned_models[name] = best_m
            tr_pred = best_m.predict(X_train)
            te_pred = best_m.predict(X_test)
            te_proba = best_m.predict_proba(X_test)[:, 1]
            
        tuned_results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, tr_pred),
            'Test Accuracy': accuracy_score(y_test, te_pred),
            'Precision': precision_score(y_test, te_pred),
            'Recall': recall_score(y_test, te_pred),
            'F1 Score': f1_score(y_test, te_pred),
            'AUC': roc_auc_score(y_test, te_proba)
        })
        
    return pd.DataFrame(tuned_results), tuned_models

tuned_df, tuned_models = run_tuned_models()


# ==========================================
# HELPER: CONFUSION MATRIX PLOTTING
# ==========================================
def render_confusion_matrices(models_dict, title_prefix):
    fig, axes = plt.subplots(len(models_dict), 2, figsize=(10, 4 * len(models_dict)))
    axes = axes.flatten()
    idx = 0
    
    for name, model in models_dict.items():
        tr_input = X_train_scaled if name == 'KNN' else X_train
        te_input = X_test_scaled if name == 'KNN' else X_test
        
        tr_pred = model.predict(tr_input)
        te_pred = model.predict(te_input)
        
        cm_train = confusion_matrix(y_train, tr_pred)
        cm_train_pct = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] * 100
        
        cm_test = confusion_matrix(y_test, te_pred)
        cm_test_pct = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100
        
        annot_train = np.array([f"{v}\n({p:.1f}%)" for v, p in zip(cm_train.flatten(), cm_train_pct.flatten())]).reshape(2, 2)
        annot_test = np.array([f"{v}\n({p:.1f}%)" for v, p in zip(cm_test.flatten(), cm_test_pct.flatten())]).reshape(2, 2)
        
        # Train CM
        sns.heatmap(cm_train, annot=annot_train, fmt='', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
        axes[idx].set_title(f"{name} - Train ({title_prefix})", fontsize=11, fontweight='bold', color='#1E3A8A')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        idx += 1
        
        # Test CM
        sns.heatmap(cm_test, annot=annot_test, fmt='', cmap='Greens', ax=axes[idx], cbar=False,
                    xticklabels=['No (0)', 'Yes (1)'], yticklabels=['No (0)', 'Yes (1)'])
        axes[idx].set_title(f"{name} - Test ({title_prefix})", fontsize=11, fontweight='bold', color='#065F46')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        idx += 1
        
    plt.tight_layout()
    st.pyplot(fig)


# ==========================================
# DASHBOARD VIEWS (SECTIONS)
# ==========================================

if app_mode == "🏠 Overview & Descriptive Analytics":
    st.markdown('<p class="main-header">Universal Bank Loan Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Executive exploratory data analysis and customer targeting intelligence.</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{df.shape[1] - 1}")
    col3.metric("Accepted Loan (Yes)", f"{df['Personal Loan'].sum():,}")
    col4.metric("Acceptance Rate", f"{(df['Personal Loan'].mean() * 100):.2f}%")
    
    st.markdown("---")
    st.subheader("📋 Raw Data Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Target Distribution & Correlation Insight")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Personal Loan', data=df, palette=['#94A3B8', '#3B82F6'], ax=ax)
        ax.set_title("Personal Loan Acceptance Distribution", fontweight='bold')
        ax.set_xticklabels(['No (0)', 'Yes (1)'])
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_df = df.drop(columns=['ID', 'ZIP Code'])
        sns.heatmap(numeric_df.corr()[['Personal Loan']].sort_values(by='Personal Loan', ascending=False), 
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar=True)
        ax.set_title("Feature Correlation with Personal Loan", fontweight='bold')
        st.pyplot(fig)

elif app_mode == "⚙️ Initial Model Training & ROC":
    st.markdown('<p class="main-header">Initial Model Performance & ROC Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluating KNN, Decision Tree, Random Forest, and GBRT prior to hyperparameter tuning.</p>', unsafe_allow_html=True)
    
    st.subheader("🏆 Initial Performance Metrics Table")
    st.dataframe(init_df.style.format({
        'Train Accuracy': '{:.4f}', 'Test Accuracy': '{:.4f}', 
        'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1 Score': '{:.4f}', 'AUC': '{:.4f}'
    }), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📉 Combined ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B']
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})', color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.6)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves - Initial Models', fontsize=14, fontweight='bold', color='#1E3A8A')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

elif app_mode == "📊 Confusion Matrices (Train & Test)":
    st.markdown('<p class="main-header">Confusion Matrix Analysis (Counts & Percentages)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed breakdown of correct and incorrect classifications across training and testing splits.</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 Initial Models Confusion Matrices", "⚙️ Tuned Models Confusion Matrices"])
    
    with tab1:
        st.subheader("Initial Models (Train vs. Test)")
        render_confusion_matrices(trained_models, title_prefix="Initial")
        
    with tab2:
        st.subheader("Tuned Models with GridSearchCV (Train vs. Test)")
        render_confusion_matrices(tuned_models, title_prefix="Tuned")

elif app_mode == "🌲 Feature Importance (DT Family)":
    st.markdown('<p class="main-header">Feature Importance Chart (Decision Tree Family)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualizing what drives customer loan decisions across Decision Trees, Random Forest, and GBRT.</p>', unsafe_allow_html=True)
    
    dt_family = {
        'Decision Tree': trained_models['Decision Tree'],
        'Random Forest': trained_models['Random Forest'],
        'GBRT': trained_models['GBRT']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#0EA5E9', '#10B981', '#6366F1']
    for i, (name, model) in enumerate(dt_family.items()):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
        feat_imp.plot(kind='barh', ax=axes[i], color=colors[i])
        axes[i].set_title(f"{name}", fontsize=12, fontweight='bold', color='#1E3A8A')
        axes[i].set_xlabel("Importance Score", fontweight='bold')
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    st.pyplot(fig)

elif app_mode == "🔍 Hyperparameter Tuning (GridSearch)":
    st.markdown('<p class="main-header">Hyperparameter Tuning via 5-Fold Grid Search CV</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimized performance metrics using Stratified K-Fold cross validation.</p>', unsafe_allow_html=True)
    
    st.subheader("🌟 Tuned Model Performance Metrics Table")
    st.dataframe(tuned_df.style.format({
        'Train Accuracy': '{:.4f}', 'Test Accuracy': '{:.4f}', 
        'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1 Score': '{:.4f}', 'AUC': '{:.4f}'
    }), use_container_width=True)
    
    st.markdown("""
    > **Grid Search Configuration Highlights:**
    > * **Cross-Validation:** 5-Fold Stratified
    > * **Scoring Metric:** F1 Score (Optimizing minority class balance)
    > * **KNN:** Tuned over `n_neighbors` and `weights`.
    > * **Decision Tree & Random Forest:** Tuned over `max_depth` and `min_samples_split`/`n_estimators`.
    > * **GBRT:** Tuned over `learning_rate` and `n_estimators`.
    """)

elif app_mode == "⚖️ Side-by-Side Performance Comparison":
    st.markdown('<p class="main-header">Side-by-Side Performance Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Direct side-by-side contrast of metrics before and after Grid Search CV hyperparameter tuning.</p>', unsafe_allow_html=True)
    
    df_before = init_df.copy()
    df_after = tuned_df.copy()
    
    df_before = df_before.rename(columns={
        'Train Accuracy': 'Train Acc (Before)', 'Test Accuracy': 'Test Acc (Before)',
        'Precision': 'Precision (Before)', 'Recall': 'Recall (Before)', 
        'F1 Score': 'F1 Score (Before)', 'AUC': 'AUC (Before)'
    })
    
    df_after = df_after.rename(columns={
        'Train Accuracy': 'Train Acc (After)', 'Test Accuracy': 'Test Acc (After)',
        'Precision': 'Precision (After)', 'Recall': 'Recall (After)', 
        'F1 Score': 'F1 Score (After)', 'AUC': 'AUC (After)'
    })
    
    comparison_table = pd.merge(df_before, df_after, on='Model')
    comparison_table.set_index('Model', inplace=True)
    
    st.subheader("🔍 Comprehensive Before vs. After Grid Search CV Matrix")
    styled_comp = (comparison_table
                   .style
                   .format("{:.4f}")
                   .background_gradient(cmap='Blues', subset=['Test Acc (Before)', 'F1 Score (Before)'])
                   .background_gradient(cmap='Greens', subset=['Test Acc (After)', 'F1 Score (After)']))
    
    st.dataframe(styled_comp, use_container_width=True)