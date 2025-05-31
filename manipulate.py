# import sys
# !{sys.executable} -m pip install --upgrade pip
# !{sys.executable} -m pip install pandas numpy matplotlib seaborn scikit-learn imblearn shap

# Import core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as display

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB # Removed due to low performance
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# XAI
import shap

# Load dataset
df = pd.read_csv('(CSV)dataset-uci.csv')

# Inspect dataset shape and columns
print(f'Dataset shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')

target_col = 'Gallstone Status'  # Target variable

# -------------------------------
# Rigorous EDA starts here

# Step 1: Check for missing values
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Step 2: Descriptive statistics for numerical features (including labs and others)
num_features = df.select_dtypes(include=np.number).columns.tolist()
num_features = [f for f in num_features if f != target_col]

print("\nDescriptive statistics for numerical features:")
display.display(df[num_features].describe())

# Step 3: Visualize distributions for selected numerical features (lab tests + others)
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(5, 5, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Step 4: Boxplots to check for outliers in numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(5, 5, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Step 5: Correlation heatmap for numerical features
plt.figure(figsize=(12, 10))
sns.heatmap(df[num_features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Step 6: Overall class distribution
print("\nOverall class distribution:")
print(df[target_col].value_counts())
print("\nOverall class distribution (percentage):")
print(df[target_col].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,4))
sns.countplot(x=target_col, data=df)
plt.title('Overall Class Distribution')
plt.xlabel('Gallstone Status')
plt.ylabel('Number of Patients')
plt.show()

# Step 7: Gender-wise class distribution
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='Gender', hue=target_col)
gender_labels = ['Male', 'Female']
ax.set_xticks([0, 1])
ax.set_xticklabels(gender_labels)
plt.title('Gender-wise Distribution of Gallstone Status')
plt.xlabel('Gender')
plt.ylabel('Number of Patients')
plt.legend(title='Gallstone Status', labels=['Without Gallstones', 'With Gallstones'])
plt.show()

# Step 8: Class distribution per categorical variable (other than gender)
categorical_features = ['Comorbidity', 'Coronary Artery Disease (CAD)', 'Hypothyroidism', 
                        'Hyperlipidemia', 'Diabetes Mellitus (DM)', 'Hepatic Fat Accumulation (HFA)']

for cat in categorical_features:
    plt.figure(figsize=(6,4))
    ax = sns.countplot(data=df, x=cat, hue=target_col)
    plt.title(f'Gallstone Status by {cat}')
    plt.xlabel(cat)
    plt.ylabel('Number of Patients')
    plt.legend(title='Gallstone Status', labels=['Without Gallstones', 'With Gallstones'])
    plt.show()

# -------------------------------
# Proceed with preprocessing as before

# Lab test features (blood work and medical tests)
lab_features = ['Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)', 
               'Triglyceride', 'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)', 
               'Alkaline Phosphatase (ALP)', 'Creatinine', 'Glomerular Filtration Rate (GFR)', 
               'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D']

# Categorical features (binary/discrete variables) — repeated here for clarity
categorical_features = ['Gender', 'Comorbidity', 'Coronary Artery Disease (CAD)', 'Hypothyroidism', 
                       'Hyperlipidemia', 'Diabetes Mellitus (DM)', 'Hepatic Fat Accumulation (HFA)']

# Numerical features (all remaining features that are not categorical or target)
numerical_features = [col for col in df.columns if col not in categorical_features + lab_features + [target_col]]

# Preprocessing pipeline (already defined earlier)
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train-test split
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Apply preprocessing on train and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Feature selection with ANOVA F-value
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train_preprocessed, y_train)

f_scores = selector.scores_
feature_names_num = numerical_features
feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

feature_scores_df = pd.DataFrame({'Feature': all_feature_names, 'F_Score': f_scores})

plt.figure(figsize=(10,6))
sns.barplot(x='F_Score', y='Feature', data=feature_scores_df.sort_values(by='F_Score', ascending=False))
plt.title('ANOVA F-scores for All Features')
plt.show()

threshold = feature_scores_df['F_Score'].quantile(0.70)
print(f"Chosen F-score threshold: {threshold:.3f}")

selected_features_mask = f_scores >= threshold
selected_feature_names = all_feature_names[selected_features_mask]

print(f"Number of selected features: {selected_features_mask.sum()}")
print("Selected features:", selected_feature_names)

X_train_selected = X_train_preprocessed[:, selected_features_mask]
X_test_selected = X_test_preprocessed[:, selected_features_mask]

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Bagging Classifier': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
}

results = []

for model_name, model in models.items():
    print(f"\nCross-validating {model_name}...")
    cross_val_evaluate(model, X_train_selected, y_train, cv)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_selected)[:,1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test_selected)
        y_proba = (y_proba - y_proba.min())/(y_proba.max()-y_proba.min())
    else:
        y_proba = None
    print(f"Test set evaluation for {model_name}:")
    evaluate_model(y_test, y_pred, y_proba)

    results.append({
        "Method": model_name,
        "Precision": round(precision_score(y_test, y_pred), 2),
        "Recall": round(recall_score(y_test, y_pred), 2),
        "F-measure": round(f1_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, y_proba), 2) if y_proba is not None else None
    })
