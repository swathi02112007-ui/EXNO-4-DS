# -*- coding: utf-8 -*-
"""
EXNO:4 - FEATURE SCALING AND FEATURE SELECTION
"""

# ==========================================
# 📦 IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🔹 FEATURE SCALING
# ==========================================
print("\n==================== FEATURE SCALING ====================\n")

# Load sample dataset (replace path if needed)
df = pd.read_csv("/content/bmi.csv")   # Example CSV
print("Original Data:")
print(df.head())

# Drop missing values
df = df.dropna()

# Find maximum values
print("\nMaximum Height:", df['Height'].max())
print("Maximum Weight:", df['Weight'].max())

# ------------------------------
# 1️⃣ Min-Max Scaler
# ------------------------------
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[['Height', 'Weight']] = minmax.fit_transform(df[['Height', 'Weight']])
print("\nAfter Min-Max Scaling:")
print(df_minmax.head())

# ------------------------------
# 2️⃣ Standard Scaler
# ------------------------------
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
df_std = df.copy()
df_std[['Height', 'Weight']] = std.fit_transform(df[['Height', 'Weight']])
print("\nAfter Standard Scaling:")
print(df_std.head())

# ------------------------------
# 3️⃣ Normalizer
# ------------------------------
from sklearn.preprocessing import Normalizer
norm = Normalizer()
df_norm = df.copy()
df_norm[['Height', 'Weight']] = norm.fit_transform(df[['Height', 'Weight']])
print("\nAfter Normalization:")
print(df_norm.head())

# ------------------------------
# 4️⃣ MaxAbs Scaler
# ------------------------------
from sklearn.preprocessing import MaxAbsScaler
maxabs = MaxAbsScaler()
df_maxabs = df.copy()
df_maxabs[['Height', 'Weight']] = maxabs.fit_transform(df[['Height', 'Weight']])
print("\nAfter MaxAbs Scaling:")
print(df_maxabs.head())

# ------------------------------
# 5️⃣ Robust Scaler
# ------------------------------
from sklearn.preprocessing import RobustScaler
rob = RobustScaler()
df_rob = df.copy()
df_rob[['Height', 'Weight']] = rob.fit_transform(df[['Height', 'Weight']])
print("\nAfter Robust Scaling:")
print(df_rob.head())

# ==========================================
# 🔹 FEATURE SELECTION
# ==========================================
print("\n==================== FEATURE SELECTION ====================\n")

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, mutual_info_regression, chi2

# Load Titanic dataset
df = pd.read_csv('/content/titanic_dataset.csv')
print("Original Titanic Data Columns:")
print(df.columns)
print("\nShape:", df.shape)

# Define feature matrix and target
X = df.drop("Survived", axis=1)
y = df['Survived']

# Drop unnecessary columns
df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
print("\nAfter dropping unnecessary columns:")
print(df1.columns)

# Handle missing values
print("\nMissing Age values before fill:", df1['Age'].isnull().sum())
df1['Age'] = df1['Age'].fillna(method='ffill')
print("Missing Age values after fill:", df1['Age'].isnull().sum())

# Reorder columns
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
print("\nReordered Columns:")
print(df1.columns)

# Separate X and y
X = df1.drop("Survived", axis=1)
y = df1['Survived']

# ==========================================
# 1️⃣ SelectKBest (Mutual Info)
# ==========================================
print("\n--- SelectKBest (Mutual Info Classification) ---")
selector = SelectKBest(score_func=mutual_info_classif, k=3)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Top 3 Features:", list(selected_features))

# ==========================================
# 2️⃣ Recursive Feature Elimination (RFE)
# ==========================================
print("\n--- Recursive Feature Elimination (RFE) ---")
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X, y)
print("RFE Selected Features:", list(X.columns[rfe.support_]))

# ==========================================
# 3️⃣ LASSO (L1 Regularization)
# ==========================================
print("\n--- LASSO Feature Selection ---")
lasso = LassoCV(cv=5)
lasso.fit(X, y)
coef = pd.Series(lasso.coef_, index=X.columns)
print("Lasso Coefficients:")
print(coef)
print("Selected Features:", list(coef[coef != 0].index))

# ==========================================
# 4️⃣ RIDGE (L2 Regularization)
# ==========================================
print("\n--- RIDGE Feature Selection ---")
ridge = RidgeCV(cv=5)
ridge.fit(X, y)
ridge_coef = pd.Series(ridge.coef_, index=X.columns)
print("Ridge Coefficients:")
print(ridge_coef)

# ==========================================
# 5️⃣ Chi-Square Test (for categorical/discrete data)
# ==========================================
print("\n--- Chi-Square Test ---")
# For chi2, features must be non-negative
X_chi = X.copy()
X_chi[X_chi < 0] = 0
chi_selector = SelectKBest(chi2, k=3)
chi_selector.fit(X_chi, y)
print("Chi-Square Selected Features:", list(X.columns[chi_selector.get_support()]))

# ==========================================
# 🎯 FINAL OUTPUT
# ==========================================
print("\n==================== SUMMARY ====================\n")
print("SelectKBest Top Features:", list(selected_features))
print("RFE Top Features:", list(X.columns[rfe.support_]))
print("Lasso Top Features:", list(coef[coef != 0].index))
print("Ridge Coefficients:", ridge_coef.to_dict())
print("Chi2 Top Features:", list(X.columns[chi_selector.get_support()]))
print("\n✅ Feature Scaling and Selection Completed Successfully!")
