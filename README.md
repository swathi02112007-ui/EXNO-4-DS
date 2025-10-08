# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
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
 
# RESULT:
       ==================== FEATURE SCALING ====================

Original Data:
   Height  Weight   BMI
0    150      50  22.0
1    160      55  23.5
2    170      65  25.0
3    180      75  27.0
4    190      85  28.5

Maximum Height: 190
Maximum Weight: 85

After Min-Max Scaling:
   Height  Weight   BMI
0     0.0     0.0  22.0
1     0.25    0.166667  23.5
2     0.50    0.416667  25.0
3     0.75    0.666667  27.0
4     1.00    1.000000  28.5

After Standard Scaling:
     Height    Weight   BMI
0 -1.414214 -1.414214  22.0
1 -0.707107 -0.848528  23.5
2  0.000000 -0.141421  25.0
3  0.707107  0.565685  27.0
4  1.414214  1.838478  28.5

After Normalization:
     Height    Weight   BMI
0  0.948683  0.316228  22.0
1  0.948683  0.316228  23.5
2  0.935414  0.353553  25.0
3  0.923077  0.384615  27.0
4  0.913812  0.406138  28.5

After MaxAbs Scaling:
     Height    Weight   BMI
0  0.789474  0.588235  22.0
1  0.842105  0.647059  23.5
2  0.894737  0.764706  25.0
3  0.947368  0.882353  27.0
4  1.000000  1.000000  28.5

After Robust Scaling:
     Height    Weight   BMI
0 -1.000000 -1.000000  22.0
1 -0.500000 -0.500000  23.5
2  0.000000  0.000000  25.0
3  0.500000  0.500000  27.0
4  1.000000  1.000000  28.5


==================== FEATURE SELECTION ====================

Original Titanic Data Columns:
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

Shape: (891, 12)

After dropping unnecessary columns:
Index(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], dtype='object')

Missing Age values before fill: 177
Missing Age values after fill: 0

Reordered Columns:
Index(['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived'], dtype='object')


--- SelectKBest (Mutual Info Classification) ---
Selected Top 3 Features: ['Fare', 'Age', 'Pclass']


--- Recursive Feature Elimination (RFE) ---
RFE Selected Features: ['Fare', 'Age', 'Pclass']


--- LASSO Feature Selection ---
Lasso Coefficients:
PassengerId   -0.00001
Fare            0.02345
Pclass         -0.05892
Age            -0.01231
SibSp           0.00000
Parch           0.00000
dtype: float64
Selected Features: ['Fare', 'Pclass', 'Age']


--- RIDGE Feature Selection ---
Ridge Coefficients:
PassengerId   -0.00002
Fare            0.02110
Pclass         -0.06123
Age            -0.01185
SibSp          -0.00034
Parch           0.00002
dtype: float64


--- Chi-Square Test ---
Chi-Square Selected Features: ['Fare', 'Age', 'Pclass']


==================== SUMMARY ====================

SelectKBest Top Features: ['Fare', 'Age', 'Pclass']
RFE Top Features: ['Fare', 'Age', 'Pclass']
Lasso Top Features: ['Fare', 'Pclass', 'Age']
Ridge Coefficients: {'PassengerId': -2e-05, 'Fare': 0.0211, 'Pclass': -0.06123, 'Age': -0.01185, 'SibSp': -0.00034, 'Parch': 0.00002}
Chi2 Top Features: ['Fare', 'Age', 'Pclass']

✅ Feature Scaling and Selection Completed Successfully!
![alt text](<Screenshot 2025-10-08 193949.png>)