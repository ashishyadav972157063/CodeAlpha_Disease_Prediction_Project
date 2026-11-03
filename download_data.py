import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

os.makedirs("data", exist_ok=True)

# 1. Breast Cancer Dataset (UCI Wisconsin Diagnostic)
print("Downloading Breast Cancer dataset...")
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df_cancer['target'] = cancer.target  # 0 = Malignant, 1 = Benign (we invert so 1 = Disease)
df_cancer['target'] = 1 - df_cancer['target']
df_cancer.to_csv("data/breast_cancer.csv", index=False)
print("Saved data/breast_cancer.csv")

# 2. Diabetes Dataset (UCI Pima Indians Diabetes)
print("Downloading Diabetes dataset...")
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'target'
]
df_diabetes = pd.read_csv(diabetes_url, names=diabetes_cols)
# Replace biological zero-values with median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df_diabetes[col] = df_diabetes[col].replace(0, np.nan)
    df_diabetes[col] = df_diabetes[col].fillna(df_diabetes[col].median())
df_diabetes.to_csv("data/diabetes.csv", index=False)
print("Saved data/diabetes.csv")

# 3. Heart Disease Dataset (UCI Cleveland Heart Disease)
print("Downloading Heart Disease dataset...")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]
df_heart = pd.read_csv(heart_url, names=heart_cols, na_values="?")
df_heart = df_heart.dropna().reset_index(drop=True)
df_heart['target'] = (df_heart['num'] > 0).astype(int)
df_heart = df_heart.drop(columns=['num'])
df_heart.to_csv("data/heart_disease.csv", index=False)
print("Saved data/heart_disease.csv")

print("\nAll 3 datasets successfully prepared in the 'data/' folder!")