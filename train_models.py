import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)

datasets = {
    "Diabetes": "data/diabetes.csv",
    "Heart Disease": "data/heart_disease.csv",
    "Breast Cancer": "data/breast_cancer.csv"
}

algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

all_metrics = []
trained_bundles = {}

for disease_name, file_path in datasets.items():
    print(f"\n================ {disease_name.upper()} ================")
    df = pd.read_csv(file_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    disease_models = {}

    for algo_name, clf in algorithms.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clf)
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        all_metrics.append({
            "Disease": disease_name,
            "Algorithm": algo_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        disease_models[algo_name] = {
            "pipeline": pipeline,
            "confusion_matrix": cm,
            "accuracy": acc,
            "roc_auc": auc
        }

        print(f"{algo_name:20s} | Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    trained_bundles[disease_name] = {
        "models": disease_models,
        "features": list(X.columns),
        "feature_stats": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean())
            } for col in X.columns
        }
    }

# Save everything for the Streamlit dashboard
joblib.dump(trained_bundles, "models/all_disease_models.pkl")
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("models/metrics_comparison.csv", index=False)
print("\nTraining complete! Saved 'models/all_disease_models.pkl' and 'models/metrics_comparison.csv'.")