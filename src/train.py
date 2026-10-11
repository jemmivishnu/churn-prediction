"""Trains churn models, tracks every run in MLflow, and exports the best model for the API."""

import json
import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "telco-churn-prediction"
REGISTERED_MODEL = "telco-churn-model"
SEED = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    X = df[NUMERIC + CATEGORICAL]
    return X, y


def preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def candidates(pos_weight):
    return [
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"C": 1.0}),
        ("random_forest", RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
         {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}),
        ("xgboost", XGBClassifier(random_state=SEED, eval_metric="logloss", scale_pos_weight=pos_weight),
         {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8}),
        ("xgboost", XGBClassifier(random_state=SEED, eval_metric="logloss", scale_pos_weight=pos_weight),
         {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
          "colsample_bytree": 0.8}),
        ("xgboost", XGBClassifier(random_state=SEED, eval_metric="logloss", scale_pos_weight=pos_weight),
         {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9}),
    ]


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_plots(pipeline, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=["No Churn", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    clf = pipeline.named_steps["model"]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        importances = abs(clf.coef_[0])
    names = pipeline.named_steps["prep"].get_feature_names_out()
    top = pd.Series(importances, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax)
    ax.set_title("Top 15 Features")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []
    for name, estimator, params in candidates(pos_weight):
        estimator.set_params(**params)
        pipeline = Pipeline([("prep", preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=name) as run:
            cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc").mean()
            pipeline.fit(X_train, y_train)
            metrics = {"cv_roc_auc": cv_auc, **evaluate(pipeline, X_test, y_test)}

            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            log_plots(pipeline, X_test, y_test)
            example = X_train.head(3)
            model_info = mlflow.sklearn.log_model(
                pipeline, name="model", input_example=example, serialization_format="cloudpickle",
                signature=infer_signature(example, pipeline.predict_proba(example)),
            )
            results.append({"run_id": run.info.run_id, "model_uri": model_info.model_uri, "model": name,
                            **params, **metrics, "pipeline": pipeline})
            print(f"{name:<20} cv_roc_auc={cv_auc:.4f}  test_roc_auc={metrics['test_roc_auc']:.4f}  "
                  f"test_f1={metrics['test_f1']:.4f}")

    # Select on cross-validation score so the test set stays an unbiased final check.
    best = max(results, key=lambda r: r["cv_roc_auc"])
    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_DIR / "model.joblib")
    metadata = {
        "model_type": best["model"],
        "mlflow_run_id": best["run_id"],
        "registered_model": f"{REGISTERED_MODEL} v{version.version}",
        "metrics": {k: round(v, 4) for k, v in best.items() if k.startswith(("cv_", "test_"))},
        "features": NUMERIC + CATEGORICAL,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nBest model: {best['model']} (run {best['run_id'][:8]}), "
          f"registered as {REGISTERED_MODEL} v{version.version} with alias 'champion'")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
