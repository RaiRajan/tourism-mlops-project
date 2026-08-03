"""Train, tune, track and export the Wellness Tourism Package classifier."""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
EXPERIMENT = "wellness-tourism-package"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]


def log_metrics(prefix, y_true, y_pred, y_prob):
    metrics = {
        f"{prefix}_accuracy":  accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1":        f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc":   roc_auc_score(y_true, y_prob),
    }
    mlflow.log_metrics(metrics)
    print(f"\n{prefix.upper()} performance")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}")
    print("  confusion matrix:", confusion_matrix(y_true, y_pred).tolist())
    return metrics


def main():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    print(f"Xtrain {Xtrain.shape}   Xtest {Xtest.shape}")

    scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
    print("scale_pos_weight:", round(scale_pos_weight, 3))

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL_FEATURES),
    ])

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "model__n_estimators":  [200, 350],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
        "model__reg_lambda":    [1, 5],
    }

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    print("MLflow tracking URI:", tracking_uri)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)

    with mlflow.start_run(run_name="xgboost_grid_search") as parent:
        search.fit(Xtrain, ytrain)

        # Log every hyperparameter candidate as a nested run
        results = search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
                mlflow.log_metric("cv_std_f1", results["std_test_score"][i])

        mlflow.log_params(search.best_params_)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("best_cv_f1", search.best_score_)

        best = search.best_estimator_

        log_metrics("train", ytrain, best.predict(Xtrain),
                    best.predict_proba(Xtrain)[:, 1])
        test_metrics = log_metrics("test", ytest, best.predict(Xtest),
                                   best.predict_proba(Xtest)[:, 1])

        print("\nTest classification report")
        print(classification_report(ytest, best.predict(Xtest), zero_division=0))

        print("Best hyperparameters:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")
        print("Best CV F1:", round(search.best_score_, 4))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        print("\nModel saved to", MODEL_PATH)
        print("MLflow parent run id:", parent.info.run_id)


if __name__ == "__main__":
    main()
