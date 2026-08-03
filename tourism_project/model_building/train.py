"""Train, tune, track and export the Wellness Tourism Package classifier."""

import os
import json
import joblib
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier

RANDOM_STATE = 42
DEPLOY_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_tourism_model_v1.joblib")

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


def build_pipeline(scale_pos_weight):
    """Preprocessing and XGBoost combined in a single sklearn pipeline."""
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_tf, NUMERIC_FEATURES),
        ("cat", categorical_tf, CATEGORICAL_FEATURES),
    ])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,   # compensates for class imbalance
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main():
    # 1. Load the train/test splits produced by the previous job (workflow artifact)
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest  = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns").astype(int)
    ytest  = pd.read_csv("ytest.csv").squeeze("columns").astype(int)
    print(f"Loaded Xtrain {Xtrain.shape}, Xtest {Xtest.shape}")

    # 2. Define the model and the hyperparameter grid
    spw = float((ytrain == 0).sum() / max((ytrain == 1).sum(), 1))
    print(f"scale_pos_weight for class imbalance: {spw:.3f}")
    pipe = build_pipeline(spw)

    param_grid = {
        "model__n_estimators":  [200, 350],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
    }

    # 3. Configure MLflow tracking (server if reachable, else local file store)
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "http://127.0.0.1:5000"))
        mlflow.set_experiment("wellness-tourism-package")
    except Exception as e:
        print("Tracking server unavailable, falling back to file store:", e)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("wellness-tourism-package")

    with mlflow.start_run(run_name="xgb_gridsearch"):
        # 4. Tune the model, optimising F1 because of the class imbalance
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring="f1",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1, verbose=1, refit=True,
        )
        search.fit(Xtrain, ytrain)
        best = search.best_estimator_

        # 5. Log every hyperparameter combination tried, as nested runs
        results = pd.DataFrame(search.cv_results_)
        for _, row in results.iterrows():
            with mlflow.start_run(nested=True):
                mlflow.log_params({k: row[f"param_{k}"] for k in param_grid})
                mlflow.log_metric("cv_mean_f1", float(row["mean_test_score"]))
                mlflow.log_metric("cv_std_f1",  float(row["std_test_score"]))

        mlflow.log_params(search.best_params_)
        mlflow.log_param("scale_pos_weight", spw)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))

        # 6. Evaluate the best model on both splits
        for name, Xs, ys in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
            pred  = best.predict(Xs)
            proba = best.predict_proba(Xs)[:, 1]
            metrics = {
                f"{name}_accuracy":  accuracy_score(ys, pred),
                f"{name}_precision": precision_score(ys, pred, zero_division=0),
                f"{name}_recall":    recall_score(ys, pred, zero_division=0),
                f"{name}_f1":        f1_score(ys, pred, zero_division=0),
                f"{name}_roc_auc":   roc_auc_score(ys, proba),
            }
            mlflow.log_metrics(metrics)
            print(f"\n===== {name.upper()} PERFORMANCE =====")
            print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
            print(classification_report(ys, pred, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(ys, pred))

        # 7. Save the best model so the pipeline can commit it to the repository
        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best CV F1     : {search.best_score_:.4f}")
        print(f"Model saved to : {MODEL_PATH}")


if __name__ == "__main__":
    main()
