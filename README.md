# tourism-mlops-project

Predicts whether a customer will purchase the Wellness Tourism Package,
so the sales team can prioritise outreach before making contact.

## Repository structure

    tourism_project/
      data/tourism.csv                     registered dataset
      model_building/data_register.py      schema validation and summary
      model_building/prep.py               cleaning and stratified splitting
      model_building/train.py              tuning, MLflow tracking, evaluation
      deployment/app.py                    Streamlit front end
      deployment/requirements.txt          deployment dependencies
      deployment/best_tourism_model_v1.joblib   model committed by the pipeline
      requirements.txt                     pipeline dependencies
    .github/workflows/pipeline.yml         CI/CD workflow

## Pipeline

Three chained GitHub Actions jobs run on every push to `main`:
`register-dataset` validates the schema, `data-prep` cleans and splits the
data and publishes the splits as an artifact, and `model-training` tunes an
XGBoost classifier, logs all runs to MLflow, evaluates the best model and
commits it back to `main`.

## Model performance (held-out test set)

| Metric | Score |
|--------|-------|
| F1 | 0.858 |
| Precision | 0.901 |
| Recall | 0.819 |
| ROC AUC | 0.961 |

## Live application

https://tourism-package-predictor.streamlit.app/
"""

with open("README.md", "w") as f:
    f.write(readme)

sha = repo.get_contents("README.md", ref=BRANCH).sha
repo.update_file("README.md", "add project README", readme, sha, branch=BRANCH)
print("README pushed.")
