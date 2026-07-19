from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import FeatureEngineer
from preprocess import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_FILENAME,
    FORM_FIELDS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    clean_telco_dataframe,
    load_telco_dataset,
    split_features_and_target,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / DATASET_FILENAME
MODEL_DIR = BASE_DIR / "model"
EDA_DIR = BASE_DIR / "static" / "images" / "eda"


def ensure_directories() -> None:
    for directory in (MODEL_DIR, EDA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_model_registry() -> dict[str, object]:
    registry: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }
    if XGBClassifier is not None:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    if LGBMClassifier is not None:
        registry["LightGBM"] = LGBMClassifier(random_state=42, n_estimators=300)
    if CatBoostClassifier is not None:
        registry["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=42,
        )
    return registry


def generate_eda(frame: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plot_specs = [
        ("churn_distribution.png", frame[TARGET_COLUMN].map({0: "No", 1: "Yes"}), "Churn Distribution", "count"),
        ("gender_distribution.png", frame["gender"], "Gender Distribution", "count"),
        ("contract_type.png", frame["contract"], "Contract Type", "count"),
        ("monthly_charges.png", frame["monthly_charges"], "Monthly Charges", "hist"),
        ("total_charges.png", frame["total_charges"], "Total Charges", "hist"),
        ("tenure_distribution.png", frame["tenure"], "Tenure", "hist"),
        ("internet_service.png", frame["internet_service"], "Internet Service", "count"),
        ("payment_method.png", frame["payment_method"], "Payment Method", "count"),
        ("senior_citizen.png", frame["senior_citizen"].map({0: "No", 1: "Yes"}), "Senior Citizen", "count"),
        ("partner.png", frame["partner"], "Partner", "count"),
        ("dependents.png", frame["dependents"], "Dependents", "count"),
    ]

    for filename, series, title, plot_type in plot_specs:
        plt.figure(figsize=(10, 5))
        if plot_type == "count":
            order = series.value_counts().index
            ax = sns.countplot(x=series, order=order, palette="crest")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        else:
            sns.histplot(series, kde=True, color="#0f766e")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(EDA_DIR / filename, dpi=180)
        plt.close()


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "fpr": roc_curve(y_test, probabilities)[0].tolist(),
        "tpr": roc_curve(y_test, probabilities)[1].tolist(),
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def save_best_model_assets(
    best_name: str,
    best_pipeline: Pipeline,
    metrics_by_model: dict[str, dict[str, object]],
    X_train: pd.DataFrame,
) -> None:
    feature_engineer = best_pipeline.named_steps["feature_engineer"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    engineered_background = feature_engineer.transform(X_train.sample(min(150, len(X_train)), random_state=42))
    background_transformed = preprocessor.transform(engineered_background)
    background_transformed = (
        background_transformed.toarray() if hasattr(background_transformed, "toarray") else np.asarray(background_transformed)
    )
    feature_names = preprocessor.get_feature_names_out().tolist()
    background_means = background_transformed.mean(axis=0).tolist()

    joblib.dump(best_pipeline, MODEL_DIR / "model.pkl")

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    joblib.dump(numeric_pipeline.named_steps["scaler"], MODEL_DIR / "scaler.pkl")
    joblib.dump(categorical_pipeline.named_steps["encoder"], MODEL_DIR / "encoder.pkl")
    joblib.dump(
        {
            "background_transformed": background_transformed,
            "background_means": background_means,
            "feature_names": feature_names,
        },
        MODEL_DIR / "explanation_assets.pkl",
    )

    metadata = {
        "best_model": best_name,
        "dataset_file": DATASET_FILENAME,
        "base_features": BASE_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
        "form_fields": FORM_FIELDS,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")


def save_evaluation_plots(best_name: str, metrics_by_model: dict[str, dict[str, object]]) -> None:
    comparison_rows = []
    for model_name, result in metrics_by_model.items():
        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
    comparison_frame.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_frame, x="roc_auc", y="model", palette="mako")
    plt.title("Model ROC AUC Comparison")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_comparison.png", dpi=180)
    plt.close()

    best_metrics = metrics_by_model[best_name]
    fpr = np.array(best_metrics["fpr"])
    tpr = np.array(best_metrics["tpr"])
    roc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC = {roc_score:.3f})", color="#0f766e", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "roc_curve.png", dpi=180)
    plt.close()

    confusion = np.array(best_metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_directories()
    frame = load_telco_dataset(DATASET_PATH)
    generate_eda(frame)
    X, y = split_features_and_target(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_registry = build_model_registry()
    metrics_by_model: dict[str, dict[str, object]] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for model_name, estimator in model_registry.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", FeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        evaluation = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        metrics_by_model[model_name] = evaluation["metrics"]

        if evaluation["metrics"]["roc_auc"] > best_score:
            best_score = evaluation["metrics"]["roc_auc"]
            best_name = model_name
            best_pipeline = evaluation["pipeline"]

    if best_pipeline is None:
        raise RuntimeError("No model pipeline was successfully trained.")

    save_best_model_assets(best_name, best_pipeline, metrics_by_model, X_train)
    save_evaluation_plots(best_name, metrics_by_model)
    print(f"Training complete. Best model: {best_name} (ROC AUC = {best_score:.4f})")


if __name__ == "__main__":
    main()
