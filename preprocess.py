from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import FeatureEngineer

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "churn"

RAW_TO_CANONICAL = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

DISPLAY_NAMES = {
    "customer_name": "Customer Name",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "Tenure",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
}

BASE_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]

ENGINEERED_NUMERIC_FEATURES = [
    "avg_monthly_spend",
    "service_count",
    "total_services_used",
    "support_service_count",
    "streaming_service_count",
    "support_gap",
    "streaming_user",
    "internet_intensity",
    "charge_to_tenure_ratio",
    "high_risk_customer",
    "long_term_customer",
    "premium_customer",
    "engagement_score",
]

ENGINEERED_CATEGORICAL_FEATURES = ["tenure_group"]

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    *ENGINEERED_NUMERIC_FEATURES,
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    *ENGINEERED_CATEGORICAL_FEATURES,
]

PREDICTION_DEFAULTS = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 75.0,
    "total_charges": 900.0,
}

FORM_FIELDS = [
    {"name": "customer_name", "label": "Customer Name", "type": "text", "placeholder": "Acme Telecom - John Doe"},
    {"name": "gender", "label": "Gender", "type": "select", "options": ["Female", "Male"]},
    {"name": "senior_citizen", "label": "Senior Citizen", "type": "select", "options": ["0", "1"]},
    {"name": "partner", "label": "Partner", "type": "select", "options": ["No", "Yes"]},
    {"name": "dependents", "label": "Dependents", "type": "select", "options": ["No", "Yes"]},
    {"name": "tenure", "label": "Tenure (months)", "type": "number", "min": 0, "max": 120, "step": 1},
    {"name": "phone_service", "label": "Phone Service", "type": "select", "options": ["No", "Yes"]},
    {"name": "multiple_lines", "label": "Multiple Lines", "type": "select", "options": ["No", "Yes", "No phone service"]},
    {"name": "internet_service", "label": "Internet Service", "type": "select", "options": ["DSL", "Fiber optic", "No"]},
    {"name": "online_security", "label": "Online Security", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "online_backup", "label": "Online Backup", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "device_protection", "label": "Device Protection", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "tech_support", "label": "Tech Support", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "streaming_tv", "label": "Streaming TV", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "streaming_movies", "label": "Streaming Movies", "type": "select", "options": ["No", "Yes", "No internet service"]},
    {"name": "contract", "label": "Contract", "type": "select", "options": ["Month-to-month", "One year", "Two year"]},
    {"name": "paperless_billing", "label": "Paperless Billing", "type": "select", "options": ["No", "Yes"]},
    {
        "name": "payment_method",
        "label": "Payment Method",
        "type": "select",
        "options": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    },
    {"name": "monthly_charges", "label": "Monthly Charges", "type": "number", "min": 0, "max": 200, "step": 0.01},
    {"name": "total_charges", "label": "Total Charges", "type": "number", "min": 0, "max": 10000, "step": 0.01},
]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={key: value for key, value in RAW_TO_CANONICAL.items() if key in frame.columns})


def _standardize_string(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else np.nan
    return value


def load_telco_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Add {DATASET_FILENAME} to the dataset directory."
        )
    frame = pd.read_csv(dataset_path)
    return clean_telco_dataframe(frame)


def clean_telco_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_columns(frame.copy())
    # `DataFrame.applymap` is not available across every pandas version we may run on.
    cleaned = cleaned.apply(lambda column: column.map(_standardize_string))
    cleaned = cleaned.drop_duplicates()

    if "total_charges" in cleaned.columns:
        cleaned["total_charges"] = pd.to_numeric(cleaned["total_charges"], errors="coerce")
    if "monthly_charges" in cleaned.columns:
        cleaned["monthly_charges"] = pd.to_numeric(cleaned["monthly_charges"], errors="coerce")
    if "tenure" in cleaned.columns:
        cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    if "senior_citizen" in cleaned.columns:
        cleaned["senior_citizen"] = pd.to_numeric(cleaned["senior_citizen"], errors="coerce").fillna(0).astype(int)

    if {"monthly_charges", "tenure", "total_charges"}.issubset(cleaned.columns):
        inferred_total = cleaned["monthly_charges"].fillna(0) * cleaned["tenure"].fillna(0)
        cleaned["total_charges"] = cleaned["total_charges"].fillna(inferred_total)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = (
            cleaned[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_engineer = FeatureEngineer()
    engineered = feature_engineer.fit_transform(frame.copy())
    return engineered


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame[BASE_FEATURES].copy()
    y = frame[TARGET_COLUMN].copy()
    return X, y


def _payload_aliases(field_name: str) -> set[str]:
    display_name = DISPLAY_NAMES.get(field_name, field_name.replace("_", " ").title())
    aliases = {
        field_name,
        display_name,
        display_name.lower(),
        display_name.replace(" ", "_").lower(),
        display_name.replace(" ", ""),
    }
    for raw, canonical in RAW_TO_CANONICAL.items():
        if canonical == field_name:
            aliases.add(raw)
    return aliases


def canonicalize_payload(payload: dict[str, Any]) -> pd.DataFrame:
    normalized: dict[str, Any] = {}
    for field_name in BASE_FEATURES:
        value = None
        for alias in _payload_aliases(field_name):
            if alias in payload and payload[alias] not in ("", None):
                value = payload[alias]
                break
        if value is None:
            value = PREDICTION_DEFAULTS[field_name]
        normalized[field_name] = value

    numeric_fields = {"senior_citizen", "tenure", "monthly_charges", "total_charges"}
    for field_name in numeric_fields:
        normalized[field_name] = pd.to_numeric(normalized[field_name], errors="coerce")
        if pd.isna(normalized[field_name]):
            normalized[field_name] = PREDICTION_DEFAULTS[field_name]

    normalized["senior_citizen"] = int(normalized["senior_citizen"])
    normalized["tenure"] = int(float(normalized["tenure"]))
    normalized["monthly_charges"] = float(normalized["monthly_charges"])
    normalized["total_charges"] = float(normalized["total_charges"])

    return pd.DataFrame([normalized], columns=BASE_FEATURES)
