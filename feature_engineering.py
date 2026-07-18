from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

SUPPORT_COLUMNS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
]

STREAMING_COLUMNS = [
    "streaming_tv",
    "streaming_movies",
]

YES_LIKE_VALUES = {
    "yes",
    "dsl",
    "fiber optic",
    "fiber",
    "true",
    "1",
    1,
    True,
}


def _is_yes_like(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in YES_LIKE_VALUES)
    return int(value in YES_LIKE_VALUES)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features that usually improve churn performance."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        frame = X.copy()
        monthly = pd.to_numeric(frame.get("monthly_charges"), errors="coerce")
        tenure = pd.to_numeric(frame.get("tenure"), errors="coerce")
        self.monthly_charge_threshold_ = float(monthly.median()) if monthly.notna().any() else 70.0
        self.tenure_threshold_ = float(tenure.median()) if tenure.notna().any() else 24.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        for column in ("tenure", "monthly_charges", "total_charges", "senior_citizen"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["tenure"] = frame.get("tenure", pd.Series(dtype=float)).fillna(0)
        frame["monthly_charges"] = frame.get("monthly_charges", pd.Series(dtype=float)).fillna(0.0)
        frame["total_charges"] = frame.get("total_charges", pd.Series(dtype=float)).fillna(0.0)
        frame["senior_citizen"] = frame.get("senior_citizen", pd.Series(dtype=float)).fillna(0).astype(int)

        service_count = pd.Series(0, index=frame.index, dtype=float)
        support_count = pd.Series(0, index=frame.index, dtype=float)
        streaming_count = pd.Series(0, index=frame.index, dtype=float)

        for column in SERVICE_COLUMNS:
            values = frame.get(column, pd.Series(index=frame.index, dtype=object))
            service_count += values.apply(_is_yes_like)

        for column in SUPPORT_COLUMNS:
            values = frame.get(column, pd.Series(index=frame.index, dtype=object))
            support_count += values.apply(_is_yes_like)

        for column in STREAMING_COLUMNS:
            values = frame.get(column, pd.Series(index=frame.index, dtype=object))
            streaming_count += values.apply(_is_yes_like)

        tenure_denominator = frame["tenure"].replace(0, np.nan)
        avg_monthly_spend = (frame["total_charges"] / tenure_denominator).replace([np.inf, -np.inf], np.nan)
        avg_monthly_spend = avg_monthly_spend.fillna(frame["monthly_charges"])

        frame["avg_monthly_spend"] = avg_monthly_spend.round(2)
        frame["service_count"] = service_count.astype(int)
        frame["total_services_used"] = frame["service_count"]
        frame["support_service_count"] = support_count.astype(int)
        frame["streaming_service_count"] = streaming_count.astype(int)
        frame["support_gap"] = (support_count == 0).astype(int)
        frame["streaming_user"] = (streaming_count > 0).astype(int)
        frame["internet_intensity"] = (
            service_count
            / np.maximum(
                frame["tenure"].replace(0, np.nan),
                1,
            )
        ).fillna(service_count)
        frame["charge_to_tenure_ratio"] = (
            frame["monthly_charges"] / np.maximum(frame["tenure"], 1)
        ).round(4)
        frame["long_term_customer"] = (frame["tenure"] >= 24).astype(int)
        frame["premium_customer"] = (
            frame["monthly_charges"] >= self.monthly_charge_threshold_
        ).astype(int)
        frame["high_risk_customer"] = (
            (
                frame.get("contract", pd.Series(index=frame.index, dtype=object))
                .fillna("")
                .astype(str)
                .str.lower()
                .eq("month-to-month")
            )
            & (frame["tenure"] < max(self.tenure_threshold_, 12))
            & (frame["monthly_charges"] >= self.monthly_charge_threshold_)
        ).astype(int)

        frame["engagement_score"] = (
            frame["service_count"]
            + frame["long_term_customer"]
            + frame.get("partner", pd.Series(index=frame.index, dtype=object))
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("yes")
            .astype(int)
        )

        frame["tenure_group"] = pd.cut(
            frame["tenure"],
            bins=[-1, 12, 24, 48, 72, np.inf],
            labels=["0-12", "13-24", "25-48", "49-72", "72+"],
        ).astype(str)

        return frame
