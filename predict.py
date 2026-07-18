from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preprocess import BASE_FEATURES, DISPLAY_NAMES, canonicalize_payload

try:
    import shap
except Exception:  # pragma: no cover - graceful fallback when SHAP is unavailable
    shap = None


class ModelNotReadyError(RuntimeError):
    pass


class PredictionService:
    def __init__(self, model_dir: str | Path = "model") -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.pkl"
        self.metadata_path = self.model_dir / "metadata.json"
        self.explanation_assets_path = self.model_dir / "explanation_assets.pkl"
        self._pipeline = None
        self._metadata: dict[str, Any] = {}
        self._explanation_assets: dict[str, Any] = {}
        self._explainer = None

    def is_ready(self) -> bool:
        return self.model_path.exists()

    def _load_assets(self) -> None:
        if self._pipeline is not None:
            return
        if not self.model_path.exists():
            raise ModelNotReadyError(
                "No trained model found. Add the Telco churn CSV to dataset/ and run `python train.py`."
            )

        self._pipeline = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.explanation_assets_path.exists():
            self._explanation_assets = joblib.load(self.explanation_assets_path)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._load_assets()
        frame = canonicalize_payload(payload)
        probability = self._predict_probability(frame)
        prediction = "Likely to Churn" if probability >= 0.5 else "Not Likely to Churn"
        risk_level = self._risk_level(probability)
        explanation = self._build_explanation(frame, probability)
        recommendations = self._build_recommendations(frame.iloc[0].to_dict(), risk_level)

        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "probability_score": round(probability, 4),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "explanation": explanation,
            "input": self._format_input(frame.iloc[0].to_dict()),
        }

    def _predict_probability(self, frame: pd.DataFrame) -> float:
        classifier = self._pipeline.named_steps["classifier"]
        if hasattr(self._pipeline, "predict_proba"):
            return float(self._pipeline.predict_proba(frame)[0, 1])
        if hasattr(classifier, "decision_function"):
            score = float(self._pipeline.decision_function(frame)[0])
            return float(1 / (1 + np.exp(-score)))
        prediction = self._pipeline.predict(frame)[0]
        return float(prediction)

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.75:
            return "High"
        if probability >= 0.45:
            return "Medium"
        return "Low"

    def _build_explanation(self, frame: pd.DataFrame, probability: float) -> dict[str, Any]:
        classifier = self._pipeline.named_steps["classifier"]
        feature_engineer = self._pipeline.named_steps["feature_engineer"]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        engineered = feature_engineer.transform(frame.copy())
        transformed = preprocessor.transform(engineered)
        dense_transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        feature_names = self._explanation_assets.get("feature_names")
        if not feature_names:
            feature_names = preprocessor.get_feature_names_out().tolist()

        contributions = self._compute_contributions(classifier, dense_transformed, feature_names)
        top_positive = [item for item in contributions if item["impact"] > 0][:4]
        top_negative = [item for item in reversed(contributions) if item["impact"] < 0][:3]

        if probability >= 0.5:
            summary = "The model sees a meaningful churn risk driven by price sensitivity, service mix, and contract profile."
        else:
            summary = "The model sees a relatively stable customer profile, with retention-oriented signals outweighing churn drivers."

        return {
            "summary": summary,
            "positive_drivers": top_positive,
            "negative_drivers": top_negative,
        }

    def _compute_contributions(
        self,
        classifier: Any,
        transformed: np.ndarray,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        impacts: np.ndarray | None = None

        if shap is not None:
            impacts = self._compute_shap_contributions(classifier, transformed, feature_names)

        if impacts is None:
            if hasattr(classifier, "coef_"):
                coef = np.asarray(classifier.coef_).reshape(-1)
                impacts = transformed[0] * coef[: transformed.shape[1]]
            elif hasattr(classifier, "feature_importances_"):
                importances = np.asarray(classifier.feature_importances_)
                background_means = np.asarray(
                    self._explanation_assets.get("background_means", np.zeros_like(importances))
                )
                impacts = (transformed[0] - background_means[: transformed.shape[1]]) * importances[: transformed.shape[1]]
            else:
                impacts = transformed[0]

        contribution_rows = []
        for feature_name, impact in zip(feature_names, impacts):
            if abs(float(impact)) < 1e-6:
                continue
            contribution_rows.append(
                {
                    "feature": self._humanize_feature_name(feature_name),
                    "impact": round(float(impact), 4),
                    "direction": "Increases churn risk" if impact > 0 else "Reduces churn risk",
                }
            )

        contribution_rows.sort(key=lambda item: item["impact"], reverse=True)
        return contribution_rows

    def _compute_shap_contributions(
        self,
        classifier: Any,
        transformed: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray | None:
        background = self._explanation_assets.get("background_transformed")
        if background is None or len(background) == 0:
            return None

        if self._explainer is None:
            try:
                if hasattr(classifier, "feature_importances_"):
                    self._explainer = shap.TreeExplainer(classifier)
                elif hasattr(classifier, "coef_"):
                    self._explainer = shap.LinearExplainer(classifier, background)
                else:
                    self._explainer = shap.Explainer(classifier.predict_proba, background, feature_names=feature_names)
            except Exception:
                return None

        try:
            shap_values = self._explainer(transformed)
            values = getattr(shap_values, "values", shap_values)
            values = np.asarray(values)
            if values.ndim == 3:
                return values[0, :, 1]
            if values.ndim == 2:
                return values[0]
            return values
        except Exception:
            return None

    def _humanize_feature_name(self, feature_name: str) -> str:
        normalized = feature_name
        if "__" in normalized:
            normalized = normalized.split("__", 1)[1]
        for base_feature in BASE_FEATURES + ["tenure_group"]:
            prefix = f"{base_feature}_"
            if normalized == base_feature:
                return DISPLAY_NAMES.get(base_feature, base_feature.replace("_", " ").title())
            if normalized.startswith(prefix):
                feature_label = DISPLAY_NAMES.get(base_feature, base_feature.replace("_", " ").title())
                option_label = normalized[len(prefix) :].replace("_", " ")
                return f"{feature_label}: {option_label}"
        return normalized.replace("_", " ").title()

    def _format_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        formatted = {}
        for field_name, value in payload.items():
            formatted[DISPLAY_NAMES.get(field_name, field_name.replace("_", " ").title())] = value
        return formatted

    def _build_recommendations(self, payload: dict[str, Any], risk_level: str) -> list[str]:
        recommendations: list[str] = []

        if str(payload.get("contract", "")).lower() == "month-to-month":
            recommendations.append("Promote a one-year or two-year contract with a retention discount.")
        if str(payload.get("tech_support", "")).lower() == "no":
            recommendations.append("Bundle proactive tech support to increase stickiness and reduce friction.")
        if str(payload.get("online_security", "")).lower() == "no":
            recommendations.append("Offer an online security add-on to improve perceived service value.")
        if float(payload.get("monthly_charges", 0)) >= 75:
            recommendations.append("Create a targeted price optimization or loyalty credit for this account.")
        if int(payload.get("tenure", 0)) < 12:
            recommendations.append("Trigger an onboarding nurture journey with loyalty rewards in the first year.")
        if str(payload.get("payment_method", "")).lower() == "electronic check":
            recommendations.append("Encourage auto-pay migration to a lower-friction payment method.")

        if risk_level == "Low" and not recommendations:
            recommendations.append("Maintain engagement with value-add communication and cross-sell campaigns.")

        if not recommendations:
            recommendations.append("Review the account with customer success for a personalized retention offer.")

        return recommendations[:5]
