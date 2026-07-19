from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash, generate_password_hash

from predict import ModelNotReadyError, PredictionService
from preprocess import DATASET_FILENAME, FORM_FIELDS, load_telco_dataset

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
DATABASE_PATH = INSTANCE_DIR / "churn_app.db"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_PATH = BASE_DIR / "dataset" / DATASET_FILENAME
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "replace-this-with-a-secure-key")
app.config["DATABASE"] = DATABASE_PATH

prediction_service = PredictionService(BASE_DIR / "model")


def ensure_directories() -> None:
    for directory in (
        BASE_DIR / "dataset",
        BASE_DIR / "model",
        BASE_DIR / "templates",
        BASE_DIR / "static" / "css",
        BASE_DIR / "static" / "js",
        BASE_DIR / "static" / "images" / "eda",
        REPORTS_DIR,
        INSTANCE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception: Exception | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            customer_name TEXT NOT NULL,
            input_json TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            churn_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendations_json TEXT NOT NULL,
            explanation_json TEXT NOT NULL,
            monthly_charges REAL,
            tenure INTEGER,
            contract TEXT,
            gender TEXT,
            payment_method TEXT,
            internet_service TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    db.commit()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/") or wants_json_response():
                return (
                    jsonify(
                        {
                            "error": "Authentication required.",
                            "login_url": url_for("login"),
                        }
                    ),
                    401,
                )
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def fetch_one(query: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    return get_db().execute(query, parameters).fetchone()


def fetch_all(query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return get_db().execute(query, parameters).fetchall()


def current_user() -> sqlite3.Row | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return fetch_one("SELECT id, full_name, email FROM users WHERE id = ?", (user_id,))


@app.context_processor
def inject_template_context() -> dict[str, Any]:
    return {
        "current_user": current_user(),
        "model_ready": prediction_service.is_ready(),
        "dataset_ready": DATASET_PATH.exists(),
    }


def serialize_prediction_row(row: sqlite3.Row) -> dict[str, Any]:
    recommendations = json.loads(row["recommendations_json"])
    explanation = json.loads(row["explanation_json"])
    return {
        "id": row["id"],
        "customer_name": row["customer_name"],
        "prediction": row["prediction_label"],
        "probability": round(float(row["churn_probability"]) * 100, 2),
        "probability_score": round(float(row["churn_probability"]), 4),
        "risk_level": row["risk_level"],
        "recommendations": recommendations,
        "explanation": explanation,
        "created_at": row["created_at"],
        "report_url": url_for("download_report", prediction_id=row["id"]),
    }


def wants_json_response() -> bool:
    return (
        request.path.startswith("/api/")
        or request.is_json
        or request.args.get("format") == "json"
        or request.accept_mimetypes.best == "application/json"
    )


def save_prediction(user_id: int, payload: dict[str, Any], result: dict[str, Any]) -> int:
    customer_name = str(payload.get("customer_name") or "Anonymous Customer").strip() or "Anonymous Customer"
    probability_score = float(result["probability_score"])
    input_data = result["input"]
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO predictions (
            user_id,
            customer_name,
            input_json,
            prediction_label,
            churn_probability,
            risk_level,
            recommendations_json,
            explanation_json,
            monthly_charges,
            tenure,
            contract,
            gender,
            payment_method,
            internet_service,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            customer_name,
            json.dumps(input_data),
            result["prediction"],
            probability_score,
            result["risk_level"],
            json.dumps(result["recommendations"]),
            json.dumps(result["explanation"]),
            float(input_data.get("Monthly Charges", 0)),
            int(float(input_data.get("Tenure", 0))),
            input_data.get("Contract", "Unknown"),
            input_data.get("Gender", "Unknown"),
            input_data.get("Payment Method", "Unknown"),
            input_data.get("Internet Service", "Unknown"),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def load_dataset_summary() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        return {}

    frame = load_telco_dataset(DATASET_PATH)
    return {
        "total_customers": int(len(frame)),
        "churn_rate": round(frame["churn"].mean() * 100, 2),
        "contract_distribution": frame["contract"].value_counts().to_dict(),
        "gender_distribution": frame["gender"].value_counts().to_dict(),
        "payment_method_distribution": frame["payment_method"].value_counts().to_dict(),
        "internet_service_distribution": frame["internet_service"].value_counts().to_dict(),
        "monthly_charge_bins": pd.cut(
            frame["monthly_charges"],
            bins=[0, 30, 60, 90, 120, float("inf")],
            labels=["0-30", "31-60", "61-90", "91-120", "120+"],
            include_lowest=True,
        )
        .value_counts()
        .sort_index()
        .to_dict(),
        "tenure_bins": pd.cut(
            frame["tenure"],
            bins=[0, 12, 24, 48, 72, float("inf")],
            labels=["0-12", "13-24", "25-48", "49-72", "72+"],
            include_lowest=True,
        )
        .value_counts()
        .sort_index()
        .to_dict(),
        "avg_monthly_charges": round(frame["monthly_charges"].mean(), 2),
    }


def load_model_summary() -> dict[str, Any]:
    metrics_path = MODEL_DIR / "metrics.json"
    metadata_path = MODEL_DIR / "metadata.json"
    if not metrics_path.exists() or not metadata_path.exists():
        return {}

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        best_model = metadata.get("best_model")
        best_metrics = metrics.get(best_model, {}) if best_model else {}
        return {
            "best_model": best_model,
            "roc_auc": round(float(best_metrics.get("roc_auc", 0)), 4) if best_metrics else None,
            "accuracy": round(float(best_metrics.get("accuracy", 0)) * 100, 2) if best_metrics else None,
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def build_dashboard_payload(user_id: int) -> dict[str, Any]:
    rows = fetch_all(
        """
        SELECT id, customer_name, prediction_label, churn_probability, risk_level,
               monthly_charges, tenure, contract, gender, payment_method,
               internet_service, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at ASC
        """,
        (user_id,),
    )
    dataset_summary = load_dataset_summary()

    total_predictions = len(rows)
    total_customers = len({row["customer_name"] for row in rows if row["customer_name"]})
    high_risk_customers = sum(1 for row in rows if row["risk_level"] == "High")
    avg_monthly_charges = round(
        sum(float(row["monthly_charges"] or 0) for row in rows) / total_predictions,
        2,
    ) if total_predictions else dataset_summary.get("avg_monthly_charges", 0)

    churn_predictions = sum(1 for row in rows if row["prediction_label"] == "Likely to Churn")
    predicted_churn_rate = round((churn_predictions / total_predictions) * 100, 2) if total_predictions else dataset_summary.get("churn_rate", 0)

    date_counter: dict[str, int] = defaultdict(int)
    churn_counter: dict[str, int] = defaultdict(int)
    contract_counter: Counter[str] = Counter()
    gender_counter: Counter[str] = Counter()
    payment_counter: Counter[str] = Counter()
    internet_counter: Counter[str] = Counter()
    charge_counter: Counter[str] = Counter()
    tenure_counter: Counter[str] = Counter()

    for row in rows:
        date_key = row["created_at"][:10]
        date_counter[date_key] += 1
        if row["prediction_label"] == "Likely to Churn":
            churn_counter[date_key] += 1
        contract_counter[row["contract"] or "Unknown"] += 1
        gender_counter[row["gender"] or "Unknown"] += 1
        payment_counter[row["payment_method"] or "Unknown"] += 1
        internet_counter[row["internet_service"] or "Unknown"] += 1

        monthly_charge = float(row["monthly_charges"] or 0)
        if monthly_charge <= 30:
            charge_counter["0-30"] += 1
        elif monthly_charge <= 60:
            charge_counter["31-60"] += 1
        elif monthly_charge <= 90:
            charge_counter["61-90"] += 1
        elif monthly_charge <= 120:
            charge_counter["91-120"] += 1
        else:
            charge_counter["120+"] += 1

        tenure = int(row["tenure"] or 0)
        if tenure <= 12:
            tenure_counter["0-12"] += 1
        elif tenure <= 24:
            tenure_counter["13-24"] += 1
        elif tenure <= 48:
            tenure_counter["25-48"] += 1
        elif tenure <= 72:
            tenure_counter["49-72"] += 1
        else:
            tenure_counter["72+"] += 1

    return {
        "cards": {
            "total_customers": total_customers or dataset_summary.get("total_customers", 0),
            "total_predictions": total_predictions,
            "high_risk_customers": high_risk_customers,
            "avg_monthly_charges": avg_monthly_charges,
            "churn_rate": predicted_churn_rate,
        },
        "charts": {
            "monthly_predictions": dict(sorted(date_counter.items())) or {"No predictions": 0},
            "churn_trend": dict(sorted(churn_counter.items())) or {"No churn alerts": 0},
            "contract_distribution": dict(contract_counter) or dataset_summary.get("contract_distribution", {}),
            "gender_distribution": dict(gender_counter) or dataset_summary.get("gender_distribution", {}),
            "payment_method_distribution": dict(payment_counter) or dataset_summary.get("payment_method_distribution", {}),
            "internet_service_distribution": dict(internet_counter) or dataset_summary.get("internet_service_distribution", {}),
            "monthly_charge_distribution": dict(charge_counter) or dataset_summary.get("monthly_charge_bins", {}),
            "tenure_distribution": dict(tenure_counter) or dataset_summary.get("tenure_bins", {}),
        },
    }


def create_pdf_report(row: sqlite3.Row) -> Path:
    report_path = REPORTS_DIR / f"prediction_{row['id']}.pdf"
    input_data = json.loads(row["input_json"])
    recommendations = json.loads(row["recommendations_json"])
    explanation = json.loads(row["explanation_json"])

    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 110, width, 110, fill=True, stroke=False)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(40, height - 55, "Customer Churn Intelligence Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 80, f"Generated on {row['created_at']}")

    y = height - 140
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Prediction Summary")
    y -= 22

    summary_lines = [
        f"Customer: {row['customer_name']}",
        f"Prediction: {row['prediction_label']}",
        f"Churn Probability: {round(float(row['churn_probability']) * 100, 2)}%",
        f"Risk Level: {row['risk_level']}",
    ]
    pdf.setFont("Helvetica", 11)
    for line in summary_lines:
        pdf.drawString(52, y, line)
        y -= 18

    y -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Customer Inputs")
    y -= 22
    pdf.setFont("Helvetica", 10)
    for key, value in input_data.items():
        pdf.drawString(52, y, f"{key}: {value}")
        y -= 15
        if y < 120:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    y -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Key Drivers")
    y -= 22
    pdf.setFont("Helvetica", 10)
    for driver in explanation.get("positive_drivers", []):
        pdf.drawString(52, y, f"+ {driver['feature']} ({driver['direction']})")
        y -= 15
    for driver in explanation.get("negative_drivers", []):
        pdf.drawString(52, y, f"- {driver['feature']} ({driver['direction']})")
        y -= 15

    y -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Recommendations")
    y -= 22
    pdf.setFont("Helvetica", 10)
    for recommendation in recommendations:
        pdf.drawString(52, y, f"- {recommendation}")
        y -= 15

    pdf.save()
    return report_path


def build_history_payload(user_id: int, query: str = "") -> dict[str, Any]:
    params: list[Any] = [user_id]
    sql = """
        SELECT *
        FROM predictions
        WHERE user_id = ?
    """
    if query:
        sql += " AND LOWER(customer_name) LIKE ?"
        params.append(f"%{query.lower()}%")
    sql += " ORDER BY created_at DESC"
    rows = fetch_all(sql, tuple(params))
    return {"items": [serialize_prediction_row(row) for row in rows]}


def handle_prediction_request() -> tuple[dict[str, Any], int]:
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = prediction_service.predict(payload)
    except ModelNotReadyError as exc:
        return {"error": str(exc)}, 400

    prediction_id = save_prediction(session["user_id"], payload, result)
    result["prediction_id"] = prediction_id
    result["report_url"] = url_for("download_report", prediction_id=prediction_id)
    return result, 200


@app.route("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    dataset_summary = load_dataset_summary()
    model_summary = load_model_summary()
    landing_stats = {
        "customers": dataset_summary.get("total_customers", 7043),
        "churn_rate": dataset_summary.get("churn_rate", 26.54),
        "avg_monthly_charges": dataset_summary.get("avg_monthly_charges", 64.76),
        "best_model": model_summary.get("best_model", "ML Pipeline Ready"),
        "roc_auc": model_summary.get("roc_auc", 0.8463),
        "accuracy": model_summary.get("accuracy", 80.0),
    }
    return render_template("landing.html", landing_stats=landing_stats)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not full_name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if fetch_one("SELECT id FROM users WHERE email = ?", (email,)):
            flash("An account with that email already exists.", "warning")
            return render_template("signup.html")

        get_db().execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name, email, generate_password_hash(password), datetime.utcnow().isoformat(timespec="seconds")),
        )
        get_db().commit()
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = fetch_one("SELECT * FROM users WHERE email = ?", (email,))

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        flash("Welcome back.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if wants_json_response():
        return jsonify(build_dashboard_payload(session["user_id"]))
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_page():
    if request.method == "POST":
        payload, status_code = handle_prediction_request()
        return jsonify(payload), status_code
    return render_template("predict.html", form_fields=FORM_FIELDS)


@app.route("/history")
@login_required
def history_page():
    if wants_json_response():
        query = request.args.get("q", "").strip()
        return jsonify(build_history_payload(session["user_id"], query))
    return render_template("history.html")


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload, status_code = handle_prediction_request()
    return jsonify(payload), status_code


@app.route("/api/history")
@login_required
def api_history():
    query = request.args.get("q", "").strip()
    return jsonify(build_history_payload(session["user_id"], query))


@app.route("/customers")
@app.route("/api/customers")
@login_required
def api_customers():
    rows = fetch_all(
        """
        SELECT customer_name, prediction_label, churn_probability, risk_level, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],),
    )
    customers = [
        {
            "customer_name": row["customer_name"],
            "prediction": row["prediction_label"],
            "probability": round(float(row["churn_probability"]) * 100, 2),
            "risk_level": row["risk_level"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return jsonify({"items": customers})


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    payload = build_dashboard_payload(session["user_id"])
    return jsonify(payload)


@app.route("/prediction", methods=["DELETE"])
@app.route("/api/prediction/<int:prediction_id>", methods=["DELETE"])
@login_required
def delete_prediction(prediction_id: int | None = None):
    if prediction_id is None:
        request_payload = request.get_json(silent=True) or request.args.to_dict()
        prediction_id = int(request_payload.get("id", 0))
    if not prediction_id:
        return jsonify({"error": "Prediction id is required."}), 400

    db = get_db()
    db.execute(
        "DELETE FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, session["user_id"]),
    )
    db.commit()
    return jsonify({"message": "Prediction deleted successfully."})


@app.route("/history/export")
@login_required
def export_history():
    rows = fetch_all(
        """
        SELECT customer_name, prediction_label, churn_probability, risk_level, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],),
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Customer Name", "Prediction", "Probability (%)", "Risk Level", "Timestamp"])
    for row in rows:
        writer.writerow(
            [
                row["customer_name"],
                row["prediction_label"],
                round(float(row["churn_probability"]) * 100, 2),
                row["risk_level"],
                row["created_at"],
            ]
        )

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


@app.route("/prediction/<int:prediction_id>/report")
@login_required
def download_report(prediction_id: int):
    row = fetch_one(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, session["user_id"]),
    )
    if row is None:
        flash("Prediction report not found.", "error")
        return redirect(url_for("history_page"))

    report_path = create_pdf_report(row)
    return send_file(report_path, as_attachment=True, download_name=report_path.name)


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


ensure_directories()
with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(debug=True)
