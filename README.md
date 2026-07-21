# ChurnNet: Customer Churn Prediction System

ChurnNet is a production-style customer churn prediction platform built with Flask, scikit-learn, SQLite/PostgreSQL, Bootstrap, Chart.js, and explainable ML outputs. It predicts churn risk for telecom customers, highlights the main drivers behind each result, stores prediction history, generates PDF reports, and exposes REST endpoints for programmatic access.

## Demo Gallery

The visuals below are repo-local illustrative mockups aligned to the current templates, so the README stays demo-friendly even without a live deployment.

![ChurnNet demo overview](docs/assets/demo-overview.svg)

![ChurnNet operations demo](docs/assets/demo-operations.svg)

## Pictorial Flowchart

![ChurnNet pictorial flowchart](docs/assets/project-flowchart.svg)

## Highlights

- Authentication with signup, login, logout, and session-based access control
- Churn prediction form with probability score, risk level, business recommendations, and downloadable PDF output
- Explainability layer with feature-driver summaries for positive and negative churn signals
- Analytics dashboard with KPI cards and multiple Chart.js visualizations
- Prediction history with search, delete, CSV export, and report download support
- ML training pipeline with preprocessing, feature engineering, model comparison, evaluation, and best-model selection
- SQLite for local development and PostgreSQL support through `DATABASE_URL`
- Render-ready deployment files: `Procfile`, `runtime.txt`, and `requirements.txt`

## Tech Stack

### Backend

- Python
- Flask
- SQLite
- PostgreSQL
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- ReportLab
- Joblib

### Optional ML Extras

- SHAP
- XGBoost
- LightGBM
- CatBoost

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Chart.js
- SweetAlert2

## How It Works

1. Load the Telco Customer Churn dataset from `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. Clean and normalize the data in `preprocess.py`.
3. Engineer features such as `avg_monthly_spend`, `tenure_group`, `service_count`, `premium_customer`, and `high_risk_customer`.
4. Train and compare multiple models in `train.py`.
5. Save the best model plus encoder, scaler, and metadata under `model/`.
6. Serve live predictions through the Flask app in `app.py`.
7. Turn predictions into dashboards, history records, PDF reports, and API responses.

## Project Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main Flask application with routes, auth flow, dashboard APIs, prediction handling, and report generation |
| `feature_engineering.py` | Feature creation helpers used by the ML pipeline |
| `predict.py` | Prediction service that loads saved model artifacts and returns churn outputs |
| `preprocess.py` | Dataset loading, cleaning, normalization, and form field definitions |
| `train.py` | Model training, comparison, evaluation, and artifact export pipeline |
| `setup_windows.ps1` | Windows setup script for creating the environment and installing dependencies |
| `requirements.txt` | Core Python dependencies |
| `requirements-optional.txt` | Optional ML and explainability dependencies |
| `runtime.txt` | Runtime version hint for deployment |
| `Procfile` | Process definition for deployment platforms such as Render |
| `README.md` | Project documentation |
| `dataset/` | Dataset storage location for the Telco Customer Churn CSV |
| `docs/assets/` | README demo visuals and pictorial flowchart assets |
| `instance/` | Local SQLite database and instance-specific runtime files |
| `model/` | Saved model, encoder, scaler, metrics, plots, and metadata artifacts |
| `reports/` | Generated PDF churn reports |
| `static/css/` | Stylesheets for the frontend |
| `static/images/` | EDA charts and image assets |
| `static/js/` | Frontend JavaScript for dashboard, history, app behavior, and prediction flow |
| `templates/` | Jinja HTML templates for landing, auth, dashboard, history, and prediction pages |

## Dataset

This project expects the Kaggle Telco Customer Churn dataset:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place it here before training:

```text
dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is not bundled in this repository.

## ML Pipeline

The training workflow in [train.py](/C:/Users/My Pc/Desktop/ML project/train.py) performs:

- Missing-value handling
- Duplicate removal
- Column normalization
- Numeric conversion for `total_charges`, `monthly_charges`, `tenure`, and `senior_citizen`
- Feature engineering for spend, tenure, service usage, and risk segmentation
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Model comparison across Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and SVM
- Optional training with XGBoost, LightGBM, and CatBoost when extras are installed
- Best-model selection using ROC AUC

Saved artifacts:

- `model/model.pkl`
- `model/scaler.pkl`
- `model/encoder.pkl`
- `model/explanation_assets.pkl`
- `model/metadata.json`
- `model/metrics.json`
- `model/model_comparison.csv`
- `model/model_comparison.png`
- `model/confusion_matrix.png`
- `model/roc_curve.png`

EDA images are automatically saved under `static/images/eda/`.

## Setup

### Recommended on Windows

```powershell
.\setup_windows.ps1
.\.venv313\Scripts\Activate.ps1
```

That script:

- creates `.venv313` if it does not already exist
- prefers the local Python 3.13 install when available
- upgrades `pip`, `setuptools`, and `wheel`
- installs the core dependencies from `requirements.txt`

Optional ML extras:

```powershell
.\.venv313\Scripts\python.exe -m pip install --only-binary=:all: --default-timeout 120 --retries 10 -r requirements-optional.txt
```

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Conda fallback

```powershell
conda env create -f environment.yml
conda activate churnnet311
```

## Training

1. Add the dataset to `dataset/`.
2. Activate your environment.
3. Run:

```bash
python train.py
```

This will:

- train multiple models
- select the best performer
- save evaluation plots under `model/`
- save EDA images under `static/images/eda/`

If optional dependencies are not installed, the app still works with the core scikit-learn models.

## Run The App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Database Modes

ChurnNet supports two database modes:

- Local development with SQLite at `instance/churn_app.db`
- Deployment with PostgreSQL through `DATABASE_URL`

If `DATABASE_URL` is set, the app automatically uses PostgreSQL and creates the required tables at startup. Otherwise it falls back to SQLite.

## Product Features

### Authentication

- Signup
- Login
- Logout
- Session management

### Prediction Workspace

- Customer profile form
- Churn prediction label
- Probability score
- Risk level
- Explanation summary
- Positive and negative driver cards
- Actionable retention recommendations
- PDF report download

### Dashboard

- Total customers
- Total predictions
- High risk customers
- Average monthly charges
- Churn rate
- Monthly predictions
- Churn trend
- Contract distribution
- Gender distribution
- Payment method distribution
- Internet service distribution
- Monthly charge distribution
- Tenure distribution

### History

- Search by customer name
- Delete saved predictions
- Export CSV
- Download PDF reports

## REST API

Primary API routes:

- `POST /api/predict`
- `GET /api/history`
- `GET /api/dashboard`
- `GET /api/customers`
- `DELETE /api/prediction/<id>`

Friendly aliases:

- `POST /predict`
- `GET /history?format=json`
- `GET /dashboard?format=json`
- `GET /customers`
- `DELETE /prediction?id=<prediction_id>`

### Example: Predict

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"customer_name\":\"John Doe\",\"gender\":\"Male\",\"senior_citizen\":0,\"partner\":\"Yes\",\"dependents\":\"No\",\"tenure\":8,\"phone_service\":\"Yes\",\"multiple_lines\":\"No\",\"internet_service\":\"Fiber optic\",\"online_security\":\"No\",\"online_backup\":\"No\",\"device_protection\":\"No\",\"tech_support\":\"No\",\"streaming_tv\":\"Yes\",\"streaming_movies\":\"Yes\",\"contract\":\"Month-to-month\",\"paperless_billing\":\"Yes\",\"payment_method\":\"Electronic check\",\"monthly_charges\":89.5,\"total_charges\":716.0}"
```

## Reports

Each saved prediction can be exported as a PDF containing:

- Customer information
- Prediction
- Probability
- Risk level
- Feature drivers
- Recommendations
- Timestamp

Reports are generated into `reports/`.

## Deployment On Render

1. Push the project to GitHub.
2. Create a new Render web service.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `gunicorn app:app`.
5. Add environment variables:

- `SECRET_KEY`
- `DATABASE_URL`

For free deployment, use an external Postgres provider such as Neon or Supabase and paste the connection string into `DATABASE_URL`.

Example:

```text
postgresql://username:password@host/database?sslmode=require
```

6. Set `PYTHON_VERSION` to a supported `3.11.x` version in Render.
7. Make sure model artifacts are created before deployment, or run `python train.py` during your release workflow.
8. Because free Render web services do not support persistent disks, PostgreSQL is the recommended way to persist signup users and prediction history.

## Future Scope

- Bulk CSV prediction
- Admin dashboard
- Role-based authentication
- Email alerts for high-risk customers
- Customer segmentation
- LIME explanations
- Docker support
- GitHub Actions CI/CD
- Retraining workflow
- Power BI integration

## License

Use this project for learning, portfolio, and interview preparation. If you plan to publish it publicly, add your preferred open-source license before release.
