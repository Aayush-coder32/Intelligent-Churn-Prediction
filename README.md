# ChurnIQ: Customer Churn Prediction System

ChurnIQ is a production-style customer churn prediction platform built with Flask, scikit-learn, SQLite, Bootstrap, Chart.js, and SHAP-style explanations. It predicts churn risk for telecom customers, explains the main drivers behind each prediction, stores prediction history, generates PDF reports, and exposes REST endpoints for programmatic access.

## Highlights

- Authentication with signup, login, logout, and session-based access control
- Churn prediction form with probability score, risk level, and business recommendations
- Explainability layer with SHAP-first contributions and model-aware fallbacks
- Analytics dashboard with KPI cards and multiple Chart.js visualizations
- Prediction history with search, delete, CSV export, and downloadable PDF reports
- ML training pipeline with feature engineering, model comparison, evaluation, and auto-selection of the best model
- EDA image generation saved automatically to `static/images/eda/`
- Render-ready deployment files: `Procfile`, `runtime.txt`, `requirements.txt`

## Tech Stack

### Backend

- Python
- Flask
- SQLite
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Joblib
- ReportLab

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Chart.js
- SweetAlert2

## Project Structure

```text
ML project/
├── app.py
├── feature_engineering.py
├── preprocess.py
├── predict.py
├── train.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── dataset/
├── model/
├── reports/
├── static/
│   ├── css/
│   ├── images/
│   │   └── eda/
│   └── js/
└── templates/
```

## Dataset

This project expects the Kaggle dataset:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place the file here before training:

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
- Feature engineering such as `avg_monthly_spend`, `tenure_group`, `service_count`, `premium_customer`, and `high_risk_customer`
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Model comparison across:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - SVM
  - XGBoost if installed
  - LightGBM if installed
  - CatBoost if installed
- Best-model selection using ROC AUC

Saved artifacts:

- `model/model.pkl`
- `model/scaler.pkl`
- `model/encoder.pkl`
- `model/explanation_assets.pkl`
- `model/metadata.json`
- `model/metrics.json`
- `model/model_comparison.csv`

## Training

1. Install dependencies.

Recommended on this Windows machine:

```powershell
.\setup_windows.ps1
.\.venv313\Scripts\Activate.ps1
```

That script:

- creates a fresh `.venv313`
- uses the local 64-bit Python 3.13 install
- upgrades `pip`, `setuptools`, and `wheel`
- installs only wheel packages with a longer network timeout
- installs the smaller core dependency set from `requirements.txt`

Optional model/explainability extras:

```powershell
.\.venv313\Scripts\python.exe -m pip install --only-binary=:all: --default-timeout 120 --retries 10 -r requirements-optional.txt
```

Alternative manual setup:

```powershell
python -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel --default-timeout 120 --retries 10
pip install --only-binary=:all: --default-timeout 120 --retries 10 -r requirements.txt
```

Conda fallback:

```powershell
conda env create -f environment.yml
conda activate churniq311
```

2. Add the dataset to `dataset/`.

3. Train the model:

```bash
python train.py
```

This will:

- train multiple models
- select the best performer
- save evaluation plots under `model/`
- save EDA images under `static/images/eda/`

If `requirements-optional.txt` is not installed, training still works with:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM

The app already treats SHAP, XGBoost, LightGBM, and CatBoost as optional and falls back gracefully when they are unavailable.

## Run The App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

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
- Positive and negative feature drivers
- Actionable retention recommendations

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

Friendly aliases matching the brief:

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
2. Create a new Render Web Service.
3. Set the build command:

```bash
pip install -r requirements.txt
```

4. Set the start command:

```bash
gunicorn app:app
```

5. Add environment variables:

- `SECRET_KEY`

6. Make sure model artifacts are created before deployment, or add the dataset and run `python train.py` during your release workflow.

## Suggested Demo Assets

After training and running the app locally, add screenshots for:

- Login page
- Dashboard
- Prediction form
- Prediction result panel
- Prediction history table
- PDF report preview

## Future Scope

- Bulk CSV prediction
- Admin dashboard
- Role-based authentication
- Email alerts for high-risk customers
- K-Means customer segmentation
- LIME explanations
- Docker support
- GitHub Actions CI/CD
- Retraining module
- Power BI integration

## License

Use this project for learning, portfolio, and interview preparation. If you plan to publish it publicly, add your preferred open-source license file before release.
