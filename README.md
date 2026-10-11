# Telco Customer Churn Prediction – XGBoost + MLflow + FastAPI + Docker

An end-to-end Machine Learning project that predicts whether a telecom customer will leave (churn).
It trains and compares Logistic Regression, Random Forest, and XGBoost models, tracks every experiment
in **MLflow**, serves the best model as a **FastAPI** REST API, and packages it with **Docker**.

Everything runs locally on a laptop for free. No cloud account or paid tool is needed.

## What is this project for? (in simple words)

**The business problem:** A telecom company (like Jio or Airtel) loses money every time a customer cancels
their plan and moves to a competitor. This is called **churn**. Keeping an existing customer is much cheaper
than finding a new one, so the company wants to know **in advance** which customers are likely to leave,
so it can offer them a discount or better support before they go.

**What we built:** A system that looks at a customer's details (how long they have been a customer,
monthly bill, contract type, internet service, payment method, etc.) and predicts the **probability that
the customer will leave**, for example "91% chance of churn – High risk".

**What happens in each part:**

| Part | What it does | Real-world equivalent |
| --- | --- | --- |
| `download_data.py` | Gets 7,043 real customer records | Collecting data from the company database |
| `train.py` (scikit-learn + XGBoost) | Cleans the data and teaches 5 different models to recognise customers who leave; picks the best one | Data scientist building the model |
| MLflow | Records every model trained: its settings, scores, and charts, and stores the best model in a "model registry" | Lab notebook + version control for models |
| FastAPI (`app/main.py`) | Turns the model into a web service: send customer details, get a prediction back | How a company's app or website would use the model |
| Docker | Packs the API and everything it needs into one container that runs the same on any computer or server | How companies ship ML models to production |
| `pytest` | Automatically checks the API gives correct answers | Quality testing before release |

This full flow (data → model → experiment tracking → API → container) is what companies call **MLOps**,
and it is exactly what entry-level ML Engineer roles ask for.

## Dataset

The dataset is **real and public**; it was not created by us.

- **Name:** Telco Customer Churn (published by IBM as a sample dataset)
- **Downloaded from:** IBM's official GitHub repository –
  https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
  (`src/download_data.py` downloads it automatically)
- **Same dataset on Kaggle:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size:** 7,043 customers, 21 columns. The target column `Churn` is "Yes" for about 26.5% of customers.
- **Main columns:** gender, SeniorCitizen, Partner, Dependents, tenure (months), PhoneService,
  InternetService, OnlineSecurity, TechSupport, StreamingTV, Contract, PaperlessBilling, PaymentMethod,
  MonthlyCharges, TotalCharges, Churn.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.846 | 0.841 | 0.614 |
| Random Forest | 0.848 | 0.842 | 0.628 |
| **XGBoost (best)** | **0.848** | **0.846** | **0.632** |

Best model recall: 0.80 (catches 80% of customers who actually churn).
Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) – 7,043 customers, 19 features.

## Tech Stack

Python, Pandas, scikit-learn, XGBoost, MLflow (experiment tracking + model registry), FastAPI, Pydantic,
Docker, Pytest

## Project Structure

```text
churn-prediction/
  src/download_data.py     # downloads the dataset
  src/train.py             # preprocessing, training, MLflow tracking, model export
  app/main.py              # FastAPI prediction service
  tests/test_api.py        # API tests
  models/                  # exported best model + metrics (created by train.py)
  Dockerfile               # container for the API
  requirements.txt         # local development packages
  requirements-api.txt     # minimal packages for the Docker image
```

## Step-by-step: run it on Windows

### Step 1 – Install the free tools (one time)

1. **Python 3.13** – https://www.python.org/downloads/ (tick "Add Python to PATH" during install).
2. **VS Code or Cursor** – any code editor.
3. **Git** (optional, for GitHub) – https://git-scm.com/download/win
4. **Docker Desktop** – https://www.docker.com/products/docker-desktop/ (free for personal use and education).
   - During install keep **"Use WSL 2"** ticked, then restart the laptop.
   - Open Docker Desktop once and wait until it says **"Engine running"**.
   - Check it works: open PowerShell and run `docker run hello-world`.

### Step 2 – Create a virtual environment and install packages

Open PowerShell in the `churn-prediction` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Step 3 – Download the dataset

```powershell
python src\download_data.py
```

### Step 4 – Train the models (tracked in MLflow)

```powershell
python src\train.py
```

This trains 5 models, logs parameters, metrics, plots, and models to MLflow, registers the best one as
`telco-churn-model` with the alias `champion`, and saves it to `models/model.joblib`.

### Step 5 – Explore the experiments in the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://127.0.0.1:5000 → **Model training** → **telco-churn-prediction** → **Training runs**.
Select runs and click **Compare** to see metrics side by side; open a run to see the confusion matrix
and feature importance plots. Open **Models** to see the registered model. Press `Ctrl+C` to stop.

### Step 6 – Run the API locally and test it

```powershell
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs → **POST /predict** → **Try it out** → **Execute**.
Example response:

```json
{"churn_probability": 0.9152, "prediction": "Churn", "risk_level": "High"}
```

Run the automated tests (in a second terminal with the venv activated):

```powershell
pytest -q
```

### Step 7 – Build and run with Docker

Make sure Docker Desktop is running, then:

```powershell
docker build -t churn-api .
docker run -d -p 8000:8000 --name churn-api churn-api
```

Open http://127.0.0.1:8000/docs – the same API, now running inside a container.

Useful commands:

```powershell
docker ps                 # list running containers
docker logs churn-api     # see API logs
docker stop churn-api     # stop the container
docker rm churn-api       # remove it
docker images             # list images
```

### Step 8 (optional) – Share the image on Docker Hub (free account)

```powershell
docker login
docker tag churn-api <your-dockerhub-username>/churn-api:1.0
docker push <your-dockerhub-username>/churn-api:1.0
```

### Step 9 – Push to GitHub

```powershell
git init
git add .
git commit -m "Churn prediction with XGBoost, MLflow, FastAPI and Docker"
git branch -M main
git remote add origin https://github.com/<your-username>/churn-prediction.git
git push -u origin main
```

`.gitignore` already excludes the virtual environment, dataset, and MLflow database.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `address already in use` on port 8000 or 5000 | Another server is still running; press `Ctrl+C` in its terminal or use `--port 8001` |
| `docker: command not found` | Install Docker Desktop and open it until it shows "Engine running" |
| Browser shows "This site can't be reached" | The server terminal must stay open while you use the page |
| Laptop freezes | The API and MLflow use very little memory; close extra browser tabs, update graphics drivers, and run only one server per terminal |

## Sample request (curl)

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"gender\":\"Female\",\"SeniorCitizen\":0,\"Partner\":\"No\",\"Dependents\":\"No\",\"tenure\":2,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"Fiber optic\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"No\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"Yes\",\"StreamingMovies\":\"Yes\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":95.5,\"TotalCharges\":191.0}"
```
