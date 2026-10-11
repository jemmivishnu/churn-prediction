from fastapi.testclient import TestClient

from app.main import Customer, app

client = TestClient(app)
EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_high_risk_customer():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] == "Churn"


def test_predict_loyal_customer():
    loyal = {**EXAMPLE, "tenure": 70, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
             "OnlineSecurity": "Yes", "TechSupport": "Yes", "MonthlyCharges": 60.0, "TotalCharges": 4200.0}
    response = client.post("/predict", json=loyal)
    assert response.status_code == 200
    assert response.json()["prediction"] == "No Churn"


def test_invalid_input_rejected():
    response = client.post("/predict", json={**EXAMPLE, "Contract": "Ten years"})
    assert response.status_code == 422
