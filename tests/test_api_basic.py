import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.deployment_step_06_api_server_skeleton import app

client = TestClient(app)


def valid_payload():
    return {
        "tenure_months": 12,
        "monthly_charges": 85.5,
        "total_charges_clean": 1026.0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "payment_method": "Electronic check",
        "paperless_billing": 1,
        "partner": 0,
        "dependents": 0,
        "senior_citizen": 0
    }


def test_health_check():
    response = client.get('/health')
    print("\n[test_health_check] status:", response.status_code)
    print("[test_health_check] body:", response.json())
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_valid_payload():
    payload = valid_payload()
    response = client.post('/predict', json=payload)
    body = response.json()
    print("\n[test_predict_valid_payload] payload:", payload)
    print("[test_predict_valid_payload] status:", response.status_code)
    print("[test_predict_valid_payload] body:", body)
    assert response.status_code == 200
    assert 'model_version' in body
    assert 'threshold_used' in body
    assert 'churn_probability' in body
    assert 'predicted_label' in body


def test_predict_missing_fields():
    payload = {
        "tenure_months": 12,
        "monthly_charges": 85.5,
        "contract": "Month-to-month"
    }
    response = client.post('/predict', json=payload)
    print("\n[test_predict_missing_fields] payload:", payload)
    print("[test_predict_missing_fields] status:", response.status_code)
    print("[test_predict_missing_fields] body:", response.text)
    assert response.status_code == 422


def test_predict_wrong_types():
    payload = valid_payload()
    payload['tenure_months'] = 'twelve'
    payload['monthly_charges'] = 'eighty five'
    response = client.post('/predict', json=payload)
    print("\n[test_predict_wrong_types] payload:", payload)
    print("[test_predict_wrong_types] status:", response.status_code)
    print("[test_predict_wrong_types] body:", response.text)
    assert response.status_code == 422


def test_predict_extra_field_behavior():
    payload = valid_payload()
    payload['unexpected_field'] = 'extra'
    response = client.post('/predict', json=payload)
    print("\n[test_predict_extra_field_behavior] payload:", payload)
    print("[test_predict_extra_field_behavior] status:", response.status_code)
    print("[test_predict_extra_field_behavior] body:", response.text)
    assert response.status_code in [200, 422]


if __name__ == '__main__':
    print('=== RUNNING BASIC API TESTS MANUALLY ===')
    test_health_check()
    test_predict_valid_payload()
    test_predict_missing_fields()
    test_predict_wrong_types()
    test_predict_extra_field_behavior()
    print("\nAll manual test functions executed.")
