from pathlib import Path
import json
import joblib
from uuid import uuid4
from datetime import datetime, UTC
import hashlib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_pipeline.joblib'
METADATA_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_metadata.json'
LOG_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'inference_log.jsonl'

app = FastAPI(title='Churn Prediction API', version='0.1.0')


class ChurnRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tenure_months: int
    monthly_charges: float
    total_charges_clean: float
    contract: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    payment_method: str
    paperless_billing: int
    partner: int
    dependents: int
    senior_citizen: int


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return model, metadata


def hash_payload(payload: dict) -> str:
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_json.encode('utf-8')).hexdigest()


def log_inference(entry: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


@app.get('/health')
def health_check():
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: ChurnRequest):
    try:
        model, metadata = load_artifacts()
        payload_dict = payload.model_dump()
        input_df = pd.DataFrame([payload_dict])

        churn_probability = float(model.predict_proba(input_df)[0, 1])
        threshold = float(metadata['threshold'])
        predicted_label = int(churn_probability >= threshold)

        response = {
            'request_id': str(uuid4()),
            'timestamp_utc': datetime.now(UTC).isoformat(),
            'model_version': metadata['model_version'],
            'threshold_used': threshold,
            'churn_probability': round(churn_probability, 6),
            'predicted_label': predicted_label
        }

        log_entry = response.copy()
        log_entry['input_feature_hash'] = hash_payload(payload_dict)
        log_inference(log_entry)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    print('Run with: uvicorn src.deployment.deployment_step_06_api_server_skeleton:app --reload')
