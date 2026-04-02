
    from pathlib import Path
    import json
    import joblib
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_pipeline.joblib'
    METADATA_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_metadata.json'
    SCHEMA_PATH = PROJECT_ROOT / 'artifacts' / 'schemas' / 'inference_api_contract.json'


    def load_artifacts():
        model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return model, metadata, schema


    def validate_input(payload: dict, schema: dict):
        required_fields = set(schema['input_schema'].keys())
        incoming_fields = set(payload.keys())

        missing = required_fields - incoming_fields
        extra = incoming_fields - required_fields

        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected fields: {sorted(extra)}")


    def predict_one(payload: dict):
        model, metadata, schema = load_artifacts()
        validate_input(payload, schema)

        input_df = pd.DataFrame([payload])
        churn_probability = float(model.predict_proba(input_df)[0, 1])
        threshold = float(metadata['threshold'])
        predicted_label = int(churn_probability >= threshold)

        return {
            'model_version': metadata['model_version'],
            'threshold_used': threshold,
            'churn_probability': round(churn_probability, 6),
            'predicted_label': predicted_label
        }


    def main():
        sample_payload = {
            'tenure_months': 12,
            'monthly_charges': 85.5,
            'total_charges_clean': 1026.0,
            'contract': 'Month-to-month',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'online_backup': 'Yes',
            'device_protection': 'No',
            'tech_support': 'No',
            'payment_method': 'Electronic check',
            'paperless_billing': 1,
            'partner': 0,
            'dependents': 0,
            'senior_citizen': 0
        }

        result = predict_one(sample_payload)
        print('=== DEPLOYMENT / INFERENCE DESIGN STEP 3: INFERENCE WRAPPER TEST ===')
        print('
Sample payload:')
        print(sample_payload)
        print('
Prediction result:')
        print(result)


    if __name__ == '__main__':
        main()
