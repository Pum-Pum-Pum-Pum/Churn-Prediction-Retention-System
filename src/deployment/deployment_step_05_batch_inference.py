
from pathlib import Path
import json
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_pipeline.joblib'
METADATA_PATH = PROJECT_ROOT / 'artifacts' / 'models' / 'logistic_baseline_metadata.json'
SCHEMA_PATH = PROJECT_ROOT / 'artifacts' / 'schemas' / 'inference_api_contract.json'
INPUT_BATCH_FILE = PROJECT_ROOT / 'data' / 'processed' / 'baseline_model_input.csv'
OUTPUT_BATCH_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'batch_predictions_output.csv'


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return model, metadata, schema


def validate_batch_columns(df: pd.DataFrame, schema: dict):
    required_fields = list(schema['input_schema'].keys())
    missing = [col for col in required_fields if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return required_fields


def main():
    model, metadata, schema = load_artifacts()

    df = pd.read_csv(INPUT_BATCH_FILE)
    required_fields = validate_batch_columns(df, schema)
    input_df = df[required_fields].copy()

    churn_probability = model.predict_proba(input_df)[:, 1]
    threshold = float(metadata['threshold'])
    predicted_label = (churn_probability >= threshold).astype(int)

    output_df = input_df.copy()
    output_df['churn_probability'] = churn_probability.round(6)
    output_df['predicted_label'] = predicted_label
    output_df['threshold_used'] = threshold
    output_df['model_version'] = metadata['model_version']

    OUTPUT_BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_BATCH_FILE, index=False)

    print('=== DEPLOYMENT / INFERENCE DESIGN STEP 5: BATCH INFERENCE ===')
    print('Input batch file:', INPUT_BATCH_FILE)
    print('Output predictions file:', OUTPUT_BATCH_FILE)
    print('Batch shape:', input_df.shape)
    print('Predicted positives:', int(output_df['predicted_label'].sum()))
    print('Predicted positive pct:', round(output_df['predicted_label'].mean() * 100, 2))
    print('Preview:')
    print(output_df.head())


if __name__ == '__main__':
    main()
