
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'baseline_model_input.csv'
MODEL_DIR = PROJECT_ROOT / 'artifacts' / 'models'
REPORTS_DIR = PROJECT_ROOT / 'artifacts' / 'reports'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / 'logistic_baseline_pipeline.joblib'
METADATA_PATH = MODEL_DIR / 'logistic_baseline_metadata.json'

FINAL_THRESHOLD = 0.60
MODEL_VERSION = 'logistic_baseline_v1'


def main():
    df = pd.read_csv(INPUT_FILE)
    target_col = 'churn_value'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= FINAL_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metadata = {
        'model_version': MODEL_VERSION,
        'model_family': 'logistic_regression',
        'artifact_path': str(MODEL_PATH),
        'input_file': str(INPUT_FILE),
        'threshold': FINAL_THRESHOLD,
        'target_column': target_col,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'metrics': {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        },
        'notes': 'Official baseline deployment artifact. Threshold chosen from business-cost operating policy.'
    }

    joblib.dump(model, MODEL_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('=== DEPLOYMENT / INFERENCE DESIGN STEP 2: MODEL ARTIFACT SAVING ===')
    print('Saved pipeline artifact to:', MODEL_PATH)
    print('Saved metadata to:', METADATA_PATH)
    print('Metadata preview:')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
