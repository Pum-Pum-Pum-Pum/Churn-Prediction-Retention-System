import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"


def evaluate_thresholds(y_true, y_prob, thresholds):
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_FILE)

    target_col = "churn_value"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    models = {
        "logreg_default": LogisticRegression(max_iter=1000, random_state=42),
        "logreg_balanced": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    }

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    print("=== MODELING STEP 2: THRESHOLD + CLASS WEIGHT EXPERIMENT PLAN ===")
    print("\nIMPORTANT: This step fits on training data and inspects behavior on the held-out test set for learning purposes.")
    print("In a stricter production workflow, threshold selection should ideally be done on validation/CV, not final test.")

    for model_name, classifier in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob >= 0.50).astype(int)

        print(f"\n\n=== MODEL: {model_name} ===")
        print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
        print("Default threshold (0.50) precision:", round(precision_score(y_test, y_pred_default, zero_division=0), 4))
        print("Default threshold (0.50) recall:", round(recall_score(y_test, y_pred_default, zero_division=0), 4))
        print("Default threshold (0.50) f1:", round(f1_score(y_test, y_pred_default, zero_division=0), 4))
        print("Confusion matrix at 0.50:")
        print(confusion_matrix(y_test, y_pred_default))

        threshold_df = evaluate_thresholds(y_test, y_prob, thresholds)
        print("\nThreshold comparison:")
        print(threshold_df)

if __name__ == "__main__":
    main()
