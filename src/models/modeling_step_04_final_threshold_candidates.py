import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_positive_count": int(y_pred.sum())
    }, y_pred


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

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    candidate_thresholds = {
        "high_recall_candidate": 0.40,
        "balanced_candidate": 0.60,
        "capacity_limited_candidate": 0.70
    }

    print("=== MODELING STEP 4: FINAL THRESHOLD CANDIDATE EVALUATION ===")
    print("\nModel: LogisticRegression(class_weight='balanced')")
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    for label, threshold in candidate_thresholds.items():
        metrics_dict, y_pred = evaluate_at_threshold(y_test, y_prob, threshold)
        print(f"\n\n=== {label.upper()} | THRESHOLD = {threshold} ===")
        print(metrics_dict)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\n=== RECOMMENDED INTERPRETATION ===")
    print("- 0.40: best when recall is prioritized and business accepts broader outreach.")
    print("- 0.60: best when a balanced operating point is needed.")
    print("- 0.70: best when outreach capacity is tight and precision matters more.")
    print("Choose threshold based on business cost, not just one metric.")


if __name__ == "__main__":
    main()
