import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"


def evaluate_thresholds(y_true, y_prob, thresholds):
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "threshold": threshold,
            "predicted_positive_count": int(y_pred.sum()),
            "predicted_positive_pct": round(y_pred.mean() * 100, 2),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
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
        ("imputer", SimpleImputer(strategy="median"))
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
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_df = evaluate_thresholds(y_test, y_prob, thresholds)

    print("=== CHALLENGER MODEL STEP 2: RANDOM FOREST THRESHOLD REVIEW ===")
    print("\nModel: RandomForestClassifier(class_weight='balanced_subsample')")
    print("ROC-AUC on held-out test probabilities:", round(roc_auc_score(y_test, y_prob), 4))

    print("\n=== THRESHOLD EVALUATION TABLE ===")
    print(threshold_df)

    high_recall_candidates = threshold_df[threshold_df["recall"] >= 0.80]
    if not high_recall_candidates.empty:
        print("\nHigh-recall candidates (recall >= 0.80):")
        print(high_recall_candidates)
    else:
        print("\nNo thresholds met recall >= 0.80 in this grid.")

    best_f1_row = threshold_df.sort_values(by="f1", ascending=False).head(1)
    print("\nBest F1 candidate:")
    print(best_f1_row)


if __name__ == "__main__":
    main()
