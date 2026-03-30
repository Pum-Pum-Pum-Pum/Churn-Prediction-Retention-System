import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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

FINAL_RF_THRESHOLD = 0.50  # chosen as closest balanced comparison to logistic baseline operating volume


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
    y_pred = (y_prob >= FINAL_RF_THRESHOLD).astype(int)

    print("=== CHALLENGER MODEL STEP 3: FINAL RANDOM FOREST HOLDOUT EVALUATION ===")
    print("\nModel: RandomForestClassifier(class_weight='balanced_subsample')")
    print("Chosen threshold:", FINAL_RF_THRESHOLD)

    print("\n=== FINAL TEST METRICS ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\n=== COMPARE AGAINST LOGISTIC BASELINE @ 0.60 ===")
    print("Logistic baseline benchmark:")
    print("- Accuracy: 0.7715")
    print("- Precision: 0.5537")
    print("- Recall: 0.7166")
    print("- F1: 0.6247")
    print("- ROC-AUC: 0.8477")
    print("- Confusion matrix: [[819, 216], [106, 268]]")

    print("\n=== BUSINESS DECISION CHECKPOINT ===")
    print("1. Does Random Forest improve recall at similar outreach volume?")
    print("2. Is the precision acceptable relative to the logistic baseline?")
    print("3. Is the metric lift worth the added model complexity and overfitting risk?")


if __name__ == "__main__":
    main()
