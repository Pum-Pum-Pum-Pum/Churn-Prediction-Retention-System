import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"


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
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    results_df = pd.DataFrame(cv_results)

    print("=== MODELING STEP 1: BASELINE LOGISTIC REGRESSION CV RESULTS ===")
    print("\nInput file:", INPUT_FILE)

    print("\n=== TRAIN / TEST SPLIT ===")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train distribution:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\n=== PER-FOLD CV RESULTS ===")
    print(results_df)

    print("\n=== MEAN CV METRICS (TEST) ===")
    for metric in scoring.keys():
        print(metric, ":", round(results_df[f"test_{metric}"].mean(), 4))
        print(metric, "std:", round(results_df[f"test_{metric}"].std(), 4))

    print("\n=== MEAN CV METRICS (TRAIN) ===")
    for metric in scoring.keys():
        print(metric, ":", round(results_df[f"train_{metric}"].mean(), 4))
        print(metric, "std:", round(results_df[f"train_{metric}"].std(), 4))

if __name__ == "__main__":
    main()
