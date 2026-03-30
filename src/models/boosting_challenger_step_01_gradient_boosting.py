import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

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
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
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

    print("=== BOOSTING CHALLENGER STEP 1: GRADIENT BOOSTING CV RESULTS ===")
    print("\nInput file:", INPUT_FILE)

    print("\n=== MODEL CONFIGURATION ===")
    print(model)

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

    print("\n=== WHAT TO COMPARE ===")
    print("1. Does Gradient Boosting beat logistic baseline on ROC-AUC, recall, and F1?")
    print("2. Does it outperform Random Forest while controlling overfitting better?")
    print("3. Is train-vs-test gap acceptable?")
    print("4. Is this now the strongest challenger family so far?")


if __name__ == "__main__":
    main()
