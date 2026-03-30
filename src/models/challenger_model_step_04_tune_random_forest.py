import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 8, 12],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [2, 5],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True
    )

    print("=== CHALLENGER MODEL STEP 4: RANDOM FOREST TUNING PLAN ===")
    print("\nInput file:", INPUT_FILE)
    print("\nScoring objective for tuning: recall")
    print("\nParameter grid:")
    for k, v in param_grid.items():
        print(k, ":", v)

    print("\n=== WHAT YOU SHOULD RUN NOW ===")
    print("This script will run GridSearchCV on the training split only.")
    print("It will search Random Forest class weights and hyperparameters while optimizing recall.")
    print("The test set remains untouched until after best params are selected.")

    grid.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid.cv_results_)
    cols_to_show = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "param_classifier__n_estimators",
        "param_classifier__max_depth",
        "param_classifier__min_samples_split",
        "param_classifier__min_samples_leaf",
        "param_classifier__class_weight"
    ]

    print("\n=== BEST PARAMETERS ===")
    print(grid.best_params_)

    print("\n=== BEST CV RECALL ===")
    print(round(grid.best_score_, 4))

    print("\n=== TOP 10 GRID SEARCH RESULTS ===")
    print(cv_results_df[cols_to_show].sort_values(by="rank_test_score").head(10))

    print("\n=== INTERPRETATION CHECKPOINT ===")
    print("1. Did tuned RF improve recall over untuned RF CV recall (~0.7498)?")
    print("2. Which class_weight worked best?")
    print("3. Did shallower trees reduce overfitting risk?")
    print("4. Are the gains large enough to justify tuning complexity?")


if __name__ == "__main__":
    main()
