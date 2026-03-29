import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    print("=== FEATURE ENGINEERING STEP 5: CV-BASED PIPELINE SETUP ===")
    print("\nInput file:", INPUT_FILE)

    print("\n=== HOLDOUT STRATEGY ===")
    print("Train size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape)
    print("Train target distribution:")
    print(y_train.value_counts(normalize=True).round(4))
    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\n=== CV STRATEGY ===")
    print(cv)
    print("CV folds: 5, stratified, shuffled, random_state=42")

    print("\n=== NUMERIC FEATURES ===")
    print(numeric_features)

    print("\n=== CATEGORICAL FEATURES ===")
    print(categorical_features)

    print("\n=== PREPROCESSOR BLUEPRINT ===")
    print(preprocessor)

    print("\n=== BASELINE MODEL PIPELINE ===")
    print(model)

    print("\n=== WHY THIS SETUP ===")
    print("1. Test set remains untouched until final evaluation.")
    print("2. Cross-validation happens only on training data.")
    print("3. Preprocessing is fit inside each fold, preventing leakage.")
    print("4. Logistic regression gives an interpretable baseline.")
    print("5. This becomes the benchmark for later tree-based challengers.")


if __name__ == "__main__":
    main()
