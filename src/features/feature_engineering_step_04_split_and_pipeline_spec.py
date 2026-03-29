import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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

    print("=== FEATURE ENGINEERING STEP 4: SPLIT + PIPELINE SPEC ===")
    print("\nInput file:", INPUT_FILE)

    print("\n=== FEATURE MATRIX / TARGET SHAPE ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\n=== TRAIN / TEST SPLIT SHAPE ===")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\n=== STRATIFICATION CHECK ===")
    print("Overall target distribution:")
    print(y.value_counts(normalize=True).round(4))
    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True).round(4))
    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\n=== NUMERIC FEATURES FOR PIPELINE ===")
    print(numeric_features)

    print("\n=== CATEGORICAL FEATURES FOR PIPELINE ===")
    print(categorical_features)

    print("\n=== BASELINE PREPROCESSING BLUEPRINT ===")
    print("Numeric transformer:")
    print("- SimpleImputer(strategy='median')")
    print("- Optional StandardScaler() for logistic regression baseline")

    print("\nCategorical transformer:")
    print("- SimpleImputer(strategy='most_frequent')")
    print("- OneHotEncoder(handle_unknown='ignore')")

    print("\nColumnTransformer design:")
    print("- numeric pipeline -> numeric_features")
    print("- categorical pipeline -> categorical_features")

    print("\nBaseline model pipeline candidate:")
    print("- ColumnTransformer(preprocessing) + LogisticRegression")

    print("\n=== IMPLEMENTATION DECISIONS TO CONFIRM ===")
    print("1. Use 80/20 stratified split for first baseline.")
    print("2. Keep preprocessing inside sklearn pipeline for train/inference consistency.")
    print("3. Use handle_unknown='ignore' for categorical safety in test/inference.")
    print("4. Add scaling only if needed for chosen linear model baseline.")
    print("5. Keep test set untouched until evaluation stage.")


if __name__ == "__main__":
    main()
