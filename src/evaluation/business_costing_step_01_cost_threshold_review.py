import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"
COST_CONFIG_FILE = CONFIG_DIR / "business_cost_assumptions.json"
OUTPUT_FILE = REPORTS_DIR / "business_cost_threshold_summary.csv"

DEFAULT_COSTS = {
    "false_positive_cost": 1,
    "false_negative_cost": 5,
    "true_positive_benefit": 0,
    "retention_team_capacity_per_cycle": 500,
    "notes": "Default placeholder assumptions used because business input is not finalized."
}


def load_cost_config():
    if COST_CONFIG_FILE.exists():
        with open(COST_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    merged = DEFAULT_COSTS.copy()
    for key in DEFAULT_COSTS:
        value = config.get(key)
        if value is not None:
            merged[key] = value
    return merged


def threshold_cost_table(y_true, y_prob, thresholds, costs):
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (
            fp * costs["false_positive_cost"] +
            fn * costs["false_negative_cost"] -
            tp * costs["true_positive_benefit"]
        )

        rows.append({
            "threshold": threshold,
            "predicted_positive_count": int(y_pred.sum()),
            "predicted_positive_pct": round(y_pred.mean() * 100, 2),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "estimated_total_cost": round(total_cost, 2),
            "within_capacity": int(y_pred.sum()) <= costs["retention_team_capacity_per_cycle"]
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_FILE)
    costs = load_cost_config()

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

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    cost_df = threshold_cost_table(y_test, y_prob, thresholds, costs)
    cost_df.to_csv(OUTPUT_FILE, index=False)

    print("=== BUSINESS COSTING STEP 1: COST-BASED THRESHOLD REVIEW ===")
    print("\nCost assumptions used:")
    print(costs)

    print("\n=== THRESHOLD COST TABLE ===")
    print(cost_df)

    print("\n=== LOWEST ESTIMATED COST OVERALL ===")
    print(cost_df.sort_values(by="estimated_total_cost", ascending=True).head(5))

    print("\n=== LOWEST ESTIMATED COST WITHIN CAPACITY ===")
    within_capacity_df = cost_df[cost_df["within_capacity"] == True]
    if not within_capacity_df.empty:
        print(within_capacity_df.sort_values(by="estimated_total_cost", ascending=True).head(5))
    else:
        print("No thresholds satisfy the current capacity constraint.")

    print("\nSaved cost summary to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
