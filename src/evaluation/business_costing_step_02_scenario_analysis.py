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
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"
OUTPUT_FILE = REPORTS_DIR / "business_cost_scenario_summary.csv"

SCENARIOS = [
    {
        "scenario_name": "high_churn_capture_priority",
        "false_positive_cost": 1,
        "false_negative_cost": 8,
        "true_positive_benefit": 0,
        "capacity": 800
    },
    {
        "scenario_name": "balanced_business_case",
        "false_positive_cost": 1,
        "false_negative_cost": 5,
        "true_positive_benefit": 0,
        "capacity": 500
    },
    {
        "scenario_name": "capacity_constrained_campaign",
        "false_positive_cost": 2,
        "false_negative_cost": 5,
        "true_positive_benefit": 0,
        "capacity": 400
    }
]


def cost_table_for_scenario(y_true, y_prob, thresholds, scenario):
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (
            fp * scenario["false_positive_cost"] +
            fn * scenario["false_negative_cost"] -
            tp * scenario["true_positive_benefit"]
        )
        rows.append({
            "scenario_name": scenario["scenario_name"],
            "threshold": threshold,
            "predicted_positive_count": int(y_pred.sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "estimated_total_cost": round(total_cost, 2),
            "within_capacity": int(y_pred.sum()) <= scenario["capacity"]
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

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    scenario_frames = []
    for scenario in SCENARIOS:
        scenario_df = cost_table_for_scenario(y_test, y_prob, thresholds, scenario)
        scenario_frames.append(scenario_df)

    final_df = pd.concat(scenario_frames, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("=== BUSINESS COSTING STEP 2: SCENARIO ANALYSIS ===")
    print("\nScenario results saved to:", OUTPUT_FILE)

    for scenario in SCENARIOS:
        print(f"\n\n=== SCENARIO: {scenario['scenario_name']} ===")
        print("Assumptions:", scenario)
        scenario_df = final_df[final_df["scenario_name"] == scenario["scenario_name"]]

        print("\nLowest cost overall:")
        print(scenario_df.sort_values(by="estimated_total_cost", ascending=True).head(3))

        print("\nLowest cost within capacity:")
        within_capacity_df = scenario_df[scenario_df["within_capacity"] == True]
        if not within_capacity_df.empty:
            print(within_capacity_df.sort_values(by="estimated_total_cost", ascending=True).head(3))
        else:
            print("No threshold satisfies capacity.")

    print("\n=== INTERPRETATION CHECKPOINT ===")
    print("1. Does the recommended threshold shift across scenarios?")
    print("2. Which scenario pushes threshold lower to catch more churners?")
    print("3. Which scenario pushes threshold higher due to cost or capacity pressure?")
    print("4. This is why threshold choice must be tied to business assumptions.")


if __name__ == "__main__":
    main()
