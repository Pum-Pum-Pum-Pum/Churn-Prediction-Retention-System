import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = DATA_RAW / "Telco_customer_churn.xlsx"


def main():
    df = pd.read_excel(RAW_DATA_FILE)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df["total_charges_clean"] = pd.to_numeric(
        df["total_charges"].astype(str).str.strip(),
        errors="coerce"
    )

    target_col = "churn_value"

    drop_list = [
        "customerid",
        "count",
        "country",
        "state",
        "churn_label",
        "churn_reason",
        "churn_score",
        "total_charges"
    ]
    drop_list = [c for c in drop_list if c in df.columns]

    baseline_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges_clean",
        "contract",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "payment_method",
        "paperless_billing",
        "partner",
        "dependents",
        "senior_citizen"
    ]
    baseline_features = [c for c in baseline_features if c in df.columns]

    baseline_df = df[baseline_features + [target_col]].copy()

    numeric_features = baseline_df[baseline_features].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = baseline_df[baseline_features].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("=== FEATURE ENGINEERING STEP 2: BASELINE PREPROCESSING DESIGN ===")
    print("\nTarget column:", target_col)

    print("\n=== FINAL DROP LIST FOR BASELINE ===")
    print(drop_list)

    print("\n=== FINAL BASELINE FEATURE LIST ===")
    print(baseline_features)

    print("\n=== NUMERIC FEATURES ===")
    print(numeric_features)

    print("\n=== CATEGORICAL FEATURES ===")
    print(categorical_features)

    print("\n=== NUMERIC MISSINGNESS PLAN ===")
    numeric_missing = pd.DataFrame({
        "missing_count": baseline_df[numeric_features].isna().sum(),
        "missing_pct": (baseline_df[numeric_features].isna().sum() / len(baseline_df) * 100).round(2)
    }).sort_values(by="missing_pct", ascending=False)
    print(numeric_missing)
    print("Plan: median imputation for baseline numeric pipeline.")

    print("\n=== CATEGORICAL MISSINGNESS PLAN ===")
    categorical_missing = pd.DataFrame({
        "missing_count": baseline_df[categorical_features].isna().sum(),
        "missing_pct": (baseline_df[categorical_features].isna().sum() / len(baseline_df) * 100).round(2)
    }).sort_values(by="missing_pct", ascending=False)
    print(categorical_missing)
    print("Plan: most_frequent imputation for baseline categorical pipeline if needed.")

if __name__ == "__main__":
    main()
