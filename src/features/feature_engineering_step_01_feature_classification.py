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

    drop_immediately = [
        "customerid",
        "count",
        "country",
        "state",
        "churn_label",
        "churn_reason",
        "churn_score"
    ]

    review_carefully = [
        "cltv",
        "zip_code",
        "latitude",
        "longitude",
        "lat_long",
        "city",
        "gender",
        "phone_service",
        "multiple_lines",
        "streaming_tv",
        "streaming_movies"
    ]

    keep_core = [
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

    drop_immediately = [c for c in drop_immediately if c in df.columns]
    review_carefully = [c for c in review_carefully if c in df.columns]
    keep_core = [c for c in keep_core if c in df.columns]

    all_classified = set(drop_immediately + review_carefully + keep_core + [target_col])
    unclassified = [c for c in df.columns if c not in all_classified]

    print("=== FEATURE ENGINEERING STEP 1: FEATURE CLASSIFICATION ===")
    print("\nTarget column:", target_col)

    print("\n=== DROP IMMEDIATELY ===")
    print(drop_immediately)

    print("\n=== REVIEW CAREFULLY ===")
    print(review_carefully)

    print("\n=== KEEP CORE FEATURES ===")
    print(keep_core)

    print("\n=== UNCLASSIFIED COLUMNS (DECIDE EXPLICITLY) ===")
    print(unclassified)

    print("\n=== RECOMMENDED FIRST MODEL FEATURE SET (SAFE START) ===")
    safe_start = keep_core.copy()
    print(safe_start)

    print("\n=== MISSINGNESS CHECK FOR SAFE START FEATURES ===")
    missing_summary = pd.DataFrame({
        "missing_count": df[safe_start].isna().sum(),
        "missing_pct": (df[safe_start].isna().sum() / len(df) * 100).round(2)
    }).sort_values(by="missing_pct", ascending=False)
    print(missing_summary)

    print("\n=== CATEGORICAL VS NUMERIC IN SAFE START ===")
    safe_df = df[safe_start]
    numeric_cols = safe_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = safe_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)

if __name__ == "__main__":
    main()
