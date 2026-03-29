import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RAW_DATA_FILE = DATA_RAW / "Telco_customer_churn.xlsx"
OUTPUT_FILE = DATA_PROCESSED / "baseline_model_input.csv"


def main():
    df = pd.read_excel(RAW_DATA_FILE)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # 1. clean numeric field
    df["total_charges_clean"] = pd.to_numeric(
        df["total_charges"].astype(str).str.strip(),
        errors="coerce"
    )

    # 2. business-rule imputation for known billing-lifecycle issue
    # if tenure is 0 and total charges is missing, set to 0
    condition = (df["total_charges_clean"].isna()) & (df["tenure_months"] == 0)
    df.loc[condition, "total_charges_clean"] = 0

    # 3. define baseline columns
    target_col = "churn_value"
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

    model_df = df[baseline_features + [target_col]].copy()

    # 4. binary mapping for clean yes/no style columns
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = ["paperless_billing", "partner", "dependents", "senior_citizen"]
    binary_cols = [c for c in binary_cols if c in model_df.columns]

    for col in binary_cols:
        unique_values = set(model_df[col].dropna().unique())
        if unique_values.issubset(set(binary_map.keys())):
            model_df[col] = model_df[col].map(binary_map)

    print("=== FEATURE ENGINEERING STEP 3: BASELINE DATA PREPARATION ===")
    print("\nOutput file:", OUTPUT_FILE)

    print("\n=== FINAL MODEL DATAFRAME SHAPE ===")
    print(model_df.shape)

    print("\n=== FINAL COLUMNS ===")
    print(model_df.columns.tolist())

    print("\n=== TOTAL_CHARGES_CLEAN AFTER BUSINESS-RULE IMPUTATION ===")
    print("Remaining missing count:", int(model_df["total_charges_clean"].isna().sum()))
    print("Rows set to 0 by tenure rule:", int(condition.sum()))

    print("\n=== DTYPES AFTER BINARY MAPPING ===")
    print(model_df.dtypes)

    print("\n=== SAMPLE ROWS ===")
    print(model_df.head())

    print("\n=== TARGET DISTRIBUTION ===")
    print(model_df[target_col].value_counts())
    print(model_df[target_col].value_counts(normalize=True).round(4))

    model_df.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved baseline model input to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
