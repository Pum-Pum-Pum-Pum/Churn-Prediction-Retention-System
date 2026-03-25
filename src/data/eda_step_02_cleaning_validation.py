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

    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== OBJECT COLUMNS ===")
    object_cols = df.select_dtypes(include="object").columns.tolist()
    print(object_cols)

    print("\n=== BLANK / WHITESPACE CHECK IN OBJECT COLUMNS ===")
    blank_summary = []
    for col in object_cols:
        blank_count = df[col].astype(str).str.strip().eq("").sum()
        blank_summary.append({"column": col, "blank_count": blank_count})
    blank_summary_df = pd.DataFrame(blank_summary).sort_values(by="blank_count", ascending=False)
    print(blank_summary_df[blank_summary_df["blank_count"] > 0])

    print("\n=== TOTAL_CHARGES RAW SAMPLE (TOP 10 DISTINCT VALUES) ===")
    if "total_charges" in df.columns:
        print(df["total_charges"].astype(str).str.strip().value_counts(dropna=False).head(10))

        df["total_charges_clean"] = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")

        print("\n=== TOTAL_CHARGES CONVERSION CHECK ===")
        print("Original dtype:", df["total_charges"].dtype)
        print("Converted dtype:", df["total_charges_clean"].dtype)
        print("Nulls after conversion:", df["total_charges_clean"].isna().sum())

        print("\nRows where total_charges became NaN after conversion:")
        invalid_total_charges = df[df["total_charges_clean"].isna()][["customerid", "tenure_months", "monthly_charges", "total_charges", "churn_label", "churn_value"]]
        print(invalid_total_charges.head(20))

    print("\n=== CATEGORY LEVEL REVIEW FOR LOW-CARDINALITY OBJECT COLUMNS ===")
    low_card_cols = [col for col in object_cols if df[col].nunique(dropna=False) <= 10]
    for col in low_card_cols:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))

    id_cols = [col for col in ["customerid"] if col in df.columns]
    target_cols = [col for col in ["churn_value", "churn_label"] if col in df.columns]
    leakage_review_cols = [
        col for col in ["churn_reason", "churn_score", "cltv"] if col in df.columns
    ]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = object_cols.copy()

    print("\n=== COLUMN GROUPING FOR NEXT STEPS ===")
    print("ID columns:", id_cols)
    print("Target columns:", target_cols)
    print("Leakage review columns:", leakage_review_cols)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)


if __name__ == "__main__":
    main()
