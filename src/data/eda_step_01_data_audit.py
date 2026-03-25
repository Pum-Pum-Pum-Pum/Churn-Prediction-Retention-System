import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = DATA_RAW / "Telco_customer_churn.xlsx"


def main():
    df = pd.read_excel(RAW_DATA_FILE)

    print("=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== ORIGINAL COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    clean_df = df.copy()
    clean_df.columns = (
        clean_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    print("\n=== CLEANED COLUMNS ===")
    print(clean_df.columns.tolist())

    missing_summary = pd.DataFrame({
        "missing_count": clean_df.isna().sum(),
        "missing_pct": (clean_df.isna().sum() / len(clean_df) * 100).round(2)
    }).sort_values(by="missing_pct", ascending=False)

    print("\n=== MISSING VALUE SUMMARY ===")
    print(missing_summary[missing_summary["missing_count"] > 0])

    print("\n=== DUPLICATE CHECK ===")
    print("Duplicate rows:", clean_df.duplicated().sum())

    if "customerid" in clean_df.columns:
        print("Duplicate customerid:", clean_df["customerid"].duplicated().sum())
    elif "customer_id" in clean_df.columns:
        print("Duplicate customer_id:", clean_df["customer_id"].duplicated().sum())

    unique_summary = pd.DataFrame({
        "n_unique": clean_df.nunique(dropna=False),
        "dtype": clean_df.dtypes.astype(str)
    }).sort_values(by="n_unique", ascending=False)

    print("\n=== UNIQUE VALUE SUMMARY ===")
    print(unique_summary)

    leakage_candidates = [
        "customerid", "customer_id",
        "count",
        "country",
        "state",
        "churn_label", "churn_value",
        "churn_score",
        "cltv",
        "churn_reason"
    ]

    print("\n=== REVIEW COLUMNS (LEAKAGE / LOW VALUE / ID) ===")
    print([col for col in leakage_candidates if col in clean_df.columns])

    if "churn_value" in clean_df.columns:
        print("\n=== TARGET DISTRIBUTION: churn_value ===")
        print(clean_df["churn_value"].value_counts(dropna=False))
        print(clean_df["churn_value"].value_counts(normalize=True, dropna=False).round(4))
    elif "churn_label" in clean_df.columns:
        print("\n=== TARGET DISTRIBUTION: churn_label ===")
        print(clean_df["churn_label"].value_counts(dropna=False))
        print(clean_df["churn_label"].value_counts(normalize=True, dropna=False).round(4))


if __name__ == "__main__":
    main()
