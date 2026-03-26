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

    numeric_cols_to_review = [
        "tenure_months",
        "monthly_charges",
        "total_charges_clean",
        "churn_score",
        "cltv",
        "latitude",
        "longitude",
        "zip_code"
    ]

    numeric_cols_to_review = [col for col in numeric_cols_to_review if col in df.columns]

    print("=== NUMERIC UNIVARIATE SUMMARY ===")
    summary_rows = []

    for col in numeric_cols_to_review:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = series[(series < lower_bound) | (series > upper_bound)].count()

        summary_rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "mean": round(series.mean(), 2),
            "median": round(series.median(), 2),
            "std": round(series.std(), 2),
            "min": round(series.min(), 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "max": round(series.max(), 2),
            "skew": round(series.skew(), 2),
            "outlier_count_iqr": int(outlier_count)
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)

    print("\n=== DETAILED VALUE CHECKS ===")
    for col in numeric_cols_to_review:
        print(f"\n--- {col} ---")
        print("Top 10 smallest values:")
        print(df[col].sort_values().head(10).tolist())
        print("Top 10 largest values:")
        print(df[col].sort_values(ascending=False).head(10).tolist())

    print("\n=== BUSINESS CHECKS ===")
    if "tenure_months" in df.columns:
        print("Tenure = 0 count:", int((df["tenure_months"] == 0).sum()))
        print("Tenure <= 1 count:", int((df["tenure_months"] <= 1).sum()))

    if "monthly_charges" in df.columns:
        print("Monthly charges = 0 count:", int((df["monthly_charges"] == 0).sum()))

    if "total_charges_clean" in df.columns:
        print("Total charges = 0 count:", int((df["total_charges_clean"] == 0).sum()))
        print("Total charges missing count:", int(df["total_charges_clean"].isna().sum()))


if __name__ == "__main__":
    main()
