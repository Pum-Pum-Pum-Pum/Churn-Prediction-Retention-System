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

    numeric_cols_to_compare = [
        "tenure_months",
        "monthly_charges",
        "total_charges_clean",
        "churn_score",
        "cltv"
    ]
    numeric_cols_to_compare = [col for col in numeric_cols_to_compare if col in df.columns]

    print("=== NUMERIC FEATURES VS CHURN ===")

    rows = []
    for col in numeric_cols_to_compare:
        grouped = df.groupby(target_col)[col]

        non_churn = df[df[target_col] == 0][col]
        churn = df[df[target_col] == 1][col]

        rows.append({
            "feature": col,
            "non_churn_mean": round(non_churn.mean(), 2),
            "churn_mean": round(churn.mean(), 2),
            "mean_diff_churn_minus_nonchurn": round(churn.mean() - non_churn.mean(), 2),
            "non_churn_median": round(non_churn.median(), 2),
            "churn_median": round(churn.median(), 2),
            "median_diff_churn_minus_nonchurn": round(churn.median() - non_churn.median(), 2),
            "non_churn_std": round(non_churn.std(), 2),
            "churn_std": round(churn.std(), 2),
            "missing_count": int(df[col].isna().sum())
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df)

    print("\n=== DECILE-STYLE GROUP CHECKS FOR KEY NUMERIC FEATURES ===")
    key_cols = [col for col in ["tenure_months", "monthly_charges", "total_charges_clean"] if col in df.columns]

    for col in key_cols:
        print(f"\n--- {col} ---")

        temp = df[[col, target_col]].copy()
        temp = temp.dropna(subset=[col])

        try:
            temp["bin"] = pd.qcut(temp[col], q=10, duplicates="drop")
            decile_summary = temp.groupby("bin", observed=False).agg(
                customer_count=(target_col, "size"),
                churn_rate=(target_col, "mean")
            )
            decile_summary["churn_rate"] = (decile_summary["churn_rate"] * 100).round(2)
            print(decile_summary)
        except ValueError:
            print("Could not create quantile bins for this feature.")

    print("\n=== BUSINESS CHECK: TENURE BY CHURN ===")
    if "tenure_months" in df.columns:
        tenure_bins = pd.cut(
            df["tenure_months"],
            bins=[-1, 1, 6, 12, 24, 48, 72],
            labels=["0-1", "2-6", "7-12", "13-24", "25-48", "49-72"]
        )
        tenure_summary = df.groupby(tenure_bins, observed=False).agg(
            customer_count=(target_col, "size"), churn_rate=(target_col, "mean")
        )
        tenure_summary["churn_rate"] = (tenure_summary["churn_rate"] * 100).round(2)
        print(tenure_summary)


if __name__ == "__main__":
    main()
