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

    categorical_cols_to_review = [
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method"
    ]

    categorical_cols_to_review = [col for col in categorical_cols_to_review if col in df.columns]
    target_col = "churn_value"

    print("=== CATEGORICAL DISTRIBUTION + CHURN RATE ===")

    for feature in categorical_cols_to_review:
        print(f"\n\n===== {feature.upper()} =====")

        summary = (
            df.groupby(feature, dropna=False)
              .agg(
                  customer_count=(target_col, "size"),
                  churn_count=(target_col, "sum"),
                  churn_rate=(target_col, "mean")
              )
              .sort_values(by="customer_count", ascending=False)
        )

        summary["customer_pct"] = (summary["customer_count"] / len(df) * 100).round(2)
        summary["churn_rate"] = (summary["churn_rate"] * 100).round(2)
        summary["churn_pct_of_all_customers"] = (summary["churn_count"] / len(df) * 100).round(2)

        summary = summary[["customer_count", "customer_pct", "churn_count", "churn_rate", "churn_pct_of_all_customers"]]
        print(summary)

    print("\n\n=== TOP CATEGORY LEVELS BY CHURN RATE (MIN 100 CUSTOMERS) ===")
    rows = []

    for feature in categorical_cols_to_review:
        tmp = (
            df.groupby(feature, dropna=False)
              .agg(customer_count=(target_col, "size"), churn_rate=(target_col, "mean"))
              .reset_index()
              .rename(columns={feature: "category"})
        )
        tmp = tmp[tmp["customer_count"] >= 100].copy()
        tmp["feature"] = feature
        tmp["churn_rate"] = (tmp["churn_rate"] * 100).round(2)
        rows.append(tmp[["feature", "category", "customer_count", "churn_rate"]])

    combined = pd.concat(rows, ignore_index=True)

    print("\nTop 15 highest churn-rate categories:")
    print(combined.sort_values(by="churn_rate", ascending=False).head(15))

    print("\nTop 15 lowest churn-rate categories:")
    print(combined.sort_values(by="churn_rate", ascending=True).head(15))


if __name__ == "__main__":
    main()
