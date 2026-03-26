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

    numeric_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges_clean",
        "churn_score",
        "cltv",
        "churn_value"
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print("=== NUMERIC CORRELATION MATRIX ===")
    corr = df[numeric_cols].corr(numeric_only=True).round(3)
    print(corr)

    print("\n=== TOP ABSOLUTE CORRELATIONS AMONG NUMERIC FEATURES ===")
    corr_pairs = corr.abs().unstack().reset_index()
    corr_pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    corr_pairs = corr_pairs[corr_pairs["feature_1"] != corr_pairs["feature_2"]]
    corr_pairs["pair_key"] = corr_pairs.apply(
        lambda row: tuple(sorted([row["feature_1"], row["feature_2"]])), axis=1
    )
    corr_pairs = corr_pairs.drop_duplicates(subset="pair_key").drop(columns="pair_key")
    print(corr_pairs.sort_values(by="abs_corr", ascending=False).head(15))

    print("\n=== BUSINESS DEPENDENCY CHECKS ===")

    dependency_checks = [
        ("internet_service", "online_security"),
        ("internet_service", "online_backup"),
        ("internet_service", "device_protection"),
        ("internet_service", "tech_support"),
        ("internet_service", "streaming_tv"),
        ("internet_service", "streaming_movies"),
        ("phone_service", "multiple_lines")
    ]

    for parent_col, child_col in dependency_checks:
        if parent_col in df.columns and child_col in df.columns:
            print(f"\n--- {parent_col} vs {child_col} ---")
            print(pd.crosstab(df[parent_col], df[child_col]))

    print("\n=== COMBINED BUSINESS PATTERN CHECKS ===")

    if {"contract", "paperless_billing", "churn_value"}.issubset(df.columns):
        print("\n--- contract vs paperless_billing (mean churn) ---")
        contract_paperless = pd.pivot_table(
            df,
            index="contract",
            columns="paperless_billing",
            values="churn_value",
            aggfunc="mean"
        )
        print((contract_paperless * 100).round(2))

    if {"internet_service", "contract", "churn_value"}.issubset(df.columns):
        print("\n--- internet_service vs contract (mean churn) ---")
        internet_contract = pd.pivot_table(
            df,
            index="internet_service",
            columns="contract",
            values="churn_value",
            aggfunc="mean"
        )
        print((internet_contract * 100).round(2))

    if {"payment_method", "contract", "churn_value"}.issubset(df.columns):
        print("\n--- payment_method vs contract (mean churn) ---")
        payment_contract = pd.pivot_table(
            df,
            index="payment_method",
            columns="contract",
            values="churn_value",
            aggfunc="mean"
        )
        print((payment_contract * 100).round(2))


if __name__ == "__main__":
    main()
