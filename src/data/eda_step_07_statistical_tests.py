import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = DATA_RAW / "Telco_customer_churn.xlsx"


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / max((kcorr - 1), (rcorr - 1))) if max((kcorr - 1), (rcorr - 1)) > 0 else np.nan


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

    categorical_cols = [
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
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    numeric_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges_clean"
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print("=== CHI-SQUARE TESTS: CATEGORICAL FEATURES VS CHURN ===")
    chi_rows = []

    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        chi_rows.append({
            "feature": col,
            "chi2_stat": round(chi2, 3),
            "p_value": p_value,
            "degrees_freedom": int(dof),
            "cramers_v": round(cramers_v(contingency), 3)
        })

    chi_df = pd.DataFrame(chi_rows).sort_values(by=["cramers_v", "chi2_stat"], ascending=[False, False])
    print(chi_df)

    print("\n=== MANN-WHITNEY U TESTS: NUMERIC FEATURES VS CHURN ===")
    mw_rows = []

    for col in numeric_cols:
        temp = df[[col, target_col]].dropna()
        group_0 = temp[temp[target_col] == 0][col]
        group_1 = temp[temp[target_col] == 1][col]

        stat, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")

        mw_rows.append({
            "feature": col,
            "non_churn_median": round(group_0.median(), 2),
            "churn_median": round(group_1.median(), 2),
            "median_diff": round(group_1.median() - group_0.median(), 2),
            "mannwhitney_u": round(stat, 3),
            "p_value": p_value
        })

    mw_df = pd.DataFrame(mw_rows).sort_values(by="p_value", ascending=True)
    print(mw_df)

    print("\n=== PRACTICAL SIGNIFICANCE HINTS ===")
    print("For categorical features, look at both p-value and Cramer's V.")
    print("For numeric features, do not rely only on p-value; examine effect size through median differences and business meaning.")
    print("Large sample sizes can make weak effects statistically significant.")


if __name__ == "__main__":
    main()
