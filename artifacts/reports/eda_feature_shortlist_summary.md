# EDA Feature Shortlist Summary

## Dataset overview
- Rows: 7043
- Columns: 33 raw columns
- Target: churn_value (1 = churn, 0 = retained)
- Class balance: ~26.5% churn, ~73.5% non-churn

## Key data quality findings
- No duplicate rows
- No duplicate customer IDs
- `total_charges` contained 11 hidden blank values and was converted to `total_charges_clean`
- `churn_reason` is missing for most non-churned customers because it is post-event information

## Structural category findings
- `No internet service` and `No phone service` are structural business states, not missing values
- These states are dependency-driven and should be encoded carefully

## Strong churn signals observed during EDA
### Numeric
- Lower `tenure_months` strongly associated with churn
- Higher `monthly_charges` associated with churn
- Lower `total_charges_clean` among churners due to shorter tenure

### Categorical
- Strongest categorical signals included: `contract`, `dependents`, `online_security`, `tech_support`, `internet_service`, `payment_method`
- Early evidence suggests strong interaction effects for combinations such as month-to-month + electronic check and month-to-month + fiber optic

## Leakage / exclusion review
### Definite or near-definite exclusion
- `customerid` -> identifier only
- `count` -> constant field
- `country` -> constant field
- `state` -> constant field
- `churn_label` -> duplicate target representation if using `churn_value`
- `churn_reason` -> post-event leakage
- `churn_score` -> likely precomputed model output / leakage risk

### Review carefully before modeling
- `cltv` -> derived business metric, possible leakage or deployment inconsistency
- `zip_code`, `latitude`, `longitude`, `lat_long`, `city` -> may add location signal but risk sparse/high-cardinality or proxy behavior

## Recommended first-pass feature buckets
### Keep candidates
- `tenure_months`
- `monthly_charges`
- `total_charges_clean`
- `contract`
- `internet_service`
- `online_security`
- `online_backup`
- `device_protection`
- `tech_support`
- `streaming_tv`
- `streaming_movies`
- `payment_method`
- `paperless_billing`
- `partner`
- `dependents`
- `senior_citizen`
- `multiple_lines`
- `phone_service`
- `gender`

### Review or engineer carefully
- Service bundle depth / add-on count
- Contract commitment features
- Automatic payment vs manual payment
- Internet-service-dependent fields
- Geographic fields

## Business takeaways
- Churn risk is highest early in the lifecycle, especially within the first year
- Month-to-month customers are much riskier than long-contract customers
- High monthly charges combined with low tenure appears especially risky
- Service adoption patterns and support/security relationships appear strongly tied to churn

## EDA conclusion
This dataset is suitable for churn modeling after careful leakage removal, type cleaning, and structured feature preparation. The strongest early candidates combine lifecycle, pricing, contract, service configuration, and billing behavior.
