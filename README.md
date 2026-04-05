# Customer Churn Prediction and Retention Decision System

## Project objective
Build a production-minded churn prediction system that:
- identifies customers likely to churn
- supports threshold-based retention decisions
- records model, schema, threshold, and experiment governance artifacts
- prepares the project for deployment and monitoring

## Dataset
- IBM Telco Customer Churn dataset
- 7043 rows
- 33 raw columns
- Target used: `churn_value`

## Workflow completed
### 1. EDA
- data audit
- missing value validation
- numeric and categorical analysis
- leakage review
- structural category review
- hypothesis-testing-style significance checks
- feature shortlist summary

### 2. Feature engineering
- baseline feature selection
- leakage-safe drop list
- cleaned `total_charges_clean`
- binary mapping for selected fields
- baseline processed dataset creation
- preprocessing and split strategy design

### 3. Modeling
- Logistic Regression baseline
- threshold review and class-weight review
- final baseline holdout evaluation
- Random Forest challenger
- tuned Random Forest challenger
- Gradient Boosting challenger
- model comparison summary

### 4. Business costing
- cost-based threshold review
- multi-scenario threshold comparison
- operating policy artifact

### 5. Deployment / inference design
- inference schema contract
- model artifact saving
- inference wrapper
- inference logging
- batch inference
- API skeleton
- drift monitoring starter

## Final model status
### Official baseline
- Logistic Regression
- threshold = 0.60
- chosen for simplicity, interpretability, and strong stable performance

### Leading challenger
- Tuned Random Forest
- improved recall and F1, but not promoted due to marginal gain vs added complexity

## Key artifacts
### Reports
- `artifacts/reports/eda_feature_shortlist_summary.md`
- `artifacts/reports/model_comparison_summary.md`
- `artifacts/reports/model_selection_decision_log.md`
- `artifacts/reports/threshold_decision_artifact.md`
- `artifacts/reports/business_cost_operating_policy.md`
- `artifacts/reports/deployment_inference_design_note.md`
- `artifacts/reports/inference_logging_plan.md`
- `artifacts/reports/model_artifact_persistence_plan.md`

### Structured tracking
- `artifacts/reports/model_metrics_summary.csv`
- `artifacts/reports/model_metrics_summary.json`
- `artifacts/registry/experiment_registry.csv`
- `artifacts/registry/experiment_registry.json`
- `artifacts/schemas/feature_schema_contract.json`
- `artifacts/schemas/inference_api_contract.json`

### Model / deployment files
- `src/models/train_official_baseline.py`
- `src/deployment/deployment_step_03_inference_wrapper.py`
- `src/deployment/deployment_step_04_inference_logging.py`
- `src/deployment/deployment_step_05_batch_inference.py`
- `src/deployment/deployment_step_06_api_server_skeleton.py`
- `src/monitoring/deployment_step_07_drift_monitoring_starter.py`

## Official baseline holdout metrics
- Accuracy: 0.7715
- Precision: 0.5537
- Recall: 0.7166
- F1: 0.6247
- ROC-AUC: 0.8477
- Confusion matrix: `[[819, 216], [106, 268]]`

## Business decision summary
- Under the balanced business scenario, threshold `0.60` was selected as the lowest-cost feasible operating point within campaign capacity.
- Lower thresholds improved recall but exceeded operational capacity.
- Higher thresholds reduced outreach volume but increased missed churners.

## Production-minded lessons from this project
- leakage prevention must happen before modeling
- threshold selection is a business decision, not only an ML decision
- model comparison must include business operating point, not just ROC-AUC
- deployment readiness requires schema, metadata, logging, and artifact governance
- monitoring must be planned even before full production rollout

## Recommended next steps
- add API request/response tests
- add categorical drift monitoring and alert rules
- add cost assumptions from real business stakeholders
- optionally evaluate stronger boosting models (e.g. XGBoost / LightGBM)
- package the service for deployment (container / cloud / scheduler)

## How to retrain the official baseline
```bash
python src/models/train_official_baseline.py
```

## How to run batch inference
```bash
python src/deployment/deployment_step_05_batch_inference.py
```

## How to test inference wrapper
```bash
python src/deployment/deployment_step_03_inference_wrapper.py
```

## How to review drift starter
```bash
python src/monitoring/deployment_step_07_drift_monitoring_starter.py
```

This completed the project!
