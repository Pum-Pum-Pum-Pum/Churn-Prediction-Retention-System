# Final Project Wrap-up Summary

## Project completed
Customer Churn Prediction and Retention Decision System

## What was built
- full EDA workflow from audit to statistical testing
- feature engineering and baseline dataset preparation
- logistic regression official baseline
- random forest challengers (untuned and tuned)
- gradient boosting challenger review
- cost-based threshold analysis
- threshold operating policy artifact
- model comparison summary and structured experiment tracking
- saved model artifact and metadata
- inference wrapper
- inference logging
- batch inference support
- FastAPI serving skeleton
- API manual and automated-style test artifacts
- starter drift monitoring and prediction-behavior alerting
- README and governance / deployment artifacts

## Final modeling decision
### Official baseline
- Logistic Regression
- chosen threshold: 0.60
- selected for strong performance, interpretability, stability, and lower deployment complexity

### Leading challenger
- Tuned Random Forest
- improved recall and F1 slightly, but not enough to clearly replace logistic regression for first deployment

## Key holdout benchmark for official baseline
- Accuracy: 0.7715
- Precision: 0.5537
- Recall: 0.7166
- F1: 0.6247
- ROC-AUC: 0.8477
- Confusion matrix: [[819, 216], [106, 268]]

## Key business lesson
The best threshold is not universal. It depends on:
- false-positive cost
- false-negative cost
- retention team capacity
- current business objective

## Key production lesson
A good ML project is not only about model metrics. It must also include:
- leakage review
- schema contracts
- experiment tracking
- threshold policy
- reproducible training
- inference logging
- monitoring starter
- handoff documentation

## What remains for future improvement
- stricter automated API testing with pytest in environment
- richer experiment logging and model versioning
- categorical drift monitoring
- better production alert routing
- stronger challenger families (e.g. XGBoost / LightGBM)
- containerization and deployment packaging

## Personal learning outcome
This project covered the workflow from raw data to deployable ML design, and moved beyond notebook-only analysis into production-minded decision making.
