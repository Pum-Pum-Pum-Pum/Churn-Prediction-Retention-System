# Inference Logging Plan

## Goal
Capture enough information at prediction time to support:
- debugging
- monitoring
- auditability
- drift analysis
- threshold review

## Recommended fields to log per inference
- timestamp
- request_id
- model_version
- threshold_used
- churn_probability
- predicted_label
- input_feature_hash or raw feature snapshot (depending on privacy rules)
- API latency
- optional downstream outcome label when available later

## Why this matters
- helps diagnose prediction issues
- supports data drift and concept drift checks
- enables later recalibration and threshold review
- creates traceability for business decisions

## Governance note
Do not log sensitive raw customer data unless privacy policy explicitly allows it.
Prefer hashed identifiers and carefully selected feature snapshots.
