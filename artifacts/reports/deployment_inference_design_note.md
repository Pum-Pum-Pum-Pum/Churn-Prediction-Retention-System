# Deployment / Inference Design Note

## Chosen model for first deployment design
- Logistic Regression official baseline

## Why this model is chosen first
- interpretable
- simpler deployment path
- strong baseline performance
- easier monitoring and debugging

## First deployment assumptions
- batch or API inference can both use same sklearn pipeline artifact
- threshold defaults to business policy threshold 0.60
- model artifact loaded from artifacts/models/logistic_baseline_pipeline.joblib
- schema contract must be validated before scoring

## Immediate next implementation steps
1. train and persist official baseline model artifact
2. create inference wrapper function
3. add request/response schema validation
4. add inference logging
5. prepare simple API serving layer
