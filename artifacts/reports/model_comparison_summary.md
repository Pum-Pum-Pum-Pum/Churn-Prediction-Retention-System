# Model Comparison Summary

## Models compared
1. Logistic Regression baseline
2. Untuned Random Forest challenger
3. Tuned Random Forest challenger
4. Gradient Boosting challenger

## Official baseline benchmark
### Logistic Regression @ threshold 0.60
- Accuracy: 0.7715
- Precision: 0.5537
- Recall: 0.7166
- F1: 0.6247
- ROC-AUC: 0.8477
- Confusion Matrix: [[819, 216], [106, 268]]

## Challenger comparisons
### Untuned Random Forest @ threshold 0.50
- Accuracy: 0.7693
- Precision: 0.5495
- Recall: 0.7273
- F1: 0.6260
- ROC-AUC: 0.8493
- Confusion Matrix: [[812, 223], [102, 272]]
- Interpretation: slightly better churn capture than logistic, but only marginal improvement with more complexity.

### Tuned Random Forest @ threshold 0.50
- Accuracy: 0.7672
- Precision: 0.5434
- Recall: 0.7701
- F1: 0.6372
- ROC-AUC: 0.8512
- Confusion Matrix: [[793, 242], [86, 288]]
- Interpretation: best recall among evaluated finalists, but also more false positives and higher model complexity.

### Gradient Boosting CV summary
- Accuracy CV: 0.8080
- Precision CV: 0.6629
- Recall CV: 0.5639
- F1 CV: 0.6093
- ROC-AUC CV: 0.8625
- Interpretation: strongest ranking quality, but weaker recall and F1 for churn objective than Random Forest.

## Current decision
### Official baseline
- Logistic Regression remains the official baseline benchmark.

### Leading challenger
- Tuned Random Forest is the strongest challenger so far.

## Why logistic remains the baseline
- simpler and easier to explain
- easier to maintain and deploy
- strong and stable performance
- only slightly worse than tuned Random Forest on holdout metrics

## Why tuned Random Forest is the leading challenger
- catches more churners than logistic baseline
- improves recall and F1
- slightly improves ROC-AUC
- but adds complexity and more false positives

## Recommended next move
- If business prioritizes churn capture strongly, continue with tuned Random Forest as top challenger.
- If simplicity, transparency, and operational stability matter more, keep logistic as preferred deployable model.
- Move next toward business-cost-based threshold selection, calibration review, or deployment/logging preparation.
