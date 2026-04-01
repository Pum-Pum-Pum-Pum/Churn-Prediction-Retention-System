# Model Selection Decision Log

## Official baseline: Logistic Regression
Reason:
- strong and stable performance
- easier to explain and deploy
- lower complexity and monitoring burden
- only slightly behind tuned RF on some metrics

## Leading challenger: Tuned Random Forest
Reason:
- improves recall and F1 over logistic baseline
- catches more churners
- but adds more false positives and complexity
- not promoted yet because improvement is marginal relative to operational cost

## Why Gradient Boosting was not selected
- strongest ROC-AUC among tested models
- but weaker recall and F1 for churn objective
- business goal prioritized churn capture over ranking quality alone
