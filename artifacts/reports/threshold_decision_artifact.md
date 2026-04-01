# Threshold Decision Artifact

## Logistic baseline threshold choice
Chosen threshold: 0.60
Reason:
- balanced operating point
- stronger F1 tradeoff than more aggressive outreach settings
- lower outreach volume than high-recall option

## Alternative business conditions
- Use 0.40 when missing churners is much more expensive than contacting non-churners
- Use 0.70 when outreach capacity is tightly constrained

## Random Forest tuned threshold
Chosen comparison threshold: 0.50
Reason:
- fair comparison region against logistic baseline operating volume
