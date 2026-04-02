# Business Costing Operating Policy

## Chosen business scenario
Balanced business case

### Scenario assumptions
- False positive cost = 1
- False negative cost = 5
- True positive benefit = 0
- Retention team capacity per cycle = 500

## Chosen operating threshold
0.60

## Why this threshold was selected
- It is the lowest estimated-cost threshold within the campaign capacity constraint under the balanced business case.
- Lower thresholds such as 0.25 or 0.40 produce lower raw cost, but they exceed outreach capacity and are therefore not operationally feasible.
- Higher thresholds such as 0.65 or 0.70 reduce outreach volume further, but they increase missed churners and raise estimated total cost.

## Operational implication at threshold 0.60
- Predicted positive count = 484
- This is within the retention team capacity of 500.
- True positives = 268
- False positives = 216
- False negatives = 106
- True negatives = 819
- Estimated total cost = 746

## Business interpretation
- This threshold reflects a balanced operating point rather than a maximum-recall policy.
- The business accepts some missed churners in exchange for keeping outreach volume feasible.
- The policy is suitable when both campaign capacity and cost discipline matter.

## When to revisit this threshold
- If the cost of missed churners rises materially
- If retention capacity changes
- If campaign cost per contacted customer changes
- If the model is retrained and score distributions shift

## Governance note
This operating threshold is scenario-dependent and should not be treated as universal. Threshold selection must be revisited whenever business assumptions change.
