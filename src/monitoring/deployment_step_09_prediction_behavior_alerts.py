from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'monitoring_reference_baseline.json'
CURRENT_BATCH_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'batch_predictions_output.csv'
OUTPUT_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'prediction_behavior_alerts_report.md'

MEAN_PROB_ALERT_DELTA = 0.05
POSITIVE_RATE_ALERT_DELTA = 0.05


def main():
    with open(REFERENCE_FILE, 'r', encoding='utf-8') as f:
        reference = json.load(f)

    current_df = pd.read_csv(CURRENT_BATCH_FILE)

    current_mean_probability = round(current_df['churn_probability'].mean(), 4)
    current_predicted_positive_rate = round(current_df['predicted_label'].mean(), 4)

    mean_probability_shift = round(current_mean_probability - reference['mean_churn_probability'], 4)
    positive_rate_shift = round(current_predicted_positive_rate - reference['predicted_positive_rate'], 4)

    probability_alert = abs(mean_probability_shift) > MEAN_PROB_ALERT_DELTA
    positive_rate_alert = abs(positive_rate_shift) > POSITIVE_RATE_ALERT_DELTA

    lines = []
    lines.append('# Prediction Behavior Alert Report')
    lines.append('## Reference values')
    lines.append(f"- Reference mean churn probability: {reference['mean_churn_probability']}")
    lines.append(f"- Reference predicted positive rate: {reference['predicted_positive_rate']}")
    lines.append('## Current values')
    lines.append(f'- Current mean churn probability: {current_mean_probability}')
    lines.append(f'- Current predicted positive rate: {current_predicted_positive_rate}')
    lines.append('## Shifts')
    lines.append(f'- Mean probability shift: {mean_probability_shift}')
    lines.append(f'- Predicted positive rate shift: {positive_rate_shift}')
    lines.append('## Alert status')
    lines.append(f'- Mean probability alert triggered: {probability_alert}')
    lines.append(f'- Predicted positive rate alert triggered: {positive_rate_alert}')
    lines.append('## Governance note')
    lines.append('This comparison isstronger than using historical target prevalence, because it compares current prediction behavior against stored historical prediction behavior.')

    OUTPUT_FILE.write_text('\n'.join(lines), encoding='utf-8')

    print('=== MONITORING STEP 9: PREDICTION BEHAVIOR ALERTS ===')
    print('Reference mean churn probability:', reference['mean_churn_probability'])
    print('Current mean churn probability:', current_mean_probability)
    print('Mean probability shift:', mean_probability_shift)
    print('Reference predicted positive rate:', reference['predicted_positive_rate'])
    print('Current predicted positive rate:', current_predicted_positive_rate)
    print('Predicted positive rate shift:', positive_rate_shift)
    print('Mean probability alert triggered:', probability_alert)
    print('Predicted positive rate alert triggered:', positive_rate_alert)
    print('Report saved to:', OUTPUT_FILE)


if __name__ == '__main__':
    main()
