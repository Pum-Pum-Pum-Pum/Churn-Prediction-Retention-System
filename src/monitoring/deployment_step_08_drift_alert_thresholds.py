from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = PROJECT_ROOT / 'data' / 'processed' / 'baseline_model_input.csv'
CURRENT_BATCH_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'batch_predictions_output.csv'
OUTPUT_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'drift_alert_thresholds_report.md'

# Starter alert thresholds for monitoring demonstration
MEAN_PROB_ALERT_DELTA = 0.05
POSITIVE_RATE_ALERT_DELTA = 0.05


def main():
    baseline_df = pd.read_csv(BASELINE_FILE)
    current_df = pd.read_csv(CURRENT_BATCH_FILE)

    baseline_positive_rate = round(baseline_df['churn_value'].mean(), 4) if 'churn_value' in baseline_df.columns else None
    current_predicted_positive_rate = round(current_df['predicted_label'].mean(), 4) if 'predicted_label' in current_df.columns else None
    current_mean_probability = round(current_df['churn_probability'].mean(), 4) if 'churn_probability' in current_df.columns else None

    positive_rate_shift = None
    if baseline_positive_rate is not None and current_predicted_positive_rate is not None:
        positive_rate_shift = round(current_predicted_positive_rate - baseline_positive_rate, 4)

    probability_alert = abs(current_mean_probability - baseline_positive_rate) > MEAN_PROB_ALERT_DELTA if current_mean_probability is not None and baseline_positive_rate is not None else False
    positive_rate_alert = abs(positive_rate_shift) > POSITIVE_RATE_ALERT_DELTA if positive_rate_shift is not None else False

    lines = []
    lines.append('# Drift Alert Thresholds Report')
    lines.append('## Baseline references')
    lines.append(f'- Baseline actual churn rate: {baseline_positive_rate}')
    lines.append(f'- Current mean churn probability: {current_mean_probability}')
    lines.append(f'- Current predicted positive rate: {current_predicted_positive_rate}')
    lines.append(f'- Predicted positive rate shift: {positive_rate_shift}')
    lines.append('## Alert thresholds')
    lines.append(f'- Mean probability alert delta: {MEAN_PROB_ALERT_DELTA}')
    lines.append(f'- Predicted positive rate alert delta: {POSITIVE_RATE_ALERT_DELTA}')
    lines.append('## Alert status')
    lines.append(f'- Mean probability alert triggered: {probability_alert}')
    lines.append(f'- Predicted positive rate alert triggered: {positive_rate_alert}')
    lines.append('## Governance note')
    lines.append('These alert thresholds are starter defaults for learning purposes and must be recalibrated using real production history.')

    OUTPUT_FILE.write_text(''.join(lines), encoding='utf-8')

    print('=== DEPLOYMENT / MONITORING STEP 8: DRIFT ALERT THRESHOLDS ===')
    print('Baseline actual churn rate:', baseline_positive_rate)
    print('Current mean churn probability:', current_mean_probability)
    print('Current predicted positive rate:', current_predicted_positive_rate)
    print('Predicted positive rate shift:', positive_rate_shift)
    print('Mean probability alert triggered:', probability_alert)
    print('Predicted positive rate alert triggered:', positive_rate_alert)
    print('Report saved to:', OUTPUT_FILE)


if __name__ == '__main__':
    main()
