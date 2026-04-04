from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = PROJECT_ROOT / 'data' / 'processed' / 'baseline_model_input.csv'
BATCH_PREDICTIONS_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'batch_predictions_output.csv'
OUTPUT_FILE = PROJECT_ROOT / 'artifacts' / 'reports' / 'drift_monitoring_summary.md'


def numeric_summary(df: pd.DataFrame, cols: list):
    rows = []
    for col in cols:
        rows.append({
            'feature': col,
            'mean': round(df[col].mean(), 4),
            'std': round(df[col].std(), 4),
            'min': round(df[col].min(), 4),
            'max': round(df[col].max(), 4)
        })
    return pd.DataFrame(rows)


def main():
    baseline_df = pd.read_csv(BASELINE_FILE)
    current_df = pd.read_csv(BATCH_PREDICTIONS_FILE)

    common_numeric = [
        col for col in baseline_df.select_dtypes(include=['number']).columns
        if col in current_df.columns and col not in ['churn_value', 'predicted_label']
    ]

    baseline_stats = numeric_summary(baseline_df, common_numeric)
    current_stats = numeric_summary(current_df, common_numeric)

    merged = baseline_stats.merge(current_stats, on='feature', suffixes=('_baseline', '_current'))
    merged['mean_shift'] = (merged['mean_current'] - merged['mean_baseline']).round(4)
    merged['std_shift'] = (merged['std_current'] - merged['std_baseline']).round(4)

    mean_pred = round(current_df['churn_probability'].mean(), 4) if 'churn_probability' in current_df.columns else None
    positive_rate = round(current_df['predicted_label'].mean(), 4) if 'predicted_label' in current_df.columns else None

    lines = []
    lines.append('# Drift Monitoring Starter Summary')
    lines.append('## Purpose')
    lines.append('This starter report compares baseline feature distributions against the current scored batch to identify obvious drift signals.')
    lines.append('## Current prediction behavior')
    lines.append(f'- Mean churn probability: {mean_pred}')
    lines.append(f'- Predicted positive rate: {positive_rate}')
    lines.append('## Numeric feature shift table')
    lines.append(merged.to_markdown(index=False))    
    lines.append('## Interpretation notes')
    lines.append('- Large mean or std shifts may indicate data drift.')
    lines.append('- This is only a starter check; real monitoring should also include population stability, category drift, and eventual outcome drift.')

    OUTPUT_FILE.write_text(''.join(lines), encoding='utf-8')

    print('=== DEPLOYMENT / INFERENCE DESIGN STEP 7: DRIFT MONITORING STARTER ===')
    print('Common numeric features monitored:')
    print(common_numeric)
    print('Mean churn probability:', mean_pred)
    print('Predicted positive rate:', positive_rate)
    print('Drift summary saved to:', OUTPUT_FILE)
    print('Shift preview:')
    print(merged)


if __name__ == '__main__':
    main()
